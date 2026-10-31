//! Affine lane projections for 1-16 bf16 rows on Metal. No family or model dependency.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("frags.zig");
pub const n16_qualification = @import("lane_n16_qualification.zig");

pub const weight_layout = "u32[ceil(N/32)][K/group][32][group*bits/32], low-bit-first codes, columns past N zero";
pub const reg_layout = "u32[N/32][K/64][2][32 lanes][4] 4-bit codes, regBlock order";
pub const metadata_layout = "bf16[K/group][N][scale,bias]";
/// Lean decode-row reg launches may take the metadata indexed: u32[sb_table] distinct (scale | bias << 16) words, then u16[K/group][N]
/// indices into them. A group's words are the pair it names, so its arithmetic is unchanged.
pub const indexed_layout = "u32[sb_table] then u16[K/group][N]";
pub const sb_table = 8192;

pub fn indexedBytes(n: usize, k: usize, group: usize) usize {
    return sb_table * 4 + n * (k / group) * 2;
}
pub const Output = enum { bf16, f32 };

pub const Format = struct {
    bits: u8,
    group: usize,

    pub fn validate(f: Format) !void {
        if (f.bits < 2 or f.bits > 8 or f.bits == 7) return error.UnsupportedBits;
        if (f.group != 32 and f.group != 64 and f.group != 128) return error.UnsupportedGroup;
    }
};

/// SK, PF and group cuts are fixed per pipeline, never chosen by row count; empty ranges mean all N/32 tiles.
pub const Layout = struct {
    n: usize,
    k: usize,
    format: Format,
    sk: usize = 1,
    pf: usize = 1,
    ranges: []const [2]usize = &.{},
    groups: ?[2]usize = null,
    output: Output = .bf16,
    /// Prepared sums reuse owned scratch; callers serialize all encodes using this projection until GPU completion.
    precompute_sums: bool = false,
    cooperative: bool = false,
    /// 4-bit g64 codes in reg_layout, unpacked into tensor-op registers; every word equals the cooperative kernel's.
    reg: bool = false,
    fused_n16: bool = false,
    ordinary_n16: bool = false,
    /// Run the two logical reg chains in one SIMD group, then add chain 1 to chain 0.
    serial_sk2: bool = false,
    /// Prompt-row reg launches (prompt SwiGLU included): the lean loop addresses its words by pointers that advance a fixed
    /// step a group.
    prompt_sr: bool = false,
    /// Reg launches of at most 8 rows (decoded rows): a lane's second row is padding and gets no accumulators.
    rows8: bool = false,
    /// Reg SIMD groups prefetch only codes; metadata and row sums load in their own iteration (decode rows).
    reg_lean: bool = false,
    /// Reg codes of adjacent tiles interleave by group: u32[N/32/B][K/64][B][2][32 lanes][4]; a partial last batch keeps zero blocks.
    tile_batch: usize = 1,
    /// Fused gate/up rows of decoded rows (rows8, serial SK2): a threadgroup's four tiles are 64 gate columns and their up
    /// columns; it stores q27_mlp_xs' activation and group sums instead of the gate/up rows.
    swiglu: bool = false,
    /// Lean decode-row reg launches (not prompt_sr) read lane_projection.indexed_layout metadata.
    indexed: bool = false,
    /// Elements between output rows (default N): a member of a fused stack writes its columns of the fused rows.
    y_stride: ?usize = null,

    pub fn validate(l: Layout) !void {
        try l.format.validate();
        if (l.cooperative and l.format.bits != 4) return error.UnsupportedCooperativeFormat;
        if (l.reg and (l.cooperative or l.format.bits != 4 or l.format.group != 64 or !l.precompute_sums)) return error.UnsupportedRegFormat;
        if ((l.cooperative or l.reg) and l.n % 32 != 0) return error.InvalidShape;
        if (l.y_stride) |s| if (s < l.n or s > std.math.maxInt(i32) / 16) return error.InvalidOutputStride;
        if (l.n == 0 or l.n % 16 != 0 or l.k == 0 or l.k % l.format.group != 0 or
            l.n > std.math.maxInt(i32) / 16 or l.k > std.math.maxInt(i32) / 16) return error.InvalidShape;
        if (l.sk == 0 or l.sk > 32 or (l.pf != 1 and l.pf != 2)) return error.InvalidSchedule;
        if (l.serial_sk2 and (!l.reg or l.sk != 2)) return error.InvalidSchedule;
        if (l.prompt_sr and (!l.reg or !l.reg_lean or l.rows8)) return error.InvalidSchedule;
        if (l.rows8 and !l.reg) return error.InvalidSchedule;
        if (l.reg_lean and !l.reg) return error.InvalidSchedule;
        if (l.indexed and (!l.reg or !l.reg_lean or l.prompt_sr)) return error.InvalidSchedule;
        if (l.tile_batch == 0 or (l.tile_batch != 1 and !l.reg)) return error.InvalidSchedule;
        if (l.ordinary_n16 and (!l.reg or !l.rows8 or !l.indexed or !l.reg_lean or l.swiglu or l.fused_n16 or l.ranges.len != 0 or l.groups != null)) return error.InvalidSchedule;
        if (l.fused_n16 and (!l.swiglu or !l.rows8 or !l.indexed or !l.serial_sk2 or l.sk != 2 or l.tile_batch != 8)) return error.InvalidSchedule;
        // decoded rows: rows8 on the serial split; prompt rows: 16-row blocks on one split
        const swiglu_rows = (l.rows8 and l.serial_sk2) or (!l.rows8 and !l.serial_sk2 and l.sk == 1);
        if (l.swiglu and (!swiglu_rows or !l.reg_lean or l.output != .bf16 or l.ranges.len != 0 or l.groups != null or
            l.y_stride != null or l.n % 128 != 0)) return error.InvalidSchedule;
        const g = l.groupCut();
        if (g[1] == 0 or g[0] >= l.k / l.format.group or g[1] > l.k / l.format.group - g[0]) return error.InvalidGroupCut;
        if (l.groups != null and l.output != .f32) return error.PartialOutputNeedsFloat;
        // Conservative declared storage, including the single-SK unused reduction array.
        if (l.threadgroupBytes() > 32768) return error.ThreadgroupMemoryLimit;
        var end: usize = 0;
        for (l.ranges) |r| {
            if (r[1] == 0 or r[0] < end or r[0] >= l.tileCount() or r[1] > l.tileCount() - r[0]) return error.InvalidTileRange;
            end = r[0] + r[1];
        }
        _ = try std.math.mul(usize, l.n, l.k);
    }

    pub fn groupCut(l: Layout) [2]usize {
        return l.groups orelse .{ 0, l.k / l.format.group };
    }

    /// 32-column tiles over N; a last tile of 16 columns is padded with zero codes and never stored.
    pub fn tileCount(l: Layout) usize {
        return (l.n + 31) / 32;
    }

    pub fn tiles(l: Layout) usize {
        if (l.ranges.len == 0) return l.tileCount();
        var count: usize = 0;
        for (l.ranges) |r| count += r[1];
        return count;
    }

    pub fn threadgroupBytes(l: Layout) usize {
        return l.sk * 32 * l.format.group + @max(l.sk -| 1, 1) * 16 * 32 * @sizeOf(f32);
    }

    pub fn weightWords(l: Layout) usize {
        return l.tileCount() * 32 * (l.k / 32) * l.format.bits;
    }

    /// MLX's packed W for this layout: N rows of K*bits/32 words.
    pub fn sourceWords(l: Layout) usize {
        return l.n * (l.k / 32) * l.format.bits;
    }

    pub fn metadataElements(l: Layout) usize {
        return l.n * (l.k / l.format.group) * 2;
    }

    pub fn packedBytes(l: Layout) usize {
        return l.weightWords() * 4 + l.metadataElements() * 2;
    }
};

pub fn source(a: std.mem.Allocator, l: Layout) ![]u8 {
    try l.validate();
    const g = l.groupCut();
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(a);
    try text.print(a, "#define TF_N {d}\n#define TF_K {d}\n#define TF_BITS {d}\n#define TF_GROUP {d}\n#define TF_SK {d}\n#define TF_G0 {d}\n#define TF_GN {d}\n#define TF_PF {d}\n#define TF_OUT {s}\n", .{
        l.n, l.k, l.format.bits, l.format.group, l.sk, g[0], g[1], l.pf, if (l.output == .f32) "float" else "bfloat",
    });
    try text.appendSlice(a, "inline int tf_tile(int j) {\n");
    var at: usize = 0;
    for (l.ranges) |r| {
        try text.print(a, "  if (j < {d}) return {d} + j - {d};\n", .{ at + r[1], r[0], at });
        at += r[1];
    }
    try text.appendSlice(a, if (l.ranges.len == 0) "  return j;\n}\n" else "  return 0;\n}\n");
    try text.print(a, "#define TF_PRECOMPUTE_SUMS {d}\n", .{@intFromBool(l.precompute_sums)});
    if (l.reg) try text.appendSlice(a, "#define TF_REG 1\n");
    if (l.serial_sk2) try text.appendSlice(a, "#define TF_SERIAL_SK2 1\n");
    if (l.prompt_sr) try text.appendSlice(a, "#define TF_PROMPT_SR 1\n");
    if (l.rows8) try text.appendSlice(a, "#define TF_ROWS8 1\n");
    if (l.reg_lean) try text.appendSlice(a, "#define TF_REG_LEAN 1\n");
    if (l.tile_batch != 1) try text.print(a, "#define TF_TB {d}\n", .{l.tile_batch});
    if (l.swiglu) try text.appendSlice(a, "#define TF_SWIGLU 1\n");
    if (l.indexed) try text.print(a, "#define TF_SBI {d}\n", .{sb_table});
    if (l.y_stride) |s| try text.print(a, "#define TF_YS {d}\n", .{s});
    try text.appendSlice(a, if (l.fused_n16) ks.core_lane_fused_n16 else if (l.ordinary_n16) ks.core_lane_ordinary_n16 else ks.core_lane_projection);
    return text.toOwnedSlice(a);
}

/// Exact byte copies of MLX affine W, S and B into the kernel's layouts (no dequantization); storage disjoint.
pub fn pack(l: Layout, w: []const u32, scales: []const u16, biases: []const u16, out_w: []u32, out_sb: []u16) !void {
    try l.validate();
    const ng = l.k / l.format.group;
    const wp = l.format.group * l.format.bits / 32;
    const row_words = l.k / 32 * l.format.bits;
    if (w.len != l.sourceWords() or scales.len != l.n * ng or biases.len != l.n * ng or
        out_w.len != l.weightWords() or out_sb.len != l.metadataElements()) return error.InvalidStorage;
    if (overlap(std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(out_w)) or
        overlap(std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(out_sb)) or
        overlap(std.mem.sliceAsBytes(scales), std.mem.sliceAsBytes(out_w)) or
        overlap(std.mem.sliceAsBytes(biases), std.mem.sliceAsBytes(out_w)) or
        overlap(std.mem.sliceAsBytes(scales), std.mem.sliceAsBytes(out_sb)) or
        overlap(std.mem.sliceAsBytes(biases), std.mem.sliceAsBytes(out_sb)) or
        overlap(std.mem.sliceAsBytes(out_w), std.mem.sliceAsBytes(out_sb))) return error.OverlappingStorage;
    for (0..l.tileCount()) |tile| for (0..ng) |g| for (0..32) |col| {
        const n = tile * 32 + col;
        const dst = ((tile * ng + g) * 32 + col) * wp;
        if (n >= l.n) {
            @memset(out_w[dst..][0..wp], 0);
            continue;
        }
        @memcpy(out_w[dst..][0..wp], w[n * row_words + g * wp ..][0..wp]);
        const sb = (g * l.n + n) * 2;
        out_sb[sb] = scales[n * ng + g];
        out_sb[sb + 1] = biases[n * ng + g];
    };
}

/// reg_layout for one 32-column x 64-code block, in the tensor op's right-input register order (lane_order.metal).
pub fn regBlock(tiled: *const [256]u32, out: *[256]u32) void {
    for (out, 0..) |*o, t| {
        const lane = (t >> 2) & 31;
        const w = 4 * (t >> 7) + (t & 3);
        const at = (((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1)) * 8 + ((lane >> 3) & 1) + 4 * (w & 1);
        const shift: u5 = @intCast(16 * (lane & 1));
        const a = (tiled[at] >> shift) & 0xffff;
        const c = (tiled[at + 2] >> shift) & 0xffff;
        o.* = (a & 0xf) | ((a >> 4) & 0xf0) | ((c & 0xf) << 8) | ((c << 4) & 0xf000) | (((a >> 4) & 0xf) << 16) | (((a >> 8) & 0xf0) << 16) | (((c >> 4) & 0xf) << 24) | (((c >> 8) & 0xf0) << 24);
    }
}

pub const Order = struct { w: Ref, n: usize, k: usize };

/// The reg_layout rewrite on the GPU: one library and queue for any number of runs.
pub const RegOrder = struct {
    pipe: mtl.Pipeline,
    queue: mtl.Queue,

    pub fn init(device: mtl.Device) !RegOrder {
        const lib = try mtl.Library.fromSource(device, ks.core_lane_order, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "tf_lane_order", false);
        errdefer pipe.deinit();
        return .{ .pipe = pipe, .queue = try device.queue() };
    }

    pub fn deinit(o: RegOrder) void {
        o.pipe.deinit();
        o.queue.deinit();
    }

    /// Rewrites tile-order 4-bit g64 words in place into reg_layout and waits; once per weight, before any reg encode.
    pub fn run(o: RegOrder, jobs: []const Order) !void {
        for (jobs) |j| {
            if (j.n == 0 or j.n % 32 != 0 or j.k == 0 or j.k % 64 != 0) return error.InvalidShape;
            try checkRef(j.w, j.n * j.k / 2, 4);
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = o.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        e.setPipeline(o.pipe);
        for (jobs) |j| {
            e.setBuffer(j.w.buf, j.w.off, 0);
            e.dispatchGroups(mtl.Size.of(j.n / 32 * (j.k / 64), 1, 1), mtl.Size.of(256, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
    }
};

pub fn regOrder(device: mtl.Device, jobs: []const Order) !void {
    const o = try RegOrder.init(device);
    defer o.deinit();
    try o.run(jobs);
}

fn overlap(a: []const u8, b: []const u8) bool {
    const aa = @intFromPtr(a.ptr);
    const bb = @intFromPtr(b.ptr);
    return if (aa <= bb) bb - aa < a.len else aa - bb < b.len;
}

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

pub const Projection = struct {
    pipe: mtl.Pipeline,
    n: usize,
    k: usize,
    group_count: usize,
    weight_bytes: usize,
    metadata_bytes: usize,
    output_bytes: usize,
    y_stride: usize,
    tile_count: usize,
    sk: usize,
    sums_pipe: ?mtl.Pipeline = null,
    sums_buffer: ?mtl.Buffer = null,
    threads_per_slice: usize = 32,
    reg: bool = false,
    fused_n16: bool = false,
    /// Built with Layout.rows8: launches take at most 8 rows.
    rows8: bool = false,
    /// Built with Layout.swiglu: encodeSwiglu only.
    swiglu: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, l: Layout) !Projection {
        try l.validate();
        if (!device.tensorUnits() and (l.format.bits != 6 or l.format.group != 32 or l.cooperative or l.reg or l.precompute_sums)) return error.TensorUnitsRequired;
        const text = try source(a, l);
        defer a.free(text);
        const adapted = try frags.source(device, a, text);
        defer a.free(adapted);
        const lib = try mtl.Library.fromSource(device, adapted, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const p = try mtl.Pipeline.init(device, lib, if (l.reg) "tf_lane_reg" else if (l.cooperative) "tf_lane_pair" else "tf_lane", false);
        errdefer p.deinit();
        const physical_sk: usize = if (l.serial_sk2) 1 else l.sk;
        if (p.simdWidth() != 32 or p.maxThreads() < (if (l.fused_n16) @as(usize, 256) else if (l.swiglu) @as(usize, 128) else physical_sk * (if (l.cooperative) @as(usize, 64) else 32))) return error.UnsupportedPipelineShape;
        var sums_pipe: ?mtl.Pipeline = null;
        var sums_buffer: ?mtl.Buffer = null;
        if (l.precompute_sums) {
            sums_pipe = try mtl.Pipeline.init(device, lib, "tf_lane_sums", false);
            errdefer sums_pipe.?.deinit();
            sums_buffer = try device.buffer(@as(usize, if (l.reg) reg_rows else 16) * (l.k / l.format.group) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        }
        return .{ .pipe = p, .n = l.n, .k = l.k, .group_count = l.k / l.format.group, .weight_bytes = l.weightWords() * 4, .metadata_bytes = if (l.indexed) indexedBytes(l.n, l.k, l.format.group) else l.metadataElements() * 2, .output_bytes = if (l.output == .f32) 4 else 2, .y_stride = l.y_stride orelse l.n, .tile_count = l.tiles() * (if (l.ordinary_n16) @as(usize, 2) else 1), .sk = physical_sk, .sums_pipe = sums_pipe, .sums_buffer = sums_buffer, .threads_per_slice = if (l.cooperative) 64 else 32, .reg = l.reg, .fused_n16 = l.fused_n16, .rows8 = l.rows8, .swiglu = l.swiglu };
    }

    pub fn deinit(p: Projection) void {
        p.pipe.deinit();
        if (p.sums_pipe) |pipe| pipe.deinit();
        if (p.sums_buffer) |buffer| buffer.deinit();
    }

    /// Encodes one window; the caller owns buffers through GPU completion and orders later accesses.
    pub fn encode(p: Projection, e: mtl.ComputeEncoder, x: Ref, w: Ref, sb: Ref, y: Ref, rows: u32) !void {
        return p.encodeImpl(e, x, w, sb, null, y, rows);
    }

    /// The reg kernel on group sums its input's producer wrote (tf_lane_sums' order): no sums dispatch, no barrier.
    pub fn encodeSums(p: Projection, e: mtl.ComputeEncoder, x: Ref, w: Ref, sb: Ref, sums: Sums, y: Ref, rows: u32) !void {
        if (!p.reg) return error.SumsNeedReg;
        if (rows == 0 or rows > reg_rows or sums.stride < rows) return error.InvalidRows;
        try checkRef(sums.ref, ((p.group_count - 1) * @as(usize, sums.stride) + rows) * 4, 4);
        return p.encodeImpl(e, x, w, sb, sums, y, rows);
    }

    /// A swiglu projection: `h` gets silu(gate) * up of the rows' bf16 gate/up words (N / 2 columns a row) and `hs` its
    /// group sums at `sums`' stride, padded rows zero, as q27_mlp_xs writes them.
    pub fn encodeSwiglu(p: Projection, e: mtl.ComputeEncoder, x: Ref, w: Ref, sb: Ref, sums: Sums, h: Ref, hs: Ref, rows: u32) !void {
        if (!p.swiglu) return error.InvalidSchedule;
        const window = try swigluWindow(rows, p.rows8, sums.stride);
        const blocks = (rows + 15) / 16;
        try checkRef(x, rows * p.k * 2, 2);
        try checkRef(w, p.weight_bytes, 16);
        try checkRef(sb, p.metadata_bytes, 16);
        try checkRef(sums.ref, try std.math.mul(usize, try stridedSumsElements(p.group_count, sums.stride, window.input_rows), 4), 4);
        try checkRef(h, rows * (p.n / 2) * 2, 2);
        try checkRef(hs, try std.math.mul(usize, try stridedSumsElements(p.n / 128, sums.stride, window.output_rows), 4), 4);
        e.setPipeline(p.pipe);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(sb.buf, sb.off, 2);
        e.setValue([2]i32{ @intCast(rows), @intCast(sums.stride) }, 3);
        e.setBuffer(h.buf, h.off, 4);
        e.setBuffer(sums.ref.buf, sums.ref.off, 5);
        e.setBuffer(hs.buf, hs.off, 6);
        e.dispatchGroups(mtl.Size.of(p.n / 128, if (p.rows8) 1 else blocks, 1), mtl.Size.of(if (p.fused_n16) 256 else 128, 1, 1));
    }

    /// Only the group sums of `x` into this projection's sums buffer (what encode dispatches first), with no barrier: the
    /// caller orders it before the encodeSums dispatches that read the returned sums.
    pub fn encodeSumsOf(p: Projection, e: mtl.ComputeEncoder, x: Ref, rows: u32) !Sums {
        if (!p.reg) return error.SumsNeedReg;
        if (rows == 0 or rows > reg_rows) return error.InvalidRows;
        const pipe = p.sums_pipe orelse return error.SumsNeedReg;
        const buffer = p.sums_buffer orelse return error.SumsNeedReg;
        try checkRef(x, rows * p.k * 2, 2);
        const stride: u32 = 16 * ((rows + 15) / 16);
        e.setPipeline(pipe);
        e.setBuffer(x.buf, x.off, 0);
        e.setValue(@as(i32, @intCast(rows)), 1);
        e.setBuffer(buffer, 0, 2);
        e.setValue(@as(i32, @intCast(stride)), 3);
        e.dispatchThreads(.{ .width = stride * p.group_count }, .{ .width = 256 });
        return .{ .ref = .{ .buf = buffer }, .stride = stride };
    }

    fn encodeImpl(p: Projection, e: mtl.ComputeEncoder, x: Ref, w: Ref, sb: Ref, given: ?Sums, y: Ref, rows: u32) !void {
        if (rows == 0 or rows > @as(u32, if (p.rows8) 8 else if (p.reg) reg_rows else 16)) return error.InvalidRows;
        const blocks = (rows + 15) / 16;
        const stride: u32 = if (given) |g| g.stride else 16 * blocks;
        try checkRef(x, rows * p.k * 2, 2);
        try checkRef(w, p.weight_bytes, if (p.reg) 16 else 8);
        try checkRef(sb, p.metadata_bytes, 16);
        try checkRef(y, ((rows - 1) * p.y_stride + p.n) * p.output_bytes, p.output_bytes);
        if (given == null) if (p.sums_pipe) |pipe| {
            e.setPipeline(pipe);
            e.setBuffer(x.buf, x.off, 0);
            e.setValue(@as(i32, @intCast(rows)), 1);
            e.setBuffer(p.sums_buffer.?, 0, 2);
            e.setValue(@as(i32, @intCast(stride)), 3);
            e.dispatchThreads(.{ .width = stride * p.group_count }, .{ .width = 256 });
            e.barrier();
        };
        e.setPipeline(p.pipe);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(sb.buf, sb.off, 2);
        e.setValue([2]i32{ @intCast(rows), @intCast(stride) }, 3);
        e.setBuffer(y.buf, y.off, 4);
        if (given) |g| e.setBuffer(g.ref.buf, g.ref.off, 5) else if (p.sums_buffer) |buffer| e.setBuffer(buffer, 0, 5);
        e.dispatchGroups(mtl.Size.of(p.tile_count, if (p.reg) blocks else 1, 1), mtl.Size.of(p.threads_per_slice * p.sk, 1, 1));
        if (given == null and p.sums_buffer != null) e.barrier();
    }
};

/// Rows one reg encode takes, as 16-row blocks on the grid's y (a prompt chunk's rows in one weight read).
pub const reg_rows = 128;
/// Prompt SwiGLU may consume this many rows of producer-owned sums; owned scratch stays at reg_rows.
pub const max_supplied_rows = 2048;

const SwigluWindow = struct { input_rows: u32, output_rows: u32 };

fn swigluWindow(rows: u32, rows8: bool, stride: u32) !SwigluWindow {
    if (rows == 0 or rows > @as(u32, if (rows8) 8 else max_supplied_rows) or stride < rows) return error.InvalidRows;
    const padded: u32 = 16 * ((rows + 15) / 16);
    const input_rows = if (rows8) rows else padded;
    // Every prompt block reads its padding XS and writes sixteen HS values, zero for inactive rows.
    if (!rows8 and stride < padded) return error.InvalidRows;
    return .{ .input_rows = input_rows, .output_rows = if (rows8) stride else padded };
}

fn stridedSumsElements(groups: usize, stride: u32, final_rows: u32) !usize {
    if (groups == 0 or final_rows == 0) return error.InvalidRows;
    return std.math.add(usize, try std.math.mul(usize, groups - 1, stride), final_rows);
}

test "fused N16 is opt in and requires the indexed ROWS8 serial TB8 SwiGLU schedule" {
    var l = Layout{ .n = 34816, .k = 5120, .format = .{ .bits = 4, .group = 64 }, .sk = 2, .precompute_sums = true, .reg = true, .serial_sk2 = true, .rows8 = true, .reg_lean = true, .tile_batch = 8, .swiglu = true, .indexed = true };
    try std.testing.expect(!l.fused_n16);
    const baseline = try source(std.testing.allocator, l);
    defer std.testing.allocator.free(baseline);
    l.fused_n16 = true;
    try l.validate();
    const narrow = try source(std.testing.allocator, l);
    defer std.testing.allocator.free(narrow);
    try std.testing.expect(std.mem.indexOf(u8, baseline, "max_total_threads_per_threadgroup(256)") == null);
    try std.testing.expect(std.mem.indexOf(u8, narrow, "max_total_threads_per_threadgroup(256)") != null);
    inline for (.{ "swiglu", "rows8", "indexed", "serial_sk2" }) |field| {
        var invalid = l;
        @field(invalid, field) = false;
        try std.testing.expectError(error.InvalidSchedule, invalid.validate());
    }
    l.tile_batch = 1;
    try std.testing.expectError(error.InvalidSchedule, l.validate());
}

test "wide prompt SwiGLU validates input and output padded sums without widening decode or owned scratch" {
    const tail = try swigluWindow(1065, false, 1072);
    try std.testing.expectEqual(@as(u32, 1072), tail.input_rows);
    try std.testing.expectEqual(@as(u32, 1072), tail.output_rows);
    try std.testing.expectEqual(@as(usize, 80 * 1072), try stridedSumsElements(80, 1072, tail.input_rows));
    try std.testing.expectEqual(@as(usize, 272 * 1072), try stridedSumsElements(272, 1072, tail.output_rows));
    try std.testing.expectEqual(@as(usize, 79 * 2048 + 1072), try stridedSumsElements(80, 2048, tail.input_rows));
    try std.testing.expectEqual(@as(usize, 271 * 2048 + 1072), try stridedSumsElements(272, 2048, tail.output_rows));
    try std.testing.expectError(error.InvalidRows, swigluWindow(129, false, 129));
    try std.testing.expectError(error.InvalidRows, swigluWindow(1065, false, 1071));
    try std.testing.expectError(error.InvalidRows, swigluWindow(2049, false, 2064));
    try std.testing.expectError(error.InvalidRows, swigluWindow(0, false, 16));
    try std.testing.expectError(error.InvalidRows, swigluWindow(9, true, 16));
    const decoded = try swigluWindow(1, true, 16);
    try std.testing.expectEqual(@as(u32, 1), decoded.input_rows);
    try std.testing.expectEqual(@as(u32, 16), decoded.output_rows);
    try std.testing.expectError(error.InvalidRows, swigluWindow(17, false, 17));
    const short = try swigluWindow(17, false, 32);
    try std.testing.expectEqual(@as(u32, 32), short.input_rows);
    try std.testing.expectEqual(@as(u32, 32), short.output_rows);
    for (1..max_supplied_rows + 1) |count| {
        const rows: u32 = @intCast(count);
        const padded: u32 = 16 * ((rows + 15) / 16);
        const valid = try swigluWindow(rows, false, padded);
        try std.testing.expectEqual(padded, valid.input_rows);
        try std.testing.expectEqual(padded, valid.output_rows);
        try std.testing.expectError(error.InvalidRows, swigluWindow(rows, false, padded - 1));
    }
    const last = try swigluWindow(2048, false, 2048);
    try std.testing.expectEqual(@as(u32, 2048), last.output_rows);
    try std.testing.expectError(error.InvalidRows, stridedSumsElements(0, 1072, 1072));
    try std.testing.expectError(error.Overflow, stridedSumsElements(std.math.maxInt(usize), 1072, 1072));
}

/// Group sums of a projection's input: XS[group * stride + row], fp32, each group's values added in order from zero.
pub const Sums = struct { ref: Ref, stride: u32 };

test "rows8 is off by default, needs the reg kernel and only adds its define" {
    var l = Layout{ .n = 10240, .k = 5120, .format = .{ .bits = 4, .group = 64 }, .sk = 2, .reg = true, .precompute_sums = true, .serial_sk2 = true };
    try std.testing.expect(!l.rows8);
    const plain = try source(std.testing.allocator, l);
    defer std.testing.allocator.free(plain);
    try std.testing.expect(std.mem.indexOf(u8, plain, "#define TF_ROWS8") == null);
    l.rows8 = true;
    const rows8 = try source(std.testing.allocator, l);
    defer std.testing.allocator.free(rows8);
    const define = "#define TF_ROWS8 1\n";
    const at = std.mem.indexOf(u8, rows8, define) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings(plain[0..at], rows8[0..at]);
    try std.testing.expectEqualStrings(plain[at..], rows8[at + define.len ..]);
    l.reg = false;
    l.serial_sk2 = false;
    try std.testing.expectError(error.InvalidSchedule, l.validate());
}

test "serial SK2 keeps logical split and is off by default" {
    var l = Layout{ .n = 34816, .k = 5120, .format = .{ .bits = 4, .group = 64 }, .sk = 2, .reg = true, .precompute_sums = true };
    try std.testing.expect(!l.serial_sk2);
    const plain = try source(std.testing.allocator, l);
    defer std.testing.allocator.free(plain);
    try std.testing.expect(std.mem.indexOf(u8, plain, "#define TF_SERIAL_SK2") == null);
    l.serial_sk2 = true;
    const serial = try source(std.testing.allocator, l);
    defer std.testing.allocator.free(serial);
    try std.testing.expect(std.mem.startsWith(u8, serial, "#define TF_N 34816\n#define TF_K 5120\n"));
    try std.testing.expect(std.mem.indexOf(u8, serial, "#define TF_SK 2\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, serial, "#define TF_SERIAL_SK2 1\n") != null);
    l.sk = 1;
    try std.testing.expectError(error.InvalidSchedule, l.validate());
    l.sk = 2;
    l.reg = false;
    try std.testing.expectError(error.InvalidSchedule, l.validate());
}

fn checkRef(r: Ref, need: usize, alignment: usize) !void {
    if (r.off % alignment != 0) return error.MisalignedBuffer;
    const len = r.buf.length();
    if (r.off > len or need > len - r.off) return error.BufferTooSmall;
}

test "layouts reject unsupported formats, schedules, cuts and overlapping tile reads" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const good = Layout{ .n = 64, .k = 256, .format = .{ .bits = 4, .group = 64 } };
    const s = try source(arena.allocator(), good);
    try std.testing.expect(std.mem.indexOf(u8, s, "#define TF_BITS 4\n#define TF_GROUP 64\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, s[0 .. s.len - ks.core_lane_projection.len], "TF_YS") == null);
    var bad = good;
    for ([_]u8{ 2, 3, 5, 6, 8 }) |bits| {
        bad.format.bits = bits;
        try bad.validate();
    }
    for ([_]u8{ 0, 1, 7, 9 }) |bits| {
        bad.format.bits = bits;
        try std.testing.expectError(error.UnsupportedBits, bad.validate());
    }
    bad = good;
    bad.y_stride = 96;
    try std.testing.expect(std.mem.indexOf(u8, try source(arena.allocator(), bad), "#define TF_YS 96\n") != null);
    bad.y_stride = 63;
    try std.testing.expectError(error.InvalidOutputStride, bad.validate());
    bad = good;
    bad.n = 48;
    bad.ranges = &.{.{ 1, 1 }};
    try bad.validate();
    try std.testing.expectEqual(2, bad.tileCount());
    bad.cooperative = true;
    try std.testing.expectError(error.InvalidShape, bad.validate());
    bad.cooperative = false;
    bad.reg = true;
    bad.precompute_sums = true;
    try std.testing.expectError(error.InvalidShape, bad.validate());
    bad = good;
    bad.ranges = &.{.{ 2, 1 }};
    try std.testing.expectError(error.InvalidTileRange, bad.validate());
    bad = good;
    bad.format.group = 16;
    try std.testing.expectError(error.UnsupportedGroup, bad.validate());
    bad = good;
    bad.n = 33;
    try std.testing.expectError(error.InvalidShape, bad.validate());
    bad = good;
    bad.k = 96;
    try std.testing.expectError(error.InvalidShape, bad.validate());
    bad = good;
    bad.sk = 0;
    try std.testing.expectError(error.InvalidSchedule, bad.validate());
    bad = good;
    bad.sk = 32;
    try std.testing.expectError(error.ThreadgroupMemoryLimit, bad.validate());
    bad = good;
    bad.pf = 3;
    try std.testing.expectError(error.InvalidSchedule, bad.validate());
    bad = good;
    bad.groups = .{ 3, 2 };
    try std.testing.expectError(error.InvalidGroupCut, bad.validate());
    bad = good;
    bad.groups = .{ 0, 2 };
    try std.testing.expectError(error.PartialOutputNeedsFloat, bad.validate());
    bad = good;
    bad.ranges = &.{ .{ 0, 1 }, .{ 0, 1 } };
    try std.testing.expectError(error.InvalidTileRange, bad.validate());
}

test "every width, group and 16-row member packs MLX codes, metadata and zero padding without changing a bit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng: std.Random.DefaultPrng = .init(0x1a9e);
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| for ([_]usize{ 64, 48 }) |rows| {
        const l = Layout{ .n = rows, .k = 2 * group, .format = .{ .bits = bits, .group = group } };
        const codes = try a.alloc(u8, l.n * l.k);
        const w = try a.alloc(u32, l.sourceWords());
        @memset(w, 0);
        for (codes, 0..) |*v, i| {
            v.* = @intCast(prng.random().intRangeAtMost(u16, 0, (@as(u16, 1) << @intCast(bits)) - 1));
            // MLX's packing written bit by bit: row n is one low-bit-first stream of K codes.
            for (0..bits) |bit| w[(i * bits + bit) / 32] |= @as(u32, (v.* >> @intCast(bit)) & 1) << @intCast((i * bits + bit) % 32);
        }
        const s = try a.alloc(u16, l.n * 2);
        const b = try a.alloc(u16, l.n * 2);
        for (s, b, 0..) |*sv, *bv, i| {
            sv.* = @intCast(i);
            bv.* = @intCast(65535 - i);
        }
        const pw = try a.alloc(u32, l.weightWords());
        @memset(pw, 0xffff_ffff);
        const sb = try a.alloc(u16, l.metadataElements());
        try pack(l, w, s, b, pw, sb);
        const wp = group * bits / 32;
        for (0..l.tileCount() * 32) |n| for (0..l.k) |k| {
            const at = ((((n / 32) * 2 + k / group) * 32 + n % 32) * wp) * 32 + (k % group) * bits;
            var code: u32 = 0;
            for (0..bits) |bit| code |= ((pw[(at + bit) / 32] >> @intCast((at + bit) % 32)) & 1) << @intCast(bit);
            try std.testing.expectEqual(@as(u32, if (n < l.n) codes[n * l.k + k] else 0), code);
        };
        for (0..l.n) |n| for (0..2) |g| {
            try std.testing.expectEqual(s[n * 2 + g], sb[(g * l.n + n) * 2]);
            try std.testing.expectEqual(b[n * 2 + g], sb[(g * l.n + n) * 2 + 1]);
        };
        try std.testing.expectError(error.InvalidStorage, pack(l, w[1..], s, b, pw, sb));
        try std.testing.expectError(error.InvalidStorage, pack(l, w, s, b, pw[1..], sb));
        try std.testing.expectError(error.OverlappingStorage, pack(l, pw[0..w.len], s, b, pw, sb));
    };
}

test "reg order puts every code in the tensor op's right-input register for its lane" {
    var tiled: [256]u32 = undefined;
    for (&tiled, 0..) |*v, i| v.* = @as(u32, @intCast(i)) *% 0x9e3779b9 +% 0x7f4a7c15;
    var out: [256]u32 = undefined;
    regBlock(&tiled, &out);
    for (0..32) |lane| for (0..8) |w| for (0..8) |j| {
        // lane l's register 16 kc + 4 nj + e: K 4(l&1) + 8(l>>3&1) + 16 kc + e, column (l>>1&3) + 4(l>>4&1) + 8 nj
        const q = j & 3;
        const k = 4 * (lane & 1) + 8 * ((lane >> 3) & 1) + 16 * (2 * (w & 1) + (q >> 1)) + 2 * (q & 1) + (j >> 2);
        const col = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1);
        const want = (tiled[col * 8 + k / 8] >> @intCast(4 * (k % 8))) & 15;
        const got = (out[(w >> 2) * 128 + lane * 4 + (w & 3)] >> @intCast(4 * j)) & 15;
        try std.testing.expectEqual(want, got);
    };
    var bad = Layout{ .n = 64, .k = 256, .format = .{ .bits = 4, .group = 64 }, .reg = true };
    try std.testing.expectError(error.UnsupportedRegFormat, bad.validate());
    bad.precompute_sums = true;
    try bad.validate();
    bad.format.group = 128;
    try std.testing.expectError(error.UnsupportedRegFormat, bad.validate());
}

test "ordinary N16 layout excludes unsupported routes and preserves packing counts" {
    const yes = Layout{ .n = 10240, .k = 5120, .format = .{ .bits = 4, .group = 64 }, .sk = 2, .precompute_sums = true, .reg = true, .serial_sk2 = true, .rows8 = true, .reg_lean = true, .indexed = true, .tile_batch = 8, .ordinary_n16 = true };
    try yes.validate();
    var wide = yes;
    wide.ordinary_n16 = false;
    try std.testing.expectEqual(wide.weightWords(), yes.weightWords());
    try std.testing.expectEqual(wide.metadataElements(), yes.metadataElements());
    inline for (.{ .{ "rows8", false }, .{ "indexed", false }, .{ "reg_lean", false }, .{ "swiglu", true }, .{ "fused_n16", true } }) |change| {
        var no = yes;
        @field(no, change[0]) = change[1];
        try std.testing.expectError(error.InvalidSchedule, no.validate());
    }
}
