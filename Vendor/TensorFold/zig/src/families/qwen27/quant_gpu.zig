//! Runtime specialization preserves the original M5 group arithmetic and GPU-resident argument buffers.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
const row = @import("core").row_projection;
const sources = @import("qwen_runtime_sources");
const projection = @import("projection.zig");
const profile = @import("core").gpu_profile;
pub const Ref = projection.Ref;
/// Group sums the input's producer wrote: XS[group * stride + row], in the lane projection's order.
pub const Sums = struct { ref: Ref, stride: u32 };

const Key = struct {
    kind: Kind,
    n: u32 = 0,
    k: u32,
    slices: u32 = 0,
    tile: u32 = 0,
    tmr: u32 = 0,
    edge: bool = false,
    skip: u32 = 0,
    cut: [2]u32 = .{ 0, 0 },
    range: [2]u32 = .{ 0, 0 },
    bits: u8 = 4,
    group: u32 = 64,
    rows8: bool = false,
    lean: bool = false,
    prompt_sr: bool = false,
    batch: u32 = 1,
    swiglu: bool = false,
    sbi: bool = false,
    sbi12: bool = false,
    resid: bool = false,
    count: bool = false,
    wait: bool = false,
    tiles: u32 = 0,
    parallel_sk2: bool = false,

    /// The decode projections run as serial K chains: fused gate/up (34816), GDN qkv (10240) and attention query+gate
    /// (12288) with K split 2 (GDN z|b|a keeps its four K slices in parallel).
    fn serialSk2(key: Key) bool {
        const two = key.slices == 2 and (key.n == 34816 or key.n == 10240 or key.n == 12288);
        // GDN z|b|a runs its four K halves in parallel slices: a two-row launch has 195 tiles, and one
        // 32-thread slice per tile leaves the machine idle (measured 57.4 -> 46.8 us a call on this box).
        const four = false;
        return key.kind == .projection and key.bits == 4 and key.group == 64 and (two or four) and key.k == 5120 and key.skip == 0 and key.cut[0] == 0 and key.cut[1] == 0 and key.range[1] == 0 and !key.parallel_sk2;
    }
};

/// The larger live half-tiles favor parallel SK2 for the wide decode projections: fused gate/up (34816), GDN qkv (10240)
/// and the attention query+gate (12288), whose 384 half-tiles sit between the other two.
fn wideDecodeSk2(n: u32, input_width: u32, rows: u32, prompt: bool) bool {
    return !prompt and input_width == 5120 and (n == 34816 or n == 10240 or n == 12288) and rows >= 6 and rows <= 8;
}

test "wide decode SK2 keeps row and phase boundaries distinct" {
    for ([_]u32{ 34816, 10240 }) |n| {
        for ([_]u32{ 6, 7, 8 }) |rows| try std.testing.expect(wideDecodeSk2(n, 5120, rows, false));
        for ([_]u32{ 0, 1, 2, 3, 4, 5, 9, 16 }) |rows| try std.testing.expect(!wideDecodeSk2(n, 5120, rows, false));
        try std.testing.expect(!wideDecodeSk2(n, 5120, 8, true));
        try std.testing.expect(!wideDecodeSk2(n, 2560, 8, false));
    }
    try std.testing.expect(wideDecodeSk2(12288, 5120, 8, false));
    for ([_]u32{ 5120, 248320 }) |n| try std.testing.expect(!wideDecodeSk2(n, 5120, 8, false));
}

test "serial SK2 guard and cache keys separate every excluded projection" {
    const eligible = Key{ .kind = .projection, .n = 34816, .k = 5120, .slices = 2 };
    try std.testing.expect(eligible.serialSk2());
    try std.testing.expect((Key{ .kind = .projection, .n = 10240, .k = 5120, .slices = 2 }).serialSk2());
    try std.testing.expect((Key{ .kind = .projection, .n = 12288, .k = 5120, .slices = 2 }).serialSk2());
    inline for (.{
        .{ "kind", Kind.mlp },
        .{ "n", @as(u32, 17408) },
        .{ "k", @as(u32, 2560) },
        .{ "slices", @as(u32, 1) },
        .{ "slices", @as(u32, 4) },
        .{ "skip", @as(u32, 1) },
        .{ "cut", [2]u32{ 1, 0 } },
        .{ "cut", [2]u32{ 0, 40 } },
        .{ "parallel_sk2", true },
        .{ "bits", @as(u8, 8) },
        .{ "group", @as(u32, 128) },
    }) |change| {
        var excluded = eligible;
        @field(excluded, change[0]) = change[1];
        try std.testing.expect(!excluded.serialSk2());
        var cache = std.AutoHashMap(Key, bool).init(std.testing.allocator);
        defer cache.deinit();
        try cache.put(eligible, eligible.serialSk2());
        try cache.put(excluded, excluded.serialSk2());
        try std.testing.expectEqual(@as(u32, 2), cache.count());
        try std.testing.expect(cache.get(eligible).?);
        try std.testing.expect(!cache.get(excluded).?);
    }
    try std.testing.expectEqual(@as(?u32, 1), projection.promptSplit(34816, 5120));
}
pub const Kind = enum { projection, xsum, norm, input_norm, mlp, mlp_wide, ane_input, mlp_ane, add_ane };

pub const Kernels = struct {
    allocator: std.mem.Allocator,
    residency: ?mtl.ResidencySet.Follow = null,
    device: mtl.Device,
    profiler: ?*profile.Trace = null,
    row_pipelines: ?row.Pipelines = null,
    simd: ?row.simd.Pipelines = null,
    simd_bits: ?row.simd_bits.Pipelines = null,
    /// Raw matrices take simd_qmm_bits' tiles; off on tensor units, whose raw matrices are runs short of a lane tile.
    raw_tiles: bool = true,
    /// Per [n, k] shape: whether the scalar twin's 1-2 row bits equal the tiles' (checked once on this chip).
    twins: std.AutoHashMapUnmanaged([2]usize, bool) = .empty,
    /// The same for simd_qmm_bits, per [bits, n, k].
    bit_twins: std.AutoHashMapUnmanaged([3]usize, bool) = .empty,
    /// Per [n, k]: whether simd_qmm's chunk pairs give its tiles' bits (checked once on this chip).
    pair_twins: std.AutoHashMapUnmanaged([2]usize, bool) = .empty,
    /// Set around prompt chunks: prompt rows take projection.promptSplit, decoded rows the decode split.
    prompt: bool = false,
    lanes: std.AutoHashMapUnmanaged(Key, lane.Projection) = .empty,
    cache: std.AutoHashMapUnmanaged(Key, mtl.Pipeline) = .empty,
    /// quantCounted's count word and normAfter's arrival word, zeroed when made and left zero by each waiting norm.
    count_words: ?mtl.Buffer = null,
    /// The decoded gate/up's count and the waiting down's arrivals (encodeSwigluCounted / encodeCountedAfter), left zero.
    gu_words: ?mtl.Buffer = null,

    pub fn setResident(k: *Kernels, hook: ?mtl.ResidencySet.Follow) void {
        var it = k.lanes.valueIterator();
        while (it.next()) |p| {
            if (k.residency) |f| mtl.resident.fields.apply(p, f, false);
            if (hook) |f| mtl.resident.fields.apply(p, f, true);
        }
        k.residency = hook;
    }

    pub fn joined(k: *Kernels, p: *const lane.Projection) void {
        if (k.residency) |f| mtl.resident.fields.apply(p, f, true);
    }

    pub fn deinit(k: *Kernels) void {
        k.setResident(null);
        if (k.row_pipelines) |*pipelines| pipelines.deinit();
        if (k.simd) |*pipelines| pipelines.deinit();
        if (k.simd_bits) |*pipelines| pipelines.deinit();
        k.twins.deinit(k.allocator);
        k.bit_twins.deinit(k.allocator);
        k.pair_twins.deinit(k.allocator);
        var lane_it = k.lanes.valueIterator();
        while (lane_it.next()) |p| p.deinit();
        k.lanes.deinit(k.allocator);
        if (k.count_words) |b| b.deinit();
        if (k.gu_words) |b| b.deinit();
        var it = k.cache.valueIterator();
        while (it.next()) |p| p.deinit();
        k.cache.deinit(k.allocator);
    }

    fn get(k: *Kernels, key: Key) !mtl.Pipeline {
        if (k.cache.get(key)) |p| return p;
        const template = switch (key.kind) {
            .projection => unreachable,
            .xsum => sources.xsum,
            .norm => sources.norm_xs,
            .input_norm => sources.norm_input,
            .mlp => sources.mlp_xs,
            .mlp_wide => sources.mlp_xs_wide,
            .ane_input, .mlp_ane, .add_ane => sources.mlp_ane,
        };
        const mark = std.mem.indexOf(u8, template, "Q27_SPECIALIZE") orelse return error.BadKernelSource;
        const definitions = switch (key.kind) {
            .projection => unreachable,
            .xsum => try std.fmt.allocPrint(k.allocator, "constexpr int K={d}, GS=64;", .{key.k}),
            .norm, .input_norm => try std.fmt.allocPrint(k.allocator, "constexpr int K={d};", .{key.k}),
            .mlp, .mlp_wide => try std.fmt.allocPrint(k.allocator, "constexpr int N={d};", .{key.n}),
            .ane_input, .mlp_ane, .add_ane => try std.fmt.allocPrint(k.allocator, "constant constexpr int H={d}, N={d}, A={d};", .{ key.k, key.n, key.tile }),
        };
        defer k.allocator.free(definitions);
        const source = try std.mem.concat(k.allocator, u8, &.{ template[0..mark], definitions, template[mark + "Q27_SPECIALIZE".len ..] });
        defer k.allocator.free(source);
        const library = try mtl.Library.fromSource(k.device, source, mtl.CompileOptions.mlx());
        defer library.deinit();
        const name: [:0]const u8 = switch (key.kind) {
            .projection => unreachable,
            .xsum => "custom_kernel_q27_xsum_bfloat16_t_int32_t_float",
            .norm => "custom_kernel_q27_norm_xs_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float",
            .input_norm => "custom_kernel_q27_norm_input_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_float",
            .mlp => "custom_kernel_q27_mlp_xs_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_float",
            .mlp_wide => "custom_kernel_q27_mlp_xs_wide_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_float",
            .ane_input => "q27_ane_input",
            .mlp_ane => "q27_mlp_ane",
            .add_ane => "q27_add_ane",
        };
        const p = try mtl.Pipeline.init(k.device, library, name, false);
        errdefer p.deinit();
        try k.cache.put(k.allocator, key, p);
        return p;
    }

    pub fn quant(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32) !void {
        return k.quantSkip(input_encoder, p, x, xs, dims, y, rows, 0);
    }

    /// Decoded rows' fused gate/up projection with the MLP activation in its epilogue: `out` and `out_sums` get what
    /// `quant` then `mlp` write there, without the gate/up rows. False (nothing encoded) where the shape does not take it.
    pub fn gateUpActivated(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: Sums, out: Ref, out_sums: Ref, rows: u32) !bool {
        // prompt rows: the fused SwiGLU epilogue with the gate and up pairs on two tiles a SIMD group (gateUpActivatedPrompt)
        if (k.prompt) return k.gateUpActivatedPrompt(input_encoder, p, x, xs, out, out_sums, rows);
        if (rows == 0 or rows > 8 or p.runs.len > 0 or p.raw != null or p.tile != 32) return false;
        if (p.bits != 4 or p.group != 64) return false;
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = p.slices, .bits = p.bits, .group = p.group, .rows8 = true, .lean = true, .batch = p.batch, .swiglu = true, .sbi = p.index != null, .sbi12 = p.index != null and p.index12 };
        if (!key.serialSk2() or p.n != 34816) return false;
        var span = profile.begin(k.profiler, input_encoder, .dense);
        span.shape = .{ .n = p.n, .k = p.k, .sk = p.slices, .bits = p.bits };
        errdefer span.cancel();
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = p.bits, .group = p.group }, .sk = key.slices, .precompute_sums = true, .reg = true, .serial_sk2 = true, .tile_batch = key.batch, .rows8 = true, .reg_lean = true, .swiglu = true, .indexed = key.sbi, .indexed12 = key.sbi12 });
            k.joined(entry.value_ptr);
        }
        try entry.value_ptr.encodeSwiglu(span.encoder, .{ .buf = x.buffer, .off = x.offset }, .{ .buf = p.words.buffer, .off = p.words.offset }, metadata(p, key.sbi), .{ .ref = .{ .buf = xs.ref.buffer, .off = xs.ref.offset }, .stride = xs.stride }, .{ .buf = out.buffer, .off = out.offset }, .{ .buf = out_sums.buffer, .off = out_sums.offset }, rows);
        try span.end();
        return true;
    }

    /// Prompt rows' fused gate/up and activation: one K split, 16-row blocks, up to `lane.reg_rows` rows a launch, two
    /// tiles a SIMD group (the gate pair, then the up pair) where the code batches allow.
    fn gateUpActivatedPrompt(
        k: *Kernels,
        input_encoder: mtl.ComputeEncoder,
        p: projection.Linear,
        x: Ref,
        xs: Sums,
        out: Ref,
        out_sums: Ref,
        rows: u32,
    ) !bool {
        if (rows == 0 or p.runs.len > 0 or p.raw != null or p.tile != 32) return false;
        if (p.bits != 4 or p.group != 64 or p.n != 34816) return false;
        if ((projection.promptSplit(p.n, p.k) orelse p.slices) != 1) return false;
        const key = Key{
            .kind = .projection,
            .n = p.n,
            .k = p.k,
            .slices = 1,
            .bits = p.bits,
            .group = p.group,
            .lean = true,
            .prompt_sr = true,
            .batch = p.batch,
            .swiglu = true,
        };
        var span = profile.begin(k.profiler, input_encoder, .dense);
        span.shape = .{ .n = p.n, .k = p.k, .sk = 1, .bits = p.bits };
        errdefer span.cancel();
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{
                .n = p.n,
                .k = p.k,
                .format = .{ .bits = p.bits, .group = p.group },
                .sk = 1,
                .precompute_sums = true,
                .reg = true,
                .prompt_sr = key.prompt_sr,
                .tile_batch = key.batch,
                .reg_lean = true,
                .swiglu = true,
                // two tiles a SIMD group where the gate and up pairs each sit inside one code batch
                .prompt_tn = if (key.batch % 2 == 0 and (p.n / 64) % key.batch == 0) 2 else 1,
            });
            k.joined(entry.value_ptr);
        }
        const half = p.n / 2;
        var first: u32 = 0;
        while (first < rows) : (first += lane.reg_rows) {
            const n = @min(rows - first, lane.reg_rows);
            const xr = lane.Ref{ .buf = x.buffer, .off = x.offset + @as(usize, first) * p.k * 2 };
            const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
            const sb = lane.Ref{ .buf = p.pairs.buffer, .off = p.pairs.offset };
            const xs_off = xs.ref.offset + @as(usize, first) * 4;
            const xsr = lane.Sums{ .ref = .{ .buf = xs.ref.buffer, .off = xs_off }, .stride = xs.stride };
            const hr = lane.Ref{ .buf = out.buffer, .off = out.offset + @as(usize, first) * half * 2 };
            const hsr = lane.Ref{ .buf = out_sums.buffer, .off = out_sums.offset + @as(usize, first) * 4 };
            try entry.value_ptr.encodeSwiglu(span.encoder, xr, w, sb, xsr, hr, hsr, n);
        }
        try span.end();
        return true;
    }

    /// Prompt rows of a two-tile 5120-wide output projection with the residual add in its store: `residual` gets `hidden`
    /// + the output rows, each word as q27_norm_xs forms it, and no output rows are written; the norm after it then reads
    /// `residual` alone. False (nothing encoded) where the launch has no two-tile prompt form: decoded rows among them.
    pub fn quantResid(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, hidden: Ref, residual: Ref, rows: u32) !bool {
        if (!k.prompt or p.runs.len > 0 or p.raw != null or p.bits != 4 or p.group != 64 or p.tile != 32) return false;
        if (rows == 0 or rows > @import("gpu_frame.zig").max_rows) return false;
        const sk = projection.promptSplit(p.n, p.k) orelse p.slices;
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = sk, .bits = p.bits, .group = p.group, .lean = true, .batch = p.batch, .prompt_sr = true, .resid = true };
        const tn = promptTiles(key, p.n, 0);
        if (tn == 1 or key.serialSk2()) return false;
        var span = profile.begin(k.profiler, input_encoder, .dense);
        span.shape = .{ .n = p.n, .k = p.k, .sk = sk, .bits = p.bits };
        errdefer span.cancel();
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = p.bits, .group = p.group }, .sk = sk, .precompute_sums = true, .reg = true, .prompt_sr = true, .tile_batch = key.batch, .reg_lean = true, .prompt_tn = tn, .residual = true });
            k.joined(entry.value_ptr);
        }
        var first: u32 = 0;
        while (first < rows) : (first += lane.reg_rows) {
            const n = @min(rows - first, lane.reg_rows);
            const xr = lane.Ref{ .buf = x.buffer, .off = x.offset + @as(usize, first) * p.k * 2 };
            const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
            const hr = lane.Ref{ .buf = hidden.buffer, .off = hidden.offset + @as(usize, first) * p.n * 2 };
            const rr = lane.Ref{ .buf = residual.buffer, .off = residual.offset + @as(usize, first) * p.n * 2 };
            const sr: ?lane.Sums = if (xs) |g| .{ .ref = .{ .buf = g.ref.buffer, .off = g.ref.offset + @as(usize, first) * 4 }, .stride = g.stride } else null;
            try entry.value_ptr.encodeResid(span.encoder, xr, w, metadata(p, false), sr, hr, rr, n);
        }
        try span.end();
        return true;
    }

    /// `quant` over input groups [cut[0], cut[0] + cut[1]) only, into fp32 partial sums.
    pub fn quantCut(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: Sums, dims: Ref, y: Ref, rows: u32, cut: [2]u32) !void {
        return k.quantImpl(input_encoder, p, x, xs, dims, y, rows, 0, cut, .{ 0, 0 });
    }

    /// `quant` over output tiles [range[0], range[0] + range[1]) only (32 columns a tile), each written at its own
    /// columns with the full launch's code: the same words there, nothing elsewhere.
    pub fn quantRange(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32, range: [2]u32) !void {
        if (range[1] == 0) return error.BadCoreProjectionRows;
        return k.quantImpl(input_encoder, p, x, xs, dims, y, rows, 0, .{ 0, 0 }, range);
    }

    /// Whether `quantRange` takes `p`: the 4-bit g64 reg path of a plain matrix.
    pub fn ranged(p: projection.Linear) bool {
        return p.runs.len == 0 and p.raw == null and p.bits == 4 and p.group == 64 and p.tile == 32;
    }

    /// `quant` over a fused two-half projection (the MLP's gate_up) without the first `skip` tiles of each half.
    pub fn quantSkip(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32, skip: u32) !void {
        return k.quantImpl(input_encoder, p, x, xs, dims, y, rows, skip, .{ 0, 0 }, .{ 0, 0 });
    }

    fn quantImpl(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32, skip: u32, cut: [2]u32, range: [2]u32) !void {
        if (p.runs.len > 0) {
            if (range[1] > 0) return error.BadCoreProjectionRows;
            return k.quantRuns(input_encoder, p, x, xs, dims, y, rows, skip, cut);
        }
        return k.quantOne(input_encoder, p, x, xs, dims, y, rows, skip, cut, range);
    }

    fn quantOne(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32, skip: u32, cut: [2]u32, range: [2]u32) !void {
        if ((skip > 0 or cut[1] > 0 or range[1] > 0) and p.raw != null) return error.BadCoreProjectionRows;
        if (range[1] > 0 and (skip > 0 or cut[1] > 0 or range[0] + range[1] > (p.n + 31) / 32)) return error.BadCoreProjectionRows;
        if (skip > 0 and (p.raw != null or skip >= p.n / 64)) return error.BadCoreProjectionRows;
        if (p.raw) |raw| if (raw.bits != 4 and k.tiled(raw)) {
            if (rows == 0 or rows > 128 or p.tile != 32) return error.BadCoreProjectionRows;
            return k.strided(input_encoder, raw, x, y, rows, @intCast(raw.n));
        };
        var span = profile.begin(k.profiler, input_encoder, .dense);
        const sk = if (k.prompt) projection.promptSplit(p.n, p.k) orelse p.slices else p.slices;
        span.shape = .{ .n = p.n, .k = p.k, .sk = sk, .bits = p.bits };
        errdefer span.cancel();
        const e = span.encoder;

        _ = dims;
        if (rows == 0 or rows > @import("gpu_frame.zig").max_rows or p.tile != 32) return error.BadCoreProjectionRows;
        if (p.raw) |raw| if (raw.bits == 4 and raw.group == 64 and raw.sum == .f32 and raw.n % 8 == 0) {
            const m = row.simd.Matrix{ .w = raw.w, .w_off = raw.w_off, .scales = raw.scales, .s_off = raw.s_off, .biases = raw.biases, .b_off = raw.b_off, .n = raw.n, .k = raw.k };
            const scalar = try k.twin(m); // loads the tiles before they're read
            const call = try row.simd.call(&k.simd.?, raw.n, raw.k, rows, scalar, try k.pairTwin(m));
            row.simd.encode(e, call, m, x.buffer, x.offset, y.buffer, y.offset);
            e.barrier();
            try span.end();
            return;
        };
        if (p.raw) |raw| {
            if (k.row_pipelines == null) k.row_pipelines = try row.Pipelines.load(k.device);
            const call = try row.call(&k.row_pipelines.?, raw, rows, false);
            e.setPipeline(call.pipeline);
            e.setBuffer(x.buffer, x.offset, 0);
            e.setBuffer(raw.w, raw.w_off, 1);
            e.setBuffer(raw.scales, raw.s_off, 2);
            e.setBuffer(raw.biases, raw.b_off, 3);
            e.setValue(call.dims, 4);
            e.setBuffer(y.buffer, y.offset, 5);
            e.dispatchGroups(.{ .width = call.groups }, .{ .width = call.threads });
            e.barrier();
            try span.end();
            return;
        }
        const reg = p.bits == 4 and p.group == 64;
        if (!reg and (skip > 0 or cut[1] > 0 or range[1] > 0)) return error.BadCoreProjectionRows;
        // decoded rows (a verify's 2, a serial step's 1) take the reg kernel without each lane's padding row
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = if (reg) sk else p.slices, .skip = skip, .cut = cut, .range = range, .bits = p.bits, .group = p.group, .rows8 = reg and !k.prompt and rows <= 8, .lean = reg, .batch = if (reg) p.batch else 1, .prompt_sr = reg and k.prompt, .sbi = reg and !k.prompt and p.index != null, .sbi12 = reg and !k.prompt and p.index != null and p.index12, .parallel_sk2 = reg and sk == 2 and skip == 0 and cut[0] == 0 and cut[1] == 0 and range[1] == 0 and wideDecodeSk2(p.n, p.k, rows, k.prompt) };
        const sums: ?Sums = if (reg) xs else null; // the non-reg kernel sums its own groups
        const half = p.n / 64;
        const ranges = [_][2]usize{ .{ skip, half - skip }, .{ half + skip, half - skip } };
        const tiles = [_][2]usize{.{ range[0], range[1] }};
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            const tn = promptTiles(key, p.n, skip);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = p.bits, .group = p.group }, .sk = key.slices, .ranges = if (skip > 0) &ranges else if (range[1] > 0) &tiles else &.{}, .groups = if (cut[1] > 0) .{ cut[0], cut[1] } else null, .output = if (cut[1] > 0) .f32 else .bf16, .precompute_sums = reg, .reg = reg, .serial_sk2 = key.serialSk2() and tn == 1, .prompt_sr = key.prompt_sr, .tile_batch = key.batch, .rows8 = key.rows8, .reg_lean = key.lean, .prompt_tn = tn, .indexed = key.sbi, .indexed12 = key.sbi12 });
            k.joined(entry.value_ptr);
        }
        const step: u32 = if (entry.value_ptr.reg) lane.reg_rows else 16;
        var first: u32 = 0;
        while (first < rows) : (first += step) {
            const xr = lane.Ref{ .buf = x.buffer, .off = x.offset + @as(usize, first) * p.k * 2 };
            const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
            const sb = metadata(p, key.sbi);
            const yr = lane.Ref{ .buf = y.buffer, .off = y.offset + @as(usize, first) * p.n * @as(usize, if (cut[1] > 0) 4 else 2) };
            if (sums) |s| try entry.value_ptr.encodeSums(e, xr, w, sb, .{ .ref = .{ .buf = s.ref.buffer, .off = s.ref.offset + @as(usize, first) * 4 }, .stride = s.stride }, yr, @min(rows - first, step)) else try entry.value_ptr.encode(e, xr, w, sb, yr, @min(rows - first, step));
        }
        try span.end();
    }

    /// Compiles at load the pipeline a decode-row `quant` of `p` over `rows` takes (for shapes no target layer uses).
    pub fn prepareDecode(k: *Kernels, p: projection.Linear, rows: u32) !void {
        if (k.prompt or rows == 0 or p.runs.len > 0 or p.raw != null or p.tile != 32) return;
        const reg = p.bits == 4 and p.group == 64;
        const key = Key{
            .kind = .projection,
            .n = p.n,
            .k = p.k,
            .slices = p.slices,
            .bits = p.bits,
            .group = p.group,
            .rows8 = reg and rows <= 8,
            .lean = reg,
            .batch = if (reg) p.batch else 1,
            .sbi = reg and p.index != null,
            .sbi12 = reg and p.index != null and p.index12,
        };
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (entry.found_existing) return;
        errdefer _ = k.lanes.remove(key);
        entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{
            .n = p.n,
            .k = p.k,
            .format = .{ .bits = p.bits, .group = p.group },
            .sk = key.slices,
            .output = .bf16,
            .precompute_sums = reg,
            .reg = reg,
            .serial_sk2 = key.serialSk2(),
            .tile_batch = key.batch,
            .rows8 = key.rows8,
            .reg_lean = key.lean,
            .indexed = key.sbi, .indexed12 = key.sbi12,
        });
        k.joined(entry.value_ptr);
    }

    /// `quant` of at most 8 decoded rows over only the 32-column tiles a GPU list names (lane_projection.Layout.tile_list):
    /// tile list[1 + j]'s columns land at j * 32 of rows `stride` apart, each word the one `quant` of `p` writes there;
    /// slots past list[0] (of `slots`) get the lowest finite bf16.
    pub fn quantTiles(k: *Kernels, e: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: Sums, list: Ref, y: Ref, rows: u32, slots: u32, stride: u32) !void {
        if (rows == 0 or rows > 8) return error.BadCoreProjectionRows;
        const t = try k.tileList(p, stride);
        try t.encodeTiles(e, .{ .buf = x.buffer, .off = x.offset }, .{ .buf = p.words.buffer, .off = p.words.offset }, metadata(p, p.index != null), .{ .ref = .{ .buf = xs.ref.buffer, .off = xs.ref.offset }, .stride = xs.stride }, .{ .buf = list.buffer, .off = list.offset }, .{ .buf = y.buffer, .off = y.offset }, rows, slots);
    }

    /// quantTiles over only the list's own tiles: `args` holds the list's count, 1, 1 (written by the dispatch that wrote
    /// the list), the threadgroups of an indirect launch; `slots` bounds the count and the room `y` has a row.
    pub fn quantTilesIndirect(k: *Kernels, e: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: Sums, list: Ref, y: Ref, rows: u32, slots: u32, stride: u32, args: Ref) !void {
        if (rows == 0 or rows > 8) return error.BadCoreProjectionRows;
        const t = try k.tileList(p, stride);
        try t.encodeTilesIndirect(e, .{ .buf = x.buffer, .off = x.offset }, .{ .buf = p.words.buffer, .off = p.words.offset }, metadata(p, p.index != null), .{ .ref = .{ .buf = xs.ref.buffer, .off = xs.ref.offset }, .stride = xs.stride }, .{ .buf = list.buffer, .off = list.offset }, .{ .buf = y.buffer, .off = y.offset }, rows, slots, .{ .buf = args.buffer, .off = args.offset });
    }

    /// Compiles at load the pipeline quantTiles of `p` at `stride` takes.
    pub fn prepareTiles(k: *Kernels, p: projection.Linear, stride: u32) !void {
        _ = try k.tileList(p, stride);
    }

    /// The tile-list build of the decode-row pipeline `quant` of `p` takes for at most 8 rows (prepareDecode's key), so
    /// a listed tile's words are that launch's.
    fn tileList(k: *Kernels, p: projection.Linear, stride: u32) !*lane.Projection {
        if (k.prompt or !ranged(p) or stride == 0 or stride % 32 != 0) return error.BadCoreProjectionRows;
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = p.slices, .bits = p.bits, .group = p.group, .rows8 = true, .lean = true, .batch = p.batch, .sbi = p.index != null, .sbi12 = p.index != null and p.index12, .tiles = stride };
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = p.bits, .group = p.group }, .sk = key.slices, .output = .bf16, .precompute_sums = true, .reg = true, .serial_sk2 = key.serialSk2(), .tile_batch = key.batch, .rows8 = true, .reg_lean = true, .indexed = key.sbi, .indexed12 = key.sbi12, .y_stride = stride, .tile_list = true });
            k.joined(entry.value_ptr);
        }
        return entry.value_ptr;
    }

    /// A mixed stack: each member its own product into its columns; row kernels write the fused rows at a stride.
    fn quantRuns(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32, skip: u32, cut: [2]u32) !void {
        if (skip > 0 or cut[1] > 0 or rows == 0 or rows > 128) return error.BadCoreProjectionRows;
        for (p.runs) |run| {
            if (run.linear.raw) |raw| {
                try k.strided(input_encoder, raw, x, .{ .buffer = y.buffer, .offset = y.offset + @as(usize, run.first) * 2 }, rows, p.n);
                continue;
            }
            for (0..rows) |r| { // lane kernels take one row a launch until they write at a stride
                const xr = Ref{ .buffer = x.buffer, .offset = x.offset + r * p.k * 2 };
                const sr: ?Sums = if (xs) |s| .{ .ref = .{ .buffer = s.ref.buffer, .offset = s.ref.offset + r * 4 }, .stride = s.stride } else null;
                const yr = Ref{ .buffer = y.buffer, .offset = y.offset + (r * p.n + run.first) * 2 };
                try k.quantOne(input_encoder, run.linear, xr, sr, dims, yr, 1, 0, .{ 0, 0 }, .{ 0, 0 });
            }
        }
    }

    /// simd_qmm's tiles at any width (4-bit: its bits), else the row kernel; output row m at y + m * stride values.
    fn strided(k: *Kernels, input_encoder: mtl.ComputeEncoder, raw: row.Weights, x: Ref, y: Ref, rows: u32, stride: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .dense);
        span.shape = .{ .n = @intCast(raw.n), .k = @intCast(raw.k), .sk = 1, .bits = raw.bits };
        errdefer span.cancel();
        const e = span.encoder;
        if (k.tiled(raw)) {
            if (k.simd_bits == null) k.simd_bits = try row.simd_bits.Pipelines.load(k.device);
            const m = row.simd_bits.Matrix{ .w = raw.w, .w_off = raw.w_off, .scales = raw.scales, .s_off = raw.s_off, .biases = raw.biases, .b_off = raw.b_off };
            const scalar = rows <= row.simd.max_scalar_rows and try k.bitTwin(m, raw); // loads the tiles before they're read
            const call = try row.simd_bits.call(&k.simd_bits.?, raw.bits, raw.n, raw.k, rows, stride, scalar);
            row.simd_bits.encode(e, call, m, x.buffer, x.offset, y.buffer, y.offset);
            e.barrier();
            try span.end();
            return;
        }
        if (k.row_pipelines == null) k.row_pipelines = try row.Pipelines.load(k.device);
        const call = try row.callStrided(&k.row_pipelines.?, raw, rows, stride);
        e.setPipeline(call.pipeline);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(raw.w, raw.w_off, 1);
        e.setBuffer(raw.scales, raw.s_off, 2);
        e.setBuffer(raw.biases, raw.b_off, 3);
        e.setValue(call.dims, 4);
        e.setBuffer(y.buffer, y.offset, 5);
        e.dispatchGroups(.{ .width = call.groups }, .{ .width = call.threads });
        e.barrier();
        try span.end();
    }

    /// Checks simd_qmm_bits' scalar twin against its tiles once per width and shape.
    fn bitTwin(k: *Kernels, m: row.simd_bits.Matrix, raw: row.Weights) !bool {
        const key = [3]usize{ raw.bits, raw.n, raw.k };
        if (k.bit_twins.get(key)) |ok| return ok;
        const ok = try row.simd_bits.twinExact(&k.simd_bits.?, k.device, raw.bits, m, raw.n, raw.k);
        if (!ok) std.log.info("qwen27: {d}-bit {d}x{d} rows take simdgroup tiles at every width (the scalar twin's bits differ on this chip)", .{ raw.bits, raw.n, raw.k });
        try k.bit_twins.put(k.allocator, key, ok);
        return ok;
    }

    /// Groups of 64 with fp32 sums at a width the tiles unpack, where raw matrices take the tiles.
    fn tiled(k: *const Kernels, raw: row.Weights) bool {
        return k.raw_tiles and raw.group == 64 and raw.sum == .f32 and raw.n % 8 == 0 and std.mem.indexOfScalar(u8, &row.simd_bits.bit_widths, raw.bits) != null;
    }

    /// Loads the tiles on first use and checks the scalar twin against them once per shape.
    fn twin(k: *Kernels, m: row.simd.Matrix) !bool {
        if (k.simd == null) k.simd = try row.simd.Pipelines.load(k.device);
        if (k.twins.get(.{ m.n, m.k })) |ok| return ok;
        const ok = try row.simd.twinExact(&k.simd.?, k.device, m);
        if (!ok) std.log.info("qwen27: {d}x{d} rows take simdgroup tiles at every width (the scalar twin's bits differ on this chip)", .{ m.n, m.k });
        try k.twins.put(k.allocator, .{ m.n, m.k }, ok);
        return ok;
    }

    /// Whether this shape's prompt rows may take simd_qmm's chunk pairs: checked against its tiles once per shape.
    fn pairTwin(k: *Kernels, m: row.simd.Matrix) !bool {
        if (k.pair_twins.get(.{ m.n, m.k })) |ok| return ok;
        const ok = try row.simd.pairedExact(&k.simd.?, k.device, m);
        if (!ok and row.simd.pairs(m.n, 2, row.simd.splits(m.n))) std.log.info("qwen27: {d}x{d} rows keep simd_qmm's plain tiles (chunk pairs' bits differ on this chip)", .{ m.n, m.k });
        try k.pair_twins.put(k.allocator, .{ m.n, m.k }, ok);
        return ok;
    }

    pub fn sum(k: *Kernels, input_encoder: mtl.ComputeEncoder, x: Ref, dims: Ref, xs: Ref, width: u32, rows: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .xsum);
        errdefer span.cancel();
        const e = span.encoder;

        e.setPipeline(try k.get(.{ .kind = .xsum, .k = width }));
        bind(e, .{ x, dims, xs });
        e.dispatchThreads(mtl.Size.of(width / 64, padded(rows), 1), mtl.Size.of(@min(width / 64, 256), 1, 1));
        try span.end();
    }

    pub fn norm(k: *Kernels, input_encoder: mtl.ComputeEncoder, h: Ref, residual: ?Ref, weight: Ref, eps: Ref, dims: Ref, combined: Ref, out: Ref, xs: Ref, width: u32, rows: u32) !void {
        if (residual) |r| try wordAligned(.{ h, r, weight, combined, out });
        var span = profile.begin(k.profiler, input_encoder, .norm);
        errdefer span.cancel();
        const e = span.encoder;

        e.setPipeline(try k.get(.{ .kind = if (residual != null) .norm else .input_norm, .k = width }));
        if (residual) |r| {
            bind(e, .{ h, r, weight, eps, dims, combined, out, xs });
            bindAt(e, out, 8); // never read: wait_for 0
            e.setValue(@as(u32, 0), 9);
        } else bind(e, .{ h, weight, eps, dims, out, xs });
        e.dispatchGroups(mtl.Size.of(1, rows, 1), mtl.Size.of(width / 16, 1, 1));
        try span.end();
    }

    /// `norm` (with `residual`) in the barrier interval of the quantCounted launch that writes `residual`: each row's
    /// threadgroup waits until that launch's `wait_for` threadgroups have counted their fenced stores.
    pub fn normAfter(k: *Kernels, input_encoder: mtl.ComputeEncoder, h: Ref, residual: Ref, weight: Ref, eps: Ref, dims: Ref, combined: Ref, out: Ref, xs: Ref, width: u32, rows: u32, wait_for: u32) !void {
        const words = k.count_words orelse return error.BadCoreProjectionRows;
        if (wait_for == 0 or rows == 0 or rows > 8) return error.BadCoreProjectionRows;
        try wordAligned(.{ h, residual, weight, combined, out });
        var span = profile.begin(k.profiler, input_encoder, .norm);
        errdefer span.cancel();
        const e = span.encoder;
        e.setPipeline(try k.get(.{ .kind = .norm, .k = width }));
        bind(e, .{ h, residual, weight, eps, dims, combined, out, xs });
        e.setBuffer(words, 0, 8);
        e.setValue(wait_for, 9);
        e.dispatchGroups(mtl.Size.of(1, rows, 1), mtl.Size.of(width / 16, 1, 1));
        try span.end();
    }

    /// `quant` of decoded rows (at most 8, given sums, a parallel K split) whose threadgroups count themselves once their
    /// stores are fenced (Layout.count), for a normAfter in the same barrier interval. Returns that norm's wait (the
    /// launch's threadgroups); null: nothing encoded, the caller takes quant.
    /// Whether `p` on these rows takes the counted launch (quantCounted, quantCountedAfter); serial split-K launches count
    /// once a threadgroup too, but only parallel split-K launches wait.
    fn countedKey(k: *const Kernels, p: projection.Linear, rows: u32, wait: bool) ?Key {
        if (k.prompt or p.runs.len > 0 or p.raw != null or p.bits != 4 or p.group != 64 or p.tile != 32 or rows == 0 or rows > 8) return null;
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = p.slices, .bits = p.bits, .group = p.group, .rows8 = true, .lean = true, .batch = p.batch, .sbi = p.index != null, .sbi12 = p.index != null and p.index12, .count = true, .wait = wait };
        if (key.serialSk2() and wait) return null;
        if (!key.serialSk2() and p.slices < 2) return null;
        return key;
    }

    /// Whether quantCounted takes `p` for `rows` decoded rows (checked before either of a pair is encoded).
    pub fn countable(k: *const Kernels, p: projection.Linear, rows: u32) bool {
        return k.countedKey(p, rows, false) != null;
    }

    pub fn quantCounted(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: Sums, y: Ref, rows: u32) !?u32 {
        const key = k.countedKey(p, rows, false) orelse return null;
        if (k.count_words == null) {
            const b = try k.device.buffer(8, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            @memset(b.contents()[0..8], 0);
            k.count_words = b;
        }
        var span = profile.begin(k.profiler, input_encoder, .dense);
        span.shape = .{ .n = p.n, .k = p.k, .sk = p.slices, .bits = p.bits };
        errdefer span.cancel();
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = p.bits, .group = p.group }, .sk = key.slices, .precompute_sums = true, .reg = true, .tile_batch = key.batch, .rows8 = true, .reg_lean = true, .indexed = key.sbi, .indexed12 = key.sbi12, .count = true, .serial_sk2 = key.serialSk2() });
            k.joined(entry.value_ptr);
        }
        const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
        try entry.value_ptr.encodeCounted(span.encoder, .{ .buf = x.buffer, .off = x.offset }, w, metadata(p, key.sbi), .{ .ref = .{ .buf = xs.ref.buffer, .off = xs.ref.offset }, .stride = xs.stride }, .{ .buf = y.buffer, .off = y.offset }, rows, .{ .buf = k.count_words.? });
        try span.end();
        return @intCast(entry.value_ptr.tile_count);
    }

    /// quantCounted for a projection whose input rows and sums come from a gateUpCounted launch in the same barrier interval:
    /// its threadgroups wait on that launch's `wait_for` threadgroups before they read them.
    pub fn quantCountedAfter(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: Sums, y: Ref, rows: u32, wait_for: u32) !u32 {
        const key = k.countedKey(p, rows, true) orelse return error.BadCoreProjectionRows;
        const gu = k.gu_words orelse return error.BadCoreProjectionRows;
        if (k.count_words == null) {
            const b = try k.device.buffer(8, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            @memset(b.contents()[0..8], 0);
            k.count_words = b;
        }
        var span = profile.begin(k.profiler, input_encoder, .dense);
        span.shape = .{ .n = p.n, .k = p.k, .sk = p.slices, .bits = p.bits };
        errdefer span.cancel();
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = p.bits, .group = p.group }, .sk = key.slices, .precompute_sums = true, .reg = true, .tile_batch = key.batch, .rows8 = true, .reg_lean = true, .indexed = key.sbi, .indexed12 = key.sbi12, .count = true, .wait = true });
            k.joined(entry.value_ptr);
        }
        const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
        try entry.value_ptr.encodeCountedAfter(span.encoder, .{ .buf = x.buffer, .off = x.offset }, w, metadata(p, key.sbi), .{ .ref = .{ .buf = xs.ref.buffer, .off = xs.ref.offset }, .stride = xs.stride }, .{ .buf = y.buffer, .off = y.offset }, rows, .{ .buf = k.count_words.? }, .{ .buf = gu }, wait_for);
        try span.end();
        return @intCast(entry.value_ptr.tile_count);
    }

    /// Decoded rows: the fused gate/up's lockstep launch (gateUpActivated's) counting its threadgroups once its activated
    /// rows and their group sums are stored, when `down` takes the counted launch that waits on them in the same barrier
    /// interval (quantCountedAfter); null (nothing encoded) otherwise.
    pub fn gateUpCounted(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, down: projection.Linear, x: Ref, xs: Sums, out: Ref, out_sums: Ref, rows: u32) !?u32 {
        if (k.prompt or rows == 0 or rows > 8 or p.runs.len > 0 or p.raw != null or p.tile != 32) return null;
        if (p.bits != 4 or p.group != 64 or p.n != 34816 or p.k != 5120 or p.batch != 8 or p.index == null) return null;
        if (k.countedKey(down, rows, true) == null) return null;
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = p.slices, .bits = p.bits, .group = p.group, .rows8 = true, .lean = true, .batch = p.batch, .swiglu = true, .sbi = true, .sbi12 = p.index12, .count = true };
        if (!key.serialSk2()) return null;
        if (k.gu_words == null) {
            const b = try k.device.buffer(8, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            @memset(b.contents()[0..8], 0);
            k.gu_words = b;
        }
        var span = profile.begin(k.profiler, input_encoder, .dense);
        span.shape = .{ .n = p.n, .k = p.k, .sk = p.slices, .bits = p.bits };
        errdefer span.cancel();
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = p.bits, .group = p.group }, .sk = key.slices, .precompute_sums = true, .reg = true, .serial_sk2 = true, .tile_batch = key.batch, .rows8 = true, .reg_lean = true, .swiglu = true, .indexed = true, .indexed12 = key.sbi12, .gu_count = true });
            k.joined(entry.value_ptr);
        }
        const n = try entry.value_ptr.encodeSwigluCounted(span.encoder, .{ .buf = x.buffer, .off = x.offset }, .{ .buf = p.words.buffer, .off = p.words.offset }, metadata(p, true), .{ .ref = .{ .buf = xs.ref.buffer, .off = xs.ref.offset }, .stride = xs.stride }, .{ .buf = out.buffer, .off = out.offset }, .{ .buf = out_sums.buffer, .off = out_sums.offset }, rows, .{ .buf = k.gu_words.? });
        try span.end();
        return n;
    }

    /// Prompt rows [base, base + 128) into the Neural Engine's fp16 input planes [hidden][128].
    pub fn aneInput(k: *Kernels, input_encoder: mtl.ComputeEncoder, x: Ref, dims: Ref, planes: Ref, hidden: u32, width: u32, columns: u32, base: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .copy);
        errdefer span.cancel();
        const e = span.encoder;
        e.setPipeline(try k.get(.{ .kind = .ane_input, .k = hidden, .n = width, .tile = columns }));
        bind(e, .{ x, dims, planes });
        e.setValue(@as(i32, @intCast(base)), 3);
        e.dispatchThreads(mtl.Size.of(128, hidden, 1), mtl.Size.of(128, 1, 1));
        try span.end();
    }

    /// `mlp` over 64-column groups from `first`; gate/up below `columns` come from the Neural Engine's planes.
    pub fn mlpAne(k: *Kernels, input_encoder: mtl.ComputeEncoder, gate: Ref, up: Ref, dims: Ref, out: Ref, xs: Ref, planes: Ref, hidden: u32, width: u32, columns: u32, rows: u32, first: u32, groups: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .activation);
        errdefer span.cancel();
        const e = span.encoder;
        e.setPipeline(try k.get(.{ .kind = .mlp_ane, .k = hidden, .n = width, .tile = columns }));
        bind(e, .{ gate, up, dims, out, xs, planes });
        e.setValue(first, 6);
        e.dispatchGroups(mtl.Size.of(groups, padded(rows), 1), mtl.Size.of(64, 1, 1));
        try span.end();
    }

    /// Add the ANE fp16 partial to the GPU fp32 partial for these rows, rounding once to bf16.
    pub fn addAne(k: *Kernels, input_encoder: mtl.ComputeEncoder, partial: Ref, planes: Ref, y: Ref, hidden: u32, width: u32, columns: u32, rows: u32, base: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .copy);
        errdefer span.cancel();
        const e = span.encoder;
        e.setPipeline(try k.get(.{ .kind = .add_ane, .k = hidden, .n = width, .tile = columns }));
        bind(e, .{ partial, planes, y });
        e.setValue(@as(i32, @intCast(base)), 3);
        e.dispatchThreads(mtl.Size.of(hidden, rows, 1), mtl.Size.of(256, 1, 1));
        try span.end();
    }

    pub fn mlp(k: *Kernels, input_encoder: mtl.ComputeEncoder, gate: Ref, up: Ref, dims: Ref, out: Ref, xs: Ref, width: u32, rows: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .activation);
        errdefer span.cancel();
        const e = span.encoder;

        // prompt rows: 64 threads of 16 columns a threadgroup (16 groups); decode rows keep a thread a column
        const wide = rows > 16 and width % 1024 == 0;
        e.setPipeline(try k.get(.{ .kind = if (wide) .mlp_wide else .mlp, .k = width, .n = width }));
        bind(e, .{ gate, up, dims, out, xs });
        e.dispatchGroups(mtl.Size.of(width / @as(u32, if (wide) 1024 else 64), if (wide) padded(rows) else rows, 1), mtl.Size.of(64, 1, 1));
        try span.end();
    }
};

/// The metadata a reg launch binds: decoded rows' indexed copy where its key takes it, else the pairs.
fn metadata(p: projection.Linear, indexed: bool) lane.Ref {
    if (indexed) if (p.index) |i| return .{ .buf = i.buffer, .off = i.offset };
    return .{ .buf = p.pairs.buffer, .off = p.pairs.offset };
}

pub fn padded(rows: u32) u32 {
    return 16 * ((rows + 15) / 16);
}

/// q27_norm_xs moves its rows and weights in 16-byte words.
fn wordAligned(refs: anytype) !void {
    inline for (refs) |r| if (r.offset % 16 != 0) return error.BadNormAlignment;
}

fn bindAt(e: mtl.ComputeEncoder, r: Ref, index: usize) void {
    e.setBuffer(r.buffer, r.offset, index);
}
fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
}

/// Adjacent tiles a prompt-row reg SIMD group computes on one register copy of each activation group: two for the
/// split-K 5120-wide projections (MLP down, attention output) and for the 5120-deep inputs (attention query+gate, GDN qkv,
/// the GDN z columns that run beside the scan, MLP gate/up), one elsewhere. A two-tile launch runs its K splits as SIMD
/// groups of one threadgroup, so the query+gate's prompt rows leave the serial split-K kernel for it.
fn promptTiles(key: Key, n: u32, skip: u32) usize {
    if (!key.prompt_sr or skip != 0 or key.batch % 2 != 0) return 1; // a SIMD group's tiles share one code batch
    // the launch's tiles in whole pairs: a range's first tile and length, or every tile
    if (if (key.range[1] != 0) key.range[0] % 2 != 0 or key.range[1] % 2 != 0 else (n + 31) / 32 % 2 != 0) return 1;
    if (key.k == 5120 and (n == 12288 or n == 10240 or n == 6240 or n == 34816)) return 2;
    return if (n == 5120 and key.slices == 2 and (key.k == 17408 or key.k == 6144)) 2 else 1;
}
