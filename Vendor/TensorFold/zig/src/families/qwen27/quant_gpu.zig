//! Runtime specialization preserves the original M5 group arithmetic and GPU-resident argument buffers.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
const row = @import("core").row_projection;
const sources = @import("qwen_runtime_sources");
const projection = @import("projection.zig");
const profile = @import("core").gpu_profile;
pub const Ref = projection.Ref;
/// Code MiB Kernels.warm reads ahead of a decoded out projection: about two thirds of its 15 MiB, as much as a star
/// scan's or an attention prefix's idle bus time covers at the stream rate.
const warm_mb = 10;
/// Kernels.warm's threadgroups at most: a few a core, so the waiting launches behind them are placed at once.
const warm_groups = 64;
const warm_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void tf_q27_warm(const device uint* W [[buffer(0)]], const device uint* M [[buffer(1)]],
    \\    constant uint2& lines [[buffer(2)]], device uint* sink [[buffer(3)]],
    \\    uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
    \\  uint acc = 0;
    \\  for (uint j = i; j < lines.x; j += 4u * n) {
    \\    const uint a = W[(size_t)j * 32u];
    \\    const uint b = j + n < lines.x ? W[(size_t)(j + n) * 32u] : 0u;
    \\    const uint c = j + 2u * n < lines.x ? W[(size_t)(j + 2u * n) * 32u] : 0u;
    \\    const uint d = j + 3u * n < lines.x ? W[(size_t)(j + 3u * n) * 32u] : 0u;
    \\    acc ^= a ^ b ^ c ^ d;
    \\  }
    \\  for (uint j = i; j < lines.y; j += n) acc ^= M[(size_t)j * 32u];
    \\  if (acc == 0x9E3779B9u && i == 0xFFFFFFFFu) sink[0] = acc; // never true: the loads stay, nothing is stored
    \\}
    \\// One word of every 128-byte line a tile-list launch reads of the tiles a GPU list names (its count, then the tiles):
    \\// each listed tile's code block of every K group, its metadata block of every K group (its first and last word: the
    \\// lines a block of at most 128 bytes covers) and the metadata's leading table lines. Every read is inside the bytes
    \\// given; nothing is stored.
    \\struct tf_warm_tiles { uint groups, batch, block, tiles, slots, code_bytes, meta_base, meta_group, meta_tile, meta_bytes, table_bytes; };
    \\kernel void tf_q27_warm_tiles(const device uchar* W [[buffer(0)]], const device uchar* M [[buffer(1)]],
    \\    const device uint* list [[buffer(2)]], constant tf_warm_tiles& a [[buffer(3)]], device uint* sink [[buffer(4)]],
    \\    uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
    \\  uint acc = 0;
    \\  const uint items = min(list[0], a.slots) * a.groups;
    \\  for (uint it = i; it < items; it += n) {
    \\    const uint tile = list[1 + it / a.groups], g = it % a.groups;
    \\    if (tile >= a.tiles) continue;
    \\    const ulong code = (ulong(tile / a.batch) * a.groups * a.batch + tile % a.batch + ulong(g) * a.batch) * a.block;
    \\    for (uint l = 0; l < a.block; l += 128u) if (code + l + 4u <= a.code_bytes) acc ^= *(const device uint*)(W + code + l);
    \\    const ulong meta = a.meta_base + ulong(g) * a.meta_group + ulong(tile) * a.meta_tile;
    \\    if (meta + a.meta_tile <= a.meta_bytes) acc ^= *(const device uint*)(M + meta) ^ *(const device uint*)(M + meta + a.meta_tile - 4u);
    \\  }
    \\  for (uint l = i * 128u; l + 4u <= a.table_bytes; l += n * 128u) acc ^= *(const device uint*)(M + l);
    \\  if (acc == 0x9E3779B9u && i == 0xFFFFFFFFu) sink[0] = acc; // never true: the loads stay, nothing is stored
    \\}
;
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
    /// Qualified target DOWN M8 pipeline; distinct from ordinary rows8/M16 entries.
    m8: bool = false,
    count: bool = false,
    wait: bool = false,
    tiles: u32 = 0,
    parallel_sk2: bool = false,
    /// Prompt rows of the attention k|v projection (2048 x 5120, K split 8): one SIMD group a tile runs the eight K chains
    /// in turn (lane_projection.Layout.serial_sk2) instead of eight SIMD groups a threadgroup.
    serial: bool = false,

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
/// input_words: q27_norm_input in 16-byte words (team-m5-08w), from the same template with Q27_NORM_WORDS defined.
pub const Kind = enum { projection, xsum, norm, input_norm, mlp, mlp_wide, ane_input, mlp_ane, add_ane, input_words };

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
    target_down_m8_qualified: u32 = 0,
    target_down_m8_rejected: u32 = 0,
    target_down_m8_dispatches: u64 = 0,
    /// Model.load sets this only when the M8 optimization is admitted; target DOWN must not silently fall back to M16.
    target_down_m8_required: bool = false,
    target_down_m8_calibration_failed: bool = false,
    target_down_m8_probe_error_logged: bool = false,
    lanes: std.AutoHashMapUnmanaged(Key, lane.Projection) = .empty,
    cache: std.AutoHashMapUnmanaged(Key, mtl.Pipeline) = .empty,
    /// quantCounted's count word and normAfter's arrival word, zeroed when made and left zero by each waiting norm.
    count_words: ?mtl.Buffer = null,
    /// The decoded gate/up's count and the waiting down's arrivals (encodeSwigluCounted / encodeCountedAfter), left zero.
    gu_words: ?mtl.Buffer = null,
    /// warm's read-only line touch and the word it could store (it never does), made on first use.
    warm_pipe: ?mtl.Pipeline = null,
    warm_sink: ?mtl.Buffer = null,
    /// Code bytes warm reads ahead of a decoded out projection (warm_mb MiB), set on first use.
    warm_bytes: ?usize = null,
    /// warmTiles' line touch (tf_q27_warm_tiles) and the word it could store (it never does), made at load
    /// (prepareWarmTiles).
    warm_tiles_pipe: ?mtl.Pipeline = null,
    warm_tiles_sink: ?mtl.Buffer = null,
    /// team-m5-08w: the 16-byte input norm failed to build on this chip (logged once): the record's kernel runs instead.
    input_words_failed: bool = false,

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
        if (k.target_down_m8_qualified != 0 or k.target_down_m8_rejected != 0 or k.target_down_m8_dispatches != 0) {
            std.log.info("qwen27: target-down M8 qualified={d} rejected={d} encoded_dispatches={d}", .{ k.target_down_m8_qualified, k.target_down_m8_rejected, k.target_down_m8_dispatches });
        }
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
        if (k.warm_pipe) |p| p.deinit();
        if (k.warm_sink) |b| b.deinit();
        if (k.warm_tiles_pipe) |p| p.deinit();
        if (k.warm_tiles_sink) |b| b.deinit();
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
            .input_norm, .input_words => sources.norm_input,
            .mlp => sources.mlp_xs,
            .mlp_wide => sources.mlp_xs_wide,
            .ane_input, .mlp_ane, .add_ane => sources.mlp_ane,
        };
        const mark = std.mem.indexOf(u8, template, "Q27_SPECIALIZE") orelse return error.BadKernelSource;
        const definitions = switch (key.kind) {
            .projection => unreachable,
            .xsum => try std.fmt.allocPrint(k.allocator, "constexpr int K={d}, GS=64;", .{key.k}),
            .norm, .input_norm => try std.fmt.allocPrint(k.allocator, "constexpr int K={d};", .{key.k}),
            .input_words => try std.fmt.allocPrint(k.allocator, "constexpr int K={d};\n#define Q27_NORM_WORDS 1", .{key.k}),
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
            .input_norm, .input_words => "custom_kernel_q27_norm_input_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_float",
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
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = if (reg) sk else p.slices, .skip = skip, .cut = cut, .range = range, .bits = p.bits, .group = p.group, .rows8 = reg and !k.prompt and rows <= 8, .lean = reg, .batch = if (reg) p.batch else 1, .prompt_sr = reg and k.prompt, .sbi = reg and !k.prompt and p.index != null, .sbi12 = reg and !k.prompt and p.index != null and p.index12, .parallel_sk2 = reg and sk == 2 and skip == 0 and cut[0] == 0 and cut[1] == 0 and range[1] == 0 and wideDecodeSk2(p.n, p.k, rows, k.prompt), .serial = reg and k.prompt and p.n == 2048 and p.k == 5120 and sk == 8 };
        const sums: ?Sums = if (reg) xs else null; // the non-reg kernel sums its own groups
        const half = p.n / 64;
        const ranges = [_][2]usize{ .{ skip, half - skip }, .{ half + skip, half - skip } };
        const tiles = [_][2]usize{.{ range[0], range[1] }};
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            const tn = promptTiles(key, p.n, skip);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = p.bits, .group = p.group }, .sk = key.slices, .ranges = if (skip > 0) &ranges else if (range[1] > 0) &tiles else &.{}, .groups = if (cut[1] > 0) .{ cut[0], cut[1] } else null, .output = if (cut[1] > 0) .f32 else .bf16, .precompute_sums = reg, .reg = reg, .serial_sk2 = (key.serialSk2() or key.serial) and tn == 1, .prompt_sr = key.prompt_sr, .tile_batch = key.batch, .rows8 = key.rows8, .reg_lean = key.lean, .prompt_tn = tn, .indexed = key.sbi, .indexed12 = key.sbi12 });
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
            .indexed = key.sbi,
            .indexed12 = key.sbi12,
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

        e.setPipeline(if (residual != null) try k.get(.{ .kind = .norm, .k = width }) else try k.inputNorm(h, weight, out, width));
        if (residual) |r| {
            bind(e, .{ h, r, weight, eps, dims, combined, out, xs });
            bindAt(e, out, 8); // never read: wait_for 0
            e.setValue(@as(u32, 0), 9);
        } else bind(e, .{ h, weight, eps, dims, out, xs });
        e.dispatchGroups(mtl.Size.of(1, rows, 1), mtl.Size.of(width / 16, 1, 1));
        try span.end();
    }

    /// team-m5-08w: the input norm's pipeline. q27_norm_input in 16-byte words where its row, its weights and its output
    /// start 16-byte aligned (every call on this engine's frames and weights), else the record's one-word kernel, which
    /// also stands in, logged once, if the 16-byte text fails to build on this chip.
    fn inputNorm(k: *Kernels, h: Ref, weight: Ref, out: Ref, width: u32) !mtl.Pipeline {
        const aligned = width % 8 == 0 and h.offset % 16 == 0 and weight.offset % 16 == 0 and out.offset % 16 == 0;
        if (aligned and !k.input_words_failed) {
            if (k.get(.{ .kind = .input_words, .k = width })) |p| return p else |err| {
                k.input_words_failed = true;
                std.log.warn("qwen27: the input norm keeps its one-word kernel (team-m5-08w): {s}", .{@errorName(err)});
            }
        }
        return k.get(.{ .kind = .input_norm, .k = width });
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
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = p.slices, .bits = p.bits, .group = p.group, .rows8 = true, .lean = true, .batch = p.batch, .sbi = p.index != null, .sbi12 = p.index != null and p.index12, .m8 = wait and p.m8_qualified and (p.m8_row_mask & (@as(u8, 1) << @as(u3, @intCast(rows - 1)))) != 0 and targetDownM8Shape(p), .count = true, .wait = wait };
        if (key.serialSk2() and wait) return null;
        if (!key.serialSk2() and p.slices < 2) return null;
        return key;
    }

    test "qualified M8 stays on counted target DOWN rows 1 through 8 with distinct wait keys" {
        var k = Kernels{ .allocator = std.testing.allocator, .device = undefined };
        var p = projection.Linear{
            .words = undefined,
            .pairs = undefined,
            .n = 5120,
            .k = 17408,
            .tile = 32,
            .slices = 4,
            .m8_qualified = false,
        };
        try std.testing.expect(targetDownM8Shape(p));
        for (1..9) |rows| {
            p.m8_qualified = false;
            const unqualified = countedKey(&k, p, @intCast(rows), false).?;
            try std.testing.expect(!unqualified.m8);
            p.m8_qualified = true;
            p.m8_row_mask = 0xFF;
            const no_wait = countedKey(&k, p, @intCast(rows), false).?;
            const wait = countedKey(&k, p, @intCast(rows), true).?;
            try std.testing.expect(!no_wait.m8 and wait.m8);
            try std.testing.expect(!std.meta.eql(no_wait, wait));
            p.m8_row_mask = 0;
            try std.testing.expect(!countedKey(&k, p, @intCast(rows), true).?.m8);
            p.m8_row_mask = @as(u8, 1) << @as(u3, @intCast(rows - 1));
            try std.testing.expect(countedKey(&k, p, @intCast(rows), true).?.m8);
        }
        try std.testing.expect(countedKey(&k, p, 9, false) == null);
        k.prompt = true;
        try std.testing.expect(countedKey(&k, p, 1, false) == null);
        k.prompt = false;
        p.m8_qualified = true;
        p.n = 34816;
        try std.testing.expect(!targetDownM8Shape(p));
        try std.testing.expect(!countedKey(&k, p, 1, false).?.m8);
    }

    fn targetDownM8Shape(p: projection.Linear) bool {
        return p.n == 5120 and p.k == 17408 and p.bits == 4 and p.group == 64 and p.tile == 32 and p.slices >= 2 and p.runs.len == 0 and p.raw == null;
    }

    /// Called during model load for each real target DOWN weight. Any probe failure keeps the M16 key.
    pub fn qualifyTargetDownM8(k: *Kernels, queue: mtl.Queue, p: *projection.Linear) bool {
        p.m8_qualified = false;
        p.m8_row_mask = 0;
        if (!targetDownM8Shape(p.*)) return false;
        const qualified = k.probeTargetDownM8(queue, p.*) catch |err| blk: {
            if (!k.target_down_m8_probe_error_logged) {
                k.target_down_m8_probe_error_logged = true;
                std.log.warn("qwen27: target-down M8 numeric qualification failed for {d}x{d} Q4/G{d}: {s}", .{ p.n, p.k, p.group, @errorName(err) });
            }
            break :blk false;
        };
        p.m8_qualified = qualified;
        if (qualified) k.target_down_m8_qualified += 1 else k.target_down_m8_rejected += 1;
        return qualified;
    }

    /// All real DOWN matrices run in each calibration cohort before any request can select M8.
    pub fn calibrateTargetDownM8(k: *Kernels, queue: mtl.Queue, down: []const projection.Linear) u8 {
        const calibration = @import("decode_m8_calibration.zig");
        k.target_down_m8_calibration_failed = false;
        if (down.len != calibration.matrices) {
            k.target_down_m8_calibration_failed = true;
            return 0;
        }
        var programs: [calibration.matrices]calibration.Program = undefined;
        for (down, 0..) |p, i| {
            if (!p.m8_qualified or !targetDownM8Shape(p)) {
                k.target_down_m8_calibration_failed = true;
                return 0;
            }
            var baseline_key = countedKey(k, p, 8, true) orelse {
                k.target_down_m8_calibration_failed = true;
                return 0;
            };
            baseline_key.m8 = false;
            var candidate_key = baseline_key;
            candidate_key.m8 = true;
            // Copy borrowed handles before another insertion can grow the pipeline map.
            const baseline = (countedProjection(k, p, baseline_key) catch {
                k.target_down_m8_calibration_failed = true;
                return 0;
            }).*;
            const candidate = (countedProjection(k, p, candidate_key) catch {
                k.target_down_m8_calibration_failed = true;
                return 0;
            }).*;
            programs[i] = .{
                .layer = i,
                .baseline = baseline,
                .candidate = candidate,
                .words = .{ .buf = p.words.buffer, .off = p.words.offset },
                .metadata = metadata(p, p.index != null),
            };
        }
        var row_mask: u8 = 0;
        for (1..9) |live_rows| {
            const evidence = calibration.run(k.device, queue, &programs, @intCast(live_rows)) catch |err| {
                k.target_down_m8_calibration_failed = true;
                std.log.info("qwen27: target-down M8 rows={d} kept M16 ({s})", .{ live_rows, @errorName(err) });
                continue;
            };
            const decision = evidence.decision;
            std.log.info("qwen27: target-down M8 rows={d} adopted={} relative={d:.6} saved={d:.9} positive={d} null={d:.6}", .{
                live_rows, decision.use_m8, decision.median_relative_gain, decision.median_saving_seconds, decision.positive_pairs, decision.max_null_relative,
            });
            if (decision.use_m8) row_mask |= @as(u8, 1) << @as(u3, @intCast(live_rows - 1));
        }
        return row_mask;
    }

    /// A shared constructor keeps qualification and production on the same complete counted/wait cache key.
    fn countedProjection(k: *Kernels, p: projection.Linear, key: Key) !*lane.Projection {
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{
                .n = p.n,
                .k = p.k,
                .format = .{ .bits = p.bits, .group = p.group },
                .sk = key.slices,
                .precompute_sums = true,
                .reg = true,
                .tile_batch = key.batch,
                .rows8 = true,
                .reg_lean = true,
                .indexed = key.sbi,
                .indexed12 = key.sbi12,
                .m8 = key.m8,
                .count = true,
                .wait = key.wait,
                .serial_sk2 = key.serialSk2(),
            });
            k.joined(entry.value_ptr);
        }
        return entry.value_ptr;
    }

    fn probeTargetDownM8(k: *Kernels, queue: mtl.Queue, p: projection.Linear) !bool {
        @setFloatMode(.strict);
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const rows_max: usize = 8;
        const sums_stride: usize = 16;
        const groups: usize = p.k / 64;
        const prefix_bytes: usize = 64;
        const suffix_bytes: usize = 64;
        const x_buf = try k.device.buffer(rows_max * p.k * 2, options);
        defer x_buf.deinit();
        const xs_buf = try k.device.buffer(groups * sums_stride * 4, options);
        defer xs_buf.deinit();
        const y_size = prefix_bytes + rows_max * p.n * 2 + suffix_bytes;
        const y16 = try k.device.buffer(y_size, options);
        defer y16.deinit();
        const y8 = try k.device.buffer(y_size, options);
        defer y8.deinit();
        const tc16 = try k.device.buffer(8, options);
        defer tc16.deinit();
        const tc8 = try k.device.buffer(8, options);
        defer tc8.deinit();
        const wc16 = try k.device.buffer(8, options);
        defer wc16.deinit();
        const wc8 = try k.device.buffer(8, options);
        defer wc8.deinit();

        var base_count = countedKey(k, p, 1, false) orelse return error.M8ProbeCountKeyUnavailable;
        base_count.m8 = false;
        var test_count = base_count;
        test_count.m8 = true;
        var base_wait = countedKey(k, p, 1, true) orelse return error.M8ProbeWaitKeyUnavailable;
        base_wait.m8 = false;
        var test_wait = base_wait;
        test_wait.m8 = true;
        // Copy each handle before another getOrPut can rehash the map and invalidate its value pointer.
        const p16 = (try countedProjection(k, p, base_count)).*;
        const p8 = (try countedProjection(k, p, test_count)).*;
        const p16w = (try countedProjection(k, p, base_wait)).*;
        const p8w = (try countedProjection(k, p, test_wait)).*;
        const xref = lane.Ref{ .buf = x_buf, .off = 0 };
        const sums = lane.Sums{ .ref = .{ .buf = xs_buf, .off = 0 }, .stride = sums_stride };
        const wref = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
        const sbref = metadata(p, p.index != null);
        const yrefs = [_]lane.Ref{ .{ .buf = y16, .off = prefix_bytes }, .{ .buf = y8, .off = prefix_bytes } };
        var state: u32 = 0;
        for ([_]u32{ 0x9E3779B9, 0xA511E9B3 }) |seed| {
            state = seed;
            const xv = x_buf.slice(u16, rows_max * p.k);
            const sv = xs_buf.slice(f32, groups * sums_stride);
            @memset(sv, 0);
            for (0..rows_max) |r| for (0..p.k) |c| {
                state = state *% 1664525 +% 1013904223;
                const f = (@as(f32, @floatFromInt(state >> 8)) / 16777216.0) * 4.0 - 2.0;
                xv[r * p.k + c] = @truncate(@as(u32, @bitCast(f)) >> 16);
            };
            for (0..groups) |g| for (0..rows_max) |r| {
                var group_sum: f32 = 0;
                for (0..64) |i| group_sum += @as(f32, @bitCast(@as(u32, xv[r * p.k + g * 64 + i]) << 16));
                sv[g * sums_stride + r] = group_sum;
            };
            for (1..rows_max + 1) |rows| {
                const used = rows * p.n * 2;
                const variants = [_]struct { proj: lane.Projection, yi: usize, wait: bool, expected: u8 }{
                    .{ .proj = p16, .yi = 0, .wait = false, .expected = 0xA5 },
                    .{ .proj = p8, .yi = 1, .wait = false, .expected = 0x5A },
                    .{ .proj = p16w, .yi = 0, .wait = true, .expected = 0xA5 },
                    .{ .proj = p8w, .yi = 1, .wait = true, .expected = 0x5A },
                };
                for (variants, 0..) |v, variant_index| {
                    const yb = if (v.yi == 0) y16 else y8;
                    const tc = if (v.yi == 0) tc16 else tc8;
                    const wc = if (v.yi == 0) wc16 else wc8;
                    @memset(yb.contents()[0..y_size], v.expected);
                    @memset(yb.contents()[prefix_bytes .. y_size - suffix_bytes], 0xFF);
                    @memset(tc.contents()[0..8], 0);
                    @memset(wc.contents()[0..8], 0);
                    if (v.wait) wc.slice(u32, 2)[0] = 1;
                    const pool = mtl.objc.Pool.push();
                    defer pool.pop();
                    const cb = queue.commandBuffer();
                    const e = cb.compute(.serial);
                    var encoder_closed = false;
                    defer if (!encoder_closed) e.end();
                    if (v.wait) {
                        try v.proj.encodeCountedAfter(e, xref, wref, sbref, sums, yrefs[v.yi], @intCast(rows), .{ .buf = tc, .off = 0 }, .{ .buf = wc, .off = 0 }, 1);
                    } else {
                        try v.proj.encodeCounted(e, xref, wref, sbref, sums, yrefs[v.yi], @intCast(rows), .{ .buf = tc, .off = 0 });
                    }
                    e.end();
                    encoder_closed = true;
                    cb.commit();
                    cb.wait();
                    if (cb.status() != .completed) return error.M8ProbeCommandFailed;
                    if (tc.slice(u32, 2)[0] != @as(u32, @intCast(v.proj.tile_count)) or tc.slice(u32, 2)[1] != 0) return error.M8ProbeCountMismatch;
                    if (v.wait and (wc.slice(u32, 2)[0] != 0 or wc.slice(u32, 2)[1] != 0)) return error.M8ProbeWaitMismatch;
                    const live_start = prefix_bytes;
                    const live_end = live_start + used;
                    for (yb.slice(u16, y_size / 2)[live_start / 2 .. live_end / 2]) |word| if ((word & 0x7f80) == 0x7f80) return error.M8ProbeNonfiniteOutput;
                    for (yb.contents()[0..prefix_bytes]) |byte| if (byte != v.expected) return error.M8ProbePrefixOverwrite;
                    for (yb.contents()[live_end .. y_size - suffix_bytes]) |byte| if (byte != 0xFF) return error.M8ProbePaddingOverwrite;
                    for (yb.contents()[y_size - suffix_bytes .. y_size]) |byte| if (byte != v.expected) return error.M8ProbeSuffixOverwrite;
                    if (variant_index == 1 or variant_index == 3) {
                        if (!std.mem.eql(u8, y16.contents()[live_start..live_end], y8.contents()[live_start..live_end])) return error.M8ProbeOutputMismatch;
                    }
                }
            }
        }
        return true;
    }

    /// Reads one word of every 128-byte line of the first warmBytes() of `p`'s code words (the stored layout, tile batches
    /// and padding included) and of all its decoded metadata (the indexed table and indices, else the pairs), with no
    /// barrier. Encoded inside a latency-bound barrier interval (a star scan, an attention prefix), it moves those lines
    /// from DRAM into the system cache while the memory bus is idle, just before the decoded launch of `p` reads them.
    /// It writes nothing anything reads, so it adds no hazard and changes no word.
    pub fn warm(k: *Kernels, e: mtl.ComputeEncoder, p: projection.Linear) !void {
        return k.warmPrefix(e, p, k.warmBytes());
    }

    /// warm of all of `p`'s code words (warmBytes() 0 still turns it off): for a window long enough for a whole matrix.
    pub fn warmAll(k: *Kernels, e: mtl.ComputeEncoder, p: projection.Linear) !void {
        if (k.warmBytes() == 0) return;
        return k.warmPrefix(e, p, std.math.maxInt(usize));
    }

    fn warmPrefix(k: *Kernels, e: mtl.ComputeEncoder, p: projection.Linear, want: usize) !void {
        if (want == 0 or p.bits != 4 or p.group != 64 or p.runs.len != 0) return;
        const codes = projection.storedColumns(p.n, p.batch) * p.k / 2;
        const meta: projection.Ref = p.index orelse p.pairs;
        const meta_bytes: usize = if (p.index == null) @as(usize, p.n) * (p.k / 64) * 4 else if (p.index12) lane.indexed12Bytes(p.n, p.k, 64) else lane.indexedBytes(p.n, p.k, 64);
        return k.warmLines(e, p.words, @min(want, codes), meta, meta_bytes);
    }

    /// tf_q27_warm over the leading `code_bytes` of `words` and `meta_bytes` of `meta`, each cut at its buffer's end.
    fn warmLines(k: *Kernels, e: mtl.ComputeEncoder, words: Ref, code_bytes: usize, meta: Ref, meta_bytes: usize) !void {
        const take = @min(code_bytes, words.buffer.length() -| words.offset);
        const meta_take = @min(meta_bytes, meta.buffer.length() -| meta.offset);
        if (take < 128) return;
        if (k.warm_pipe == null) {
            const library = try mtl.Library.fromSource(k.device, warm_source, mtl.CompileOptions.mlx());
            defer library.deinit();
            const pipe = try mtl.Pipeline.init(k.device, library, "tf_q27_warm", false);
            errdefer pipe.deinit();
            k.warm_sink = try k.device.buffer(16, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            k.warm_pipe = pipe;
        }
        const lines = [2]u32{ @intCast(take / 128), @intCast(meta_take / 128) };
        // four independent line loads a thread a pass keep enough lines in flight to stream at the bus's rate
        const groups: usize = std.math.clamp((lines[0] / 4 + 255) / 256, 1, warm_groups);
        e.setPipeline(k.warm_pipe.?);
        e.setBuffer(words.buffer, words.offset, 0);
        e.setBuffer(meta.buffer, meta.offset, 1);
        e.setValue(lines, 2);
        e.setBuffer(k.warm_sink.?, 0, 3);
        e.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// Compiles at load warmTiles' pipeline (tf_q27_warm_tiles, in warm's source) and makes its sink.
    pub fn prepareWarmTiles(k: *Kernels) !void {
        if (k.warm_tiles_pipe != null) return;
        const library = try mtl.Library.fromSource(k.device, warm_source, mtl.CompileOptions.mlx());
        defer library.deinit();
        const pipe = try mtl.Pipeline.init(k.device, library, "tf_q27_warm_tiles", false);
        errdefer pipe.deinit();
        k.warm_tiles_sink = try k.device.buffer(16, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        k.warm_tiles_pipe = pipe;
    }

    /// Encoded in the barrier interval of a quantTiles launch of `p` over `list` (its count, then its tiles; at most
    /// `slots`), behind the barrier that orders the list's writer: one word of every 128-byte line that launch reads of the
    /// listed tiles (each tile's code block and metadata block of every K group) and of the metadata's index table, all
    /// requested at once. The tile launch runs one SIMD group a tile, each walking its tile's K groups with one group of
    /// codes and indices prefetched (a few KB in flight a launch), so each group waits about one memory round trip; with the
    /// lines on their way into the caches it finds them there. It writes nothing anything reads, so it changes no word.
    /// A no-op until prepareWarmTiles made its pipeline; it touches the encoder only to encode its one dispatch.
    pub fn warmTiles(k: *Kernels, e: mtl.ComputeEncoder, p: projection.Linear, list: Ref, slots: u32) !void {
        const pipe = k.warm_tiles_pipe orelse return;
        if (!ranged(p) or slots == 0 or p.batch == 0) return;
        const meta = metadata(p, p.index != null);
        const n: usize = p.n;
        // [first block's offset, a K group's stride, a tile's block, table bytes] of the metadata quantTiles binds (the
        // lean reg loop's TF_SBI12 / TF_SBI / pairs addressing)
        const form: [4]usize = if (p.index == null) .{ 0, 4 * n, 128, 0 } else if (p.index12) .{ lane.sb_table12 * 4, 3 * n / 2, 48, lane.sb_table12 * 4 } else .{ lane.sb_table * 4, 2 * n, 64, lane.sb_table * 4 };
        const code_bytes = p.words.buffer.length() -| p.words.offset;
        const meta_bytes = meta.buf.length() -| meta.off;
        const most = std.math.maxInt(u32);
        if (code_bytes > most or meta_bytes > most or form[1] > most) return;
        const groups = p.k / 64;
        // a tile's code block of a K group: 32 columns of 64 4-bit codes
        const args = [11]u32{ groups, p.batch, 32 * 64 / 2, (p.n + 31) / 32, slots, @intCast(code_bytes), @intCast(form[0]), @intCast(form[1]), @intCast(form[2]), @intCast(meta_bytes), @intCast(@min(form[3], meta_bytes)) };
        const threads = @max(@as(usize, slots) * groups, form[3] / 128);
        e.setPipeline(pipe);
        e.setBuffer(p.words.buffer, p.words.offset, 0);
        e.setBuffer(meta.buf, meta.off, 1);
        e.setBuffer(list.buffer, list.offset, 2);
        e.setValue(args, 3);
        e.setBuffer(k.warm_tiles_sink.?, 0, 4);
        e.dispatchGroups(mtl.Size.of(std.math.clamp((threads + 255) / 256, 1, warm_groups), 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn warmBytes(k: *Kernels) usize {
        if (k.warm_bytes) |b| return b;
        k.warm_bytes = @as(usize, warm_mb) << 20;
        return k.warm_bytes.?;
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
        const entry = try k.countedProjection(p, key);
        const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
        try entry.encodeCounted(span.encoder, .{ .buf = x.buffer, .off = x.offset }, w, metadata(p, key.sbi), .{ .ref = .{ .buf = xs.ref.buffer, .off = xs.ref.offset }, .stride = xs.stride }, .{ .buf = y.buffer, .off = y.offset }, rows, .{ .buf = k.count_words.? });
        try span.end();
        if (key.m8) k.target_down_m8_dispatches += 1;
        return @intCast(entry.tile_count);
    }

    /// quantCounted for a projection whose input rows and sums come from a gateUpCounted launch in the same barrier interval:
    /// its threadgroups wait on that launch's `wait_for` threadgroups before they read them.
    pub fn quantCountedAfter(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: Sums, y: Ref, rows: u32, wait_for: u32) !u32 {
        const key = try k.targetDownM8CallKey(p, rows);
        const gu = k.gu_words orelse return error.BadCoreProjectionRows;
        if (k.count_words == null) {
            const b = try k.device.buffer(8, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            @memset(b.contents()[0..8], 0);
            k.count_words = b;
        }
        var span = profile.begin(k.profiler, input_encoder, .dense);
        span.shape = .{ .n = p.n, .k = p.k, .sk = p.slices, .bits = p.bits };
        errdefer span.cancel();
        const entry = try k.countedProjection(p, key);
        const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
        try entry.encodeCountedAfter(span.encoder, .{ .buf = x.buffer, .off = x.offset }, w, metadata(p, key.sbi), .{ .ref = .{ .buf = xs.ref.buffer, .off = xs.ref.offset }, .stride = xs.stride }, .{ .buf = y.buffer, .off = y.offset }, rows, .{ .buf = k.count_words.? }, .{ .buf = gu }, wait_for);
        try span.end();
        if (key.m8) k.target_down_m8_dispatches += 1;
        return @intCast(entry.tile_count);
    }

    /// Resolve and enforce the exact target DOWN row-stratum selected at model load. Kept pure for direct host tests.
    fn targetDownM8CallKey(k: *const Kernels, p: projection.Linear, rows: u32) !Key {
        const key = k.countedKey(p, rows, true) orelse return error.BadCoreProjectionRows;
        if (k.target_down_m8_required and targetDownM8Shape(p) and !key.m8) return error.TargetDownM8NotSelected;
        return key;
    }

    /// Decoded rows: the fused gate/up's lockstep launch (gateUpActivated's) counting its threadgroups once its activated
    /// rows and their group sums are stored, when `down` takes the counted launch that waits on them in the same barrier
    /// interval (quantCountedAfter); null (nothing encoded) otherwise.
    pub fn gateUpCounted(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, down: projection.Linear, x: Ref, xs: Sums, out: Ref, out_sums: Ref, rows: u32) !?u32 {
        if (!try k.gateUpCountedEligible(p, down, rows)) return null;
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = p.slices, .bits = p.bits, .group = p.group, .rows8 = true, .lean = true, .batch = p.batch, .swiglu = true, .sbi = true, .sbi12 = p.index12, .count = true };
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

    /// Preserve the legacy optional fast path unless this model admitted target DOWN M8. In that mode, a normal decoded
    /// target call must reach the exact counted-wait kernel with a calibrated M8 row key; otherwise fail before touching
    /// buffers or the caller's encoder instead of silently taking the M16 fallback.
    fn gateUpCountedEligible(k: *const Kernels, p: projection.Linear, down: projection.Linear, rows: u32) !bool {
        const required_target = k.target_down_m8_required and !k.prompt and rows > 0 and rows <= 8 and targetDownM8Shape(down);
        if (k.prompt or rows == 0 or rows > 8) return false;
        const gate_eligible = p.runs.len == 0 and p.raw == null and p.tile == 32 and p.bits == 4 and p.group == 64 and p.n == 34816 and p.k == 5120 and p.batch == 8 and p.index != null;
        if (!gate_eligible) {
            if (required_target) return error.TargetDownM8GateUpUnavailable;
            return false;
        }
        const down_key = k.countedKey(down, rows, true) orelse {
            if (required_target) return error.TargetDownM8GateUpUnavailable;
            return false;
        };
        if (required_target and !down_key.m8) return error.TargetDownM8NotSelected;
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = p.slices, .bits = p.bits, .group = p.group, .rows8 = true, .lean = true, .batch = p.batch, .swiglu = true, .sbi = true, .sbi12 = p.index12, .count = true };
        if (!key.serialSk2()) {
            if (required_target) return error.TargetDownM8GateUpUnavailable;
            return false;
        }
        return true;
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

test "target DOWN M8 admission selector: selected, unselected, non-target, disabled and invalid rows" {
    var k = Kernels{ .allocator = std.testing.allocator, .device = undefined, .target_down_m8_required = true };
    var p = projection.Linear{
        .words = undefined,
        .pairs = undefined,
        .n = 5120,
        .k = 17408,
        .tile = 32,
        .slices = 4,
        .m8_qualified = true,
        .m8_row_mask = 0b0000_0010,
    };
    try std.testing.expect((try k.targetDownM8CallKey(p, 2)).m8);
    try std.testing.expectError(error.TargetDownM8NotSelected, k.targetDownM8CallKey(p, 1));
    try std.testing.expectError(error.TargetDownM8NotSelected, k.targetDownM8CallKey(p, 8));
    p.n = 34816;
    try std.testing.expect(!(try k.targetDownM8CallKey(p, 1)).m8);
    p.n = 5120;
    k.target_down_m8_required = false;
    try std.testing.expect(!(try k.targetDownM8CallKey(p, 1)).m8);
    k.target_down_m8_required = true;
    try std.testing.expectError(error.BadCoreProjectionRows, k.targetDownM8CallKey(p, 0));
    try std.testing.expectError(error.BadCoreProjectionRows, k.targetDownM8CallKey(p, 9));
}

test "target DOWN M8 required gate/up entry rejects fallback before encoder access" {
    var k = Kernels{ .allocator = std.testing.allocator, .device = undefined, .target_down_m8_required = true };
    const down = projection.Linear{
        .words = undefined,
        .pairs = undefined,
        .n = 5120,
        .k = 17408,
        .tile = 32,
        .slices = 4,
        .m8_qualified = true,
        .m8_row_mask = 0,
    };
    var gate = projection.Linear{
        .words = undefined,
        .pairs = undefined,
        .index = .{ .buffer = undefined },
        .n = 34816,
        .k = 5120,
        .tile = 32,
        .slices = 2,
        .batch = 8,
    };
    try std.testing.expectError(error.TargetDownM8NotSelected, k.gateUpCounted(undefined, gate, down, undefined, undefined, undefined, undefined, 2));
    try std.testing.expectError(error.TargetDownM8NotSelected, k.quantCountedAfter(undefined, down, undefined, undefined, undefined, 2, 1));
    gate.n = 0;
    try std.testing.expectError(error.TargetDownM8GateUpUnavailable, k.gateUpCounted(undefined, gate, down, undefined, undefined, undefined, undefined, 2));
    gate.n = 34816;
    gate.slices = 1;
    var selected_down = down;
    selected_down.m8_row_mask = 1 << 1;
    try std.testing.expectError(error.TargetDownM8GateUpUnavailable, k.gateUpCounted(undefined, gate, selected_down, undefined, undefined, undefined, undefined, 2));
    gate.slices = 2;
    k.target_down_m8_required = false;
    gate.n = 0;
    try std.testing.expect((try k.gateUpCounted(undefined, gate, down, undefined, undefined, undefined, undefined, 2)) == null);
    k.target_down_m8_required = true;
    k.prompt = true;
    gate.n = 34816;
    try std.testing.expect((try k.gateUpCounted(undefined, gate, down, undefined, undefined, undefined, undefined, 2)) == null);
}

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
