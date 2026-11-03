//! Encode exact BF16 per-row top-k into caller-owned output buffers.
const std = @import("std");
const mtl = @import("metal");
pub const contract = @import("bf16_topk");
pub const source = @import("bf16_topk_sources").text;
/// bf16_topk.metal with its TF_TOPK_FOLD section (tf_bf16_topk_wide_tiles): a library of its own, so the kernels above
/// build alone if it does not (Ops.buildFold).
pub const fold_source = "#define TF_TOPK_FOLD 1\n" ++ source;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
/// Spans a wide row is split into (TF_TOPK_SPANS in the kernel) and the narrowest row split.
const spans = 16;
const wide_vocab = 4096;
/// Spans of the register-held kernels (TF_TOPK_WIDE_SPANS) and the tokens a lane holds there (TF_TOPK_HOLD): rows of at
/// most wide_spans * lanes * hold tokens take them.
const wide_spans = 64;
const hold = 8;
/// Threads per threadgroup, in every dispatch (TF_TOPK_LANES in the kernel).
const lanes = 256;
/// tf_bf16_topk_tiles' tile marks (32 a word) and slots: a row of up to 8192 tiles, max_rows rows' tiles and tile 0.
const tile_words = 256;
const tile_slots = 272;
comptime {
    @setEvalBranchQuota(100_000);
    // the kernel's arrays, strides, tournament and entry counts are all written with these
    for (.{ .{ "TF_TOPK_LANES", lanes }, .{ "TF_TOPK_MAX_K", contract.max_k }, .{ "TF_TOPK_SPANS", spans }, .{ "TF_TOPK_TILE_WORDS", tile_words }, .{ "TF_TOPK_TILE_SLOTS", tile_slots }, .{ "TF_TOPK_WIDE_SPANS", wide_spans }, .{ "TF_TOPK_HOLD", hold } }) |named| {
        const define = std.fmt.comptimePrint("#define {s} {d}\n", .{ named[0], named[1] });
        if (std.mem.indexOf(u8, source, define) == null) @compileError("bf16_topk.metal's " ++ named[0] ++ " differs from the Zig constant");
    }
    // the merge kernel gives each span's max_k entries one lane, and the tournament halves the lanes
    if (spans * contract.max_k != lanes or !std.math.isPowerOfTwo(lanes)) @compileError("bf16 top-k merge needs spans * max_k == lanes, a power of two");
    if (contract.max_rows * contract.max_k + 1 > tile_slots) @compileError("bf16 top-k tiles need a slot for every row's k tiles and tile 0");
}
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
pub const Ops = struct {
    pipeline: mtl.Pipeline,
    max_pipeline: ?mtl.Pipeline = null,
    span_pipeline: mtl.Pipeline,
    merge_pipeline: mtl.Pipeline,
    tiles_pipeline: mtl.Pipeline,
    wide_pipeline: mtl.Pipeline,
    wide_merge_pipeline: mtl.Pipeline,
    /// Every span's top k, between a wide row's two dispatches: one wide top-k may be in flight per Ops.
    part: mtl.Buffer,
    /// encodeWideTiles' launch (buildFold) and its arrival counts (the rows' word, then a word a row): zero between
    /// launches, as each word's last arrival leaves it, and the launches that share them are ordered.
    fold_pipeline: ?mtl.Pipeline = null,
    fold_counts: ?mtl.Buffer = null,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk", false);
        errdefer pipeline.deinit();
        const max_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_max", false);
        errdefer max_pipeline.deinit();
        const span_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_span", false);
        errdefer span_pipeline.deinit();
        const merge_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_merge", false);
        errdefer merge_pipeline.deinit();
        const tiles_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_tiles", false);
        errdefer tiles_pipeline.deinit();
        const wide_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_wide", false);
        errdefer wide_pipeline.deinit();
        const wide_merge_pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_wide_merge", false);
        errdefer wide_merge_pipeline.deinit();
        if (wide_pipeline.maxThreads() < lanes or wide_merge_pipeline.maxThreads() < lanes) return error.UnsupportedTopKPipeline;
        const part = try device.buffer(contract.max_rows * @max(spans, wide_spans) * @sizeOf(contract.Row), options);
        return .{ .pipeline = pipeline, .max_pipeline = max_pipeline, .span_pipeline = span_pipeline, .merge_pipeline = merge_pipeline, .tiles_pipeline = tiles_pipeline, .wide_pipeline = wide_pipeline, .wide_merge_pipeline = wide_merge_pipeline, .part = part };
    }
    pub fn deinit(ops: Ops) void {
        ops.pipeline.deinit();
        if (ops.max_pipeline) |pipeline| pipeline.deinit();
        ops.span_pipeline.deinit();
        ops.merge_pipeline.deinit();
        ops.tiles_pipeline.deinit();
        ops.wide_pipeline.deinit();
        ops.wide_merge_pipeline.deinit();
        ops.part.deinit();
        if (ops.fold_pipeline) |pipeline| pipeline.deinit();
        if (ops.fold_counts) |counts| counts.deinit();
    }
    /// Builds encodeWideTiles' launch from fold_source; errors leave `ops` as it was.
    pub fn buildFold(ops: *Ops, device: mtl.Device) !void {
        const lib = try mtl.Library.fromSource(device, fold_source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk_wide_tiles", false);
        errdefer pipeline.deinit();
        if (pipeline.maxThreads() < lanes) return error.UnsupportedTopKPipeline;
        const counts = try device.buffer((contract.max_rows + 1) * 4, options);
        @memset(counts.slice(u32, contract.max_rows + 1), 0);
        ops.dropFold();
        ops.fold_pipeline = pipeline;
        ops.fold_counts = counts;
    }
    /// Drops encodeWideTiles' launch (encode then encodeTiles run instead).
    pub fn dropFold(ops: *Ops) void {
        if (ops.fold_pipeline) |pipeline| pipeline.deinit();
        if (ops.fold_counts) |counts| counts.deinit();
        ops.fold_pipeline = null;
        ops.fold_counts = null;
    }
    /// Whether encodeWideTiles takes `p`: a built fold, and rows encode gives the wide path (more than one pick, a row of
    /// wide_vocab to wide_spans * lanes * hold tokens) whose tiles encodeTiles lists.
    pub fn foldable(ops: Ops, p: contract.Params) bool {
        contract.check(p) catch return false;
        return ops.fold_pipeline != null and p.k > 1 and p.vocab >= wide_vocab and p.vocab <= wide_spans * lanes * hold and
            p.vocab % 32 == 0 and p.vocab / 32 <= tile_words * 32;
    }
    /// encode then encodeTiles (`picked` = `output`, `rows`, `vocab` and `k` from `p`) in one launch, for foldable `p`:
    /// `output` gets encode's rows, and `list` and `map` encodeTiles' words of them.
    pub fn encodeWideTiles(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, output: Ref, ids: Ref, list: Ref, map: Ref, p: contract.Params) !void {
        if (!ops.foldable(p)) return error.BadTopKShape;
        const pipeline = ops.fold_pipeline.?;
        const counts = ops.fold_counts orelse return error.BadTopKShape;
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.vocab);
        const input_bytes = try std.math.mul(usize, cells, 2);
        const output_bytes = @as(usize, p.rows) * @sizeOf(contract.Row);
        const slots = tileSlots(p.rows, p.k);
        try check(logits, input_bytes, 2);
        try check(output, output_bytes, 4);
        if (logits.buf.id == output.buf.id and logits.off < output.off + output_bytes and output.off < logits.off + input_bytes) return error.TopKAlias;
        try check(ids, @as(usize, p.vocab) * 4, 4);
        try check(list, (@as(usize, slots) + 1) * 4, 4);
        try check(map, @as(usize, slots) * 32 * 4, 4);
        e.setPipeline(pipeline);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(ops.part, 0, 1);
        e.setValue(p, 2);
        e.setBuffer(output.buf, output.off, 3);
        e.setBuffer(counts, 0, 4);
        e.setBuffer(ids.buf, ids.off, 5);
        e.setBuffer(list.buf, list.off, 6);
        e.setBuffer(map.buf, map.off, 7);
        e.setValue([3]u32{ p.rows, p.vocab / 32, slots }, 8);
        e.dispatchGroups(mtl.Size.of(wide_spans, p.rows, 1), mtl.Size.of(lanes, 1, 1));
    }
    /// Slots a tile list of `rows` top-k rows holds: every row's k tiles and tile 0.
    pub fn tileSlots(rows: u32, k: u32) u32 {
        return rows * k + 1;
    }
    /// The distinct 32-column tiles of `rows` top-k rows' tokens (encode's output over a `vocab`-wide row), tile 0 always
    /// among them: `list` gets their count and the tiles in increasing order, `map` the ids (`ids`, one per column) of the
    /// tileSlots(rows, k) slots' columns, 0 past the count.
    pub fn encodeTiles(ops: Ops, e: mtl.ComputeEncoder, picked: Ref, ids: Ref, list: Ref, map: Ref, rows: u32, vocab: u32, k: u32) !void {
        if (rows == 0 or rows > contract.max_rows or k == 0 or k > contract.max_k or vocab == 0 or vocab % 32 != 0 or vocab / 32 > tile_words * 32) return error.BadTopKShape;
        const slots = tileSlots(rows, k);
        try check(picked, @as(usize, rows) * @sizeOf(contract.Row), 4);
        try check(ids, @as(usize, vocab) * 4, 4);
        try check(list, (@as(usize, slots) + 1) * 4, 4);
        try check(map, @as(usize, slots) * 32 * 4, 4);
        e.setPipeline(ops.tiles_pipeline);
        e.setBuffer(picked.buf, picked.off, 0);
        e.setBuffer(ids.buf, ids.off, 1);
        e.setBuffer(list.buf, list.off, 2);
        e.setBuffer(map.buf, map.off, 3);
        e.setValue([3]u32{ rows, vocab / 32, slots }, 4);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(lanes, 1, 1));
    }
    pub fn encode(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, output: Ref, p: contract.Params) !void {
        try contract.check(p);
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.vocab);
        const input_bytes = try std.math.mul(usize, cells, 2);
        const output_bytes = @as(usize, p.rows) * @sizeOf(contract.Row);
        try check(logits, input_bytes, 2);
        try check(output, output_bytes, 4);
        if (logits.buf.id == output.buf.id and logits.off < output.off + output_bytes and output.off < logits.off + input_bytes) return error.TopKAlias;
        if (p.k > 1 and p.vocab >= wide_vocab and p.vocab <= wide_spans * lanes * hold) {
            e.setPipeline(ops.wide_pipeline);
            e.setBuffer(logits.buf, logits.off, 0);
            e.setBuffer(ops.part, 0, 1);
            e.setValue(p, 2);
            e.dispatchGroups(mtl.Size.of(wide_spans, p.rows, 1), mtl.Size.of(lanes, 1, 1));
            e.barrier();
            e.setPipeline(ops.wide_merge_pipeline);
            e.setBuffer(ops.part, 0, 0);
            e.setBuffer(output.buf, output.off, 1);
            e.setValue(p, 2);
            e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(lanes, 1, 1));
            return;
        }
        if (p.k > 1 and p.vocab >= wide_vocab) {
            e.setPipeline(ops.span_pipeline);
            e.setBuffer(logits.buf, logits.off, 0);
            e.setBuffer(ops.part, 0, 1);
            e.setValue(p, 2);
            e.dispatchGroups(mtl.Size.of(spans, p.rows, 1), mtl.Size.of(lanes, 1, 1));
            e.barrier();
            e.setPipeline(ops.merge_pipeline);
            e.setBuffer(ops.part, 0, 0);
            e.setBuffer(output.buf, output.off, 1);
            e.setValue(p, 2);
            e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(lanes, 1, 1));
            return;
        }
        e.setPipeline(if (p.k == 1) ops.max_pipeline orelse ops.pipeline else ops.pipeline);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(output.buf, output.off, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(lanes, 1, 1));
    }
};
fn check(ref: Ref, bytes: usize, alignment: usize) !void {
    if (ref.off % alignment != 0 or ref.off > ref.buf.length() or bytes > ref.buf.length() - ref.off) return error.BadTopKBuffer;
}
