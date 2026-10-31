//! Encode-only BF16 target picks and deterministic verified-tree matching.
const std = @import("std");
const mtl = @import("metal");
pub const contract = @import("tree_round");
const source = @import("tree_round_sources").text;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
/// Spans a row's picks take in `argmaxSpans`: enough threadgroups to spread a 248320-wide row over the GPU.
pub const argmax_spans = 64;
/// One span's best score, its token (invalid_token when the span held none) and its nonfinite mask.
pub const ArgmaxPart = extern struct { value: f32, token: u32, nonfinite: u32 };
pub const Ops = struct {
    argmax_pipe: mtl.Pipeline,
    match_pipe: mtl.Pipeline,
    span_pipe: mtl.Pipeline,
    merge_pipe: mtl.Pipeline,
    merge_match_pipe: mtl.Pipeline,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const argmax_pipe = try mtl.Pipeline.init(device, lib, "tf_round_bf16_argmax", false);
        errdefer argmax_pipe.deinit();
        const match_pipe = try mtl.Pipeline.init(device, lib, "tf_tree_round_match", false);
        errdefer match_pipe.deinit();
        const span_pipe = try mtl.Pipeline.init(device, lib, "tf_round_bf16_argmax_span", false);
        errdefer span_pipe.deinit();
        const merge_pipe = try mtl.Pipeline.init(device, lib, "tf_round_bf16_argmax_merge", false);
        errdefer merge_pipe.deinit();
        return .{ .argmax_pipe = argmax_pipe, .match_pipe = match_pipe, .span_pipe = span_pipe, .merge_pipe = merge_pipe, .merge_match_pipe = try mtl.Pipeline.init(device, lib, "tf_round_bf16_argmax_merge_match", false) };
    }
    pub fn deinit(ops: Ops) void {
        ops.argmax_pipe.deinit();
        ops.match_pipe.deinit();
        ops.span_pipe.deinit();
        ops.merge_pipe.deinit();
        ops.merge_match_pipe.deinit();
    }
    pub fn argmax(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, picks: Ref, p: contract.Argmax) !void {
        try contract.checkArgmax(p);
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.vocab);
        const bytes = try std.math.mul(usize, cells, 2);
        try check(logits, bytes, 2);
        try check(picks, @as(usize, p.rows) * @sizeOf(contract.Pick), 4);
        try distinct(picks, @as(usize, p.rows) * @sizeOf(contract.Pick), logits, bytes);
        e.setPipeline(ops.argmax_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(picks.buf, picks.off, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }
    /// `argmax` in two passes through `parts` (rows * argmax_spans ArgmaxPart): every row's spans run in their own
    /// threadgroups, then one threadgroup a row merges them. The picks equal `argmax`'s; `parts` is the caller's scratch.
    pub fn argmaxSpans(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, parts: Ref, picks: Ref, p: contract.Argmax) !void {
        try contract.checkArgmax(p);
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.vocab);
        const bytes = try std.math.mul(usize, cells, 2);
        const part_bytes = @as(usize, p.rows) * argmax_spans * @sizeOf(ArgmaxPart);
        try check(logits, bytes, 2);
        try check(parts, part_bytes, 4);
        try check(picks, @as(usize, p.rows) * @sizeOf(contract.Pick), 4);
        try distinct(parts, part_bytes, logits, bytes);
        try distinct(picks, @as(usize, p.rows) * @sizeOf(contract.Pick), parts, part_bytes);
        try distinct(picks, @as(usize, p.rows) * @sizeOf(contract.Pick), logits, bytes);
        e.setPipeline(ops.span_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(parts.buf, parts.off, 1);
        e.setValue(p, 2);
        e.dispatchGroups(mtl.Size.of(argmax_spans, p.rows, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        e.setPipeline(ops.merge_pipe);
        e.setBuffer(parts.buf, parts.off, 0);
        e.setBuffer(picks.buf, picks.off, 1);
        e.setValue(@as(u32, argmax_spans), 2);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(64, 1, 1));
    }
    /// `argmaxSpans`' merge and `match` in one dispatch: the kernel reduces every row's spans into
    /// threadgroup picks exactly as the merge does, then runs the match body over them. The Result
    /// equals merge-then-match's; `picks` is unused (kept for the caller's buffer layout).
    pub fn argmaxSpansMatch(ops: Ops, e: mtl.ComputeEncoder, logits: Ref, parts: Ref, tokens: Ref, parents: Ref, eos: ?Ref, result: Ref, p: contract.Match) !void {
        try contract.checkMatch(p);
        if ((eos == null) != (p.eos_count == 0)) return error.BadRoundShape;
        const rows = @as(usize, p.rows);
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.vocab), p.vocab);
        const bytes = try std.math.mul(usize, cells, 2);
        const part_bytes = rows * argmax_spans * @sizeOf(ArgmaxPart);
        try check(logits, bytes, 2);
        try check(parts, part_bytes, 4);
        try check(result, @sizeOf(contract.Result), 4);
        for ([_]Ref{ tokens, parents }, [_]usize{ rows * 4, rows * 4 }) |ref, need| {
            try check(ref, need, 4);
            try distinct(result, @sizeOf(contract.Result), ref, need);
            try distinct(parts, part_bytes, ref, need);
        }
        if (eos) |ref| {
            try check(ref, @as(usize, p.eos_count) * 4, 4);
            try distinct(result, @sizeOf(contract.Result), ref, @as(usize, p.eos_count) * 4);
            try distinct(parts, part_bytes, ref, @as(usize, p.eos_count) * 4);
        }
        try distinct(parts, part_bytes, logits, bytes);
        try distinct(result, @sizeOf(contract.Result), logits, bytes);
        try distinct(result, @sizeOf(contract.Result), parts, part_bytes);
        e.setPipeline(ops.span_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setBuffer(parts.buf, parts.off, 1);
        e.setValue(contract.Argmax{ .rows = p.rows, .vocab = p.vocab, .stride = p.vocab }, 2);
        e.dispatchGroups(mtl.Size.of(argmax_spans, p.rows, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        e.setPipeline(ops.merge_match_pipe);
        e.setBuffer(parts.buf, parts.off, 0);
        e.setBuffer(tokens.buf, tokens.off, 1);
        e.setBuffer(parents.buf, parents.off, 2);
        e.setBuffer(if (eos) |ref| ref.buf else tokens.buf, if (eos) |ref| ref.off else tokens.off, 3);
        e.setBuffer(result.buf, result.off, 4);
        e.setValue(p, 5);
        e.setValue(@as(u32, argmax_spans), 6);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(64, 1, 1));
    }
    pub fn match(ops: Ops, e: mtl.ComputeEncoder, tokens: Ref, parents: Ref, picks: Ref, eos: ?Ref, result: Ref, p: contract.Match) !void {
        try contract.checkMatch(p);
        if ((eos == null) != (p.eos_count == 0)) return error.BadRoundShape;
        const rows = @as(usize, p.rows);
        try check(result, @sizeOf(contract.Result), 4);
        for ([_]Ref{ tokens, parents, picks }, [_]usize{ rows * 4, rows * 4, rows * @sizeOf(contract.Pick) }) |ref, bytes| {
            try check(ref, bytes, 4);
            try distinct(result, @sizeOf(contract.Result), ref, bytes);
        }
        if (eos) |ref| {
            try check(ref, @as(usize, p.eos_count) * 4, 4);
            try distinct(result, @sizeOf(contract.Result), ref, @as(usize, p.eos_count) * 4);
        }
        e.setPipeline(ops.match_pipe);
        for ([_]Ref{ tokens, parents, picks, eos orelse tokens, result }, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setValue(p, 5);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }
};
fn check(ref: Ref, bytes: usize, alignment: usize) !void {
    if (ref.off % alignment != 0 or ref.off > ref.buf.length() or bytes > ref.buf.length() - ref.off) return error.BadRoundBuffer;
}
fn distinct(out: Ref, bytes: usize, input: Ref, input_bytes: usize) !void {
    if (out.buf.id == input.buf.id and out.off < input.off + input_bytes and input.off < out.off + bytes) return error.RoundAlias;
}
