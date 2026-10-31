//! Borrowed BF16 draft primitives encode into caller-owned command buffers with explicit rounding boundaries.
const std = @import("std");
const mtl = @import("metal");
const source = @import("draft_ops_sources").text;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Linear = extern struct { rows: u32, n: u32, k: u32, x_stride: u32, w_stride: u32, y_stride: u32 };
pub const Norm = extern struct { rows: u32, heads: u32, dim: u32, x_stride: u32, y_stride: u32, round_normalized: u32 = 1, eps: f32 = 1e-6 };
pub const Conv = extern struct { rows: u32, width: u32, group: u32 = 16, block: u32 = 8, branch: u32, residual: u32 = 0, x_stride: u32, dynamic_stride: u32, y_stride: u32, rounding: u32 = 0 };
pub const Rope = extern struct { rows: u32, heads: u32, dim: u32, x_stride: u32, y_stride: u32, theta: f32 = 10000000 };
pub const Activation = extern struct { rows: u32, width: u32, gate_stride: u32, up_stride: u32, y_stride: u32, rounding: u32 = 0 };
const Kind = enum { linear, norm, conv, rope, swiglu, norm_sums, conv_sums, swiglu_sums, norm_rope, conv_norm_sums };
pub const Ops = struct {
    pipes: [10]mtl.Pipeline,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [10]mtl.Pipeline = undefined;
        var count: usize = 0;
        errdefer for (pipes[0..count]) |pipe| pipe.deinit();
        for (&pipes, [_][]const u8{ "tf_bf16_linear", "tf_draft_rms", "tf_dynamic_conv2", "tf_full_rope", "tf_draft_swiglu", "tf_draft_rms_sums", "tf_dynamic_conv2_sums", "tf_draft_swiglu_sums", "tf_draft_norm_rope", "tf_conv_norm_sums" }) |*pipe, name| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            count += 1;
        }
        return .{ .pipes = pipes };
    }
    pub fn deinit(ops: Ops) void {
        for (ops.pipes) |pipe| pipe.deinit();
    }
    pub fn linear(ops: Ops, e: mtl.ComputeEncoder, x: Ref, weight: Ref, out: Ref, p: Linear) !void {
        try checkLinear(p);
        const xb = try span(p.rows, p.x_stride, p.k);
        const wb = try span(p.n, p.w_stride, p.k);
        const yb = try span(p.rows, p.y_stride, p.n);
        try refs(out, yb, &.{ x, weight }, &.{ xb, wb });
        e.setPipeline(ops.pipes[@backingInt(Kind.linear)]);
        bind(e, .{ x, weight });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchGroups(mtl.Size.of((p.n + 7) / 8, p.rows, 1), mtl.Size.of(256, 1, 1));
    }
    pub fn norm(ops: Ops, e: mtl.ComputeEncoder, x: Ref, gain: Ref, out: Ref, p: Norm) !void {
        const threads = try normThreads(p);
        const width = try std.math.mul(u32, p.heads, p.dim);
        try refs(out, try span(p.rows, p.y_stride, width), &.{ x, gain }, &.{ try span(p.rows, p.x_stride, width), @as(usize, p.dim) * 2 });
        e.setPipeline(ops.pipes[@backingInt(Kind.norm)]);
        bind(e, .{ x, gain });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchGroups(mtl.Size.of(p.heads, p.rows, 1), mtl.Size.of(threads, 1, 1));
    }
    pub fn conv(ops: Ops, e: mtl.ComputeEncoder, x: Ref, dynamic: Ref, base: Ref, residual: ?Ref, out: Ref, p: Conv) !void {
        try checkConv(p);
        if ((residual != null) != (p.residual == 1)) return error.BadDraftShape;
        const xb = try span(p.rows, p.x_stride, p.width);
        const yb = try span(p.rows, p.y_stride, p.width);
        const db = try span(p.rows, p.dynamic_stride, 4 * (p.width / p.group));
        const bb = @as(usize, p.width) * 4 * 2;
        try refs(out, yb, &.{ x, dynamic, base }, &.{ xb, db, bb });
        if (residual) |r| try refs(out, yb, &.{r}, &.{xb});
        e.setPipeline(ops.pipes[@backingInt(Kind.conv)]);
        bind(e, .{ x, dynamic, base, residual orelse x });
        e.setValue(p, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(256, 1, 1));
    }
    pub fn rope(ops: Ops, e: mtl.ComputeEncoder, x: Ref, positions: Ref, out: Ref, p: Rope) !void {
        try checkRope(p);
        const width = try std.math.mul(u32, p.heads, p.dim);
        try refs(out, try span(p.rows, p.y_stride, width), &.{ x, positions }, &.{ try span(p.rows, p.x_stride, width), @as(usize, p.rows) * 8 });
        if (positions.off % 8 != 0) return error.BadDraftBuffer;
        e.setPipeline(ops.pipes[@backingInt(Kind.rope)]);
        bind(e, .{ x, positions });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchThreads(mtl.Size.of(p.dim / 2, p.heads, p.rows), mtl.Size.of(32, 1, 1));
    }
    pub fn swiglu(ops: Ops, e: mtl.ComputeEncoder, gate: Ref, up: Ref, out: Ref, p: Activation) !void {
        try checkActivation(p);
        try refs(out, try span(p.rows, p.y_stride, p.width), &.{ gate, up }, &.{ try span(p.rows, p.gate_stride, p.width), try span(p.rows, p.up_stride, p.width) });
        e.setPipeline(ops.pipes[@backingInt(Kind.swiglu)]);
        bind(e, .{ gate, up });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// Whether the sums variants take these shapes: whole 64-column groups, one tile of 256 columns at a time, stride 16.
    pub fn sumsRows(rows: u32) bool {
        return rows >= 1 and rows <= 16;
    }
    /// norm then rope of a 128-wide head's rows in one dispatch (the two kernels' values, no barrier between).
    pub fn normRope(ops: Ops, e: mtl.ComputeEncoder, x: Ref, gain: Ref, positions: Ref, out: Ref, pn: Norm, pr: Rope) !void {
        _ = try normThreads(pn);
        try checkRope(pr);
        if (pn.heads != pr.heads or pn.dim != 128 or pr.dim != 128 or pn.rows != pr.rows) return error.BadDraftShape;
        const width = pn.heads * pn.dim;
        try refs(out, try span(pr.rows, pr.y_stride, width), &.{ x, gain, positions }, &.{ try span(pn.rows, pn.x_stride, width), @as(usize, pn.dim) * 2, @as(usize, pn.rows) * 8 });
        if (positions.off % 8 != 0) return error.BadDraftBuffer;
        e.setPipeline(ops.pipes[@backingInt(Kind.norm_rope)]);
        bind(e, .{ x, gain, positions });
        e.setValue(pn, 3);
        e.setValue(pr, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.dispatchGroups(mtl.Size.of(pn.heads, pn.rows, 1), mtl.Size.of(32, 1, 1));
    }
    /// The residual conv, then the norm of its result with that result's group sums: one dispatch, a row a group.
    pub fn convNormSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, dynamic: Ref, base: Ref, residual: Ref, gain: Ref, hidden: Ref, out: Ref, sums: Ref, pc: Conv, pn: Norm) !void {
        try checkConv(pc);
        const threads = try normThreads(pn);
        if (pc.residual != 1 or pn.heads != 1 or pn.dim != pc.width or pn.dim > 5120 or pn.dim % 64 != 0 or pn.rows != pc.rows or !sumsRows(pc.rows) or pn.x_stride != pc.y_stride) return error.BadDraftShape;
        const xb = try span(pc.rows, pc.x_stride, pc.width);
        const yb = try span(pc.rows, pc.y_stride, pc.width);
        const db = try span(pc.rows, pc.dynamic_stride, 4 * (pc.width / pc.group));
        try refs(hidden, yb, &.{ x, dynamic, base, residual }, &.{ xb, db, @as(usize, pc.width) * 4 * 2, xb });
        try refs(out, try span(pn.rows, pn.y_stride, pn.dim), &.{ hidden, gain, x }, &.{ yb, @as(usize, pn.dim) * 2, xb });
        e.setPipeline(ops.pipes[@backingInt(Kind.conv_norm_sums)]);
        bind(e, .{ x, dynamic, base, residual, gain });
        e.setValue(pc, 5);
        e.setValue(pn, 6);
        e.setBuffer(hidden.buf, hidden.off, 7);
        e.setBuffer(out.buf, out.off, 8);
        e.setBuffer(sums.buf, sums.off, 9);
        e.setValue(@as(u32, 16), 10);
        e.dispatchGroups(mtl.Size.of(1, pn.rows, 1), mtl.Size.of(threads, 1, 1));
    }
    pub fn normSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, gain: Ref, out: Ref, sums: Ref, p: Norm) !void {
        const threads = try normThreads(p);
        if (p.heads != 1 or p.dim > 5120 or p.dim % 64 != 0 or !sumsRows(p.rows)) return error.BadDraftShape;
        const width = p.dim;
        try refs(out, try span(p.rows, p.y_stride, width), &.{ x, gain }, &.{ try span(p.rows, p.x_stride, width), @as(usize, p.dim) * 2 });
        e.setPipeline(ops.pipes[@backingInt(Kind.norm_sums)]);
        bind(e, .{ x, gain });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.setBuffer(sums.buf, sums.off, 4);
        e.setValue(@as(u32, 16), 5);
        e.dispatchGroups(mtl.Size.of(p.heads, p.rows, 1), mtl.Size.of(threads, 1, 1));
    }
    pub fn convSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, dynamic: Ref, base: Ref, residual: ?Ref, out: Ref, sums: Ref, p: Conv) !void {
        try checkConv(p);
        if ((residual != null) != (p.residual == 1)) return error.BadDraftShape;
        if (p.width % 256 != 0 or !sumsRows(p.rows)) return error.BadDraftShape;
        const xb = try span(p.rows, p.x_stride, p.width);
        const yb = try span(p.rows, p.y_stride, p.width);
        const db = try span(p.rows, p.dynamic_stride, 4 * (p.width / p.group));
        const bb = @as(usize, p.width) * 4 * 2;
        try refs(out, yb, &.{ x, dynamic, base }, &.{ xb, db, bb });
        if (residual) |r| try refs(out, yb, &.{r}, &.{xb});
        e.setPipeline(ops.pipes[@backingInt(Kind.conv_sums)]);
        bind(e, .{ x, dynamic, base, residual orelse x });
        e.setValue(p, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.setBuffer(sums.buf, sums.off, 6);
        e.setValue(@as(u32, 16), 7);
        e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(256, 1, 1));
    }
    pub fn swigluSums(ops: Ops, e: mtl.ComputeEncoder, gate: Ref, up: Ref, out: Ref, sums: Ref, p: Activation) !void {
        try checkActivation(p);
        if (p.width % 256 != 0 or !sumsRows(p.rows)) return error.BadDraftShape;
        try refs(out, try span(p.rows, p.y_stride, p.width), &.{ gate, up }, &.{ try span(p.rows, p.gate_stride, p.width), try span(p.rows, p.up_stride, p.width) });
        e.setPipeline(ops.pipes[@backingInt(Kind.swiglu_sums)]);
        bind(e, .{ gate, up });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.setBuffer(sums.buf, sums.off, 4);
        e.setValue(@as(u32, 16), 5);
        e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(256, 1, 1));
    }
};
fn bind(e: mtl.ComputeEncoder, values: anytype) void {
    inline for (values, 0..) |v, i| e.setBuffer(v.buf, v.off, i);
}
fn span(rows: u32, stride: u32, width: u32) !usize {
    if (rows == 0 or width == 0 or stride < width) return error.BadDraftShape;
    return std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, rows - 1, stride), width), 2);
}
fn refs(out: Ref, bytes: usize, inputs: []const Ref, lengths: []const usize) !void {
    try check(out, bytes);
    for (inputs, lengths) |r, n| {
        try check(r, n);
        if (r.buf.id == out.buf.id and r.off < out.off + bytes and out.off < r.off + n) return error.DraftAlias;
    }
}
fn check(r: Ref, bytes: usize) !void {
    if (r.off % 2 != 0 or r.off > r.buf.length() or bytes > r.buf.length() - r.off) return error.BadDraftBuffer;
}
pub fn checkLinear(p: Linear) !void {
    _ = try span(p.rows, p.x_stride, p.k);
    _ = try span(p.n, p.w_stride, p.k);
    _ = try span(p.rows, p.y_stride, p.n);
}
pub fn normThreads(p: Norm) !u32 {
    if (p.heads == 0 or p.dim == 0 or p.dim > 65536 or p.round_normalized != 1 or !std.math.isFinite(p.eps) or p.eps <= 0) return error.BadDraftShape;
    const width = try std.math.mul(u32, p.heads, p.dim);
    _ = try span(p.rows, p.x_stride, width);
    _ = try span(p.rows, p.y_stride, width);
    return @min(1024, ((p.dim + 127) / 128) * 32);
}
pub fn checkConv(p: Conv) !void {
    if (p.group == 0 or p.width % p.group != 0 or p.block == 0 or p.branch > 1 or p.residual > 1 or p.rounding != 0) return error.BadDraftShape;
    _ = try span(p.rows, p.x_stride, p.width);
    _ = try span(p.rows, p.y_stride, p.width);
    _ = try span(p.rows, p.dynamic_stride, 4 * (p.width / p.group));
}
pub fn checkRope(p: Rope) !void {
    if (p.heads == 0 or p.dim == 0 or p.dim % 2 != 0 or !std.math.isFinite(p.theta) or p.theta <= 0) return error.BadDraftShape;
    const width = try std.math.mul(u32, p.heads, p.dim);
    _ = try span(p.rows, p.x_stride, width);
    _ = try span(p.rows, p.y_stride, width);
}
pub fn checkActivation(p: Activation) !void {
    if (p.rounding != 0) return error.BadDraftShape;
    _ = try span(p.rows, p.gate_stride, p.width);
    _ = try span(p.rows, p.up_stride, p.width);
    _ = try span(p.rows, p.y_stride, p.width);
}
test "operator guards reject invalid width, stride and rounding before encoding" {
    try std.testing.expectError(error.BadDraftShape, checkLinear(.{ .rows = 0, .n = 64, .k = 128, .x_stride = 128, .w_stride = 128, .y_stride = 64 }));
    try std.testing.expectError(error.BadDraftShape, checkConv(.{ .rows = 8, .width = 64, .group = 3, .branch = 0, .x_stride = 64, .dynamic_stride = 16, .y_stride = 64 }));
    try std.testing.expectError(error.BadDraftShape, checkRope(.{ .rows = 8, .heads = 4, .dim = 127, .x_stride = 508, .y_stride = 508 }));
    try std.testing.expectError(error.BadDraftShape, normThreads(.{ .rows = 8, .heads = 4, .dim = 128, .x_stride = 512, .y_stride = 512, .round_normalized = 2 }));
}
