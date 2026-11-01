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
/// Where a `*Sums` encode writes its output's 64-column group sums: XS[group * stride + row], padded rows zero.
pub const Sums = struct { ref: Ref, stride: u32 };
pub const Ops = struct {
    pipes: [10]mtl.Pipeline,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [10]mtl.Pipeline = undefined;
        var count: usize = 0;
        errdefer for (pipes[0..count]) |pipe| pipe.deinit();
        for (&pipes, [_][]const u8{ "tf_bf16_linear", "tf_draft_rms", "tf_dynamic_conv2", "tf_full_rope", "tf_draft_swiglu", "tf_draft_rms_sums", "tf_dynamic_conv2_sums", "tf_draft_swiglu_sums", "tf_draft_rms_rope", "tf_dynamic_conv2_rms_sums" }) |*pipe, name| {
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
    /// `norm` of whole rows (one head, `dim` a multiple of 64) that also writes `sums` for the projection reading `out`.
    pub fn normSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, gain: Ref, out: Ref, p: Norm, sums: Sums) !void {
        const threads = try normThreads(p);
        if (p.heads != 1 or p.dim % 64 != 0 or sums.stride < p.rows) return error.BadDraftShape;
        try refs(out, try span(p.rows, p.y_stride, p.dim), &.{ x, gain }, &.{ try span(p.rows, p.x_stride, p.dim), @as(usize, p.dim) * 2 });
        try checkSums(sums, p.dim);
        e.setPipeline(ops.pipes[@backingInt(Kind.norm_sums)]);
        bind(e, .{ x, gain });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.setBuffer(sums.ref.buf, sums.ref.off, 4);
        e.setValue(sums.stride, 5);
        e.dispatchGroups(mtl.Size.of(1, p.rows, 1), mtl.Size.of(threads, 1, 1));
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
    /// `conv` (`width` a multiple of 64) that also writes `sums` for the projection reading `out`.
    pub fn convSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, dynamic: Ref, base: Ref, residual: ?Ref, out: Ref, p: Conv, sums: Sums) !void {
        try checkConv(p);
        if ((residual != null) != (p.residual == 1) or p.width % 64 != 0 or sums.stride < p.rows) return error.BadDraftShape;
        const xb = try span(p.rows, p.x_stride, p.width);
        const yb = try span(p.rows, p.y_stride, p.width);
        const db = try span(p.rows, p.dynamic_stride, 4 * (p.width / p.group));
        const bb = @as(usize, p.width) * 4 * 2;
        try refs(out, yb, &.{ x, dynamic, base }, &.{ xb, db, bb });
        if (residual) |r| try refs(out, yb, &.{r}, &.{xb});
        try checkSums(sums, p.width);
        e.setPipeline(ops.pipes[@backingInt(Kind.conv_sums)]);
        bind(e, .{ x, dynamic, base, residual orelse x });
        e.setValue(p, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.setBuffer(sums.ref.buf, sums.ref.off, 6);
        e.setValue(sums.stride, 7);
        e.dispatchGroups(mtl.Size.of(p.width / 64, p.rows, 1), mtl.Size.of(64, 1, 1));
    }
    /// `conv` of whole rows into `hidden`, then `normSums` of `hidden` into `out` and `sums` (a threadgroup a row).
    pub fn convNormSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, dynamic: Ref, base: Ref, residual: ?Ref, hidden: Ref, gain: Ref, out: Ref, c: Conv, p: Norm, sums: Sums) !void {
        try checkConv(c);
        const threads = try normThreads(p);
        if ((residual != null) != (c.residual == 1) or p.heads != 1 or p.dim != c.width or p.rows != c.rows or p.x_stride != c.y_stride or p.dim % 64 != 0 or sums.stride < p.rows) return error.BadDraftShape;
        const xb = try span(c.rows, c.x_stride, c.width);
        const hb = try span(c.rows, c.y_stride, c.width);
        const db = try span(c.rows, c.dynamic_stride, 4 * (c.width / c.group));
        try refs(hidden, hb, &.{ x, dynamic, base }, &.{ xb, db, @as(usize, c.width) * 4 * 2 });
        if (residual) |r| try refs(hidden, hb, &.{r}, &.{xb});
        try refs(out, try span(p.rows, p.y_stride, p.dim), &.{ hidden, gain }, &.{ hb, @as(usize, p.dim) * 2 });
        try checkSums(sums, p.dim);
        e.setPipeline(ops.pipes[@backingInt(Kind.conv_norm_sums)]);
        bind(e, .{ x, dynamic, base, residual orelse x });
        e.setValue(c, 4);
        e.setBuffer(hidden.buf, hidden.off, 5);
        e.setBuffer(gain.buf, gain.off, 6);
        e.setValue(p, 7);
        e.setBuffer(out.buf, out.off, 8);
        e.setBuffer(sums.ref.buf, sums.ref.off, 9);
        e.setValue(sums.stride, 10);
        e.dispatchGroups(mtl.Size.of(1, p.rows, 1), mtl.Size.of(threads, 1, 1));
    }
    /// `norm` of 128-wide heads then `rope` of its words into `out` (32 threads a head, as `norm` takes them).
    pub fn normRope(ops: Ops, e: mtl.ComputeEncoder, x: Ref, gain: Ref, positions: Ref, out: Ref, p: Norm, r: Rope) !void {
        if (try normThreads(p) != 32 or p.dim != 128 or r.dim != p.dim or r.heads != p.heads or r.rows != p.rows) return error.BadDraftShape;
        try checkRope(r);
        const width = try std.math.mul(u32, p.heads, p.dim);
        try refs(out, try span(p.rows, r.y_stride, width), &.{ x, gain, positions }, &.{ try span(p.rows, p.x_stride, width), @as(usize, p.dim) * 2, @as(usize, p.rows) * 8 });
        if (positions.off % 8 != 0) return error.BadDraftBuffer;
        e.setPipeline(ops.pipes[@backingInt(Kind.norm_rope)]);
        bind(e, .{ x, gain });
        e.setValue(p, 2);
        e.setBuffer(positions.buf, positions.off, 3);
        e.setValue(r, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.dispatchGroups(mtl.Size.of(p.heads, p.rows, 1), mtl.Size.of(32, 1, 1));
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
    /// `swiglu` (`width` a multiple of 64) that also writes `sums` for the projection reading `out`.
    pub fn swigluSums(ops: Ops, e: mtl.ComputeEncoder, gate: Ref, up: Ref, out: Ref, p: Activation, sums: Sums) !void {
        try checkActivation(p);
        if (p.width % 64 != 0 or sums.stride < p.rows) return error.BadDraftShape;
        try refs(out, try span(p.rows, p.y_stride, p.width), &.{ gate, up }, &.{ try span(p.rows, p.gate_stride, p.width), try span(p.rows, p.up_stride, p.width) });
        try checkSums(sums, p.width);
        e.setPipeline(ops.pipes[@backingInt(Kind.swiglu_sums)]);
        bind(e, .{ gate, up });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.setBuffer(sums.ref.buf, sums.ref.off, 4);
        e.setValue(sums.stride, 5);
        e.dispatchGroups(mtl.Size.of(p.width / 64, p.rows, 1), mtl.Size.of(64, 1, 1));
    }
};
fn checkSums(sums: Sums, width: u32) !void {
    const bytes = try std.math.mul(usize, @as(usize, width / 64) * sums.stride, 4);
    if (sums.ref.off % 4 != 0 or sums.ref.off > sums.ref.buf.length() or bytes > sums.ref.buf.length() - sums.ref.off) return error.BadDraftBuffer;
}
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
