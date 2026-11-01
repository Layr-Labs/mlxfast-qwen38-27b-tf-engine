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
const Kind = enum { linear, norm, conv, rope, swiglu, norm_xs, conv_xs, swiglu_xs, conv_norm_xs, norm_rope };
/// Group sums a sums-emitting primitive writes for the q4 projection that reads its output: f32
/// [width / 64][stride], each group's 64 output values added in order, rows past `rows` zero.
pub const Sums = struct { ref: Ref, stride: u32 };
/// Columns a sums group spans, and the widest row a sums-emitting norm holds.
pub const sums_group = 64;
pub const sums_max_width = 5120;
/// The widest head a fused norm and rope holds (tf_draft_rms_rope).
pub const rope_max_dim = 256;
pub const Ops = struct {
    pipes: [10]mtl.Pipeline,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [10]mtl.Pipeline = undefined;
        var count: usize = 0;
        errdefer for (pipes[0..count]) |pipe| pipe.deinit();
        for (&pipes, [_][]const u8{ "tf_bf16_linear", "tf_draft_rms", "tf_dynamic_conv2", "tf_full_rope", "tf_draft_swiglu", "tf_draft_rms_xs", "tf_dynamic_conv2_xs", "tf_draft_swiglu_xs", "tf_dynamic_conv2_rms_xs", "tf_draft_rms_rope" }) |*pipe, name| {
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
    /// A per-head norm of `x` into `normed` (norm's values), then the rope of `normed` into `out` (rope's), one launch.
    pub fn normRope(ops: Ops, e: mtl.ComputeEncoder, x: Ref, gain: Ref, normed: Ref, positions: Ref, out: Ref, np: Norm, rp: Rope) !void {
        const threads = try normThreads(np);
        try checkRope(rp);
        if (np.heads != rp.heads or np.dim != rp.dim or np.rows != rp.rows or np.dim > rope_max_dim or rp.x_stride != np.y_stride) return error.BadDraftShape;
        const width = try std.math.mul(u32, np.heads, np.dim);
        const xb = try span(np.rows, np.x_stride, width);
        const nb = try span(np.rows, np.y_stride, width);
        try refs(normed, nb, &.{ x, gain }, &.{ xb, @as(usize, np.dim) * 2 });
        try refs(out, try span(rp.rows, rp.y_stride, width), &.{ x, normed, positions }, &.{ xb, nb, @as(usize, rp.rows) * 8 });
        if (positions.off % 8 != 0) return error.BadDraftBuffer;
        e.setPipeline(ops.pipes[@backingInt(Kind.norm_rope)]);
        bind(e, .{ x, gain });
        e.setValue(np, 2);
        e.setBuffer(normed.buf, normed.off, 3);
        e.setBuffer(positions.buf, positions.off, 4);
        e.setValue(rp, 5);
        e.setBuffer(out.buf, out.off, 6);
        e.dispatchGroups(mtl.Size.of(np.heads, np.rows, 1), mtl.Size.of(threads, 1, 1));
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
    /// norm with its output's group sums (one head of at most sums_max_width columns).
    pub fn normSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, gain: Ref, out: Ref, sums: Sums, p: Norm) !void {
        const threads = try normThreads(p);
        if (p.heads != 1 or p.dim > sums_max_width) return error.BadDraftShape;
        try checkSums(sums, p.rows, p.dim);
        try refs(out, try span(p.rows, p.y_stride, p.dim), &.{ x, gain }, &.{ try span(p.rows, p.x_stride, p.dim), @as(usize, p.dim) * 2 });
        e.setPipeline(ops.pipes[@backingInt(Kind.norm_xs)]);
        bind(e, .{ x, gain });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.setBuffer(sums.ref.buf, sums.ref.off, 4);
        e.setValue(sums.stride, 5);
        e.dispatchGroups(mtl.Size.of(1, p.rows, 1), mtl.Size.of(threads, 1, 1));
    }
    /// conv with its output's group sums.
    /// A conv with `residual` into `conv_out` (conv's values), then normSums of it into `out` and `sums`, one launch.
    pub fn convNormSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, dynamic: Ref, base: Ref, residual: Ref, conv_out: Ref, cp: Conv, gain: Ref, out: Ref, sums: Sums, np: Norm) !void {
        try checkConv(cp);
        const threads = try normThreads(np);
        if (cp.residual != 1 or np.heads != 1 or np.dim != cp.width or np.rows != cp.rows or np.x_stride != cp.y_stride or np.dim > sums_max_width) return error.BadDraftShape;
        try checkSums(sums, np.rows, np.dim);
        const xb = try span(cp.rows, cp.x_stride, cp.width);
        const cb = try span(cp.rows, cp.y_stride, cp.width);
        const db = try span(cp.rows, cp.dynamic_stride, 4 * (cp.width / cp.group));
        const bb = @as(usize, cp.width) * 4 * 2;
        try refs(conv_out, cb, &.{ x, dynamic, base, residual }, &.{ xb, db, bb, xb });
        try refs(out, try span(np.rows, np.y_stride, np.dim), &.{ conv_out, gain, x }, &.{ cb, @as(usize, np.dim) * 2, xb });
        e.setPipeline(ops.pipes[@backingInt(Kind.conv_norm_xs)]);
        bind(e, .{ x, dynamic, base, residual });
        e.setValue(cp, 4);
        e.setBuffer(conv_out.buf, conv_out.off, 5);
        e.setBuffer(gain.buf, gain.off, 6);
        e.setValue(np, 7);
        e.setBuffer(out.buf, out.off, 8);
        e.setBuffer(sums.ref.buf, sums.ref.off, 9);
        e.setValue(sums.stride, 10);
        e.dispatchGroups(mtl.Size.of(1, np.rows, 1), mtl.Size.of(threads, 1, 1));
    }
    pub fn convSums(ops: Ops, e: mtl.ComputeEncoder, x: Ref, dynamic: Ref, base: Ref, residual: ?Ref, out: Ref, sums: Sums, p: Conv) !void {
        try checkConv(p);
        if ((residual != null) != (p.residual == 1)) return error.BadDraftShape;
        try checkSums(sums, p.rows, p.width);
        const xb = try span(p.rows, p.x_stride, p.width);
        const yb = try span(p.rows, p.y_stride, p.width);
        const db = try span(p.rows, p.dynamic_stride, 4 * (p.width / p.group));
        const bb = @as(usize, p.width) * 4 * 2;
        try refs(out, yb, &.{ x, dynamic, base }, &.{ xb, db, bb });
        if (residual) |r| try refs(out, yb, &.{r}, &.{xb});
        e.setPipeline(ops.pipes[@backingInt(Kind.conv_xs)]);
        bind(e, .{ x, dynamic, base, residual orelse x });
        e.setValue(p, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.setBuffer(sums.ref.buf, sums.ref.off, 6);
        e.setValue(sums.stride, 7);
        e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(sums_group, 1, 1));
    }
    /// swiglu with its output's group sums.
    pub fn swigluSums(ops: Ops, e: mtl.ComputeEncoder, gate: Ref, up: Ref, out: Ref, sums: Sums, p: Activation) !void {
        try checkActivation(p);
        try checkSums(sums, p.rows, p.width);
        try refs(out, try span(p.rows, p.y_stride, p.width), &.{ gate, up }, &.{ try span(p.rows, p.gate_stride, p.width), try span(p.rows, p.up_stride, p.width) });
        e.setPipeline(ops.pipes[@backingInt(Kind.swiglu_xs)]);
        bind(e, .{ gate, up });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.setBuffer(sums.ref.buf, sums.ref.off, 4);
        e.setValue(sums.stride, 5);
        e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(sums_group, 1, 1));
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
/// A sums-emitting launch: whole 64-column groups, at most `stride` rows, its f32 sums in range.
pub fn checkSums(sums: Sums, rows: u32, width: u32) !void {
    if (rows == 0 or width == 0 or width % sums_group != 0 or sums.stride < rows or sums.stride > 64) return error.BadDraftShape;
    const bytes = @as(usize, width / sums_group) * sums.stride * 4;
    if (sums.ref.off % 4 != 0 or sums.ref.off > sums.ref.buf.length() or bytes > sums.ref.buf.length() - sums.ref.off) return error.BadDraftBuffer;
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
