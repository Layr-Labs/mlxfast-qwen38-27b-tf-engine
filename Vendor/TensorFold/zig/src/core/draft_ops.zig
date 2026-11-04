//! Borrowed BF16 draft primitives encode into caller-owned command buffers with explicit rounding boundaries.
const std = @import("std");
const mtl = @import("metal");
const tune = @import("tune_once");
const source = @import("draft_ops_sources").text;
/// The tune-once pilot's twin text: the launched source with every `size_t` narrowed to `uint`.
/// The guard keeps the 64-bit incumbent unless the text carries exactly 46 such lines with 59
/// sites (this tree's draft_ops.metal); only the linear pipeline is ever built from the twin.
pub const linear32_expected_lines = 46;
pub const linear32_expected_sites = 59;
pub const linear32_lines = tune.countLinesWith(source, tune.narrow_from);
pub const linear32_sites = tune.countOccurrences(source, tune.narrow_from);
pub const linear32_guarded = linear32_lines == linear32_expected_lines and linear32_sites == linear32_expected_sites;
pub const linear32_text = tune.narrow(source);
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Linear = extern struct { rows: u32, n: u32, k: u32, x_stride: u32, w_stride: u32, y_stride: u32 };
pub const Norm = extern struct { rows: u32, heads: u32, dim: u32, x_stride: u32, y_stride: u32, round_normalized: u32 = 1, eps: f32 = 1e-6 };
pub const Conv = extern struct { rows: u32, width: u32, group: u32 = 16, block: u32 = 8, branch: u32, residual: u32 = 0, x_stride: u32, dynamic_stride: u32, y_stride: u32, rounding: u32 = 0 };
pub const Rope = extern struct { rows: u32, heads: u32, dim: u32, x_stride: u32, y_stride: u32, theta: f32 = 10000000 };
pub const Activation = extern struct { rows: u32, width: u32, gate_stride: u32, up_stride: u32, y_stride: u32, rounding: u32 = 0 };
/// A row copy's shape, in 16-bit words.
pub const Rows = extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32 };
const Kind = enum { linear, norm, conv, rope, swiglu, norm_xs, conv_xs, swiglu_xs, conv_norm_xs, norm_rope, zero, conv_xs_stack, norm_rope_ring, copy_rows };
/// Group sums a sums-emitting primitive writes for the q4 projection that reads its output: f32
/// [width / 64][stride], each group's 64 output values added in order, rows past `rows` zero.
pub const Sums = struct { ref: Ref, stride: u32 };
/// Columns a sums group spans, and the widest row a sums-emitting norm holds.
pub const sums_group = 64;
pub const sums_max_width = 5120;
/// The widest head a fused norm and rope holds (tf_draft_rms_rope).
pub const rope_max_dim = 256;
/// A counted projection's word (lane_projection's TF_COUNT) and its threadgroups: a launch given one reads that
/// projection's rows in the projection's own barrier interval, its threadgroups waiting until the count is reached.
pub const Wait = struct { ref: Ref, value: u32 };
pub const Ops = struct {
    pipes: [14]mtl.Pipeline,
    /// Taken by the next convSums, convNormSums, swigluSums, normRope, normRopeRing or copyRows (its caller sets and
    /// clears it).
    wait: ?Wait = null,
    /// The tune-once pilot's 32-bit-index twin of the linear pipeline, built only by the tune.
    linear32: ?mtl.Pipeline = null,
    /// Fixed winner per boot: linear() serves the twin only after the load-time tune adopted it.
    use_linear32: bool = false,
    /// The load-time tune ran (or was skipped): the winner is fixed, never re-tuned.
    tuned: bool = false,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [14]mtl.Pipeline = undefined;
        var count: usize = 0;
        errdefer for (pipes[0..count]) |pipe| pipe.deinit();
        for (&pipes, [_][]const u8{ "tf_bf16_linear", "tf_draft_rms", "tf_dynamic_conv2", "tf_full_rope", "tf_draft_swiglu", "tf_draft_rms_xs", "tf_dynamic_conv2_xs", "tf_draft_swiglu_xs", "tf_dynamic_conv2_rms_xs", "tf_draft_rms_rope", "tf_draft_zero", "tf_dynamic_conv2_xs_stack", "tf_draft_rms_rope_ring", "tf_draft_copy_rows" }) |*pipe, name| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            count += 1;
        }
        return .{ .pipes = pipes };
    }
    pub fn deinit(ops: Ops) void {
        for (ops.pipes) |pipe| pipe.deinit();
        if (ops.linear32) |twin| twin.deinit();
    }
    /// The wait words of the six launches that take one (buffers 14 and 15); none: a zero count, never read.
    fn bindWait(ops: Ops, e: mtl.ComputeEncoder, unused: Ref) void {
        const w = ops.wait orelse Wait{ .ref = unused, .value = 0 };
        e.setBuffer(w.ref.buf, w.ref.off, 14);
        e.setValue(w.value, 15);
    }
    /// `count` u32 words at `words` set to zero.
    pub fn zero(ops: Ops, e: mtl.ComputeEncoder, words: Ref, count: u32) !void {
        if (count == 0 or words.off % 4 != 0 or words.off > words.buf.length() or @as(usize, count) * 4 > words.buf.length() - words.off) return error.BadDraftBuffer;
        e.setPipeline(ops.pipes[@backingInt(Kind.zero)]);
        e.setBuffer(words.buf, words.off, 0);
        e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(@min(count, 256), 1, 1));
    }
    pub fn linear(ops: Ops, e: mtl.ComputeEncoder, x: Ref, weight: Ref, out: Ref, p: Linear) !void {
        try checkLinear(p);
        const xb = try span(p.rows, p.x_stride, p.k);
        const wb = try span(p.n, p.w_stride, p.k);
        const yb = try span(p.rows, p.y_stride, p.n);
        try refs(out, yb, &.{ x, weight }, &.{ xb, wb });
        e.setPipeline(ops.linearPipe(p));
        bind(e, .{ x, weight });
        e.setValue(p, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchGroups(mtl.Size.of((p.n + 7) / 8, p.rows, 1), mtl.Size.of(256, 1, 1));
    }
    /// The adopted linear pipeline: the 32-bit twin only when the tune adopted it and every
    /// index of this launch stays below 2^31 (uint arithmetic then equals size_t's bit for bit),
    /// else the 64-bit incumbent.
    fn linearPipe(ops: Ops, p: Linear) mtl.Pipeline {
        if (ops.use_linear32) {
            if (ops.linear32) |twin| {
                if (offsetsFitU31(p)) return twin;
            }
        }
        return ops.pipes[@backingInt(Kind.linear)];
    }
    /// Load-time tune-once pilot for the drafter matmul: bit-check the 32-bit-index twin against
    /// the incumbent on synthetic seeded shapes, then device-time both round robin and adopt the
    /// twin only if bit-identical and faster, else the incumbent stays. Fixed winner per boot
    /// (runs once). Holds no prompt or decode content: synthetic buffers only, freed before
    /// return. Never fails the load: any miss keeps the incumbent.
    pub fn tuneLinear32(ops: *Ops, device: mtl.Device) void {
        if (ops.tuned) return;
        ops.tuned = true;
        if (!tune.tuneEnabled()) {
            std.log.info("tune-once linear32: off (MLXFAST_TUNE=0), incumbent stays", .{});
            return;
        }
        if (!linear32_guarded) {
            std.log.warn("tune-once linear32: text guard mismatch ({d} lines, {d} sites), incumbent stays", .{ linear32_lines, linear32_sites });
            return;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = mtl.Library.fromSource(device, &linear32_text, mtl.CompileOptions.mlx()) catch {
            std.log.warn("tune-once linear32: twin library failed to compile, incumbent stays", .{});
            return;
        };
        defer lib.deinit();
        const twin = mtl.Pipeline.initOptional(device, lib, "tf_bf16_linear", false) catch {
            std.log.warn("tune-once linear32: twin pipeline failed to build, incumbent stays", .{});
            return;
        };
        const queue = device.queue() catch {
            std.log.warn("tune-once linear32: no tune queue, incumbent stays", .{});
            twin.deinit();
            return;
        };
        defer queue.deinit();
        const incumbent = ops.pipes[@backingInt(Kind.linear)];
        for (tune_shapes, 0..) |shape, i| {
            const seed: u64 = 0x9E3779B97F4A7C15 +% (@as(u64, @intCast(i)) *% 0xBF58476D1CE4E5B9);
            bitCheckLinear(device, queue, incumbent, twin, shape, seed) catch {
                std.log.warn("tune-once linear32: bit check failed (shape {d}), incumbent stays", .{i});
                twin.deinit();
                return;
            };
        }
        if (tune.tuneForced()) {
            ops.linear32 = twin;
            ops.use_linear32 = true;
            std.log.info("tune-once linear32: forced (MLXFAST_TUNE_FORCE=1), twin adopted after bit check", .{});
            return;
        }
        const outcome = timeLinear32(device, queue, incumbent, twin) catch {
            std.log.warn("tune-once linear32: timing failed, incumbent stays", .{});
            twin.deinit();
            return;
        };
        if (outcome.adopted) {
            ops.linear32 = twin;
            ops.use_linear32 = true;
        } else twin.deinit();
        std.log.info("tune-once linear32: {s} (inc {d:.3} ms, twin {d:.3} ms, wins {d}/{d})", .{ if (outcome.adopted) "twin adopted" else "incumbent stays", outcome.inc_med * 1000, outcome.cand_med * 1000, outcome.wins, outcome.cycles });
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
        ops.bindWait(e, out);
        e.dispatchGroups(mtl.Size.of(np.heads, np.rows, 1), mtl.Size.of(threads, 1, 1));
    }
    /// normRope whose rows from `from` on rotate into `ring` (rows heads * dim apart) instead of `out`, only as many as the
    /// u32 at `count` (an earlier dispatch wrote it, at most np.rows - from); `out` holds rows below `from`.
    pub fn normRopeRing(ops: Ops, e: mtl.ComputeEncoder, x: Ref, gain: Ref, normed: Ref, positions: Ref, out: Ref, ring: Ref, count: Ref, np: Norm, rp: Rope, from: u32) !void {
        const threads = try normThreads(np);
        try checkRope(rp);
        if (np.heads != rp.heads or np.dim != rp.dim or np.rows != rp.rows or np.dim > rope_max_dim or rp.x_stride != np.y_stride or from == 0 or from >= np.rows) return error.BadDraftShape;
        const width = try std.math.mul(u32, np.heads, np.dim);
        const xb = try span(np.rows, np.x_stride, width);
        const nb = try span(np.rows, np.y_stride, width);
        const pb = @as(usize, rp.rows) * 8;
        const ob = try span(from, rp.y_stride, width);
        try refs(normed, nb, &.{ x, gain }, &.{ xb, @as(usize, np.dim) * 2 });
        try refs(out, ob, &.{ x, normed, positions }, &.{ xb, nb, pb });
        try refs(ring, try span(np.rows - from, width, width), &.{ x, normed, positions, out }, &.{ xb, nb, pb, ob });
        if (positions.off % 8 != 0 or count.off % 4 != 0 or count.off > count.buf.length() or count.buf.length() - count.off < 4) return error.BadDraftBuffer;
        e.setPipeline(ops.pipes[@backingInt(Kind.norm_rope_ring)]);
        bind(e, .{ x, gain });
        e.setValue(np, 2);
        e.setBuffer(normed.buf, normed.off, 3);
        e.setBuffer(positions.buf, positions.off, 4);
        e.setValue(rp, 5);
        e.setBuffer(out.buf, out.off, 6);
        e.setBuffer(ring.buf, ring.off, 7);
        e.setBuffer(count.buf, count.off, 8);
        e.setValue(from, 9);
        ops.bindWait(e, out);
        e.dispatchGroups(mtl.Size.of(np.heads, np.rows, 1), mtl.Size.of(threads, 1, 1));
    }
    /// Rows below p.rows of `x` into `y`, their 16-bit words as they are, only as many as the u32 at `count` (an earlier
    /// dispatch wrote it).
    pub fn copyRows(ops: Ops, e: mtl.ComputeEncoder, x: Ref, y: Ref, count: Ref, p: Rows) !void {
        try refs(y, try span(p.rows, p.y_stride, p.width), &.{x}, &.{try span(p.rows, p.x_stride, p.width)});
        if (count.off % 4 != 0 or count.off > count.buf.length() or count.buf.length() - count.off < 4) return error.BadDraftBuffer;
        e.setPipeline(ops.pipes[@backingInt(Kind.copy_rows)]);
        bind(e, .{ x, y });
        e.setValue(p, 2);
        e.setBuffer(count.buf, count.off, 3);
        ops.bindWait(e, y);
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
        ops.bindWait(e, out);
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
        ops.bindWait(e, out);
        e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(sums_group, 1, 1));
    }
    /// convSums (no residual) of p.rows rows with `stack_rows` more behind them, which copy `stack`'s rows (`stack_stride`
    /// apart): every row's group sums are its output's, and sums rows past all of them are zero.
    pub fn convSumsStack(ops: Ops, e: mtl.ComputeEncoder, x: Ref, dynamic: Ref, base: Ref, out: Ref, sums: Sums, p: Conv, stack: Ref, stack_rows: u32, stack_stride: u32) !void {
        try checkConv(p);
        if (p.residual != 0 or stack_rows == 0) return error.BadDraftShape;
        const rows = try std.math.add(u32, p.rows, stack_rows);
        try checkSums(sums, rows, p.width);
        const xb = try span(p.rows, p.x_stride, p.width);
        const yb = try span(rows, p.y_stride, p.width);
        const db = try span(p.rows, p.dynamic_stride, 4 * (p.width / p.group));
        const bb = @as(usize, p.width) * 4 * 2;
        try refs(out, yb, &.{ x, dynamic, base, stack }, &.{ xb, db, bb, try span(stack_rows, stack_stride, p.width) });
        e.setPipeline(ops.pipes[@backingInt(Kind.conv_xs_stack)]);
        bind(e, .{ x, dynamic, base, x });
        e.setValue(p, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.setBuffer(sums.ref.buf, sums.ref.off, 6);
        e.setValue(sums.stride, 7);
        e.setBuffer(stack.buf, stack.off, 8);
        e.setValue([2]u32{ stack_rows, stack_stride }, 9);
        e.dispatchThreads(mtl.Size.of(p.width, rows, 1), mtl.Size.of(sums_group, 1, 1));
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
        ops.bindWait(e, out);
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
/// Whether every element index tf_bf16_linear computes for `p` stays below 2^31, so the
/// 32-bit twin's uint arithmetic equals the incumbent's size_t arithmetic bit for bit.
pub fn offsetsFitU31(p: Linear) bool {
    if (p.rows == 0 or p.n == 0) return false;
    const bar: u64 = 1 << 31;
    const x_max = (@as(u64, p.rows) - 1) * p.x_stride + p.k;
    const w_max = (@as(u64, p.n) - 1) * p.w_stride + p.k;
    const y_max = (@as(u64, p.rows) - 1) * p.y_stride + p.n;
    return x_max < bar and w_max < bar and y_max < bar;
}
/// A trial linear's shape: exact multiples, lane and column tails, strided rows, then
/// production-sized drafter shapes.
const TuneShape = struct { rows: u32, n: u32, k: u32, x_stride: u32, w_stride: u32, y_stride: u32 };
const tune_shapes = [_]TuneShape{
    .{ .rows = 1, .n = 8, .k = 32, .x_stride = 32, .w_stride = 32, .y_stride = 8 },
    .{ .rows = 3, .n = 100, .k = 1000, .x_stride = 1000, .w_stride = 1000, .y_stride = 100 },
    .{ .rows = 5, .n = 64, .k = 128, .x_stride = 160, .w_stride = 128, .y_stride = 96 },
    .{ .rows = 8, .n = 5120, .k = 5120, .x_stride = 5120, .w_stride = 5120, .y_stride = 5120 },
    .{ .rows = 2, .n = 2048, .k = 5120, .x_stride = 5120, .w_stride = 5120, .y_stride = 2048 },
};
const tune_time_shape = TuneShape{ .rows = 4, .n = 5120, .k = 5120, .x_stride = 5120, .w_stride = 5120, .y_stride = 5120 };
const TrialCtx = struct { pipe: mtl.Pipeline, x: Ref, w: Ref, y: Ref, p: Linear, chain: usize };
/// linear()'s binds and dispatch under `pipe`, `chain` launches back to back.
fn encodeTrial(e: mtl.ComputeEncoder, ctx: *anyopaque) void {
    const t: *TrialCtx = @ptrCast(@alignCast(ctx));
    var i: usize = 0;
    while (i < t.chain) : (i += 1) {
        e.setPipeline(t.pipe);
        e.setBuffer(t.x.buf, t.x.off, 0);
        e.setBuffer(t.w.buf, t.w.off, 1);
        e.setValue(t.p, 2);
        e.setBuffer(t.y.buf, t.y.off, 3);
        e.dispatchGroups(mtl.Size.of((t.p.n + 7) / 8, t.p.rows, 1), mtl.Size.of(256, 1, 1));
    }
}
/// One launch of each pipeline on synthetic seeded buffers; every written output element must
/// match bit for bit (stride padding is never written, so each row's `n` words are compared, not
/// the whole buffer). The two outputs start from different fills, so a launch that never ran also
/// mismatches.
fn bitCheckLinear(device: mtl.Device, queue: mtl.Queue, incumbent: mtl.Pipeline, twin: mtl.Pipeline, shape: TuneShape, seed: u64) !void {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const p = Linear{ .rows = shape.rows, .n = shape.n, .k = shape.k, .x_stride = shape.x_stride, .w_stride = shape.w_stride, .y_stride = shape.y_stride };
    try checkLinear(p);
    if (!offsetsFitU31(p)) return error.TuneShapeTooWide;
    const xb = try span(p.rows, p.x_stride, p.k);
    const wb = try span(p.n, p.w_stride, p.k);
    const yb = try span(p.rows, p.y_stride, p.n);
    const x = try device.buffer(xb, opts);
    defer x.deinit();
    const w = try device.buffer(wb, opts);
    defer w.deinit();
    const y0 = try device.buffer(yb, opts);
    defer y0.deinit();
    const y1 = try device.buffer(yb, opts);
    defer y1.deinit();
    tune.fillBf16(x.slice(u16, xb / 2), seed);
    tune.fillBf16(w.slice(u16, wb / 2), seed ^ 0xD1B54A32D192ED03);
    @memset(y0.contents()[0..yb], 0xA5);
    @memset(y1.contents()[0..yb], 0x5A);
    var c0 = TrialCtx{ .pipe = incumbent, .x = .{ .buf = x }, .w = .{ .buf = w }, .y = .{ .buf = y0 }, .p = p, .chain = 1 };
    var c1 = TrialCtx{ .pipe = twin, .x = .{ .buf = x }, .w = .{ .buf = w }, .y = .{ .buf = y1 }, .p = p, .chain = 1 };
    _ = try tune.launchSeconds(queue, encodeTrial, &c0);
    _ = try tune.launchSeconds(queue, encodeTrial, &c1);
    var row: u32 = 0;
    while (row < p.rows) : (row += 1) {
        const off = @as(usize, row) * p.y_stride * 2;
        const nbytes = @as(usize, p.n) * 2;
        if (!std.mem.eql(u8, y0.contents()[off..][0..nbytes], y1.contents()[off..][0..nbytes])) return error.TuneBitMismatch;
    }
}
/// The production-sized duel: two warmup cycles, then the timed round robin.
fn timeLinear32(device: mtl.Device, queue: mtl.Queue, incumbent: mtl.Pipeline, twin: mtl.Pipeline) !tune.Outcome {
    const shape = tune_time_shape;
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const p = Linear{ .rows = shape.rows, .n = shape.n, .k = shape.k, .x_stride = shape.x_stride, .w_stride = shape.w_stride, .y_stride = shape.y_stride };
    try checkLinear(p);
    const xb = try span(p.rows, p.x_stride, p.k);
    const wb = try span(p.n, p.w_stride, p.k);
    const yb = try span(p.rows, p.y_stride, p.n);
    const x = try device.buffer(xb, opts);
    defer x.deinit();
    const w = try device.buffer(wb, opts);
    defer w.deinit();
    const y = try device.buffer(yb, opts);
    defer y.deinit();
    tune.fillBf16(x.slice(u16, xb / 2), 0x243F6A8885A308D3);
    tune.fillBf16(w.slice(u16, wb / 2), 0x452821E638D01377);
    var c0 = TrialCtx{ .pipe = incumbent, .x = .{ .buf = x }, .w = .{ .buf = w }, .y = .{ .buf = y }, .p = p, .chain = 2 };
    var c1 = TrialCtx{ .pipe = twin, .x = .{ .buf = x }, .w = .{ .buf = w }, .y = .{ .buf = y }, .p = p, .chain = 2 };
    return tune.timeDuel(queue, encodeTrial, &c0, encodeTrial, &c1, tune.min_cycles, 2);
}
test "operator guards reject invalid width, stride and rounding before encoding" {
    try std.testing.expectError(error.BadDraftShape, checkLinear(.{ .rows = 0, .n = 64, .k = 128, .x_stride = 128, .w_stride = 128, .y_stride = 64 }));
    try std.testing.expectError(error.BadDraftShape, checkConv(.{ .rows = 8, .width = 64, .group = 3, .branch = 0, .x_stride = 64, .dynamic_stride = 16, .y_stride = 64 }));
    try std.testing.expectError(error.BadDraftShape, checkRope(.{ .rows = 8, .heads = 4, .dim = 127, .x_stride = 508, .y_stride = 508 }));
    try std.testing.expectError(error.BadDraftShape, normThreads(.{ .rows = 8, .heads = 4, .dim = 128, .x_stride = 512, .y_stride = 512, .round_normalized = 2 }));
}
test "linear32 twin guard holds on this tree's source" {
    try std.testing.expectEqual(@as(usize, 46), linear32_lines);
    try std.testing.expectEqual(@as(usize, 59), linear32_sites);
    try std.testing.expect(linear32_guarded);
    try std.testing.expectEqual(source.len - 59 * 2, linear32_text.len);
    try std.testing.expectEqual(@as(usize, 0), tune.countOccurrences(&linear32_text, tune.narrow_from));
    try std.testing.expect(std.mem.indexOf(u8, &linear32_text, "uint(row)*a.x_stride") != null);
    try std.testing.expect(std.mem.indexOf(u8, &linear32_text, "uint(col)*a.w_stride") != null);
    try std.testing.expect(std.mem.indexOf(u8, &linear32_text, "uint(row)*a.y_stride") != null);
}
test "linear32 serves only offsets below 2^31" {
    try std.testing.expect(offsetsFitU31(.{ .rows = 8, .n = 5120, .k = 5120, .x_stride = 5120, .w_stride = 5120, .y_stride = 5120 }));
    try std.testing.expect(!offsetsFitU31(.{ .rows = 0, .n = 8, .k = 8, .x_stride = 8, .w_stride = 8, .y_stride = 8 }));
    try std.testing.expect(offsetsFitU31(.{ .rows = 2, .n = 2, .k = 1, .x_stride = (1 << 31) - 2, .w_stride = 1, .y_stride = 2 }));
    try std.testing.expect(!offsetsFitU31(.{ .rows = 2, .n = 2, .k = 2, .x_stride = (1 << 31) - 2, .w_stride = 1, .y_stride = 2 }));
    try std.testing.expect(!offsetsFitU31(.{ .rows = 3, .n = 2, .k = 1, .x_stride = 1 << 30, .w_stride = 1, .y_stride = 2 }));
    try std.testing.expect(!offsetsFitU31(.{ .rows = 2049, .n = 1 << 20, .k = 8, .x_stride = 8, .w_stride = 8, .y_stride = 1 << 20 }));
}
