//! Typed draft callbacks share the target queue and encode primitives into one owned frame.
const std = @import("std");
const op = @import("operators.zig");
const ck = @import("../checkpoint.zig");
const Config = @import("config.zig").Config;
const Backend = @import("runtime_backend.zig").Backend;
const Ref = @import("runtime_backend.zig").Ref;
const Span = @import("runtime_backend.zig").Span;
const context = @import("runtime_context.zig");
fn backend(ptr: *anyopaque) *Backend {
    return @ptrCast(@alignCast(ptr));
}
fn begin(ptr: *anyopaque) !u64 {
    const b = backend(ptr);
    if (b.failed) return error.DraftBackendFailed;
    b.input_sums = null;
    b.hazards.clear();
    return b.frame.begin();
}
fn release(ptr: *anyopaque, epoch: u64) void {
    const b = backend(ptr);
    if (epoch != b.frame.epoch) return;
    // a frame abandoned before finish (an error path) now runs, and its context appends may overwrite ring rows
    if (b.frame.active and !b.frame.completed) b.failed = true;
    b.frame.release();
}
fn project(ptr: *anyopaque, epoch: u64, x: op.View, w: ck.Tensor, mode: op.Mode) !op.View {
    const b = backend(ptr);
    errdefer b.input_sums = null;
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(x.rows, @intCast(w.shape[0]), .bf16);
    const xs = try b.span(x);
    const ys = try b.span(y);
    if (mode == .bf16_reference) {
        b.input_sums = null;
        b.access(e, &.{xs}, &.{ys});
        try b.primitives.linear(e, try b.frame.ref(x), try b.weight(w), try b.frame.ref(y), .{ .rows = x.rows, .n = y.width, .k = x.width, .x_stride = x.stride, .w_stride = x.width, .y_stride = y.stride });
    } else if (b.tiled.get(@intFromPtr(w.bytes.ptr))) |t| {
        b.input_sums = null;
        if (x.stride != x.width or y.stride != y.width) return error.DraftPreparationUnqualified;
        const input = try b.frame.ref(x);
        const output = try b.frame.ref(y);
        const s = @import("core").row_projection.simd;
        b.access(e, &.{xs}, &.{ys});
        s.encode(e, try s.call(&b.simd.?, t.n, t.k, x.rows, false, false), t.matrix(), input.buf, input.off, output.buf, output.off);
    } else {
        const adapter = b.prepared.get(@intFromPtr(w.bytes.ptr)) orelse return error.DraftPreparationUnqualified;
        try encodePrepared(b, e, x, y, adapter);
        return y;
    }
    e.barrier();
    return y;
}
/// A prepared adapter's projection of `x` into `y`, sharing the input sums of an earlier projection of the same view.
fn encodePrepared(b: *Backend, e: @import("metal").ComputeEncoder, x: op.View, y: op.View, adapter: @import("core").affine4_lane.Adapter) !void {
    {
        const xs = try b.span(x);
        const ys = try b.span(y);
        const input = try b.frame.ref(x);
        const output = try b.frame.ref(y);
        const p = adapter.forRows(x.rows);
        const cacheable = p.reg and p.sums_pipe != null and p.sums_buffer != null and
            x.dtype == .bf16 and x.rows >= 1 and x.rows <= 16 and x.stride == x.width and
            x.width == p.k and p.k % 64 == 0 and p.group_count == p.k / 64 and
            y.stride == p.y_stride and p.output_bytes == 2;
        if (!cacheable or (if (b.input_sums) |s| !std.meta.eql(s.view, x) else false)) b.input_sums = null;
        if (b.input_sums) |s| {
            b.access(e, &.{ xs, Span.whole(s.sums.ref.buf) }, &.{ys});
            try p.encodeSums(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = adapter.words }, .{ .buf = adapter.metadata }, s.sums, .{ .buf = output.buf, .off = output.off }, x.rows);
        } else if (cacheable) {
            // the sums, a barrier, the product: no barrier behind it, so independent work runs beside it
            const sums_span = Span.whole(p.sums_buffer.?);
            b.access(e, &.{xs}, &.{sums_span});
            const sums = try p.encodeSumsOf(e, .{ .buf = input.buf, .off = input.off }, x.rows);
            b.access(e, &.{ xs, sums_span }, &.{ys});
            try p.encodeSums(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = adapter.words }, .{ .buf = adapter.metadata }, sums, .{ .buf = output.buf, .off = output.off }, x.rows);
            b.input_sums = .{ .view = x, .sums = sums };
        } else {
            if (p.sums_buffer) |sb| b.access(e, &.{xs}, &.{ ys, Span.whole(sb) }) else b.access(e, &.{xs}, &.{ys});
            // reg adapters take a prompt absorb's rows 128 at a time (16-row blocks on the grid, each block's words as alone)
            const step: u32 = if (p.reg) @import("core").lane_projection.reg_rows else 16;
            var first: u32 = 0;
            while (first < x.rows) : (first += step) {
                try adapter.encode(e, .{ .buf = input.buf, .off = input.off + @as(usize, first) * x.stride * 2 }, .{ .buf = output.buf, .off = output.off + @as(usize, first) * y.stride * 2 }, @min(x.rows - first, step));
            }
        }
    }
}
fn projectGroup(ptr: *anyopaque, epoch: u64, x: op.View, w: ck.Tensor, kind: u32, mode: op.Mode) !?op.View {
    const b = backend(ptr);
    if (mode == .bf16_reference or b.fused_count == 0) return null;
    const adapter = b.prepared.get(@intFromPtr(w.bytes.ptr) + kind) orelse return null;
    errdefer b.input_sums = null;
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(x.rows, @intCast(adapter.projection.n), .bf16);
    try encodePrepared(b, e, x, y, adapter);
    return y;
}
fn sliceCols(ptr: *anyopaque, epoch: u64, x: op.View, first: u32, width: u32) !op.View {
    const b = backend(ptr);
    _ = try b.encoder(epoch);
    if (first > x.width or width > x.width - first) return error.DraftOperatorShape;
    const r = try b.frame.ref(x);
    return b.frame.add(.{ .buf = r.buf, .off = r.off + @as(usize, first) * 2 }, x.rows, width, x.stride, .bf16);
}
fn norm(ptr: *anyopaque, epoch: u64, x: op.View, p: op.Norm) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(x.rows, x.width, .bf16);
    b.access(e, &.{try b.span(x)}, &.{try b.span(y)});
    try b.primitives.norm(e, try b.frame.ref(x), try b.weight(p.weight), try b.frame.ref(y), .{ .rows = x.rows, .heads = p.heads, .dim = p.head_dim, .x_stride = x.stride, .y_stride = y.stride, .eps = p.eps });
    return y;
}
fn embed(ptr: *anyopaque, epoch: u64, _: ck.Linear, ids: []const u32) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const tokens = try b.frame.alloc(@intCast(ids.len), 1, .u32);
    const r = try b.frame.ref(tokens);
    @memcpy(r.buf.slice(u32, ids.len), ids);
    const y = try b.frame.alloc(@intCast(ids.len), b.config.hidden, .bf16);
    const out = try b.frame.ref(y);
    b.access(e, &.{try b.span(tokens)}, &.{try b.span(y)});
    try b.target.glue.embedding(e, b.target.config, b.target.weights.embed, .{ .buffer = r.buf, .offset = r.off }, .{ .buffer = out.buf, .offset = out.off }, @intCast(ids.len));
    return y;
}
fn conv(ptr: *anyopaque, epoch: u64, x: op.View, p: op.Conv) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(x.rows, x.width, .bf16);
    if (p.residual) |r| b.access(e, &.{ try b.span(x), try b.span(p.dynamic), try b.span(r) }, &.{try b.span(y)}) else b.access(e, &.{ try b.span(x), try b.span(p.dynamic) }, &.{try b.span(y)});
    try b.primitives.conv(e, try b.frame.ref(x), try b.frame.ref(p.dynamic), try b.weight(p.base), if (p.residual) |r| try b.frame.ref(r) else null, try b.frame.ref(y), .{ .rows = x.rows, .width = x.width, .group = p.group, .block = p.block, .branch = p.branch, .residual = @intFromBool(p.residual != null), .x_stride = x.stride, .dynamic_stride = p.dynamic.stride, .y_stride = y.stride });
    return y;
}
fn rope(ptr: *anyopaque, epoch: u64, x: op.View, p: op.Rotary) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const positions = try b.frame.alloc(x.rows, 1, .u64);
    const r = try b.frame.ref(positions);
    @memcpy(r.buf.slice(u64, p.positions.len), p.positions);
    const y = try b.frame.alloc(x.rows, x.width, .bf16);
    b.access(e, &.{ try b.span(x), try b.span(positions) }, &.{try b.span(y)});
    try b.primitives.rope(e, try b.frame.ref(x), r, try b.frame.ref(y), .{ .rows = x.rows, .heads = p.heads, .dim = p.dim, .x_stride = x.stride, .y_stride = y.stride, .theta = @floatCast(p.theta) });
    return y;
}
fn swiglu(ptr: *anyopaque, epoch: u64, g: op.View, u: op.View) !op.View {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(g.rows, g.width, .bf16);
    b.access(e, &.{ try b.span(g), try b.span(u) }, &.{try b.span(y)});
    try b.primitives.swiglu(e, try b.frame.ref(g), try b.frame.ref(u), try b.frame.ref(y), .{ .rows = g.rows, .width = g.width, .gate_stride = g.stride, .up_stride = u.stride, .y_stride = y.stride });
    return y;
}
fn maskRows(ptr: *anyopaque, epoch: u64, x: op.View, first: u32, rows: u32) !op.View {
    const b = backend(ptr);
    _ = try b.encoder(epoch);
    if (first > x.rows or rows > x.rows - first) return error.DraftOperatorShape;
    const r = try b.frame.ref(x);
    return b.frame.add(.{ .buf = r.buf, .off = r.off + @as(usize, first) * x.stride * 2 }, rows, x.width, x.stride, .bf16);
}
fn head(ptr: *anyopaque, epoch: u64, x: op.View) !op.Head {
    const b = backend(ptr);
    const e = try b.encoder(epoch);
    const y = try b.frame.alloc(x.rows, b.head.linear.n, .bf16);
    const input = try b.frame.ref(x);
    const output = try b.frame.ref(y);
    // the group sums of `x` a projection just computed (the same tf_lane_sums values the head's own pass would
    // produce): the head reads them, so it dispatches no sums and no barriers of its own
    const Sums = @import("../quant_gpu.zig").Sums;
    var sums: ?Sums = null;
    if (b.input_sums) |s| if (std.meta.eql(s.view, x) and x.rows <= s.sums.stride) {
        sums = .{ .ref = .{ .buffer = s.sums.ref.buf, .offset = s.sums.ref.off }, .stride = s.sums.stride };
    };
    if (sums) |s| b.access(e, &.{ try b.span(x), Span.whole(s.ref.buffer) }, &.{try b.span(y)}) else b.access(e, &.{try b.span(x)}, &.{try b.span(y)});
    try b.target.kernels.quant(e, b.head.linear, .{ .buffer = input.buf, .offset = input.off }, sums, b.target.frame.get(.dims), .{ .buffer = output.buf, .offset = output.off }, x.rows);
    return .{ .logits = y, .token_ids = if (b.head.ids) |ids| try b.frame.add(.{ .buf = ids }, 1, b.head.linear.n, b.head.linear.n, .u32) else null };
}
fn lattice(ptr: *anyopaque, epoch: u64, h: op.Head, projected: op.View, c: Config) !op.Lattice {
    const b = backend(ptr);
    _ = try b.encoder(epoch);
    const output = try b.frame.alloc(h.logits.rows, @sizeOf(@import("core").bf16_topk.Row), .u8);
    const logits = try b.frame.ref(h.logits);
    const out = try b.frame.ref(output);
    b.access(b.frame.encoder, &.{try b.span(h.logits)}, &.{ try b.span(output), Span.whole(b.topk.part) });
    try b.topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = out.buf, .off = out.off }, .{ .rows = h.logits.rows, .vocab = h.logits.width, .k = c.topk, .stride = h.logits.stride });
    b.frame.finishBridged() catch |err| {
        b.failed = true;
        return err;
    };
    return @import("runtime_lattice.zig").read(b.allocator, &b.frame, output, projected, c, h);
}
/// `lattice`'s first half: the top-k (and a frame's pick behind it) encoded and the frame's last command buffer
/// committed, nothing waited for.
fn latticeBegin(ptr: *anyopaque, epoch: u64, h: op.Head, projected: op.View, c: Config) !op.View {
    const b = backend(ptr);
    _ = try b.encoder(epoch);
    const output = try b.frame.alloc(h.logits.rows, @sizeOf(@import("core").bf16_topk.Row), .u8);
    const logits = try b.frame.ref(h.logits);
    const out = try b.frame.ref(output);
    b.access(b.frame.encoder, &.{try b.span(h.logits)}, &.{ try b.span(output), Span.whole(b.topk.part) });
    try b.topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = out.buf, .off = out.off }, .{ .rows = h.logits.rows, .vocab = h.logits.width, .k = c.topk, .stride = h.logits.stride });
    if (b.frame.pick) |pick| {
        b.frame.pick = null;
        // the pick reads the top-k rows and the projected row: everything before it ends first
        b.barrier(b.frame.encoder);
        try pick.encode(pick.ptr, b.frame.encoder, out, if (h.token_ids) |ids| try b.frame.ref(ids) else null, h.logits.width, try b.frame.ref(projected));
    }
    b.frame.commitLast() catch |err| {
        b.failed = true;
        return err;
    };
    return output;
}
/// `lattice`'s second half: the frame waited for and the picks of `latticeBegin` read back.
fn latticeEnd(ptr: *anyopaque, epoch: u64, output: op.View, h: op.Head, projected: op.View, c: Config) !op.Lattice {
    const b = backend(ptr);
    if (!b.frame.active or b.frame.epoch != epoch) return error.DraftFrameNotActive;
    b.frame.waitAll() catch |err| {
        b.failed = true;
        return err;
    };
    return @import("runtime_lattice.zig").read(b.allocator, &b.frame, output, projected, c, h);
}
/// Rows [first, first + rows) of `x`, an alias in the frame.
fn rowsOf(b: *Backend, x: op.View, first: u32, rows: u32) !op.View {
    if (first > x.rows or rows > x.rows - first) return error.DraftOperatorShape;
    const r = try b.frame.ref(x);
    return b.frame.add(.{ .buf = r.buf, .off = r.off + @as(usize, first) * x.stride * x.dtype.size() }, rows, x.width, x.stride, x.dtype);
}
fn lattices(ptr: *anyopaque, epoch: u64, h: op.Head, projected: op.View, c: Config, block: u32, out: []op.Lattice) !void {
    const b = backend(ptr);
    _ = try b.encoder(epoch);
    if (block < 2 or h.logits.rows != out.len * block or out.len > 8) return error.DraftOperatorShape;
    // top-k takes one stream's rows a call (at most 16), all before the frame's one wait
    var heads: [8]op.Head = undefined;
    var picks: [8]op.View = undefined;
    for (heads[0..out.len], picks[0..out.len], 0..) |*hs, *pick, s| {
        const first: u32 = @intCast(s * block + 1); // a stream's anchor row drafts nothing
        hs.* = .{ .logits = try rowsOf(b, h.logits, first, block - 1), .token_ids = h.token_ids };
        pick.* = try b.frame.alloc(block - 1, @sizeOf(@import("core").bf16_topk.Row), .u8);
        const logits = try b.frame.ref(hs.logits);
        const rows = try b.frame.ref(pick.*);
        b.access(b.frame.encoder, &.{try b.span(hs.logits)}, &.{ try b.span(pick.*), Span.whole(b.topk.part) });
        try b.topk.encode(b.frame.encoder, .{ .buf = logits.buf, .off = logits.off }, .{ .buf = rows.buf, .off = rows.off }, .{ .rows = block - 1, .vocab = hs.logits.width, .k = c.topk, .stride = hs.logits.stride });
    }
    b.frame.finish() catch |err| {
        b.failed = true;
        return err;
    };
    var made: usize = 0;
    errdefer for (out[0..made]) |*l| l.deinit();
    for (out, heads[0..out.len], picks[0..out.len], 0..) |*l, hs, pick, s| {
        l.* = try @import("runtime_lattice.zig").read(b.allocator, &b.frame, pick, try rowsOf(b, projected, @intCast(s * block + 1), block - 1), c, hs);
        made += 1;
    }
}
pub const vtable = op.Ops.VTable{ .begin = begin, .release = release, .project = project, .norm = norm, .embed = embed, .conv = conv, .rope = rope, .attention = context.attention, .swiglu = swiglu, .mask_rows = maskRows, .head = head, .lattice = lattice, .lattices = lattices, .context_begin = context.begin, .context_append = context.append, .context_commit = context.commit, .context_stage = context.stage, .context_abort = context.abort, .stream_append = context.streamAppend, .project_group = projectGroup, .slice_cols = sliceCols, .lattice_begin = latticeBegin, .lattice_end = latticeEnd };
