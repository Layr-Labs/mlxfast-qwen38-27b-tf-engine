// Modified by Layr Labs, Inc. in 2026; see THIRD_PARTY_NOTICES.md.
//! A draft frame owns reusable buffers until its serial command buffers complete; parts commit in order behind a fence.
const std = @import("std");
const mtl = @import("metal");
const op = @import("operators.zig");
pub const Ref = @import("core").draft_ops.Ref;
const Entry = struct { ref: Ref, view: op.View };
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
/// Whether a frame's command buffer failed on the GPU, its Metal error logged.
fn failed(cb: mtl.CommandBuffer) bool {
    const msg = cb.failure() orelse return false;
    std.log.err("qwen27: a draft frame failed on the GPU: {s}", .{msg});
    return true;
}

/// Keeps the GPU busy from a proposal's last command buffer until the verify behind it starts. The host reads the
/// proposal before it encodes the verify, and a GPU left idle meanwhile starts the verify's first command buffer a few
/// hundred microseconds after its commit; a busy one starts it within a few tens. One 32-thread group spins behind the
/// proposal until the verify's first encoder stores the spin's epoch (`end`), or for `bound` iterations (a few
/// milliseconds) when no verify follows. It reads and writes nothing else, so no result depends on it.
pub const Bridge = struct {
    spin: mtl.Pipeline,
    stop: mtl.Pipeline,
    /// [0]: the epoch of the last stop; [1]: a sink the spin never writes in practice (keeps its loop alive).
    flag: mtl.Buffer,
    epoch: u32 = 0,
    open: bool = false,
    const bound: u32 = 50_000;
    const source =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void tf_bridge_spin(device atomic_uint* flag [[buffer(0)]], constant uint2& p [[buffer(1)]],
        \\                           uint tid [[thread_position_in_grid]]) {
        \\  uint x = tid + 0x9e3779b9u;
        \\  for (uint i = 0; i < p.x; ++i) {
        \\    if ((i & 31u) == 0u && int(atomic_load_explicit(flag, memory_order_relaxed) - p.y) >= 0) break;
        \\    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        \\  }
        \\  if (x == 0x12345678u) atomic_store_explicit(flag + 1, x, memory_order_relaxed);
        \\}
        \\kernel void tf_bridge_stop(device atomic_uint* flag [[buffer(0)]], constant uint& epoch [[buffer(1)]]) {
        \\  atomic_store_explicit(flag, epoch, memory_order_relaxed);
        \\}
    ;
    pub fn init(device: mtl.Device) !Bridge {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const spin = try mtl.Pipeline.init(device, lib, "tf_bridge_spin", false);
        errdefer spin.deinit();
        const stop = try mtl.Pipeline.init(device, lib, "tf_bridge_stop", false);
        errdefer stop.deinit();
        // untracked: the verify's stop must not wait for the spin it ends
        const flag = try device.buffer(16, options);
        @memset(flag.slice(u32, 4), 0);
        return .{ .spin = spin, .stop = stop, .flag = flag };
    }
    pub fn deinit(b: *Bridge) void {
        b.spin.deinit();
        b.stop.deinit();
        b.flag.deinit();
    }
    /// A spin in its own command buffer, started when `fence` (updated by the proposal's last encoder) is reached.
    fn launch(b: *Bridge, queue: mtl.Queue, fence: mtl.objc.Id) void {
        b.epoch +%= 1;
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        mtl.objc.msg(void, e.id, "waitForFence:", .{fence});
        e.setPipeline(b.spin);
        e.setBuffer(b.flag, 0, 0);
        e.setValue([2]u32{ bound, b.epoch }, 1);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        b.open = true;
    }
    /// First in the verify's first encoder, beside its own work: ends the open spin once the verify runs.
    fn end(b: *Bridge, e: mtl.ComputeEncoder) void {
        if (!b.open) return;
        b.open = false;
        e.setPipeline(b.stop);
        e.setBuffer(b.flag, 0, 0);
        e.setValue(b.epoch, 1);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }
};

pub const Frame = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    pool: std.ArrayList(mtl.Buffer) = .empty,
    entries: std.ArrayList(Entry) = .empty,
    cursor: usize = 0,
    epoch: u64 = 0,
    active: bool = false,
    completed: bool = false,
    /// The last command buffer is committed (commitLast); the frame completes at waitLast.
    committed: bool = false,
    /// An early frame (runtime_model.prepareNext), encoded and committed behind a verify inside its autorelease pool:
    /// one command buffer, retained until the frame completes or is released, its bridge spin launched by launchBridge
    /// behind work committed after it. (On an M4, a command buffer committed behind one that has not started yet starts
    /// only once that one ends: work committed after the frame starts runs beside it, work committed before it does not.)
    early: bool = false,
    /// The bridge's spin is due behind the frame (launchBridge).
    bridge_due: bool = false,
    cb: mtl.CommandBuffer = undefined,
    encoder: mtl.ComputeEncoder = undefined,
    external: ?Entry = null,
    external_id: u64 = 0,
    external_epoch: u64 = 0,
    fence: ?mtl.objc.Id = null,
    parts: [8]mtl.CommandBuffer = undefined,
    part_count: usize = 0,
    ops: u32 = 0,
    residency: ?mtl.ResidencySet.Follow = null,
    /// Bridges a proposal's frame to the verify behind it (finishBridged, endBridge); none if it failed to build.
    bridge: ?Bridge = null,
    /// Signalled behind a frame's last command: finish returns once the GPU reaches it, and the frame's command buffers
    /// are waited for and checked when the next frame begins (checkUnchecked).
    poll_event: ?mtl.SharedEvent = null,
    poll_value: u64 = 0,
    unchecked: [9]mtl.CommandBuffer = undefined,
    n_unchecked: usize = 0,
    pub fn setResident(f: *Frame, hook: ?mtl.ResidencySet.Follow) void {
        for (f.pool.items) |buffer| {
            if (f.residency) |h| h.leave(h.ctx, buffer);
            if (hook) |h| h.join(h.ctx, buffer);
        }
        f.residency = hook;
    }
    pub fn deinit(f: *Frame) void {
        f.release();
        f.checkUnchecked() catch {};
        if (f.poll_event) |ev| ev.deinit();
        f.setResident(null);
        if (f.fence) |fence| mtl.objc.release(fence);
        if (f.bridge) |*b| b.deinit();
        for (f.pool.items) |buffer| buffer.deinit();
        f.pool.deinit(f.allocator);
        f.entries.deinit(f.allocator);
    }
    pub fn cacheBytes(f: *const Frame) u64 {
        var bytes: u64 = 0;
        for (f.pool.items) |buffer| bytes +|= @intCast(buffer.length());
        return bytes;
    }
    /// Release idle reusable buffers. A live frame stays visible so phase close fails closed.
    pub fn drain(f: *Frame) u64 {
        if (f.active) return f.cacheBytes();
        for (f.pool.items) |buffer| {
            if (f.residency) |h| h.leave(h.ctx, buffer);
            buffer.deinit();
        }
        f.pool.clearRetainingCapacity();
        f.entries.clearRetainingCapacity();
        f.cursor = 0;
        f.external = null;
        return f.cacheBytes();
    }
    /// Waits for and checks the command buffers a finished frame returned before they reported completion.
    pub fn checkUnchecked(f: *Frame) !void {
        var bad = false;
        for (f.unchecked[0..f.n_unchecked]) |cb| {
            cb.wait();
            bad = failed(cb) or bad;
            cb.release();
        }
        f.n_unchecked = 0;
        if (bad) return error.DraftGpuFailure;
    }
    pub fn begin(f: *Frame) !u64 {
        return f.beginEarly(false);
    }
    /// begin, an `early` frame (see Frame.early) when asked.
    pub fn beginEarly(f: *Frame, early: bool) !u64 {
        if (f.active or f.epoch >= 0x7fffffff) return error.DraftFrameBusy;
        try f.checkUnchecked();
        f.epoch += 1;
        if (f.external_epoch != f.epoch) f.external = null;
        f.cursor = 0;
        f.entries.clearRetainingCapacity();
        f.early = early;
        f.bridge_due = false;
        f.cb = f.open();
        f.encoder = f.cb.compute(.concurrent);
        f.active = true;
        f.completed = false;
        f.committed = false;
        f.part_count = 0;
        f.ops = 0;
        return f.epoch;
    }
    fn open(f: *Frame) mtl.CommandBuffer {
        const cb = f.queue.commandBuffer();
        return if (f.early) cb.retain() else cb;
    }
    /// Ends an early frame's hold on its command buffers.
    fn drop(f: *Frame) void {
        if (!f.early) return;
        for (f.parts[0..f.part_count]) |cb| cb.release();
        f.cb.release();
    }
    /// Commits the work so far and opens the next buffer behind a fence, so the GPU starts early; between ops only.
    pub fn part(f: *Frame) !void {
        if (!f.active or f.completed or f.committed or f.early or f.part_count == f.parts.len) return;
        if (f.fence == null) f.fence = mtl.objc.msg(?mtl.objc.Id, f.device.id, "newFence", .{}) orelse return error.NoFence;
        mtl.objc.msg(void, f.encoder.id, "updateFence:", .{f.fence.?});
        f.encoder.end();
        f.cb.commit();
        f.parts[f.part_count] = f.cb;
        f.part_count += 1;
        f.cb = f.open();
        f.encoder = f.cb.compute(.concurrent);
        mtl.objc.msg(void, f.encoder.id, "waitForFence:", .{f.fence.?});
    }
    pub fn finish(f: *Frame) !void {
        return f.finishWith(false);
    }
    /// finish, leaving the bridge's spin running behind the frame for the verify that follows a proposal.
    pub fn finishBridged(f: *Frame) !void {
        return f.finishWith(f.bridge != null);
    }
    /// Ends a bridge's open spin from the verify's first encoder (a forward.Tail on the frame).
    pub fn endBridge(ptr: *anyopaque, e: mtl.ComputeEncoder) anyerror!void {
        const f: *Frame = @ptrCast(@alignCast(ptr));
        if (f.bridge) |*b| b.end(e);
    }
    fn finishWith(f: *Frame, bridged: bool) !void {
        try f.commitLast(bridged);
        return f.waitLast();
    }
    /// Ends and commits the frame's last command buffer, the bridge's spin behind it when `bridged` (an early frame's at
    /// launchBridge), without a wait: waitLast returns once the GPU reaches its end. Nothing more is encoded into the frame.
    pub fn commitLast(f: *Frame, bridged: bool) !void {
        if (!f.active or f.completed or f.committed) return error.DraftFrameNotActive;
        if (bridged and f.fence == null) f.fence = mtl.objc.msg(?mtl.objc.Id, f.device.id, "newFence", .{}) orelse return error.NoFence;
        if (bridged) mtl.objc.msg(void, f.encoder.id, "updateFence:", .{f.fence.?});
        f.encoder.end();
        if (f.poll_event == null) f.poll_event = f.device.sharedEvent() catch null;
        if (f.poll_event) |ev| {
            f.poll_value += 1;
            f.cb.signal(ev, f.poll_value);
        }
        f.cb.commit();
        f.committed = true;
        f.bridge_due = bridged;
        if (!f.early) f.launchBridge();
    }
    /// The bridge's spin behind a committed frame, if it is due.
    pub fn launchBridge(f: *Frame) void {
        if (!f.active or !f.committed or !f.bridge_due) return;
        f.bridge_due = false;
        f.bridge.?.launch(f.queue, f.fence.?);
    }
    /// After commitLast: returns once the GPU reached the frame's last command (its buffers are checked at the next begin).
    pub fn waitLast(f: *Frame) !void {
        if (!f.active or f.completed or !f.committed) return error.DraftFrameNotActive;
        // the frame's tokens gate the next verify's encode: return once the GPU reaches the frame's last command
        if (f.poll_event) |ev| {
            while (ev.value() < f.poll_value and f.cb.status() != .@"error") std.atomic.spinLoopHint();
            if (f.cb.status() != .@"error") {
                for (f.parts[0..f.part_count]) |cb| {
                    f.unchecked[f.n_unchecked] = cb.retain();
                    f.n_unchecked += 1;
                }
                f.unchecked[f.n_unchecked] = f.cb.retain();
                f.n_unchecked += 1;
                f.drop();
                f.completed = true;
                return;
            }
        }
        // poll its status rather than sleep until the completion handler wakes this thread, then wait() returns at once
        while (true) switch (f.cb.status()) {
            .completed, .@"error" => break,
            else => std.atomic.spinLoopHint(),
        };
        f.cb.wait();
        f.completed = true;
        defer f.drop();
        for (f.parts[0..f.part_count]) |cb| {
            cb.wait();
            if (failed(cb)) return error.DraftGpuFailure;
        }
        if (failed(f.cb)) return error.DraftGpuFailure;
    }
    pub fn release(f: *Frame) void {
        if (f.active and !f.completed) {
            if (!f.committed) {
                f.encoder.end();
                // an abandoned frame's buffer still goes to the queue: one left uncommitted holds a queue slot for good,
                // and enough of them block every later commandBuffer() (a hang, not an error). The caller fails the backend.
                f.cb.commit();
            }
            f.cb.wait();
            for (f.parts[0..f.part_count]) |cb| cb.wait(); // committed parts still use the pool
            f.drop();
        }
        f.active = false;
        f.external = null;
    }
    pub fn require(f: *const Frame, epoch: u64) !void {
        if (!f.active or f.completed or f.committed or f.epoch != epoch) return error.DraftFrameNotActive;
    }
    pub fn add(f: *Frame, r: Ref, rows: u32, width: u32, stride: u32, dtype: @import("../checkpoint.zig").DType) !op.View {
        const view = op.View{ .handle = (f.epoch << 32) | (f.entries.items.len + 1), .rows = rows, .width = width, .stride = stride, .dtype = dtype };
        try view.check(rows, width, dtype);
        const bytes = try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, rows - 1, stride), width), dtype.size());
        if (r.off > r.buf.length() or bytes > r.buf.length() - r.off) return error.DraftOperatorShape;
        try f.entries.append(f.allocator, .{ .ref = r, .view = view });
        return view;
    }
    pub fn alloc(f: *Frame, rows: u32, width: u32, dtype: @import("../checkpoint.zig").DType) !op.View {
        if (!f.active or f.completed or f.committed or rows == 0 or width == 0) return error.DraftFrameNotActive;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, rows, width), dtype.size());
        // Each slot keeps less than 64 KiB of spare scratch as context scores grow across rounds.
        const reserved = (try std.math.add(usize, bytes, 65535)) & ~@as(usize, 65535);
        if (f.cursor == f.pool.items.len) {
            const buffer = try f.device.buffer(reserved, options);
            f.pool.append(f.allocator, buffer) catch |err| {
                buffer.deinit();
                return err;
            };
            if (f.residency) |h| h.join(h.ctx, buffer);
        } else if (f.pool.items[f.cursor].length() < bytes) {
            const buffer = try f.device.buffer(reserved, options);
            if (f.residency) |h| h.leave(h.ctx, f.pool.items[f.cursor]);
            f.pool.items[f.cursor].deinit();
            f.pool.items[f.cursor] = buffer;
            if (f.residency) |h| h.join(h.ctx, buffer);
        }
        const buffer = f.pool.items[f.cursor];
        f.cursor += 1;
        return f.add(.{ .buf = buffer }, rows, width, width, dtype);
    }
    pub fn ref(f: *const Frame, v: op.View) !Ref {
        if (v.handle & (@as(u64, 1) << 63) != 0) {
            if (!f.active or f.external_epoch != f.epoch) return error.StaleDraftView;
            const entry = f.external orelse return error.StaleDraftView;
            if (!std.meta.eql(v, entry.view)) return error.StaleDraftView;
            return entry.ref;
        }
        const index: usize = @intCast(v.handle & 0xffffffff);
        if (!f.active or v.handle >> 32 != f.epoch or index == 0 or index > f.entries.items.len) return error.StaleDraftView;
        const entry = f.entries.items[index - 1];
        if (!std.meta.eql(v, entry.view)) return error.StaleDraftView;
        return entry.ref;
    }
    pub fn borrow(f: *Frame, r: Ref, rows: u32, width: u32) !op.View {
        if (f.active or rows == 0 or width == 0 or r.off > r.buf.length() or @as(usize, rows) * width * 2 > r.buf.length() - r.off) return error.DraftFrameBusy;
        f.external_id += 1;
        if (f.external_id >= (@as(u64, 1) << 63)) return error.DraftFrameBusy;
        const view = op.View{ .handle = (@as(u64, 1) << 63) | f.external_id, .rows = rows, .width = width, .stride = width, .dtype = .bf16 };
        f.external = .{ .ref = r, .view = view };
        f.external_epoch = f.epoch + 1;
        return view;
    }
};
