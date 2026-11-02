//! The Metal 4 command path (macOS 26+): an MTL4CommandQueue behind the same Queue, CommandBuffer and ComputeEncoder
//! the rest of TensorFold uses (command.zig dispatches on their `m4` field). Every kernel, pipeline and dispatch is the
//! one the classic path encodes; what changes is how the commands are recorded and submitted:
//! - a command buffer is a reusable slot owning one MTL4CommandAllocator, one MTL4ArgumentTable and its upload chunks,
//!   recycled (allocator reset, chunks and retained buffers returned) only once the queue's timeline event shows its
//!   last epoch complete, so nothing the GPU may still read is reused or freed;
//! - setBuffer binds the buffer's GPU address in the argument table; setBytes copies into the slot's upload chunk and
//!   binds that address; every bound allocation joins the queue's residency set (Metal 4 makes nothing resident
//!   implicitly) and is retained until the slot completes (Metal 4 command buffers retain nothing);
//! - event waits and signals, which Metal 4 issues on the queue, split the buffer where the caller placed them
//!   (mtl4_book.Plan), so their order against the encoded work is the classic one; each commit ends with a timeline
//!   signal that status() and wait() read;
//! - each encoder opens behind a queue barrier and every `barrier()` is an encoder barrier over all stages, at least as
//!   strong as the classic ordering.
//! TF_MTL4=0 keeps every queue on the classic path; a device or OS without Metal 4 keeps it too.
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const book = @import("mtl4_book.zig");

const Id = objc.Id;

/// A spin lock over the queue's short bookkeeping sections (slot, chunk and residency updates).
const Spin = struct {
    m: std.atomic.Mutex = .unlocked,
    fn lock(s: *Spin) void {
        while (!s.m.tryLock()) std.atomic.spinLoopHint();
    }
    fn unlock(s: *Spin) void {
        s.m.unlock();
    }
};

/// MTLStageAll (NSIntegerMax) and MTL4VisibilityOptionDevice.
const stages_all: usize = std.math.maxInt(isize);
const visibility_device: usize = 1 << 0;
/// Buffer slots an argument table binds ([[buffer(0..30)]], Metal's limit for compute).
const table_buffers: usize = 31;
/// Upload chunks a queue holds (64 KB each): far more than the command buffers a queue keeps in flight.
const upload_chunks: u32 = 256;
/// How long wait() blocks before it reports a stuck epoch (it keeps waiting).
const wait_report_ms: u64 = 20_000;
/// Allocations in the queue's residency set past which an idle queue rebuilds it.
const trim_at: usize = 4096;

/// Whether queues take the Metal 4 path: TF_MTL4 not "0", macOS 26 or later, and a device that makes MTL4 queues.
pub fn enabled(device: Id) bool {
    if (std.c.getenv("TF_MTL4")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return false;
    if (objc.osVersion().major < 26) return false;
    inline for (.{ "newMTL4CommandQueue", "newCommandAllocator", "newArgumentTableWithDescriptor:error:", "newResidencySetWithDescriptor:error:" }) |name|
        if (!objc.msg(bool, device, "respondsToSelector:", .{objc.sel(name)})) return false;
    return true;
}

pub const Queue = struct {
    gpa: std.mem.Allocator,
    device: Id,
    queue: Id,
    timeline: Id,
    epoch: u64 = 0,
    lock: Spin = .{},
    slots: std.ArrayListUnmanaged(*Slot) = .empty,
    upload: Id,
    upload_bytes: [*]u8,
    upload_gpu: u64,
    chunks: book.Chunks,
    resident: Id,
    resident_ids: std.AutoHashMapUnmanaged(usize, void) = .empty,
    resident_dirty: bool = false,
    sets: std.AutoHashMapUnmanaged(usize, void) = .empty,

    /// A Metal 4 queue with its timeline, upload buffer and residency set; null when any part cannot be made (the
    /// caller then makes a classic queue).
    pub fn init(gpa: std.mem.Allocator, device: Id) ?*Queue {
        const queue = objc.msg(?Id, device, "newMTL4CommandQueue", .{}) orelse return null;
        const timeline = objc.msg(?Id, device, "newSharedEvent", .{}) orelse {
            objc.release(queue);
            return null;
        };
        const upload = objc.msg(?Id, device, "newBufferWithLength:options:", .{ @as(usize, upload_chunks) * book.Chunks.size, types.ResourceOptions.shared | types.ResourceOptions.untracked }) orelse {
            objc.release(timeline);
            objc.release(queue);
            return null;
        };
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTLResidencySetDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        objc.msg(void, desc, "setInitialCapacity:", .{@as(usize, 256)});
        var err: ?Id = null;
        const resident = objc.msg(?Id, device, "newResidencySetWithDescriptor:error:", .{ desc, &err }) orelse {
            std.log.warn("metal4: no residency set ({s}): classic queue", .{objc.errorText(err)});
            objc.release(upload);
            objc.release(timeline);
            objc.release(queue);
            return null;
        };
        const q = gpa.create(Queue) catch return null;
        q.* = .{ .gpa = gpa, .device = device, .queue = queue, .timeline = timeline, .upload = upload, .upload_bytes = objc.msg([*]u8, upload, "contents", .{}), .upload_gpu = objc.msg(u64, upload, "gpuAddress", .{}), .chunks = book.Chunks.init(gpa, upload_chunks) catch {
            gpa.destroy(q);
            return null;
        }, .resident = resident };
        objc.msg(void, resident, "addAllocation:", .{upload});
        q.resident_ids.put(gpa, @intFromPtr(upload), {}) catch {};
        objc.msg(void, resident, "commit", .{});
        objc.msg(void, queue, "addResidencySet:", .{resident});
        return q;
    }

    fn seen(q: *Queue) u64 {
        return objc.msg(u64, q.timeline, "signaledValue", .{});
    }

    /// A buffer the GPU reads through an address: resident through the queue's set from the next commit on.
    fn hold(q: *Queue, buffer: Id) void {
        const gop = q.resident_ids.getOrPut(q.gpa, @intFromPtr(buffer)) catch return;
        if (gop.found_existing) return;
        objc.msg(void, q.resident, "addAllocation:", .{buffer});
        q.resident_dirty = true;
    }

    pub fn addResidencySet(q: *Queue, set: Id) void {
        q.lock.lock();
        defer q.lock.unlock();
        const gop = q.sets.getOrPut(q.gpa, @intFromPtr(set)) catch return;
        if (gop.found_existing) return;
        objc.msg(void, q.queue, "addResidencySet:", .{set});
    }

    pub fn removeResidencySet(q: *Queue, set: Id) void {
        q.lock.lock();
        defer q.lock.unlock();
        if (!q.sets.remove(@intFromPtr(set))) return;
        objc.msg(void, q.queue, "removeResidencySet:", .{set});
    }

    /// The residency set holds every allocation any slot ever bound; buffers made per request would grow it without
    /// end, so when it passes `trim_at` and no slot is in flight it starts again from the upload buffer.
    fn trim(q: *Queue) void {
        if (q.resident_ids.count() < trim_at) return;
        for (q.slots.items) |s| if (s.state != .free) return;
        objc.msg(void, q.resident, "removeAllAllocations", .{});
        q.resident_ids.clearRetainingCapacity();
        objc.msg(void, q.resident, "addAllocation:", .{q.upload});
        q.resident_ids.put(q.gpa, @intFromPtr(q.upload), {}) catch {};
        objc.msg(void, q.resident, "commit", .{});
        q.resident_dirty = false;
    }

    /// Returns every committed slot the timeline shows complete to the free state.
    fn reclaim(q: *Queue) void {
        const now = q.seen();
        for (q.slots.items) |s| if (s.state == .committed and book.done(s.epoch, now)) s.recycle();
    }

    /// A free slot, open for encoding: a completed one reused, else a new one.
    pub fn slot(q: *Queue) *Slot {
        q.lock.lock();
        defer q.lock.unlock();
        q.reclaim();
        q.trim();
        for (q.slots.items) |s| if (s.state == .free) {
            s.state = .open;
            return s;
        };
        const s = Slot.init(q) catch |e| std.debug.panic("metal4: no command buffer slot: {t}", .{e});
        q.slots.append(q.gpa, s) catch std.debug.panic("metal4: out of memory", .{});
        s.state = .open;
        return s;
    }

    /// An upload chunk, waiting for the oldest committed slot to complete when every chunk is taken.
    fn chunk(q: *Queue) u32 {
        while (true) {
            if (q.chunks.take()) |c| return c;
            q.reclaim();
            if (q.chunks.take()) |c| return c;
            var oldest: ?*Slot = null;
            for (q.slots.items) |s| if (s.state == .committed and (oldest == null or s.epoch < oldest.?.epoch)) {
                oldest = s;
            };
            const o = oldest orelse std.debug.panic("metal4: every upload chunk is held by open command buffers", .{});
            q.lock.unlock();
            q.waitEpoch(o.epoch);
            q.lock.lock();
        }
    }

    fn waitEpoch(q: *Queue, epoch: u64) void {
        var waited: u64 = 0;
        while (!book.done(epoch, q.seen())) {
            if (objc.msg(bool, q.timeline, "waitUntilSignaledValue:timeoutMS:", .{ epoch, @as(u64, 1000) })) return;
            waited += 1000;
            if (waited % wait_report_ms == 0) std.log.warn("metal4: still waiting for epoch {d} (at {d}) after {d} s", .{ epoch, q.seen(), waited / 1000 });
        }
    }

    pub fn deinit(q: *Queue) void {
        q.lock.lock();
        if (q.epoch != 0) {
            q.lock.unlock();
            q.waitEpoch(q.epoch);
            q.lock.lock();
        }
        for (q.slots.items) |s| s.deinit();
        q.slots.deinit(q.gpa);
        q.chunks.deinit(q.gpa);
        q.resident_ids.deinit(q.gpa);
        q.sets.deinit(q.gpa);
        objc.release(q.resident);
        objc.release(q.upload);
        objc.release(q.timeline);
        objc.release(q.queue);
        q.lock.unlock();
        q.gpa.destroy(q);
    }
};

pub const Slot = struct {
    q: *Queue,
    generation: u32 = 1,
    state: enum { free, open, committed } = .free,
    epoch: u64 = 0,
    allocator: Id,
    table: Id,
    plan: book.Plan(Id) = .{},
    /// Metal 4 command buffers begun by this slot since its last reuse, and ones free to begin again.
    used: std.ArrayListUnmanaged(Id) = .empty,
    spare: std.ArrayListUnmanaged(Id) = .empty,
    taken: std.ArrayListUnmanaged(u32) = .empty,
    cursor: book.Cursor = .{},
    retained: std.ArrayListUnmanaged(Id) = .empty,
    /// Buffers this slot retained, with their GPU addresses: a buffer bound again costs one lookup.
    known: std.AutoHashMapUnmanaged(usize, u64) = .empty,

    fn init(q: *Queue) !*Slot {
        const allocator = objc.msg(?Id, q.device, "newCommandAllocator", .{}) orelse return error.NoCommandAllocator;
        errdefer objc.release(allocator);
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTL4ArgumentTableDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        objc.msg(void, desc, "setMaxBufferBindCount:", .{table_buffers});
        var err: ?Id = null;
        const table = objc.msg(?Id, q.device, "newArgumentTableWithDescriptor:error:", .{ desc, &err }) orelse {
            std.log.err("metal4: argument table: {s}", .{objc.errorText(err)});
            return error.NoArgumentTable;
        };
        const s = try q.gpa.create(Slot);
        s.* = .{ .q = q, .allocator = allocator, .table = table };
        return s;
    }

    /// Back to free: the allocator reset, its chunks and retained buffers returned. Called under the queue lock once
    /// the slot's epoch completed.
    fn recycle(s: *Slot) void {
        objc.msg(void, s.allocator, "reset", .{});
        for (s.taken.items) |c| s.q.chunks.give(c);
        s.taken.clearRetainingCapacity();
        for (s.retained.items) |r| objc.release(r);
        s.retained.clearRetainingCapacity();
        s.known.clearRetainingCapacity();
        s.spare.appendSlice(s.q.gpa, s.used.items) catch {
            for (s.used.items) |cb| objc.release(cb);
        };
        s.used.clearRetainingCapacity();
        s.plan.reset();
        s.cursor = .{};
        s.epoch = 0;
        s.generation +%= 1;
        if (s.generation == 0) s.generation = 1;
        s.state = .free;
    }

    fn deinit(s: *Slot) void {
        if (s.state == .committed) s.recycle();
        for (s.retained.items) |r| objc.release(r);
        for (s.used.items) |cb| objc.release(cb);
        for (s.spare.items) |cb| objc.release(cb);
        s.retained.deinit(s.q.gpa);
        s.known.deinit(s.q.gpa);
        s.used.deinit(s.q.gpa);
        s.spare.deinit(s.q.gpa);
        s.taken.deinit(s.q.gpa);
        s.plan.deinit(s.q.gpa);
        objc.release(s.table);
        objc.release(s.allocator);
        s.q.gpa.destroy(s);
    }

    // mtl4_book.Plan's callbacks: begin a buffer on the slot's allocator, end one, drop an empty one.
    pub fn make(s: *Slot) !Id {
        const cb = s.spare.pop() orelse (objc.msg(?Id, s.q.device, "newCommandBuffer", .{}) orelse return error.NoCommandBuffer);
        try s.used.append(s.q.gpa, cb);
        objc.msg(void, cb, "beginCommandBufferWithAllocator:", .{s.allocator});
        return cb;
    }
    pub fn close(_: *Slot, cb: Id) void {
        objc.msg(void, cb, "endCommandBuffer", .{});
    }
    pub fn drop(_: *Slot, _: Id) void {} // stays in `used`, reused after the slot recycles

    /// A compute encoder on the open buffer: the slot's argument table bound, behind a barrier on all earlier work of
    /// the queue (the classic path's encoders are at least this ordered).
    pub fn encoder(s: *Slot) Id {
        const cb = s.plan.work(s) catch |e| std.debug.panic("metal4: command buffer: {t}", .{e});
        const enc = objc.msg(?Id, cb, "computeCommandEncoder", .{}) orelse std.debug.panic("metal4: no compute encoder", .{});
        objc.msg(void, enc, "setArgumentTable:", .{s.table});
        objc.msg(void, enc, "barrierAfterQueueStages:beforeStages:visibilityOptions:", .{ stages_all, stages_all, visibility_device });
        return enc;
    }

    /// `buffer`'s address plus `offset` at `index`; the buffer stays alive and resident until the slot completes.
    pub fn bind(s: *Slot, buffer: Id, offset: usize, index: usize) void {
        objc.msg(void, s.table, "setAddress:atIndex:", .{ s.keep(buffer) + offset, index });
    }

    /// The address `buffer` plus `offset` has, after keeping the buffer (for indirect arguments and copies).
    pub fn address(s: *Slot, buffer: Id, offset: usize) u64 {
        return s.keep(buffer) + offset;
    }

    /// Keeps `buffer` alive and resident until the slot completes; its GPU address.
    pub fn keep(s: *Slot, buffer: Id) u64 {
        const gop = s.known.getOrPut(s.q.gpa, @intFromPtr(buffer)) catch std.debug.panic("metal4: out of memory", .{});
        if (gop.found_existing) return gop.value_ptr.*;
        gop.value_ptr.* = objc.msg(u64, buffer, "gpuAddress", .{});
        s.retained.append(s.q.gpa, objc.retain(buffer)) catch std.debug.panic("metal4: out of memory", .{});
        s.q.lock.lock();
        defer s.q.lock.unlock();
        s.q.hold(buffer);
        return gop.value_ptr.*;
    }

    /// `bytes` copied into the slot's upload chunk and bound at `index` (the classic setBytes).
    pub fn bytes(s: *Slot, data: []const u8, index: usize) void {
        if (data.len > book.Chunks.size) std.debug.panic("metal4: setBytes of {d} bytes", .{data.len});
        const at = s.cursor.place(data.len) orelse blk: {
            s.q.lock.lock();
            const c = s.q.chunk();
            s.q.lock.unlock();
            s.taken.append(s.q.gpa, c) catch std.debug.panic("metal4: out of memory", .{});
            s.cursor.start(c);
            break :blk s.cursor.place(data.len).?;
        };
        @memcpy(s.q.upload_bytes[at..][0..data.len], data);
        objc.msg(void, s.table, "setAddress:atIndex:", .{ s.q.upload_gpu + at, index });
    }

    pub fn waitFor(s: *Slot, event: Id, value: u64) void {
        s.plan.wait(s.q.gpa, s, event, value) catch std.debug.panic("metal4: out of memory", .{});
    }

    pub fn signal(s: *Slot, event: Id, value: u64) void {
        s.plan.signal(s.q.gpa, s, event, value) catch std.debug.panic("metal4: out of memory", .{});
    }

    /// The buffer's steps to the queue in order, then the timeline signal its status and wait read.
    pub fn commit(s: *Slot) void {
        s.plan.close(s.q.gpa, s) catch std.debug.panic("metal4: out of memory", .{});
        const q = s.q;
        q.lock.lock();
        defer q.lock.unlock();
        if (q.resident_dirty) {
            objc.msg(void, q.resident, "commit", .{});
            q.resident_dirty = false;
        }
        for (s.plan.items.items) |item| switch (item) {
            .wait => |w| objc.msg(void, q.queue, "waitForEvent:value:", .{ w.event, w.value }),
            .signal => |g| objc.msg(void, q.queue, "signalEvent:value:", .{ g.event, g.value }),
            .cb => |cb| {
                var one = [1]Id{cb};
                objc.msg(void, q.queue, "commit:count:", .{ @as([*]const Id, &one), @as(usize, 1) });
            },
        };
        q.epoch += 1;
        objc.msg(void, q.queue, "signalEvent:value:", .{ q.timeline, q.epoch });
        s.epoch = q.epoch;
        s.state = .committed;
    }

    pub fn status(s: *Slot, generation: u32) types.CommandBufferStatus {
        if (s.generation != generation) return .completed; // reused: the buffer it held completed
        return switch (s.state) {
            .free => .completed,
            .open => .not_enqueued,
            .committed => if (book.done(s.epoch, s.q.seen())) .completed else .committed,
        };
    }

    pub fn wait(s: *Slot, generation: u32) void {
        if (s.generation != generation or s.state != .committed) return;
        s.q.waitEpoch(s.epoch);
    }
};
