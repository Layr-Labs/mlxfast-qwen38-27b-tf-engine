//! Staggered prompt segments, for any family: each segment on its own queue does what a serial chunk of its rows does, so the output equals the chunks run in order.
const std = @import("std");
const mtl = @import("metal");
pub const Fence = @import("fence.zig").Fence;
const stagger = @import("stagger.zig");

/// Segments one chunk can take.
pub const MAX = stagger.MAX;

/// Where layer i waits for the segment before (core/stagger.zig).
pub const Wait = stagger.Wait;

/// A segment's command buffer and open encoder; hooks encode into `enc` and leave it open.
pub const Lane = struct {
    k: usize,
    cb: mtl.CommandBuffer,
    enc: mtl.ComputeEncoder,
    fence: Fence, // orders the segment's encoders: buffers are untracked, so encoders could overlap
};

pub const rows = stagger.rows;
pub const start = stagger.start;
pub const Call = stagger.Call;
pub const next = stagger.next;

/// Hooks leave the encoder open and commit nothing; what segment k+1 reads of segment k must be written by k's mixer or earlier.
pub fn run(device: mtl.Device, queues: []const mtl.Queue, layers: usize, mode: mtl.DispatchType, fam: anytype) !f64 {
    return runCommitting(device, queues, layers, mode, fam, 0);
}

/// `run`, committing every `every` layers (0: once), fenced in order; returns the chunk's summed GPU seconds.
pub fn runCommitting(device: mtl.Device, queues: []const mtl.Queue, layers: usize, mode: mtl.DispatchType, fam: anytype, every: usize) !f64 {
    return runCommittingFrom(device, queues, layers, mode, fam, every, every);
}

/// `runCommitting` whose first commit comes after `first` layers, so the GPU starts while the CPU encodes the rest.
/// A family that declares `beforeCommit` runs it once, right before the pass's first commit (whichever chunk that is).
/// A polled single-lane pass signals `poll_event` behind its last command and the host returns once the GPU reaches it,
/// tens of microseconds before the command buffers report themselves completed. Their waits and failure checks run at
/// the start of the next pass (`checkUnchecked`); the GPU has finished their work by then, so no buffer is reused early.
var poll_event: ?mtl.SharedEvent = null;
var poll_value: u64 = 0;
var unchecked: [17]mtl.CommandBuffer = undefined;
var n_unchecked: usize = 0;
/// A polled single-lane pass's fence, made once for its queue and kept: a pass's first encoder waits on nothing and
/// each later one on the update just before it, so a kept fence orders a pass exactly as a new one does.
var poll_fence: ?struct { fence: Fence, queue: mtl.objc.Id } = null;

/// The family's work due right before a pass's first commit, if it declares any (runCommittingFrom).
fn beforeCommit(fam: anytype) !void {
    if (comptime @hasDecl(std.meta.Child(@TypeOf(fam)), "beforeCommit")) try fam.beforeCommit();
}

/// Waits for and checks the command buffers a polled pass returned before they reported completion.
pub fn checkUnchecked() !void {
    var failed = false;
    for (unchecked[0..n_unchecked]) |cb| {
        // a polled pass signals behind its last command and returns with its earlier chunks retained: the queue ran
        // them before the command the poll waited for, so a chunk that already reports completion needs no blocking
        // wait, only its failure check (Metal reports failure once a command buffer has completed)
        switch (cb.status()) {
            .completed, .@"error" => {},
            else => cb.wait(),
        }
        if (cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            failed = true;
        }
        cb.release();
    }
    n_unchecked = 0;
    if (failed) return error.GpuFailed;
}

pub fn runCommittingFrom(device: mtl.Device, queues: []const mtl.Queue, layers: usize, mode: mtl.DispatchType, fam: anytype, first: usize, every: usize) !f64 {
    return runCommittingEarly(device, queues, layers, mode, fam, first, every, false);
}

/// `runCommittingFrom`; `early` (one lane) also commits after the first layer's mixer and after each of the first three
/// layers, so a long pass's GPU starts on the first mixer while the CPU encodes the rest of that layer and never waits
/// for the first `every` layers' encode.
pub fn runCommittingEarly(device: mtl.Device, queues: []const mtl.Queue, layers: usize, mode: mtl.DispatchType, fam: anytype, first: usize, every: usize, early: bool) !f64 {
    try checkUnchecked();
    const n = queues.len;
    if (n == 0 or n > MAX) return error.Segments;
    var evs: [MAX - 1]mtl.Event = undefined;
    var n_evs: usize = 0;
    defer for (evs[0..n_evs]) |e| e.deinit();
    while (n_evs + 1 < n) : (n_evs += 1) evs[n_evs] = try device.event();
    var lanes: [MAX]Lane = undefined;
    var n_lanes: usize = 0;
    // decoded rows poll one lane (as the poll event and the unchecked buffers above): that lane keeps its queue's fence
    const kept_fence = n == 1 and (if (poll_fence) |kept| kept.queue == queues[0].id else true) and (if (comptime @hasDecl(std.meta.Child(@TypeOf(fam)), "polls")) fam.polls() else false);
    defer if (!kept_fence) for (lanes[0..n_lanes]) |l| l.fence.deinit();
    for (queues) |q| {
        const fence = if (kept_fence) (if (poll_fence) |kept| kept.fence else made: {
            const f = try Fence.init(device);
            poll_fence = .{ .fence = f, .queue = q.id };
            break :made f;
        }) else try Fence.init(device);
        const cb = q.commandBuffer();
        lanes[n_lanes] = .{ .k = n_lanes, .cb = cb, .enc = cb.compute(mode), .fence = fence };
        n_lanes += 1;
    }
    var done: [16]mtl.CommandBuffer = undefined;
    var n_done: usize = 0;
    const X = struct {
        fam: @TypeOf(fam),
        lanes: []Lane,
        evs: []const mtl.Event,
        mode: mtl.DispatchType,
        queue: mtl.Queue,
        every: usize,
        layers: usize,
        first: usize,
        early: bool,
        done: *[16]mtl.CommandBuffer,
        n_done: *usize,

        /// Commits the lane's command buffer so far and opens the next behind the lane's fence (one lane only); the
        /// family's beforeCommit runs before the first.
        fn commitLane(x: *const @This(), k: usize) !void {
            if (x.lanes.len != 1 or x.n_done.* == x.done.len) return;
            if (x.n_done.* == 0) try beforeCommit(x.fam);
            const l = &x.lanes[k];
            l.fence.update(l.enc);
            l.enc.end();
            l.cb.commit();
            x.done[x.n_done.*] = l.cb;
            x.n_done.* += 1;
            l.cb = x.queue.commandBuffer();
            l.enc = l.cb.compute(x.mode);
            l.fence.wait(l.enc);
        }

        /// Ends the lane's encoder, waits for (or signals) `value` between its encoders, and opens the next behind the fence.
        fn sync(x: *const @This(), l: *Lane, ev: mtl.Event, value: u64, wait_: bool) void {
            l.fence.update(l.enc);
            l.enc.end();
            if (wait_) l.cb.waitForEvent(ev, value) else l.cb.signalEvent(ev, value);
            l.enc = l.cb.compute(x.mode);
            l.fence.wait(l.enc);
        }
        pub fn begin(x: *const @This(), k: usize) !void {
            try x.fam.begin(&x.lanes[k]);
        }
        pub fn wait(x: *const @This(), i: usize) Wait {
            return x.fam.wait(i);
        }
        pub fn waitFor(x: *const @This(), k: usize, i: usize) !void {
            x.sync(&x.lanes[k], x.evs[k - 1], i + 1, true);
            try x.fam.handoff(&x.lanes[k], i);
        }
        pub fn signal(x: *const @This(), k: usize, i: usize) !void {
            x.sync(&x.lanes[k], x.evs[k], i + 1, false);
        }
        pub fn pre(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.pre(&x.lanes[k], i);
        }
        pub fn mixer(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.mixer(&x.lanes[k], i);
            if (x.early and i == 0 and i + 1 < x.layers) try x.commitLane(k);
        }
        pub fn post(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.post(&x.lanes[k], i);
            if (x.every == 0 or i + 1 >= x.layers) return;
            const end = i + 1;
            const paced = if (comptime @hasField(std.meta.Child(@TypeOf(x.fam)), "pace_verify")) x.fam.pace_verify else false;
            if (paced and x.lanes.len == 1 and end != x.first) {
                if (end % 4 != 0) return;
                // Keep one buffer queued behind the one executing. Poll only the
                // penultimate submission. Each 16th layer is an unconditional handoff.
                const submitted = x.n_done.*;
                if (end % 16 != 0 and submitted >= 2 and x.done[submitted - 2].status() != .completed) return;
            } else if (end != x.first and end % x.every != 0 and !(x.early and i < 3)) return;
            try x.commitLane(k);
        }
        pub fn finish(x: *const @This(), k: usize) !void {
            try x.fam.finish(&x.lanes[k]);
        }
    };
    const x: X = .{ .fam = fam, .lanes = lanes[0..n], .evs = evs[0..n_evs], .mode = mode, .queue = queues[0], .every = every, .layers = layers, .first = first, .early = early, .done = &done, .n_done = &n_done };
    stagger.drive(n, layers, &x) catch |err| { // close the open encoders, drop the open command buffers, let committed chunks finish
        for (lanes[0..n]) |l| l.enc.end();
        for (done[0..n_done]) |cb| cb.wait();
        return err;
    };
    for (lanes[0..n]) |l| l.enc.end();
    if (n_done == 0) try beforeCommit(fam);
    const polled = if (comptime @hasDecl(std.meta.Child(@TypeOf(fam)), "polls")) fam.polls() else false;
    // one polled lane signals the event behind its last command (see poll_event)
    if (polled and n == 1 and poll_event == null) poll_event = device.sharedEvent() catch null;
    const signalled = polled and n == 1 and poll_event != null;
    if (signalled) {
        poll_value += 1;
        lanes[0].cb.signal(poll_event.?, poll_value);
    }
    for (lanes[0..n]) |l| l.cb.commit();
    // work the family queues behind the last chunk before the host waits on it (a round's keep)
    const queued: anyerror!void = if (comptime @hasDecl(std.meta.Child(@TypeOf(fam)), "committed")) fam.committed() else {};
    if (signalled) {
        const ev = poll_event.?;
        while (ev.value() < poll_value and lanes[0].cb.status() != .@"error") std.atomic.spinLoopHint();
        if (lanes[0].cb.status() != .@"error") {
            for (done[0..n_done]) |cb| {
                unchecked[n_unchecked] = cb.retain();
                n_unchecked += 1;
            }
            unchecked[n_unchecked] = lanes[0].cb.retain();
            n_unchecked += 1;
            try queued;
            return 0;
        }
    }
    // a family that polls (decoded rows) spins on its last chunks' status rather than sleeping until the completion wakes it
    if (polled) for (lanes[0..n]) |l| while (true) switch (l.cb.status()) {
        .completed, .@"error" => break,
        else => std.atomic.spinLoopHint(),
    };
    for (lanes[0..n]) |l| l.cb.wait(); // every segment ends before a failure returns and its buffers are reused
    var gpu: f64 = 0;
    for (done[0..n_done]) |cb| {
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        gpu += cb.gpuSeconds();
    }
    for (lanes[0..n]) |l| {
        if (l.cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        gpu = if (n_done > 0) gpu + l.cb.gpuSeconds() else @max(gpu, l.cb.gpuSeconds());
    }
    try queued;
    return gpu;
}

test {
    _ = stagger;
}
