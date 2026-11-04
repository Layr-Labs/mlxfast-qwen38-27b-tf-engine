// Modified by Layr Labs, Inc. in 2026; see THIRD_PARTY_NOTICES.md.
//! Host-controlled draft rounds keep only consumed rows and leave the final emitted bonus token pending.
const std = @import("std");
const mtl = @import("metal");
const TargetSession = @import("../session.zig");
const Runner = @import("../decode_round.zig").Runner;
const gpuFailed = @import("../decode_round.zig").gpuFailed;
const Round = @import("../round_plan.zig").Round;
const Draft = @import("runtime_model.zig").Model;
const audit_journal = @import("audit.zig");
pub const AuditJournal = audit_journal.Journal;
pub const appendTo = audit_journal.appendTo;
pub const Proposer = struct { ptr: *anyopaque, call: *const fn (*anyopaque, u32, u32) anyerror!@import("selector.zig").Tree };
pub const Result = struct { tokens: []u32, rounds: usize, verified: usize, matched: usize, min_rows: u32 = 0, extended: u32 = 0, extension_landed: u32 = 0 };
/// What the context says follows `path` (the pending token first), at most `max` tokens; borrowed until the next call.
pub const Extender = struct { ptr: *anyopaque, call: *const fn (*anyopaque, []const u32, u32) anyerror![]const u32 };
/// The weight of a round in Generation.accept_share, and the share below which frames draft on q4 copies.
const accept_step: f32 = 0.125;
const precise_below: f32 = 0.97;
/// The target's first layer's input projections, whole (quant_gpu.Kernels.warmAll): the first weights a verify streams.
fn warmFirstLayer(ptr: *anyopaque, e: mtl.ComputeEncoder) anyerror!void {
    const m: *@import("../model.zig").Model = @ptrCast(@alignCast(ptr));
    if (m.weights.layers.len == 0) return;
    switch (m.weights.layers[0].mixer) {
        .linear => |g| {
            try m.kernels.warmAll(e, g.qkv);
            try m.kernels.warmAll(e, g.zba);
        },
        .attention => |at| {
            try m.kernels.warmAll(e, at.q);
            try m.kernels.warmAll(e, at.kv);
        },
    }
}
pub const Generation = struct {
    allocator: std.mem.Allocator,
    runner: *Runner,
    draft: *Draft,
    kept: mtl.Buffer,
    taps: mtl.Buffer,
    proposer: ?Proposer = null,
    matcher: @import("round_gpu.zig").Matcher,
    commit: @import("core").tree_commit_gpu.Ops,
    keep: ?mtl.CommandBuffer = null, // the last round's cache and state commit, running behind the drafter
    /// Signalled behind each keep's last command (keep_signal: the pending keep's value, 0 if it signals none): a verify's
    /// settle sees the keep's work done tens of microseconds before its completion is reported, which comes only behind
    /// the drafter frame's, committed before it (settleKeep).
    keep_event: ?mtl.SharedEvent = null,
    keep_value: u64 = 0,
    keep_signal: u64 = 0,
    /// A keep whose work settleKeep saw done before its completion was reported: checked once the verify behind it ran,
    /// before that verify's result is read (checkKeep).
    keep_done: ?mtl.CommandBuffer = null,
    picked: ?@import("core").segments.Fence = null, // updated behind a verify's picks; its keep waits on it
    pending: ?u32 = null, // runRound's next token as the last accept picked it; anything else that moves the target clears it
    journal: ?*AuditJournal = null,
    /// Moving share of rounds a sibling landed in (each round's weight `accept_step`): below `precise_below` the frames
    /// draft on the q4 copies of the linears drafted below 4 bits, which miss fewer siblings where they are missed.
    accept_share: f32 = 1,
    /// The drafter scratch's refill (Frame.refill) while a prefill's pass runs: joined before the drafter's first frame.
    refill: ?std.Thread = null,
    /// The rows of the prompt chunk in flight, for aheadDue.
    ahead_total: u64 = 0,
    pub fn init(a: std.mem.Allocator, runner: *Runner, draft: *Draft) !Generation {
        if (runner.model != draft.backend.target or runner.model.config.hidden != draft.graph.config.hidden or runner.model.config.layers != draft.graph.config.target_layers) return error.TargetBinding;
        runner.capture_taps = true;
        for (draft.graph.config.taps, &runner.taps.ids) |id, *to| to.* = id;
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const kept = try runner.model.device.buffer(128 * 4, options);
        errdefer kept.deinit();
        const taps = try runner.model.device.buffer(128 * @as(usize, draft.graph.config.tapWidth()) * 2, options);
        errdefer taps.deinit();
        const matcher = try @import("round_gpu.zig").Matcher.init(runner.model.device);
        errdefer matcher.deinit();
        const commit = try @import("core").tree_commit_gpu.Ops.init(runner.model.device);
        // the greedy verify head's sieve: its 2-bit copy now, not in the first timed round; one that fails to build
        // leaves the full head
        if (runner.headSieve()) |_| {} else |err| std.log.warn("qwen27: head sieve off: {t}", .{err});
        // the bridge spin behind each early frame first streams the next verify's first input projections into the system cache
        if (draft.backend.frame.bridge) |*bridge| bridge.warm = .{ .ptr = runner.model, .encode = warmFirstLayer };
        return .{ .allocator = a, .runner = runner, .draft = draft, .kept = kept, .taps = taps, .matcher = matcher, .commit = commit };
    }
    pub fn deinit(g: Generation) void {
        // an early frame waits on the picks' fence and reads the matcher's result: it ends before they go
        g.draft.drainNext() catch {};
        if (g.keep_done) |cb| {
            cb.wait();
            cb.release();
        }
        if (g.keep) |cb| {
            cb.wait();
            cb.release();
        }
        if (g.keep_event) |ev| ev.deinit();
        g.commit.deinit();
        if (g.picked) |f| f.deinit();
        g.matcher.deinit();
        g.kept.deinit();
        g.taps.deinit();
    }
    fn absorb(g: *Generation, path: []const u32) !void {
        try g.absorbRows(path);
        if (g.draft.committed_end != g.runner.offsets[0]) return error.BadDraftPosition;
    }
    /// A prompt pass's taps go straight into the drafter's ring at its next slots (Taps.into) when the pass's rows fit
    /// there without wrapping: the ring then holds the words the tap gather would copy, with no gather after the pass.
    fn tapsInto(g: *Generation, rows: usize) bool {
        const r = g.draft.ring();
        const slot: usize = @intCast(r.end.* % r.window);
        if (g.keep != null or !g.runner.capture_taps or rows > g.runner.taps.capacity or rows > r.window - slot) return false;
        g.runner.taps.into = .{ .buffer = r.buf, .offset = slot * r.stride, .stride = r.stride };
        return true;
    }
    /// A prompt pass's rows into the drafter: `direct`, its taps already stand in the ring's next slots (tapsInto); else
    /// gathered 128 rows a call (its frame's), as the target's pass may be longer.
    fn absorbChunk(g: *Generation, rows: usize, direct: bool) !void {
        if (direct) {
            const t = &g.runner.taps;
            try t.complete();
            if (t.rows != rows) return error.BadTapGather;
            t.begin(); // the planes hold none of this pass's rows: nothing gathers them
            g.draft.ring().end.* += rows;
            if (g.draft.committed_end != g.runner.offsets[0]) return error.BadDraftPosition;
            return;
        }
        var path: [128]u32 = undefined;
        var first: usize = 0;
        while (first < rows) : (first += path.len) {
            const n = @min(path.len, rows - first);
            for (path[0..n], 0..) |*index, i| index.* = @intCast(first + i);
            try g.absorbRows(path[0..n]);
        }
        if (g.draft.committed_end != g.runner.offsets[0]) return error.BadDraftPosition;
    }
    fn absorbRows(g: *Generation, path: []const u32) !void {
        std.debug.assert(g.keep == null);
        @memcpy(g.kept.slice(u32, path.len), path);
        const cb = g.runner.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try g.runner.taps.accepted(g.runner.model.glue, e, path, .{ .buffer = g.kept }, .{ .buffer = g.taps });
        e.end();
        cb.commit();
        cb.wait();
        if (gpuFailed(cb, "a draft taps gather")) return error.DraftTapGpuFailure;
        try g.draft.absorb(.{ .buf = g.taps }, @intCast(path.len));
    }
    pub fn advance(g: *Generation, token: u32) !void {
        g.pending = null;
        std.debug.assert(g.keep == null);
        try (TargetSession.Session{ .runner = g.runner }).step(token);
        try g.absorb(&.{0});
    }
    pub fn promptChunk(g: *Generation, ids: []const u32, last: bool) !void {
        g.pending = null;
        std.debug.assert(g.keep == null);
        if (g.runner.failed or g.draft.backend.failed or g.draft.committed_end != g.runner.offsets[0] or ids.len == 0 or ids.len > g.runner.model.frame.capacity) return error.BadDraftContext;
        const direct = g.tapsInto(ids.len);
        defer g.runner.taps.into = null;
        // the taps land in the ring's slots: each segment but the last of a staggered pass gives its rows' taps to the
        // drafter as it ends on the GPU, while the later segments still run (Model.absorbAhead); only a chunk that is the
        // whole prompt (the last, with nothing committed before it), so `ahead_total` is the prompt's rows
        g.ahead_total = ids.len;
        if (direct and last) g.runner.rows_done = .{ .ptr = g, .due = aheadDue, .run = aheadRun };
        defer g.runner.rows_done = null;
        try (TargetSession.Session{ .runner = g.runner }).promptChunk(ids, last);
        try g.absorbChunk(ids.len, direct);
    }
    fn aheadDue(ptr: *anyopaque, end: u32) bool {
        const g: *Generation = @ptrCast(@alignCast(ptr));
        return g.keep == null and g.draft.canAbsorbAhead(end, g.ahead_total);
    }
    fn aheadRun(ptr: *anyopaque, end: u32) anyerror!void {
        const g: *Generation = @ptrCast(@alignCast(ptr));
        if (g.refill) |t| { // the scratch the frame takes, made beside the pass
            t.join();
            g.refill = null;
        }
        try g.draft.absorbAhead(end, g.ahead_total);
    }
    /// `promptChunk` with decoded rows' arithmetic (an earlier reply's tokens).
    pub fn decodeChunk(g: *Generation, ids: []const u32, last: bool) !void {
        g.pending = null;
        std.debug.assert(g.keep == null);
        if (g.runner.failed or g.draft.backend.failed or g.draft.committed_end != g.runner.offsets[0] or ids.len == 0 or ids.len > g.runner.model.frame.capacity) return error.BadDraftContext;
        const direct = g.tapsInto(ids.len);
        defer g.runner.taps.into = null;
        try (TargetSession.Session{ .runner = g.runner }).decodeChunk(ids, last);
        try g.absorbChunk(ids.len, direct);
    }
    pub fn prefill(g: *Generation, ids: []const u32, chunk: usize) !void {
        g.pending = null;
        g.accept_share = 1;
        g.draft.backend.precise = false;
        std.debug.assert(g.keep == null);
        if (g.runner.failed or g.draft.backend.failed or g.draft.backend.frame.active or g.draft.backend.transaction != null or g.draft.committed_end != 0 or g.runner.offsets[0] != 0 or chunk == 0 or chunk > g.runner.model.frame.capacity) return error.BadDraftContext;
        // the drafter's scratch, freed at the last phase close, is made again on a helper thread while the GPU runs this
        // pass (Frame.refill): the catch-up frame behind it then waits on no allocation. Nothing here touches the frame.
        const frame = &g.draft.backend.frame;
        g.refill = if (frame.refillDue()) std.Thread.spawn(.{}, @TypeOf(frame.*).refill, .{frame}) catch null else null;
        defer if (g.refill) |t| {
            t.join();
            g.refill = null;
        };
        var first: usize = 0;
        while (first < ids.len) {
            const count = @min(ids.len - first, g.runner.promptRows(@intCast(chunk)));
            try g.promptChunk(ids[first..][0..count], first + count == ids.len);
            first += count;
        }
    }
    /// Ends a pending keep and an early frame and forgets the picked token, before anything but runRound uses the slot.
    pub fn settle(g: *Generation) !void {
        g.pending = null;
        const drained = g.draft.drainNext();
        try g.finishKeep();
        return drained;
    }
    /// A round's keep runs behind the drafter; it ends before the next verify's GPU work and before a run returns.
    fn finishKeep(g: *Generation) !void {
        const done = g.checkKeep();
        try g.finishPending();
        return done;
    }
    /// The keep settleKeep saw done on the GPU: its completion has been reported by now (a verify ran behind it), and it
    /// is checked as finishKeep checks a keep.
    fn checkKeep(g: *Generation) !void {
        const cb = g.keep_done orelse return;
        g.keep_done = null;
        defer cb.release();
        while (true) switch (cb.status()) {
            .completed, .@"error" => break,
            else => std.atomic.spinLoopHint(),
        };
        if (gpuFailed(cb, "a draft round's keep")) {
            g.runner.failed = true;
            return error.DraftKeepGpuFailure;
        }
    }
    fn finishPending(g: *Generation) !void {
        const cb = g.keep orelse return;
        g.keep = null;
        defer cb.release();
        // its completion is reported behind the drafter frame's (committed before it), right before the next verify's
        // first commit (settle): poll its status rather than sleep until the completion wakes this thread
        while (true) switch (cb.status()) {
            .completed, .@"error" => break,
            else => std.atomic.spinLoopHint(),
        };
        if (gpuFailed(cb, "a draft round's keep")) {
            g.runner.failed = true;
            return error.DraftKeepGpuFailure;
        }
    }
    /// The round's picks, accepted rows' taps and head row: the verify's tail, behind its head in its last command buffer.
    const Picks = struct {
        g: *Generation,
        ids: []const u32,
        parents: []const i32,
        budget: u32,
        eos: []const u32,
        round: *const Round,
        shape: ?@import("core").tree_round.Match = null,
        /// The next round's frame is committed behind the verify (Model.prepareNext), its taps gathered into the ring.
        early: bool = false,
        /// The keep waits for the verify's end to go to the queue (release), behind the early frame.
        held: bool = false,
        /// The rows' picks come from the runner's head sieve (the forward wrote no logits): no head row to keep.
        sieve: bool = false,
        fn encode(ptr: *anyopaque, e: mtl.ComputeEncoder) anyerror!void {
            const p: *Picks = @ptrCast(@alignCast(ptr));
            const g = p.g;
            // the matcher and keep fence their own dependencies; taps and head read only the matcher's result
            const logits = g.runner.model.frame.get(.logits);
            const width: u32 = @intCast(g.runner.model.config.vocab);
            const shape = if (p.sieve) sieved: {
                const sieve = if (g.runner.sieve) |*s| s else return error.DraftKeepGpuFailure;
                try sieve.encode(e, g.runner.model, @intCast(p.ids.len), g.matcher.picks());
                e.barrier();
                break :sieved try g.matcher.encodePicked(e, p.ids, p.parents, width, p.budget, p.eos);
            } else try g.matcher.encode(e, .{ .buf = logits.buffer, .off = logits.offset }, p.ids, p.parents, width, p.budget, p.eos);
            const result = @import("core").tree_commit_gpu.Ref{ .buf = g.matcher.buffers[4] };
            const taps = @import("core").tree_commit_gpu.Taps{ .rows = shape.rows, .width = g.runner.taps.width, .capacity = g.runner.taps.capacity, .planes = 5 };
            if (p.early) {
                // the kept taps (at most the budget's 2) go straight to the drafter's ring at its next slots
                const r = g.draft.ring();
                try g.commit.tapsInto(e, result, .{ .buf = g.runner.taps.buffer }, .{ .buf = r.buf, .off = @as(usize, @intCast(r.end.* % r.window)) * r.stride }, taps, @min(2, shape.rows));
            } else try g.commit.taps(e, result, .{ .buf = g.runner.taps.buffer }, .{ .buf = g.taps }, taps);
            // a sieved round's row 0 is no logits row: verifyAccept sets Runner.head_token in its place
            if (!p.sieve) try g.commit.head(e, result, .{ .buf = logits.buffer, .off = logits.offset }, .{ .rows = shape.rows, .width = width, .stride = width });
            p.shape = shape;
            g.picked.?.update(e);
        }
        /// The keep, encoded and committed behind the verify's last command buffer while the GPU still runs it,
        /// so its replay starts on the picks' fence instead of after the host wakes, reads and encodes. An early round
        /// commits the next round's frame there instead and the keep once the verify ran (release), beside that frame.
        fn keep(ptr: *anyopaque) anyerror!void {
            const p: *Picks = @ptrCast(@alignCast(ptr));
            const g = p.g;
            // the last round's keep ended before this verify's first commit (settle): never two keeps pending
            if (g.keep != null) try g.finishKeep();
            const result = @import("core").tree_commit_gpu.Ref{ .buf = g.matcher.buffers[4] };
            const keep_cb = g.runner.model.queue.commandBuffer();
            const k = keep_cb.compute(.concurrent);
            g.picked.?.wait(k);
            var kept = false;
            errdefer if (!kept) k.end();
            try g.runner.keepDevice(p.round, k, g.commit, result);
            k.end();
            kept = true;
            // its event behind its last command, for the next verify's settle (settleKeep)
            if (g.keep_event == null) g.keep_event = g.runner.model.device.sharedEvent() catch null;
            g.keep_signal = 0;
            if (g.keep_event) |ev| {
                g.keep_value += 1;
                keep_cb.signal(ev, g.keep_value);
                g.keep_signal = g.keep_value;
            }
            // the next round's frame goes first, on the picks' fence (its words picked on the GPU from this round's
            // result); the keep follows once the verify ran, so it runs beside the frame (Frame.early)
            if (p.early) {
                const prepared = g.draft.prepareNext(.{ .ptr = p, .encode = lead }, 2) catch |err| {
                    keep_cb.commit();
                    g.keep = keep_cb.retain();
                    return err;
                };
                if (prepared) {
                    g.keep = keep_cb.retain();
                    p.held = true;
                    return;
                }
            }
            // committed without a wait: the drafter must not read or write what the keep writes until finishKeep
            keep_cb.commit();
            g.keep = keep_cb.retain();
        }
        /// A held keep to the queue, then the early frame's bridge spin behind it.
        fn release(p: *Picks) void {
            if (!p.held) return;
            p.held = false;
            p.g.keep.?.commit();
            p.g.draft.bridgeNext();
        }
        /// First in the next round's frame: once the picks are in, the RoundArgs of the count they admitted.
        fn lead(ptr: *anyopaque, e: mtl.ComputeEncoder, blocks: @import("runtime_backend.zig").Ref) anyerror!void {
            const p: *Picks = @ptrCast(@alignCast(ptr));
            const g = p.g;
            const RoundArgs = @import("runtime_backend.zig").RoundArgs;
            const shape = p.shape orelse return error.DraftKeepGpuFailure;
            g.picked.?.wait(e);
            try g.commit.select(e, .{ .buf = g.matcher.buffers[4] }, .{ .buf = blocks.buf, .off = blocks.off }, .{ .rows = shape.rows, .words = RoundArgs.words, .blocks = RoundArgs.blocks - 1, .token = RoundArgs.token_word, .status = RoundArgs.status_word, .fallback = g.draft.graph.config.mask });
        }
    };
    /// A keep left pending (runRound), waited for right before the next verify's first GPU work (Runner.settle): once its
    /// event is seen the GPU ran its last command, and its buffer is checked once the verify ran (checkKeep), as its
    /// completion is reported only behind the frame's. One already reported, or that signals nothing, is checked now.
    fn settleKeep(ptr: *anyopaque) anyerror!void {
        const g: *Generation = @ptrCast(@alignCast(ptr));
        const cb = g.keep orelse return;
        const ev = g.keep_event orelse return g.finishKeep();
        if (g.keep_signal == 0) return g.finishKeep();
        while (ev.value() < g.keep_signal and cb.status() != .@"error") std.atomic.spinLoopHint();
        switch (cb.status()) {
            .completed, .@"error" => return g.finishKeep(),
            else => {},
        }
        try g.checkKeep();
        g.keep = null;
        g.keep_done = cb;
    }
    /// The verify with its picks encoded behind the head and its keep committed behind them: the host waits once. `early`:
    /// the kept taps go to the drafter's ring, and the next round's frame is committed behind the verify, the keep once
    /// the verify ran.
    fn verifyAccept(g: *Generation, round: *const Round, ids: []const u32, parents: []const i32, budget: u32, eos: []const u32, early: bool) !@import("core").tree_round.Decoded {
        if (g.picked == null) g.picked = try @import("core").segments.Fence.init(g.runner.model.device);
        // the picks rewrite the matcher's result, which a pending keep still reads, and the verify's metadata uploads
        // and launches rewrite what it reads: a pending keep ends right before the verify's first GPU work (settle)
        g.runner.settle = if (g.keep != null) .{ .ptr = g, .run = settleKeep } else null;
        defer g.runner.settle = null;
        // a greedy verify of at most the sieve's rows picks through it; wider ones (copies past the tree) take the full head
        const sieve = g.runner.sieve != null and ids.len <= @import("../head_sieve.zig").HeadSieve.rows_max;
        var picks = Picks{ .g = g, .ids = ids, .parents = parents, .budget = budget, .eos = eos, .round = round, .early = early, .sieve = sieve };
        g.runner.tail = .{ .ptr = &picks, .encode = Picks.encode };
        defer g.runner.tail = null;
        g.runner.head_sieve = sieve;
        defer g.runner.head_sieve = false;
        g.runner.after = .{ .ptr = &picks, .run = Picks.keep };
        defer g.runner.after = null;
        g.runner.lead = .{ .ptr = &g.draft.backend.frame, .encode = @import("runtime_frame.zig").Frame.endBridge };
        defer g.runner.lead = null;
        // a frame the keep hook committed runs on: an error waits for it (and fails the drafter if it admitted nothing)
        errdefer g.draft.drainNext() catch {};
        errdefer picks.release();
        try g.runner.verifyHead(round, .all);
        picks.release();
        errdefer g.runner.failed = true;
        // the last round's keep, seen done at this verify's settle, is checked before the verify's result is read
        try g.checkKeep();
        try g.runner.taps.complete();
        const shape = picks.shape orelse return error.DraftKeepGpuFailure;
        if (g.keep == null) return error.DraftKeepGpuFailure;
        const accepted = try g.matcher.read(shape);
        if (sieve) {
            // what keep_head would have left in row 0: the accepted path's last row, whose pick is its argmax
            if (accepted.path.len == 0) return error.EmptyAcceptedDraft;
            g.runner.head_token = g.matcher.pickAt(accepted.path[accepted.path.len - 1]).token;
            g.runner.sieve.?.after();
        }
        try g.runner.completeDeviceKeep(round, accepted.path);
        if (early) try g.draft.commitNext(@intCast(accepted.path.len)) else try g.draft.absorb(.{ .buf = g.taps }, @intCast(accepted.path.len));
        if (g.draft.committed_end != g.runner.offsets[0]) return error.BadDraftPosition;
        return accepted;
    }
    pub fn run(g: *Generation, count: usize, force_length: bool, nodes: usize) !Result {
        return g.runImpl(count, force_length, nodes, null, false);
    }
    pub fn runBatch(g: *Generation, count: usize, nodes: usize, eos: []const u32, already_delivered: bool) !Result {
        if (count == 0 or count > 16 or eos.len > 16) return error.BadDraftOptions;
        return g.runImpl(count, false, nodes, eos, already_delivered);
    }
    /// `extend`: a copy from the context shares the tree's agreeing nodes and adds up to `extra` rows past them.
    pub fn runRound(g: *Generation, remaining: u32, policy: @import("selector.zig").Policy, eos: []const u32, already_delivered: bool, extend: ?Extender, extra: u32) !Result {
        const nodes = policy.nodes;
        const block = try policy.block(g.draft.tree_block);
        if (policy.max_depth != null and (g.proposer != null or extend != null or extra != 0)) return error.BadDraftOptions;
        // the drafter's frame may be running only as the last round's early one, for this round to draft from
        if (g.runner.failed or g.runner.active != null or g.runner.offsets[0] == 0 or g.draft.backend.failed or (g.draft.backend.frame.active and g.draft.early == null) or g.draft.backend.transaction != null or g.draft.committed_end != g.runner.offsets[0] or g.draft.committed_end == 0 or nodes == 0 or nodes > 15 or eos.len > 16) return error.BadDraftOptions;
        for (eos) |token| if (token >= g.runner.model.config.vocab) return error.BadDraftOptions;
        if (remaining == 0) {
            try g.draft.drainNext();
            return .{ .tokens = try g.allocator.alloc(u32, 0), .rounds = 0, .verified = 0, .matched = 0 };
        }
        const pending = g.pending orelse try (TargetSession.Session{ .runner = g.runner }).greedy();
        if (!already_delivered) {
            // Seed delivery, not a draft round (rounds = 0): the bench seed token, never journaled, so the
            // journal's committed sum equals the free-run count that follows it.
            try g.draft.drainNext();
            // the first round's pending token is this one: nothing moves the target before it, so it need not
            // take the logits' argmax again
            g.pending = pending;
            const tokens = try g.allocator.alloc(u32, 1);
            tokens[0] = pending;
            return .{ .tokens = tokens, .rounds = 0, .verified = 0, .matched = 0 };
        }
        if (std.mem.indexOfScalar(u32, eos, pending) != null) {
            try g.draft.drainNext();
            return .{ .tokens = try g.allocator.alloc(u32, 0), .rounds = 0, .verified = 0, .matched = 0 };
        }
        const available = g.runner.capacity - g.runner.offsets[0];
        if (available == 0) {
            try g.draft.drainNext();
            return error.ContextFull;
        }
        const offered: u32 = @intCast(@min(nodes, available - 1));
        var offered_policy = policy;
        offered_policy.nodes = offered;
        if (g.journal) |journal| try journal.reserve(g.allocator, 1);
        if (g.proposer != null) try g.draft.drainNext();
        // a tree drafted from the early frame waits for it: what waits on nothing it makes goes first (ahead)
        if (g.proposer == null and g.draft.draftsNext(pending, offered_policy)) try g.ahead();
        // a first command buffer made ahead that this round's verify did not take goes to the queue with its stop
        errdefer @import("core").segments.flushFirst();
        var tree = if (g.proposer) |proposer| try proposer.call(proposer.ptr, pending, offered) else try g.draft.proposePolicy(pending, offered_policy);
        defer tree.deinit();
        try offered_policy.checkTree(tree.parents);
        const drafted = try tree.verifyParents(g.allocator);
        defer g.allocator.free(drafted);
        // a copy from the root shares the tree's nodes while they agree, and its rest hangs below the last shared one
        const room = @min(extra, @as(u32, @intCast(@import("core").tree_round.max_rows - drafted.len)), available - 1 - @as(u32, @intCast(tree.tokens.len)));
        const copy: []const u32 = if (extend) |x| (if (room > 0) try x.call(x.ptr, &.{pending}, room + @as(u32, @intCast(tree.tokens.len))) else &.{}) else &.{};
        var last: i32 = 0;
        var shared: usize = 0;
        follow: for (copy) |token| {
            for (drafted[1..], tree.tokens, 1..) |parent, drafted_token, row| if (parent == last and drafted_token == token) {
                last = @intCast(row);
                shared += 1;
                continue :follow;
            };
            break;
        }
        const copied = copy[shared..@min(copy.len, shared + room)];
        const parents = try g.allocator.alloc(i32, drafted.len + copied.len);
        defer g.allocator.free(parents);
        @memcpy(parents[0..drafted.len], drafted);
        for (parents[drafted.len..], drafted.len..) |*p, row| p.* = if (row == drafted.len) last else @intCast(row - 1);
        const ids = try g.allocator.alloc(u32, parents.len);
        defer g.allocator.free(ids);
        ids[0] = pending;
        @memcpy(ids[1..drafted.len], tree.tokens);
        @memcpy(ids[drafted.len..], copied);
        var round = try Round.init(g.allocator, &.{.{ .slot = 0, .start = g.runner.offsets[0], .capacity = g.runner.capacity, .ids = ids, .parents = parents }}, @intCast(g.runner.model.config.conv_kernel), g.runner.slots, @intCast(g.runner.model.config.vocab));
        defer round.deinit();
        // the last round's keep (behind its verify, beside this round's frame) is waited for right before this
        // verify's first commit (verifyAccept): its completion, reported only behind the frame's, overlaps the planning
        // and encode
        errdefer g.finishKeep() catch {};
        // a depth-one round of the drafter's 2-row block: the next round's frame is committed behind this verify, unless a
        // next round may not follow (its budget may end the run) or either count of kept taps would change its launches
        const early = remaining > 2 and (policy.max_depth orelse 0) == 1 and block == 2 and g.proposer == null and extend == null and extra == 0 and eos.len == 0 and g.draft.canPrepare(block);
        // a depth-one round's keep (keepDevice, behind the verify) stashes its GDN rows for the next verifies' star scans
        g.runner.defer_gdn = (policy.max_depth orelse 0) == 1;
        defer g.runner.defer_gdn = false;
        const accepted = try g.verifyAccept(&round, ids, parents, policy.budget(remaining), eos, early);
        // a depth-bounded round that takes its whole budget stops at its length with the bonus emitted: that pick is the
        // next round's pending token, as a bonus stop's is, so the next round need not take it from the logits again
        g.pending = accepted.pending orelse accepted.bonus;
        if (accepted.path.len == 0 or accepted.tokens.len == 0 or accepted.tokens.len > @min(@import("core").tree_round.max_rows, remaining)) return error.EmptyAcceptedDraft;
        g.accept_share += (@as(f32, @floatFromInt(@intFromBool(accepted.matched > 0))) - g.accept_share) * accept_step;
        g.draft.backend.precise = g.accept_share < precise_below;
        var landed: u32 = 0;
        for (accepted.path) |row| landed += @intFromBool(row >= drafted.len);
        const tokens = try g.allocator.dupe(u32, accepted.tokens);
        if (g.journal) |journal| journal.recordAssumeCapacity(.{
            .committed = @intCast(tokens.len),
            .drafted = @intCast(ids.len - 1),
            .accepted = @intCast(accepted.matched),
        });
        return .{ .tokens = tokens, .rounds = 1, .verified = ids.len, .matched = accepted.matched, .min_rows = @intCast(ids.len), .extended = @intCast(copied.len), .extension_landed = landed };
    }
    /// The host work of a round drafted from the early frame that waits on nothing the frame makes, done while the frame
    /// runs: the last verify's command buffers checked (segments.checkUnchecked, else this verify's first host work),
    /// and this verify's first command buffer and encoder made with the bridge's stop encoded first in it, as the
    /// verify's lead would encode it (Frame.endBridge; the lead then finds the bridge stopped and encodes nothing).
    fn ahead(g: *Generation) !void {
        @import("core").segments.checkUnchecked() catch |err| {
            g.runner.failed = true;
            g.draft.drainNext() catch {};
            g.finishKeep() catch {};
            return err;
        };
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        @import("core").segments.prepareFirst(g.runner.model.queue, .concurrent, .{ .ptr = &g.draft.backend.frame, .encode = @import("runtime_frame.zig").Frame.endBridge });
    }
    fn runImpl(g: *Generation, count: usize, force_length: bool, nodes: usize, eos_override: ?[]const u32, already_delivered: bool) !Result {
        g.pending = null;
        if (g.runner.failed or g.runner.active != null or g.runner.offsets[0] == 0 or g.draft.backend.failed or g.draft.backend.frame.active or g.draft.backend.transaction != null or g.draft.committed_end != g.runner.offsets[0] or g.draft.committed_end == 0 or nodes == 0 or nodes > 15 or count > 65536) return error.BadDraftOptions;
        const eos: []const u32 = if (force_length) &.{} else eos_override orelse g.runner.model.config.eos[0..g.runner.model.config.eos_count];
        for (eos) |token| if (token >= g.runner.model.config.vocab) return error.BadDraftOptions;
        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(g.allocator);
        errdefer g.finishKeep() catch {};
        var rounds: usize = 0;
        var verified: usize = 0;
        var matched: usize = 0;
        var min_rows: u32 = 0;
        if (count == 0) return .{ .tokens = try tokens.toOwnedSlice(g.allocator), .rounds = 0, .verified = 0, .matched = 0 };
        const session = TargetSession.Session{ .runner = g.runner };
        var pending = try session.greedy();
        if (!already_delivered) try tokens.append(g.allocator, pending);
        while (tokens.items.len < count and std.mem.indexOfScalar(u32, eos, pending) == null) {
            const available = g.runner.capacity - g.runner.offsets[0];
            if (available == 0) return error.ContextFull;
            const offered: u32 = @intCast(@min(nodes, available - 1));
            var tree = if (g.proposer) |proposer| try proposer.call(proposer.ptr, pending, offered) else try g.draft.propose(pending, offered);
            defer tree.deinit();
            const parents = try tree.verifyParents(g.allocator);
            defer g.allocator.free(parents);
            const ids = try g.allocator.alloc(u32, tree.tokens.len + 1);
            defer g.allocator.free(ids);
            ids[0] = pending;
            @memcpy(ids[1..], tree.tokens);
            var round = try Round.init(g.allocator, &.{.{ .slot = 0, .start = g.runner.offsets[0], .capacity = g.runner.capacity, .ids = ids, .parents = parents }}, @intCast(g.runner.model.config.conv_kernel), g.runner.slots, @intCast(g.runner.model.config.vocab));
            defer round.deinit();
            try g.finishKeep();
            const accepted = try g.verifyAccept(&round, ids, parents, @intCast(count - tokens.items.len), eos, false);
            rounds += 1;
            verified += ids.len;
            min_rows = if (min_rows == 0) @intCast(ids.len) else @min(min_rows, @as(u32, @intCast(ids.len)));
            try tokens.appendSlice(g.allocator, accepted.tokens);
            matched += accepted.matched;
            if (accepted.path.len == 0) return error.EmptyAcceptedDraft;
            if (accepted.pending) |next| pending = next else break;
        }
        try g.finishKeep();
        return .{ .tokens = try tokens.toOwnedSlice(g.allocator), .rounds = rounds, .verified = verified, .matched = matched, .min_rows = min_rows };
    }
};

test "true request budget keeps a verified non-prefix continuation across delivery boundaries" {
    const r = @import("core").tree_round;
    const ids = [_]u32{ 1, 9, 2, 9, 3, 9, 4 };
    const parents = [_]i32{ -1, 0, 0, 2, 2, 4, 4 };
    const picks = [_]r.Pick{ .{ .token = 2 }, .{ .token = 0 }, .{ .token = 3 }, .{ .token = 0 }, .{ .token = 4 }, .{ .token = 0 }, .{ .token = 5 } };
    var full = try r.oracle(.{ .rows = ids.len, .vocab = 10, .budget = 33 }, &ids, &parents, &picks, &.{});
    const accepted = try r.decode(&full, .{ .rows = ids.len, .vocab = 10, .budget = 33 });
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4, 6 }, accepted.path);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4, 5 }, accepted.tokens);
    try std.testing.expectEqual(@as(?u32, 5), accepted.pending);
    var clipped = try r.oracle(.{ .rows = ids.len, .vocab = 10, .budget = 3 }, &ids, &parents, &picks, &.{});
    const terminal = try r.decode(&clipped, .{ .rows = ids.len, .vocab = 10, .budget = 3 });
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4 }, terminal.path);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4 }, terminal.tokens);
    try std.testing.expect(terminal.pending == null);
}
