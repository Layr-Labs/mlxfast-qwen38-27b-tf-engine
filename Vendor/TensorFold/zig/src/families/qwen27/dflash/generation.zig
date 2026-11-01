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
pub const Generation = struct {
    allocator: std.mem.Allocator,
    runner: *Runner,
    draft: *Draft,
    kept: mtl.Buffer,
    /// Row ids 0..n for a whole prompt pass's tap gather into the drafter's ring (absorbChunk), made on first use.
    kept_all: ?mtl.Buffer = null,
    taps: mtl.Buffer,
    proposer: ?Proposer = null,
    matcher: @import("round_gpu.zig").Matcher,
    commit: @import("core").tree_commit_gpu.Ops,
    keep: ?mtl.CommandBuffer = null, // the last round's cache and state commit, running behind the drafter
    picked: ?@import("core").segments.Fence = null, // updated behind a verify's picks; its keep waits on it
    pending: ?u32 = null, // runRound's next token as the last accept picked it; anything else that moves the target clears it
    journal: ?*AuditJournal = null,
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
        return .{ .allocator = a, .runner = runner, .draft = draft, .kept = kept, .taps = taps, .matcher = matcher, .commit = commit };
    }
    pub fn deinit(g: Generation) void {
        if (g.keep) |cb| {
            cb.wait();
            cb.release();
        }
        g.commit.deinit();
        if (g.picked) |f| f.deinit();
        g.matcher.deinit();
        g.kept.deinit();
        if (g.kept_all) |b| b.deinit();
        g.taps.deinit();
    }
    fn absorb(g: *Generation, path: []const u32) !void {
        try g.absorbRows(path);
        if (g.draft.committed_end != g.runner.offsets[0]) return error.BadDraftPosition;
    }
    /// A prompt pass's rows into the drafter, 128 a call (its frame's); the target's pass may be longer.
    fn absorbChunk(g: *Generation, rows: usize) !void {
        var path: [128]u32 = undefined;
        var first: usize = 0;
        // The whole pass in one gather, written straight into the drafter's ring at its slots when they do not wrap.
        {
            const r = g.draft.ring();
            const slot: usize = @intCast(r.end.* % r.window);
            if (g.keep == null and rows <= g.runner.taps.rows and rows <= r.window - slot) {
                if (g.kept_all == null) g.kept_all = try g.runner.model.device.buffer(@as(usize, g.runner.taps.capacity) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                const ids = g.kept_all.?.slice(u32, rows);
                for (ids, 0..) |*index, i| index.* = @intCast(i);
                const cb = g.runner.model.queue.commandBuffer();
                const e = cb.compute(.serial);
                g.runner.taps.accepted(g.runner.model.glue, e, ids, .{ .buffer = g.kept_all.? }, .{ .buffer = r.buf, .offset = slot * r.stride }) catch |err| {
                    e.end();
                    cb.commit();
                    cb.wait();
                    return err;
                };
                e.end();
                cb.commit();
                cb.wait();
                if (gpuFailed(cb, "a draft taps gather")) return error.DraftTapGpuFailure;
                r.end.* += rows;
                if (g.draft.committed_end != g.runner.offsets[0]) return error.BadDraftPosition;
                return;
            }
        }
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
        // the prompt's last pass: while its GPU work runs, the drafter's frame scratch for the first round is remade
        if (last) g.runner.idle = .{ .ptr = g.draft, .run = regrowDraftFrame };
        defer g.runner.idle = null;
        try (TargetSession.Session{ .runner = g.runner }).promptChunk(ids, last);
        try g.absorbChunk(ids.len);
    }
    /// The drafter frame's remembered scratch (Frame.regrow): host allocations only, no GPU work and no content.
    fn regrowDraftFrame(ptr: *anyopaque) anyerror!void {
        const draft: *Draft = @ptrCast(@alignCast(ptr));
        draft.backend.frame.regrow();
    }
    /// `promptChunk` with decoded rows' arithmetic (an earlier reply's tokens).
    pub fn decodeChunk(g: *Generation, ids: []const u32, last: bool) !void {
        g.pending = null;
        std.debug.assert(g.keep == null);
        if (g.runner.failed or g.draft.backend.failed or g.draft.committed_end != g.runner.offsets[0] or ids.len == 0 or ids.len > g.runner.model.frame.capacity) return error.BadDraftContext;
        try (TargetSession.Session{ .runner = g.runner }).decodeChunk(ids, last);
        try g.absorbChunk(ids.len);
    }
    pub fn prefill(g: *Generation, ids: []const u32, chunk: usize) !void {
        g.pending = null;
        std.debug.assert(g.keep == null);
        if (g.runner.failed or g.draft.backend.failed or g.draft.backend.frame.active or g.draft.backend.transaction != null or g.draft.committed_end != 0 or g.runner.offsets[0] != 0 or chunk == 0 or chunk > g.runner.model.frame.capacity) return error.BadDraftContext;
        var first: usize = 0;
        while (first < ids.len) {
            const count = @min(ids.len - first, g.runner.promptRows(@intCast(chunk)));
            try g.promptChunk(ids[first..][0..count], first + count == ids.len);
            first += count;
        }
    }
    /// Ends a pending keep and forgets the picked token, before anything but runRound uses the slot.
    pub fn settle(g: *Generation) !void {
        g.pending = null;
        return g.finishKeep();
    }
    /// A round's keep runs behind the drafter; it ends before the next verify and before a run returns.
    fn finishKeep(g: *Generation) !void {
        const cb = g.keep orelse return;
        g.keep = null;
        defer cb.release();
        cb.wait();
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
        fn encode(ptr: *anyopaque, e: mtl.ComputeEncoder) anyerror!void {
            const p: *Picks = @ptrCast(@alignCast(ptr));
            const g = p.g;
            // the matcher and keep fence their own dependencies; taps and head read only the matcher's result
            const logits = g.runner.model.frame.get(.logits);
            const width: u32 = @intCast(g.runner.model.config.vocab);
            const shape = try g.matcher.encode(e, .{ .buf = logits.buffer, .off = logits.offset }, p.ids, p.parents, width, p.budget, p.eos);
            const result = @import("core").tree_commit_gpu.Ref{ .buf = g.matcher.buffers[4] };
            try g.commit.taps(e, result, .{ .buf = g.runner.taps.buffer }, .{ .buf = g.taps }, .{ .rows = shape.rows, .width = g.runner.taps.width, .capacity = g.runner.taps.capacity, .planes = 5 });
            try g.commit.head(e, result, .{ .buf = logits.buffer, .off = logits.offset }, .{ .rows = shape.rows, .width = width, .stride = width });
            p.shape = shape;
            g.picked.?.update(e);
        }
        /// The keep, encoded and committed behind the verify's last command buffer while the GPU still runs it,
        /// so its replay starts on the picks' fence instead of after the host wakes, reads and encodes.
        fn keep(ptr: *anyopaque) anyerror!void {
            const p: *Picks = @ptrCast(@alignCast(ptr));
            const g = p.g;
            const result = @import("core").tree_commit_gpu.Ref{ .buf = g.matcher.buffers[4] };
            const keep_cb = g.runner.model.queue.commandBuffer();
            const k = keep_cb.compute(.concurrent);
            g.picked.?.wait(k);
            var kept = false;
            errdefer if (!kept) k.end();
            try g.runner.keepDevice(p.round, k, g.commit, result);
            k.end();
            kept = true;
            // committed without a wait: the drafter must not read or write what the keep writes until finishKeep
            keep_cb.commit();
            g.keep = keep_cb.retain();
        }
    };
    /// The verify with its picks encoded behind the head and its keep committed behind them: the host waits once.
    fn verifyAccept(g: *Generation, round: *const Round, ids: []const u32, parents: []const i32, budget: u32, eos: []const u32) !@import("core").tree_round.Decoded {
        // the picks rewrite the matcher's result, which a pending keep still reads
        std.debug.assert(g.keep == null);
        if (g.picked == null) g.picked = try @import("core").segments.Fence.init(g.runner.model.device);
        var picks = Picks{ .g = g, .ids = ids, .parents = parents, .budget = budget, .eos = eos, .round = round };
        g.runner.tail = .{ .ptr = &picks, .encode = Picks.encode };
        defer g.runner.tail = null;
        g.runner.after = .{ .ptr = &picks, .run = Picks.keep };
        defer g.runner.after = null;
        g.runner.lead = .{ .ptr = &g.draft.backend.frame, .encode = @import("runtime_frame.zig").Frame.endBridge };
        defer g.runner.lead = null;
        try g.runner.verifyHead(round, .all);
        errdefer g.runner.failed = true;
        try g.runner.taps.complete();
        const shape = picks.shape orelse return error.DraftKeepGpuFailure;
        if (g.keep == null) return error.DraftKeepGpuFailure;
        const accepted = try g.matcher.read(shape);
        try g.runner.completeDeviceKeep(round, accepted.path);
        try g.draft.absorb(.{ .buf = g.taps }, @intCast(accepted.path.len));
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
        _ = try policy.block(g.draft.tree_block);
        if (policy.max_depth != null and (g.proposer != null or extend != null or extra != 0)) return error.BadDraftOptions;
        if (g.runner.failed or g.runner.active != null or g.runner.offsets[0] == 0 or g.draft.backend.failed or g.draft.backend.frame.active or g.draft.backend.transaction != null or g.draft.committed_end != g.runner.offsets[0] or g.draft.committed_end == 0 or nodes == 0 or nodes > 15 or eos.len > 16) return error.BadDraftOptions;
        for (eos) |token| if (token >= g.runner.model.config.vocab) return error.BadDraftOptions;
        if (remaining == 0) return .{ .tokens = try g.allocator.alloc(u32, 0), .rounds = 0, .verified = 0, .matched = 0 };
        const pending = g.pending orelse try (TargetSession.Session{ .runner = g.runner }).greedy();
        if (!already_delivered) {
            // Seed delivery, not a draft round (rounds = 0): the bench seed token, never journaled, so the
            // journal's committed sum equals the free-run count that follows it.
            const tokens = try g.allocator.alloc(u32, 1);
            tokens[0] = pending;
            return .{ .tokens = tokens, .rounds = 0, .verified = 0, .matched = 0 };
        }
        if (std.mem.indexOfScalar(u32, eos, pending) != null) return .{ .tokens = try g.allocator.alloc(u32, 0), .rounds = 0, .verified = 0, .matched = 0 };
        const available = g.runner.capacity - g.runner.offsets[0];
        if (available == 0) return error.ContextFull;
        const offered: u32 = @intCast(@min(nodes, available - 1));
        var offered_policy = policy;
        offered_policy.nodes = offered;
        if (g.journal) |journal| try journal.reserve(g.allocator, 1);
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
        errdefer g.finishKeep() catch {};
        try g.finishKeep();
        const accepted = try g.verifyAccept(&round, ids, parents, policy.budget(remaining), eos);
        // a depth-bounded round that takes its whole budget stops at its length with the bonus emitted: that pick is the
        // next round's pending token, as a bonus stop's is, so the next round need not take it from the logits again
        g.pending = accepted.pending orelse accepted.bonus;
        if (accepted.path.len == 0 or accepted.tokens.len == 0 or accepted.tokens.len > @min(@import("core").tree_round.max_rows, remaining)) return error.EmptyAcceptedDraft;
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
            const accepted = try g.verifyAccept(&round, ids, parents, @intCast(count - tokens.items.len), eos);
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
