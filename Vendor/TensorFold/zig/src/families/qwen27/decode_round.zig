//! One command buffer verifies every layer; cache publication waits for the accepted path.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Frame = @import("gpu_frame.zig").Frame;
const View = @import("gpu_frame.zig").View;
const segments = @import("core").segments;
const forward = @import("forward.zig");
/// Layers a verify encodes before committing them, so the GPU starts while the CPU encodes the rest.
const commit_every = 8;
/// Layers a verify encodes before its first commit: the idle GPU starts after two layers' encode.
const first_commit = 2;
const wts = @import("gpu_weights.zig");
const attn = @import("attention.zig");
const abi = @import("gdn_contract.zig");
const GdnPool = @import("gdn_pool_gpu.zig").Pool;
const GdnHook = @import("gdn_hook.zig").Hook;
const GdnPlan = @import("gdn_plan.zig").Plan;
const Round = @import("round_plan.zig").Round;
const Taps = @import("taps.zig").Taps;
const image_rows = @import("image_rows.zig");
const mrope = @import("core").qwen_vl.rope;

/// Whether `cb` failed on the GPU; its Metal error (out of memory, a timeout) is logged with `what` it ran.
pub fn gpuFailed(cb: mtl.CommandBuffer, what: []const u8) bool {
    const msg = cb.failure() orelse return false;
    std.log.err("qwen27: {s} failed on the GPU: {s}", .{ what, msg });
    return true;
}

pub const Runner = struct {
    allocator: std.mem.Allocator,
    residency: ?mtl.ResidencySet.Follow = null,
    model: *Model,
    geometry: attn.planning.Geometry = .{},
    gdn: GdnPool,
    attention: attn.Attention,
    scratch: attn.Scratch,
    bindings: []attn.Binding,
    caches: []attn.Cache,
    linear_index: []u32,
    attention_index: []u32,
    offsets: []u32,
    owned: std.ArrayList(mtl.Buffer),
    taps: Taps,
    slots: u32,
    ready: u32, // slots [0, ready) have their K/V rows; a later slot takes its own when first used (ensure)
    capacity: u32,
    active: ?[32]u8 = null,
    failed: bool = false,
    capture_taps: bool = false,
    /// Each attention layer's own keep K/V lanes, made at the first moving keep: the 16 gathers then run side by side.
    keep_lanes: ?mtl.Buffer = null,
    last_gpu_seconds: f64 = 0,
    /// Encoded after the head of the next verify, in its last command buffer.
    tail: ?forward.Tail = null,
    /// Committed right after the next verify's last command buffer, then cleared by its caller.
    after: ?forward.Hook = null,
    /// Encoded first in the next verify's first command buffer, then cleared by its caller.
    lead: ?forward.Tail = null,
    /// A split prompt chunk's second segment's own attention bindings and dims (32 bytes a segment), made at its first.
    side_bindings: ?[]attn.Binding = null,
    side_dims: ?mtl.Buffer = null,
    /// Each slot's rows' M-RoPE layout from its stream's images (empty: text, plain rope at the cache row).
    rope: [attn.planning.max_streams]mrope.Layout = @splat(.{}),
    /// A prompt pass's image rows, while it runs.
    images: ?image_rows.Images = null,
    /// A kept state's blit still reading a slot, by slot: the slot's next write waits for it (settleSlot).
    reading: [attn.planning.max_streams]?mtl.CommandBuffer = @splat(null),

    pub fn init(a: std.mem.Allocator, m: *Model, slots: u32, capacity: u32) !Runner {
        return initReady(a, m, slots, capacity, slots);
    }

    /// A runner of `slots` whose first `ready` take their K/V rows now, the rest when first used (ensure).
    pub fn initReady(a: std.mem.Allocator, m: *Model, slots: u32, capacity: u32, ready: u32) !Runner {
        if (slots == 0 or slots > attn.planning.max_streams or capacity == 0 or m.frame.capacity % 16 != 0 or ready == 0 or ready > slots) return error.BadRoundCapacity;
        const geometry = try attn.planning.Geometry.fromConfig(m.config, m.device.tensorUnits());
        const linear_index = try a.alloc(u32, m.config.layers);
        errdefer a.free(linear_index);
        const attention_index = try a.alloc(u32, m.config.layers);
        errdefer a.free(attention_index);
        var ng: u32 = 0;
        var na: u32 = 0;
        for (linear_index, attention_index, 0..) |*g, *v, block| {
            g.* = std.math.maxInt(u32);
            v.* = std.math.maxInt(u32);
            if (m.config.kind(block) == .linear) {
                g.* = ng;
                ng += 1;
            } else {
                v.* = na;
                na += 1;
            }
        }
        var pool = try GdnPool.init(m.device, try abi.Shape.init(m.config), ng, slots, m.frame.capacity);
        errdefer pool.deinit();
        const units = m.device.tensorUnits();
        // before M5 the tail attention reads its tensor-op results in the simdgroup-matrix layout (simd_attention.zig)
        if (!units) try @import("simd_attention.zig").check(m.device, m.queue);
        const text = try @import("simd_attention.zig").library(a, units, true);
        defer a.free(text);
        // before M5 the prompt kernels are optional: a chip that cannot build them keeps the decode path for prompts
        const library = mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx()) catch |err| blk: {
            if (units) return err;
            std.log.info("qwen27: the prompt attention kernels do not build here ({t}); prompt rows take the decode path", .{err});
            const base = try @import("simd_attention.zig").library(a, units, false);
            defer a.free(base);
            break :blk try mtl.Library.fromSource(m.device, base, mtl.CompileOptions.mlx());
        };
        defer library.deinit();
        var attention = try attn.Attention.init(m.device, library);
        errdefer attention.deinit();
        const limits = attn.Limits{ .rows = m.frame.capacity, .streams = slots, .keys = capacity, .shared_prefix = geometry.shared_prefix, .geometry = geometry };
        var scratch = try attn.Scratch.init(m.device, limits);
        errdefer scratch.deinit();
        const bindings = try a.alloc(attn.Binding, na);
        var made: usize = 0;
        errdefer {
            for (bindings[0..made]) |*b| b.deinit();
            a.free(bindings);
        }
        for (bindings) |*b| {
            b.* = try attn.Binding.init(m.device, limits);
            made += 1;
        }
        const caches = try a.alloc(attn.Cache, na * slots);
        errdefer a.free(caches);
        var owned: std.ArrayList(mtl.Buffer) = .empty;
        errdefer {
            for (owned.items) |b| b.deinit();
            owned.deinit(a);
        }
        for (0..na) |layer| for (0..ready) |slot| {
            caches[layer * slots + slot] = try slotCache(a, m, &owned, capacity);
        };
        const offsets = try a.alloc(u32, slots);
        errdefer a.free(offsets);
        @memset(offsets, 0);
        const taps = try Taps.init(m.device, m.frame.capacity, @intCast(m.config.hidden));
        return .{ .allocator = a, .model = m, .geometry = geometry, .gdn = pool, .attention = attention, .scratch = scratch, .bindings = bindings, .caches = caches, .linear_index = linear_index, .attention_index = attention_index, .offsets = offsets, .owned = owned, .taps = taps, .slots = slots, .ready = ready, .capacity = capacity };
    }

    /// One slot's K/V rows across the attention layers.
    pub fn slotBytes(r: *const Runner) usize {
        return r.caches.len / r.slots * 2 * @as(usize, r.capacity) * r.model.config.kvDim() * 2;
    }

    /// Slot `slot`'s K/V rows (the next unready slot, as free slots are taken lowest first); its new buffers.
    pub fn ensure(r: *Runner, slot: u32) ![]const mtl.Buffer {
        if (slot < r.ready) return &.{};
        if (slot != r.ready) return error.SlotOutOfOrder;
        const first = r.owned.items.len;
        errdefer {
            for (r.owned.items[first..]) |b| b.deinit();
            r.owned.shrinkRetainingCapacity(first);
        }
        const layers = r.caches.len / r.slots;
        for (0..layers) |layer| r.caches[layer * r.slots + slot] = try slotCache(r.allocator, r.model, &r.owned, r.capacity);
        r.ready += 1;
        if (r.residency) |f| for (r.owned.items[first..]) |b| f.join(f.ctx, b);
        return r.owned.items[first..];
    }

    /// The K/V buffers of the ready slots (for a residency set).
    pub fn readyBuffers(r: *const Runner, a: std.mem.Allocator, out: *std.ArrayList(mtl.Buffer)) !void {
        for (0..r.caches.len / r.slots) |layer| for (0..r.ready) |slot| {
            const c = r.caches[layer * r.slots + slot];
            try out.appendSlice(a, &.{ c.keys.buffer, c.values.buffer });
        };
    }

    pub fn setResident(r: *Runner, hook: ?mtl.ResidencySet.Follow) void {
        @import("residency.zig").setRunner(r, hook);
    }

    pub fn deinit(r: *Runner) void {
        for (0..r.slots) |slot| r.settleSlot(@intCast(slot));
        r.setResident(null);
        r.taps.deinit();
        r.gdn.deinit();
        r.attention.deinit();
        r.scratch.deinit();
        for (r.bindings) |*b| b.deinit();
        if (r.side_bindings) |side| {
            for (side) |*b| b.deinit();
            r.allocator.free(side);
        }
        if (r.side_dims) |b| b.deinit();
        for (r.owned.items) |b| b.deinit();
        r.owned.deinit(r.allocator);
        r.allocator.free(r.bindings);
        r.allocator.free(r.caches);
        r.allocator.free(r.linear_index);
        r.allocator.free(r.attention_index);
        r.allocator.free(r.offsets);
    }

    pub fn profile(r: *Runner, trace: ?*@import("core").gpu_profile.Trace) void {
        r.model.kernels.profiler = trace;
        r.model.glue.profiler = trace;
        r.gdn.profiler = trace;
        r.attention.profiler = trace;
    }

    pub fn verify(r: *Runner, round: *const Round) !void {
        return r.verifyHead(round, .all);
    }

    /// Prompt chunks may skip nonfinal logits or project just their last row, using the identical lane math.
    pub fn verifyHead(r: *Runner, round: *const Round, head: forward.Head) !void {
        return r.verifyImpl(round, head, false);
    }

    /// Ordinary decode and prompt: every chain row is kept, no snapshots or replay; a GPU failure ends the runner.
    pub fn advance(r: *Runner, round: *const Round, head: forward.Head) !void {
        return r.verifyImpl(round, head, true);
    }

    fn verifyImpl(r: *Runner, round: *const Round, head: forward.Head, accepted: bool) !void {
        if (r.failed or r.active != null or round.ids.len > r.model.frame.capacity) return error.RoundNotReady;
        for (round.slots, round.windows) |slot, window| if (slot >= r.ready or r.offsets[slot] != window.start) return error.CachePositionDiffers;
        const paths = try r.allocator.alloc([]const u32, round.windows.len);
        defer r.allocator.free(paths);
        @memset(paths, &.{});
        var accepted_rows: [@import("gpu_frame.zig").max_rows]u32 = undefined;
        if (accepted) for (round.windows, round.firsts, paths) |window, first, *path| {
            for (window.parents, 0..) |parent, row| {
                if (parent != (if (row == 0) @as(i32, -1) else @as(i32, @intCast(row - 1)))) return error.NotAcceptedChain;
                accepted_rows[first + row] = @intCast(row);
            }
            path.* = accepted_rows[first..][0..window.parents.len];
        };
        var recurrent = try round.recurrent(try r.params(round.ids.len), paths);
        defer recurrent.deinit();
        if (r.gdn.marks) |m| {
            if (!accepted or !r.model.kernels.prompt or r.capture_taps) return error.BadPromptMarks;
            try m.check(recurrent);
        }
        var attention_plan = try r.attentionPlan(round);
        defer attention_plan.deinit();
        try r.gdn.prepare(r.model.device, recurrent, accepted);
        try r.model.frame.setRows(round.ids, round.feed_positions);
        const placed = image_rows.place(&r.model.frame, &r.rope, r.images, r.model.config.mrope_section, round);
        if (r.segmentRows(round, head, accepted, placed)) |rows| return r.advanceSegments(round, head, rows);
        var region = attn.Region{ .scratch = &r.scratch };
        try region.upload(attention_plan);
        var view = Frame.whole(&r.model.frame, @intCast(round.ids.len));
        if (r.capture_taps) try r.taps.begin(view.rows, 1);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const terminal_rows = r.terminalRows(head, accepted);
        var pass = Pass{ .runner = r, .round = round, .gdn = &recurrent, .attention = &attention_plan, .view = &view, .region = &region, .bindings = r.bindings, .placed = placed, .terminal_head = if (terminal_rows) head else .all, .accepted_chain = accepted };
        var stages = forward.Stages{ .terminal_rows = terminal_rows, .model = r.model, .view = &view, .rows = view.rows, .hooks = pass.hooks(), .head = head, .tail = r.tail, .lead = r.lead, .after = r.after, .images = placed.images };
        // a keep committed behind the forward's last command buffer (stages.after) keeps this round
        if (!accepted and r.after != null) r.active = fingerprint(round);
        r.last_gpu_seconds = segments.runCommittingFrom(r.model.device, &.{r.model.queue}, r.model.config.layers, .concurrent, &stages, first_commit, commit_every) catch |err| {
            r.failed = true;
            r.active = null;
            return err;
        };
        if (r.model.ane) |share| share.drain() catch |err| {
            r.failed = true;
            return err;
        };
        if (r.capture_taps) try r.taps.complete();
        if (accepted) {
            for (round.slots, round.windows) |slot, window| r.offsets[slot] = window.start + @as(u32, @intCast(window.parents.len));
        } else r.active = fingerprint(round);
    }

    pub fn keep(r: *Runner, round: *const Round, paths: []const []const u32) !void {
        const active = r.active orelse return error.RoundNotReady;
        if (r.failed or !std.mem.eql(u8, &active, &fingerprint(round))) return error.RoundNotReady;
        var recurrent = try round.recurrent(try r.params(round.ids.len), paths);
        defer recurrent.deinit();
        var attention_plan = try r.attentionPlan(round);
        defer attention_plan.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        var gi: usize = 0;
        var ai: usize = 0;
        for (r.model.weights.layers) |layer| switch (layer.mixer) {
            .linear => {
                try r.gdn.commit(gi, .{ .host = recurrent, .metadata = r.metadata() }, e);
                gi += 1;
            },
            .attention => {
                var views: [attn.planning.max_streams]attn.Cache = undefined;
                const caches = r.layerCaches(ai, round, &views);
                const kept = try r.bindings[ai].uploadKeep(attention_plan, caches, paths);
                try r.attention.keep(e, &r.scratch, &r.bindings[ai], kept, caches);
                ai += 1;
            },
        };
        e.end();
        cb.commit();
        cb.wait();
        if (gpuFailed(cb, "a round's keep")) {
            r.failed = true;
            return error.RoundGpuFailure;
        }
        for (round.slots, round.windows, paths) |slot, window, path| r.offsets[slot] = (try window.keep(path)).length;
        r.active = null;
    }

    /// A validated single-window round consumes authoritative GPU keep metadata without host path selection.
    pub fn keepDevice(r: *Runner, round: *const Round, e: mtl.ComputeEncoder, ops: @import("core").tree_commit_gpu.Ops, result: @import("core").tree_commit_gpu.Ref) !void {
        const active = r.active orelse return error.RoundNotReady;
        if (r.failed or round.windows.len != 1 or round.ids.len > @import("core").tree_round.max_rows or !std.mem.eql(u8, &active, &fingerprint(round))) return error.RoundNotReady;
        var recurrent = try round.recurrent(try r.params(round.ids.len), &.{&.{}});
        defer recurrent.deinit();
        var attention_plan = try r.attentionPlan(round);
        defer attention_plan.deinit();
        const m = r.metadata();
        try m.upload(recurrent);
        const p = @import("core").tree_commit_gpu.Meta{ .rows = @intCast(round.ids.len), .slot = round.slots[0] };
        const keep_ref = @import("core").tree_commit_gpu.Ref{ .buf = m.keeps.buffer, .off = m.keeps.offset };
        const rows = @import("core").tree_commit_gpu.Ref{ .buf = m.kept_rows.buffer, .off = m.kept_rows.offset };
        try ops.metadata(e, result, keep_ref, rows, null, p);
        // a chain's kept path is a prefix of its rows, which already stand at their positions: the attention keep would
        // copy every kept row onto itself
        const moves = !chainWindow(round.windows[0].parents);
        // every attention layer's keep map beside the GDN metadata, under its one barrier: each map is its own binding's
        // buffer, and the shared keep/rows words they rewrite hold the same values the first dispatch writes
        if (moves) {
            var ai: usize = 0;
            for (r.model.weights.layers) |layer| switch (layer.mixer) {
                .linear => {},
                .attention => {
                    var views: [attn.planning.max_streams]attn.Cache = undefined;
                    const caches = r.layerCaches(ai, round, &views);
                    try r.bindings[ai].admitDeviceKeep(attention_plan, caches, p.rows);
                    const map = r.bindings[ai].get(.keep_map);
                    try ops.metadata(e, result, keep_ref, rows, .{ .buf = map.buffer, .off = map.offset }, p);
                    ai += 1;
                },
            };
        }
        e.barrier();
        const plan = @import("gdn_pool_gpu.zig").DevicePlan{ .host = recurrent, .metadata = m };
        for (0..r.gdn.budget.layers) |gi| try r.gdn.commitScan(gi, plan, e);
        e.barrier();
        for (0..r.gdn.budget.layers) |gi| try r.gdn.commitPublish(gi, plan, e);
        // the attention moves touch only their caches, maps and lanes: they run beside the GDN publish
        if (!moves) return e.barrier();
        const g = r.scratch.limits.geometry;
        const lane = @as(usize, r.scratch.limits.rows) * g.kv_heads * g.dim * 2;
        const lanes = r.keep_lanes orelse made: {
            const buffer = try r.model.device.buffer(lane * 2 * r.bindings.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            r.owned.append(r.allocator, buffer) catch |err| {
                buffer.deinit();
                return err;
            };
            r.keep_lanes = buffer;
            break :made buffer;
        };
        e.useResource(lanes, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        // every layer gathers into its own lanes in one launch under one barrier, then every layer scatters in one more
        var ai: usize = 0;
        for (r.model.weights.layers) |layer| switch (layer.mixer) {
            .linear => {},
            .attention => {
                var views: [attn.planning.max_streams]attn.Cache = undefined;
                const caches = r.layerCaches(ai, round, &views);
                try attn.Attention.keepUse(e, &r.bindings[ai], p.rows, caches);
                ai += 1;
            },
        };
        var table: [attn.max_keep_layers]attn.KeepLayer = undefined;
        if (ai > table.len) return error.AttentionKeep;
        const words: u32 = @intCast(lane / 2);
        try r.attention.keepGatherLayers(e, r.bindings[0..ai], p.rows, table[0..ai], lanes, words);
        e.barrier();
        r.attention.keepScatterLayers(e, g, p.rows, table[0..ai], lanes, words);
        e.barrier();
    }

    /// Whether every row's parent is the row before it (a chain), so any kept path is a prefix of the rows.
    fn chainWindow(parents: []const i32) bool {
        for (parents, 0..) |parent, row| if (parent != @as(i32, @intCast(row)) - 1) return false;
        return true;
    }

    pub fn completeDeviceKeep(r: *Runner, round: *const Round, path: []const u32) !void {
        const active = r.active orelse return error.RoundNotReady;
        if (r.failed or round.windows.len != 1 or path.len == 0 or !std.mem.eql(u8, &active, &fingerprint(round))) return error.RoundNotReady;
        r.offsets[round.slots[0]] = (try round.windows[0].keep(path)).length;
        r.active = null;
    }

    /// After a GPU failure, once no stream holds a slot: rounds run again, each slot reset by its next prompt.
    pub fn recover(r: *Runner) void {
        r.failed = false;
        r.active = null;
    }

    /// An idle or newly lone host may drop scratch only after the previous verify has been kept.
    pub fn trimBefore(r: *Runner) void {
        if (r.active == null) r.gdn.releaseBefore();
    }

    /// A kept state's blit reading `slot` lands before the slot is written: its reset, next chunk or first round.
    pub fn settleSlot(r: *Runner, slot: u32) void {
        const cb = r.reading[slot] orelse return;
        r.reading[slot] = null;
        cb.wait();
        cb.release();
    }

    /// `cb` reads `slot` (committed, not waited): the slot's next write settles it first.
    pub fn readingSlot(r: *Runner, slot: u32, cb: mtl.CommandBuffer) void {
        r.settleSlot(slot);
        r.reading[slot] = cb.retain();
    }

    pub fn reset(r: *Runner, slot: u32) !void {
        r.settleSlot(slot);
        if (r.failed or r.active != null or slot >= r.ready) return error.RoundNotReady;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try r.gdn.clearSlot(slot, e);
        e.end();
        cb.commit();
        cb.wait();
        if (gpuFailed(cb, "a slot reset")) {
            r.failed = true;
            return error.RoundGpuFailure;
        }
        r.offsets[slot] = 0;
    }

    fn params(r: *Runner, rows: usize) !abi.Params {
        return r.gdn.budget.shape.params(rows, r.slots, .{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 });
    }

    fn metadata(r: *Runner) @import("gdn_pool_gpu.zig").Metadata {
        const f = &r.model.frame;
        return .{ .parents = f.get(.parents), .windows = f.get(.windows), .row_slots = f.get(.row_slots), .segments = f.get(.segments), .keeps = f.get(.keeps), .kept_rows = f.get(.kept_rows), .slot_steps = f.get(.slot_steps) };
    }

    pub fn promptRows(r: *const Runner, requested: u32) u32 {
        return r.geometry.promptRows(@min(requested, r.model.frame.capacity));
    }

    fn attentionPlan(r: *Runner, round: *const Round) !attn.Plan {
        var windows: [attn.planning.max_streams]attn.planning.Window = undefined;
        if (round.windows.len > windows.len) return error.AttentionStreams;
        for (round.windows, 0..) |window, i| windows[i] = .{ .parents = window.parents, .start = window.start, .capacity = r.capacity };
        return attn.Plan.init(r.allocator, windows[0..round.windows.len], r.geometry);
    }

    fn layerCaches(r: *Runner, layer: usize, round: *const Round, views: *[attn.planning.max_streams]attn.Cache) []const attn.Cache {
        for (round.slots, 0..) |slot, i| views[i] = r.caches[layer * r.slots + slot];
        return views[0..round.slots.len];
    }

    /// Whether a pass of `head` needs nothing of its final block's other rows past its cache updates.
    fn terminalRows(r: *const Runner, head: forward.Head, accepted: bool) bool {
        return accepted and r.model.kernels.prompt and head != .all and r.tail == null and r.after == null and r.lead == null and r.model.ane == null and
            (!r.capture_taps or std.mem.indexOfScalar(usize, &r.taps.ids, r.model.config.layers - 1) == null);
    }

    /// The first segment's rows when a prompt chunk runs as two staggered segments (advanceSegments), else null: an
    /// accepted chain of prompt rows on tensor units, 512 rows or more, with nothing that wants one pass (marks, the
    /// Neural Engine's share, image rows or M-RoPE planes, a tail, lead or after hook, every row's logits).
    fn segmentRows(r: *const Runner, round: *const Round, head: forward.Head, accepted: bool, placed: image_rows.Placed) ?u32 {
        const rows = round.ids.len;
        if (!accepted or !r.model.kernels.prompt or !r.model.device.tensorUnits() or rows < 512 or round.windows.len != 1 or head == .all) return null;
        if (r.gdn.marks != null or r.model.ane != null or r.images != null or placed.images != null or placed.h_end != 0 or placed.w_end != 0 or r.tail != null or r.after != null or r.lead != null) return null;
        return @intCast(rows / 2 / 128 * 128); // the first segment's projections take whole 128-row blocks
    }

    /// The chunk as two staggered segments on two queues (core/segments.zig), each the serial chunk of its rows: the
    /// second waits at every layer's mixer for the first's states and keys, its projections run beside the first's
    /// scans, and the bits are the serial chunks'.
    fn advanceSegments(r: *Runner, round: *const Round, head: forward.Head, first_rows: u32) !void {
        const side = try r.sideLane();
        const rows: u32 = @intCast(round.ids.len);
        var segs: [2]Seg = undefined;
        var made: usize = 0;
        defer for (segs[0..made]) |*s| s.deinit();
        for (&segs, 0..) |*s, k| {
            const start: u32 = if (k == 0) 0 else first_rows;
            const tiles: u32 = if (k == 0) 0 else segs[0].attention.tiles;
            s.* = try r.segment(round, @intCast(k), start, if (k == 0) first_rows else rows - first_rows, if (k == 0) .none else head, .{ .buffer = side.dims, .offset = 32 * k }, tiles);
            made += 1;
        }
        if (r.capture_taps) try r.taps.begin(rows, 2);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        for (&segs) |*s| s.wire(r, if (s.view.part == 0) r.bindings else side.bindings);
        var hooks = Segmented{ .segs = &segs };
        r.last_gpu_seconds = segments.runCommittingFrom(r.model.device, &.{ r.model.queue, side.queue }, r.model.config.layers, .concurrent, &hooks, first_commit, commit_every) catch |err| {
            r.failed = true;
            return err;
        };
        if (r.capture_taps) try r.taps.complete();
        r.offsets[round.slots[0]] = round.windows[0].start + rows;
    }

    /// Segment `k` of `round`: rows [start, start + rows) as the serial chunk at that place, its plans uploaded to its
    /// own regions, `tiles` tiles of the segments before it in the attention scratch.
    fn segment(r: *Runner, round: *const Round, k: u32, start: u32, rows: u32, head: forward.Head, dims: @import("projection.zig").Ref, tiles: u32) !Seg {
        const window = round.windows[0];
        var own = try Round.init(r.allocator, &.{.{ .slot = round.slots[0], .start = window.start + start, .capacity = r.capacity, .ids = round.ids[start..][0..rows] }}, @intCast(r.model.config.conv_kernel), r.slots, @intCast(r.model.config.vocab));
        errdefer own.deinit();
        var path: [@import("gpu_frame.zig").max_rows]u32 = undefined;
        for (path[0..rows], 0..) |*p, i| p.* = @intCast(i);
        var gdn = try own.recurrent(try r.params(rows), &.{path[0..rows]});
        errdefer gdn.deinit();
        try r.gdn.prepare(r.model.device, gdn, true);
        var attention = try r.attentionPlan(&own);
        errdefer attention.deinit();
        var region = attn.Region{ .scratch = &r.scratch, .rows = start, .tiles = tiles };
        try region.upload(attention);
        const view = View.init(&r.model.frame, start, rows, dims, 16 * ((start + 15) / 16), k);
        view.setDims();
        return .{ .round = own, .gdn = gdn, .attention = attention, .view = view, .region = region, .head = head };
    }

    /// The second lane's own queue, attention bindings and dims, made at the first split chunk.
    fn sideLane(r: *Runner) !Side {
        if (r.side_bindings == null) {
            const bindings = try r.allocator.alloc(attn.Binding, r.bindings.len);
            var made: usize = 0;
            errdefer {
                for (bindings[0..made]) |*b| b.deinit();
                r.allocator.free(bindings);
            }
            for (bindings) |*b| {
                b.* = try attn.Binding.init(r.model.device, r.scratch.limits);
                made += 1;
            }
            const dims = try r.model.device.buffer(64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            if (r.residency) |f| {
                for (bindings) |*b| mtl.resident.fields.apply(b, f, true);
                f.join(f.ctx, dims);
            }
            r.side_bindings = bindings;
            r.side_dims = dims;
        }
        return .{ .queue = try r.model.sideQueue(), .bindings = r.side_bindings.?, .dims = r.side_dims.? };
    }
};

/// What one pass's mixer hooks read: its rows' round and plans, its view of the frame, its attention scratch region
/// and bindings. A round is one pass; a split prompt chunk is one a segment.
const Pass = struct {
    runner: *Runner,
    round: *const Round,
    gdn: *const GdnPlan,
    attention: *const attn.Plan,
    view: *View,
    region: *attn.Region,
    bindings: []attn.Binding,
    placed: image_rows.Placed = .{},
    terminal_head: forward.Head = .all,
    accepted_chain: bool = false,

    fn hooks(p: *Pass) forward.Mixer {
        return .{ .ptr = p, .gdn = gdnForward, .attention = attentionForward, .tap = if (p.runner.capture_taps) tap else null };
    }
};

/// The second lane's own queue, bindings and dims (the first lane's are the runner's).
const Side = struct { queue: mtl.Queue, bindings: []attn.Binding, dims: mtl.Buffer };

/// One segment of a split prompt chunk: the serial chunk of its rows, on a lane of its own.
const Seg = struct {
    round: Round,
    gdn: GdnPlan,
    attention: attn.Plan,
    view: View,
    region: attn.Region,
    head: forward.Head,
    pass: Pass = undefined,
    stages: forward.Stages = undefined,

    fn deinit(s: *Seg) void {
        s.attention.deinit();
        s.gdn.deinit();
        s.round.deinit();
    }

    /// The pass and stages over this segment, once it stands where it stays.
    fn wire(s: *Seg, r: *Runner, bindings: []attn.Binding) void {
        const terminal_rows = r.terminalRows(s.head, true);
        s.pass = .{ .runner = r, .round = &s.round, .gdn = &s.gdn, .attention = &s.attention, .view = &s.view, .region = &s.region, .bindings = bindings, .terminal_head = if (terminal_rows) s.head else .all, .accepted_chain = true };
        s.stages = .{ .terminal_rows = terminal_rows, .model = r.model, .view = &s.view, .rows = s.view.rows, .hooks = s.pass.hooks(), .head = s.head, .own_sums = true };
    }
};

/// A split chunk's hooks (core/segments.zig): lane k runs segment k's stages, waiting at the mixer for the lane before.
const Segmented = struct {
    segs: []Seg,

    pub fn begin(h: *Segmented, lane: *segments.Lane) !void {
        return h.segs[lane.k].stages.begin(lane);
    }
    pub fn wait(_: *Segmented, _: usize) segments.Wait {
        return .mixer;
    }
    pub fn handoff(_: *Segmented, _: *segments.Lane, _: usize) !void {}
    pub fn pre(h: *Segmented, lane: *segments.Lane, index: usize) !void {
        return h.segs[lane.k].stages.pre(lane, index);
    }
    pub fn mixer(h: *Segmented, lane: *segments.Lane, index: usize) !void {
        return h.segs[lane.k].stages.mixer(lane, index);
    }
    pub fn post(h: *Segmented, lane: *segments.Lane, index: usize) !void {
        return h.segs[lane.k].stages.post(lane, index);
    }
    pub fn finish(h: *Segmented, lane: *segments.Lane) !void {
        return h.segs[lane.k].stages.finish(lane);
    }
    pub fn polls(_: *const Segmented) bool {
        return false;
    }
    pub fn committed(_: *Segmented) !void {}
};

fn gdnForward(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Gdn, view: *View, rows: u32) !void {
    const p: *Pass = @ptrCast(@alignCast(ptr));
    const r = p.runner;
    var hook = GdnHook{ .pool = &r.gdn, .quant = &r.model.kernels, .glue = r.model.glue, .plan = p.gdn, .linear_index = r.linear_index, .accepted_chain = p.accepted_chain, .row = view.start };
    try GdnHook.forward(&hook, e, block, weights, view, rows);
}

fn attentionForward(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Attention, f: *View, rows: u32) !void {
    const p: *Pass = @ptrCast(@alignCast(ptr));
    const r = p.runner;
    const plan = p.attention.*;
    const layer = r.attention_index[block];
    var views: [attn.planning.max_streams]attn.Cache = undefined;
    const caches = r.layerCaches(layer, p.round, &views);
    try p.bindings[layer].upload(plan, caches);
    // the terminal block's other queries have no reader: their stale rows only reach the rows post drops
    const dead = if (p.terminal_head != .all and block + 1 == r.model.config.layers and r.attention.promptPath(plan, r.model.kernels.prompt)) (if (p.terminal_head == .last) rows - 1 else rows) else 0;
    if (dead < rows) {
        var input = f.get(.input);
        input.offset += @as(usize, dead) * weights.q.k * 2;
        var query_sums = forward.sums(f, .sums, rows);
        query_sums.ref.offset += @as(usize, dead) * 4;
        var query = f.get(.q_gate);
        query.offset += @as(usize, dead) * weights.q.n * 2;
        try r.model.kernels.quant(e, weights.q, input, query_sums, f.get(.dims), query, rows - dead);
    }
    try r.model.kernels.quant(e, weights.kv, f.get(.input), forward.sums(f, .sums, rows), f.get(.dims), f.get(.mixed), rows);
    e.barrier();
    // keys and values read in place from the [key | value] rows
    const kv: u32 = @intCast(r.model.config.kvDim());
    const mixed = f.get(.mixed);
    var value = mixed;
    value.offset += @as(usize, kv) * 2;
    // a prompt plan's query goes straight to the prompt kernel's packed rows
    const prepacked = r.attention.promptPath(plan, r.model.kernels.prompt);
    try r.attention.preprocess(e, .{ .qg = f.get(.q_gate), .key = mixed, .key_stride = 2 * kv, .q_gain = wts.ref(weights.q_norm), .k_gain = wts.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = if (prepacked) p.region.get(.qa) else f.get(.query), .packed_rows = if (prepacked) plan.tiles * 16 else 0, .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta, .h_end = p.placed.h_end, .w_end = p.placed.w_end, .plane = f.frame.capacity, .geometry = r.geometry });
    // decoded rows: the merge writes the out projection's group sums into `.sums` (read and cleared by forward.post)
    const out_sums = !prepacked and !r.attention.promptPath(plan, r.model.kernels.prompt);
    const xs = forward.sums(f, .sums, rows);
    try r.attention.encode(e, p.region, &p.bindings[layer], plan, f.get(.query), f.get(.rotated_key), value, 2 * kv, f.get(.mixer_out), caches, f.get(.q_gate), r.model.kernels.prompt, prepacked, if (out_sums) .{ .ref = xs.ref, .stride = xs.stride } else null);
    f.mixer_sums = out_sums;
}

fn tap(ptr: *anyopaque, e: mtl.ComputeEncoder, layer: usize, hidden: @import("projection.zig").Ref, rows: u32) !void {
    const p: *Pass = @ptrCast(@alignCast(ptr));
    try p.runner.taps.record(p.runner.model.glue, e, layer, hidden, p.view.start, rows, @intCast(p.view.part));
}

fn slotCache(a: std.mem.Allocator, m: *Model, owned: *std.ArrayList(mtl.Buffer), capacity: u32) !attn.Cache {
    const bytes = @as(usize, capacity) * m.config.kvDim() * 2;
    const key = try allocation(a, m.device, owned, bytes);
    const value = try allocation(a, m.device, owned, bytes);
    return .{ .keys = .{ .buffer = key }, .values = .{ .buffer = value }, .capacity = capacity };
}

fn allocation(a: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), bytes: usize) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    owned.append(a, b) catch |err| {
        b.deinit();
        return err;
    };
    return b;
}

fn fingerprint(round: *const Round) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(round.ids));
    hash.update(std.mem.sliceAsBytes(round.slots));
    hash.update(std.mem.sliceAsBytes(round.feed_positions));
    hash.update(std.mem.sliceAsBytes(round.draw_positions));
    hash.update(std.mem.sliceAsBytes(round.firsts));
    for (round.windows) |window| {
        hash.update(std.mem.asBytes(&window.start));
        hash.update(std.mem.asBytes(&window.taps));
        hash.update(std.mem.sliceAsBytes(window.parents));
        hash.update(std.mem.sliceAsBytes(window.conv));
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}
