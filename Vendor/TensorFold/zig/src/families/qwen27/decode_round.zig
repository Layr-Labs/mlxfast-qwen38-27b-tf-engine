//! One command buffer verifies every layer; cache publication waits for the accepted path.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Frame = @import("gpu_frame.zig").Frame;
const forward = @import("forward.zig");
/// Layers a verify encodes before committing them, so the GPU starts while the CPU encodes the rest.
const commit_every = 16;
/// Layers a verify encodes before its first commit: the GPU (kept busy by the draft bridge) starts after one layer's
/// encode, and the host encodes the next chunk's layers while the GPU runs that one.
const first_commit = 1;
/// Prompt rows a pass needs before it runs as staggered segments on their own queues (core/stagger.zig): each later
/// segment's mixer of a layer waits for the one before it, so one segment's sequential GDN scan runs beside another's
/// projections.
const stagger_rows = 768;
/// Segments a staggered prompt pass takes by default (TF_Q27_STAGGER: 0 or 1 keeps one, up to core/stagger.MAX).
const stagger_parts = 3;
const wts = @import("gpu_weights.zig");
const attn = @import("attention.zig");
const abi = @import("gdn_contract.zig");
const GdnPool = @import("gdn_pool_gpu.zig").Pool;
const GdnHook = @import("gdn_hook.zig").Hook;
const GdnPlan = @import("gdn_plan.zig").Plan;
const gdn_pool = @import("gdn_pool.zig");
const Round = @import("round_plan.zig").Round;
const Taps = @import("taps.zig").Taps;
const image_rows = @import("image_rows.zig");
const mrope = @import("core").qwen_vl.rope;
const HeadSieve = @import("head_sieve.zig").HeadSieve;

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
    accepted_chain: bool = false,
    last_gpu_seconds: f64 = 0,
    current_round: ?*const Round = null,
    /// Encoded after the head of the next verify, in its last command buffer.
    tail: ?forward.Tail = null,
    /// Committed right after the next verify's last command buffer, then cleared by its caller.
    after: ?forward.Hook = null,
    /// Encoded first in the next verify's first command buffer, then cleared by its caller.
    lead: ?forward.Tail = null,
    /// Run once before the next verify's first GPU work (its first commit, or a deferred GDN replay it lands first),
    /// then cleared: work its caller left pending (Generation: the last round's keep), so planning and encoding overlap
    /// it.
    settle: ?forward.Hook = null,
    current_gdn: ?*const GdnPlan = null,
    current_attention: ?*const attn.Plan = null,
    /// The current pass's attention plan goes up to the scratch with its first attention layer, which reads it first.
    scratch_due: bool = false,
    terminal_head: forward.Head = .all,
    current_placed: image_rows.Placed = .{},
    /// Each slot's rows' M-RoPE layout from its stream's images (empty: text, plain rope at the cache row).
    rope: [attn.planning.max_streams]mrope.Layout = @splat(.{}),
    /// A prompt pass's image rows, while it runs.
    images: ?image_rows.Images = null,
    /// A kept state's blit still reading a slot, by slot: the slot's next write waits for it (settleSlot).
    reading: [attn.planning.max_streams]?mtl.CommandBuffer = @splat(null),
    /// The next verify's keep is keepDevice's, committed behind it (Generation's depth-one rounds): a lone depth-one star
    /// verify then defers its GDN replay (deferGdn).
    defer_gdn: bool = false,
    /// The GDN replay the last verify deferred: its scans applied the pending rows, and its keep stashes its own behind them.
    deferred: ?abi.Deferred = null,
    /// The greedy verify head's sieve (head_sieve.zig), made by the first headSieve call that fits; `sieve_tried` once made
    /// or refused.
    sieve: ?HeadSieve = null,
    sieve_tried: bool = false,
    /// The next verify's tail picks its rows' tokens through `sieve` (Generation's greedy verifies of at most 8 rows): the
    /// forward skips the full head and writes no logits row.
    head_sieve: bool = false,
    /// Set after a sieved verify: the token the argmax of row 0 of the logits would give had the full head run (the
    /// accepted path's last row's, which tree_commit's keep_head copies there). Any forward with a head clears it, as
    /// its logits replace that row; Session.greedy and fingerprint read it in place of row 0.
    head_token: ?u32 = null,
    /// A staggered prompt pass's later segments (the first runs in the model's frame on its queue): each one's rows'
    /// frame, attention scratch and bindings, GDN scratch and queue, made by the first pass that splits that far.
    second: [@import("core").segments.MAX - 1]?Second = @splat(null),
    /// The residency set the model's queue holds (residency.zig): the later segments' queues hold it as well.
    resident_set: ?mtl.ResidencySet = null,

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

    /// The greedy verify head's sieve, made on the first call (its 2-bit copy and scratch join `owned`); null where the
    /// head does not fit. A failed build leaves the full head and is not retried.
    pub fn headSieve(r: *Runner) !?*HeadSieve {
        if (!r.sieve_tried) {
            r.sieve_tried = true;
            r.sieve = try HeadSieve.init(r.allocator, r.model, &r.owned, r.residency);
        }
        return if (r.sieve) |*s| s else null;
    }

    pub fn deinit(r: *Runner) void {
        for (0..r.slots) |slot| r.settleSlot(@intCast(slot));
        r.setResident(null);
        for (&r.second) |*second| if (second.*) |*s| s.deinit(r.allocator);
        if (r.sieve) |*s| s.deinit();
        r.taps.deinit();
        r.gdn.deinit();
        r.attention.deinit();
        r.scratch.deinit();
        for (r.bindings) |*b| b.deinit();
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
        // a sieved verify writes picks, not logits: only a full head's rows (or the sieve's own token) stand for row 0
        const sieved = r.head_sieve and head == .all and round.ids.len <= HeadSieve.rows_max and r.sieve != null;
        if (r.head_sieve and !sieved) return error.RoundNotReady;
        if (head != .none) r.head_token = null;
        for (round.slots, round.windows) |slot, window| if (slot >= r.ready or r.offsets[slot] != window.start) return error.CachePositionDiffers;
        if (r.staggers(round, head, accepted)) return r.staggered(round, head);
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
        r.accepted_chain = accepted;
        defer r.accepted_chain = false;
        var recurrent = try round.recurrent(try r.params(round.ids.len), paths);
        defer recurrent.deinit();
        if (r.gdn.marks) |m| {
            if (!accepted or !r.model.kernels.prompt or r.capture_taps) return error.BadPromptMarks;
            try m.check(recurrent);
        }
        const deferred = try r.deferGdn(round, recurrent, accepted);
        var attention_plan = try r.attentionPlan(round);
        defer attention_plan.deinit();
        try r.gdn.prepare(r.model.device, recurrent, accepted);
        try r.model.frame.setRows(round.ids, round.feed_positions);
        const placed = image_rows.place(&r.model.frame, &r.rope, r.images, r.model.config.mrope_section, round);
        // no launch before the first attention layer reads the scratch: its plan goes up there, after the first commit
        try r.scratch.admit(attention_plan);
        r.current_round = round;
        r.current_gdn = &recurrent;
        r.current_attention = &attention_plan;
        r.current_placed = placed;
        r.scratch_due = true;
        defer {
            r.current_round = null;
            r.current_gdn = null;
            r.current_attention = null;
            r.current_placed = .{};
            r.scratch_due = false;
        }
        if (r.capture_taps) r.taps.begin();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const terminal_rows = accepted and r.model.kernels.prompt and head != .all and r.tail == null and r.after == null and r.lead == null and r.model.ane == null and
            (!r.capture_taps or std.mem.indexOfScalar(usize, &r.taps.ids, r.model.config.layers - 1) == null);
        r.terminal_head = if (terminal_rows) head else .all;
        defer r.terminal_head = .all;
        var stages = forward.Stages{ .terminal_rows = terminal_rows, .model = r.model, .rows = @intCast(round.ids.len), .hooks = .{ .ptr = r, .gdn = gdnForward, .attention = attentionForward, .tap = if (r.capture_taps) tap else null }, .head = head, .sieve = sieved, .tail = r.tail, .lead = r.lead, .after = r.after, .settle = .{ .ptr = r, .run = settleRound }, .images = placed.images };
        r.deferred = deferred;
        // a write-back lands every pending row: the keep stashes this round's from the first
        if (deferred) |d| if (d.persist != 0) {
            r.gdn.pending.rows = 0;
            r.gdn.pending.rounds = 0;
        };
        // a keep committed behind the forward's last command buffer (stages.after) keeps this round
        if (!accepted and r.after != null) r.active = fingerprint(round);
        // a prompt-sized pass encodes a layer in milliseconds: its GPU starts on the first mixer and is fed layer by layer
        const early = round.ids.len > 16;
        r.last_gpu_seconds = @import("core").segments.runCommittingEarly(r.model.device, &.{r.model.queue}, r.model.config.layers, .concurrent, &stages, first_commit, commit_every, early) catch |err| {
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

    /// Whether a pass runs as two staggered segments: an accepted prompt window of `stagger_rows` rows or more, text only,
    /// with none of a verify's hooks around it. TF_Q27_STAGGER=0 keeps one segment.
    fn staggers(r: *const Runner, round: *const Round, head: forward.Head, accepted: bool) bool {
        if (!accepted or !r.model.kernels.prompt or head == .all or round.windows.len != 1 or round.ids.len < stagger_rows) return false;
        if (r.tail != null or r.after != null or r.lead != null or r.model.ane != null or r.gdn.marks != null or r.images != null or r.head_sieve) return false;
        return staggerParts() > 1;
    }

    /// The segments a staggered pass takes: stagger_parts, or TF_Q27_STAGGER's count (2 to core/stagger.MAX; 0 or 1 none).
    fn staggerParts() usize {
        const v = std.c.getenv("TF_Q27_STAGGER") orelse return stagger_parts;
        const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return stagger_parts;
        return @min(n, @import("core").segments.MAX);
    }

    /// Later segment k's (1 and on) frame, scratch, bindings and queue, made once (their buffers join the runner's
    /// residency).
    fn laterSegment(r: *Runner, k: usize) !*Second {
        const at = &r.second[k - 1];
        if (at.* == null) {
            at.* = try Second.init(r.allocator, r.model, &r.gdn, r.scratch.limits, r.bindings.len, r.model.frame.capacity / 2 + 64);
            if (r.residency) |f| at.*.?.follow(f, true);
            if (r.resident_set) |set| at.*.?.queue.addResidencySet(set);
            std.log.info("qwen27: staggered prompt passes: segment {d} of up to {d} rows on its own queue", .{ k, at.*.?.frame.capacity });
        }
        return &at.*.?;
    }

    /// A prompt window as two segments (core/stagger.zig): the first `na` rows (a multiple of 64) in the model's frame on
    /// the model's queue, the rest in the second segment's frame on its own queue. Each layer's mixer of the second waits
    /// for the first's (its GDN states and conv windows, its K/V rows); everything else of the two runs side by side. Each
    /// segment's rows take the kernels and the arithmetic a prompt chunk of those rows takes, so every word is the one two
    /// chunks run in order give.
    fn staggered(r: *Runner, round: *const Round, head: forward.Head) !void {
        const max = @import("core").segments.MAX;
        const n: u32 = @intCast(round.ids.len);
        const parts = staggerParts();
        const slot = round.slots[0];
        const start = round.windows[0].start;
        for (round.windows[0].parents, 0..) |parent, row| if (parent != (if (row == 0) @as(i32, -1) else @as(i32, @intCast(row - 1)))) return error.NotAcceptedChain;
        // segment k's first row: k n / parts, down to a multiple of 64; the last runs to n
        var firsts: [max + 1]u32 = undefined;
        for (0..parts) |k| firsts[k] = @intCast(k * n / parts / 64 * 64);
        firsts[parts] = n;
        for (0..parts) |k| if (firsts[k + 1] <= firsts[k]) return error.BadFrameRows;
        var frames: [max]*Frame = undefined;
        var scratches: [max]*attn.Scratch = undefined;
        var bindings: [max][]attn.Binding = undefined;
        var owns: [max]?@import("gdn_pool_gpu.zig").Own = @splat(null);
        var queues: [max]mtl.Queue = undefined;
        frames[0] = &r.model.frame;
        scratches[0] = &r.scratch;
        bindings[0] = r.bindings;
        queues[0] = r.model.queue;
        for (1..parts) |k| {
            const later = try r.laterSegment(k);
            if (firsts[k + 1] - firsts[k] > later.frame.capacity) return error.BadFrameRows;
            frames[k] = &later.frame;
            scratches[k] = &later.scratch;
            bindings[k] = later.bindings;
            owns[k] = later.gdn;
            queues[k] = later.queue;
        }
        const taps: u32 = @intCast(r.model.config.conv_kernel);
        const vocab: u32 = @intCast(r.model.config.vocab);
        var rounds: [max]Round = undefined;
        var made_rounds: usize = 0;
        defer for (rounds[0..made_rounds]) |*x| x.deinit();
        for (0..parts) |k| {
            rounds[k] = try Round.init(r.allocator, &.{.{ .slot = slot, .start = start + firsts[k], .capacity = r.capacity, .ids = round.ids[firsts[k]..firsts[k + 1]] }}, taps, r.slots, vocab);
            made_rounds += 1;
        }
        var chain: [@import("gpu_frame.zig").max_rows]u32 = undefined;
        for (chain[0..n], 0..) |*v, i| v.* = @intCast(i);
        r.accepted_chain = true;
        defer r.accepted_chain = false;
        var plans: [max]GdnPlan = undefined;
        var made_plans: usize = 0;
        defer for (plans[0..made_plans]) |*x| x.deinit();
        for (0..parts) |k| {
            const rows = firsts[k + 1] - firsts[k];
            plans[k] = try rounds[k].recurrent(try r.params(rows), &.{chain[0..rows]});
            made_plans += 1;
        }
        // an accepted chain defers nothing: the caller's pending work and any pending GDN rows land first
        if (try r.deferGdn(&rounds[0], plans[0], true) != null) return error.RoundNotReady;
        r.deferred = null;
        var attention_plans: [max]attn.Plan = undefined;
        var made_attention: usize = 0;
        defer for (attention_plans[0..made_attention]) |*x| x.deinit();
        for (0..parts) |k| {
            attention_plans[k] = try r.attentionPlan(&rounds[k]);
            made_attention += 1;
        }
        for (0..parts) |k| try r.gdn.prepare(r.model.device, plans[k], true);
        var placed: [max]image_rows.Placed = undefined;
        for (0..parts) |k| {
            try frames[k].setRows(rounds[k].ids, rounds[k].feed_positions);
            placed[k] = image_rows.place(frames[k], &r.rope, null, r.model.config.mrope_section, &rounds[k]);
            try scratches[k].admit(attention_plans[k]);
        }
        if (r.capture_taps) r.taps.begin();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        // as a prompt chunk's: the last block keeps only the rows a head or a tap reads
        const terminal_rows = !r.capture_taps or std.mem.indexOfScalar(usize, &r.taps.ids, r.model.config.layers - 1) == null;
        var due: [max]bool = @splat(true);
        var segments: [max]Segment = undefined;
        var stages: [max]forward.Stages = undefined;
        var family = Stagger{};
        for (0..parts) |k| {
            const last = k + 1 == parts;
            const segment_head: forward.Head = if (last) head else .none;
            segments[k] = .{ .r = r, .gdn = &plans[k], .own = owns[k], .attention = .{ .plan = &attention_plans[k], .round = &rounds[k], .placed = placed[k], .scratch = scratches[k], .bindings = bindings[k], .scratch_due = &due[k], .terminal_head = if (terminal_rows) segment_head else .all }, .tap_row = firsts[k], .total = n };
            stages[k] = .{ .terminal_rows = terminal_rows, .model = r.model, .rows = firsts[k + 1] - firsts[k], .hooks = segments[k].mixer(), .head = segment_head, .frame = frames[k], .lane_k = k, .logits = if (k == 0) null else r.model.frame.get(.logits), .settle = if (k == 0) .{ .ptr = r, .run = settleRound } else null };
            family.stages[k] = &stages[k];
        }
        r.last_gpu_seconds = @import("core").segments.runCommittingEarly(r.model.device, queues[0..parts], r.model.config.layers, .concurrent, &family, first_commit, commit_every, true) catch |err| {
            r.failed = true;
            r.active = null;
            return err;
        };
        if (r.capture_taps) try r.taps.complete();
        r.offsets[slot] = start + n;
    }

    pub fn keep(r: *Runner, round: *const Round, paths: []const []const u32) !void {
        const active = r.active orelse return error.RoundNotReady;
        if (r.failed or !std.mem.eql(u8, &active, &fingerprint(round))) return error.RoundNotReady;
        // the replays below read the committed states: rows still pending land there first
        try r.settleGdn();
        r.deferred = null;
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
        // a chain's kept path is a prefix of its rows, which already stand at their positions: the attention keep would
        // copy every kept row onto itself
        const moves = !chainWindow(round.windows[0].parents);
        // the keep map depends only on the verify result, never on the layer: the first attention binding's map is
        // written once, beside the GDN keep and rows, and every layer's gather and scatter read that one copy
        var shared_map: ?@import("core").tree_commit_gpu.Ref = null;
        if (moves) {
            var ai: usize = 0;
            for (r.model.weights.layers) |layer| switch (layer.mixer) {
                .linear => {},
                .attention => {
                    var views: [attn.planning.max_streams]attn.Cache = undefined;
                    const caches = r.layerCaches(ai, round, &views);
                    try r.bindings[ai].admitDeviceKeep(attention_plan, caches, p.rows);
                    ai += 1;
                },
            };
            if (ai > 0) {
                const map = r.bindings[0].get(.keep_map);
                shared_map = .{ .buf = map.buffer, .off = map.offset };
            }
        }
        try ops.metadata(e, result, keep_ref, rows, shared_map, p);
        e.barrier();
        const plan = @import("gdn_pool_gpu.zig").DevicePlan{ .host = recurrent, .metadata = m };
        if (r.deferred != null) {
            // a deferred replay: every layer's kept rows go behind the pending ones in one launch, for the next star scans,
            // and every layer's conv window lands in the same one launch over the pooled spans
            try r.gdn.stash(plan, e);
            try r.gdn.commitConvKeepLayers(plan, e);
        } else {
            if (r.gdn.pending.rows != 0) return error.RoundNotReady; // a verify that defers nothing lands them first
            // Each layer replays its state and copies all conv taps before overwriting that channel's history.
            for (0..r.gdn.budget.layers) |gi| try r.gdn.commitKeep(gi, plan, e);
        }
        // The attention moves touch only their caches, maps and lanes: they run beside the GDN commits.
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
        try r.attention.keepGatherLayers(e, r.bindings[0..ai], p.rows, r.bindings[0].get(.keep_map), table[0..ai], lanes, words);
        e.barrier();
        r.attention.keepScatterLayers(e, g, p.rows, table[0..ai], lanes, words);
        e.barrier();
    }

    /// The caller's pending work (settle), at most once a verify.
    fn settleCaller(r: *Runner) !void {
        const h = r.settle orelse return;
        r.settle = null;
        try h.run(h.ptr);
    }

    /// Right before a verify's first commit: the caller's pending work lands (settle); a tree verify's GDN metadata
    /// then goes up once, and every layer's scan reads that copy in place (gdn_pool_gpu.Pool.forwardAfter, `uploaded`).
    fn settleRound(ptr: *anyopaque) anyerror!void {
        const r: *Runner = @ptrCast(@alignCast(ptr));
        try r.settleCaller();
        if (!r.accepted_chain) try r.metadata().upload((r.current_gdn orelse return error.RoundNotReady).*);
    }

    /// Whether every row's parent is the row before it (a chain), so any kept path is a prefix of the rows.
    fn chainWindow(parents: []const i32) bool {
        for (parents, 0..) |parent, row| if (parent != @as(i32, @intCast(row)) - 1) return false;
        return true;
    }

    pub fn completeDeviceKeep(r: *Runner, round: *const Round, path: []const u32) !void {
        const active = r.active orelse return error.RoundNotReady;
        if (r.failed or round.windows.len != 1 or path.len == 0 or !std.mem.eql(u8, &active, &fingerprint(round))) return error.RoundNotReady;
        if (r.deferred != null) {
            // the keep stashed the path's rows behind the pending ones, at most the rows left there (Pool.stash)
            const pending = &r.gdn.pending;
            if (pending.rows + path.len > gdn_pool.pending_rows) {
                r.failed = true;
                return error.BadGdnPending;
            }
            pending.rows += @intCast(path.len);
            pending.rounds += 1;
            r.deferred = null;
        }
        r.offsets[round.slots[0]] = (try round.windows[0].keep(path)).length;
        r.active = null;
    }

    /// After a GPU failure, once no stream holds a slot: rounds run again, each slot reset by its next prompt.
    pub fn recover(r: *Runner) void {
        r.failed = false;
        r.active = null;
        r.deferred = null;
        r.gdn.pending = .{};
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

    /// The GDN replay a verify defers: a lone depth-one star verify whose keep is keepDevice's (defer_gdn) applies the rows
    /// earlier keeps stashed in its scans, and every `defer_rounds`-th writes the state they leave back; any other verify
    /// reads the committed states, so those rows land there first.
    fn deferGdn(r: *Runner, round: *const Round, plan: GdnPlan, accepted: bool) !?abi.Deferred {
        const pending = &r.gdn.pending;
        const star = !accepted and r.defer_gdn and round.windows.len == 1 and r.gdn.starPlan(plan);
        if (!star or (pending.rows != 0 and pending.slot != round.slots[0])) {
            // the caller's pending work (a keep that may stash pending rows) lands before a replay reads them
            try r.settleCaller();
            try r.settleGdn();
        }
        if (!star) return null;
        pending.slot = round.slots[0];
        // the K-th round since the last write-back writes back, or one whose keep might not fit behind the pending rows
        const persist = pending.rows != 0 and (pending.rounds >= gdn_pool.defer_rounds or pending.rows + gdn_pool.star_kept > gdn_pool.pending_rows);
        return .{ .rows = pending.rows, .persist = @intFromBool(persist) };
    }

    /// The rows earlier keeps deferred replayed into their slot's committed states, waited for: before anything but a
    /// deferred star scan reads those states (another verify, a host keep, a snapshot, a fingerprint).
    pub fn settleGdn(r: *Runner) !void {
        if (r.gdn.pending.rows == 0) return;
        if (r.failed) return error.RoundNotReady;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        r.gdn.unstash(e) catch |err| {
            e.end();
            return err;
        };
        e.end();
        cb.commit();
        cb.wait();
        if (gpuFailed(cb, "a deferred GDN replay")) {
            r.failed = true;
            return error.RoundGpuFailure;
        }
        r.gdn.pending.rows = 0;
        r.gdn.pending.rounds = 0;
    }

    pub fn reset(r: *Runner, slot: u32) !void {
        r.settleSlot(slot);
        if (r.failed or r.active != null or slot >= r.ready) return error.RoundNotReady;
        if (r.gdn.pending.slot == slot) r.gdn.pending = .{}; // rows deferred for the slot's states, which clear below
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

    fn gdnForward(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Gdn, frame: *Frame, rows: u32) !void {
        const r: *Runner = @ptrCast(@alignCast(ptr));
        var hook = GdnHook{ .pool = &r.gdn, .quant = &r.model.kernels, .glue = r.model.glue, .plan = r.current_gdn orelse return error.RoundNotReady, .linear_index = r.linear_index, .accepted_chain = r.accepted_chain, .deferred = r.deferred, .uploaded = !r.accepted_chain };
        try GdnHook.forward(&hook, e, block, weights, frame, rows);
    }

    fn attentionForward(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Attention, f: *Frame, rows: u32) !void {
        const r: *Runner = @ptrCast(@alignCast(ptr));
        const pass = AttentionPass{ .plan = r.current_attention orelse return error.RoundNotReady, .round = r.current_round orelse return error.RoundNotReady, .placed = r.current_placed, .scratch = &r.scratch, .bindings = r.bindings, .scratch_due = &r.scratch_due, .terminal_head = r.terminal_head };
        return r.attentionPass(pass, e, block, weights, f, rows);
    }

    /// What an attention layer of a pass reads and uploads into: the pass's plan and round, its image placement, the
    /// scratch and per-layer bindings its plan goes up to, and its terminal head (a staggered prompt's segments each
    /// have their own).
    const AttentionPass = struct {
        plan: *const attn.Plan,
        round: *const Round,
        placed: image_rows.Placed,
        scratch: *attn.Scratch,
        bindings: []attn.Binding,
        scratch_due: *bool,
        terminal_head: forward.Head,
    };

    fn attentionPass(r: *Runner, pass: AttentionPass, e: mtl.ComputeEncoder, block: usize, weights: wts.Attention, f: *Frame, rows: u32) !void {
        const plan = pass.plan.*;
        if (pass.scratch_due.*) {
            try pass.scratch.upload(plan);
            pass.scratch_due.* = false;
        }
        const layer = r.attention_index[block];
        var views: [attn.planning.max_streams]attn.Cache = undefined;
        const caches = r.layerCaches(layer, pass.round, &views);
        try pass.bindings[layer].upload(plan, caches);
        // the terminal block's other queries have no reader: their stale rows only reach the rows post drops
        const dead = if (pass.terminal_head != .all and block + 1 == r.model.config.layers and r.attention.promptPath(plan, r.model.kernels.prompt)) (if (pass.terminal_head == .last) rows - 1 else rows) else 0;
        // decoded verify rows: q+gate and k|v count their threadgroups past a device fence and the preprocess that reads them
        // waits on that count in the same barrier interval, as the record's norms wait on their projections; both are counted
        // or neither, so the shared counter always returns to zero
        var after: ?attn.After = null;
        const promptPlan = r.attention.promptPath(plan, r.model.kernels.prompt);
        if (!promptPlan and dead == 0 and r.model.kernels.countable(weights.q, rows) and r.model.kernels.countable(weights.kv, rows)) {
            const a = (try r.model.kernels.quantCounted(e, weights.q, f.get(.input), forward.sums(f, .sums, rows), f.get(.q_gate), rows)) orelse return error.RoundNotReady;
            const b = (try r.model.kernels.quantCounted(e, weights.kv, f.get(.input), forward.sums(f, .sums, rows), f.get(.mixed), rows)) orelse return error.RoundNotReady;
            after = .{ .counter = r.model.kernels.count_words.?, .wait = a + b };
        } else {
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
        }
        // keys and values read in place from the [key | value] rows
        const kv: u32 = @intCast(r.model.config.kvDim());
        const mixed = f.get(.mixed);
        var value = mixed;
        value.offset += @as(usize, kv) * 2;
        // a prompt plan's query goes straight to the prompt kernel's packed rows
        const prepacked = r.attention.promptPath(plan, r.model.kernels.prompt);
        // a decoded fused chain (one stream, at most 8 rows, counted projections): the preprocess also packs the prefix's
        // query and appends the round's keys and values, and the prefix waits on its count in the same barrier interval
        const appended = after != null and !prepacked and plan.fused and plan.streams == 1 and plan.prefix_chunks > 0 and rows <= 8 and r.attention.chain_words != null;
        const decoded: ?attn.Decoded = if (appended) .{ .scratch = pass.scratch, .binding = &pass.bindings[layer], .plan = &plan, .caches = caches, .value = value, .v_stride = 2 * kv } else null;
        try r.attention.preprocess(e, .{ .qg = f.get(.q_gate), .key = mixed, .key_stride = 2 * kv, .q_gain = wts.ref(weights.q_norm), .k_gain = wts.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = if (prepacked) pass.scratch.get(.qa) else f.get(.query), .packed_rows = if (prepacked) plan.tiles * 16 else 0, .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta, .h_end = pass.placed.h_end, .w_end = pass.placed.w_end, .plane = f.capacity, .geometry = r.geometry, .after = after, .decoded = decoded });
        // Both prompt and decoded merges write post-gate group sums into `.sums`; forward.post consumes them.
        const out_sums = true;
        const xs = forward.sums(f, .sums, rows);
        var live_plan = plan;
        live_plan.live_from = dead; // the terminal block's dead rows: nothing reads their attention output
        try r.attention.encode(e, pass.scratch, &pass.bindings[layer], live_plan, f.get(.query), f.get(.rotated_key), value, 2 * kv, f.get(.mixer_out), caches, f.get(.q_gate), r.model.kernels.prompt, prepacked, if (out_sums) .{ .ref = xs.ref, .stride = xs.stride } else null, appended);
        f.mixer_sums = out_sums;
    }

    fn tap(ptr: *anyopaque, e: mtl.ComputeEncoder, layer: usize, hidden: @import("projection.zig").Ref, rows: u32) !void {
        const r: *Runner = @ptrCast(@alignCast(ptr));
        try r.taps.record(r.model.glue, e, layer, hidden, rows);
    }
};

/// A staggered prompt's second segment's own state: its rows' frame (activations and GDN metadata), the attention scratch
/// and per-layer bindings its plan goes up to at encode time (the first segment's stay untouched until its kernels have
/// read them), and its queue.
const Second = struct {
    frame: Frame,
    scratch: attn.Scratch,
    bindings: []attn.Binding,
    queue: mtl.Queue,
    gdn: @import("gdn_pool_gpu.zig").Own,

    fn init(a: std.mem.Allocator, m: *Model, pool: *const GdnPool, limits: attn.Limits, layers: usize, rows: u32) !Second {
        var frame = try Frame.init(m.device, m.config, 16 * ((rows + 15) / 16));
        errdefer frame.deinit();
        var scratch = try attn.Scratch.init(m.device, limits);
        errdefer scratch.deinit();
        const bindings = try a.alloc(attn.Binding, layers);
        var made: usize = 0;
        errdefer {
            for (bindings[0..made]) |*b| b.deinit();
            a.free(bindings);
        }
        for (bindings) |*b| {
            b.* = try attn.Binding.init(m.device, limits);
            made += 1;
        }
        const own = try @import("gdn_pool_gpu.zig").Own.init(pool, m.device);
        errdefer own.deinit();
        const queue = try m.device.queue();
        return .{ .frame = frame, .scratch = scratch, .bindings = bindings, .queue = queue, .gdn = own };
    }

    pub fn follow(s: *Second, hook: mtl.ResidencySet.Follow, joining: bool) void {
        const fields = mtl.resident.fields;
        for (s.frame.buffers) |b| fields.one(b, hook, joining);
        if (s.frame.standby) |bank| for (bank) |b| fields.one(b, hook, joining);
        fields.apply(&s.scratch, hook, joining);
        for (s.bindings) |*b| fields.apply(b, hook, joining);
        for ([_]mtl.Buffer{ s.gdn.tails, s.gdn.recurrent, s.gdn.snapshots }) |b| fields.one(b, hook, joining);
    }

    fn deinit(s: *Second, a: std.mem.Allocator) void {
        s.frame.deinit();
        s.scratch.deinit();
        for (s.bindings) |*b| b.deinit();
        a.free(s.bindings);
        s.gdn.deinit();
        s.queue.deinit();
    }
};

/// One segment of a staggered prompt pass, as its stages' mixer hooks see it: its GDN plan, its attention pass, and
/// where its rows go in the taps.
const Segment = struct {
    r: *Runner,
    gdn: *const GdnPlan,
    /// The second segment's own every-layer GDN scratch (null: the pool's).
    own: ?@import("gdn_pool_gpu.zig").Own = null,
    attention: Runner.AttentionPass,
    tap_row: u32,
    total: u32,

    fn mixer(s: *Segment) forward.Mixer {
        return .{ .ptr = s, .gdn = gdnHook, .attention = attentionHook, .tap = if (s.r.capture_taps) tapHook else null };
    }
    fn gdnHook(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Gdn, frame: *Frame, rows: u32) !void {
        const s: *Segment = @ptrCast(@alignCast(ptr));
        var hook = GdnHook{ .pool = &s.r.gdn, .quant = &s.r.model.kernels, .glue = s.r.model.glue, .plan = s.gdn, .linear_index = s.r.linear_index, .accepted_chain = true, .deferred = null, .uploaded = false, .own = s.own };
        try GdnHook.forward(&hook, e, block, weights, frame, rows);
    }
    fn attentionHook(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Attention, f: *Frame, rows: u32) !void {
        const s: *Segment = @ptrCast(@alignCast(ptr));
        return s.r.attentionPass(s.attention, e, block, weights, f, rows);
    }
    fn tapHook(ptr: *anyopaque, e: mtl.ComputeEncoder, layer: usize, hidden: @import("projection.zig").Ref, rows: u32) !void {
        const s: *Segment = @ptrCast(@alignCast(ptr));
        try s.r.taps.recordAt(s.r.model.glue, e, layer, hidden, rows, s.tap_row, s.total);
    }
};

/// A staggered prompt pass's segments as one family for core/segments.zig: each hook goes to its lane's stages.
const Stagger = struct {
    stages: [@import("core").segments.MAX]*forward.Stages = undefined,
    const Lane = @import("core").segments.Lane;

    pub fn begin(p: *Stagger, lane: *Lane) !void {
        return p.stages[lane.k].begin(lane);
    }
    pub fn wait(_: *Stagger, _: usize) @import("core").segments.Wait {
        return .mixer;
    }
    pub fn handoff(_: *Stagger, _: *Lane, _: usize) !void {}
    pub fn pre(p: *Stagger, lane: *Lane, index: usize) !void {
        return p.stages[lane.k].pre(lane, index);
    }
    pub fn mixer(p: *Stagger, lane: *Lane, index: usize) !void {
        return p.stages[lane.k].mixer(lane, index);
    }
    pub fn post(p: *Stagger, lane: *Lane, index: usize) !void {
        return p.stages[lane.k].post(lane, index);
    }
    pub fn finish(p: *Stagger, lane: *Lane) !void {
        return p.stages[lane.k].finish(lane);
    }
    pub fn beforeCommit(p: *Stagger) !void {
        return p.stages[0].beforeCommit();
    }
};

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