//! Per-layer logs outlive verification; a single layer scratch slab is reused by caller-ordered encoders.
const std = @import("std");
const metal = @import("metal");
const abi = @import("gdn_contract.zig");
const gdn = @import("gdn.zig");
const Plan = @import("gdn_plan.zig").Plan;
const Budget = @import("gdn_pool.zig").Budget;
const pending_rows = @import("gdn_pool.zig").pending_rows;
const Ref = @import("projection.zig").Ref;
const Tensor = @import("gpu_weights.zig").Tensor;
const profile = @import("core").gpu_profile;
const Kernels = @import("gdn_library.zig").Kernels;

pub const PromptMarks = @import("gdn_marks_gpu.zig").Bank;

const Field = enum { state, history, next_history, qkv, query, key, value, decay, mixing, snapshots, tails, recurrent, pending_key, pending_value, pending_decay, pending_mixing };
/// The pending rows' fields, as tf_delta_gdn_stash, tf_delta_gdn_star_pending and tf_delta_gdn_unstash bind them.
const pending_fields = [_]Field{ .pending_key, .pending_value, .pending_decay, .pending_mixing };

pub const Metadata = struct {
    parents: Ref,
    windows: Ref,
    row_slots: Ref,
    segments: Ref,
    keeps: Ref,
    kept_rows: Ref,
    slot_steps: Ref,

    pub fn upload(m: Metadata, plan: Plan) !void {
        inline for (.{ "parents", "windows", "row_slots", "segments", "keeps", "kept_rows", "slot_steps" }) |name| {
            const r = @field(m, name);
            const bytes = std.mem.sliceAsBytes(@field(plan, name));
            if (r.offset > r.buffer.length() or bytes.len > r.buffer.length() - r.offset) return error.ShortGdnMetadata;
        }
        inline for (.{ "parents", "windows", "row_slots", "segments", "keeps", "kept_rows", "slot_steps" }) |name| {
            const r = @field(m, name);
            const bytes = std.mem.sliceAsBytes(@field(plan, name));
            @memcpy(r.buffer.contents()[r.offset..][0..bytes.len], bytes);
        }
    }
};

pub const Scalars = struct {
    conv: gdn.Span,
    a_log: gdn.Span,
    dt: gdn.Span,
    norm: gdn.Span,
    kinds: abi.WeightKinds,

    pub fn from(g: @import("gpu_weights.zig").Gdn) !Scalars {
        return .{ .conv = tensorSpan(g.conv), .a_log = tensorSpan(g.a_log), .dt = tensorSpan(g.dt), .norm = tensorSpan(g.norm), .kinds = .{ .conv = try abi.storage(g.conv.dtype), .a_log = try abi.storage(g.a_log.dtype), .dt = try abi.storage(g.dt.dtype), .norm = try abi.storage(g.norm.dtype) } };
    }
};

pub const DevicePlan = struct { host: Plan, metadata: Metadata };
/// Rows depth-one star keeps stashed (Pool.stash) instead of replaying them into `slot`'s committed states, in keep order,
/// and the rounds that stashed them since the last write-back: the committed words lag them until a star scan writes the
/// state back (abi.Deferred.persist) or unstash replays them.
pub const Pending = struct { rows: u32 = 0, rounds: u32 = 0, slot: u32 = 0 };
/// z, a and b in their own rows, or (zba_stride > 0) views into one fused zba row.
/// `sums`: the gate also writes the out projection's group sums of its rows there (gdn.Round.sums).
pub const Refs = struct { z: Ref, a: Ref, b: Ref, output: Ref, scalars: Scalars, zba_stride: u32 = 0, sums: ?gdn.Sums = null };

pub const Pool = struct {
    budget: Budget,
    buffers: [std.enums.values(Field).len]metal.Buffer,
    kernels: Kernels,
    before: ?metal.Buffer = null,
    marks: ?*PromptMarks = null, // borrowed through one synchronous plain prompt and its cache callbacks
    before_slots: usize = 0,
    profiler: ?*profile.Trace = null,
    tree_mode: gdn.TreeMode = .auto,
    residency: ?metal.ResidencySet.Follow = null,
    /// The decoded star scan's count and its waiting gate's arrivals (gdn.Round.forwardStar), left zero.
    scan_words: ?metal.Buffer = null,
    /// Read by a deferred star scan, the keep that stashes behind it and unstash only; every other reader of the committed
    /// states runs after the rows land (decode_round.Runner.settleGdn).
    pending: Pending = .{},

    pub fn setResident(pool: *Pool, hook: ?metal.ResidencySet.Follow) void {
        if (pool.residency) |f| metal.resident.fields.apply(pool, f, false);
        pool.residency = hook;
        if (hook) |f| metal.resident.fields.apply(pool, f, true);
    }

    /// One state a slot: every round's keeps stay in their slots and update them in place.
    pub fn init(device: metal.Device, shape: abi.Shape, layers: usize, slots: usize, capacity: usize) !Pool {
        const budget = try Budget.init(shape, layers, slots, capacity);
        if (capacity % 16 != 0) return error.BadGdnProjectionCapacity;
        const buffers = try allocateBuffers(device, budget);
        errdefer for (buffers) |buffer| buffer.deinit();
        const kernels = try Kernels.init(device);
        const scan_words = try device.buffer(8, metal.ResourceOptions.shared | metal.ResourceOptions.untracked);
        @memset(scan_words.contents()[0..8], 0);
        return .{ .budget = budget, .buffers = buffers, .kernels = kernels, .scan_words = scan_words };
    }

    pub fn allocateBuffers(device: metal.Device, budget: Budget) !@FieldType(Pool, "buffers") {
        const widths = budget.allocationSizes();
        var buffers: [widths.len]metal.Buffer = undefined;
        var count: usize = 0;
        errdefer for (buffers[0..count]) |buffer| buffer.deinit();
        for (widths, &buffers) |bytes, *buffer| {
            buffer.* = try device.buffer(@max(bytes, 16), metal.ResourceOptions.shared | metal.ResourceOptions.untracked);
            count += 1;
        }
        for ([_]Field{ .state, .history }) |field| @memset(buffers[@backingInt(field)].contents()[0..widths[@backingInt(field)]], 0);
        return buffers;
    }

    pub fn deinit(pool: *Pool) void {
        pool.setResident(null);
        pool.releaseBefore();
        pool.kernels.deinit();
        for (pool.buffers) |buffer| buffer.deinit();
        if (pool.scan_words) |b| b.deinit();
    }

    /// Called between completed rounds, with no pending keep, before the first encoder can use the bank.
    pub fn prepare(pool: *Pool, device: metal.Device, plan: Plan, accepted: bool) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan);
        if (!inPlace(plan)) return error.GdnSlotMoves;
        const slots = @import("gdn_pool.zig").beforeSlots(plan, !accepted and pool.advances(plan));
        if (slots == pool.before_slots) return;
        const bytes = try abi.countBytes(f32, pool.budget.shape.state, pool.budget.layers * slots);
        // The old bank has no readers now; drop it first so growth never holds two rollback banks.
        pool.releaseBefore();
        if (slots == 0) return;
        pool.before = try device.buffer(bytes, metal.ResourceOptions.shared | metal.ResourceOptions.untracked);
        pool.before_slots = slots;
        if (pool.residency) |f| f.join(f.ctx, pool.before.?);
    }

    /// No submitted work or pending keep may reference the bank when its owner calls this.
    pub fn releaseBefore(pool: *Pool) void {
        if (pool.before) |buffer| {
            if (pool.residency) |f| f.leave(f.ctx, buffer);
            buffer.deinit();
        }
        pool.before = null;
        pool.before_slots = 0;
    }

    /// Reserve the still-unallocated peak bank beside prompt caches and future stream buffers.
    pub fn missingBeforeBytes(pool: *const Pool) usize {
        if (pool.budget.slots < 2 or pool.budget.shape.c.dk != 128 or pool.tree_mode == .snapshot) return 0;
        return pool.budget.state -| (if (pool.before) |buffer| buffer.length() else 0);
    }

    /// Borrow committed bytes only after all encoded work has completed.
    pub fn committed(pool: *const Pool, layer: usize, slot: usize, recurrent: bool) ![]const u8 {
        const span = try pool.committedSpan(layer, slot, recurrent);
        return span.buffer.contents()[span.offset..][0..span.bytes];
    }

    /// Where a slot's committed state (recurrent) or conv history lives, for copies on the GPU.
    pub fn committedSpan(pool: *const Pool, layer: usize, slot: usize, recurrent: bool) !struct { buffer: metal.Buffer, offset: usize, bytes: usize } {
        const off = if (recurrent) try pool.budget.stateOffset(layer, slot) else try pool.budget.historyOffset(layer, slot);
        const bytes = if (recurrent) pool.budget.shape.state * 4 else pool.budget.shape.conv * 2;
        return .{ .buffer = pool.buffers[@backingInt(if (recurrent) Field.state else Field.history)], .offset = off, .bytes = bytes };
    }

    pub fn projected(pool: *const Pool, layer: usize) !Ref {
        const span = try pool.layerSpan(.qkv, layer);
        return .{ .buffer = span.buffer, .offset = span.offset };
    }

    /// `beside`: encoded beside the scan, before the barrier that ends it.
    /// The counted projections a decoded prepare waits on in its barrier interval (Kernels.quantCounted).
    pub const After = struct { counter: metal.Buffer, wait: u32 };

    pub fn forward(pool: *Pool, layer: usize, plan: DevicePlan, refs: Refs, e: metal.ComputeEncoder, beside: ?gdn.Beside) !void {
        return pool.forwardAfter(layer, plan, refs, e, beside, null, null);
    }

    /// `deferred`: a lone depth-one star verify's scan applies the first `rows` pending rows before node 0
    /// (tf_delta_gdn_star_pending) and, on `persist`, writes the state they leave back over the committed one.
    pub fn forwardAfter(pool: *Pool, layer: usize, plan: DevicePlan, refs: Refs, e: metal.ComputeEncoder, beside: ?gdn.Beside, after: ?After, deferred: ?abi.Deferred) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        const prepared = try pool.round(layer, plan, refs, false);
        try plan.metadata.upload(plan.host);
        var pre = profile.begin(pool.profiler, e, .gdn_prepare);
        if (after) |a| try prepared.prepareRowsAfter(pre.encoder, pool.kernels.pre_rows, a.counter, a.wait) else prepared.prepareRows(pre.encoder, pool.kernels.pre_rows);
        try pre.end();
        var scan = profile.begin(pool.profiler, e, .gdn_scan);
        if (prepared.star and prepared.usesSlots()) {
            if (deferred) |d| {
                if (d.rows > pending_rows) return error.BadGdnPending;
                const pending = gdn.Pending{ .key = try pool.layerSpan(.pending_key, layer), .value = try pool.layerSpan(.pending_value, layer), .decay = try pool.layerSpan(.pending_decay, layer), .mixing = try pool.layerSpan(.pending_mixing, layer), .use = d };
                try prepared.forwardStar(scan.encoder, pool.kernels.star_pending, pool.kernels.post_rows, beside, pool.scan_words, pending);
            } else try prepared.forwardStar(scan.encoder, pool.kernels.star_pair, pool.kernels.post_rows, beside, pool.scan_words, null);
        } else if (deferred != null) {
            return error.BadGdnPlan;
        } else try prepared.forwardPair(scan.encoder, pool.kernels.tree_pair, pool.kernels.post_rows, beside);
        try scan.end();
    }

    /// Whether a verify of `plan` scans one depth-one star window (gdn.Round's `star` and usesSlots on one segment): the
    /// rounds whose keeps may defer their replays.
    pub fn starPlan(pool: *const Pool, plan: Plan) bool {
        const c = pool.budget.shape.c;
        if (plan.segments.len != 1 or !plan.compact or pool.tree_mode == .snapshot or c.dk != 128 or c.dv % 2 != 0) return false;
        for (plan.parents, 0..) |parent, row| if (parent != @as(i32, if (row == 0) -1 else 0)) return false;
        return true;
    }

    /// Publish a fully kept chain after its per-position update; prompt chains take the wide scan.
    pub fn forwardChain(pool: *Pool, layer: usize, plan: DevicePlan, refs: Refs, input_encoder: metal.ComputeEncoder, prompt: bool, beside: ?gdn.Beside) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        for (plan.host.segments, plan.host.keeps) |segment, keep| {
            if (keep.rows != segment.rows) return error.NotAcceptedChain;
            for (0..segment.rows) |row| if (plan.host.parents[segment.first + row] != (if (row == 0) @as(i32, -1) else @as(i32, @intCast(row - 1)))) return error.NotAcceptedChain;
        }
        // Each scan thread reads and writes only its own state words, so a chain that keeps its slots scans in place.
        var prepared = try pool.round(layer, plan, refs, true);
        const in_place = inPlace(plan.host);
        if (in_place) prepared.b.next_state = prepared.b.state;
        // one row a slot kept in place: its prepare writes the next conv window, so no commit or publication follows
        const step = step_rounds and in_place and !prompt and oneRowSegments(plan.host);
        try plan.metadata.upload(plan.host);
        var pre = profile.begin(pool.profiler, input_encoder, .gdn_prepare);
        errdefer pre.cancel();
        if (step) prepared.prepareStep(pre.encoder, pool.kernels.pre_step) else prepared.prepareRows(pre.encoder, pool.kernels.pre_rows);
        try pre.end();
        var scan = profile.begin(pool.profiler, input_encoder, .gdn_scan);
        errdefer scan.cancel();
        const wide = prompt and pool.budget.shape.c.dk == 128 and pool.budget.shape.c.dv % 16 == 0;
        if (pool.marks) |m| {
            if (!wide) return error.BadPromptMarks;
            try m.scan(layer, prepared, scan.encoder, pool.kernels.chain_wide_marks);
            if (beside) |b| {
                try b.encode(b.ptr, scan.encoder);
                scan.encoder.barrier();
            }
        } else {
            // the conv windows' commit reads the history and the projected rows and writes the next windows: nothing the
            // scan or the gate reads or writes, so it runs beside the scan instead of behind the gate's barrier
            var early = EarlyCommit{ .inner = beside, .round = prepared };
            if (!step and inPlace(plan.host) and projectedWindows(plan.host, prepared.p.taps)) early.round.b.next_history = prepared.b.history;
            try prepared.scanChain(scan.encoder, if (wide) pool.kernels.chain_wide else pool.kernels.chain, wide, if (step) beside else .{ .ptr = &early, .encode = EarlyCommit.encode });
        }
        try scan.end();
        var gate = profile.begin(pool.profiler, input_encoder, .gdn_gate);
        errdefer gate.cancel();
        if (prompt) prepared.gatePrompt(gate.encoder, pool.kernels.post_rows16, pool.kernels.post_rows) else prepared.gate(gate.encoder, pool.kernels.post_rows);
        try gate.end();
        if (step) return;
        var publication = profile.begin(pool.profiler, input_encoder, .state_publish);
        errdefer publication.cancel();
        const e = publication.encoder;
        if (pool.marks) |m| try m.convolution(layer, prepared, e, pool.kernels.mark_conv);
        const commit_done = pool.marks == null; // the commit ran beside the scan (EarlyCommit)
        if (in_place and projectedWindows(plan.host, prepared.p.taps)) {
            // no kept window reads its history: each lands there directly and nothing is left to publish
            var direct = prepared;
            direct.b.next_history = prepared.b.history;
            if (!commit_done) direct.commitConv(e);
            return publication.end();
        }
        if (!commit_done) prepared.commitConv(e);
        e.setPipeline(pool.kernels.publish_conv);
        gdn.bind(e, .{prepared.b.next_history}, 0);
        e.setBuffer(plan.metadata.keeps.buffer, plan.metadata.keeps.offset, 1);
        e.setValue(prepared.p, 2);
        gdn.bind(e, .{prepared.b.history}, 3);
        e.dispatchThreads(.{ .width = pool.budget.shape.conv, .height = plan.host.keeps.len }, .{ .width = 256 }); // a word a thread
        e.barrier();
        try publication.end();
    }

    pub fn commit(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        try plan.metadata.upload(plan.host);
        return pool.commitDevice(layer, plan, e);
    }

    /// Caller-generated keep descriptors remain on the GPU while the admitted host plan fixes dispatch bounds.
    pub fn commitDevice(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try pool.commitScan(layer, plan, e);
        e.barrier();
        try pool.commitPublish(layer, plan, e);
        e.barrier();
    }

    /// A commit's replay, in place, and its conv window, unfenced.
    pub fn commitScan(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try pool.commitReplay(layer, plan, e);
        const p = try pool.budget.shape.params(plan.host.parents.len, pool.budget.slots, commit_kinds);
        const m = plan.metadata;
        const next_history = try pool.layerSpan(.next_history, layer);
        e.setPipeline(pool.kernels.pipelines.conv_commit);
        gdn.bind(e, .{ try pool.layerSpan(.history, layer), try pool.layerSpan(.qkv, layer) }, 0);
        e.setBuffer(m.kept_rows.buffer, m.kept_rows.offset, 2);
        e.setBuffer(m.keeps.buffer, m.keeps.offset, 3);
        e.setBuffer(m.windows.buffer, m.windows.offset, 4);
        e.setValue(p, 5);
        e.setBuffer(next_history.buffer, next_history.offset, 6);
        e.dispatchThreads(.{ .width = pool.budget.shape.conv, .height = plan.host.keeps.len }, .{ .width = 256 }); // a word a thread
    }

    /// commitScan and commitPublish in one fence-free pass: the replay, and the conv window written straight over
    /// history by tf_delta_gdn_conv_keep (a thread a channel reads every source tap before it writes), so the keep
    /// needs no next_history staging, no publish_conv launch and no fence between the two.
    pub fn commitKeep(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try pool.commitReplay(layer, plan, e);
        try pool.commitConvKeep(layer, plan, e);
    }

    /// commitKeep's conv window alone (a keep whose replay `stash` defers), unfenced.
    pub fn commitConvKeep(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        if (!inPlace(plan.host)) return error.GdnSlotMoves;
        const p = try pool.budget.shape.params(plan.host.parents.len, pool.budget.slots, commit_kinds);
        const m = plan.metadata;
        e.setPipeline(pool.kernels.conv_keep);
        gdn.bind(e, .{ try pool.layerSpan(.history, layer), try pool.layerSpan(.qkv, layer) }, 0);
        e.setBuffer(m.kept_rows.buffer, m.kept_rows.offset, 2);
        e.setBuffer(m.keeps.buffer, m.keeps.offset, 3);
        e.setBuffer(m.windows.buffer, m.windows.offset, 4);
        e.setValue(p, 5);
        e.dispatchThreads(.{ .width = pool.budget.shape.qkv, .height = plan.host.keeps.len }, .{ .width = 256 }); // a channel a thread
    }

    /// A lone depth-one star keep that defers its replay, every layer in one launch: the kept rows (keeps[0] and kept_rows,
    /// the device keep's: the rows tf_delta_gdn_replay would take, in its order) copied behind the pending rows for the next
    /// star scans; unfenced, beside the conv keeps. The keep's completion counts them (decode_round.Runner).
    pub fn stash(pool: *Pool, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        const c = pool.budget.shape.c;
        if (plan.host.keeps.len != 1 or !inPlace(plan.host) or c.dk != 128 or c.dv != 128 or pool.pending.rows >= pending_rows) return error.BadGdnPending;
        const p = try pool.budget.shape.params(plan.host.parents.len, pool.budget.slots, commit_kinds);
        const m = plan.metadata;
        e.setPipeline(pool.kernels.stash);
        inline for (.{ Field.key, Field.value, Field.decay, Field.mixing }, 0..) |field, index| e.setBuffer(pool.get(field), 0, index);
        e.setBuffer(m.kept_rows.buffer, m.kept_rows.offset, 4);
        e.setBuffer(m.keeps.buffer, m.keeps.offset, 5);
        e.setValue(p, 6);
        e.setValue(abi.Stash{ .base = pool.pending.rows, .room = pending_rows - pool.pending.rows, .log_rows = @intCast(pool.budget.rows), .pending_rows = pending_rows }, 7);
        inline for (pending_fields, 8..) |field, index| e.setBuffer(pool.get(field), 0, index);
        // a thread a 16-byte word of the key and value rows and a word of the gates, a layer a row of threads
        e.dispatchThreads(.{ .width = @max(c.k_heads * c.dk / 8, pool.budget.shape.value / 8, c.v_heads), .height = pool.budget.layers }, .{ .width = 256 });
    }

    /// The pending rows replayed into their slot's committed state of every layer, in place and in keep order, each by
    /// the delta update a keep's replay runs (tf_delta_gdn_unstash); one launch, unfenced. Its caller forgets them.
    pub fn unstash(pool: *Pool, e: metal.ComputeEncoder) !void {
        const c = pool.budget.shape.c;
        if (pool.pending.rows > pending_rows or pool.pending.slot >= pool.budget.slots or c.dk != 128) return error.BadGdnPending;
        const p = try pool.budget.shape.params(pending_rows, pool.budget.slots, commit_kinds);
        e.setPipeline(pool.kernels.unstash);
        inline for (pending_fields, 0..) |field, index| e.setBuffer(pool.get(field), 0, index);
        e.setBuffer(pool.get(.state), 0, 4);
        e.setValue(p, 5);
        const layer_words = std.math.cast(u32, pool.budget.slots * pool.budget.shape.state) orelse return error.BadGdnPending;
        e.setValue(abi.Unstash{ .rows = pool.pending.rows, .slot = pool.pending.slot, .pending_rows = pending_rows, .layer_words = layer_words }, 6);
        e.dispatchThreads(.{ .width = 32, .height = p.dv, .depth = p.nv * pool.budget.layers }, .{ .width = 32, .height = 4 });
    }

    /// The replay half of a commit, in place; skipped when every window's verify already stepped its state.
    fn commitReplay(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        try @import("gdn_pool.zig").validatePlan(pool.budget.shape, pool.budget.slots, pool.budget.rows, plan.host);
        if (!inPlace(plan.host)) return error.GdnSlotMoves;
        var p = try pool.budget.shape.params(plan.host.parents.len, pool.budget.slots, commit_kinds);
        p.advance = @intFromBool(pool.advances(plan.host));
        const state = try pool.layerSpan(.state, layer);
        const m = plan.metadata;
        // windows the verify stepped in place replay nothing: skip the launch when that is all of them
        const replays = p.advance == 0 or for (plan.host.segments, plan.host.keeps) |segment, keep| {
            if (segment.rows > 1 or keep.rows == 0) break true;
        } else false;
        if (replays) {
            e.setPipeline(pool.kernels.pipelines.replay);
            e.setBuffer(m.segments.buffer, m.segments.offset, 9);
            ((try pool.beforeSpan(layer, plan.host, pool.advances(plan.host))) orelse state).bind(e, 10);
            gdn.bind(e, .{ try pool.layerSpan(.key, layer), try pool.layerSpan(.value, layer), try pool.layerSpan(.decay, layer), try pool.layerSpan(.mixing, layer), state }, 0);
            e.setBuffer(m.kept_rows.buffer, m.kept_rows.offset, 5);
            e.setBuffer(m.keeps.buffer, m.keeps.offset, 6);
            e.setValue(p, 7);
            e.setBuffer(state.buffer, state.offset, 8);
            e.dispatchThreads(.{ .width = 32, .height = p.dv, .depth = p.nv * plan.host.keeps.len }, .{ .width = 32, .height = 4 });
        }
    }

    /// After commitScan's fence: the new conv window into place, unfenced.
    pub fn commitPublish(pool: *Pool, layer: usize, plan: DevicePlan, e: metal.ComputeEncoder) !void {
        if (!inPlace(plan.host)) return error.GdnSlotMoves;
        const p = try pool.budget.shape.params(plan.host.parents.len, pool.budget.slots, commit_kinds);
        const history = try pool.layerSpan(.history, layer);
        e.setPipeline(pool.kernels.publish_conv);
        gdn.bind(e, .{try pool.layerSpan(.next_history, layer)}, 0);
        e.setBuffer(plan.metadata.keeps.buffer, plan.metadata.keeps.offset, 1);
        e.setValue(p, 2);
        e.setBuffer(history.buffer, history.offset, 3);
        e.dispatchThreads(.{ .width = pool.budget.shape.conv, .height = plan.host.keeps.len }, .{ .width = 256 }); // a word a thread
    }

    pub fn clearSlot(pool: *Pool, slot: u32, e: metal.ComputeEncoder) !void {
        if (slot >= pool.budget.slots) return error.BadGdnPoolIndex;
        const k = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
        const p = try pool.budget.shape.params(1, pool.budget.slots, k);
        e.setPipeline(pool.kernels.clear);
        e.setBuffer(pool.get(.state), 0, 0);
        e.setBuffer(pool.get(.history), 0, 1);
        e.setValue(p, 2);
        e.setValue(slot, 3);
        e.dispatchThreads(.{ .width = pool.budget.shape.state, .height = pool.budget.layers }, .{ .width = 32 });
        e.barrier();
    }

    fn round(pool: *Pool, layer: usize, device_plan: DevicePlan, refs: Refs, accepted: bool) !gdn.Round {
        const plan = device_plan.host;
        if (plan.parents.len > pool.budget.rows) return error.BadGdnPoolIndex;
        var p = try pool.budget.shape.params(plan.parents.len, pool.budget.slots, refs.scalars.kinds);
        p.zba_stride = refs.zba_stride;
        const bytes = try abi.Bytes.init(pool.budget.shape, p, refs.scalars.kinds);
        const m = device_plan.metadata;
        const zs: usize = refs.zba_stride;
        const heads = pool.budget.shape.c.v_heads;
        const b = gdn.Buffers{
            .qkv = try pool.layerSpan(.qkv, layer),
            .a = try refSpan(refs.a, if (zs == 0) bytes.heads else ((p.rows - 1) * zs + heads) * 2),
            .b = try refSpan(refs.b, if (zs == 0) bytes.heads else ((p.rows - 1) * zs + heads) * 2),
            .z = try refSpan(refs.z, if (zs == 0) bytes.value else ((p.rows - 1) * zs + pool.budget.shape.value) * 2),
            .conv_weight = refs.scalars.conv,
            .a_log = refs.scalars.a_log,
            .dt = refs.scalars.dt,
            .norm = refs.scalars.norm,
            .history = try pool.layerSpan(.history, layer),
            .state = try pool.layerSpan(.state, layer),
            .before = try pool.beforeSpan(layer, plan, !accepted and pool.advances(plan)),
            .query = try pool.layerSpan(.query, layer),
            .key = try pool.layerSpan(.key, layer),
            .value = try pool.layerSpan(.value, layer),
            .decay = try pool.layerSpan(.decay, layer),
            .mixing = try pool.layerSpan(.mixing, layer),
            .tails = try refSpan(.{ .buffer = pool.get(.tails) }, bytes.tails),
            .snapshots = try refSpan(.{ .buffer = pool.get(.snapshots) }, bytes.snapshots),
            .recurrent = try refSpan(.{ .buffer = pool.get(.recurrent) }, bytes.value),
            .output = try refSpan(refs.output, bytes.value),
            .next_state = if (inPlace(plan)) try pool.layerSpan(.state, layer) else return error.GdnSlotMoves,
            .next_history = try pool.layerSpan(.next_history, layer),
            .parents = try refSpan(m.parents, plan.parents.len * 4),
            .windows = try refSpan(m.windows, plan.windows.len * 4),
            .row_slots = try refSpan(m.row_slots, plan.row_slots.len * 4),
            .segments = try refSpan(m.segments, plan.segments.len * @sizeOf(abi.Segment)),
            .keeps = try refSpan(m.keeps, plan.keeps.len * @sizeOf(abi.Keep)),
            .kept_rows = try refSpan(m.kept_rows, plan.kept_rows.len * 4),
            .slot_steps = try refSpan(m.slot_steps, plan.slot_steps.len * @sizeOf(@import("core").tree_slots.Step)),
        };
        var result = try gdn.Round.init(pool.budget.shape, refs.scalars.kinds, p, plan, b, pool.kernels.pipelines);
        result.tree_mode = pool.tree_mode;
        result.p.advance = @intFromBool(pool.advances(plan));
        if (refs.sums) |w| {
            // one fp32 sum per 64-value group of each row, `stride` rows a group (tf_lane_sums' layout)
            const groups = pool.budget.shape.value / 64;
            if (w.stride < p.rows or groups == 0 or pool.budget.shape.value % 64 != 0) return error.BadGdnBuffer;
            _ = try refSpan(.{ .buffer = w.span.buffer, .offset = w.span.offset }, ((groups - 1) * w.stride + p.rows) * 4);
            result.sums = w;
        }
        return result;
    }

    /// Whether a round's one-row windows step their state in its slot scan (shared rounds), so the replay skips them.
    fn advances(pool: *const Pool, plan: Plan) bool {
        return plan.segments.len > 1 and plan.compact and pool.budget.shape.c.dk == 128 and pool.tree_mode != .snapshot;
    }

    fn beforeSpan(pool: *const Pool, layer: usize, plan: Plan, advance: bool) !?gdn.Span {
        const slots = @import("gdn_pool.zig").beforeSlots(plan, advance);
        if (slots == 0) return null;
        if (layer >= pool.budget.layers or slots > pool.before_slots) return error.MissingGdnBefore;
        const buffer = pool.before orelse return error.MissingGdnBefore;
        const bytes = try abi.countBytes(f32, pool.budget.shape.state, pool.before_slots);
        return .{ .buffer = buffer, .offset = layer * bytes, .bytes = bytes };
    }

    fn get(pool: *const Pool, field: Field) metal.Buffer {
        return pool.buffers[@backingInt(field)];
    }

    fn layerSpan(pool: *const Pool, field: Field, layer: usize) !gdn.Span {
        if (layer >= pool.budget.layers) return error.BadGdnPoolIndex;
        const buffer = pool.get(field);
        const bytes = buffer.length() / pool.budget.layers;
        return .{ .buffer = buffer, .offset = layer * bytes, .bytes = bytes };
    }
};

fn tensorSpan(t: Tensor) gdn.Span {
    return .{ .buffer = t.buffer, .offset = t.offset, .bytes = t.bytes };
}

/// One-row rounds take tf_delta_gdn_pre_step; off, they run the general prepare, commit and publication (checks).
pub var step_rounds = true;

fn oneRowSegments(plan: Plan) bool {
    for (plan.segments) |segment| if (segment.rows != 1) return false;
    return true;
}

/// Whether every kept window (a keep's last row's next conv window) reads projected rows alone.
fn projectedWindows(plan: Plan, taps: u32) bool {
    for (plan.keeps) |keep| {
        if (keep.rows == 0) continue;
        const row: usize = plan.kept_rows[keep.first + keep.rows - 1];
        for (plan.windows[row * taps + 1 ..][0 .. taps - 1]) |source| if (source < taps - 1) return false;
    }
    return true;
}

/// The scalar kinds every commit launch states in its params.
const commit_kinds = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };

fn inPlace(plan: Plan) bool {
    for (plan.keeps) |keep| if (keep.state_slot != keep.next_slot) return false;
    return true;
}

fn refSpan(r: Ref, bytes: usize) !gdn.Span {
    if (r.offset > r.buffer.length() or bytes > r.buffer.length() - r.offset) return error.BadGdnBuffer;
    return .{ .buffer = r.buffer, .offset = r.offset, .bytes = bytes };
}

/// A prompt chain's conv commit encoded beside its scan (after any other beside work), the barrier after the scan ordering
/// it before the publication that reads the next windows.
const EarlyCommit = struct {
    inner: ?gdn.Beside,
    round: gdn.Round,
    fn encode(ptr: *anyopaque, e: metal.ComputeEncoder) anyerror!void {
        const c: *EarlyCommit = @ptrCast(@alignCast(ptr));
        if (c.inner) |b| try b.encode(b.ptr, e);
        c.round.commitConvBeside(e);
    }
};
