//! Borrowed buffers and caller-owned pipelines encode a round without allocating, committing or reading back.
const std = @import("std");
const metal = @import("metal");
const abi = @import("gdn_contract.zig");
const Plan = @import("gdn_plan.zig").Plan;

pub const Span = struct {
    buffer: metal.Buffer,
    offset: usize = 0,
    bytes: usize,

    fn check(s: Span, required: usize) !void {
        try abi.checkSpan(s.buffer.length(), s.offset, s.bytes, required, 2);
    }

    pub fn bind(s: Span, encoder: metal.ComputeEncoder, index: usize) void {
        encoder.setBuffer(s.buffer, s.offset, index);
    }

    fn overlaps(a: Span, b: Span) bool {
        return a.buffer.id == b.buffer.id and a.offset < b.offset + b.bytes and b.offset < a.offset + a.bytes;
    }
};

pub const TreeMode = enum { auto, snapshot };

pub const Pipelines = @import("core_delta").Pipelines;

pub const Buffers = struct {
    qkv: Span,
    a: Span,
    b: Span,
    z: Span,
    conv_weight: Span,
    a_log: Span,
    dt: Span,
    norm: Span,
    history: Span,
    state: Span,
    query: Span,
    key: Span,
    value: Span,
    decay: Span,
    mixing: Span,
    tails: Span,
    snapshots: Span,
    recurrent: Span,
    output: Span,
    next_state: Span,
    before: ?Span = null,
    next_history: Span,
    parents: Span,
    windows: Span,
    row_slots: Span,
    segments: Span,
    keeps: Span,
    kept_rows: Span,
    slot_steps: Span,

    pub fn check(b: Buffers, sizes: abi.Bytes, kinds: abi.WeightKinds, p: abi.Params, plan: Plan) !void {
        try b.qkv.check(sizes.qkv);
        if (b.before) |before| {
            if (p.slots == 0) return error.BadGdnDispatch;
            const slots = @import("gdn_pool.zig").beforeSlots(plan, true);
            try before.check(sizes.state / p.slots * slots);
            if (before.offset % 4 != 0) return error.BadGdnBuffer;
        }
        inline for (.{ "a", "b", "mixing" }) |name| try @field(b, name).check(sizes.heads);
        try b.z.check(sizes.value);
        try b.conv_weight.check(sizes.conv_weight);
        try b.a_log.check(sizes.a_log);
        try b.dt.check(sizes.dt);
        try b.norm.check(sizes.norm);
        inline for (.{ "history", "next_history" }) |name| try @field(b, name).check(sizes.history);
        inline for (.{ "state", "next_state" }) |name| try @field(b, name).check(sizes.state);
        inline for (.{ "query", "key" }) |name| try @field(b, name).check(sizes.query);
        inline for (.{ "value", "recurrent", "output" }) |name| try @field(b, name).check(sizes.value);
        try b.decay.check(sizes.decay);
        try b.tails.check(sizes.tails);
        try b.snapshots.check(sizes.snapshots);
        try b.parents.check(try abi.countBytes(i32, p.rows, 1));
        try b.windows.check(try abi.countBytes(u32, p.rows, p.taps));
        try b.row_slots.check(try abi.countBytes(u32, p.rows, 1));
        try b.segments.check(try abi.countBytes(abi.Segment, plan.segments.len, 1));
        try b.keeps.check(try abi.countBytes(abi.Keep, plan.keeps.len, 1));
        try b.slot_steps.check(plan.slot_steps.len * @sizeOf(@import("core").tree_slots.Step));
        try b.kept_rows.check(try abi.countBytes(u32, plan.kept_rows.len, 1));
        inline for (.{ "state", "next_state", "decay", "snapshots", "parents", "windows", "row_slots", "segments", "keeps", "kept_rows", "slot_steps" }) |name| if (@field(b, name).offset % 4 != 0) return error.BadGdnBuffer;
        if ((kinds.conv == .f32 and b.conv_weight.offset % 4 != 0) or (kinds.a_log == .f32 and b.a_log.offset % 4 != 0) or (kinds.dt == .f32 and b.dt.offset % 4 != 0) or (kinds.norm == .f32 and b.norm.offset % 4 != 0)) return error.BadGdnBuffer;
        const reads = [_]Span{ b.qkv, b.a, b.b, b.z, b.conv_weight, b.a_log, b.dt, b.norm, b.history, b.state, b.parents, b.windows, b.row_slots, b.segments, b.keeps, b.kept_rows, b.slot_steps };
        // an in-place round's next state is its state: each scan thread reads its own words before it writes them
        const in_place = b.next_state.buffer.id == b.state.buffer.id and b.next_state.offset == b.state.offset and b.next_state.bytes == b.state.bytes;
        const writes = [_]Span{ b.query, b.key, b.value, b.decay, b.mixing, b.tails, b.snapshots, b.recurrent, b.output, if (in_place) .{ .buffer = b.state.buffer, .bytes = 0 } else b.next_state, b.next_history, b.before orelse Span{ .buffer = b.state.buffer, .bytes = 0 } };
        inline for (writes, 0..) |write, index| {
            inline for (reads) |read| if (write.overlaps(read)) return error.AliasedGdnState;
            inline for (writes[0..index]) |other| if (write.overlaps(other)) return error.AliasedGdnState;
        }
    }
};

/// Group sums a gate writes for the projection that reads its rows; `written` is set once a dispatch writes them.
pub const Sums = struct { span: Span, stride: u32, written: *bool };
/// Work encoded beside a scan, before the barrier that ends it: rows only the gate reads (the z projection's).
pub const Beside = struct { ptr: *anyopaque, encode: *const fn (*anyopaque, metal.ComputeEncoder) anyerror!void };
/// A deferred replay's rows for a star scan (tf_delta_gdn_star_pending): the layer's pending key, value, decay and mixing
/// rows, and how many it applies and whether it writes the state back.
pub const Pending = struct { key: Span, value: Span, decay: Span, mixing: Span, use: abi.Deferred };

pub const Round = struct {
    p: abi.Params,
    b: Buffers,
    pipelines: Pipelines,
    segments: usize,
    conv_width: usize,
    tree_mode: TreeMode = .auto,
    /// The out projection's group sums `gate` writes beside its rows (tf_lane_sums' layout, `stride` rows a group).
    sums: ?Sums = null,
    compact: bool,
    slot_kernel: usize, // which tree_slots kernel holds the plan's peak of live states
    /// Every segment a depth-one star (node 0, the rest its children) on a verify that steps no state: forwardStar.
    star: bool = false,

    pub fn init(shape: abi.Shape, kinds: abi.WeightKinds, p: abi.Params, plan: Plan, b: Buffers, pipelines: Pipelines) !Round {
        if (plan.parents.len != p.rows or plan.segments.len == 0 or plan.keeps.len != plan.segments.len or (plan.compact and plan.slot_steps.len != p.rows)) return error.BadGdnPlan;
        try b.check(try abi.Bytes.init(shape, p, kinds), kinds, p, plan);
        inline for (.{ pipelines.pre, pipelines.post, pipelines.conv_commit }) |pipeline| if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < 32) return error.UnsupportedGdnPipeline;
        inline for (.{ pipelines.tree, pipelines.replay }) |pipeline| if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < 128) return error.UnsupportedGdnPipeline;
        for (pipelines.tree_slots) |pipeline| if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < 128) return error.UnsupportedGdnPipeline;
        const slots = @import("core_delta").slotKernel(plan.slot_peak);
        if (plan.compact) for (plan.segments) |segment| try @import("core").tree_slots.validate(plan.parents[segment.first..][0..segment.rows], plan.slot_steps[segment.first..][0..segment.rows], segment.first, @import("core_delta").slot_kernels[slots]);
        var star = plan.compact and p.advance == 0 and b.before == null and p.dk == 128 and p.dv % 2 == 0;
        for (plan.segments) |segment| for (plan.parents[segment.first..][0..segment.rows], 0..) |parent, i| {
            if (parent != @as(i32, if (i == 0) -1 else 0)) star = false;
        };
        return .{ .slot_kernel = slots, .compact = plan.compact, .star = star, .p = p, .b = b, .pipelines = pipelines, .segments = plan.segments.len, .conv_width = shape.conv };
    }

    pub fn prepare(r: Round, e: metal.ComputeEncoder) void {
        e.setPipeline(r.pipelines.pre);
        bind(e, .{ r.b.qkv, r.b.a, r.b.b, r.b.history, r.b.conv_weight, r.b.a_log, r.b.dt, r.b.windows, r.b.row_slots }, 0);
        e.setValue(r.p, 9);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.tails }, 10);
        e.dispatchGroups(.{ .width = 1, .height = 2 * r.p.nk + r.p.nv, .depth = r.p.rows }, .{ .width = 32, .height = 1, .depth = 1 });
        e.barrier();
    }

    /// prepare for 128-wide heads and 4 taps by `rows` (tf_delta_gdn_pre_rows): 16 rows a group of 128.
    pub fn prepareRows(r: Round, e: metal.ComputeEncoder, rows: metal.Pipeline) void {
        if (r.p.rows == 0 or r.p.dk != 128 or r.p.dv != 128 or r.p.taps != 4) return r.prepare(e);
        inline for (.{ r.b.qkv, r.b.history, r.b.query, r.b.key, r.b.value }) |s| if (s.offset % 8 != 0) return r.prepare(e); // bfloat4 rows
        e.setPipeline(rows);
        bind(e, .{ r.b.qkv, r.b.a, r.b.b, r.b.history, r.b.conv_weight, r.b.a_log, r.b.dt, r.b.windows, r.b.row_slots }, 0);
        e.setValue(r.p, 9);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing }, 10);
        r.b.qkv.bind(e, 15); // never read: wait_for 0
        e.setValue(@as(u32, 0), 16);
        e.dispatchGroups(.{ .width = 1, .height = 2 * r.p.nk + r.p.nv, .depth = (r.p.rows + 15) / 16 }, .{ .width = 128, .height = 1, .depth = 1 });
        e.barrier();
    }

    /// prepareRows in the barrier interval of the counted decoded projections it reads (no barrier before it): each group
    /// waits until `counter`'s first word reaches `wait` (Kernels.quantCounted). Only the tf_delta_gdn_pre_rows path waits,
    /// so any other shape is refused rather than run unsynchronised.
    pub fn prepareRowsAfter(r: Round, e: metal.ComputeEncoder, rows: metal.Pipeline, counter: metal.Buffer, wait: u32) !void {
        if (r.p.rows == 0 or r.p.dk != 128 or r.p.dv != 128 or r.p.taps != 4) return error.BadGdnPrepare;
        inline for (.{ r.b.qkv, r.b.history, r.b.query, r.b.key, r.b.value }) |s| if (s.offset % 8 != 0) return error.BadGdnPrepare;
        e.setPipeline(rows);
        bind(e, .{ r.b.qkv, r.b.a, r.b.b, r.b.history, r.b.conv_weight, r.b.a_log, r.b.dt, r.b.windows, r.b.row_slots }, 0);
        e.setValue(r.p, 9);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing }, 10);
        e.setBuffer(counter, 0, 15);
        e.setValue(wait, 16);
        e.dispatchGroups(.{ .width = 1, .height = 2 * r.p.nk + r.p.nv, .depth = (r.p.rows + 15) / 16 }, .{ .width = 128, .height = 1, .depth = 1 });
        e.barrier();
    }

    /// One-row segments kept in place: tf_delta_gdn_pre_step writes each slot's next conv window into its history.
    pub fn prepareStep(r: Round, e: metal.ComputeEncoder, step: metal.Pipeline) void {
        e.setPipeline(step);
        bind(e, .{ r.b.qkv, r.b.a, r.b.b, r.b.history, r.b.conv_weight, r.b.a_log, r.b.dt, r.b.windows, r.b.row_slots }, 0);
        e.setValue(r.p, 9);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing }, 10);
        const wide = r.p.dk == 128 and r.p.dv == 128; // four simdgroups a head (tf_delta_gdn_pre_step4)
        e.dispatchGroups(.{ .width = 1, .height = 2 * r.p.nk + r.p.nv, .depth = r.p.rows }, .{ .width = if (wide) 128 else 32, .height = 1, .depth = 1 });
        e.barrier();
    }

    pub fn usesSlots(r: Round) bool {
        return r.p.dk == 128 and r.compact and r.tree_mode != .snapshot;
    }

    /// forward, a chain's one-slot tree (tf_delta_gdn_tree_pair) taking two value columns a simdgroup.
    pub fn forwardPair(r: Round, e: metal.ComputeEncoder, pair: metal.Pipeline, post_rows: metal.Pipeline, beside: ?Beside) !void {
        if (!r.usesSlots() or r.slot_kernel != 0 or r.p.dv % 2 != 0) return r.forward(e, post_rows, beside);
        e.setPipeline(pair);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state, r.b.parents, r.b.segments }, 0);
        e.setValue(r.p, 8);
        bind(e, .{ r.b.recurrent, r.b.snapshots }, 9);
        r.b.slot_steps.bind(e, 11);
        r.b.state.bind(e, 12);
        (r.b.before orelse r.b.state).bind(e, 13);
        e.dispatchThreads(.{ .width = 32, .height = r.p.dv / 2, .depth = r.p.nv * r.segments }, .{ .width = 32, .height = 4, .depth = 1 });
        if (beside) |b| try b.encode(b.ptr, e);
        e.barrier();
        r.gate(e, post_rows);
    }

    /// `post_rows`: tf_delta_gdn_post_rows, as `gate` takes it.
    pub fn forward(r: Round, e: metal.ComputeEncoder, post_rows: metal.Pipeline, beside: ?Beside) !void {
        e.setPipeline(if (r.usesSlots()) r.pipelines.tree_slots[r.slot_kernel] else r.pipelines.tree);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state, r.b.parents, r.b.segments }, 0);
        e.setValue(r.p, 8);
        bind(e, .{ r.b.recurrent, r.b.snapshots }, 9);
        if (r.usesSlots()) {
            r.b.slot_steps.bind(e, 11);
            r.b.state.bind(e, 12);
            (r.b.before orelse r.b.state).bind(e, 13);
        }
        r.stateGrid(e);
        if (beside) |b| try b.encode(b.ptr, e);
        e.barrier();
        r.gate(e, post_rows);
    }

    /// forward for a depth-one star (`star`): tf_delta_gdn_star_pair, two value columns a simdgroup, each node of each
    /// column tf_delta_gdn_tree_slots' arithmetic from the same parent state.
    /// `counter` (two words, left zero): with nothing beside the scan and the gate on post_rows, the scan counts its
    /// threadgroups once their rows are stored and the gate waits on that count in the same barrier interval.
    /// `pending` (one window): `star` is tf_delta_gdn_star_pending, which applies the pending rows before node 0.
    pub fn forwardStar(r: Round, e: metal.ComputeEncoder, star: metal.Pipeline, post_rows: metal.Pipeline, beside: ?Beside, counter: ?metal.Buffer, pending: ?Pending) !void {
        if (!r.star or !r.usesSlots() or (pending != null and r.segments != 1)) return error.BadGdnPlan;
        const counted = counter != null and beside == null and r.gateRows() and r.p.dv % 8 == 0;
        e.setPipeline(star);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state }, 0);
        r.b.segments.bind(e, 7);
        e.setValue(r.p, 8);
        r.b.recurrent.bind(e, 9);
        if (counted) e.setBuffer(counter.?, 0, 10) else r.b.recurrent.bind(e, 10); // never written: count_out 0
        e.setValue(@as(u32, @intFromBool(counted)), 11);
        if (pending) |q| {
            bind(e, .{ q.key, q.value, q.decay, q.mixing }, 12);
            e.setValue(q.use, 16);
        }
        e.dispatchThreads(.{ .width = 32, .height = r.p.dv / 2, .depth = r.p.nv * r.segments }, .{ .width = 32, .height = 4, .depth = 1 });
        if (beside) |b| try b.encode(b.ptr, e);
        if (counted) return r.gateAfter(e, post_rows, counter.?, @intCast((r.p.dv / 8) * r.p.nv * r.segments));
        e.barrier();
        r.gate(e, post_rows);
    }

    pub fn forwardChain(r: Round, e: metal.ComputeEncoder, chain: metal.Pipeline, post_rows: metal.Pipeline) !void {
        try r.scanChain(e, chain, false, null);
        r.gate(e, post_rows);
        r.commitConv(e);
    }
    /// `wide`: tf_delta_gdn_chain_wide's grid, 16 value columns a simdgroup (prompt chunks, dk = 128).
    pub fn scanChain(r: Round, e: metal.ComputeEncoder, chain: metal.Pipeline, wide: bool, beside: ?Beside) !void {
        e.setPipeline(chain);
        bind(e, .{ r.b.query, r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state, r.b.segments }, 0);
        e.setValue(r.p, 7);
        bind(e, .{ r.b.recurrent, r.b.next_state }, 8);
        // two simdgroups a threadgroup: twice the groups, so the cores' shares of them come out more even
        if (wide) e.dispatchThreads(.{ .width = 32, .height = r.p.dv / 16, .depth = r.p.nv * r.segments }, .{ .width = 32, .height = 2, .depth = 1 }) else r.stateGrid(e);
        if (beside) |b| try b.encode(b.ptr, e);
        e.barrier();
    }
    /// tf_delta_gdn_post a simdgroup a (row, head); 128-wide heads on bfloat4 rows take `rows` (tf_delta_gdn_post_rows),
    /// 4 rows a group of 128.
    /// Whether the gate takes tf_delta_gdn_post_rows (128-wide heads on bfloat4 rows).
    pub fn gateRows(r: Round) bool {
        var wide = r.p.dv == 128 and r.p.zba_stride % 4 == 0;
        inline for (.{ r.b.recurrent, r.b.z, r.b.output }) |s| wide = wide and s.offset % 8 == 0; // bfloat4 rows
        return wide;
    }
    /// gate on post_rows in the star scan's barrier interval: each group waits until `counter`'s first word reaches `wait`
    /// (the scan's threadgroups) before it reads the scan's rows; the last group leaves both words zero.
    pub fn gateAfter(r: Round, e: metal.ComputeEncoder, rows: metal.Pipeline, counter: metal.Buffer, wait: u32) void {
        e.setPipeline(rows);
        bind(e, .{ r.b.recurrent, r.b.z, r.b.norm }, 0);
        e.setValue(r.p, 3);
        r.b.output.bind(e, 4);
        if (r.sums) |w| {
            w.span.bind(e, 5);
            e.setValue(w.stride, 6);
            w.written.* = true;
        } else {
            r.b.output.bind(e, 5); // never read: a zero stride writes no sums
            e.setValue(@as(u32, 0), 6);
        }
        e.setBuffer(counter, 0, 7);
        e.setValue(wait, 8);
        e.dispatchGroups(.{ .width = 1, .height = r.p.nv, .depth = (r.p.rows + 3) / 4 }, .{ .width = 128 });
        e.barrier();
    }
    pub fn gate(r: Round, e: metal.ComputeEncoder, rows: metal.Pipeline) void {
        const wide = r.gateRows();
        e.setPipeline(if (wide) rows else r.pipelines.post);
        bind(e, .{ r.b.recurrent, r.b.z, r.b.norm }, 0);
        e.setValue(r.p, 3);
        r.b.output.bind(e, 4);
        if (wide) if (r.sums) |w| {
            w.span.bind(e, 5);
            e.setValue(w.stride, 6);
            w.written.* = true;
        } else {
            r.b.output.bind(e, 5); // never read: a zero stride writes no sums
            e.setValue(@as(u32, 0), 6);
        };
        if (wide) {
            r.b.output.bind(e, 7); // never read: wait_for 0
            e.setValue(@as(u32, 0), 8);
        }
        e.dispatchGroups(.{ .width = 1, .height = r.p.nv, .depth = if (wide) (r.p.rows + 3) / 4 else r.p.rows }, .{ .width = if (wide) 128 else 32 });
        e.barrier();
    }
    /// gate for a prompt chain on `rows16` (tf_delta_gdn_post_rows16: post_rows' words and group sums, 16 rows a group of
    /// 128); shapes post_rows does not take keep `gate`.
    pub fn gatePrompt(r: Round, e: metal.ComputeEncoder, rows16: metal.Pipeline, post_rows: metal.Pipeline) void {
        if (!r.gateRows()) return r.gate(e, post_rows);
        e.setPipeline(rows16);
        bind(e, .{ r.b.recurrent, r.b.z, r.b.norm }, 0);
        e.setValue(r.p, 3);
        r.b.output.bind(e, 4);
        if (r.sums) |w| {
            w.span.bind(e, 5);
            e.setValue(w.stride, 6);
            w.written.* = true;
        } else {
            r.b.output.bind(e, 5); // never read: a zero stride writes no sums
            e.setValue(@as(u32, 0), 6);
        }
        e.dispatchGroups(.{ .width = 1, .height = r.p.nv, .depth = (r.p.rows + 15) / 16 }, .{ .width = 128 });
        e.barrier();
    }
    pub fn commitConv(r: Round, e: metal.ComputeEncoder) void {
        r.commitConvBeside(e);
        e.barrier();
    }

    /// commitConv without the barrier after it, for a caller that encodes it beside work that neither reads its next
    /// windows nor writes what it reads (the prompt chain's scan).
    pub fn commitConvBeside(r: Round, e: metal.ComputeEncoder) void {
        e.setPipeline(r.pipelines.conv_commit);
        bind(e, .{ r.b.history, r.b.qkv, r.b.kept_rows, r.b.keeps, r.b.windows }, 0);
        e.setValue(r.p, 5);
        r.b.next_history.bind(e, 6);
        e.dispatchThreads(.{ .width = r.conv_width, .height = r.segments }, .{ .width = 256 }); // a word a thread
    }

    pub fn commit(r: Round, e: metal.ComputeEncoder) void {
        e.setPipeline(r.pipelines.replay);
        bind(e, .{ r.b.key, r.b.value, r.b.decay, r.b.mixing, r.b.state, r.b.kept_rows, r.b.keeps }, 0);
        e.setValue(r.p, 7);
        r.b.next_state.bind(e, 8);
        r.b.segments.bind(e, 9);
        (r.b.before orelse r.b.state).bind(e, 10);
        r.stateGrid(e);
        e.barrier();
        e.setPipeline(r.pipelines.conv_commit);
        bind(e, .{ r.b.history, r.b.qkv, r.b.kept_rows, r.b.keeps, r.b.windows }, 0);
        e.setValue(r.p, 5);
        r.b.next_history.bind(e, 6);
        e.dispatchThreads(.{ .width = r.conv_width, .height = r.segments, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        e.barrier();
    }

    fn stateGrid(r: Round, e: metal.ComputeEncoder) void {
        e.dispatchThreads(.{ .width = 32, .height = r.p.dv, .depth = r.p.nv * r.segments }, .{ .width = 32, .height = 4, .depth = 1 });
    }
};

pub fn bind(e: metal.ComputeEncoder, buffers: anytype, start: usize) void {
    inline for (buffers, 0..) |buffer, index| buffer.bind(e, start + index);
}
