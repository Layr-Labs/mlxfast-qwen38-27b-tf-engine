//! The layer program preserves Python's pending residual and only exposes post-block native tap rows.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const wts = @import("gpu_weights.zig");
const Config = @import("config.zig").Config;
const Ref = @import("projection.zig").Ref;
const Frame = @import("gpu_frame.zig").Frame;
const ane_mlp = @import("ane_mlp.zig");

pub const Mixer = struct {
    ptr: *anyopaque,
    gdn: *const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Gdn, *Frame, u32) anyerror!void,
    attention: *const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Attention, *Frame, u32) anyerror!void,
    tap: ?*const fn (*anyopaque, mtl.ComputeEncoder, usize, Ref, u32) anyerror!void = null,
};

/// Serial decode and prompt segments own separate frames and order shared caches at the mixer.
pub const Head = enum { all, last, none };
/// Work a round encodes behind the head in the forward's last command buffer (its greedy picks): no buffer of its own.
pub const Tail = struct {
    ptr: *anyopaque,
    encode: *const fn (*anyopaque, mtl.ComputeEncoder) anyerror!void,
};
/// Work committed right after the forward's last command buffer, before the host waits on it (a round's keep).
pub const Hook = struct {
    ptr: *anyopaque,
    run: *const fn (*anyopaque) anyerror!void,
};
pub const Stages = struct {
    model: *Model,
    rows: u32,
    hooks: Mixer,
    pending: ?Ref = null,
    /// The previous block's MLP down projection stored hidden + its output into `residual` (Kernels.quantResid): the next
    /// norm reads `residual` alone.
    summed: bool = false,
    /// Decoded rows: the MLP down was a quantCounted launch with no barrier after it, so the next norm (the next pre, or
    /// finish) waits on its `wait` counted threadgroups in that barrier interval (Kernels.normAfter).
    norm_wait: u32 = 0,
    hidden: Ref = undefined,
    residual: Ref = undefined,
    head: Head = .all,
    tail: ?Tail = null,
    /// Work encoded first in the forward's first command buffer, beside its embedding (a draft's bridge stop).
    lead: ?Tail = null,
    after: ?Hook = null,
    /// No consumer needs the final block's other rows after its cache updates.
    terminal_rows: bool = false,
    compact: bool = false,
    /// An image prompt's rows: the frame's `takes` rows take theirs over their placeholders' embeddings.
    images: ?Ref = null,
    pub fn begin(s: *Stages, lane: *@import("core").segments.Lane) !void {
        const m = s.model;
        if (s.rows == 0 or s.rows > m.frame.capacity or lane.k != 0) return error.BadFrameRows;
        s.hidden = m.frame.get(.hidden);
        s.residual = m.frame.get(.residual);
        if (s.lead) |t| try t.encode(t.ptr, lane.enc);
        try m.glue.embedding(lane.enc, m.config, m.weights.embed, m.frame.get(.ids), s.hidden, s.rows);
        lane.enc.barrier();
        if (s.images) |rows| {
            m.glue.imageRows(lane.enc, s.hidden, rows, m.frame.get(.takes), @intCast(m.config.hidden), s.rows);
            lane.enc.barrier();
        }
    }
    pub fn wait(_: *Stages, _: usize) @import("core").segments.Wait {
        return .mixer;
    }
    pub fn handoff(_: *Stages, _: *@import("core").segments.Lane, _: usize) !void {}
    pub fn pre(s: *Stages, lane: *@import("core").segments.Lane, index: usize) !void {
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        const layer = m.weights.layers[index];
        const h: u32 = @intCast(m.config.hidden);
        if (s.norm_wait != 0) {
            try m.kernels.normAfter(e, s.hidden, s.pending.?, wts.ref(layer.norm), f.get(.eps), f.get(.dims), s.residual, f.get(.input), f.get(.sums), h, s.rows, s.norm_wait);
        } else try m.kernels.norm(e, if (s.summed) s.residual else s.hidden, if (s.summed) null else s.pending, wts.ref(layer.norm), f.get(.eps), f.get(.dims), s.residual, f.get(.input), f.get(.sums), h, s.rows);
        e.barrier();
        s.norm_wait = 0;
        s.summed = false;
        if (s.pending != null) {
            std.mem.swap(Ref, &s.hidden, &s.residual);
            if (s.hooks.tap) |tap| try tap(s.hooks.ptr, e, index - 1, s.hidden, s.rows);
        }
    }
    pub fn mixer(s: *Stages, lane: *@import("core").segments.Lane, index: usize) !void {
        const m = s.model;
        switch (m.weights.layers[index].mixer) {
            .linear => |g| try s.hooks.gdn(s.hooks.ptr, lane.enc, index, g, &m.frame, s.rows),
            .attention => |a| try s.hooks.attention(s.hooks.ptr, lane.enc, index, a, &m.frame, s.rows),
        }
    }
    pub fn post(s: *Stages, lane: *@import("core").segments.Lane, index: usize) !void {
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        const layer = m.weights.layers[index];
        const h: u32 = @intCast(m.config.hidden);
        const out = switch (layer.mixer) {
            .linear => |g| g.out,
            .attention => |a| a.out,
        };
        // read and cleared before any return, and used only by the block whose GDN gate or attention merge wrote it
        const gated = f.mixer_sums;
        f.mixer_sums = false;
        var mixer_out = f.get(.mixer_out);
        if (s.terminal_rows and index + 1 == m.config.layers) {
            if (s.head == .none) return;
            mixer_out.offset += @as(usize, s.rows - 1) * out.k * 2;
            s.hidden.offset += @as(usize, s.rows - 1) * h * 2;
            s.rows = 1;
            s.compact = true;
        }
        const dims = f.get(if (s.compact) .last_dims else .dims);
        // decoded rows whose fused gate/up takes the counted launch: the post-norm counts its rows too, and the gate/up
        // waits on them in the same barrier interval (the count is decided before either is encoded)
        var post_wait: u32 = 0;
        // the mixer's group sums of mixer_out (its rows, uncompacted) spare the projection its own sums pass; prompt rows'
        // two-tile launch stores hidden + its output as the residual stream's next rows, which the norm then reads alone
        if (!s.compact and try m.kernels.quantResid(e, out, mixer_out, if (gated) sums(f, .sums, s.rows) else null, s.hidden, s.residual, s.rows)) {
            e.barrier();
            try m.kernels.norm(e, s.residual, null, wts.ref(layer.post_norm), f.get(.eps), dims, s.residual, f.get(.input), f.get(.sums), h, s.rows);
        } else if (if (gated and !s.compact) try m.kernels.quantCounted(e, out, mixer_out, sums(f, .sums, s.rows), f.get(.output), s.rows) else null) |counted| {
            // decoded rows: the norm shares the projection's barrier interval and waits on its counted threadgroups
            const chain = m.kernels.gateUpCountable(layer.gu, layer.down, s.rows);
            try m.kernels.normAfterCounted(e, s.hidden, f.get(.output), wts.ref(layer.post_norm), f.get(.eps), dims, s.residual, f.get(.input), f.get(.sums), h, s.rows, counted, chain);
            if (chain) post_wait = s.rows;
        } else {
            try m.kernels.quant(e, out, mixer_out, if (gated and !s.compact) sums(f, .sums, s.rows) else null, dims, f.get(.output), s.rows);
            e.barrier();
            try m.kernels.norm(e, s.hidden, f.get(.output), wts.ref(layer.post_norm), f.get(.eps), dims, s.residual, f.get(.input), f.get(.sums), h, s.rows);
        }
        if (post_wait == 0) e.barrier();
        std.mem.swap(Ref, &s.hidden, &s.residual);
        var up = f.get(.mixed);
        up.offset += @as(usize, m.config.intermediate) * 2;
        const width: u32 = @intCast(m.config.intermediate);
        const shared = if (m.ane) |sh| m.kernels.prompt and s.rows <= sh.capacity() else false;
        if (if (shared) m.ane else null) |share| {
            // ANE slices run the shared columns; the GPU adds its remaining columns.
            const used: u32 = (s.rows + ane_mlp.width - 1) / ane_mlp.width;
            const v = share.next(index, used);
            const planes = Ref{ .buffer = share.outputs[0].buffer };
            const own = share.columns / 64;
            const all = width / 64;
            for (0..used) |slice| try m.kernels.aneInput(e, f.get(.input), f.get(.dims), .{ .buffer = share.inputs[slice].buffer }, h, width, share.columns, @intCast(slice * ane_mlp.width));
            try share.signal(lane, v[0]);
            try m.kernels.quantSkip(lane.enc, layer.gu, f.get(.input), sums(f, .sums, s.rows), f.get(.dims), f.get(.mixed), s.rows, share.columns / 32);
            lane.enc.barrier();
            try m.kernels.mlpAne(lane.enc, f.get(.mixed), up, f.get(.dims), f.get(.activated), f.get(.act_sums), planes, h, width, share.columns, s.rows, own, all - own);
            lane.enc.barrier();
            try m.kernels.quantCut(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), f.get(.dims), .{ .buffer = share.partial }, s.rows, .{ own, all - own });
            share.wait(lane, v[1]);
            for (0..used) |slice| {
                const base: u32 = @intCast(slice * ane_mlp.width);
                try m.kernels.addAne(lane.enc, .{ .buffer = share.partial }, .{ .buffer = share.outputs[slice].buffer }, f.get(.output), h, width, share.columns, @min(ane_mlp.width, s.rows - base), base);
            }
            lane.enc.barrier();
            s.pending = f.get(.output);
            return;
        }
        // decoded rows: a counted gate/up shares its barrier interval with the counted down that waits on it
        const gu_wait: ?u32 = if (!s.compact) try m.kernels.gateUpCountedAfter(e, layer.gu, layer.down, f.get(.input), sums(f, .sums, s.rows), f.get(.activated), f.get(.act_sums), s.rows, post_wait) else null;
        if (post_wait != 0 and gu_wait == null) return error.BadCoreProjectionRows; // the post-norm counted for a gate/up that did not wait
        if (gu_wait == null) {
            if (!try m.kernels.gateUpActivated(e, layer.gu, f.get(.input), sums(f, .sums, s.rows), f.get(.activated), f.get(.act_sums), s.rows)) {
                try m.kernels.quant(e, layer.gu, f.get(.input), sums(f, .sums, s.rows), dims, f.get(.mixed), s.rows);
                e.barrier();
                try m.kernels.mlp(e, f.get(.mixed), up, dims, f.get(.activated), f.get(.act_sums), width, s.rows);
            }
            lane.enc.barrier();
        }
        if (gu_wait) |n| {
            s.norm_wait = try m.kernels.quantCountedAfter(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), f.get(.output), s.rows, n);
        } else if (!s.compact and try m.kernels.quantResid(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), s.hidden, s.residual, s.rows)) {
            s.summed = true;
        } else if (if (!s.compact) try m.kernels.quantCounted(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), f.get(.output), s.rows) else null) |counted| {
            s.norm_wait = counted;
        } else try m.kernels.quant(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), dims, f.get(.output), s.rows);
        // a counted down shares its barrier interval with the norm that waits on it
        if (s.norm_wait == 0) lane.enc.barrier();
        s.pending = f.get(.output);
    }
    pub fn finish(s: *Stages, lane: *@import("core").segments.Lane) !void {
        if (s.terminal_rows and s.head == .none) return;
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        const dims = f.get(if (s.compact) .last_dims else .dims);
        if (s.norm_wait != 0) {
            try m.kernels.normAfter(e, s.hidden, s.pending.?, wts.ref(m.weights.norm), f.get(.eps), dims, s.residual, f.get(.input), f.get(.sums), @intCast(m.config.hidden), s.rows, s.norm_wait);
        } else try m.kernels.norm(e, if (s.summed) s.residual else s.hidden, if (s.summed) null else s.pending, wts.ref(m.weights.norm), f.get(.eps), dims, s.residual, f.get(.input), f.get(.sums), @intCast(m.config.hidden), s.rows);
        e.barrier();
        s.norm_wait = 0;
        s.summed = false;
        std.mem.swap(Ref, &s.hidden, &s.residual);
        if (s.hooks.tap) |tap| try tap(s.hooks.ptr, e, m.config.layers - 1, s.hidden, s.rows);
        if (s.head == .all and s.rows > @import("gpu_frame.zig").logit_rows) return error.BadFrameRows;
        if (s.head != .none) {
            var input = f.get(.input);
            var head_sums = sums(f, .sums, s.rows);
            if (s.head == .last) {
                input.offset += @as(usize, s.rows - 1) * m.config.hidden * 2;
                head_sums.ref.offset += @as(usize, s.rows - 1) * 4;
            }
            // a decoded round's rows read the copy whose eight adjacent tiles interleave; a prompt's last row the unbatched one
            const head = if (s.head == .all) m.weights.head_decode else m.weights.head;
            try m.kernels.quant(e, head, input, head_sums, f.get(.dims), f.get(.logits), if (s.head == .last) 1 else s.rows);
        }
        e.barrier();
        if (s.tail) |t| try t.encode(t.ptr, e);
    }
    /// Decoded rows (a verify's, a serial step's): the host polls the forward's last command buffer, its picks gate the round.
    pub fn polls(s: *const Stages) bool {
        return s.rows <= 16;
    }
    pub fn committed(s: *Stages) !void {
        if (s.after) |h| try h.run(h.ptr);
    }
};

pub fn encode(m: *Model, e: mtl.ComputeEncoder, rows: u32, mixer: Mixer) !void {
    var stages = Stages{ .model = m, .rows = rows, .hooks = mixer };
    var lane = @import("core").segments.Lane{ .k = 0, .cb = undefined, .enc = e, .fence = undefined };
    try stages.begin(&lane);
    for (0..m.config.layers) |index| {
        try stages.pre(&lane, index);
        try stages.mixer(&lane, index);
        try stages.post(&lane, index);
    }
    try stages.finish(&lane);
}

/// A norm's or the MLP gate's group sums for the frame's padded rows.
pub fn sums(f: *Frame, field: @import("gpu_frame.zig").Field, rows: u32) @import("quant_gpu.zig").Sums {
    return .{ .ref = f.get(field), .stride = @import("quant_gpu.zig").padded(rows) };
}
