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
    /// A split GDN layer's stages (Stages.splitLinear), each without a barrier; `begin` answers whether the layer splits.
    gdn_stage: ?*const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Gdn, *Frame, u32, GdnStage) anyerror!bool = null,
};

/// Rows [first, first + rows) of a pass.
pub const Half = struct { first: u32, rows: u32 };
pub const GdnStage = union(enum) { begin: u32, project: Half, prepare: Half, scan: Half, gate: Half, publish };

/// Where a prompt pass of `rows` rows splits its GDN layers (Stages.split_first): a multiple of the 128-row prompt
/// launches, so each half's projections are launches the whole pass makes; 0 below two launches.
pub fn splitRow(rows: u32) u32 {
    return if (rows < 256) 0 else 128 * (rows / 256);
}

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
    /// Nonzero: GDN layers run as row halves [0, split_first) and [split_first, rows) (splitLinear).
    split_first: u32 = 0,
    /// The layer's post stage ran inside its split mixer.
    post_done: bool = false,
    pub fn begin(s: *Stages, lane: *@import("core").segments.Lane) !void {
        const m = s.model;
        if (s.rows == 0 or s.rows > m.frame.capacity or lane.k != 0) return error.BadFrameRows;
        if (s.split_first >= s.rows) s.split_first = 0;
        if (s.split_first != 0) {
            // the halves' post-norm row counts beside the pass's (dims words 2-5; frame.setRows writes 0 and 1)
            const dims = m.frame.get(.dims).buffer.slice(i32, 8);
            dims[2] = @intCast(s.split_first);
            dims[3] = dims[1];
            dims[4] = @intCast(s.rows - s.split_first);
            dims[5] = dims[1];
        }
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
        try m.kernels.norm(e, s.hidden, s.pending, wts.ref(layer.norm), f.get(.eps), f.get(.dims), s.residual, f.get(.input), f.get(.sums), h, s.rows);
        e.barrier();
        if (s.pending != null) {
            std.mem.swap(Ref, &s.hidden, &s.residual);
            if (s.hooks.tap) |tap| try tap(s.hooks.ptr, e, index - 1, s.hidden, s.rows);
        }
    }
    pub fn mixer(s: *Stages, lane: *@import("core").segments.Lane, index: usize) !void {
        const m = s.model;
        switch (m.weights.layers[index].mixer) {
            .linear => |g| {
                if (try s.splits(lane.enc, index, g)) return s.splitLinear(lane.enc, index, g);
                try s.hooks.gdn(s.hooks.ptr, lane.enc, index, g, &m.frame, s.rows);
            },
            .attention => |a| try s.hooks.attention(s.hooks.ptr, lane.enc, index, a, &m.frame, s.rows),
        }
    }

    fn splits(s: *Stages, e: mtl.ComputeEncoder, index: usize, g: wts.Gdn) !bool {
        if (s.split_first == 0) return false;
        const stage = s.hooks.gdn_stage orelse return false;
        const m = s.model;
        if (s.terminal_rows and index + 1 == m.config.layers) return false;
        if (!@import("quant_gpu.zig").Kernels.fusesPromptGateUp(m.weights.layers[index].gu)) return false;
        return stage(s.hooks.ptr, e, index, g, &m.frame, s.rows, .{ .begin = s.split_first });
    }

    /// A GDN layer as row halves A [0, split_first) and B [split_first, rows): each barrier interval holds a stage of
    /// each half, so A's scan runs beside B's projections, B's scan beside A's out projection and A's gate/up beside B's
    /// out projection. Each stage is `mixer`'s or `post`'s over its half's rows: a projection's 128-row launches are the
    /// whole pass's (split_first is a multiple of 128), B's scan continues A's state, and the per-row kernels see the same
    /// words, so every row's output is the unsplit layer's.
    fn splitLinear(s: *Stages, e: mtl.ComputeEncoder, index: usize, g: wts.Gdn) !void {
        const m = s.model;
        const f = &m.frame;
        const stage = s.hooks.gdn_stage.?;
        const ptr = s.hooks.ptr;
        const layer = m.weights.layers[index];
        const a = Half{ .first = 0, .rows = s.split_first };
        const b = Half{ .first = s.split_first, .rows = s.rows - s.split_first };
        _ = try stage(ptr, e, index, g, f, s.rows, .{ .project = a });
        e.barrier();
        _ = try stage(ptr, e, index, g, f, s.rows, .{ .prepare = a });
        e.barrier();
        _ = try stage(ptr, e, index, g, f, s.rows, .{ .scan = a });
        _ = try stage(ptr, e, index, g, f, s.rows, .{ .project = b });
        e.barrier();
        _ = try stage(ptr, e, index, g, f, s.rows, .{ .gate = a });
        _ = try stage(ptr, e, index, g, f, s.rows, .{ .prepare = b });
        e.barrier();
        _ = try stage(ptr, e, index, g, f, s.rows, .{ .scan = b });
        try s.outHalf(e, g, a);
        e.barrier();
        _ = try stage(ptr, e, index, g, f, s.rows, .{ .gate = b });
        try s.normHalf(e, layer, a);
        e.barrier();
        try s.gateUpHalf(e, layer, a);
        try s.outHalf(e, g, b);
        _ = try stage(ptr, e, index, g, f, s.rows, .publish);
        e.barrier();
        try s.downHalf(e, layer, a);
        try s.normHalf(e, layer, b);
        e.barrier();
        try s.gateUpHalf(e, layer, b);
        e.barrier();
        try s.downHalf(e, layer, b);
        e.barrier();
        std.mem.swap(Ref, &s.hidden, &s.residual);
        s.pending = f.get(.output);
        f.mixer_sums = false;
        s.post_done = true;
    }

    /// `post`'s out projection over half `h`, reading the group sums its gate wrote.
    fn outHalf(s: *Stages, e: mtl.ComputeEncoder, g: wts.Gdn, h: Half) !void {
        const f = &s.model.frame;
        try s.model.kernels.quant(e, g.out, rowsAt(f.get(.mixer_out), h.first, g.out.k), sumsAt(f, .sums, s.rows, h.first), f.get(.dims), rowsAt(f.get(.output), h.first, g.out.n), h.rows);
    }

    /// `post`'s norm over half `h`: its rows of hidden + output into residual, input and group sums.
    fn normHalf(s: *Stages, e: mtl.ComputeEncoder, layer: wts.Layer, h: Half) !void {
        const m = s.model;
        const f = &m.frame;
        const width: u32 = @intCast(m.config.hidden);
        const dims = Ref{ .buffer = f.get(.dims).buffer, .offset = if (h.first == 0) 8 else 16 }; // begin's words
        var xs = f.get(.sums);
        xs.offset += @as(usize, h.first) * 4;
        try m.kernels.norm(e, rowsAt(s.hidden, h.first, width), rowsAt(f.get(.output), h.first, width), wts.ref(layer.post_norm), f.get(.eps), dims, rowsAt(s.residual, h.first, width), rowsAt(f.get(.input), h.first, width), xs, width, h.rows);
    }

    fn gateUpHalf(s: *Stages, e: mtl.ComputeEncoder, layer: wts.Layer, h: Half) !void {
        const f = &s.model.frame;
        var act_sums = f.get(.act_sums);
        act_sums.offset += @as(usize, h.first) * 4;
        if (!try s.model.kernels.gateUpActivated(e, layer.gu, rowsAt(f.get(.input), h.first, layer.gu.k), sumsAt(f, .sums, s.rows, h.first), rowsAt(f.get(.activated), h.first, layer.gu.n / 2), act_sums, h.rows)) return error.BadSplitLayer;
    }

    fn downHalf(s: *Stages, e: mtl.ComputeEncoder, layer: wts.Layer, h: Half) !void {
        const f = &s.model.frame;
        try s.model.kernels.quant(e, layer.down, rowsAt(f.get(.activated), h.first, layer.down.k), sumsAt(f, .act_sums, s.rows, h.first), f.get(.dims), rowsAt(f.get(.output), h.first, layer.down.n), h.rows);
    }

    pub fn post(s: *Stages, lane: *@import("core").segments.Lane, index: usize) !void {
        if (s.post_done) {
            s.post_done = false;
            return;
        }
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        const layer = m.weights.layers[index];
        const h: u32 = @intCast(m.config.hidden);
        const out = switch (layer.mixer) {
            .linear => |g| g.out,
            .attention => |a| a.out,
        };
        // read and cleared before any return, and used only by the block whose GDN gate wrote it
        const gated = f.mixer_sums and layer.mixer == .linear;
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
        // a GDN gate's group sums of mixer_out (its rows, uncompacted) spare the projection its own sums pass
        try m.kernels.quant(e, out, mixer_out, if (gated and !s.compact) sums(f, .sums, s.rows) else null, dims, f.get(.output), s.rows);
        e.barrier();
        try m.kernels.norm(e, s.hidden, f.get(.output), wts.ref(layer.post_norm), f.get(.eps), dims, s.residual, f.get(.input), f.get(.sums), h, s.rows);
        e.barrier();
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
        } else if (!try m.kernels.gateUpActivated(e, layer.gu, f.get(.input), sums(f, .sums, s.rows), f.get(.activated), f.get(.act_sums), s.rows)) {
            try m.kernels.quant(e, layer.gu, f.get(.input), sums(f, .sums, s.rows), dims, f.get(.mixed), s.rows);
            e.barrier();
            try m.kernels.mlp(e, f.get(.mixed), up, dims, f.get(.activated), f.get(.act_sums), width, s.rows);
        }
        lane.enc.barrier();
        try m.kernels.quant(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), dims, f.get(.output), s.rows);
        lane.enc.barrier();
        s.pending = f.get(.output);
    }
    pub fn finish(s: *Stages, lane: *@import("core").segments.Lane) !void {
        if (s.terminal_rows and s.head == .none) return;
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        const dims = f.get(if (s.compact) .last_dims else .dims);
        try m.kernels.norm(e, s.hidden, s.pending, wts.ref(m.weights.norm), f.get(.eps), dims, s.residual, f.get(.input), f.get(.sums), @intCast(m.config.hidden), s.rows);
        e.barrier();
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
            try m.kernels.quant(e, m.weights.head, input, head_sums, f.get(.dims), f.get(.logits), if (s.head == .last) 1 else s.rows);
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

/// `sums` from row `first` on: the same stride, so row i of a launch there is the pass's row first + i.
fn sumsAt(f: *Frame, field: @import("gpu_frame.zig").Field, rows: u32, first: u32) @import("quant_gpu.zig").Sums {
    var at = sums(f, field, rows);
    at.ref.offset += @as(usize, first) * 4;
    return at;
}

/// bf16 rows of `width` values from row `first` on.
fn rowsAt(r: Ref, first: u32, width: usize) Ref {
    return .{ .buffer = r.buffer, .offset = r.offset + @as(usize, first) * width * 2 };
}
