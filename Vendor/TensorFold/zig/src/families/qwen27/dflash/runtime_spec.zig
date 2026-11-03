//! A depth-one round's absorb, done ahead beside its verify: the verify's command buffer ends behind its last layer's
//! block with an event signalled there (forward.Split), and a command buffer on a queue of its own, waiting for that
//! event, runs the early frame's committed-tap launches (fusion projection, hidden norm, K/V projections, the keys' norm
//! and rope) over every row the verify checks, into rows of its own, beside the verify's final norm, head and picks.
//! The kept rows are only known from the verify's picks; the early frame then copies those rows' keys and values
//! (Spec.gather, beside its RoundArgs select) instead of computing them, and appends them as before. Row 0 (the pending
//! token's) always sits at the committed end and any kept row past it one position later, so every row's rope position
//! is known before the picks.
//!
//! Every launch is the early frame's own for its count of rows (the same pipelines, grids, K splits and arithmetic, the
//! rows' count the only difference): a row's words do not depend on the other rows or their count (lane_projection's
//! rows8 reg kernel and simd_qmm's 8-row tiles compute each row alone, its group sums alone; norms and ropes run a row and
//! head a threadgroup), so each kept row's keys and values are the words the frame computes for it.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const cfg = @import("config.zig");
const weights = @import("weights.zig");
const Backend = @import("runtime_backend.zig").Backend;
const Glue = @import("../glue_gpu.zig").Kernels;

/// Rows a verify of a depth-one round checks at most: the pending token and its children.
pub const rows_max = 8;

/// The verify's tap planes: plane j's rows are layer j's taps of the verify's rows (taps.zig).
pub const Planes = struct { buffer: mtl.Buffer, capacity: u32, width: u32 };

pub const Spec = struct {
    queue: mtl.Queue,
    /// Signalled at the end of the verify's command buffer that holds its last layer's block (forward.Split).
    taps_in: mtl.SharedEvent,
    /// Signalled behind the absorb's last launch: the early frame's command buffer waits for it.
    done: mtl.SharedEvent,
    /// The last value reserved (one a verify, whether or not an absorb follows it: no wait ever sees an old signal), and
    /// the last one an absorb was committed for.
    value: u64 = 0,
    committed: u64 = 0,
    /// The last absorb's command buffer, retained until the next one (or deinit) checks it.
    last: ?mtl.CommandBuffer = null,
    /// The K/V projection rows a layer's values sit in: the fused K+V rows with tensor units (values at `voff`), else
    /// the values' own rows.
    vw: u32,
    voff: u32,
    /// The verify's rows' taps as the ring holds a row: [rows_max][tap width].
    taps: mtl.Buffer,
    fusion: mtl.Buffer,
    hidden: mtl.Buffer,
    fusion_sums: mtl.Buffer,
    kv_sums: mtl.Buffer,
    /// [layers][rows_max][vw]: each layer's K+V rows (tensor units) or values (without).
    kv: mtl.Buffer,
    /// [layers][rows_max][kv width]: the keys before their norm (without tensor units), normed, roped.
    keys: mtl.Buffer,
    normed: mtl.Buffer,
    roped: mtl.Buffer,
    positions: mtl.Buffer,
    identity: mtl.Buffer,
    /// The early frame's two rows (kept, in order): [2][layers][kv width] keys and [2][layers][vw] values.
    kept_keys: mtl.Buffer,
    kept_values: mtl.Buffer,

    pub fn init(device: mtl.Device, c: cfg.Config, units: bool) !Spec {
        const kvw: usize = c.kvWidth();
        const layers: usize = c.layers;
        const vw: u32 = if (units) 2 * c.kvWidth() else c.kvWidth();
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const sizes = [_]usize{
            rows_max * @as(usize, c.tapWidth()) * 2,
            rows_max * @as(usize, c.hidden) * 2,
            rows_max * @as(usize, c.hidden) * 2,
            @as(usize, c.tapWidth() / 64) * 16 * 4,
            @as(usize, c.hidden / 64) * 16 * 4,
            layers * rows_max * vw * 2,
            layers * rows_max * kvw * 2,
            layers * rows_max * kvw * 2,
            layers * rows_max * kvw * 2,
            256,
            256,
            2 * layers * kvw * 2,
            2 * layers * vw * 2,
        };
        var bufs: [sizes.len]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |b| b.deinit();
        while (made < sizes.len) : (made += 1) bufs[made] = try device.buffer(sizes[made], opts);
        for (bufs[10].slice(u32, rows_max), 0..) |*row, i| row.* = @intCast(i);
        const queue = try device.queue();
        errdefer queue.deinit();
        const taps_in = try device.sharedEvent();
        errdefer taps_in.deinit();
        const done = try device.sharedEvent();
        return .{ .queue = queue, .taps_in = taps_in, .done = done, .vw = vw, .voff = vw - c.kvWidth(), .taps = bufs[0], .fusion = bufs[1], .hidden = bufs[2], .fusion_sums = bufs[3], .kv_sums = bufs[4], .kv = bufs[5], .keys = bufs[6], .normed = bufs[7], .roped = bufs[8], .positions = bufs[9], .identity = bufs[10], .kept_keys = bufs[11], .kept_values = bufs[12] };
    }

    pub fn deinit(s: *Spec) void {
        s.release(s.committed);
        if (s.last) |cb| {
            cb.wait();
            cb.release();
        }
        s.last = null;
        inline for (.{ s.taps, s.fusion, s.hidden, s.fusion_sums, s.kv_sums, s.kv, s.keys, s.normed, s.roped, s.positions, s.identity, s.kept_keys, s.kept_values }) |b| b.deinit();
        s.done.deinit();
        s.taps_in.deinit();
        s.queue.deinit();
    }

    /// A new value for a verify to signal `taps_in` at, and its absorb's events.
    pub fn reserve(s: *Spec) u64 {
        s.value += 1;
        return s.value;
    }

    /// A committed absorb whose verify may never signal (an error path): its wait is let through from the host, so its
    /// command buffer and any frame waiting on it run out; what it writes is then never read.
    pub fn release(s: *Spec, value: u64) void {
        if (value != 0 and s.taps_in.value() < value) s.taps_in.set(value);
    }

    /// Encodes and commits the absorb of the verify's `rows` rows (their taps in `planes`), the rows that follow the
    /// committed taps ending at `end`, behind `taps_in` reaching `value` (the last reserved); `done` takes it at the end.
    /// An error before the commit leaves `committed` as it was; one after it (the absorb runs, its rows unread) sets it.
    pub fn commit(s: *Spec, b: *Backend, g: *const weights.Graph, glue: Glue, planes: Planes, rows: u32, end: u64, value: u64) !void {
        const c = g.config;
        if (value != s.value or value <= s.committed or rows == 0 or rows > rows_max or rows > planes.capacity or planes.width != c.hidden) return error.DraftDeviceRound;
        // the last absorb ran before its frame, which completed before this verify was encoded (an absorb nothing
        // waited for: its verify ran too, or an error let it through)
        if (s.last) |cb| {
            s.last = null;
            defer cb.release();
            s.release(s.committed);
            cb.wait();
            if (cb.failure() != null) return error.DraftGpuFailure;
        }
        const positions = s.positions.slice(u64, rows_max);
        for (positions, 0..) |*p, i| p.* = end + @intFromBool(i > 0);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = s.queue.commandBuffer();
        cb.waitFor(s.taps_in, value);
        const e = cb.compute(.concurrent);
        // an encode error still commits what it holds, so the queue's slot and its wait are never left behind
        const encoded = encode(s, b, g, glue, planes, e, rows);
        e.end();
        cb.signal(s.done, value);
        cb.commit();
        s.committed = value;
        s.last = cb.retain();
        return encoded;
    }

    /// The early frame's launches of encodeAppendTo up to its appends, for `rows` rows, each stage behind a barrier.
    fn encode(s: *Spec, b: *Backend, g: *const weights.Graph, glue: Glue, planes: Planes, e: mtl.ComputeEncoder, rows: u32) !void {
        const c = g.config;
        const kvw = c.kvWidth();
        const layers = g.layers.len;
        if (layers > 8) return error.DraftDeviceRound;
        // the verify's rows' taps, as the picks' gather lays a kept row into the ring
        try glue.tapRows(e, .{ .buffer = planes.buffer }, .{ .buffer = s.identity }, .{ .buffer = s.taps }, .{ .rows = rows, .width = planes.width, .capacity = planes.capacity, .planes = 5 });
        e.barrier();
        const plane_bytes = @as(usize, rows_max) * kvw * 2;
        const vplane_bytes = @as(usize, rows_max) * s.vw * 2;
        if (b.simd) |*pipes| {
            // no tensor units: simd_qmm tiles, as Backend.project takes them (an early frame's Args name its rows)
            const sq = core.row_projection.simd;
            const fusion = b.tiled.get(@intFromPtr(g.fusion.bytes.ptr)) orelse return error.DraftPreparationUnqualified;
            sq.encodeArgs(e, try sq.call(pipes, fusion.n, fusion.k, rows, false, false), fusion.matrix(), s.taps, 0, s.fusion, 0, null);
            e.barrier();
            try b.primitives.norm(e, .{ .buf = s.fusion }, try b.weight(g.hidden_norm), .{ .buf = s.hidden }, .{ .rows = rows, .heads = 1, .dim = c.hidden, .x_stride = c.hidden, .y_stride = c.hidden, .eps = c.eps });
            e.barrier();
            for (g.layers, 0..) |layer, i| {
                const k = b.tiled.get(@intFromPtr(layer.attention.k.bytes.ptr)) orelse return error.DraftPreparationUnqualified;
                const v = b.tiled.get(@intFromPtr(layer.attention.v.bytes.ptr)) orelse return error.DraftPreparationUnqualified;
                sq.encodeArgs(e, try sq.call(pipes, k.n, k.k, rows, false, false), k.matrix(), s.hidden, 0, s.keys, i * plane_bytes, null);
                sq.encodeArgs(e, try sq.call(pipes, v.n, v.k, rows, false, false), v.matrix(), s.hidden, 0, s.kv, i * vplane_bytes, null);
            }
        } else {
            // tensor units: the prepared lane adapters, their input group sums first, as encodePrepared's cacheable path
            const fkey = @intFromPtr(g.fusion.bytes.ptr);
            const fusion = (if (b.precise) b.prepared.get(fkey | Backend.precise_key) else null) orelse b.prepared.get(fkey) orelse return error.DraftPreparationUnqualified;
            const fp = fusion.forRows(rows);
            const fsums = try fp.encodeSumsInto(e, .{ .buf = s.taps }, rows, .{ .buf = s.fusion_sums });
            e.barrier();
            try fp.encodeSumsAt(e, .{ .buf = s.taps }, .{ .buf = fusion.words }, fusion.meta(), fsums, .{ .buf = s.fusion }, rows, null);
            e.barrier();
            try b.primitives.norm(e, .{ .buf = s.fusion }, try b.weight(g.hidden_norm), .{ .buf = s.hidden }, .{ .rows = rows, .heads = 1, .dim = c.hidden, .x_stride = c.hidden, .y_stride = c.hidden, .eps = c.eps });
            e.barrier();
            var ksums: ?core.lane_projection.Sums = null;
            for (g.layers, 0..) |layer, i| {
                const base = @intFromPtr(layer.attention.k.bytes.ptr);
                const kv = (if (b.precise) b.prepared.get((base | Backend.precise_key) + @import("runtime_backend.zig").group_kv) else null) orelse b.prepared.get(base + @import("runtime_backend.zig").group_kv) orelse return error.DraftPreparationUnqualified;
                const p = kv.forRows(rows);
                if (p.n != s.vw) return error.DraftDeviceRound;
                // one sums pass for every layer's K+V projection (they share the hidden rows), as the frame reuses them
                if (ksums == null) {
                    ksums = try p.encodeSumsInto(e, .{ .buf = s.hidden }, rows, .{ .buf = s.kv_sums });
                    e.barrier();
                }
                try p.encodeSumsAt(e, .{ .buf = s.hidden }, .{ .buf = kv.words }, kv.meta(), ksums.?, .{ .buf = s.kv, .off = i * vplane_bytes }, rows, null);
            }
        }
        e.barrier();
        // each layer's keys: the per-head norm and the rope in one launch, at the rows' positions
        const kstride: u32 = if (b.simd != null) kvw else s.vw;
        for (g.layers, 0..) |layer, i| {
            const raw: core.draft_ops.Ref = if (b.simd != null) .{ .buf = s.keys, .off = i * plane_bytes } else .{ .buf = s.kv, .off = i * vplane_bytes };
            const np: core.draft_ops.Norm = .{ .rows = rows, .heads = c.kv_heads, .dim = c.head_dim, .x_stride = kstride, .y_stride = kvw, .eps = c.eps };
            const rp: core.draft_ops.Rope = .{ .rows = rows, .heads = c.kv_heads, .dim = c.head_dim, .x_stride = kvw, .y_stride = kvw, .theta = @floatCast(c.theta) };
            try b.primitives.normRope(e, raw, try b.weight(layer.attention.k_norm), .{ .buf = s.normed, .off = i * plane_bytes }, .{ .buf = s.positions }, .{ .buf = s.roped, .off = i * plane_bytes }, np, rp);
        }
    }

    /// In the early frame's first interval, beside its RoundArgs select: the kept rows' keys and values (the verify's
    /// result names them, at most 2 of its `rows`) into `kept_keys` and `kept_values`, in kept order.
    pub fn gather(s: *const Spec, e: mtl.ComputeEncoder, ops: core.tree_commit_gpu.Ops, result: core.tree_commit_gpu.Ref, rows: u32, c: cfg.Config) !void {
        const room = @min(2, rows);
        try ops.tapsInto(e, result, .{ .buf = s.roped }, .{ .buf = s.kept_keys }, .{ .rows = rows, .width = c.kvWidth(), .capacity = rows_max, .planes = c.layers }, room);
        try ops.tapsInto(e, result, .{ .buf = s.kv }, .{ .buf = s.kept_values }, .{ .rows = rows, .width = s.vw, .capacity = rows_max, .planes = c.layers }, room);
    }
};
