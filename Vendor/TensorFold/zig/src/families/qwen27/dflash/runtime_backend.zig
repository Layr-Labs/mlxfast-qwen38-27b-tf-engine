//! The draft adapter owns workspace and rolling caches while the target owns embedding, head and accepted state.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const op = @import("operators.zig");
const cfg = @import("config.zig");
const ck = @import("../checkpoint.zig");
const Target = @import("../model.zig").Model;
const callbacks = @import("runtime_callbacks.zig");
pub const Head = @import("runtime_head.zig").Head;
pub const Frame = @import("runtime_frame.zig").Frame;
pub const frame_mod = @import("runtime_frame.zig");
pub const Ref = core.draft_ops.Ref;
const caches_impl = @import("caches.zig");
pub const Cache = caches_impl.Pair(mtl.Buffer);
pub fn newCaches(a: std.mem.Allocator, device: mtl.Device, c: cfg.Config, layers: usize) ![]Cache {
    return caches_impl.newCaches(mtl.Buffer, a, device, c, layers);
}
pub fn freeCaches(a: std.mem.Allocator, caches: []Cache) void {
    caches_impl.freeCaches(mtl.Buffer, a, caches);
}
const simd = core.row_projection.simd;
/// Chips without tensor units: a prepared q4 matrix in MLX layout on simdgroup-matrix tiles.
pub const Tiled = struct {
    w: mtl.Buffer,
    scales: mtl.Buffer,
    biases: mtl.Buffer,
    n: usize,
    k: usize,
    pub fn init(device: mtl.Device, p: core.affine4.Prepared) !Tiled {
        const w = try device.buffer(p.words.len * 4, options);
        errdefer w.deinit();
        const scales = try device.buffer(p.scales.len * 2, options);
        errdefer scales.deinit();
        const biases = try device.buffer(p.biases.len * 2, options);
        @memcpy(w.slice(u32, p.words.len), p.words);
        @memcpy(scales.slice(u16, p.scales.len), p.scales);
        @memcpy(biases.slice(u16, p.biases.len), p.biases);
        return .{ .w = w, .scales = scales, .biases = biases, .n = p.n, .k = p.k };
    }
    pub fn deinit(t: Tiled) void {
        t.w.deinit();
        t.scales.deinit();
        t.biases.deinit();
    }
    pub fn matrix(t: Tiled) simd.Matrix {
        return .{ .w = t.w, .scales = t.scales, .biases = t.biases, .n = t.n, .k = t.k };
    }
};
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const Stashed = struct { prepared: core.affine4.Prepared, key: usize };
pub const Stash = std.StringHashMapUnmanaged(Stashed);
/// Fused weight groups: one prepared matrix of several linears' rows that share an input.
pub const group_kv = 1;
pub const group_qkv = 2;
pub const group_gu = 3;
/// Ops a draft frame encodes before committing them, so the GPU starts while the CPU encodes the rest.
const part_ops = 40;
/// The first part is short, so the GPU starts while the rest is still being encoded.
const first_part_ops = 6;

/// Bytes [lo, hi) of one Metal buffer that a frame dispatch reads or writes.
pub const Span = struct {
    buf: usize,
    lo: usize,
    hi: usize,
    pub fn of(r: Ref, bytes: usize) Span {
        return .{ .buf = @intFromPtr(r.buf.id), .lo = r.off, .hi = r.off + bytes };
    }
    pub fn whole(buf: mtl.Buffer) Span {
        return .{ .buf = @intFromPtr(buf.id), .lo = 0, .hi = std.math.maxInt(usize) };
    }
    fn meets(a: Span, b: Span) bool {
        return a.buf == b.buf and a.lo < b.hi and b.lo < a.hi;
    }
};

/// What the frame's concurrent encoder ran since its last barrier: a dispatch that reads what one of them writes, or
/// writes what one of them reads or writes, waits behind a barrier; every other dispatch runs beside them.
pub const Hazards = struct {
    writes: [96]Span = undefined,
    reads: [192]Span = undefined,
    n_writes: usize = 0,
    n_reads: usize = 0,
    pub fn clear(h: *Hazards) void {
        h.n_writes = 0;
        h.n_reads = 0;
    }
    fn any(list: []const Span, x: Span) bool {
        for (list) |y| if (y.meets(x)) return true;
        return false;
    }
    /// Barrier when `reads`/`writes` conflict with the open set, then add them to it.
    pub fn order(h: *Hazards, e: mtl.ComputeEncoder, reads: []const Span, writes: []const Span) void {
        var conflict = h.n_writes + writes.len > h.writes.len or h.n_reads + reads.len > h.reads.len;
        if (!conflict) for (reads) |x| if (any(h.writes[0..h.n_writes], x)) {
            conflict = true;
            break;
        };
        if (!conflict) for (writes) |x| if (any(h.writes[0..h.n_writes], x) or any(h.reads[0..h.n_reads], x)) {
            conflict = true;
            break;
        };
        if (conflict) h.barrier(e);
        @memcpy(h.reads[h.n_reads..][0..reads.len], reads);
        h.n_reads += reads.len;
        @memcpy(h.writes[h.n_writes..][0..writes.len], writes);
        h.n_writes += writes.len;
    }
    /// Everything encoded so far completes before anything encoded next.
    pub fn barrier(h: *Hazards, e: mtl.ComputeEncoder) void {
        e.barrier();
        h.clear();
    }
};

/// A frame's words that depend on how many taps the verify before it admits (1 or 2), when the frame is encoded before the
/// host reads that verify (runtime_model.prepareNext): the host writes one block per count behind block 0 of `args`, the
/// verify's result picks one into block 0 on the GPU (tree_commit_gpu.select), and the frame's dispatches read it there.
/// Each field starts its own 256 bytes, so every binding's offset is aligned.
pub const RoundArgs = extern struct {
    /// the block's rope positions: right behind the admitted taps
    positions: [2]u64 align(256),
    /// the block's tokens: the verify's last token (written on the GPU), then the mask
    ids: [2]u32 align(256),
    /// the K/V slots of the block's own rows, behind the admitted context in the ring
    slots: [2]u32 align(256),
    /// the block's attention: the admitted context and its own rows
    attention: core.shared_attention.Args align(256),
    /// the admitted taps' K/V appends
    append: core.shared_attention.Args align(256),
    /// the admitted taps' lane projections: [rows, sums stride]
    dims: [2]i32 align(256),
    /// their group sums' rows
    count: i32 align(256),
    /// the admitted taps' projections on simdgroup tiles (no tensor units): the fusion's and each K or V one's
    fusion: simd.Args align(256),
    kv: simd.Args align(256),
    /// [admitted taps, status (0: the verify's result was valid), keys, 0]
    meta: [4]u32 align(256),

    /// Words a block takes.
    pub const words = @sizeOf(RoundArgs) / 4;
    /// The word tree_commit_gpu.select writes the verify's last token to, and the one it writes its status to.
    pub const token_word = @offsetOf(RoundArgs, "ids") / 4;
    pub const status_word = @offsetOf(RoundArgs, "meta") / 4 + 1;
    /// The blocks `args` holds: block 0 for the frame, then one per admitted count.
    pub const blocks = 3;

    /// The block for `rows` (1 or 2) taps admitted behind `context` (begin, end) and a 2-row block after them, whose
    /// attention keys reach at most `keys`; the frame's launches are those of 2 rows.
    pub fn of(c: cfg.Config, context: op.Context, rows: u32, keys: u32) RoundArgs {
        const end = context.end + rows;
        const context_rows: u32 = @intCast(end - context.begin);
        const at: u32 = @intCast(context.begin % c.window);
        return .{
            .positions = .{ end, end + 1 },
            .ids = .{ c.mask, c.mask },
            .slots = .{ at + context_rows, at + context_rows + 1 },
            .attention = .{ .rows = 2, .keys = context_rows + 2, .mask_stride = keys, .scale_bits = @bitCast(@import("runtime_context.zig").scaleOf(c)) },
            .append = .{ .rows = rows },
            .dims = .{ @intCast(rows), 16 },
            .count = @intCast(rows),
            .fusion = .{ .k = @intCast(c.tapWidth()), .n = @intCast(c.hidden), .r = @intCast(rows) },
            .kv = .{ .k = @intCast(c.hidden), .n = @intCast(c.kvWidth()), .r = @intCast(rows) },
            .meta = .{ rows, 0, context_rows + 2, 0 },
        };
    }
};

/// First in a frame encoded early (Model.prepareNext): waits for the verify's result and writes block 0 of `args`.
pub const Lead = struct { ptr: *anyopaque, encode: *const fn (*anyopaque, mtl.ComputeEncoder, Ref) anyerror!void };

pub const HeldNorm = struct { x: op.View, p: op.Norm, y: op.View };

pub const Backend = struct {
    allocator: std.mem.Allocator,
    residency: ?mtl.ResidencySet.Follow = null,
    target: *Target,
    checkpoint: *core.checkpoint_metal.Checkpoint,
    config: cfg.Config,
    primitives: core.draft_ops.Ops,
    attention: core.shared_attention.Attention,
    topk: core.bf16_topk_gpu.Ops,
    head: @import("runtime_head.zig").Head,
    frame: Frame,
    caches: []Cache,
    context: op.Context = .{ .cache = 1, .begin = 0, .end = 0 },
    batch: []const []Cache = &.{}, // a batched pass's streams' caches, by stream
    transaction: ?u64 = null,
    staged: ?u64 = null,
    appended: u32 = 0,
    appended_rows: u32 = 0,
    failed: bool = false,
    prepared: std.AutoHashMapUnmanaged(usize, core.affine4_lane.Adapter) = .empty,
    input_sums: ?struct { view: op.View, sums: core.lane_projection.Sums } = null,
    /// A conv with a residual held back for the norm that reads its output (one launch for both); the next launch of
    /// anything else encodes it first, on its own.
    held_conv: ?struct { x: op.View, p: op.Conv, y: op.View } = null,
    /// Per-head norms held back for the ropes that read their outputs (one launch for each pair); any other launch
    /// encodes them first, on their own.
    held_norms: [8]?HeldNorm = @splat(null),
    tiled: std.AutoHashMapUnmanaged(usize, Tiled) = .empty,
    simd: ?simd.Pipelines = null,
    stash: ?*Stash = null,
    fused_count: usize = 0,
    /// Linears drafted below 4 bits also keep a q4 copy under `key | precise_key` (and their fused groups under
    /// `(key | precise_key) + kind`); with `precise` a frame's projections read those copies.
    precise: bool = false,
    precise_count: usize = 0,
    hazards: Hazards = .{},
    /// The frame's drafter attention mask: every layer of a frame with the same context and query rows reads it.
    mask: ?struct { epoch: u64, context_rows: u32, rows: u32, view: op.View } = null,
    /// RoundArgs blocks: the frame's (0), then one per count the verify before it may admit.
    args: mtl.Buffer,
    /// While a frame is encoded early (its context and anchor are placeholders for the larger count): its first encoder
    /// runs this, and the launches that depend on the admitted count read their words from block 0 of `args`.
    early: ?Lead = null,
    pub fn init(a: std.mem.Allocator, target: *Target, checkpoint: *core.checkpoint_metal.Checkpoint, c: cfg.Config) !Backend {
        try c.check();
        const primitives = try core.draft_ops.Ops.init(target.device);
        errdefer primitives.deinit();
        const attention = try core.shared_attention.Attention.init(a, target.device, .{ .query_heads = c.heads, .kv_heads = c.kv_heads, .head_dim = c.head_dim, .rotary_dim = c.head_dim, .capacity = c.window + 16, .output = .bf16, .split_chunk = 64 });
        errdefer attention.deinit();
        const topk = try core.bf16_topk_gpu.Ops.init(target.device);
        errdefer topk.deinit();
        var head = try @import("runtime_head.zig").Head.init(a, target.device, target.weights.head);
        errdefer head.deinit();
        if (head_bits < 4 and target.device.tensorUnits() and @import("../quant_gpu.zig").Kernels.ranged(head.linear)) {
            head.lowBits(a, target.device, &target.checkpoint, head_bits, low_group, low_reg) catch |err| std.log.warn("draft head stays q4 alone: {s}", .{@errorName(err)});
            if (head.plain) |low| target.kernels.prepareTiles(head.linear, Head.rescored_stride) catch |err| {
                std.log.warn("draft head stays q4 alone: {s}", .{@errorName(err)});
                low.deinit();
                head.plain = null;
            };
        }
        // the trimmed head's shape is the drafter's own: its pipeline compiles here, not in the first timed round
        // (a depth-1 block heads one mask row; every head launch of at most 8 rows shares this pipeline)
        try target.kernels.prepareDecode(head.linear, 1);
        const caches = try newCaches(a, target.device, c, c.layers);
        errdefer freeCaches(a, caches);
        const args = try target.device.buffer(RoundArgs.blocks * @sizeOf(RoundArgs), options);
        return .{ .allocator = a, .target = target, .checkpoint = checkpoint, .config = c, .primitives = primitives, .attention = attention, .topk = topk, .head = head, .frame = .{ .allocator = a, .device = target.device, .queue = target.queue, .bridge = frame_mod.Bridge.init(target.device) catch null }, .caches = caches, .args = args };
    }
    pub fn setResident(b: *Backend, hook: ?mtl.ResidencySet.Follow) void {
        @import("../residency.zig").setDraft(b, hook);
    }

    pub fn deinit(b: *Backend) void {
        b.setResident(null);
        b.frame.deinit();
        var prepared = b.prepared.valueIterator();
        while (prepared.next()) |adapter| adapter.deinit();
        b.prepared.deinit(b.allocator);
        var tiled = b.tiled.valueIterator();
        while (tiled.next()) |t| t.deinit();
        b.tiled.deinit(b.allocator);
        if (b.simd) |*p| p.deinit();
        freeCaches(b.allocator, b.caches);
        b.args.deinit();
        b.head.deinit();
        b.topk.deinit();
        b.attention.deinit();
        b.primitives.deinit();
    }
    /// Prepared q4 linears use tensor adapters or SIMD tiles, keyed by the graph weight address.
    pub fn prepare(b: *Backend) !void {
        const units = b.target.device.tensorUnits();
        const order = if (units) try core.lane_projection.RegOrder.init(b.target.device) else null;
        defer if (order) |o| o.deinit();
        if (!units) b.simd = try simd.Pipelines.load(b.target.device);
        var stash: Stash = .empty;
        defer {
            var it = stash.iterator();
            while (it.next()) |entry| {
                b.allocator.free(entry.key_ptr.*);
                var p = entry.value_ptr.prepared;
                p.deinit();
            }
            stash.deinit(b.allocator);
        }
        if (units) b.stash = &stash;
        defer b.stash = null;
        var rows: std.ArrayList(u16) = .empty;
        defer rows.deinit(b.allocator);
        var kept = b.checkpoint.tensors.iterator();
        while (kept.next()) |entry| if (linear(entry.key_ptr.*, entry.value_ptr.rank)) {
            const t = entry.value_ptr.*;
            try b.prepareOne(order, entry.key_ptr.*, @intFromPtr(t.buffer.contents() + t.offset), t.dtype, t.shape, t.host(u16));
        };
        var later = b.checkpoint.deferred.iterator();
        while (later.next()) |entry| if (linear(entry.key_ptr.*, entry.value_ptr.rank)) {
            const d = entry.value_ptr.*;
            try rows.resize(b.allocator, d.bytes / 2);
            try b.checkpoint.read(entry.key_ptr.*, 0, std.mem.sliceAsBytes(rows.items));
            try b.prepareOne(order, entry.key_ptr.*, @intFromPtr((try b.checkpoint.view(entry.key_ptr.*)).ptr), d.dtype, d.shape, rows.items);
        };
        const expected = b.config.linearCount();
        if (b.stash != null) try b.buildGroups(order, &stash);
        if ((if (units) b.prepared.count() - b.fused_count - b.precise_count else b.tiled.count()) != expected) return error.DraftTensorCoverage;
    }
    /// The key bit of a linear's q4 copy beside its copy below 4 bits (no tensor address reaches it).
    pub const precise_key: usize = 1 << 62;
    /// One bf16 [n][k] linear's q4 copy under `key`: an M5 lane adapter, or (no `order`) the same bits on tiles.
    fn prepareOne(b: *Backend, order: ?core.lane_projection.RegOrder, name: []const u8, key: usize, dtype: core.checkpoint_metal.DType, shape: @FieldType(core.checkpoint_metal.Tensor, "shape"), rows: []const u16) !void {
        if (dtype != .bf16 or shape[0] % 32 != 0 or shape[1] % 64 != 0) return error.DraftPreparationUnqualified;
        const bits: u8 = if (order != null) draftBits(name) else 4;
        var prepared_words = if (bits != 4 and low_reg) try core.affine4.prepareNibblesRefined(b.allocator, rows, shape[0], shape[1], bits) else try core.affine4.prepareBits(b.allocator, rows, shape[0], shape[1], bits, if (bits != 4) low_group else 64);
        defer prepared_words.deinit();
        if (b.stash) |st| if (shortName(name)) |short| {
            const layer = layerOf(name) orelse return error.DraftTensorCoverage;
            var buf: [16]u8 = undefined;
            const id = try std.fmt.bufPrint(&buf, "{d}{s}", .{ layer, short });
            const owned = try b.allocator.dupe(u8, id);
            errdefer b.allocator.free(owned);
            const dup = core.affine4.Prepared{ .allocator = b.allocator, .n = prepared_words.n, .k = prepared_words.k, .words = try b.allocator.dupe(u32, prepared_words.words), .scales = try b.allocator.dupe(u16, prepared_words.scales), .biases = try b.allocator.dupe(u16, prepared_words.biases), .bits = prepared_words.bits, .group = prepared_words.group, .nibbles = prepared_words.nibbles };
            try st.put(b.allocator, owned, .{ .prepared = dup, .key = key });
        };
        if (order) |o| {
            const adapter = try core.affine4_lane.Adapter.init(b.allocator, b.target.device, prepared_words, try @import("../projection.zig").split(shape[0], shape[1]), o);
            try @import("../residency.zig").putPrepared(b, key, adapter);
            // a linear drafted below 4 bits keeps its q4 copy too, for frames drafted precise
            if (bits != 4) {
                var q4 = try core.affine4.prepareBits(b.allocator, rows, shape[0], shape[1], 4, 64);
                defer q4.deinit();
                if (b.stash) |st| if (shortName(name)) |short| {
                    const layer = layerOf(name) orelse return error.DraftTensorCoverage;
                    var buf: [16]u8 = undefined;
                    const id = try std.fmt.bufPrint(&buf, "{d}{c}", .{ layer, std.ascii.toUpper(short[0]) });
                    const owned = try b.allocator.dupe(u8, id);
                    errdefer b.allocator.free(owned);
                    const dup = core.affine4.Prepared{ .allocator = b.allocator, .n = q4.n, .k = q4.k, .words = try b.allocator.dupe(u32, q4.words), .scales = try b.allocator.dupe(u16, q4.scales), .biases = try b.allocator.dupe(u16, q4.biases), .bits = q4.bits, .group = q4.group, .nibbles = q4.nibbles };
                    try st.put(b.allocator, owned, .{ .prepared = dup, .key = key | precise_key });
                };
                const exact = try core.affine4_lane.Adapter.init(b.allocator, b.target.device, q4, try @import("../projection.zig").split(shape[0], shape[1]), o);
                try @import("../residency.zig").putPrepared(b, key | precise_key, exact);
                b.precise_count += 1;
            }
        } else {
            const t = try Tiled.init(b.target.device, prepared_words);
            try @import("../residency.zig").putTiled(b, key, t);
        }
    }
    /// Code width of a draft linear's prepared copy: the MLP's gate and up, attention's o and the fusion input projection
    /// fc at `mlp_bits`; q, k and v (one fused launch), the dynamic-conv kernel projections, the MLP's down and every
    /// other linear q4. In ArthurOstapenko's per-linear acceptance runs (48d75d22) q/k/v and the conv projections cost
    /// acceptance at 2 bits, and the down's 2-bit copy costs about as many extra rounds as its bytes save, so it stays
    /// q4; the head's 2-bit copy only picks tiles for its q4 scores (head_bits).
    pub const mlp_bits: u8 = 2;
    /// Code width of the draft head's requantized copy of the spans' rows, which heads of at most Head.rescored_rows rows
    /// read first to pick the tiles the q4 head computes (4: no copy, the q4 head alone).
    pub const head_bits: u8 = 2;
    /// Copies below 4 bits on the reg kernel's 2-bit words (prepareNibbles, then lane_projection.halveReg) at 64-element
    /// groups; false takes MLX bit streams at `low_group` on the plain lane kernel.
    pub const low_reg: bool = true;
    /// Elements a (scale, bias) pair covers in the plain kernel's copies below 4 bits (the draft head's included).
    pub const low_group: u32 = 128;
    fn draftBits(name: []const u8) u8 {
        const mlp = [_][]const u8{ "mlp.gate_proj.weight", "mlp.up_proj.weight", "self_attn.o_proj.weight", "fc.weight" };
        for (mlp) |m| if (std.mem.endsWith(u8, name, m)) return mlp_bits;
        return 4;
    }
    fn shortName(name: []const u8) ?[]const u8 {
        const table = [_][2][]const u8{ .{ "self_attn.q_proj.weight", "q" }, .{ "self_attn.k_proj.weight", "k" }, .{ "self_attn.v_proj.weight", "v" }, .{ "mlp.gate_proj.weight", "g" }, .{ "mlp.up_proj.weight", "u" } };
        for (table) |t| if (std.mem.endsWith(u8, name, t[0])) return t[1];
        return null;
    }
    fn layerOf(name: []const u8) ?usize {
        const at = std.mem.indexOf(u8, name, "layers.") orelse return null;
        const rest = name[at + 7 ..];
        const end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        return std.fmt.parseInt(usize, rest[0..end], 10) catch null;
    }
    fn fuse(b: *Backend, order: ?core.lane_projection.RegOrder, st: *Stash, layer: usize, parts: []const u8, key_part: u8, kind: usize) !void {
        var n: usize = 0;
        var k: usize = 0;
        var bits: u8 = 0;
        var size: u32 = 0;
        var nibbles = false;
        var buf: [16]u8 = undefined;
        for (parts) |p| {
            const got = st.get(try std.fmt.bufPrint(&buf, "{d}{c}", .{ layer, p })) orelse return;
            n += got.prepared.n;
            if (k != 0 and k != got.prepared.k) return error.DraftPreparationUnqualified;
            k = got.prepared.k;
            if (bits != 0 and (bits != got.prepared.bits or size != got.prepared.group or nibbles != got.prepared.nibbles)) return;
            bits = got.prepared.bits;
            size = got.prepared.group;
            nibbles = got.prepared.nibbles;
        }
        const word_bits: usize = if (nibbles) 4 else bits;
        const words = try b.allocator.alloc(u32, n * k / 32 * word_bits);
        defer b.allocator.free(words);
        const scales = try b.allocator.alloc(u16, n * k / size);
        defer b.allocator.free(scales);
        const biases = try b.allocator.alloc(u16, n * k / size);
        defer b.allocator.free(biases);
        var at: usize = 0;
        for (parts) |p| {
            const got = st.get(try std.fmt.bufPrint(&buf, "{d}{c}", .{ layer, p })).?;
            @memcpy(words[at * k / 32 * word_bits ..][0..got.prepared.words.len], got.prepared.words);
            @memcpy(scales[at * k / size ..][0..got.prepared.scales.len], got.prepared.scales);
            @memcpy(biases[at * k / size ..][0..got.prepared.biases.len], got.prepared.biases);
            at += got.prepared.n;
        }
        const key = (st.get(try std.fmt.bufPrint(&buf, "{d}{c}", .{ layer, key_part })).?).key + kind;
        const joined = core.affine4.Prepared{ .allocator = b.allocator, .n = n, .k = k, .words = words, .scales = scales, .biases = biases, .bits = bits, .group = size, .nibbles = nibbles };
        const adapter = try core.affine4_lane.Adapter.init(b.allocator, b.target.device, joined, try @import("../projection.zig").split(n, k), order);
        try @import("../residency.zig").putPrepared(b, key, adapter);
        b.fused_count += 1;
    }
    fn buildGroups(b: *Backend, order: ?core.lane_projection.RegOrder, st: *Stash) !void {
        for (0..b.config.layers) |layer| {
            try b.fuse(order, st, layer, "kv", 'k', group_kv);
            try b.fuse(order, st, layer, "qkv", 'q', group_qkv);
            try b.fuse(order, st, layer, "gu", 'g', group_gu);
            // the q4 copies' gate/up group of a layer drafted below 4 bits (stashed as G and U)
            const before = b.fused_count;
            try b.fuse(order, st, layer, "GU", 'G', group_gu);
            if (b.fused_count != before) {
                b.fused_count -= 1;
                b.precise_count += 1;
            }
        }
    }
    /// The draft's linears: its rank-2 weights, which q4 drafts read only through their prepared copies.
    pub fn linear(name: []const u8, rank: usize) bool {
        return rank == 2 and std.mem.endsWith(u8, name, ".weight");
    }
    pub fn ops(b: *Backend) op.Ops {
        return .{ .ptr = b, .vtable = &callbacks.vtable };
    }
    pub fn source(b: *Backend) ck.Source {
        return .{ .ptr = b, .getFn = get };
    }
    /// A tensor's host view; a deferred linear's view only names it (its key), its bytes read once by prepare.
    fn get(ptr: *anyopaque, name: []const u8) !ck.Tensor {
        const b: *Backend = @ptrCast(@alignCast(ptr));
        const d = b.checkpoint.deferred.get(name) orelse return cpu(try b.checkpoint.get(name));
        var shape: @FieldType(ck.Tensor, "shape") = @splat(1);
        @memcpy(shape[0..d.rank], d.shape[0..d.rank]);
        return .{ .dtype = d.dtype, .rank = @intCast(d.rank), .shape = shape, .bytes = try b.checkpoint.view(name) };
    }
    pub fn cpu(t: core.checkpoint_metal.Tensor) ck.Tensor {
        var shape: @FieldType(ck.Tensor, "shape") = @splat(1);
        @memcpy(shape[0..t.rank], t.shape[0..t.rank]);
        return .{ .dtype = t.dtype, .rank = @intCast(t.rank), .shape = shape, .bytes = t.buffer.contents()[t.offset .. t.offset + t.bytes] };
    }
    pub fn weight(b: *Backend, t: ck.Tensor) !Ref {
        var it = b.checkpoint.tensors.valueIterator();
        while (it.next()) |candidate| if (candidate.buffer.contents() + candidate.offset == t.bytes.ptr and candidate.bytes == t.bytes.len and candidate.dtype == t.dtype) return .{ .buf = candidate.buffer, .off = candidate.offset };
        return error.UnownedDraftWeight;
    }
    pub fn encoder(b: *Backend, epoch: u64) !mtl.ComputeEncoder {
        if (b.failed) return error.DraftBackendFailed;
        try b.frame.require(epoch);
        b.frame.ops += 1;
        if (b.frame.ops == first_part_ops or b.frame.ops % part_ops == 0) {
            const parts = b.frame.part_count;
            try b.frame.part();
            if (b.frame.part_count != parts) b.hazards.clear(); // a new encoder behind the fence: nothing open
        }
        try b.flushHeld(true);
        return b.frame.encoder;
    }
    /// `encoder` for a launch that takes or holds a per-head norm: the other held norms stay held.
    pub fn encoderHolding(b: *Backend, epoch: u64) !mtl.ComputeEncoder {
        if (b.failed) return error.DraftBackendFailed;
        try b.frame.require(epoch);
        b.frame.ops += 1;
        if (b.frame.ops == first_part_ops or b.frame.ops % part_ops == 0) {
            const parts = b.frame.part_count;
            try b.frame.part();
            if (b.frame.part_count != parts) b.hazards.clear(); // a new encoder behind the fence: nothing open
        }
        try b.flushHeld(false);
        return b.frame.encoder;
    }
    /// Encodes the held conv (and with `norms` the held norms) on their own.
    fn flushHeld(b: *Backend, norms: bool) !void {
        if (b.held_conv) |h| {
            b.held_conv = null;
            try b.encodeConv(b.frame.encoder, h.x, h.p, h.y);
        }
        if (!norms) return;
        for (&b.held_norms) |*slot| {
            if (slot.*) |h| {
                slot.* = null;
                try b.encodeNorm(b.frame.encoder, h.x, h.p, h.y);
            }
        }
    }
    /// The held norm whose output is `x`, for `heads` heads of `dim`, taken out of its slot.
    pub fn takeHeldNorm(b: *Backend, x: op.View, heads: u32, dim: u32) ?HeldNorm {
        for (&b.held_norms) |*slot| {
            if (slot.*) |h| {
                if (h.p.heads == heads and h.p.head_dim == dim and std.meta.eql(h.y, x)) {
                    slot.* = null;
                    return h;
                }
            }
        }
        return null;
    }
    /// A free held-norm slot.
    pub fn heldNormSlot(b: *Backend) ?*?HeldNorm {
        for (&b.held_norms) |*slot| {
            if (slot.* == null) return slot;
        }
        return null;
    }
    /// The norm launch of `x` into `y` (tf_draft_rms).
    pub fn encodeNorm(b: *Backend, e: mtl.ComputeEncoder, x: op.View, p: op.Norm, y: op.View) !void {
        b.access(e, &.{try b.span(x)}, &.{try b.span(y)});
        try b.primitives.norm(e, try b.frame.ref(x), try b.weight(p.weight), try b.frame.ref(y), .{ .rows = x.rows, .heads = p.heads, .dim = p.head_dim, .x_stride = x.stride, .y_stride = y.stride, .eps = p.eps });
    }
    /// The conv launch of `x` into `y` (tf_dynamic_conv2).
    pub fn encodeConv(b: *Backend, e: mtl.ComputeEncoder, x: op.View, p: op.Conv, y: op.View) !void {
        if (p.residual) |r| b.access(e, &.{ try b.span(x), try b.span(p.dynamic), try b.span(r) }, &.{try b.span(y)}) else b.access(e, &.{ try b.span(x), try b.span(p.dynamic) }, &.{try b.span(y)});
        try b.primitives.conv(e, try b.frame.ref(x), try b.frame.ref(p.dynamic), try b.weight(p.base), if (p.residual) |r| try b.frame.ref(r) else null, try b.frame.ref(y), .{ .rows = x.rows, .width = x.width, .group = p.group, .block = p.block, .branch = p.branch, .residual = @intFromBool(p.residual != null), .x_stride = x.stride, .dynamic_stride = p.dynamic.stride, .y_stride = y.stride });
    }
    /// A frame view's bytes, for hazard tracking.
    pub fn span(b: *Backend, v: op.View) !Span {
        return Span.of(try b.frame.ref(v), (@as(usize, v.rows - 1) * v.stride + v.width) * v.dtype.size());
    }
    pub fn access(b: *Backend, e: mtl.ComputeEncoder, reads: []const Span, writes: []const Span) void {
        b.hazards.order(e, reads, writes);
    }
    pub fn barrier(b: *Backend, e: mtl.ComputeEncoder) void {
        b.hazards.barrier(e);
    }
    /// Field `name` of the frame's RoundArgs block.
    pub fn roundArg(b: *const Backend, comptime name: []const u8) Ref {
        return .{ .buf = b.args, .off = @offsetOf(RoundArgs, name) };
    }
    /// The frame is an early one's append: its taps' count is the verify's, read from `args`.
    pub fn admitting(b: *const Backend) bool {
        return b.early != null and b.transaction != null;
    }
    pub fn copy(b: *Backend, e: mtl.ComputeEncoder, x: Ref, y: Ref, rows: u32, width: u32, stride: u32) !void {
        try b.target.glue.unstack(e, .{ .buffer = x.buf, .offset = x.off }, .{ .buffer = y.buf, .offset = y.off }, .{ .rows = rows, .width = width, .stride = stride, .offset = 0 });
    }
};
