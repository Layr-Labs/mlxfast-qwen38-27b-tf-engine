//! The greedy verify head's sieve (kernels/metal/qwen27/head_sieve.metal), the CUDA head_sieve's method on the reg lane
//! layout: a 2-bit copy of the vocabulary head's codes (code >> 2, the same metadata words) built when a drafter binds;
//! a greedy verify of at most 8 rows then streams it in place of the 4-bit words (half the head's code bytes), recomputes
//! each row's first tile (the 32-column tile of its best approximate logit) with the head's own tile-list launch
//! (Kernels.quantTiles: each word the one the full launch writes), and then, in one indirect tile-list launch, only the
//! tiles whose approximate logit plus kappa spreads of the dropped bits' term reaches the lower of that tile's exact best
//! and its best approximate logit. Each row's token is the best of those exact words, the full head's tie rule: the
//! full head's token whenever the best id's dropped bits' term exceeds both kappa spreads and the first tile's best-a
//! id's by less than kappa spreads. Statistical (kappa spreads), not a certified bound, as on the CUDA track.
//! No logits row is written: the round's picks go straight to the matcher, and Runner.head_token stands in for the
//! accepted row's logits that tree_commit's keep_head would have left in row 0.
//! kappa is 7.
const std = @import("std");
const mtl = @import("metal");
const projection = @import("projection.zig");
const quant = @import("quant_gpu.zig");
const Model = @import("model.zig").Model;
const sources = @import("qwen_runtime_sources");

const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

pub const HeadSieve = struct {
    /// Rows a sieved verify takes: the tile-list launch's decoded rows.
    pub const rows_max = 8;
    /// The first tiles' rows: a slot (32 columns) a row, rows_max slots.
    const first_stride = rows_max * 32;
    const sums_stride = 16; // quant.padded(rows) for rows_max rows at most

    pack: mtl.Pipeline,
    prep: mtl.Pipeline,
    scan: mtl.Pipeline,
    top: mtl.Pipeline,
    select: mtl.Pipeline,
    pick: mtl.Pipeline,
    /// The head the sieve stands for (Weights.head_decode): its words and metadata, which the exact launches read.
    head: projection.Linear,
    tiles: u32,
    groups: u32,
    kappa: f32,
    /// The 2-bit copy and the per-round scratch: both owned by the runner's `owned` list (freed and made resident with it).
    words: mtl.Buffer,
    scratch: mtl.Buffer,
    o_xq: usize,
    o_hm: usize,
    o_first: usize,
    o_y1: usize,
    o_list: usize,
    o_args: usize,
    rounds: u64 = 0,
    listed: u64 = 0,
    listed_max: u32 = 0,
    next_log: u64 = 1,

    /// Whether the sieve takes `head`: the vocabulary head on the reg lane layout (4-bit g64, tensor units).
    pub fn fits(device: mtl.Device, head: projection.Linear) bool {
        return device.tensorUnits() and quant.Kernels.ranged(head) and head.n % 256 == 0 and head.n >= 65536 and head.k % 64 == 0;
    }

    /// The sieve for `m`'s decoded head, its two buffers appended to `owned` (joined to `residency` when given); null when
    /// the head does not fit.
    pub fn init(a: std.mem.Allocator, m: *Model, owned: *std.ArrayList(mtl.Buffer), residency: ?mtl.ResidencySet.Follow) !?HeadSieve {
        const head = m.weights.head_decode;
        if (!fits(m.device, head)) {
            std.log.info("qwen27: head sieve off: head n {d} k {d} batch {d}", .{ head.n, head.k, head.batch });
            return null;
        }
        const kappa: f32 = 7.0;
        const tiles: u32 = head.n / 32;
        const groups: u32 = head.k / 64;
        // the head's shape and metadata form ahead of the kernels; kappa a literal (each value its own library)
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(a);
        try text.print(a, "#define TF_SIEVE_SHAPED 1\n#define TF_N {d}\n#define TF_K {d}\n#define TF_TB {d}\n#define TF_SIEVE_KAPPA {d:.4}f\n", .{ head.n, head.k, head.batch, kappa });
        if (head.index != null) try text.print(a, "#define TF_SBI {d}\n", .{@as(usize, if (head.index12) @import("core").lane_projection.sb_table12 else @import("core").lane_projection.sb_table)});
        if (head.index != null and head.index12) try text.appendSlice(a, "#define TF_SBI12 1\n");
        try text.appendSlice(a, sources.head_sieve);
        const lib = try mtl.Library.fromSource(m.device, text.items, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var made: [6]mtl.Pipeline = undefined;
        var count: usize = 0;
        errdefer for (made[0..count]) |p| p.deinit();
        for ([_][]const u8{ "tf_head_sieve_pack", "tf_head_sieve_prep", "tf_head_sieve_scan", "tf_head_sieve_top", "tf_head_sieve_select", "tf_head_sieve_pick" }) |name| {
            made[count] = try mtl.Pipeline.init(m.device, lib, name, false);
            count += 1;
        }
        // the scan runs a SIMD group a tile; top and pick run a 1024-thread threadgroup
        if (made[2].simdWidth() != 32 or made[3].maxThreads() < 1024 or made[5].maxThreads() < 1024) return error.UnsupportedPipelineShape;
        // the exact launches' pipelines (the first tiles' rows, then the listed tiles' rows at the vocabulary's stride)
        try m.kernels.prepareTiles(head, first_stride);
        try m.kernels.prepareTiles(head, head.n);
        var at: usize = 0;
        const take = struct {
            fn f(p: *usize, bytes: usize) usize {
                const o = p.*;
                p.* += (bytes + 255) / 256 * 256;
                return o;
            }
        }.f;
        const o_xq = take(&at, @as(usize, groups) * sums_stride * 4);
        const o_hm = take(&at, @as(usize, tiles) * 16 * 4);
        const o_first = take(&at, (1 + rows_max) * 4);
        const o_y1 = take(&at, rows_max * first_stride * 2);
        const o_list = take(&at, (1 + @as(usize, tiles)) * 4);
        const o_args = take(&at, 4 * 4);
        const words = try m.device.buffer(@as(usize, tiles) * groups * 512, options);
        owned.append(a, words) catch |err| {
            words.deinit();
            return err;
        };
        const scratch = try m.device.buffer(at, options);
        owned.append(a, scratch) catch |err| {
            scratch.deinit();
            return err;
        };
        if (residency) |f| {
            f.join(f.ctx, words);
            f.join(f.ctx, scratch);
        }
        // the 2-bit copy, once: a thread a lane's uint4 of a block
        {
            const cb = m.queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(made[0]);
            e.setBuffer(head.words.buffer, head.words.offset, 0);
            e.setBuffer(words, 0, 1);
            e.dispatchThreads(mtl.Size.of(@as(usize, tiles) * groups * 32, 1, 1), mtl.Size.of(256, 1, 1));
            e.end();
            cb.commit();
            cb.wait();
            if (@import("decode_round.zig").gpuFailed(cb, "the head sieve's 2-bit copy")) return error.HeadSieveGpuFailure;
        }
        std.log.info("qwen27: head sieve: kappa {d:.2}, 2-bit copy {d} MiB, metadata {s}", .{ kappa, @as(usize, tiles) * groups * 512 >> 20, if (head.index == null) "pairs" else if (head.index12) "12-bit indices" else "16-bit indices" });
        return .{ .pack = made[0], .prep = made[1], .scan = made[2], .top = made[3], .select = made[4], .pick = made[5], .head = head, .tiles = tiles, .groups = groups, .kappa = kappa, .words = words, .scratch = scratch, .o_xq = o_xq, .o_hm = o_hm, .o_first = o_first, .o_y1 = o_y1, .o_list = o_list, .o_args = o_args };
    }

    /// The pipelines; the buffers go with the runner's `owned`.
    pub fn deinit(s: *HeadSieve) void {
        for ([_]mtl.Pipeline{ s.pack, s.prep, s.scan, s.top, s.select, s.pick }) |p| p.deinit();
    }

    fn ref(s: *const HeadSieve, offset: usize) projection.Ref {
        return .{ .buffer = s.scratch, .offset = offset };
    }

    /// A greedy verify's picks over `rows` of the frame's final normed rows (`input`, with the norm's group sums), behind
    /// the barrier that follows that norm: each row's token and nonfinite mask into `picks` (tree_round.Pick a row), as
    /// the full head's argmax would write them under the sieve's kappa condition. The frame's logits buffer takes the
    /// listed tiles' exact rows (no row of it is a logits row afterwards). Encoded into the forward's last encoder.
    pub fn encode(s: *HeadSieve, e: mtl.ComputeEncoder, m: *Model, rows: u32, picks: mtl.Buffer) !void {
        if (rows == 0 or rows > rows_max or quant.padded(rows) != sums_stride) return error.BadSieveRows;
        const f = &m.frame;
        const x = f.get(.input);
        const xs = quant.Sums{ .ref = f.get(.sums), .stride = sums_stride };
        const logits = f.get(.logits);
        if (logits.buffer.length() < @as(usize, rows) * s.head.n * 2) return error.BadSieveRows;
        const meta: projection.Ref = if (s.head.index) |i| i else s.head.pairs;
        // 1: each group's sums of squares of the rows; the list cleared
        e.setPipeline(s.prep);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setValue(@as(i32, @intCast(rows)), 1);
        e.setBuffer(s.scratch, s.o_xq, 2);
        e.setBuffer(s.scratch, s.o_list, 3);
        e.setBuffer(s.scratch, s.o_args, 4);
        e.dispatchThreads(mtl.Size.of(@as(usize, s.groups) * sums_stride, 1, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        // 2: every tile's best a + k s and best a, a row each, through the 2-bit copy
        e.setPipeline(s.scan);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(s.words, 0, 1);
        e.setBuffer(meta.buffer, meta.offset, 2);
        e.setValue([2]i32{ @intCast(rows), sums_stride }, 3);
        e.setBuffer(s.scratch, s.o_hm, 4);
        e.setBuffer(xs.ref.buffer, xs.ref.offset, 5);
        e.setBuffer(s.scratch, s.o_xq, 6);
        e.dispatchGroups(mtl.Size.of(s.tiles, 1, 1), mtl.Size.of(32, 1, 1));
        e.barrier();
        // 3: each row's first tile
        e.setPipeline(s.top);
        e.setBuffer(s.scratch, s.o_hm, 0);
        e.setValue(@as(i32, @intCast(rows)), 1);
        e.setBuffer(s.scratch, s.o_first, 2);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        e.barrier();
        // 4: the first tiles exactly, the head's own tile-list launch (slot r: row r's first tile)
        try m.kernels.quantTiles(e, s.head, x, xs, s.ref(s.o_first), s.ref(s.o_y1), rows, rows, first_stride);
        e.barrier();
        // 5: the rows' thresholds, and every tile some row's bound reaches
        e.setPipeline(s.select);
        e.setBuffer(s.scratch, s.o_hm, 0);
        e.setBuffer(s.scratch, s.o_y1, 1);
        e.setBuffer(s.scratch, s.o_first, 2);
        e.setValue(@as(i32, @intCast(rows)), 3);
        e.setBuffer(s.scratch, s.o_list, 4);
        e.setBuffer(s.scratch, s.o_args, 5);
        e.dispatchGroups(mtl.Size.of((s.tiles + 255) / 256, 1, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        // 6: the listed tiles exactly, as many threadgroups as tiles listed, rows the vocabulary apart in the logits buffer
        try m.kernels.quantTilesIndirect(e, s.head, x, xs, s.ref(s.o_list), logits, rows, s.tiles, s.head.n, s.ref(s.o_args));
        e.barrier();
        // 7: each row's token from those words
        e.setPipeline(s.pick);
        e.setBuffer(logits.buffer, logits.offset, 0);
        e.setBuffer(s.scratch, s.o_list, 1);
        e.setBuffer(picks, 0, 2);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    /// After a sieved round's command buffer ran: the tiles it listed, logged at rounds 1, 2, 4, ...
    pub fn after(s: *HeadSieve) void {
        const n = s.scratch.slice(u32, s.o_list / 4 + 1)[s.o_list / 4];
        s.rounds += 1;
        s.listed += n;
        s.listed_max = @max(s.listed_max, n);
        if (s.rounds != s.next_log) return;
        s.next_log *= 2;
        std.log.info("qwen27: head sieve: rounds={d} listed tiles avg={d} max={d} of {d}", .{ s.rounds, s.listed / s.rounds, s.listed_max, s.tiles });
    }
};
