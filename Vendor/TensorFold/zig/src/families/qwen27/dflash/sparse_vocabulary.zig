//! Single-row DFlash proposal retrieval. The target keeps its complete original vocabulary projection.
const mtl = @import("metal");
const core = @import("core");
const op = @import("operators.zig");
const k = 5120;
const groups = k / 64;
const n = 98592;
const tiles = (n + 255) / 256;
const padded = tiles * 256;
const count = tiles * 4;

pub const Search = struct {
    pipelines: [4]mtl.Pipeline,
    codes: mtl.Buffer,
    pairs: mtl.Buffer,
    map: mtl.Buffer,
    rms: mtl.Buffer,
    raw: mtl.Buffer,
    scales: mtl.Buffer,
    biases: mtl.Buffer,

    fn permute(row: usize) usize {
        // Coprime stride distributes neighbors across the 256-token buckets without a lookup.
        return (12347 * row + 2131) % n;
    }

    pub fn init(head: anytype, device: mtl.Device, words: []const u32, sc: []const u16, bi: []const u16) !Search {
        if (words.len != n * k / 8 or sc.len != n * groups or bi.len != sc.len) return error.BadDraftHead;
        const lib = try mtl.Library.fromSource(device, @embedFile("sparse_vocabulary.metal"), mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [4]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        for ([_][:0]const u8{ "selectChannels", "sectorDown", "pickBucket", "proposal_gather" }, &pipes) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.simdWidth() != 32) return error.DraftPreparationUnqualified;
        }
        const start = head.owned.items.len;
        errdefer while (head.owned.items.len > start) head.owned.pop().?.deinit();
        const s = Search{
            .pipelines = pipes,
            .codes = try head.allocate(device, padded * k / 2),
            .pairs = try head.allocate(device, padded * groups * 4),
            .map = try head.allocate(device, padded * 4),
            .rms = try head.allocate(device, k * 4),
            .raw = try head.allocate(device, words.len * 4),
            .scales = try head.allocate(device, sc.len * 2),
            .biases = try head.allocate(device, bi.len * 2),
        };
        @memcpy(s.raw.slice(u32, words.len), words);
        @memcpy(s.scales.slice(u16, sc.len), sc);
        @memcpy(s.biases.slice(u16, bi.len), bi);
        const order = s.map.slice(u32, padded);
        const ids = head.ids.?.slice(u32, n);
        for (order, 0..) |*token, row| token.* = if (row < n) ids[permute(row)] else 0;
        const bytes = s.codes.slice(u8, padded * k / 2);
        const meta = s.pairs.slice(u32, padded * groups);
        for (0..tiles) |tile| for (0..128) |pair| {
            const row = tile * 256 + pair * 2;
            const lo = if (row < n) permute(row) else 0;
            const hi = if (row + 1 < n) permute(row + 1) else 0;
            for (0..groups) |g| {
                const at = tile * groups + g;
                const a = lo * groups + g;
                const b = hi * groups + g;
                meta[at * 256 + pair * 2] = (@as(u32, bi[a]) << 16) | sc[a];
                meta[at * 256 + pair * 2 + 1] = (@as(u32, bi[b]) << 16) | sc[b];
                for (0..8) |j| {
                    const wa = words[a * 8 + j];
                    const wb = words[b * 8 + j];
                    for (0..8) |v| {
                        const sh: u5 = @intCast(v * 4);
                        bytes[(at * 64 + j * 8 + v) * 128 + pair] = @intCast(((wa >> sh) & 15) | (((wb >> sh) & 15) << 4));
                    }
                }
            }
        };
        // Estimate each channel's weight variance from a deterministic, checkpoint-only sample.
        var mean: [k]f64 = @splat(0);
        var squares: [k]f64 = @splat(0);
        for (0..4096) |sample| {
            const row = permute(sample);
            for (0..groups) |g| {
                const at = row * groups + g;
                const scale: f64 = core.affine4.asFloat(sc[at]);
                const bias: f64 = core.affine4.asFloat(bi[at]);
                for (0..64) |i| {
                    const digit = (words[at * 8 + i / 8] >> @as(u5, @intCast(4 * (i % 8)))) & 15;
                    const w = bias + scale * @as(f64, @floatFromInt(digit));
                    mean[g * 64 + i] += w;
                    squares[g * 64 + i] += w * w;
                }
            }
        }
        const gain = s.rms.slice(f32, k);
        for (gain, mean, squares) |*v, sum, sq| v.* = @floatCast(@sqrt(@max(0, sq / 4096 - (sum / 4096) * (sum / 4096))));
        return s;
    }

    pub fn deinit(s: Search) void {
        // Head owns and registers every persistent buffer.
        for (s.pipelines) |pipe| pipe.deinit();
    }

    pub fn encode(s: *const Search, b: anytype, e: mtl.ComputeEncoder, x: op.View) !op.Head {
        try x.check(1, k, .bf16);
        const mask = try b.frame.alloc(1, groups * 16, .u8);
        const values = try b.frame.alloc(1, groups * 16, .u32);
        const partials = try b.frame.alloc(1, padded * 16, .u32);
        const ids = try b.frame.alloc(1, count, .u32);
        const logits = try b.frame.alloc(1, count, .bf16);
        const inp = try b.frame.ref(x);
        const indices = try b.frame.ref(mask);
        const vals = try b.frame.ref(values);
        const parts = try b.frame.ref(partials);
        const map = try b.frame.ref(ids);
        const scores = try b.frame.ref(logits);
        b.access(e, &.{try b.span(x)}, &.{ try b.span(mask), try b.span(values) });
        e.setPipeline(s.pipelines[0]);
        e.setBuffer(inp.buf, inp.off, 0);
        e.setBuffer(s.rms, 0, 1);
        e.setBuffer(indices.buf, indices.off, 2);
        e.setBuffer(vals.buf, vals.off, 3);
        e.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(32, 1, 1));
        b.access(e, &.{ try b.span(mask), try b.span(values) }, &.{try b.span(partials)});
        e.setPipeline(s.pipelines[1]);
        e.setBuffer(s.codes, 0, 0);
        e.setBuffer(s.pairs, 0, 1);
        e.setBuffer(indices.buf, indices.off, 2);
        e.setBuffer(vals.buf, vals.off, 3);
        e.setBuffer(parts.buf, parts.off, 4);
        e.dispatchGroups(mtl.Size.of(tiles, 1, 4), mtl.Size.of(128, 1, 1));
        b.access(e, &.{try b.span(partials)}, &.{try b.span(ids)});
        e.setPipeline(s.pipelines[2]);
        e.setBuffer(parts.buf, parts.off, 0);
        e.setBuffer(s.map, 0, 1);
        e.setBuffer(map.buf, map.off, 2);
        e.dispatchGroups(mtl.Size.of(tiles, 1, 1), mtl.Size.of(32, 1, 1));
        b.access(e, &.{ try b.span(ids), try b.span(x) }, &.{try b.span(logits)});
        e.setPipeline(s.pipelines[3]);
        e.setBuffer(s.raw, 0, 0);
        e.setBuffer(s.scales, 0, 1);
        e.setBuffer(s.biases, 0, 2);
        e.setBuffer(inp.buf, inp.off, 3);
        e.setBuffer(map.buf, map.off, 4);
        e.setBuffer(scores.buf, scores.off, 5);
        e.setValue(@as(u32, count), 6);
        e.dispatchGroups(mtl.Size.of((count + 3) / 4, 1, 1), mtl.Size.of(128, 1, 1));
        return .{ .logits = logits, .token_ids = ids };
    }
};
