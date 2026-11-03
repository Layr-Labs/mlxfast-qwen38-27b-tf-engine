//! Shared-context attention, per-row RoPE and cache append; byte masks alone decide what each row sees.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Output = enum { f32, bf16 };
pub const Rope = enum { split_half, interleaved };
pub const Shape = struct {
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope: Rope = .split_half,
    capacity: usize,
    output: Output = .f32,
    split_chunk: usize = 256,

    pub fn validate(s: Shape) !void {
        if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or
            s.query_heads > 128 or s.head_dim == 0 or s.head_dim > 256 or s.head_dim % 32 != 0 or
            s.rotary_dim > s.head_dim or s.rotary_dim % 2 != 0 or s.capacity == 0 or s.capacity > std.math.maxInt(u32) or
            (s.split_chunk != 64 and s.split_chunk != 256)) return error.InvalidAttentionShape;
        _ = try std.math.mul(usize, s.capacity, try std.math.mul(usize, s.kv_heads, s.head_dim));
    }
};
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
/// Chunks covering `keys` (tf_shared_lane_attention_split's TF_CHUNK).
fn chunksOf(keys: usize, chunk: usize) usize {
    return (keys + chunk - 1) / chunk;
}
pub const Args = extern struct { rows: u32, keys: u32 = 0, mask_stride: u32 = 0, heads: u32 = 0, scale_bits: u32 = 0 };
/// A dispatch's Args: the host's, or the words an earlier dispatch wrote at a Ref.
const Bound = union(enum) { value: Args, at: Ref };
fn bindArgs(e: mtl.ComputeEncoder, args: Bound, index: usize) void {
    switch (args) {
        .value => |a| e.setValue(a, index),
        .at => |r| e.setBuffer(r.buf, r.off, index),
    }
}

pub fn source(a: std.mem.Allocator, s: Shape) ![]u8 {
    try s.validate();
    return std.fmt.allocPrint(a, "#define TF_QH {d}\n#define TF_KVH {d}\n#define TF_D {d}\n#define TF_RD {d}\n#define TF_INTERLEAVED {d}\n#define TF_OUT {s}\n#define TF_CHUNK {d}\n{s}", .{
        s.query_heads, s.kv_heads, s.head_dim, s.rotary_dim, @intFromBool(s.rope == .interleaved), if (s.output == .f32) "float" else "bfloat", s.split_chunk, sources.core_shared_attention,
    });
}

pub const Attention = struct {
    shape: Shape,
    attend: mtl.Pipeline,
    decode: mtl.Pipeline,
    rope: mtl.Pipeline,
    append: mtl.Pipeline,
    append_strided: mtl.Pipeline,
    split: mtl.Pipeline,
    merge: mtl.Pipeline,
    merge_xs: mtl.Pipeline,
    /// encodeSplit's chunk partials: [16 rows][kv heads][chunks][query heads a kv head][head_dim + 2] fp32.
    part: mtl.Buffer,
    /// encodeSplitSums' one launch (buildMerged): the split whose last chunk threadgroup of each query head and row merges
    /// it. Null keeps the split, a barrier and merge_xs.
    split_merge_xs: ?mtl.Pipeline = null,
    /// split_merge_xs' arrival counts, u32 [16 rows][query heads]: zero between launches (each launch's last arrivals
    /// leave their words zero), and launches that share `part` are ordered, so they share these too.
    arrivals: ?mtl.Buffer = null,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, s: Shape) !Attention {
        const text = try source(a, s);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        const attend = try mtl.Pipeline.init(device, lib, "tf_shared_lane_attention", false);
        errdefer attend.deinit();
        const decode = try mtl.Pipeline.init(device, lib, "tf_core_decode_attention", false);
        errdefer decode.deinit();
        const rope = try mtl.Pipeline.init(device, lib, "tf_lane_rope", false);
        errdefer rope.deinit();
        const append = try mtl.Pipeline.init(device, lib, "tf_lane_append_kv", false);
        errdefer append.deinit();
        const append_strided = try mtl.Pipeline.init(device, lib, "tf_lane_append_kv_strided", false);
        errdefer append_strided.deinit();
        const split = try mtl.Pipeline.init(device, lib, "tf_shared_lane_attention_split", false);
        errdefer split.deinit();
        const merge = try mtl.Pipeline.init(device, lib, "tf_shared_lane_attention_merge", false);
        errdefer merge.deinit();
        const merge_xs = try mtl.Pipeline.init(device, lib, "tf_shared_lane_attention_merge_xs", false);
        errdefer merge_xs.deinit();
        if (attend.simdWidth() != 32 or attend.maxThreads() < 32 or split.simdWidth() != 32 or merge.simdWidth() != 32) return error.UnsupportedAttentionPipeline;
        const part = try device.buffer(16 * s.query_heads * chunksOf(s.capacity, s.split_chunk) * (s.head_dim + 2) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return .{ .shape = s, .attend = attend, .decode = decode, .rope = rope, .append = append, .append_strided = append_strided, .split = split, .merge = merge, .merge_xs = merge_xs, .part = part };
    }
    pub fn deinit(p: Attention) void {
        p.attend.deinit();
        p.decode.deinit();
        p.rope.deinit();
        p.append.deinit();
        p.append_strided.deinit();
        p.split.deinit();
        p.merge.deinit();
        p.merge_xs.deinit();
        p.part.deinit();
        if (p.split_merge_xs) |m| m.deinit();
        if (p.arrivals) |r| r.deinit();
    }

    /// Rows a split_merge_xs launch takes (`part`'s rows).
    const arrival_rows = 16;
    /// Builds split_merge_xs (its own library, so a failure there leaves the others as they are) and keeps it only where
    /// mergedMatches finds the words of the split, barrier and merge_xs on this GPU; an error or a difference leaves it null.
    pub fn buildMerged(p: *Attention, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
        if (p.split_merge_xs) |m| m.deinit();
        if (p.arrivals) |r| r.deinit();
        p.split_merge_xs = null;
        p.arrivals = null;
        const base = try source(a, p.shape);
        defer a.free(base);
        const text = try std.mem.concat(a, u8, &.{ "#define TF_SPLIT_MERGE 1\n", base });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "tf_shared_lane_attention_split_merge_xs", false);
        errdefer pipe.deinit();
        if (pipe.simdWidth() != 32 or pipe.maxThreads() < 32) return error.UnsupportedAttentionPipeline;
        const arrivals = try device.buffer(arrival_rows * p.shape.query_heads * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer arrivals.deinit();
        @memset(arrivals.slice(u32, arrival_rows * p.shape.query_heads), 0);
        if (!try p.mergedMatches(pipe, arrivals, device, queue)) return error.MergedAttentionDiffers;
        p.split_merge_xs = pipe;
        p.arrivals = arrivals;
    }

    /// Encode key chunks and merge their fp32 partials in chunk order.
    pub fn encodeSplit(p: Attention, e: mtl.ComputeEncoder, rows: usize, keys: usize, stride: usize, scale: f32, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref) !void {
        try p.checkViews(rows, keys, stride, scale, q, k, v, mask, out);
        p.dispatchSplit(e, rows, keys, .{ .value = .{ .rows = @intCast(rows), .keys = @intCast(keys), .mask_stride = @intCast(stride), .scale_bits = @bitCast(scale) } }, q, k, v, mask, out, null, null);
    }

    /// encodeSplit on the Args an earlier dispatch wrote at `args`: at most `rows` rows, `keys` keys and a mask stride of
    /// at most `stride`, which bound the views and the grid (chunks past the Args' keys exit at once).
    pub fn encodeSplitAt(p: Attention, e: mtl.ComputeEncoder, rows: usize, keys: usize, stride: usize, args: Ref, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref) !void {
        try p.checkViews(rows, keys, stride, 1, q, k, v, mask, out);
        try check(args, @sizeOf(Args), 4);
        p.dispatchSplit(e, rows, keys, .{ .at = args }, q, k, v, mask, out, null, null);
    }

    /// Group sums of a bf16 output for a q4 projection that reads it: f32 [query heads * head_dim / 64][stride].
    pub const Sums = struct { ref: Ref, stride: u32 };
    /// The last `rows` keys (of the Args' keys) read from `k`/`v` rows at their own strides (elements), not from the
    /// cache views: the block's own rows where their producers left them.
    pub const Own = struct { k: Ref, v: Ref, rows: u32, k_stride: u32, v_stride: u32 };
    /// encodeSplit (on the host's Args, or with `args_at` on the words an earlier dispatch wrote, as encodeSplitAt) whose
    /// merge also writes the bf16 output's group sums: each group's 64 values added in column order, rows past the Args'
    /// rows zero. With `own`, the K/V views hold only the keys before the own rows.
    pub fn encodeSplitSums(p: Attention, e: mtl.ComputeEncoder, rows: usize, keys: usize, stride: usize, scale: f32, args_at: ?Ref, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref, xs: Sums, own: ?Own) !void {
        if (p.shape.output != .bf16 or p.shape.head_dim % 64 != 0 or xs.stride < rows or xs.stride > 64) return error.InvalidAttentionView;
        try check(xs.ref, p.shape.query_heads * p.shape.head_dim / 64 * @as(usize, xs.stride) * 4, 4);
        const own_rows: usize = if (own) |o| o.rows else 0;
        if (own_rows >= keys) return error.InvalidAttentionView;
        try p.checkViews(rows, keys, stride, if (args_at != null) 1 else scale, q, k, v, mask, out);
        if (args_at) |r| try check(r, @sizeOf(Args), 4);
        if (own) |o| {
            const kv_width = p.shape.kv_heads * p.shape.head_dim;
            if (o.k_stride < kv_width or o.v_stride < kv_width) return error.InvalidAttentionView;
            if (o.rows > 0) {
                try check(o.k, ((o.rows - 1) * @as(usize, o.k_stride) + kv_width) * 2, 2);
                try check(o.v, ((o.rows - 1) * @as(usize, o.v_stride) + kv_width) * 2, 2);
            }
        }
        const args: Bound = if (args_at) |r| .{ .at = r } else .{ .value = .{ .rows = @intCast(rows), .keys = @intCast(keys), .mask_stride = @intCast(stride), .scale_bits = @bitCast(scale) } };
        p.dispatchSplit(e, rows, keys, args, q, k, v, mask, out, xs, own);
    }

    fn dispatchSplit(p: Attention, e: mtl.ComputeEncoder, rows: usize, keys: usize, args: Bound, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref, xs: ?Sums, own: ?Own) void {
        p.dispatchSplitWith(e, p.split_merge_xs, p.arrivals, rows, keys, args, q, k, v, mask, out, xs, own);
    }
    /// The split's buffers 0-8: the views, Args, partials and own rows.
    fn bindSplit(p: Attention, e: mtl.ComputeEncoder, args: Bound, q: Ref, k: Ref, v: Ref, mask: Ref, own: ?Own) void {
        for ([_]Ref{ q, k, v, mask }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        bindArgs(e, args, 4);
        e.setBuffer(p.part, 0, 5);
        const kv_width: u32 = @intCast(p.shape.kv_heads * p.shape.head_dim);
        const o = own orelse Own{ .k = k, .v = v, .rows = 0, .k_stride = kv_width, .v_stride = kv_width };
        e.setBuffer(o.k.buf, o.k.off, 6);
        e.setBuffer(o.v.buf, o.v.off, 7);
        e.setValue([3]u32{ o.rows, o.k_stride, o.v_stride }, 8);
    }
    /// dispatchSplit, in one launch (with sums, `merged` and its `arrivals`, and at most 16 rows): each query head and row
    /// merged by its last chunk threadgroup, with no barrier and no merge launch.
    fn dispatchSplitWith(p: Attention, e: mtl.ComputeEncoder, merged: ?mtl.Pipeline, arrivals: ?mtl.Buffer, rows: usize, keys: usize, args: Bound, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref, xs: ?Sums, own: ?Own) void {
        if (merged) |pipe| if (arrivals) |counts| if (xs) |s| if (rows <= arrival_rows) {
            e.setPipeline(pipe);
            p.bindSplit(e, args, q, k, v, mask, own);
            e.setBuffer(out.buf, out.off, 9);
            e.setBuffer(s.ref.buf, s.ref.off, 10);
            e.setValue(s.stride, 11);
            e.setBuffer(counts, 0, 12);
            e.dispatchGroups(mtl.Size.of(p.shape.query_heads, rows, chunksOf(keys, p.shape.split_chunk)), mtl.Size.of(32, 1, 1));
            return;
        };
        e.setPipeline(p.split);
        p.bindSplit(e, args, q, k, v, mask, own);
        e.dispatchGroups(mtl.Size.of(p.shape.query_heads, rows, chunksOf(keys, p.shape.split_chunk)), mtl.Size.of(32, 1, 1));
        e.barrier();
        e.setPipeline(if (xs != null) p.merge_xs else p.merge);
        e.setBuffer(p.part, 0, 0);
        bindArgs(e, args, 1);
        e.setBuffer(out.buf, out.off, 2);
        if (xs) |s| {
            e.setBuffer(s.ref.buf, s.ref.off, 3);
            e.setValue(s.stride, 4);
        }
        e.dispatchGroups(mtl.Size.of(p.shape.query_heads, rows, 1), mtl.Size.of(32, 1, 1));
    }

    /// Whether `merged` (split_merge_xs on `arrivals`) writes, on this GPU, the words of the split, barrier and merge_xs it
    /// replaces: three rows of a fixed bf16 query, keys and values over 35 chunks and 3 keys (or the capacity), a mask with
    /// holes, a strided Own K/V view, and both host-value and device-backed Args. Every output and group-sum word (including
    /// rows past three) is compared bit for bit. Each path runs in its own command buffer, waited on, since both use `part`.
    fn mergedMatches(p: Attention, merged: mtl.Pipeline, arrivals: mtl.Buffer, device: mtl.Device, queue: mtl.Queue) !bool {
        const s = p.shape;
        if (s.output != .bf16 or s.head_dim % 64 != 0) return false;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const rows: usize = 3;
        const xs_stride: u32 = 16;
        const keys: usize = @min(s.capacity, 35 * s.split_chunk + 3);
        const stride = keys;
        const qw = s.query_heads * s.head_dim;
        const kvw = s.kv_heads * s.head_dim;
        const groups = qw / 64;
        const q = try device.buffer(rows * qw * 2, mtl.ResourceOptions.shared);
        defer q.deinit();
        const k = try device.buffer(keys * kvw * 2, mtl.ResourceOptions.shared);
        defer k.deinit();
        const v = try device.buffer(keys * kvw * 2, mtl.ResourceOptions.shared);
        defer v.deinit();
        const own_rows: usize = 3;
        const own_k_stride = kvw + 7;
        const own_v_stride = kvw + 13;
        const own_k_off = 8 * @sizeOf(u16);
        const own_k = try device.buffer(own_k_off + own_rows * own_k_stride * 2, mtl.ResourceOptions.shared);
        defer own_k.deinit();
        const own_v_off = 12 * @sizeOf(u16);
        const own_v = try device.buffer(own_v_off + own_rows * own_v_stride * 2, mtl.ResourceOptions.shared);
        defer own_v.deinit();
        const mask = try device.buffer(rows * stride, mtl.ResourceOptions.shared);
        defer mask.deinit();
        const args_off = @sizeOf(Args);
        const args_buffer = try device.buffer(args_off + @sizeOf(Args), mtl.ResourceOptions.shared);
        defer args_buffer.deinit();
        var outs: [2]mtl.Buffer = undefined;
        var sums: [2]mtl.Buffer = undefined;
        var made: usize = 0;
        defer for (outs[0..made], sums[0..made]) |o, x| {
            o.deinit();
            x.deinit();
        };
        while (made < 2) : (made += 1) {
            outs[made] = try device.buffer(rows * qw * 2, mtl.ResourceOptions.shared);
            sums[made] = device.buffer(groups * xs_stride * 4, mtl.ResourceOptions.shared) catch |err| {
                outs[made].deinit();
                return err;
            };
            // a NaN pattern neither path writes: a word one path leaves unwritten differs from the other's
            @memset(outs[made].slice(u16, rows * qw), 0x7fc1);
            @memset(sums[made].slice(u32, groups * xs_stride), 0x7fc00001);
        }
        fill(q.slice(u16, rows * qw), 7919, 4001, 2000.0);
        fill(k.slice(u16, keys * kvw), 104729, 3001, 1500.0);
        fill(v.slice(u16, keys * kvw), 1299709, 2003, 1000.0);
        fill(own_k.slice(u16, own_k_off / 2 + own_rows * own_k_stride)[own_k_off / 2 ..], 104729, 3001, 1500.0);
        fill(own_v.slice(u16, own_v_off / 2 + own_rows * own_v_stride)[own_v_off / 2 ..], 1299709, 2003, 1000.0);
        for (mask.slice(u8, rows * stride), 0..) |*m, i| m.* = @intFromBool((i * 7 + i / stride) % 11 != 0);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(s.head_dim)));
        const args_value = Args{ .rows = @intCast(rows), .keys = @intCast(keys), .mask_stride = @intCast(stride), .scale_bits = @bitCast(scale) };
        args_buffer.slice(Args, 2)[1] = args_value;
        const own: Own = .{ .k = .{ .buf = own_k, .off = own_k_off }, .v = .{ .buf = own_v, .off = own_v_off }, .rows = own_rows, .k_stride = @intCast(own_k_stride), .v_stride = @intCast(own_v_stride) };
        const cases = [_]struct { has_own: bool, args_at: bool }{
            .{ .has_own = false, .args_at = false },
            .{ .has_own = true, .args_at = false },
            .{ .has_own = false, .args_at = true },
            .{ .has_own = true, .args_at = true },
        };
        var check_attention = p;
        check_attention.split_merge_xs = merged;
        check_attention.arrivals = arrivals;
        for (cases) |case| {
            const bound: ?Ref = if (case.args_at) .{ .buf = args_buffer, .off = args_off } else null;
            const selected_own: ?Own = if (case.has_own) own else null;
            for (0..2) |path| {
                @memset(outs[path].slice(u16, rows * qw), 0x7fc1);
                @memset(sums[path].slice(u32, groups * xs_stride), 0x7fc00001);
                // A concurrent encoder, as the frame's: fallback's barrier orders its merge behind the split.
                var selected_attention = check_attention;
                if (path == 0) selected_attention.split_merge_xs = null;
                const cb = queue.commandBuffer();
                const e = cb.compute(.concurrent);
                try selected_attention.encodeSplitSums(e, rows, keys, stride, scale, bound, .{ .buf = q }, .{ .buf = k }, .{ .buf = v }, .{ .buf = mask }, .{ .buf = outs[path] }, .{ .ref = .{ .buf = sums[path] }, .stride = xs_stride }, selected_own);
                e.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.log.warn("draft attention merge check: {s}", .{msg});
                    return false;
                }
            }
            // The fast dispatch must drain every arrival for both cache-backed and Own K/V rows.
            if (!std.mem.allEqual(u32, arrivals.slice(u32, arrival_rows * s.query_heads), 0)) return false;
            if (!std.mem.eql(u16, outs[0].slice(u16, rows * qw), outs[1].slice(u16, rows * qw)) or
                !std.mem.eql(u32, sums[0].slice(u32, groups * xs_stride), sums[1].slice(u32, groups * xs_stride))) return false;
        }
        return true;
    }
    /// bf16 words of moderate values: ((i * step + 13) % period - range) / 1000, truncated to bf16.
    fn fill(words: []u16, step: usize, period: usize, range: f32) void {
        for (words, 0..) |*w, i| {
            const f: f32 = (@as(f32, @floatFromInt((i * step + 13) % period)) - range) / 1000.0;
            w.* = @intCast(@as(u32, @bitCast(f)) >> 16);
        }
    }

    /// Q is rotated, cached K was rotated at append; no causal mask is inferred: mask[row*stride+key] decides.
    pub fn encode(p: Attention, e: mtl.ComputeEncoder, rows: usize, keys: usize, stride: usize, scale: f32, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref) !void {
        try p.checkViews(rows, keys, stride, scale, q, k, v, mask, out);
        p.dispatchAttend(e, rows, .{ .value = .{ .rows = @intCast(rows), .keys = @intCast(keys), .mask_stride = @intCast(stride), .scale_bits = @bitCast(scale) } }, q, k, v, mask, out);
    }

    /// encode on the Args an earlier dispatch wrote at `args`, bounded as encodeSplitAt's.
    pub fn encodeAt(p: Attention, e: mtl.ComputeEncoder, rows: usize, keys: usize, stride: usize, args: Ref, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref) !void {
        try p.checkViews(rows, keys, stride, 1, q, k, v, mask, out);
        try check(args, @sizeOf(Args), 4);
        p.dispatchAttend(e, rows, .{ .at = args }, q, k, v, mask, out);
    }

    fn dispatchAttend(p: Attention, e: mtl.ComputeEncoder, rows: usize, args: Bound, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref) void {
        e.setPipeline(p.attend);
        for ([_]Ref{ q, k, v, mask }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        bindArgs(e, args, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.dispatchGroups(mtl.Size.of(p.shape.query_heads, rows, 1), mtl.Size.of(32, 1, 1));
    }

    fn checkViews(p: Attention, rows: usize, keys: usize, stride: usize, scale: f32, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref) !void {
        if (rows == 0 or rows > 16 or keys == 0 or keys > p.shape.capacity or stride < keys or stride > std.math.maxInt(u32) or
            !std.math.isFinite(scale)) return error.InvalidAttentionView;
        try check(q, rows * p.shape.query_heads * p.shape.head_dim * 2, 2);
        try check(out, rows * p.shape.query_heads * p.shape.head_dim * p.outputBytes(), p.outputBytes());
        try check(k, keys * p.shape.kv_heads * p.shape.head_dim * 2, 2);
        try check(v, keys * p.shape.kv_heads * p.shape.head_dim * 2, 2);
        try check(mask, try std.math.mul(usize, rows, stride), 1);
        try disjoint(out, q);
        try disjoint(out, k);
        try disjoint(out, v);
        try disjoint(out, mask);
    }

    fn outputBytes(p: Attention) usize {
        return if (p.shape.output == .f32) 4 else 2;
    }

    /// Ordinary one-sequence decode, all keys visible. Its fp32 arithmetic is exactly the shared row's.
    pub fn ordinary(p: Attention, e: mtl.ComputeEncoder, keys: usize, scale: f32, q: Ref, k: Ref, v: Ref, out: Ref) !void {
        if (keys == 0 or keys > p.shape.capacity or !std.math.isFinite(scale)) return error.InvalidAttentionView;
        try check(q, p.shape.query_heads * p.shape.head_dim * 2, 2);
        try check(out, p.shape.query_heads * p.shape.head_dim * p.outputBytes(), p.outputBytes());
        try check(k, keys * p.shape.kv_heads * p.shape.head_dim * 2, 2);
        try check(v, keys * p.shape.kv_heads * p.shape.head_dim * 2, 2);
        try disjoint(out, q);
        try disjoint(out, k);
        try disjoint(out, v);
        e.setPipeline(p.decode);
        for ([_]Ref{ q, k, v }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Args{ .rows = 1, .keys = @intCast(keys), .scale_bits = @bitCast(scale) }, 3);
        e.setBuffer(out.buf, out.off, 4);
        e.dispatchGroups(mtl.Size.of(p.shape.query_heads, 1, 1), mtl.Size.of(32, 1, 1));
    }

    /// positions u32[rows], frequencies f32[rotary_dim/2], storage bf16[rows,heads,D]; prompt chunks too.
    pub fn rotate(p: Attention, e: mtl.ComputeEncoder, rows: usize, heads: usize, x: Ref, positions: Ref, frequencies: Ref, y: Ref) !void {
        if (rows == 0 or rows > p.shape.capacity or (heads != p.shape.query_heads and heads != p.shape.kv_heads)) return error.InvalidRopeView;
        const elements = try std.math.mul(usize, rows, heads * p.shape.head_dim);
        if (elements > std.math.maxInt(u32)) return error.InvalidRopeView;
        try check(x, elements * 2, 2);
        try check(y, elements * 2, 2);
        try check(positions, rows * 4, 4);
        try check(frequencies, @max(p.shape.rotary_dim / 2 * 4, 4), 4);
        try disjoint(x, y);
        try disjoint(y, positions);
        try disjoint(y, frequencies);
        e.setPipeline(p.rope);
        for ([_]Ref{ x, positions, frequencies }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Args{ .rows = @intCast(rows), .heads = @intCast(heads) }, 3);
        e.setBuffer(y.buf, y.off, 4);
        e.dispatchThreads(mtl.Size.of(elements, 1, 1), mtl.Size.of(128, 1, 1));
    }

    /// Prepared K and unmodified V go directly to unique physical slots; caller orders subsequent attention.
    /// appendKv from row-strided sources (element strides), no overlap checks beyond the cache's.
    pub fn appendKvStrided(p: Attention, e: mtl.ComputeEncoder, slots: []const u32, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref, k_stride: u32, v_stride: u32) !void {
        try p.checkSlots(slots);
        try p.stridedViews(slots.len, k, v, cache_k, cache_v, k_stride, v_stride);
        p.dispatchAppend(e, p.append_strided, slots.len, .{ .host = slots }, .{ .value = .{ .rows = @intCast(slots.len) } }, k, v, cache_k, cache_v, .{ k_stride, v_stride });
    }
    /// appendKvStrided taking its row count from the Args an earlier dispatch wrote at `rows` (at most slots.len):
    /// rows past it are not written.
    pub fn appendKvStridedRows(p: Attention, e: mtl.ComputeEncoder, slots: []const u32, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref, k_stride: u32, v_stride: u32, rows: Ref) !void {
        try p.checkSlots(slots);
        try p.stridedViews(slots.len, k, v, cache_k, cache_v, k_stride, v_stride);
        try check(rows, @sizeOf(Args), 4);
        p.dispatchAppend(e, p.append_strided, slots.len, .{ .host = slots }, .{ .at = rows }, k, v, cache_k, cache_v, .{ k_stride, v_stride });
    }
    /// appendKvStrided of `count` rows to the u32 slots an earlier dispatch wrote at `slots`, which the writer keeps
    /// distinct and inside the cache.
    pub fn appendKvStridedAt(p: Attention, e: mtl.ComputeEncoder, count: usize, slots: Ref, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref, k_stride: u32, v_stride: u32) !void {
        if (count == 0 or count > 16) return error.InvalidAppendView;
        try p.stridedViews(count, k, v, cache_k, cache_v, k_stride, v_stride);
        try check(slots, count * 4, 4);
        p.dispatchAppend(e, p.append_strided, count, .{ .at = slots }, .{ .value = .{ .rows = @intCast(count) } }, k, v, cache_k, cache_v, .{ k_stride, v_stride });
    }
    pub fn appendKv(p: Attention, e: mtl.ComputeEncoder, slots: []const u32, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref) !void {
        try p.checkSlots(slots);
        try p.packedViews(slots.len, k, v, cache_k, cache_v);
        p.dispatchAppend(e, p.append, slots.len, .{ .host = slots }, .{ .value = .{ .rows = @intCast(slots.len) } }, k, v, cache_k, cache_v, null);
    }
    /// appendKv taking its row count from the Args an earlier dispatch wrote at `rows` (at most slots.len).
    pub fn appendKvRows(p: Attention, e: mtl.ComputeEncoder, slots: []const u32, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref, rows: Ref) !void {
        try p.checkSlots(slots);
        try p.packedViews(slots.len, k, v, cache_k, cache_v);
        try check(rows, @sizeOf(Args), 4);
        p.dispatchAppend(e, p.append, slots.len, .{ .host = slots }, .{ .at = rows }, k, v, cache_k, cache_v, null);
    }

    fn checkSlots(p: Attention, slots: []const u32) !void {
        if (slots.len == 0 or slots.len > 16) return error.InvalidAppendView;
        for (slots, 0..) |slot, i| {
            if (slot >= p.shape.capacity) return error.InvalidAppendSlot;
            for (slots[0..i]) |prior| if (prior == slot) return error.DuplicateAppendSlot;
        }
    }
    fn stridedViews(p: Attention, count: usize, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref, k_stride: u32, v_stride: u32) !void {
        const width = p.shape.kv_heads * p.shape.head_dim;
        if (k_stride < width or v_stride < width) return error.InvalidAppendView;
        try check(k, ((count - 1) * k_stride + width) * 2, 2);
        try check(v, ((count - 1) * v_stride + width) * 2, 2);
        try check(cache_k, p.shape.capacity * width * 2, 2);
        try check(cache_v, p.shape.capacity * width * 2, 2);
    }
    fn packedViews(p: Attention, count: usize, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref) !void {
        const width = p.shape.kv_heads * p.shape.head_dim;
        try check(k, count * width * 2, 2);
        try check(v, count * width * 2, 2);
        try check(cache_k, p.shape.capacity * width * 2, 2);
        try check(cache_v, p.shape.capacity * width * 2, 2);
        try disjoint(k, cache_k);
        try disjoint(v, cache_k);
        try disjoint(k, cache_v);
        try disjoint(v, cache_v);
        try disjoint(cache_k, cache_v);
    }
    /// One append dispatch of `count` rows' threads; the kernel writes the rows its Args name.
    fn dispatchAppend(p: Attention, e: mtl.ComputeEncoder, pipeline: mtl.Pipeline, count: usize, slots: union(enum) { host: []const u32, at: Ref }, args: Bound, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref, strides: ?[2]u32) void {
        const width = p.shape.kv_heads * p.shape.head_dim;
        e.setPipeline(pipeline);
        e.setBuffer(k.buf, k.off, 0);
        e.setBuffer(v.buf, v.off, 1);
        switch (slots) {
            .host => |s| e.setBytes(std.mem.sliceAsBytes(s), 2),
            .at => |r| e.setBuffer(r.buf, r.off, 2),
        }
        bindArgs(e, args, 3);
        e.setBuffer(cache_k.buf, cache_k.off, 4);
        e.setBuffer(cache_v.buf, cache_v.off, 5);
        if (strides) |s| e.setValue(s, 6);
        e.dispatchThreads(mtl.Size.of(count * width, 1, 1), mtl.Size.of(128, 1, 1));
    }
};

fn check(r: Ref, need: usize, alignment: usize) !void {
    if (r.off % alignment != 0) return error.MisalignedAttentionBuffer;
    const len = r.buf.length();
    if (r.off > len or need > len - r.off) return error.AttentionBufferTooSmall;
}
fn disjoint(a: Ref, b: Ref) !void {
    // Conservative: use separate Metal resources for read/write operations, even at disjoint byte offsets.
    if (a.buf.id == b.buf.id) return error.AliasedAttentionBuffer;
}

test "attention shapes keep grouped-query and rotary dimensions explicit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const good = Shape{ .query_heads = 4, .kv_heads = 2, .head_dim = 64, .rotary_dim = 32, .capacity = 128 };
    const text = try source(arena.allocator(), good);
    try std.testing.expect(std.mem.indexOf(u8, text, "#define TF_QH 4\n#define TF_KVH 2\n") != null);
    var bad = good;
    bad.query_heads = 3;
    try std.testing.expectError(error.InvalidAttentionShape, bad.validate());
    bad = good;
    bad.rotary_dim = 33;
    try std.testing.expectError(error.InvalidAttentionShape, bad.validate());
    bad = good;
    bad.head_dim = 31;
    try std.testing.expectError(error.InvalidAttentionShape, bad.validate());
}
