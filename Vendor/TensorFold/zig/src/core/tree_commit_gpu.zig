//! Verified GPU results drive accepted-path metadata, tap gathering and final-logit restoration.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("tree_round");
const source = @import("tree_commit_sources").text;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Meta = extern struct { rows: u32, slot: u32 };
pub const Taps = extern struct { rows: u32, width: u32, capacity: u32, planes: u32 };
pub const Head = extern struct { rows: u32, width: u32, stride: u32 };
/// tf_round_select: `blocks` host blocks of `words` u32 words, the last token at word `token`, the status at `status`.
pub const Select = extern struct { rows: u32, words: u32, blocks: u32, token: u32, status: u32, fallback: u32 };
pub const Ops = struct {
    metadata_pipe: mtl.Pipeline,
    taps_pipe: mtl.Pipeline,
    head_pipe: mtl.Pipeline,
    select_pipe: mtl.Pipeline,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const metadata_pipe = try mtl.Pipeline.init(device, lib, "tf_round_keep_metadata", false);
        errdefer metadata_pipe.deinit();
        const taps_pipe = try mtl.Pipeline.init(device, lib, "tf_round_keep_taps", false);
        errdefer taps_pipe.deinit();
        const head_pipe = try mtl.Pipeline.init(device, lib, "tf_round_keep_head", false);
        errdefer head_pipe.deinit();
        return .{ .metadata_pipe = metadata_pipe, .taps_pipe = taps_pipe, .head_pipe = head_pipe, .select_pipe = try mtl.Pipeline.init(device, lib, "tf_round_select", false) };
    }
    pub fn deinit(ops: Ops) void {
        ops.metadata_pipe.deinit();
        ops.taps_pipe.deinit();
        ops.head_pipe.deinit();
        ops.select_pipe.deinit();
    }
    pub fn metadata(ops: Ops, e: mtl.ComputeEncoder, result: Ref, keep: Ref, rows: Ref, map: ?Ref, p: Meta) !void {
        if (p.rows == 0 or p.rows > contract.max_rows or p.slot >= 64) return error.BadCommitShape;
        try check(result, @sizeOf(contract.Result), 4);
        const outputs = [_]Ref{ keep, rows, map orelse rows };
        const sizes = [_]usize{ 16, @as(usize, p.rows) * 4, @as(usize, p.rows) * 16 };
        const count: usize = if (map != null) 3 else 2;
        for (outputs[0..count], sizes[0..count], 0..) |out, bytes, i| {
            try check(out, bytes, 4);
            try distinct(out, bytes, result, @sizeOf(contract.Result));
            for (outputs[0..i], sizes[0..i]) |prior, prior_bytes| try distinct(out, bytes, prior, prior_bytes);
        }
        e.setPipeline(ops.metadata_pipe);
        bind(e, .{ result, keep, rows, map orelse rows });
        e.setValue(p, 4);
        e.setValue(@as(u32, @intFromBool(map != null)), 5);
        e.dispatchThreads(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(32, 1, 1));
    }
    pub fn taps(ops: Ops, e: mtl.ComputeEncoder, result: Ref, input: Ref, out: Ref, p: Taps) !void {
        return ops.tapsInto(e, result, input, out, p, p.rows);
    }
    /// taps into room for `room` of the round's rows: rows past it are not written, however many the result kept.
    pub fn tapsInto(ops: Ops, e: mtl.ComputeEncoder, result: Ref, input: Ref, out: Ref, p: Taps, room: u32) !void {
        if (p.rows == 0 or p.rows > contract.max_rows or p.rows > p.capacity or p.width == 0 or p.planes == 0 or p.planes > 8 or room == 0 or room > p.rows) return error.BadCommitShape;
        const width = try std.math.mul(usize, p.width, p.planes);
        const in_bytes = try std.math.mul(usize, try std.math.mul(usize, p.capacity, width), 2);
        const out_bytes = try std.math.mul(usize, try std.math.mul(usize, room, width), 2);
        try check(result, @sizeOf(contract.Result), 4);
        try check(input, in_bytes, 2);
        try check(out, out_bytes, 2);
        try distinct(out, out_bytes, input, in_bytes);
        try distinct(out, out_bytes, result, @sizeOf(contract.Result));
        e.setPipeline(ops.taps_pipe);
        bind(e, .{ result, input, out });
        e.setValue(p, 3);
        e.dispatchThreads(mtl.Size.of(width, room, 1), mtl.Size.of(256, 1, 1));
    }
    pub fn head(ops: Ops, e: mtl.ComputeEncoder, result: Ref, logits: Ref, p: Head) !void {
        if (p.rows == 0 or p.rows > contract.max_rows or p.width == 0 or p.stride < p.width) return error.BadCommitShape;
        const cells = try std.math.add(usize, try std.math.mul(usize, p.rows - 1, p.stride), p.width);
        const bytes = try std.math.mul(usize, cells, 2);
        try check(result, @sizeOf(contract.Result), 4);
        try check(logits, bytes, 2);
        try distinct(logits, bytes, result, @sizeOf(contract.Result));
        e.setPipeline(ops.head_pipe);
        bind(e, .{ result, logits });
        e.setValue(p, 2);
        e.dispatchThreads(mtl.Size.of(p.width, 1, 1), mtl.Size.of(256, 1, 1));
    }
    /// Block `consumed_count` of `p.blocks` (the host's blocks behind block 0 in `blocks`) into block 0, with the result's
    /// last emitted token at word `p.token` and its status at `p.status` (0 valid; 1: block 1 and `p.fallback` taken).
    pub fn select(ops: Ops, e: mtl.ComputeEncoder, result: Ref, blocks: Ref, p: Select) !void {
        if (p.rows == 0 or p.rows > contract.max_rows or p.words == 0 or p.blocks == 0 or p.token >= p.words or p.status >= p.words or p.token == p.status) return error.BadCommitShape;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, p.blocks + 1, p.words), 4);
        try check(result, @sizeOf(contract.Result), 4);
        try check(blocks, bytes, 4);
        try distinct(blocks, bytes, result, @sizeOf(contract.Result));
        e.setPipeline(ops.select_pipe);
        bind(e, .{ result, blocks });
        e.setValue(p, 2);
        e.dispatchThreads(mtl.Size.of(p.words, 1, 1), mtl.Size.of(256, 1, 1));
    }
};
fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}
fn check(r: Ref, bytes: usize, alignment: usize) !void {
    if (r.off % alignment != 0 or r.off > r.buf.length() or bytes > r.buf.length() - r.off) return error.BadCommitBuffer;
}
fn distinct(out: Ref, bytes: usize, input: Ref, input_bytes: usize) !void {
    if (out.buf.id == input.buf.id and out.off < input.off + input_bytes and input.off < out.off + bytes) return error.CommitAlias;
}
