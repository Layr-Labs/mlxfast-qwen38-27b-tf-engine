//! Allocation geometry shared by prompt admission and the actual attention scratch owner.
const std = @import("std");
const planning = @import("attention_plan.zig");
pub const Layout = struct {
    tiles: u32,
    chunks: u32,
    tail_chunks: u32,
    sizes: [15]usize,
    pub fn init(limits: anytype) Layout {
        const tiles = (limits.rows * limits.geometry.group() + 15) / 16 + limits.streams;
        const chunks = @max((limits.keys + 511) / 512, (planning.fine_until + 128 + planning.fine_chunk - 1) / planning.fine_chunk); // fine chunks below fine_until
        const tail_chunks = if (limits.shared_prefix) 3 else chunks; // a tail of up to 191 keys over 128-key chunks
        return .{ .tiles = tiles, .chunks = chunks, .tail_chunks = tail_chunks, .sizes = offsets(limits, chunks, tail_chunks, limits.rows, tiles) };
    }
};

/// Each buffer's bytes for `rows` rows and `tiles` tiles at the allocation's chunks: its size at the allocation's
/// counts, and where a pass behind that many rows and tiles of another pass starts.
pub fn offsets(limits: anytype, chunks: u32, tail_chunks: u32, rows: usize, tiles: usize) [15]usize {
    const hk: usize = limits.geometry.kv_heads;
    const dim: usize = limits.geometry.dim;
    const rp: usize = tiles * 16;
    const a: usize = if (limits.shared_prefix) hk * @as(usize, chunks) * rp else 0;
    const b: usize = hk * @as(usize, tail_chunks) * rows * 16;
    return .{ tiles * 4, rp * 4, rows * planning.node_words * 4, rows * planning.path_width * 4, rows * 4, hk * rp * dim * 2, hk * rows * 16 * dim * 2, a * dim * 4, a * 4, a * 4, b * dim * 4, b * 4, b * 4, rows * hk * dim * 2, rows * hk * dim * 2 };
}

test "a region behind another pass's rows and tiles starts where its bytes end and stays inside the allocation" {
    const limits = .{ .rows = @as(u32, 2048), .streams = @as(u32, 2), .keys = @as(u32, 32784), .shared_prefix = true, .geometry = planning.Geometry{} };
    const l = Layout.init(limits);
    try std.testing.expectEqual(l.sizes, offsets(limits, l.chunks, l.tail_chunks, limits.rows, l.tiles));
    // a 1065-row chunk split 512 and 553: the second segment's region behind the first's 192 tiles, its own 208 tiles
    const first = offsets(limits, l.chunks, l.tail_chunks, 512, 192);
    const second = offsets(limits, l.chunks, l.tail_chunks, 553, 208);
    for (first, second, l.sizes) |at, bytes, size| try std.testing.expect(at + bytes <= size);
}
