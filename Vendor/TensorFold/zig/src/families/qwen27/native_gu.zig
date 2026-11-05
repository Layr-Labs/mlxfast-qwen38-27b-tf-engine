//! Pure policy, canonical Q4 packing and byte offsets for the optional prompt GU.
const std = @import("std");
pub const n: u32 = 34816;
pub const k: u32 = 5120;
pub const columns: u32 = n / 2;
pub const groups: u32 = k / 64;
pub const bytes: usize = @as(usize, n) * k / 2;
pub const up_word_offset: usize = bytes / 2;
pub const up_pair_offset: usize = @as(usize, columns) * 4;
pub const Part = struct { words: []const u32, rows: usize };
pub const Params = extern struct { rows: u32, x_stride: u32, sb_stride: u32, y_stride: u32, sum_stride: u32, tiles: u32 };

pub fn enabled(text: ?[]const u8, disabled: bool) bool {
    return !disabled and if (text) |v| std.mem.eql(u8, v, "1") else true;
}
pub fn admit(requested: bool, units: bool, out: usize, inner: usize, bits: u8, group: usize, equal_halves: bool) bool {
    return requested and units and out == n and inner == k and bits == 4 and group == 64 and equal_halves;
}
pub fn route(present: bool, units: bool, rows: u32, out: u32, inner: u32, bits: u8, group: u32, tile: u32, runs: usize, raw: bool) bool {
    return present and units and rows > 0 and out == n and inner == k and bits == 4 and group == 64 and tile == 32 and runs == 0 and !raw;
}

/// [tile64][group64][half32][column64][16 bytes]. Copy original low-nibble-first bytes, never reg words.
pub fn pack(out: []u8, parts: []const Part, cols: usize, inner: usize) !void {
    if (cols == 0 or inner == 0 or cols % 64 != 0 or inner % 64 != 0) return error.BadNativeGUShape;
    if (out.len != try std.math.mul(usize, cols, inner) / 2) return error.BadNativeGUShape;
    var total: usize = 0;
    for (parts) |part| {
        if (part.words.len != try std.math.mul(usize, part.rows, inner / 8)) return error.BadNativeGUShape;
        total = try std.math.add(usize, total, part.rows);
    }
    if (total != cols) return error.BadNativeGUShape;
    var base: usize = 0;
    for (parts) |part| {
        const src = std.mem.sliceAsBytes(part.words);
        for (0..part.rows) |r| for (0..inner / 64) |g| for (0..2) |half| {
            const row = base + r;
            const from = r * (inner / 2) + g * 32 + half * 16;
            const to = ((((row / 64) * (inner / 64) + g) * 2 + half) * 64 + row % 64) * 16;
            @memcpy(out[to..][0..16], src[from..][0..16]);
        };
        base += part.rows;
    }
}
pub fn promptParams(chunk_rows: u32, original_stride: u32) Params {
    return .{ .rows = chunk_rows, .x_stride = k, .sb_stride = n, .y_stride = columns, .sum_stride = original_stride, .tiles = n / 128 };
}
pub const Chunk = struct { rows: u32, x: usize, y: usize, sums: usize };
pub fn chunk(first: u32, rows: u32) Chunk {
    return .{ .rows = @min(rows -| first, 128), .x = @as(usize, first) * k * 2, .y = @as(usize, first) * columns * 2, .sums = @as(usize, first) * 4 };
}

test "native GU requests device qualification by default and preserves explicit opt-out" {
    const t = std.testing;
    try t.expect(enabled(null, false));
    try t.expect(!enabled(null, true));
    try t.expect(enabled("1", false));
    try t.expect(!enabled("1", true));
    for ([_][]const u8{ "0", "", "true", "2" }) |v| try t.expect(!enabled(v, false));
    try t.expect(admit(true, true, n, k, 4, 64, true));
    try t.expect(!admit(false, true, n, k, 4, 64, true));
    try t.expect(!admit(true, false, n, k, 4, 64, true));
    try t.expect(!admit(true, true, n, k, 4, 64, false));
    try t.expect(!admit(true, true, n - 1, k, 4, 64, true));
    try t.expect(!admit(true, true, n, k - 64, 4, 64, true));
    try t.expect(!admit(true, true, n, k, 2, 64, true));
    try t.expect(!admit(true, true, n, k, 4, 32, true));
    try t.expect(route(true, true, 361, n, k, 4, 64, 32, 0, false));
    try t.expect(!route(false, true, 361, n, k, 4, 64, 32, 0, false));
    try t.expect(!route(true, false, 361, n, k, 4, 64, 32, 0, false));
    try t.expect(!route(true, true, 0, n, k, 4, 64, 32, 0, false));
    try t.expect(!route(true, true, 361, n, k, 4, 64, 32, 1, false));
    try t.expect(!route(true, true, 361, n, k, 4, 64, 32, 0, true));
}

test "native GU canonical pack preserves every nibble across tiles groups halves and stack seam" {
    const a = std.testing.allocator;
    const cols = 128;
    const inner = 5120;
    const source = try a.alloc(u32, cols * inner / 8);
    defer a.free(source);
    @memset(source, 0);
    for (0..cols) |row| for (0..inner) |col| {
        const code: u32 = @intCast((row * 7 + col * 11 + row * col % 13) & 15);
        source[row * (inner / 8) + col / 8] |= code << @as(u5, @intCast((col % 8) * 4));
    };
    const storage = try a.alloc(u8, cols * inner / 2 + 32);
    defer a.free(storage);
    @memset(storage, 0xa5);
    const dst = storage[16..][0 .. cols * inner / 2];
    try pack(dst, &.{ .{ .words = source[0 .. 48 * inner / 8], .rows = 48 }, .{ .words = source[48 * inner / 8 ..], .rows = 80 } }, cols, inner);
    for (0..cols) |row| for (0..inner) |col| {
        const at = ((((row / 64) * (inner / 64) + col / 64) * 2 + (col % 64) / 32) * 64 + row % 64) * 16 + (col % 32) / 2;
        const got = (dst[at] >> @as(u3, @intCast((col & 1) * 4))) & 15;
        const expected: u8 = @intCast((row * 7 + col * 11 + row * col % 13) & 15);
        try std.testing.expectEqual(expected, got);
    };
    for (storage[0..16]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
    for (storage[storage.len - 16 ..]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
    try std.testing.expectError(error.BadNativeGUShape, pack(dst[1..], &.{.{ .words = source, .rows = cols }}, cols, inner));
    try std.testing.expectError(error.BadNativeGUShape, pack(dst, &.{.{ .words = source[1..], .rows = cols }}, cols, inner));
}

test "native GU prompt chunk offsets retain global sums stride and tail bounds" {
    const t = std.testing;
    const middle = chunk(128, 361);
    const tail = chunk(256, 361);
    try t.expectEqual(@as(u32, 128), middle.rows);
    try t.expectEqual(@as(usize, 128 * 5120 * 2), middle.x);
    try t.expectEqual(@as(usize, 128 * 17408 * 2), middle.y);
    try t.expectEqual(@as(usize, 128 * 4), middle.sums);
    try t.expectEqual(@as(u32, 105), tail.rows);
    try t.expectEqual(@as(usize, 256 * 4), tail.sums);
    try t.expectEqual(@as(u32, 0), chunk(384, 361).rows);
    const params = promptParams(tail.rows, 361);
    try t.expectEqual(@as(u32, 105), params.rows);
    try t.expectEqual(@as(u32, 361), params.sum_stride);
    try t.expectEqual(@as(u32, 272), params.tiles);
    try t.expectEqual(@as(usize, 24), @sizeOf(Params));
    try t.expectEqual(@as(usize, 44564480), up_word_offset);
    try t.expectEqual(@as(usize, 69632), up_pair_offset);
}

/// Only before frames/residency exist and after every startup command buffer is drained.
/// Duplicate optional references and duplicate owned IDs are cleared/released exactly once.
pub fn releaseOwned(layers: anytype, owned: anytype) void {
    for (layers) |*layer| {
        const ref = layer.gu.native_gu orelse continue;
        const id = ref.buffer.id;
        var original = false;
        for (layers) |*other| {
            original = original or other.gu.words.buffer.id == id or other.gu.pairs.buffer.id == id;
            if (other.gu.native_gu) |value| if (value.buffer.id == id) {
                other.gu.native_gu = null;
            };
        }
        // A malformed optional alias must never release original matrix/metadata storage.
        if (original) continue;
        var released = false;
        var i: usize = 0;
        while (i < owned.items.len) {
            if (owned.items[i].id == id) {
                const buffer = owned.orderedRemove(i);
                if (!released) {
                    buffer.deinit();
                    released = true;
                }
            } else i += 1;
        }
    }
}

test "native GU actual optional ownership release clears aliases once and preserves originals" {
    const B = struct {
        id: u32,
        calls: *[4]u32,
        count: *usize,
        fn deinit(b: @This()) void {
            b.calls[b.count.*] = b.id;
            b.count.* += 1;
        }
    };
    const R = struct { buffer: B };
    const L = struct { gu: struct { words: R, pairs: R, native_gu: ?R } };
    var calls: [4]u32 = @splat(0);
    var count: usize = 0;
    const side = B{ .id = 10, .calls = &calls, .count = &count };
    const words = B{ .id = 20, .calls = &calls, .count = &count };
    const pairs = B{ .id = 30, .calls = &calls, .count = &count };
    const unrelated = B{ .id = 40, .calls = &calls, .count = &count };
    const borrowed = B{ .id = 99, .calls = &calls, .count = &count };
    var layers = [_]L{
        .{ .gu = .{ .words = .{ .buffer = words }, .pairs = .{ .buffer = pairs }, .native_gu = .{ .buffer = side } } },
        .{ .gu = .{ .words = .{ .buffer = words }, .pairs = .{ .buffer = pairs }, .native_gu = .{ .buffer = side } } },
        .{ .gu = .{ .words = .{ .buffer = words }, .pairs = .{ .buffer = pairs }, .native_gu = .{ .buffer = borrowed } } },
        .{ .gu = .{ .words = .{ .buffer = words }, .pairs = .{ .buffer = pairs }, .native_gu = .{ .buffer = words } } },
    };
    var owned: std.ArrayList(B) = .empty;
    defer owned.deinit(std.testing.allocator);
    try owned.appendSlice(std.testing.allocator, &.{ side, words, side, pairs, unrelated });
    releaseOwned(&layers, &owned);
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expectEqual(@as(u32, 10), calls[0]);
    try std.testing.expectEqual(@as(usize, 3), owned.items.len);
    for (layers) |l| try std.testing.expect(l.gu.native_gu == null);
    for (owned.items, [_]u32{ 20, 30, 40 }) |value, id| try std.testing.expectEqual(id, value.id);
    releaseOwned(&layers, &owned);
    try std.testing.expectEqual(@as(usize, 1), count);
}
