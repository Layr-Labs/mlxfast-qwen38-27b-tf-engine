//! Pure integer conversion from the promoted lane register layout to native Q4 packed words.
const std = @import("std");

/// Convert p.words after lane_projection.regOrder into [tile][K/64][column][8 K nibbles].
/// The existing checkpoint and p.words remain untouched; caller owns the returned slice.
pub fn fromReg(allocator: std.mem.Allocator, reg_words: []const u32, n: usize, k: usize, batch: usize) ![]u32 {
    if (n == 0 or k == 0 or batch == 0 or n % 32 != 0 or k % 64 != 0) return error.InvalidShape;
    const batch_columns = try std.math.mul(usize, 32, batch);
    if (n % batch_columns != 0) return error.InvalidShape;
    const tiles = n / 32;
    const groups = k / 64;
    const blocks = try std.math.mul(usize, tiles, groups);
    const expected_words = try std.math.mul(usize, blocks, 256);
    if (reg_words.len != expected_words) return error.InvalidStorage;
    const out = try allocator.alloc(u32, expected_words);
    errdefer allocator.free(out);
    @memset(out, 0);

    for (0..tiles) |tile| for (0..groups) |group| {
        // Reg batches are stored as [batch-start][K-group][tile-within-batch].
        const block = (((tile / batch) * groups + group) * batch + tile % batch) * 256;
        for (0..32) |lane| for (0..8) |w| {
            const reg_word = block + (w / 4) * 128 + lane * 4 + (w % 4);
            const column = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1);
            for (0..8) |j| {
                const q = j & 3;
                const k_index = 4 * (lane & 1) + 8 * ((lane >> 3) & 1) + 16 * (2 * (w & 1) + (q >> 1)) + 2 * (q & 1) + (j >> 2);
                const code = (reg_words[reg_word] >> @intCast(4 * j)) & 0xF;
                const output_index = ((tile * groups + group) * 32 + column) * 8 + k_index / 8;
                out[output_index] |= code << @intCast(4 * (k_index % 8));
            }
        };
    };
    return out;
}

fn makeFixture(allocator: std.mem.Allocator, n: usize, k: usize, batch: usize) !struct { reg: []u32, canonical: []u32 } {
    const tiles = n / 32;
    const groups = k / 64;
    const word_count = tiles * groups * 256;
    const reg = try allocator.alloc(u32, word_count);
    errdefer allocator.free(reg);
    const canonical = try allocator.alloc(u32, word_count);
    errdefer allocator.free(canonical);
    @memset(reg, 0);
    @memset(canonical, 0);
    for (0..tiles) |tile| for (0..groups) |group| {
        const canonical_base = (tile * groups + group) * 256;
        const reg_base = (((tile / batch) * groups + group) * batch + tile % batch) * 256;
        for (0..32) |column| for (0..64) |ki| {
            const ordinal = (tile * groups + group) * 2048 + column * 64 + ki;
            const code: u32 = @intCast((ordinal *% 13 +% ordinal / 7 +% 5) & 15);
            canonical[canonical_base + column * 8 + ki / 8] |= code << @intCast(4 * (ki % 8));
        };
        // Build the known register fixture by distributing each logical nibble to its documented lane register.
        for (0..32) |lane| for (0..8) |w| {
            const reg_word = reg_base + (w / 4) * 128 + lane * 4 + (w % 4);
            const column = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1);
            for (0..8) |j| {
                const q = j & 3;
                const ki = 4 * (lane & 1) + 8 * ((lane >> 3) & 1) + 16 * (2 * (w & 1) + (q >> 1)) + 2 * (q & 1) + (j >> 2);
                const code = (canonical[canonical_base + column * 8 + ki / 8] >> @intCast(4 * (ki % 8))) & 15;
                reg[reg_word] |= code << @intCast(4 * j);
            }
        };
    };
    return .{ .reg = reg, .canonical = canonical };
}

test "reg adapter preserves every code across batches and K groups" {
    const allocator = std.testing.allocator;
    inline for (.{ 1, 2, 4, 8 }) |batch| {
        const fixture = try makeFixture(allocator, 512, 192, batch);
        defer allocator.free(fixture.reg);
        defer allocator.free(fixture.canonical);
        const actual = try fromReg(allocator, fixture.reg, 512, 192, batch);
        defer allocator.free(actual);
        try std.testing.expectEqualSlices(u32, fixture.canonical, actual);
        var codes: u16 = 0;
        for (actual) |word| for (0..8) |i| {
            codes |= @as(u16, 1) << @intCast((word >> @intCast(4 * i)) & 15);
        };
        try std.testing.expectEqual(@as(u16, 0xFFFF), codes);
    }
}

test "reg adapter rejects invalid dimensions, strides, lengths and overflow" {
    const allocator = std.testing.allocator;
    const valid: [256]u32 = @splat(0);
    try std.testing.expectError(error.InvalidShape, fromReg(allocator, &valid, 33, 64, 1));
    try std.testing.expectError(error.InvalidShape, fromReg(allocator, &valid, 32, 65, 1));
    try std.testing.expectError(error.InvalidShape, fromReg(allocator, &valid, 32, 64, 0));
    try std.testing.expectError(error.InvalidShape, fromReg(allocator, &valid, 32, 64, 2));
    try std.testing.expectError(error.InvalidStorage, fromReg(allocator, valid[0..255], 32, 64, 1));
    try std.testing.expectError(error.Overflow, fromReg(allocator, &.{}, std.math.maxInt(usize) - 31, std.math.maxInt(usize) - 63, 1));
}
