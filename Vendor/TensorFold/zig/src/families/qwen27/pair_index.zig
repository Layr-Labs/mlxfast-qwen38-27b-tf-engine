//! A reg matrix's (scale, bias) words as a u16 index into its table of distinct words: lossless, built at load.
const std = @import("std");

/// Distinct (scale, bias) words a table may hold; a matrix with more keeps only its pairs.
pub const table_limit = 16384;

/// Each u32 (scale, bias) word's index into `table`, distinct words in first-seen order; null when more than
/// table.len words are distinct. `slots` (a power of two of at least twice table.len) is scratch.
pub fn indexPairs(words: []const u32, index: []u16, table: []u32, slots: []u32) ?usize {
    std.debug.assert(index.len == words.len and table.len <= 65536 and slots.len >= 2 * table.len and std.math.isPowerOfTwo(slots.len));
    @memset(slots, 0); // 0 empty, else the table position + 1
    const mask = slots.len - 1;
    var count: usize = 0;
    for (words, index) |word, *at| {
        var probe = (@as(usize, word *% 0x9E3779B1) >> 7) & mask;
        while (true) : (probe = (probe + 1) & mask) {
            const slot = slots[probe];
            if (slot == 0) {
                if (count == table.len) return null;
                table[count] = word;
                count += 1;
                slots[probe] = @intCast(count);
                at.* = @intCast(count - 1);
                break;
            }
            if (table[slot - 1] == word) {
                at.* = @intCast(slot - 1);
                break;
            }
        }
    }
    return count;
}

test "every word comes back through its index, distinct words in first-seen order" {
    const a = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(11);
    const r = rng.random();
    for ([_]usize{ 1, 7, 3000, 6000, table_limit }) |distinct| {
        const pool = try a.alloc(u32, distinct);
        defer a.free(pool);
        for (pool, 0..) |*w, i| w.* = @as(u32, @intCast(i)) *% 2654435761 +% r.int(u32) % 4; // near and colliding hashes
        const words = try a.alloc(u32, 200_000);
        defer a.free(words);
        for (words, 0..) |*w, i| w.* = if (i < distinct) pool[i] else pool[r.uintLessThan(usize, distinct)];
        const index = try a.alloc(u16, words.len);
        defer a.free(index);
        const table = try a.alloc(u32, table_limit);
        defer a.free(table);
        const slots = try a.alloc(u32, 2 * table_limit);
        defer a.free(slots);
        const count = indexPairs(words, index, table, slots) orelse return error.TableFull;
        var seen = std.AutoHashMap(u32, void).init(a);
        defer seen.deinit();
        for (pool) |w| try seen.put(w, {});
        try std.testing.expectEqual(@as(usize, seen.count()), count);
        for (words, index) |w, at| try std.testing.expectEqual(w, table[at]);
    }
}

test "more distinct words than the table holds keeps the pairs" {
    const a = std.testing.allocator;
    const words = try a.alloc(u32, table_limit + 1);
    defer a.free(words);
    for (words, 0..) |*w, i| w.* = @intCast(i * 3 + 1);
    const index = try a.alloc(u16, words.len);
    defer a.free(index);
    var table: [table_limit]u32 = undefined;
    const slots = try a.alloc(u32, 2 * table_limit);
    defer a.free(slots);
    try std.testing.expect(indexPairs(words, index, &table, slots) == null);
}
