//! Plan large contiguous writes to a circular KV cache; all other inputs use the scatter path.
const std = @import("std");

pub const Plan = struct { slot: u32, first: u32, second: u32 };

pub fn contiguous(positions: []const u64, window: u32) ?Plan {
    if (window == 0 or positions.len <= 16 or positions.len > window) return null;
    for (positions[1..], 1..) |position, i| {
        const prev = positions[i - 1];
        if (prev == std.math.maxInt(u64) or position != prev + 1) return null;
    }
    const slot: u32 = @intCast(positions[0] % window);
    const first: u32 = @min(@as(u32, @intCast(positions.len)), window - slot);
    return .{ .slot = slot, .first = first, .second = @as(u32, @intCast(positions.len)) - first };
}

test "contiguous ring writes split only at the wrap and reject unsafe streams" {
    var positions: [20]u64 = undefined;
    for (&positions, 0..) |*p, i| p.* = 60 + i;
    try std.testing.expectEqual(Plan{ .slot = 60, .first = 4, .second = 16 }, contiguous(&positions, 64).?);
    for (&positions, 0..) |*p, i| p.* = 10 + i;
    try std.testing.expectEqual(Plan{ .slot = 10, .first = 20, .second = 0 }, contiguous(&positions, 64).?);
    positions[9] += 1;
    try std.testing.expect(contiguous(&positions, 64) == null);
    try std.testing.expect(contiguous(positions[0..16], 64) == null);
    try std.testing.expect(contiguous(&positions, 8) == null);
    try std.testing.expect(contiguous(&positions, 0) == null);
}

test "planned runs match sequential scatter slots for all wraps" {
    var positions: [64]u64 = undefined;
    for (17..65) |len| for (0..128) |start| {
        for (positions[0..len], 0..) |*p, i| p.* = start + i;
        const plan = contiguous(positions[0..len], 64).?;
        for (0..len) |i| {
            const planned = if (i < plan.first) plan.slot + i else i - plan.first;
            try std.testing.expectEqual(positions[i] % 64, planned);
        }
    };
    positions[0] = std.math.maxInt(u64);
    positions[1] = 0;
    try std.testing.expect(contiguous(positions[0..17], 64) == null);
}

test "two planned runs preserve strided key and value rows" {
    const width = 4;
    const stride = 11;
    var positions: [32]u64 = undefined;
    var input: [32 * stride]u16 = undefined;
    var expected: [64 * width]u16 = undefined;
    var actual: [64 * width]u16 = undefined;
    for (0..128) |start| {
        for (&positions, 0..) |*p, i| p.* = start + i;
        for (&input, 0..) |*v, i| v.* = @intCast(i);
        @memset(&expected, 0xffff);
        @memset(&actual, 0xffff);
        for (positions, 0..) |position, row| {
            const slot: usize = @intCast(position % 64);
            @memcpy(expected[slot * width ..][0..width], input[row * stride ..][0..width]);
        }
        const plan = contiguous(&positions, 64).?;
        for (0..plan.first) |row| @memcpy(actual[(@as(usize, plan.slot) + row) * width ..][0..width], input[row * stride ..][0..width]);
        for (0..plan.second) |row| @memcpy(actual[row * width ..][0..width], input[(@as(usize, plan.first) + row) * stride ..][0..width]);
        try std.testing.expectEqualSlices(u16, &expected, &actual);
    }
}
