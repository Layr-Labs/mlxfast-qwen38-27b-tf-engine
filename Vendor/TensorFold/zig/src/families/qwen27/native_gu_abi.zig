//! Pure admission for the exact SG1 M16 HALF pipeline's runtime QUERY/RUN branch.
const std = @import("std");
pub const records = 64;
pub const record_words = 388;
pub const payload_words = 4 + records * record_words;
pub const guard_words = 8;
pub const storage_words = payload_words + 2 * guard_words;
pub const storage_bytes = storage_words * 4;
pub const offset_bytes = guard_words * 4;
const poison: u32 = 0xDEADDEAD;
const guard: u32 = 0xD3D3D3D3;

pub fn initialize(storage: []u32) !void {
    if (storage.len != storage_words) return error.NativeGuAbi;
    @memset(storage, poison);
    @memset(storage[0..guard_words], guard);
    @memset(storage[storage.len - guard_words ..], guard);
    storage[guard_words] = 1;
}
pub fn validate(storage: []const u32) !void {
    if (storage.len != storage_words) return error.NativeGuAbi;
    for (storage[0..guard_words]) |v| if (v != guard) return error.NativeGuGuard;
    for (storage[storage.len - guard_words ..]) |v| if (v != guard) return error.NativeGuGuard;
    const words = storage[guard_words..][0..payload_words];
    if (words[0] != 1 or words[1] != 1 or words[2] != records or words[3] != record_words) return error.NativeGuAbi;
    var seen: [2 * 16 * 32]bool = @splat(false);
    for (0..records) |tid| {
        const base = 4 + tid * record_words;
        if (words[base] != 16 or words[base + 1] != tid / 32 or words[base + 2] != tid % 32 or words[base + 3] != 1) return error.NativeGuAbi;
        for (0..128) |slot| {
            const at = base + 4 + slot * 3;
            if (slot >= 16) {
                if (words[at] != 0 or words[at + 1] != std.math.maxInt(u32) or words[at + 2] != std.math.maxInt(u32)) return error.NativeGuAbi;
                continue;
            }
            const col = words[at + 1];
            const row = words[at + 2];
            if (words[at] != 1 or col >= 32 or row >= 16) return error.NativeGuAbi;
            const index = (tid / 32) * 512 + row * 32 + col;
            if (seen[index]) return error.NativeGuAbi;
            seen[index] = true;
        }
    }
    for (seen) |v| if (!v) return error.NativeGuAbi;
}
/// The same optional-buffer residency transfer used by the runtime owner.
pub fn follow(buffer: anytype, previous: anytype, next: @TypeOf(previous)) void {
    if (previous) |f| f.leave(f.ctx, buffer);
    if (next) |f| f.join(f.ctx, buffer);
}
pub fn activate(storage: []u32) !void {
    try validate(storage);
    storage[guard_words] = 0;
}

fn fixture(storage: []u32) !void {
    try initialize(storage);
    const w = storage[guard_words..][0..payload_words];
    w[1] = 1;
    w[2] = records;
    w[3] = record_words;
    for (0..records) |tid| {
        const b = 4 + tid * record_words;
        w[b] = 16;
        w[b + 1] = @intCast(tid / 32);
        w[b + 2] = @intCast(tid % 32);
        w[b + 3] = 1;
        for (0..128) |i| {
            const at = b + 4 + 3 * i;
            w[at] = @intFromBool(i < 16);
            w[at + 1] = if (i < 16) @intCast(((tid % 32) * 16 + i) % 32) else std.math.maxInt(u32);
            w[at + 2] = if (i < 16) @intCast(((tid % 32) * 16 + i) / 32) else std.math.maxInt(u32);
        }
    }
}
test "HALF GU exact production ABI validator rejects incomplete maps and activates last" {
    const a = std.testing.allocator;
    const good = try a.alloc(u32, storage_words);
    defer a.free(good);
    const bad = try a.alloc(u32, storage_words);
    defer a.free(bad);
    try fixture(good);
    try validate(good);
    for ([_]struct { at: usize, value: u32 }{
        .{ .at = guard_words + 1, .value = 2 },
        .{ .at = guard_words + 4, .value = 15 },
        .{ .at = guard_words + 4, .value = 17 },
        .{ .at = guard_words + 5, .value = 1 },
        .{ .at = guard_words + 6, .value = 1 },
        .{ .at = guard_words + 8, .value = 0 },
        .{ .at = guard_words + 9, .value = 32 },
        .{ .at = guard_words + 10, .value = 16 },
        .{ .at = guard_words + 8 + 16 * 3, .value = 1 },
    }) |fault| {
        @memcpy(bad, good);
        bad[fault.at] = fault.value;
        const before = try a.dupe(u32, bad);
        defer a.free(before);
        try std.testing.expectError(error.NativeGuAbi, activate(bad));
        try std.testing.expectEqualSlices(u32, before, bad);
    }
    @memcpy(bad, good);
    bad[guard_words + 8 + 3 + 1] = bad[guard_words + 9];
    bad[guard_words + 8 + 3 + 2] = bad[guard_words + 10];
    try std.testing.expectError(error.NativeGuAbi, validate(bad));
    @memcpy(bad, good);
    bad[0] = 0;
    const damaged_prefix = try a.dupe(u32, bad);
    defer a.free(damaged_prefix);
    try std.testing.expectError(error.NativeGuGuard, activate(bad));
    try std.testing.expectEqualSlices(u32, damaged_prefix, bad);
    @memcpy(bad, good);
    bad[bad.len - 1] = 0;
    const damaged_suffix = try a.dupe(u32, bad);
    defer a.free(damaged_suffix);
    try std.testing.expectError(error.NativeGuGuard, activate(bad));
    try std.testing.expectEqualSlices(u32, damaged_suffix, bad);
    try std.testing.expectError(error.NativeGuAbi, validate(good[1..]));
    try activate(good);
    try std.testing.expectEqual(@as(u32, 0), good[guard_words]);
    try std.testing.expectError(error.NativeGuAbi, validate(good));
}

test "HALF GU ABI residency transfers leave before join and clear exactly once" {
    const Mock = struct {
        events: [4]u32 = @splat(0),
        len: usize = 0,
        const Self = @This();
        const F = struct { ctx: *Self, join: *const fn (*Self, u32) void, leave: *const fn (*Self, u32) void };
        fn join(self: *@This(), b: u32) void {
            self.events[self.len] = 100 + b;
            self.len += 1;
        }
        fn leave(self: *@This(), b: u32) void {
            self.events[self.len] = 200 + b;
            self.len += 1;
        }
    };
    var m: Mock = .{};
    const f: ?Mock.F = .{ .ctx = &m, .join = Mock.join, .leave = Mock.leave };
    follow(@as(u32, 7), @as(?Mock.F, null), f);
    follow(@as(u32, 7), f, f);
    follow(@as(u32, 7), f, null);
    try std.testing.expectEqualSlices(u32, &.{ 107, 207, 107, 207 }, &m.events);
    try std.testing.expectEqual(@as(usize, 4), m.len);
    follow(@as(u32, 7), @as(?Mock.F, null), null);
    try std.testing.expectEqual(@as(usize, 4), m.len);
}
