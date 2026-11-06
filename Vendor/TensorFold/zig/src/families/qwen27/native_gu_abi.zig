//! Pure admission for the exact SG1 M16 packed U4 pipeline's runtime QUERY/RUN branch.
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
pub fn inspect(storage: []const u32) !u32 {
    if (storage.len != storage_words) return error.NativeGuAbi;
    for (storage[0..guard_words]) |v| if (v != guard) return error.NativeGuGuard;
    for (storage[storage.len - guard_words ..]) |v| if (v != guard) return error.NativeGuGuard;
    const words = storage[guard_words..][0..payload_words];
    if (words[0] != 1 or words[1] != 1 or words[2] != records or words[3] != record_words) return error.NativeGuAbi;
    var seen: [2 * 16 * 32]bool = @splat(false);
    var maximum: u32 = 0;
    for (0..records) |tid| {
        const base = 4 + tid * record_words;
        const capacity = words[base];
        if (capacity == 0 or capacity > 128) return error.NativeGuAbi;
        maximum = @max(maximum, capacity);
        if (words[base + 1] != tid / 32 or words[base + 2] != tid % 32 or words[base + 3] != 1) return error.NativeGuAbi;
        for (0..128) |slot| {
            const at = base + 4 + slot * 3;
            if (words[at] == 0) {
                if (words[at] != 0 or words[at + 1] != std.math.maxInt(u32) or words[at + 2] != std.math.maxInt(u32)) return error.NativeGuAbi;
                continue;
            }
            const col = words[at + 1];
            const row = words[at + 2];
            if (slot >= capacity or words[at] != 1 or col >= 32 or row >= 16) return error.NativeGuAbi;
            const index = (tid / 32) * 512 + row * 32 + col;
            if (seen[index]) return error.NativeGuAbi;
            seen[index] = true;
        }
    }
    for (seen) |v| if (!v) return error.NativeGuAbi;
    return maximum;
}
pub fn validate(storage: []const u32) !void {
    _ = try inspect(storage);
}
/// The same optional-buffer residency transfer used by the runtime owner.
pub fn follow(buffer: anytype, previous: anytype, next: @TypeOf(previous)) void {
    if (previous) |f| f.leave(f.ctx, buffer);
    if (next) |f| f.join(f.ctx, buffer);
}
pub fn activateFor(storage: []u32, compiled_capacity: u32) !void {
    if (compiled_capacity != 16 and compiled_capacity != 32 and compiled_capacity != 64 and compiled_capacity != 128) return error.NativeGuAbi;
    if (try inspect(storage) > compiled_capacity) return error.NativeGuCapacity;
    storage[guard_words] = 0;
}

/// Legacy callers remain conservative; no generic map can enable a CAP16 PSO unchecked.
pub fn activate(storage: []u32) !void {
    try activateFor(storage, 16);
}
/// Production compile/query selection. Each attempt owns a fresh PSO and QUERY buffer.
/// The probe releases partial resources on any failure; adoption transfers only the fit PSO.
pub fn admit(probe: anytype) !void {
    defer probe.discard();
    var compiled: u32 = 16;
    for (0..4) |_| {
        try probe.prepare(compiled);
        const actual = try inspect(probe.storage());
        if (actual <= compiled) {
            try activateFor(probe.storage(), compiled);
            probe.adopt();
            return;
        }
        probe.discard();
        compiled = try nextCapacity(compiled, actual);
    }
    return error.NativeGuCapacity;
}
pub fn nextCapacity(current: u32, actual: u32) !u32 {
    if (actual <= current or actual > 128) return error.NativeGuCapacity;
    for ([_]u32{ 16, 32, 64, 128 }) |candidate| {
        if (candidate > current and candidate >= actual) return candidate;
    }
    return error.NativeGuCapacity;
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
test "packed U4 GU exact production ABI validator rejects incomplete maps and activates last" {
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
        .{ .at = guard_words + 4, .value = 129 },
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

test "packed U4 GU ABI residency transfers leave before join and clear exactly once" {
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

fn capacityFixture(storage: []u32, capacity: u32) !void {
    try fixture(storage);
    const w = storage[guard_words..][0..payload_words];
    for (0..records) |tid| {
        const b = 4 + tid * record_words;
        w[b] = capacity;
        for (0..128) |slot| {
            const at = b + 4 + 3 * slot;
            w[at] = 0;
            w[at + 1] = std.math.maxInt(u32);
            w[at + 2] = std.math.maxInt(u32);
        }
        for (0..16) |logical| {
            const slot = logical * (capacity / 16);
            const at = b + 4 + 3 * slot;
            w[at] = 1;
            w[at + 1] = @intCast(((tid % 32) * 16 + logical) % 32);
            w[at + 2] = @intCast(((tid % 32) * 16 + logical) / 32);
        }
    }
}

test "capacity ABI accepts holes and bounds compiled storage before mode zero" {
    const a = std.testing.allocator;
    const w = try a.alloc(u32, storage_words);
    defer a.free(w);
    for ([_]u32{ 16, 32, 64, 128 }) |capacity| {
        try capacityFixture(w, capacity);
        try std.testing.expectEqual(capacity, try inspect(w));
        if (capacity > 16) {
            try std.testing.expectError(error.NativeGuCapacity, activateFor(w, capacity / 2));
            try std.testing.expectEqual(@as(u32, 1), w[guard_words]);
        }
        try activateFor(w, capacity);
        try std.testing.expectEqual(@as(u32, 0), w[guard_words]);
    }
    try capacityFixture(w, 128);
    // Invalid interior slots require sentinel coordinates, not just valid padding.
    w[guard_words + 4 + 4 + 3 + 1] = 0;
    try std.testing.expectError(error.NativeGuAbi, inspect(w));
    try capacityFixture(w, 32);
    // Valid coordinates beyond reported storage are forbidden, including a hole map.
    w[guard_words + 4] = 16;
    try std.testing.expectError(error.NativeGuAbi, inspect(w));
    try capacityFixture(w, 64);
    // Duplicate a coordinate in SG1 while SG0 remains complete.
    const base = guard_words + 4 + 32 * record_words + 4;
    w[base + 4 * 3 + 1] = w[base + 1];
    w[base + 4 * 3 + 2] = w[base + 2];
    try std.testing.expectError(error.NativeGuAbi, inspect(w));
}

test "production capacity policy freshly queries upgrades and releases every partial attempt" {
    const Mock = struct {
        words: []u32,
        capacities: []const u32,
        fail_attempt: ?usize = null,
        fail_stage: u32 = 0,
        corrupt: bool = false,
        attempts: usize = 0,
        compiled: [4]u32 = @splat(0),
        live: u32 = 0,
        released: u32 = 0,
        adopted: bool = false,
        adopted_mode: u32 = 99,
        selected_capacity: u32 = 0,
        pub fn prepare(self: *@This(), cap: u32) !void {
            const i = self.attempts;
            self.compiled[i] = cap;
            self.attempts += 1;
            self.live = 1; // PSO allocated.
            if (self.fail_attempt == i and self.fail_stage == 1) return error.MockCompile;
            self.live = 2; // ABI allocated.
            if (self.fail_attempt == i and self.fail_stage == 2) return error.MockAllocation;
            try capacityFixture(self.words, self.capacities[i]);
            if (self.corrupt) self.words[guard_words + 4 + 4] = 0;
            if (self.fail_attempt == i and self.fail_stage == 3) return error.MockQuery;
        }
        pub fn storage(self: *@This()) []u32 {
            return self.words;
        }
        pub fn discard(self: *@This()) void {
            self.released += self.live;
            self.live = 0;
        }
        pub fn adopt(self: *@This()) void {
            self.adopted = true;
            self.adopted_mode = self.words[guard_words];
            self.selected_capacity = self.compiled[self.attempts - 1];
            self.live = 0; // Exact last PSO/ABI transferred to their owner.
        }
    };
    const a = std.testing.allocator;
    const words = try a.alloc(u32, storage_words);
    defer a.free(words);
    for ([_][]const u32{ &.{16}, &.{ 32, 32 }, &.{ 64, 64 }, &.{ 128, 128 }, &.{ 32, 64, 128, 128 } }) |sequence| {
        var m: Mock = .{ .words = words, .capacities = sequence };
        try admit(&m);
        try std.testing.expect(m.adopted);
        try std.testing.expectEqual(@as(u32, 0), m.adopted_mode);
        try std.testing.expectEqual(sequence.len, m.attempts);
        try std.testing.expectEqual(sequence[sequence.len - 1], m.selected_capacity);
        try std.testing.expectEqual(@as(u32, @intCast(2 * (sequence.len - 1))), m.released);
        try std.testing.expectEqual(@as(u32, 0), m.live);
        for (m.compiled[1..m.attempts], m.compiled[0 .. m.attempts - 1]) |now, previous| try std.testing.expect(now > previous);
    }
    for (0..4) |failure| for (1..4) |stage| {
        var m: Mock = .{ .words = words, .capacities = &.{ 32, 64, 128, 128 }, .fail_attempt = failure, .fail_stage = @intCast(stage) };
        const result = admit(&m);
        try std.testing.expectError(if (stage == 1) error.MockCompile else if (stage == 2) error.MockAllocation else error.MockQuery, result);
        try std.testing.expect(!m.adopted);
        try std.testing.expectEqual(@as(u32, 0), m.live);
        try std.testing.expectEqual(@as(u32, @intCast(2 * failure + @min(stage, 2))), m.released);
    };
    var bad: Mock = .{ .words = words, .capacities = &.{32}, .corrupt = true };
    try std.testing.expectError(error.NativeGuAbi, admit(&bad));
    try std.testing.expect(!bad.adopted);
    try std.testing.expectEqual(@as(u32, 2), bad.released);
    try std.testing.expectError(error.NativeGuCapacity, nextCapacity(128, 129));
    try std.testing.expectError(error.NativeGuCapacity, nextCapacity(32, 32));
}
