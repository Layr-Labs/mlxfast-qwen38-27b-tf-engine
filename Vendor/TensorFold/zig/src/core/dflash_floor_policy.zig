//! Bench-only sibling floor override and opt-in first-call engagement diagnostics.
const std = @import("std");
pub const default_floor: f64 = 0.015;

pub fn parse(text: ?[]const u8) !f64 {
    const value = if (text) |s| std.fmt.parseFloat(f64, s) catch return error.BadDflashFloor else default_floor;
    if (!std.math.isFinite(value) or value < 0 or value >= 1) return error.BadDflashFloor;
    return value;
}

pub const Config = struct { floor: f64, report: bool = false };
pub const Counts = struct { seed_calls: u64 = 0, round_calls: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, committed: u64 = 0, mismatched_floor_calls: u64 = 0 };
pub const State = struct {
    config: ?Config = null,
    counts: Counts = .{},

    pub fn get(s: *State) !f64 {
        if (s.config) |c| return c.floor;
        const floor = try parse(if (std.c.getenv("MLX_DFLASH_FLOOR")) |v| std.mem.span(v) else null);
        const report = if (std.c.getenv("MLX_DFLASH_FLOOR_REPORT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        s.config = .{ .floor = floor, .report = report };
        return floor;
    }

    /// Successful Generation.runRound calls only; no counting when the report is disabled.
    pub fn record(s: *State, floor: f64, seed: bool, nodes: usize, verified: usize, matched: usize, committed: usize) void {
        const c = s.config orelse return;
        if (!c.report) return;
        if (@as(u64, @bitCast(floor)) != @as(u64, @bitCast(c.floor))) s.counts.mismatched_floor_calls +|= 1;
        if (seed) {
            s.counts.seed_calls +|= 1;
        } else {
            s.counts.round_calls +|= 1;
            s.counts.drafted +|= verified -| 1;
            s.counts.accepted +|= matched;
            s.counts.committed +|= committed;
        }
        const calls = if (seed) s.counts.seed_calls else s.counts.round_calls;
        if (calls != 1) return;
        // Diagnostics only: timings use REPORT=0. No repeated per-round I/O.
        std.log.info("DFLASH_FLOOR event={s} floor_bits={d} nodes={d} verified={d} matched={d} committed={d} calls={d} mismatched_floor_calls={d}", .{
            if (seed) "seed" else "round", @as(u64, @bitCast(floor)), nodes, verified,
            matched,                       committed,                 calls, s.counts.mismatched_floor_calls,
        });
    }
};

test "DFlash floor defaults overrides and invalid input" {
    try std.testing.expectEqual(@as(f64, 0.015), try parse(null));
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, 0.007))), @as(u64, @bitCast(try parse("0.007"))));
    try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, 0.015))), @as(u64, @bitCast(try parse("0.015"))));
    try std.testing.expectEqual(@as(f64, 0), try parse("0"));
    for ([_][]const u8{ "", "invalid", "nan", "inf", "-inf", "-0.001", "1", "1.1", "1e999" }) |value|
        try std.testing.expectError(error.BadDflashFloor, parse(value));
}

test "DFlash floor real environment caches both knobs" {
    const C = struct {
        extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
        extern "c" fn unsetenv([*:0]const u8) c_int;
    };
    const a = std.testing.allocator;
    const keys = [_][:0]const u8{ "MLX_DFLASH_FLOOR", "MLX_DFLASH_FLOOR_REPORT" };
    var saved: [2]?[:0]u8 = @splat(null);
    for (keys, &saved) |key, *old| old.* = if (std.c.getenv(key)) |v| try a.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer for (keys, saved) |key, old| {
        if (old) |v| {
            _ = C.setenv(key, v, 1);
            a.free(v);
        } else _ = C.unsetenv(key);
    };
    for (keys) |key| try std.testing.expectEqual(@as(c_int, 0), C.unsetenv(key));
    var defaults = State{};
    try std.testing.expectEqual(@as(f64, 0.015), try defaults.get());
    try std.testing.expect(!defaults.config.?.report);
    try std.testing.expectEqual(@as(c_int, 0), C.setenv(keys[0], "0.007", 1));
    try std.testing.expectEqual(@as(c_int, 0), C.setenv(keys[1], "1", 1));
    var enabled = State{};
    try std.testing.expectEqual(@as(f64, 0.007), try enabled.get());
    try std.testing.expect(enabled.config.?.report);
    try std.testing.expectEqual(@as(f64, 0.015), try defaults.get());
    try std.testing.expectEqual(@as(c_int, 0), C.setenv(keys[0], "nan", 1));
    try std.testing.expectEqual(@as(f64, 0.007), try enabled.get());
    var invalid = State{};
    try std.testing.expectError(error.BadDflashFloor, invalid.get());
    try std.testing.expect(invalid.config == null);
}

test "DFlash floor counters are opt in and record successful actual arguments" {
    var off = State{ .config = .{ .floor = 0.015 } };
    off.record(0.015, true, 7, 0, 0, 1);
    off.record(0.015, false, 7, 8, 1, 2);
    try std.testing.expectEqualDeep(Counts{}, off.counts);
    var on = State{ .config = .{ .floor = 0.015, .report = true } };
    on.record(0.015, true, 7, 0, 0, 1);
    on.record(0.015, false, 7, 8, 1, 2);
    on.record(0.007, false, 7, 4, 0, 1);
    try std.testing.expectEqualDeep(Counts{ .seed_calls = 1, .round_calls = 2, .drafted = 10, .accepted = 1, .committed = 3, .mismatched_floor_calls = 1 }, on.counts);
}
