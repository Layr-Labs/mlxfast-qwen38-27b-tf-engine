//! Startup-only, prompt-free selection of an unchanged prompt GU computation.
const std = @import("std");
pub const modes = [_]u8{ 0, 2, 4, 22, 8 };
pub const pair_count = 8;
pub const null_pairs = 4;
pub const Pair = struct { baseline: f64, candidate: f64 };

pub fn positive(p: Pair) bool {
    return std.math.isFinite(p.baseline) and std.math.isFinite(p.candidate) and p.baseline > 0 and p.candidate > 0;
}
pub fn meanGain(values: []const Pair) ?f64 {
    if (values.len != pair_count) return null;
    var baseline: f64 = 0;
    var candidate: f64 = 0;
    for (values) |p| {
        if (!positive(p) or p.candidate >= p.baseline) return null;
        baseline += p.baseline;
        candidate += p.candidate;
    }
    return 1 - candidate / baseline;
}
pub fn qualifies(values: []const Pair, controls: []const Pair) bool {
    const gain = meanGain(values) orelse return false;
    if (controls.len != 2 * null_pairs) return false;
    var envelope: f64 = 0;
    for (controls) |p| {
        if (!positive(p)) return false;
        envelope = @max(envelope, @abs(1 - p.candidate / p.baseline));
    }
    return gain >= 0.08 and gain > 3 * envelope;
}

test "prompt GU walk policy refuses noise missing evidence and nonfinite data" {
    const good: [pair_count]Pair = @splat(.{ .baseline = 1, .candidate = 0.9 });
    const controls: [2 * null_pairs]Pair = @splat(.{ .baseline = 1, .candidate = 1 });
    try std.testing.expect(qualifies(&good, &controls));
    try std.testing.expect(!qualifies(good[0 .. pair_count - 1], &controls));
    try std.testing.expect(!qualifies(&good, controls[0..null_pairs]));
    var bad = good;
    bad[3].candidate = 1.01;
    try std.testing.expect(!qualifies(&bad, &controls));
    bad = good;
    bad[0].baseline = std.math.nan(f64);
    try std.testing.expect(!qualifies(&bad, &controls));
    bad = @splat(.{ .baseline = 1, .candidate = 0.93 });
    try std.testing.expect(!qualifies(&bad, &controls));
    var noisy = controls;
    noisy[0].candidate = 1.04;
    try std.testing.expect(!qualifies(&good, &noisy));
}

test "prompt GU walk policy independent holdout must qualify again" {
    const selection: [pair_count]Pair = @splat(.{ .baseline = 1, .candidate = 0.9 });
    const holdout: [pair_count]Pair = @splat(.{ .baseline = 1, .candidate = 0.95 });
    const controls: [2 * null_pairs]Pair = @splat(.{ .baseline = 1, .candidate = 1 });
    try std.testing.expect(qualifies(&selection, &controls));
    try std.testing.expect(!qualifies(&holdout, &controls));
}

/// Final admission requires independent holdout evidence after selection.
pub fn confirmed(mode: u8, selection: []const Pair, controls: []const Pair, holdout: []const Pair, holdout_controls: []const Pair) u8 {
    if (mode != 2 and mode != 4 and mode != 22 and mode != 8) return 0;
    if (!qualifies(selection, controls) or !qualifies(holdout, holdout_controls)) return 0;
    return mode;
}

test "prompt GU walk final admission includes independent holdout and valid mode" {
    const good: [pair_count]Pair = @splat(.{ .baseline = 1, .candidate = 0.9 });
    const weak: [pair_count]Pair = @splat(.{ .baseline = 1, .candidate = 0.95 });
    const controls: [2 * null_pairs]Pair = @splat(.{ .baseline = 1, .candidate = 1 });
    for (modes[1..]) |mode| {
        try std.testing.expectEqual(mode, confirmed(mode, &good, &controls, &good, &controls));
        try std.testing.expectEqual(@as(u8, 0), confirmed(mode, &good, &controls, &weak, &controls));
        try std.testing.expectEqual(@as(u8, 0), confirmed(mode, &weak, &controls, &good, &controls));
    }
    try std.testing.expectEqual(@as(u8, 0), confirmed(99, &good, &controls, &good, &controls));
}

/// Production and host fault checks share the exact selection/holdout call sequence.
pub fn select(probe: anytype, expected: anytype) !u8 {
    var winner: usize = 0;
    var best_gain: f64 = 0;
    var winning_pairs: [pair_count]Pair = undefined;
    var winning_controls: [2 * null_pairs]Pair = undefined;
    for (1..modes.len) |arm| {
        const controls = try probe.nulls(arm, expected);
        const samples = try probe.pairs(arm, expected);
        if (qualifies(&samples, &controls)) {
            const gain = meanGain(&samples).?;
            if (gain > best_gain) {
                winner = arm;
                best_gain = gain;
                winning_pairs = samples;
                winning_controls = controls;
            }
        }
    }
    if (winner == 0) return 0;
    // Fresh measurements after selection, including fresh nulls, cannot reuse the winning samples.
    const controls = try probe.nulls(winner, expected);
    const holdout = try probe.pairs(winner, expected);
    return confirmed(modes[winner], &winning_pairs, &winning_controls, &holdout, &controls);
}

const Script = struct {
    events: [2 * modes.len]u8 = undefined,
    len: usize = 0,
    weak_holdout: bool = false,
    noisy_holdout: bool = false,
    identity_error: bool = false,
    fn note(s: *Script, arm: usize, code: u8) void {
        s.events[s.len] = @as(u8, @intCast(10 * arm)) + code;
        s.len += 1;
    }
    pub fn nulls(s: *Script, arm: usize, _: void) ![2 * null_pairs]Pair {
        s.note(arm, 1);
        var result: [2 * null_pairs]Pair = @splat(.{ .baseline = 1, .candidate = 1 });
        if (s.noisy_holdout and s.len == 2 * modes.len - 1) result[0].candidate = 1.1;
        return result;
    }
    pub fn pairs(s: *Script, arm: usize, _: void) ![pair_count]Pair {
        s.note(arm, 2);
        if (s.identity_error and s.len == 2 * modes.len) return error.PlantedIdentity;
        const time = if (s.weak_holdout and s.len == 2 * modes.len) @as(f64, 0.99) else 0.9 - 0.01 * @as(f64, @floatFromInt(arm));
        return @splat(.{ .baseline = 1, .candidate = time });
    }
};

test "prompt GU walk production orchestration consumes fresh holdout and nulls" {
    var good: Script = .{};
    try std.testing.expectEqual(@as(u8, 8), try select(&good, @as(void, {})));
    try std.testing.expectEqualSlices(u8, &.{ 11, 12, 21, 22, 31, 32, 41, 42, 41, 42 }, good.events[0..good.len]);
    var weak: Script = .{ .weak_holdout = true };
    try std.testing.expectEqual(@as(u8, 0), try select(&weak, @as(void, {})));
    try std.testing.expectEqual(@as(usize, 2 * modes.len), weak.len);
    var noisy: Script = .{ .noisy_holdout = true };
    try std.testing.expectEqual(@as(u8, 0), try select(&noisy, @as(void, {})));
    var broken: Script = .{ .identity_error = true };
    try std.testing.expectError(error.PlantedIdentity, select(&broken, @as(void, {})));
    try std.testing.expectEqual(@as(usize, 2 * modes.len), broken.len);
}

pub fn requireIdentity(expected: [32]u8, actual: [32]u8) !void {
    if (!std.mem.eql(u8, &expected, &actual)) return error.PromptWalkIdentity;
}

test "prompt GU walk actual hash guard detects a changed output bit" {
    const expected: [32]u8 = @splat(0xA7);
    try requireIdentity(expected, expected);
    var changed = expected;
    changed[17] ^= 1;
    try std.testing.expectError(error.PromptWalkIdentity, requireIdentity(expected, changed));
}
