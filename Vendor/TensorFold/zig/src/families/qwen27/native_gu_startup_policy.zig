//! Prompt-free device admission. Adapted from team ticket d8f76fd4 / R8 08949fe.
const std = @import("std");
pub const pair_count = 8;
pub const null_count = 4;
pub const identity_rows = [_]u32{ 128, 1, 17, 37 };
pub const identity_cohorts = identity_rows.len * 2 * 2;
pub const warm_cohorts = 4;
pub const max_cohorts = identity_cohorts + warm_cohorts + 2 * (4 * null_count + 4 * pair_count);
pub fn candidateTurn(index: usize, turn: usize, holdout: bool) bool {
    return ((turn == 1 or turn == 2) != ((index / 2 + @intFromBool(holdout)) % 2 != 0));
}
pub const rows = 128;
pub const k = 5120;
pub const Hash = [32]u8;
pub const Pair = struct { baseline: f64, candidate: f64 };
pub const Mode = enum { off, adaptive, required };
pub fn cohortLimit(mode: Mode) usize {
    return switch (mode) {
        .off => 0,
        .adaptive => max_cohorts,
        .required => max_cohorts,
    };
}
pub const Status = enum(u8) { off, unsupported, shape, unavailable, identity, guard, sums, gpu, time, budget, selection, holdout, native };
pub const Decision = struct { status: Status, native: bool = false };
/// Required admission never silently falls back after a failed performance guard.
pub fn requireDecision(mode: Mode, decision: Decision) !void {
    if (mode != .required or decision.native) return;
    return switch (decision.status) {
        .selection => error.NativeGuPerformanceSelection,
        .holdout => error.NativeGuPerformanceHoldout,
        else => error.NativeGuRequired,
    };
}
pub fn budgetFits(allocated: u64, maximum: u64, scratch: u64) bool {
    return allocated <= maximum and scratch <= maximum - allocated;
}
fn positive(p: Pair) bool {
    return std.math.isFinite(p.baseline) and std.math.isFinite(p.candidate) and p.baseline > 0 and p.candidate > 0;
}
pub fn pairGood(p: Pair) bool {
    return positive(p) and p.candidate <= 0.92 * p.baseline;
}
fn envelope(controls: []const Pair) ?f64 {
    if (controls.len != null_count) return null;
    var value: f64 = 0;
    for (controls) |p| {
        if (!positive(p)) return null;
        value = @max(value, @abs(1 - p.candidate / p.baseline));
    }
    return value;
}
pub fn qualifies(pairs: []const Pair, controls: []const Pair) bool {
    if (pairs.len != pair_count) return false;
    const noise = envelope(controls) orelse return false;
    var a: f64 = 0;
    var b: f64 = 0;
    for (pairs) |p| {
        if (!pairGood(p)) return false;
        a += p.baseline;
        b += p.candidate;
    }
    return std.math.isFinite(a) and std.math.isFinite(b) and 1 - b / a > 3 * noise;
}
/// Required mode also consumes the unchanged performance selection and holdout.
/// Production and fake probes share this complete preflight/admission orchestration.
pub fn admit(probe: anytype, mode: Mode) !Decision {
    if (mode == .off) return .{ .status = .off };
    var expected: [2]Hash = undefined;
    for (0..2) |seed| {
        try probe.inputs(@intCast(seed));
        for (identity_rows) |active| {
            const hash = try probe.original(@intCast(seed), active);
            try probe.candidate(@intCast(seed), active, hash);
            if (active == rows) expected[seed] = hash;
        }
    }
    try probe.beginWarm();
    for ([_]bool{ false, true, true, false }) |native| try probe.warm(native, expected[0]);
    return select(probe, expected);
}

/// Both production and fault probes execute this exact fresh-null/fresh-holdout protocol.
pub fn select(probe: anytype, expected: [2]Hash) !Decision {
    for (0..2) |phase| {
        const holdout = phase != 0;
        const fail: Decision = .{ .status = if (holdout) .holdout else .selection };
        const controls = try probe.nulls(expected, holdout);
        _ = envelope(&controls) orelse return fail;
        var pairs: [pair_count]Pair = undefined;
        for (&pairs, 0..) |*p, i| {
            p.* = try probe.pair(i, expected, holdout);
            // Admission requires EVERY pair. One failed pair makes completion pointless.
            if (!pairGood(p.*)) return fail;
        }
        if (!qualifies(&pairs, &controls)) return fail;
    }
    return .{ .status = .native, .native = true };
}
pub fn requireIdentity(expected: Hash, actual: Hash) !void {
    if (!std.mem.eql(u8, &expected, &actual)) return error.NativeGuIdentity;
}
/// BF16 words cover ordinary exponent/mantissa patterns rather than a small integer lattice.
pub fn inputs(x: []u16, xs: []f32, seed: u32) !void {
    @setFloatMode(.strict);
    if (x.len != rows * k or xs.len != rows * (k / 64)) return error.NativeGuShape;
    var state: u32 = 0x9E3779B9 ^ (seed *% 0x85EBCA6B);
    for (x, 0..) |*v, i| {
        state = state *% 1664525 +% 1013904223;
        const exponent: u16 = @intCast(112 + ((state >> 8) % 21)); // unbiased -15..5
        v.* = @as(u16, @intCast(state >> 31)) << 15 | exponent << 7 | @as(u16, @intCast((state >> 17) & 127));
        if (i % 8192 == 0) v.* = if ((i / 8192) % 2 == 0) 0 else 0x8000;
    }
    for (0..k / 64) |g| for (0..rows) |r| {
        var sum: f32 = 0;
        for (0..64) |j| sum += @as(f32, @bitCast(@as(u32, x[r * k + g * 64 + j]) << 16));
        xs[g * rows + r] = sum;
    };
}

/// Production probe validation, also exercised with forced padding and HS faults without a GPU.
pub fn checkRows(y: []const u16, hs: []const f32, width: usize, stride: usize, active: usize, ordered: bool) !void {
    @setFloatMode(.strict);
    if (width == 0 or width % 64 != 0 or stride == 0 or active == 0 or active > stride or y.len != width * stride or hs.len != (width / 64) * stride) return error.NativeGuShape;
    for (0..stride) |r| for (y[r * width ..][0..width]) |value| {
        if (r < active) {
            if (value & 0x7F80 == 0x7F80) return error.NativeGuNonfinite;
        } else if (value != 0xFFFF) return error.NativeGuGuard;
    };
    for (0..width / 64) |g| for (0..stride) |r| {
        const value = hs[g * stride + r];
        if (r < active) {
            if (!std.math.isFinite(value)) return error.NativeGuNonfinite;
            if (ordered) {
                var sum: f32 = 0;
                for (0..64) |j| sum += @as(f32, @bitCast(@as(u32, y[r * width + g * 64 + j]) << 16));
                if (@as(u32, @bitCast(sum)) != @as(u32, @bitCast(value))) return error.NativeGuSums;
            }
        } else if (@as(u32, @bitCast(value)) != 0xFFFFFFFF) return error.NativeGuGuard;
    };
}

const Fake = struct {
    events: [64]u8 = undefined,
    len: usize = 0,
    weak_pair: ?usize = null,
    weak_holdout: bool = false,
    noisy_holdout: bool = false,
    identity_error: bool = false,
    swapped: bool = false,
    bad_null: bool = false,
    budget_error: bool = false,
    cohorts: usize = 0,
    numerical_error: bool = false,
    tail_error: bool = false,
    fn note(f: *Fake, value: u8) void {
        f.events[f.len] = value;
        f.len += 1;
    }
    pub fn inputs(f: *Fake, seed: u32) !void {
        f.note(@intCast(30 + seed));
    }
    fn fixtureHash(seed: u32, active: u32) Hash {
        var hash: Hash = if (seed == 0) @splat(0xa7) else @splat(0xb8);
        if (active != rows) hash[0] ^= @intCast(active);
        return hash;
    }
    fn shapeIndex(active: u32) !usize {
        const shapes = [_]u32{ 128, 1, 17, 37 };
        for (shapes, 0..) |value, index| if (value == active) return index;
        return error.PlantedShape;
    }
    pub fn original(f: *Fake, seed: u32, active: u32) !Hash {
        f.note(@intCast(50 + seed * 4 + try shapeIndex(active)));
        f.cohorts += 1;
        return fixtureHash(seed, active);
    }
    pub fn candidate(f: *Fake, seed: u32, active: u32, expected: Hash) !void {
        f.note(@intCast(60 + seed * 4 + try shapeIndex(active)));
        f.cohorts += 1;
        var actual = fixtureHash(seed, active);
        if (f.numerical_error) actual[7] ^= 1;
        if (f.tail_error and seed == 1 and active == 37) return error.NativeGuGuard;
        try requireIdentity(expected, actual);
    }
    pub fn beginWarm(f: *Fake) !void {
        f.note(40);
    }
    pub fn warm(f: *Fake, native: bool, expected: Hash) !void {
        f.note(if (native) 42 else 41);
        f.cohorts += 1;
        try requireIdentity(@splat(0xa7), expected);
    }
    pub fn nulls(f: *Fake, _: [2]Hash, holdout: bool) ![null_count]Pair {
        f.note(if (holdout) 20 else 10);
        f.cohorts += 4 * null_count;
        var v: [null_count]Pair = @splat(.{ .baseline = 1, .candidate = 1 });
        if (holdout and f.noisy_holdout) v[0].candidate = 1.04;
        if (f.bad_null) v[0].baseline = std.math.nan(f64);
        return v;
    }
    pub fn pair(f: *Fake, i: usize, expected: [2]Hash, holdout: bool) !Pair {
        f.note(@as(u8, if (holdout) 21 else 11) + @as(u8, @intCast(i)));
        f.cohorts += 4;
        if (holdout and f.budget_error) return error.NativeGuBudget;
        const known: [2]Hash = .{ @splat(0xa7), @splat(0xb8) };
        var actual = known[i % 2];
        if (holdout and f.identity_error) actual[7] ^= 1;
        try requireIdentity(expected[i % 2], actual);
        const p: Pair = .{ .baseline = 1, .candidate = if ((holdout and f.weak_holdout) or f.weak_pair == i) 0.95 else 0.9 };
        return if (f.swapped) .{ .baseline = p.candidate, .candidate = p.baseline } else p;
    }
};
test "native GU startup production protocol requires fresh holdout nulls and all eight pairs" {
    const expected: [2]Hash = .{ @splat(0xa7), @splat(0xb8) };
    var good: Fake = .{};
    const d = try select(&good, expected);
    try std.testing.expect(d.native and d.status == .native);
    try std.testing.expectEqual(@as(usize, max_cohorts), good.cohorts + identity_cohorts + warm_cohorts);
    try std.testing.expectEqualSlices(u8, &.{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28 }, good.events[0..good.len]);
    var seed_swap: Fake = .{};
    try std.testing.expectError(error.NativeGuIdentity, select(&seed_swap, .{ expected[1], expected[0] }));
    var weak: Fake = .{ .weak_holdout = true };
    try std.testing.expect(!(try select(&weak, expected)).native);
    try std.testing.expectEqual(@as(usize, 11), weak.len);
    var noisy: Fake = .{ .noisy_holdout = true };
    try std.testing.expect(!(try select(&noisy, expected)).native);
    var identity: Fake = .{ .identity_error = true };
    try std.testing.expectError(error.NativeGuIdentity, select(&identity, expected));
    var budget: Fake = .{ .budget_error = true };
    try std.testing.expectError(error.NativeGuBudget, select(&budget, expected));
    var one_bad: Fake = .{ .weak_pair = 0 };
    try std.testing.expect(!(try select(&one_bad, expected)).native);
    try std.testing.expectEqual(@as(usize, 2), one_bad.len);
    try std.testing.expectEqual(@as(usize, identity_cohorts + warm_cohorts + 20), one_bad.cohorts + identity_cohorts + warm_cohorts);
    var swapped: Fake = .{ .swapped = true };
    try std.testing.expect(!(try select(&swapped, expected)).native);
    var invalid_null: Fake = .{ .bad_null = true };
    try std.testing.expect(!(try select(&invalid_null, expected)).native);
    try std.testing.expectEqual(@as(usize, 1), invalid_null.len);
}
test "native GU startup every-pair rule rejects a weak pair despite a strong mean" {
    const controls: [null_count]Pair = @splat(.{ .baseline = 1, .candidate = 1 });
    var pairs: [pair_count]Pair = @splat(.{ .baseline = 1, .candidate = 0.75 });
    try std.testing.expect(qualifies(&pairs, &controls));
    pairs[3].candidate = 0.99;
    try std.testing.expect(!qualifies(&pairs, &controls));
    pairs[3].candidate = std.math.inf(f64);
    try std.testing.expect(!qualifies(&pairs, &controls));
    try std.testing.expect(!qualifies(pairs[0..7], &controls));
}
test "native GU startup actual hash guard rejects a changed output bit" {
    const hash: Hash = @splat(0x77);
    try requireIdentity(hash, hash);
    var other = hash;
    other[31] ^= 1;
    try std.testing.expectError(error.NativeGuIdentity, requireIdentity(hash, other));
}
test "native GU startup synthetic words exercise exponent mantissa zeros and exact ordered XS" {
    const a = std.testing.allocator;
    const x = try a.alloc(u16, rows * k);
    defer a.free(x);
    const xs = try a.alloc(f32, rows * (k / 64));
    defer a.free(xs);
    try inputs(x, xs, 0);
    const first = x[17];
    var exponents: [21]bool = @splat(false);
    var mantissas: [128]bool = @splat(false);
    var negative_zero = false;
    for (x) |v| {
        if (v == 0x8000) negative_zero = true;
        if (v & 0x7fff == 0) continue;
        const e = (v >> 7) & 255;
        try std.testing.expect(e >= 112 and e <= 132);
        exponents[e - 112] = true;
        mantissas[v & 127] = true;
    }
    try std.testing.expect(negative_zero);
    for (exponents) |v| try std.testing.expect(v);
    for (mantissas) |v| try std.testing.expect(v);
    var reversed_differs = false;
    for (0..16) |g| {
        var sum: f32 = 0;
        var reverse: f32 = 0;
        for (0..64) |j| {
            sum += @as(f32, @bitCast(@as(u32, x[g * 64 + j]) << 16));
            reverse += @as(f32, @bitCast(@as(u32, x[g * 64 + 63 - j]) << 16));
        }
        try std.testing.expectEqual(@as(u32, @bitCast(sum)), @as(u32, @bitCast(xs[g * rows])));
        reversed_differs = reversed_differs or @as(u32, @bitCast(sum)) != @as(u32, @bitCast(reverse));
    }
    try std.testing.expect(reversed_differs);
    try inputs(x, xs, 1);
    try std.testing.expect(x[17] != first);
}

test "native GU startup actual ABBA construction balances both seeds in selection and holdout" {
    for ([_]bool{ false, true }) |holdout| {
        var inner: [2]usize = @splat(0);
        var outer: [2]usize = @splat(0);
        for (0..pair_count) |i| {
            const first = candidateTurn(i, 0, holdout);
            try std.testing.expectEqual(first, candidateTurn(i, 3, holdout));
            try std.testing.expectEqual(!first, candidateTurn(i, 1, holdout));
            try std.testing.expectEqual(!first, candidateTurn(i, 2, holdout));
            if (first) outer[i % 2] += 1 else inner[i % 2] += 1;
        }
        for (0..2) |seed| {
            try std.testing.expectEqual(@as(usize, 2), inner[seed]);
            try std.testing.expectEqual(@as(usize, 2), outer[seed]);
        }
    }
}

test "native GU startup actual allocation budget fails closed at bounds and overflow" {
    try std.testing.expect(budgetFits(100, 200, 100));
    try std.testing.expect(!budgetFits(101, 200, 100));
    try std.testing.expect(!budgetFits(201, 200, 0));
    try std.testing.expect(budgetFits(0, 0, 0));
    try std.testing.expect(budgetFits(std.math.maxInt(u64) - 3, std.math.maxInt(u64), 3));
    try std.testing.expect(!budgetFits(std.math.maxInt(u64) - 3, std.math.maxInt(u64), 4));
    try std.testing.expect(!budgetFits(std.math.maxInt(u64), 2, std.math.maxInt(u64)));
}

test "native GU startup actual partial-row validator catches padding finite and HS faults" {
    try std.testing.expectEqualSlices(u32, &.{ 128, 1, 17, 37 }, &identity_rows);
    try std.testing.expectEqual(@as(usize, 116), max_cohorts);
    var y: [4 * 64]u16 = @splat(0xFFFF);
    @memset(y[0 .. 2 * 64], 0x3F80);
    var hs: [4]f32 = .{ 64, 64, @as(f32, @bitCast(@as(u32, 0xFFFFFFFF))), @as(f32, @bitCast(@as(u32, 0xFFFFFFFF))) };
    try checkRows(&y, &hs, 64, 4, 2, true);
    y[2 * 64] = 0;
    try std.testing.expectError(error.NativeGuGuard, checkRows(&y, &hs, 64, 4, 2, true));
    y[2 * 64] = 0xFFFF;
    hs[2] = 0;
    try std.testing.expectError(error.NativeGuGuard, checkRows(&y, &hs, 64, 4, 2, true));
    hs[2] = @bitCast(@as(u32, 0xFFFFFFFF));
    y[0] = 0x7FC1;
    try std.testing.expectError(error.NativeGuNonfinite, checkRows(&y, &hs, 64, 4, 2, true));
    y[0] = 0x3F80;
    hs[0] = 63;
    try std.testing.expectError(error.NativeGuSums, checkRows(&y, &hs, 64, 4, 2, true));
    try checkRows(&y, &hs, 64, 4, 2, false);
    try std.testing.expectError(error.NativeGuShape, checkRows(&y, &hs, 64, 4, 5, true));
    var wide_y: [4 * 128]u16 = @splat(0xFFFF);
    for (0..2) |r| {
        @memset(wide_y[r * 128 ..][0..64], 0x3F80);
        @memset(wide_y[r * 128 + 64 ..][0..64], 0x4000);
    }
    const nan: f32 = @bitCast(@as(u32, 0xFFFFFFFF));
    var wide_hs: [8]f32 = .{ 64, 64, nan, nan, 128, 128, nan, nan };
    try checkRows(&wide_y, &wide_hs, 128, 4, 2, true);
    wide_hs[4] = 64;
    try std.testing.expectError(error.NativeGuSums, checkRows(&wide_y, &wide_hs, 128, 4, 2, true));
    wide_hs[4] = 128;
    wide_hs[6] = 0;
    try std.testing.expectError(error.NativeGuGuard, checkRows(&wide_y, &wide_hs, 128, 4, 2, true));
}

test "HALF GU required admission consumes numerical checks and unchanged performance guard" {
    var strong: Fake = .{};
    try std.testing.expect((try admit(&strong, .required)).native);
    try std.testing.expectEqual(@as(usize, 116), strong.cohorts);
    var slow: Fake = .{ .weak_pair = 0 };
    const rejected = try admit(&slow, .required);
    try std.testing.expect(!rejected.native and rejected.status == .selection);
    var holdout: Fake = .{ .weak_holdout = true };
    const held = try admit(&holdout, .required);
    try std.testing.expect(!held.native and held.status == .holdout);
    var noisy: Fake = .{ .noisy_holdout = true };
    try std.testing.expect(!(try admit(&noisy, .required)).native);
    var identity: Fake = .{ .numerical_error = true };
    try std.testing.expectError(error.NativeGuIdentity, admit(&identity, .required));
    var tail: Fake = .{ .tail_error = true };
    try std.testing.expectError(error.NativeGuGuard, admit(&tail, .required));
    var off: Fake = .{};
    try std.testing.expect(!(try admit(&off, .off)).native);
    try std.testing.expectEqual(@as(usize, 0), off.len);
    try std.testing.expectEqual(@as(usize, 116), cohortLimit(.required));
    try std.testing.expectEqual(@as(usize, 116), cohortLimit(.adaptive));
}

test "HALF GU required production decision returns precise admission failures" {
    try requireDecision(.required, .{ .status = .native, .native = true });
    try requireDecision(.off, .{ .status = .selection });
    try requireDecision(.adaptive, .{ .status = .holdout });
    try std.testing.expectError(error.NativeGuPerformanceSelection, requireDecision(.required, .{ .status = .selection }));
    try std.testing.expectError(error.NativeGuPerformanceHoldout, requireDecision(.required, .{ .status = .holdout }));
    try std.testing.expectError(error.NativeGuRequired, requireDecision(.required, .{ .status = .shape }));
}
