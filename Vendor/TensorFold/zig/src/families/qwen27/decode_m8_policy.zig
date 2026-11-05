//! Conservative, pure-host decision for an already-qualified M8 decode specialization.
//! Timings are complete 64-projection cohorts; this module does not measure or qualify kernels.
const std = @import("std");

pub const cohort_pairs: usize = 8;
pub const null_pairs_per_arm: usize = 8;
pub const min_relative_gain: f64 = 0.10;
pub const min_cohort_saving_seconds: f64 = 0.0005;
pub const min_positive_pairs: usize = 7;
pub const null_margin: f64 = 0.02;

pub const NullPair = struct {
    first_seconds: f64,
    second_seconds: f64,
};

pub const Refusal = enum {
    none,
    identity_mismatch,
    incomplete,
    invalid_timing,
};

pub const Decision = struct {
    use_m8: bool = false,
    refusal: Refusal = .none,
    positive_pairs: usize = 0,
    median_relative_gain: f64 = 0,
    median_saving_seconds: f64 = 0,
    max_null_relative: f64 = 0,
    relative_gate: bool = false,
    absolute_gate: bool = false,
    consistency_gate: bool = false,
    null_gate: bool = false,
    pair_relative_gain: [cohort_pairs]f64 = @splat(0),
    pair_saving_seconds: [cohort_pairs]f64 = @splat(0),
};

/// Decide from all eight paired 64-matrix cohorts and both complete eight-pair null series.
/// The caller must supply paired samples in their original AB/BA order and hash-check their
/// outputs before calling; this function deliberately discards no sample.
pub fn decide(
    identity_ok: bool,
    m16_seconds: []const f64,
    m8_seconds: []const f64,
    m16_null: []const NullPair,
    m8_null: []const NullPair,
) Decision {
    var result: Decision = .{};
    if (!identity_ok) {
        result.refusal = .identity_mismatch;
        return result;
    }
    if (m16_seconds.len != cohort_pairs or m8_seconds.len != cohort_pairs or
        m16_null.len != null_pairs_per_arm or m8_null.len != null_pairs_per_arm)
    {
        result.refusal = .incomplete;
        return result;
    }

    var relative: [cohort_pairs]f64 = undefined;
    var savings: [cohort_pairs]f64 = undefined;
    for (m16_seconds, m8_seconds, 0..) |base, candidate, i| {
        if (!validTime(base) or !validTime(candidate)) {
            result.refusal = .invalid_timing;
            return result;
        }
        savings[i] = base - candidate;
        relative[i] = savings[i] / base;
        if (!std.math.isFinite(savings[i]) or !std.math.isFinite(relative[i])) {
            result.refusal = .invalid_timing;
            return result;
        }
        if (savings[i] > 0) result.positive_pairs += 1;
    }

    var null_max: f64 = 0;
    inline for (0..2) |arm| {
        const series = if (arm == 0) m16_null else m8_null;
        for (series) |pair| {
            if (!validTime(pair.first_seconds) or !validTime(pair.second_seconds)) {
                result.refusal = .invalid_timing;
                return result;
            }
            const center = (pair.first_seconds + pair.second_seconds) * 0.5;
            const deviation = @abs(pair.first_seconds - pair.second_seconds) / center;
            if (!std.math.isFinite(center) or center <= 0 or !std.math.isFinite(deviation)) {
                result.refusal = .invalid_timing;
                return result;
            }
            null_max = @max(null_max, deviation);
        }
    }

    result.pair_relative_gain = relative;
    result.pair_saving_seconds = savings;
    result.median_relative_gain = median(relative);
    result.median_saving_seconds = median(savings);
    result.max_null_relative = null_max;
    result.relative_gate = result.median_relative_gain >= min_relative_gain;
    result.absolute_gate = result.median_saving_seconds >= min_cohort_saving_seconds;
    result.consistency_gate = result.positive_pairs >= min_positive_pairs;
    result.null_gate = result.median_relative_gain > null_max + null_margin;
    result.use_m8 = result.relative_gate and result.absolute_gate and result.consistency_gate and result.null_gate;
    return result;
}

fn validTime(value: f64) bool {
    return std.math.isFinite(value) and value > 0;
}

fn median(values: [cohort_pairs]f64) f64 {
    var sorted = values;
    std.mem.sort(f64, &sorted, {}, std.sort.asc(f64));
    return (sorted[3] + sorted[4]) * 0.5;
}

fn nulls(comptime a: f64, comptime b: f64) [null_pairs_per_arm]NullPair {
    return @splat(.{ .first_seconds = a, .second_seconds = b });
}

fn times(comptime value: f64) [cohort_pairs]f64 {
    return @splat(value);
}

test "clear cohort win clears every gate" {
    const d = decide(true, &times(0.010), &times(0.0088), &nulls(0.010, 0.010), &nulls(0.0088, 0.0088));
    try std.testing.expect(d.use_m8);
    try std.testing.expectEqual(@as(usize, 8), d.positive_pairs);
    try std.testing.expectApproxEqAbs(@as(f64, 0.12), d.median_relative_gain, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0012), d.median_saving_seconds, 1e-12);
}

test "identity mismatch and incomplete evidence fail closed" {
    const full = times(0.010);
    const candidate = times(0.008);
    const ns = nulls(0.010, 0.010);
    try std.testing.expectEqual(Refusal.identity_mismatch, decide(false, &full, &candidate, &ns, &ns).refusal);
    try std.testing.expectEqual(Refusal.incomplete, decide(true, full[0..7], &candidate, &ns, &ns).refusal);
    try std.testing.expect(!decide(true, &full, &candidate, ns[0..7], &ns).use_m8);
}

test "timing values must be finite and positive" {
    var bad = times(0.008);
    bad[3] = std.math.nan(f64);
    const base = times(0.010);
    const ns = nulls(0.010, 0.010);
    try std.testing.expectEqual(Refusal.invalid_timing, decide(true, &base, &bad, &ns, &ns).refusal);
    bad = times(0.008);
    bad[3] = 0;
    try std.testing.expectEqual(Refusal.invalid_timing, decide(true, &base, &bad, &ns, &ns).refusal);
}

test "finite inputs whose null arithmetic overflows are refused" {
    const base = times(1.0);
    const candidate = times(0.8);
    const huge = nulls(std.math.floatMax(f64), std.math.floatMax(f64));
    try std.testing.expectEqual(Refusal.invalid_timing, decide(true, &base, &candidate, &huge, &huge).refusal);
}

test "relative gain threshold is enforced" {
    const base = times(0.010);
    const candidate = times(0.00901); // 9.9%: a lowered 9% threshold would incorrectly accept.
    const ns = nulls(0.010, 0.010);
    const d = decide(true, &base, &candidate, &ns, &ns);
    try std.testing.expect(!d.use_m8);
    try std.testing.expect(d.median_relative_gain < min_relative_gain);
}

test "absolute cohort saving threshold is enforced" {
    const base = times(0.00489);
    const candidate = times(0.00438); // 0.51 ms clears the boundary with float margin.
    const ns = nulls(0.00489, 0.00489);
    try std.testing.expect(decide(true, &base, &candidate, &ns, &ns).use_m8);
    const below = times(0.0043901); // 0.4999 ms must fail despite >10% relative gain.
    try std.testing.expect(!decide(true, &base, &below, &ns, &ns).use_m8);
}

test "at least seven paired cohorts must improve" {
    const base = times(0.010);
    var candidate = times(0.008);
    candidate[6] = 0.010001;
    candidate[7] = 0.010001;
    const ns = nulls(0.010, 0.010);
    const d = decide(true, &base, &candidate, &ns, &ns);
    try std.testing.expectEqual(@as(usize, 6), d.positive_pairs);
    try std.testing.expect(d.median_relative_gain >= min_relative_gain);
    try std.testing.expect(!d.use_m8);
}

test "full null envelope plus margin rejects an apparent small win" {
    const base = times(0.010);
    const candidate = times(0.00895); // 10.5% median, but null max is 9% plus 2% margin.
    var baseline_null = nulls(0.010, 0.010);
    baseline_null[0] = .{ .first_seconds = 0.010, .second_seconds = 0.0109 };
    const candidate_null = nulls(0.00895, 0.00895);
    const d = decide(true, &base, &candidate, &baseline_null, &candidate_null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.105), d.median_relative_gain, 1e-12);
    try std.testing.expect(d.max_null_relative > 0.08);
    try std.testing.expect(!d.use_m8);
}

test "negative cohorts are retained and cannot pass by cherry-picking" {
    const base = times(0.010);
    const candidate = [_]f64{ 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.0101, 0.0101 };
    const ns = nulls(0.010, 0.010);
    const d = decide(true, &base, &candidate, &ns, &ns);
    try std.testing.expectEqual(@as(usize, 6), d.positive_pairs);
    try std.testing.expect(!d.use_m8);
}
