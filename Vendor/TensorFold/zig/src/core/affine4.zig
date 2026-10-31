//! CPU affine q4/group64 preparation preserves BF16 metadata and low-bit-first MLX words.
const std = @import("std");
pub const Prepared = struct {
    allocator: std.mem.Allocator,
    n: usize,
    k: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,
    pub fn deinit(p: *Prepared) void {
        p.allocator.free(p.words);
        p.allocator.free(p.scales);
        p.allocator.free(p.biases);
        p.* = undefined;
    }
};
pub fn asFloat(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
pub fn asBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
pub fn prepare(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize) !Prepared {
    if (n == 0 or k == 0 or k % 64 != 0 or weights.len != try std.math.mul(usize, n, k)) return error.BadAffineShape;
    const words = try a.alloc(u32, weights.len / 8);
    errdefer a.free(words);
    const scales = try a.alloc(u16, weights.len / 64);
    errdefer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    errdefer a.free(biases);
    // groups are independent: workers take contiguous ranges, so every byte matches one serial pass
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, scales.len / 4096));
    const per = (scales.len + workers - 1) / workers;
    var results: [max_workers]Error!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |w| threads[w] = std.Thread.spawn(.{}, run, .{ &results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per) }) catch null;
    run(&results[0], weights, words, scales, biases, 0, @min(scales.len, per));
    for (1..workers) |w| if (threads[w]) |t| t.join() else run(&results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per));
    for (results[0..workers]) |r| try r;
    return .{ .allocator = a, .n = n, .k = k, .words = words, .scales = scales, .biases = biases };
}
const max_workers = 16;
const Error = error{ NonFiniteAffineInput, NonFiniteAffineRange };
fn run(out: *Error!void, weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize) void {
    out.* = groups(weights, words, scales, biases, first, last);
}
/// Starting ranges of the refit: the group's [min, max] (narrower starts fit the weights closer but drafted worse).
const shrink = [_]f32{1.0};
/// Code assignments and least-squares refits the start runs.
const passes = 8;

/// The least-squares (scale, bias) of `values` on fixed codes `q`, rounded to bf16; null when the codes are all equal or
/// the fit is not a finite nonzero scale.
fn fit(values: *const [64]f32, q: *const [64]f32) ?[2]f32 {
    var qm: f64 = 0;
    var wm: f64 = 0;
    for (values, q) |v, c| {
        qm += c;
        wm += v;
    }
    qm /= 64;
    wm /= 64;
    var variance: f64 = 0;
    var covariance: f64 = 0;
    for (values, q) |v, c| {
        variance += (c - qm) * (c - qm);
        covariance += (c - qm) * (v - wm);
    }
    if (variance == 0) return null;
    const scale = asFloat(asBf16(@floatCast(covariance / variance)));
    const bias = asFloat(asBf16(@floatCast(wm - @as(f64, scale) * qm)));
    if (!std.math.isFinite(scale) or !std.math.isFinite(bias) or scale == 0) return null;
    return .{ scale, bias };
}

/// Each value's nearest code under (scale, bias).
fn assign(values: *const [64]f32, scale: f32, bias: f32, q: *[64]f32) void {
    for (values, q) |v, *c| c.* = @min(@as(f32, 15), @max(@as(f32, 0), @round((v - bias) / scale)));
}

/// The squared reconstruction error of codes `q` under (scale, bias).
fn squared(values: *const [64]f32, scale: f32, bias: f32, q: *const [64]f32) f64 {
    var e: f64 = 0;
    for (values, q) |v, c| {
        const d: f64 = @as(f64, scale) * c + bias - v;
        e += d * d;
    }
    return e;
}

/// Groups [first, last): each group's 64 inputs to one scale, one bias and eight low-bit-first words. The MLX range fit
/// below is refined toward the bf16 weights: from the group's [min, max], codes are assigned and (scale, bias) refit by
/// least squares eight times, and the codes and bf16 pair with the smaller squared error are kept (the range fit when
/// the refit does not beat it). The drafter's arithmetic on these words is unchanged; its weights only
/// sit closer to the bf16 ones.
fn groups(weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize) Error!void {
    for (first..last) |group| {
        const values = weights[group * 64 ..][0..64];
        for (values) |bits| if (!std.math.isFinite(asFloat(bits))) return error.NonFiniteAffineInput;
        var lower: f32 = std.math.inf(f32);
        var upper: f32 = 0;
        for (values) |bits| {
            const v = asFloat(bits);
            lower = @min(lower, v);
            upper = @max(upper, v);
        }
        var bias: f32 = if (@abs(lower) > @abs(upper)) lower else upper;
        var scale: f32 = @max((upper - lower) / @as(f32, 15), @as(f32, 1e-7));
        if (bias >= 0) scale = -scale;
        const zero = @round(bias / scale);
        if (zero != 0) scale = bias / zero else bias = 0;
        if (!std.math.isFinite(scale) or scale == 0) return error.NonFiniteAffineRange;
        var w: [64]f32 = undefined;
        for (values, &w) |bits, *v| v.* = asFloat(bits);
        var best: [64]f32 = undefined;
        for (&w, &best) |v, *c| c.* = @min(@as(f32, 15), @max(@as(f32, 0), @round((v - bias) / scale)));
        var best_scale = asFloat(asBf16(scale));
        var best_bias = asFloat(asBf16(bias));
        var best_error = squared(&w, best_scale, best_bias, &best);
        const low = std.mem.min(f32, &w);
        const high = std.mem.max(f32, &w);
        if (high > low) for (shrink) |fraction| {
            const centre = (high + low) / 2;
            const radius = (high - low) / 2 * fraction;
            var s = asFloat(asBf16(@max(2 * radius / 15, 1e-12)));
            var b = asFloat(asBf16(centre - radius));
            var q: [64]f32 = undefined;
            var fitted = true;
            for (0..passes) |_| {
                assign(&w, s, b, &q);
                const pair = fit(&w, &q) orelse {
                    fitted = false;
                    break;
                };
                s = pair[0];
                b = pair[1];
            }
            if (!fitted) continue;
            const e = squared(&w, s, b, &q);
            if (e < best_error) {
                best_error = e;
                best_scale = s;
                best_bias = b;
                best = q;
            }
        };
        scales[group] = asBf16(best_scale);
        biases[group] = asBf16(best_bias);
        @memset(words[group * 8 ..][0..8], 0);
        for (best, 0..) |c, index| {
            const code: u32 = @intFromFloat(c);
            words[group * 8 + index / 8] |= code << @as(u5, @intCast(4 * (index % 8)));
        }
    }
}
test "q4 shape and finite input checks occur before preparation" {
    var w: [64]u16 = @splat(0);
    try std.testing.expectError(error.BadAffineShape, prepare(std.testing.allocator, &w, 1, 63));
    w[7] = 0x7fc0;
    try std.testing.expectError(error.NonFiniteAffineInput, prepare(std.testing.allocator, &w, 1, 64));
}
test "a threaded preparation gives the serial pass's bytes" {
    const a = std.testing.allocator;
    const n = 64;
    const k = 4096;
    const w = try a.alloc(u16, n * k);
    defer a.free(w);
    var prng = std.Random.DefaultPrng.init(7);
    for (w) |*x| x.* = asBf16(prng.random().floatNorm(f32));
    var p = try prepare(a, w, n, k);
    defer p.deinit();
    const words = try a.alloc(u32, w.len / 8);
    defer a.free(words);
    const scales = try a.alloc(u16, w.len / 64);
    defer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    defer a.free(biases);
    try groups(w, words, scales, biases, 0, scales.len);
    try std.testing.expectEqualSlices(u32, words, p.words);
    try std.testing.expectEqualSlices(u16, scales, p.scales);
    try std.testing.expectEqualSlices(u16, biases, p.biases);
}
test "constant positive, negative and tiny groups retain signed scales and exact zero codes" {
    const a = std.testing.allocator;
    var w: [3 * 64]u16 = undefined;
    @memset(w[0..64], 0x3f80);
    @memset(w[64..128], 0xbf80);
    @memset(w[128..], asBf16(1e-20));
    var p = try prepare(a, &w, 3, 64);
    defer p.deinit();
    try std.testing.expectEqual(@as(u16, 0xb3d7), p.scales[0]);
    try std.testing.expectEqual(@as(u16, 0x3d89), p.scales[1]);
    try std.testing.expectEqual(@as(u16, 0), p.biases[2]);
    for (p.words) |word| try std.testing.expectEqual(@as(u32, 0), word);
}

test "the refit reconstructs normal weights with less squared error than the range fit" {
    const a = std.testing.allocator;
    const n = 32;
    const k = 1024;
    const w = try a.alloc(u16, n * k);
    defer a.free(w);
    var prng = std.Random.DefaultPrng.init(11);
    for (w) |*x| x.* = asBf16(prng.random().floatNorm(f32) * 0.02);
    var p = try prepare(a, w, n, k);
    defer p.deinit();
    var refit: f64 = 0;
    var range: f64 = 0;
    for (0..w.len / 64) |g| {
        const s = asFloat(p.scales[g]);
        const b = asFloat(p.biases[g]);
        var lower: f32 = std.math.inf(f32);
        var upper: f32 = 0;
        for (w[g * 64 ..][0..64]) |x| {
            lower = @min(lower, asFloat(x));
            upper = @max(upper, asFloat(x));
        }
        var rb: f32 = if (@abs(lower) > @abs(upper)) lower else upper;
        var rs: f32 = @max((upper - lower) / @as(f32, 15), @as(f32, 1e-7));
        if (rb >= 0) rs = -rs;
        const zero = @round(rb / rs);
        if (zero != 0) rs = rb / zero else rb = 0;
        for (w[g * 64 ..][0..64], 0..) |x, i| {
            const code: f32 = @floatFromInt((p.words[g * 8 + i / 8] >> @as(u5, @intCast(4 * (i % 8)))) & 15);
            const d = s * code + b - asFloat(x);
            refit += d * d;
            const rc = @min(@as(f32, 15), @max(@as(f32, 0), @round((asFloat(x) - rb) / rs)));
            const r = asFloat(asBf16(rs)) * rc + asFloat(asBf16(rb)) - asFloat(x);
            range += r * r;
        }
    }
    try std.testing.expect(refit < 0.9 * range);
}

pub fn signature() [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("affine4.zig"), &result, .{});
    return result;
}
