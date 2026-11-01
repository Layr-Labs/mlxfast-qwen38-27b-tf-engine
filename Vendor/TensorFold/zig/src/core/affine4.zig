//! CPU affine q4/group64 preparation preserves BF16 metadata and low-bit-first MLX words.
const std = @import("std");
pub const Prepared = struct {
    allocator: std.mem.Allocator,
    n: usize,
    k: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,
    /// Code width: 4 for `prepare`, else `prepareBits`' MLX bit stream of `bits`-bit codes.
    bits: u8 = 4,
    /// Elements a (scale, bias) pair covers: 64 for `prepare`; `prepareBits` also takes 128.
    group: u32 = 64,
    /// `prepareNibbles`: `bits`-bit codes stored one a nibble in `prepare`'s layout (the reg path packs them down).
    nibbles: bool = false,
    /// Bits a code takes in `words`.
    pub fn wordBits(p: Prepared) u8 {
        return if (p.nibbles) 4 else p.bits;
    }
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
    for (1..workers) |w| threads[w] = std.Thread.spawn(.{}, run, .{ &results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per), 15 }) catch null;
    run(&results[0], weights, words, scales, biases, 0, @min(scales.len, per), 15);
    for (1..workers) |w| if (threads[w]) |t| t.join() else run(&results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per), 15);
    for (results[0..workers]) |r| try r;
    return .{ .allocator = a, .n = n, .k = k, .words = words, .scales = scales, .biases = biases };
}
/// `prepare`'s layout and rule at 2^bits - 1 levels (bits 2 or 3): codes 0 to 2^bits - 1, one a nibble, the scale the
/// group's range over those levels.
pub fn prepareNibbles(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, bits: u8) !Prepared {
    if (bits != 2 and bits != 3) return error.BadAffineShape;
    if (n == 0 or k == 0 or k % 64 != 0 or weights.len != try std.math.mul(usize, n, k)) return error.BadAffineShape;
    const levels: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
    const words = try a.alloc(u32, weights.len / 8);
    errdefer a.free(words);
    const scales = try a.alloc(u16, weights.len / 64);
    errdefer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    errdefer a.free(biases);
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, scales.len / 4096));
    const per = (scales.len + workers - 1) / workers;
    var results: [max_workers]Error!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |w| threads[w] = std.Thread.spawn(.{}, run, .{ &results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per), levels }) catch null;
    run(&results[0], weights, words, scales, biases, 0, @min(scales.len, per), levels);
    for (1..workers) |w| if (threads[w]) |t| t.join() else run(&results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per), levels);
    for (results[0..workers]) |r| try r;
    return .{ .allocator = a, .n = n, .k = k, .words = words, .scales = scales, .biases = biases, .bits = bits, .nibbles = true };
}
const max_workers = 16;
/// `prepare` with `bits`-bit codes (2, 3, 4, 5, 6 or 8): the same affine rule at 2^bits - 1 levels, each row's codes one
/// low-bit-first bit stream (MLX's packed W), so a group of `group` elements owns group * bits / 32 whole words.
pub fn prepareBits(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, bits: u8, group: u32) !Prepared {
    if (bits == 4 and group == 64) return prepare(a, weights, n, k);
    if (bits < 2 or bits > 8 or bits == 7 or (group != 64 and group != 128)) return error.BadAffineShape;
    if (n == 0 or k == 0 or k % group != 0 or weights.len != try std.math.mul(usize, n, k)) return error.BadAffineShape;
    const words = try a.alloc(u32, weights.len / 32 * bits);
    errdefer a.free(words);
    const scales = try a.alloc(u16, weights.len / group);
    errdefer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    errdefer a.free(biases);
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, scales.len / 4096));
    const per = (scales.len + workers - 1) / workers;
    var results: [max_workers]Error!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |w| threads[w] = std.Thread.spawn(.{}, runBits, .{ &results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per), bits, group }) catch null;
    runBits(&results[0], weights, words, scales, biases, 0, @min(scales.len, per), bits, group);
    for (1..workers) |w| if (threads[w]) |t| t.join() else runBits(&results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per), bits, group);
    for (results[0..workers]) |r| try r;
    return .{ .allocator = a, .n = n, .k = k, .words = words, .scales = scales, .biases = biases, .bits = bits, .group = group };
}
fn runBits(out: *Error!void, weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, bits: u8, size: u32) void {
    out.* = groupsBits(weights, words, scales, biases, first, last, bits, size);
}
fn groupsBits(weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, bits: u8, size: u32) Error!void {
    const levels: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    const per_group = @as(usize, size) * bits / 32;
    for (first..last) |group| {
        const values = weights[group * size ..][0..size];
        for (values) |b| if (!std.math.isFinite(asFloat(b))) return error.NonFiniteAffineInput;
        var lower: f32 = std.math.inf(f32);
        var upper: f32 = 0;
        for (values) |b| {
            const v = asFloat(b);
            lower = @min(lower, v);
            upper = @max(upper, v);
        }
        var bias: f32 = if (@abs(lower) > @abs(upper)) lower else upper;
        var scale: f32 = @max((upper - lower) / levels, @as(f32, 1e-7));
        if (bias >= 0) scale = -scale;
        const zero = @round(bias / scale);
        if (zero != 0) scale = bias / zero else bias = 0;
        if (!std.math.isFinite(scale) or scale == 0) return error.NonFiniteAffineRange;
        scales[group] = asBf16(scale);
        biases[group] = asBf16(bias);
        const out = words[group * per_group ..][0..per_group];
        @memset(out, 0);
        for (values, 0..) |b, index| {
            const code: u32 = @intFromFloat(@min(levels, @max(@as(f32, 0), @round((asFloat(b) - bias) / scale))));
            const bit = index * bits;
            const shift: u5 = @intCast(bit % 32);
            out[bit / 32] |= code << shift;
            if (bit % 32 + bits > 32) out[bit / 32 + 1] |= code >> @intCast(32 - @as(u32, shift));
        }
    }
}

const Error = error{ NonFiniteAffineInput, NonFiniteAffineRange };
fn run(out: *Error!void, weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, levels: u32) void {
    out.* = groups(weights, words, scales, biases, first, last, levels);
}
/// Groups [first, last): each group's 64 inputs to one scale, one bias and eight low-bit-first words (codes 0 to `levels`).
fn groups(weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, levels_int: u32) Error!void {
    const levels: f32 = @floatFromInt(levels_int);
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
        if (levels_int < 15) {
            try clipped(values, words[group * 8 ..][0..8], &scales[group], &biases[group], lower, upper, levels);
            continue;
        }
        var bias: f32 = if (@abs(lower) > @abs(upper)) lower else upper;
        var scale: f32 = @max((upper - lower) / levels, @as(f32, 1e-7));
        if (bias >= 0) scale = -scale;
        const zero = @round(bias / scale);
        if (zero != 0) scale = bias / zero else bias = 0;
        if (!std.math.isFinite(scale) or scale == 0) return error.NonFiniteAffineRange;
        scales[group] = asBf16(scale);
        biases[group] = asBf16(bias);
        @memset(words[group * 8 ..][0..8], 0);
        for (values, 0..) |bits, index| {
            const code: u32 = @intFromFloat(@min(levels, @max(@as(f32, 0), @round((asFloat(bits) - bias) / scale))));
            words[group * 8 + index / 8] |= code << @as(u5, @intCast(4 * (index % 8)));
        }
    }
}
/// Below 4 bits the full range wastes levels on a group's outliers: the same rule (bias at the extreme of larger
/// magnitude, zero point snapped to an integer) on the range shrunk by each factor below, keeping the one whose bf16
/// scale and bias give the least squared error over the group. Codes clamp to the levels, so clipped values saturate.
fn clipped(values: []const u16, out: []u32, scale_out: *u16, bias_out: *u16, lower: f32, upper: f32, levels: f32) Error!void {
    const shrinks = [_]f32{ 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4 };
    var best_err: f32 = std.math.inf(f32);
    var best_scale: u16 = 0;
    var best_bias: u16 = 0;
    for (shrinks) |shrink| {
        const lo = lower * shrink;
        const hi = upper * shrink;
        var bias: f32 = if (@abs(lo) > @abs(hi)) lo else hi;
        var scale: f32 = @max((hi - lo) / levels, @as(f32, 1e-7));
        if (bias >= 0) scale = -scale;
        const zero = @round(bias / scale);
        if (zero != 0) scale = bias / zero else bias = 0;
        if (!std.math.isFinite(scale) or scale == 0) continue;
        const s16 = asBf16(scale);
        const b16 = asBf16(bias);
        const s = asFloat(s16);
        const b = asFloat(b16);
        if (s == 0 or !std.math.isFinite(s)) continue;
        var err: f32 = 0;
        for (values) |bits| {
            const v = asFloat(bits);
            const code = @min(levels, @max(@as(f32, 0), @round((v - b) / s)));
            const d = v - (b + s * code);
            err += d * d;
        }
        if (err < best_err) {
            best_err = err;
            best_scale = s16;
            best_bias = b16;
        }
    }
    if (!std.math.isFinite(best_err)) return error.NonFiniteAffineRange;
    scale_out.* = best_scale;
    bias_out.* = best_bias;
    const s = asFloat(best_scale);
    const b = asFloat(best_bias);
    @memset(out, 0);
    for (values, 0..) |bits, index| {
        const code: u32 = @intFromFloat(@min(levels, @max(@as(f32, 0), @round((asFloat(bits) - b) / s))));
        out[index / 8] |= code << @as(u5, @intCast(4 * (index % 8)));
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
    try groups(w, words, scales, biases, 0, scales.len, 15);
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

pub fn signature() [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("affine4.zig"), &result, .{});
    return result;
}

/// Refine only a drafter's runtime Q2 copy; the checkpoint and Q4 preparation stay unchanged.
/// A fractional zero point still uses the same affine four-code representation.
pub fn prepareNibblesRefined(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, bits: u8) !Prepared {
    var prepared = try prepareNibbles(a, weights, n, k, bits);
    errdefer prepared.deinit();
    if (bits != 2) return prepared;
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, prepared.scales.len / 4096));
    const per = (prepared.scales.len + workers - 1) / workers;
    var results: [max_workers]Error!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |w| threads[w] = std.Thread.spawn(.{}, refineRun, .{ &results[w], weights, prepared.words, prepared.scales, prepared.biases, w * per, @min(prepared.scales.len, (w + 1) * per) }) catch null;
    refineRun(&results[0], weights, prepared.words, prepared.scales, prepared.biases, 0, @min(prepared.scales.len, per));
    for (1..workers) |w| if (threads[w]) |thread| thread.join() else refineRun(&results[w], weights, prepared.words, prepared.scales, prepared.biases, w * per, @min(prepared.scales.len, (w + 1) * per));
    for (results[0..workers]) |result| try result;
    return prepared;
}
fn refineRun(out: *Error!void, weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize) void {
    out.* = refineGroups(weights, words, scales, biases, first, last);
}
fn refineGroups(weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize) Error!void {
    @setFloatMode(.strict);
    for (first..last) |group| {
        const values = weights[group * 64 ..][0..64];
        const output = words[group * 8 ..][0..8];
        const initial_scale: f64 = @abs(@as(f64, asFloat(scales[group])));
        const original_scale: f64 = asFloat(scales[group]);
        const original_bias: f64 = asFloat(biases[group]);
        var best_error: f64 = 0;
        for (values, 0..) |value, i| {
            const code: f64 = @floatFromInt((output[i / 8] >> @as(u5, @intCast(4 * (i % 8)))) & 15);
            const delta = @as(f64, asFloat(value)) - (original_bias + original_scale * code);
            best_error += delta * delta;
        }
        if (best_error == 0) continue;
        // Keep the original quantization unless a representable BF16 grid strictly improves its error.
        for ([_]f64{ 0, 0.5, 1, 1.5, 2, 2.5, 3 }) |zero| {
            var scale = initial_scale;
            for (0..8) |_| {
                const scale_bits = asBf16(@floatCast(scale));
                const s = asFloat(scale_bits);
                const bias_bits = asBf16(@floatCast(-zero * @as(f64, s)));
                const bias = asFloat(bias_bits);
                if (s == 0 or !std.math.isFinite(s) or !std.math.isFinite(bias)) break;
                var code_words: [8]u32 = @splat(0);
                var err: f64 = 0;
                var numerator: f64 = 0;
                var denominator: f64 = 0;
                for (values, 0..) |value, i| {
                    const v = asFloat(value);
                    const q = @min(@as(f32, 3), @max(@as(f32, 0), @round((v - bias) / s)));
                    const code: u32 = @intFromFloat(q);
                    code_words[i / 8] |= code << @as(u5, @intCast(4 * (i % 8)));
                    const delta = @as(f64, v) - (@as(f64, bias) + @as(f64, s) * @as(f64, q));
                    err += delta * delta;
                    const level = @as(f64, q) - zero;
                    numerator += @as(f64, v) * level;
                    denominator += level * level;
                }
                if (!std.math.isFinite(err)) return error.NonFiniteAffineRange;
                if (err < best_error) {
                    best_error = err;
                    scales[group] = scale_bits;
                    biases[group] = bias_bits;
                    @memcpy(output, &code_words);
                }
                if (denominator == 0) break;
                scale = @max(numerator / denominator, @as(f64, 1e-7));
            }
        }
    }
}
fn refinedSquaredError(values: []const u16, p: Prepared) f64 {
    @setFloatMode(.strict);
    var total: f64 = 0;
    for (values, 0..) |value, i| {
        const group = i / 64;
        const code: f64 = @floatFromInt((p.words[i / 8] >> @as(u5, @intCast(4 * (i % 8)))) & 15);
        const d = @as(f64, asFloat(value)) - (@as(f64, asFloat(p.biases[group])) + @as(f64, asFloat(p.scales[group])) * code);
        total += d * d;
    }
    return total;
}
test "fractional zero preserves an exact symmetric four-level BF16 grid" {
    const a = std.testing.allocator;
    var w: [64]u16 = undefined;
    for (&w, 0..) |*value, i| value.* = asBf16(([_]f32{ -1.5, -0.5, 0.5, 1.5 })[i % 4]);
    var original = try prepareNibbles(a, &w, 1, 64, 2);
    defer original.deinit();
    var refined = try prepareNibblesRefined(a, &w, 1, 64, 2);
    defer refined.deinit();
    try std.testing.expect(refinedSquaredError(&w, original) > 0);
    try std.testing.expectEqual(@as(f64, 0), refinedSquaredError(&w, refined));
}
test "refined Q2 never increases representable error and leaves Q3 preparation unchanged" {
    const a = std.testing.allocator;
    var w: [256 * 64]u16 = undefined;
    var random: u32 = 0x20261213;
    for (&w, 0..) |*value, i| {
        random = random *% 1664525 +% 1013904223;
        const v = (@as(f32, @floatFromInt(@as(i32, @intCast(random % 20001)) - 10000))) / 8192.0;
        value.* = asBf16(if (i / 64 == 0) 0 else if (i / 64 == 1) 0.125 else v);
    }
    var original = try prepareNibbles(a, &w, 256, 64, 2);
    defer original.deinit();
    var refined = try prepareNibblesRefined(a, &w, 256, 64, 2);
    defer refined.deinit();
    var repeated = try prepareNibblesRefined(a, &w, 256, 64, 2);
    defer repeated.deinit();
    try std.testing.expectEqualSlices(u32, refined.words, repeated.words);
    try std.testing.expectEqualSlices(u16, refined.scales, repeated.scales);
    try std.testing.expectEqualSlices(u16, refined.biases, repeated.biases);
    for (0..256) |g| {
        var orig_group = original;
        orig_group.words = original.words[g * 8 ..][0..8]; orig_group.scales = original.scales[g..][0..1]; orig_group.biases = original.biases[g..][0..1];
        var new_group = refined;
        new_group.words = refined.words[g * 8 ..][0..8]; new_group.scales = refined.scales[g..][0..1]; new_group.biases = refined.biases[g..][0..1];
        try std.testing.expect(refinedSquaredError(w[g * 64 ..][0..64], new_group) <= refinedSquaredError(w[g * 64 ..][0..64], orig_group));
    }
    for (refined.words) |word| try std.testing.expect((word & 0xcccccccc) == 0);
    var three = try prepareNibbles(a, &w, 256, 64, 3);
    defer three.deinit();
    var three_refined = try prepareNibblesRefined(a, &w, 256, 64, 3);
    defer three_refined.deinit();
    try std.testing.expectEqualSlices(u32, three.words, three_refined.words);
    try std.testing.expectEqualSlices(u16, three.scales, three_refined.scales);
    try std.testing.expectEqualSlices(u16, three.biases, three_refined.biases);
}
