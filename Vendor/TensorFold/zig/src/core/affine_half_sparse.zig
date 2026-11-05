//! Exact FP32 affine Q4 coefficients represented by an FP16 high part and sparse FP32 residuals.
const std = @import("std");

pub const Matrix = struct {
    allocator: std.mem.Allocator,
    rows: usize,
    k: usize,
    high: []f16,
    offsets: []u32,
    columns: []u16,
    deltas: []f32,

    pub fn deinit(m: *Matrix) void {
        m.allocator.free(m.deltas);
        m.allocator.free(m.columns);
        m.allocator.free(m.offsets);
        m.allocator.free(m.high);
        m.* = undefined;
    }
};

/// Convert row-major low-nibble-first Q4 affine tensors to FP16 plus every exact FP32 residual.
/// Each BF16 scale and bias is widened exactly. An integer-rational check rejects coefficients
/// whose FP32 multiply-add is not the exact real affine value, so callers can retain Q4 fallback.
pub fn convert(
    allocator: std.mem.Allocator,
    rows: usize,
    k: usize,
    codes: []const u32,
    scales_bf16: []const u16,
    biases_bf16: []const u16,
) !Matrix {
    if (rows == 0 or k == 0 or k % 64 != 0 or k > std.math.maxInt(u16) + 1) return error.BadShape;
    const count = try std.math.mul(usize, rows, k);
    const code_count = try std.math.mul(usize, rows, k / 8);
    const group_count = try std.math.mul(usize, rows, k / 64);
    if (codes.len != code_count or scales_bf16.len != group_count or biases_bf16.len != group_count) return error.BadShape;

    var correction_count: usize = 0;
    for (0..rows) |r| for (0..k) |col| {
        const coeff = try coefficient(r, col, k, codes, scales_bf16, biases_bf16);
        const high: f16 = @floatCast(coeff);
        if (!std.math.isFinite(high)) return error.HalfOverflow;
        const delta = residual(coeff, high);
        try checkReconstruction(coeff, high, delta);
        if (hasResidual(delta)) correction_count += 1;
    };
    if (correction_count > std.math.maxInt(u32)) return error.TooManyCorrections;

    const high = try allocator.alloc(f16, count);
    errdefer allocator.free(high);
    const offsets = try allocator.alloc(u32, rows + 1);
    errdefer allocator.free(offsets);
    const columns = try allocator.alloc(u16, correction_count);
    errdefer allocator.free(columns);
    const deltas = try allocator.alloc(f32, correction_count);
    errdefer allocator.free(deltas);

    var cursor: usize = 0;
    offsets[0] = 0;
    for (0..rows) |r| {
        for (0..k) |col| {
            const coeff = try coefficient(r, col, k, codes, scales_bf16, biases_bf16);
            const h: f16 = @floatCast(coeff);
            const d = residual(coeff, h);
            try checkReconstruction(coeff, h, d);
            high[r * k + col] = h;
            if (hasResidual(d)) {
                columns[cursor] = @intCast(col);
                deltas[cursor] = d;
                cursor += 1;
            }
        }
        offsets[r + 1] = @intCast(cursor);
    }
    std.debug.assert(cursor == correction_count);
    return .{ .allocator = allocator, .rows = rows, .k = k, .high = high, .offsets = offsets, .columns = columns, .deltas = deltas };
}

fn coefficient(r: usize, col: usize, k: usize, codes: []const u32, scales: []const u16, biases: []const u16) !f32 {
    const word_index = r * (k / 8) + col / 8;
    const q: u32 = (codes[word_index] >> @intCast(4 * (col % 8))) & 0xf;
    const group_index = r * (k / 64) + col / 64;
    const scale = bf16ToF32(scales[group_index]);
    const bias = bf16ToF32(biases[group_index]);
    if (!std.math.isFinite(scale) or !std.math.isFinite(bias)) return error.NonFiniteInput;

    const product: f32 = scale * @as(f32, @floatFromInt(q));
    const value: f32 = product + bias;
    if (!std.math.isFinite(value)) return error.NonFiniteCoefficient;
    try checkExactAffine(scale, q, bias, value);
    return value;
}

const Rational = struct { significand: i128, exponent: i32 };

fn f32Rational(value: f32) Rational {
    const bits: u32 = @bitCast(value);
    const exponent_bits = (bits >> 23) & 0xff;
    const fraction = bits & 0x7fffff;
    var significand: i128 = if (exponent_bits == 0) fraction else (1 << 23) | fraction;
    if ((bits >> 31) != 0) significand = -significand;
    return .{ .significand = significand, .exponent = if (exponent_bits == 0) -149 else @as(i32, @intCast(exponent_bits)) - 150 };
}

fn normalized(value: Rational) Rational {
    if (value.significand == 0) return .{ .significand = 0, .exponent = 0 };
    const magnitude: u128 = @intCast(@abs(value.significand));
    const trailing = @ctz(magnitude);
    const divisor: i128 = @as(i128, 1) << @intCast(trailing);
    return .{ .significand = @divExact(value.significand, divisor), .exponent = value.exponent + @as(i32, @intCast(trailing)) };
}

fn shifted(value: i128, by: u32) i128 {
    const factor: i128 = @as(i128, 1) << @intCast(by);
    return value * factor;
}

fn checkExactAffine(scale: f32, q: u32, bias: f32, value: f32) !void {
    const s = f32Rational(scale);
    const b = f32Rational(bias);
    const product = Rational{ .significand = s.significand * @as(i128, @intCast(q)), .exponent = s.exponent };
    if (product.significand == 0 and b.significand == 0) {
        if (value != 0) return error.CoefficientNotFP32Exact;
        return;
    }
    const common_exp = if (product.significand == 0) b.exponent else if (b.significand == 0) product.exponent else @min(product.exponent, b.exponent);
    const product_shift: u32 = if (product.significand == 0) 0 else @intCast(product.exponent - common_exp);
    const bias_shift: u32 = if (b.significand == 0) 0 else @intCast(b.exponent - common_exp);
    // Beyond this span, exact alignment could exceed the signed 128-bit accumulator.
    if (product_shift > 80 or bias_shift > 80) return error.CoefficientExponentSpan;
    const sum = shifted(product.significand, product_shift) + shifted(b.significand, bias_shift);
    const exact = normalized(.{ .significand = sum, .exponent = common_exp });
    const actual = normalized(f32Rational(value));
    if (exact.significand != actual.significand or exact.exponent != actual.exponent) return error.CoefficientNotFP32Exact;
}

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn residual(value: f32, high: f16) f32 {
    const wide: f32 = high;
    if (value == 0 and wide == 0) return @bitCast(@as(u32, @bitCast(value)) & 0x80000000);
    return value - wide;
}

fn checkReconstruction(value: f32, high: f16, delta: f32) !void {
    if (!std.math.isFinite(delta)) return error.NonFiniteResidual;
    const reconstructed: f32 = @as(f32, high) + delta;
    if (@as(u32, @bitCast(reconstructed)) != @as(u32, @bitCast(value))) return error.NonReconstructible;
}

fn hasResidual(delta: f32) bool {
    // Negative zero must remain explicit: adding it to a negative-zero high preserves the raw sign.
    return @as(u32, @bitCast(delta)) != 0;
}

test "all sixteen nibble codes and signed scales reconstruct from independent FP64 coefficients" {
    const k = 64;
    var words: [8]u32 = @splat(0);
    for (0..k) |i| words[i / 8] |= @as(u32, @intCast(i % 16)) << @intCast((i % 8) * 4);
    const scales = [_]u16{0xbd56}; // negative BF16 scale
    const biases = [_]u16{0x3d91};
    var matrix = try convert(std.testing.allocator, 1, k, &words, &scales, &biases);
    defer matrix.deinit();
    for (0..k) |col| {
        const q: u32 = @intCast(col % 16);
        const scale = bf16ToF32(scales[0]);
        const bias = bf16ToF32(biases[0]);
        const exact64 = @as(f64, scale) * @as(f64, @floatFromInt(q)) + @as(f64, bias);
        const exact32: f32 = @floatCast(exact64);
        try std.testing.expectEqual(@as(u32, @bitCast(exact32)), @as(u32, @bitCast(@as(f32, matrix.high[col]) + correctionAt(matrix, 0, col))));
    }
}

test "double-round-sensitive pinned row 144 correction survives and omission changes BF16" {
    const k = 3392;
    const rows = 145;
    const words = try std.testing.allocator.alloc(u32, rows * (k / 8));
    defer std.testing.allocator.free(words);
    @memset(words, 0);
    // The audited coefficient is row 144, k=3357, q=1; all other packed codes are zero.
    const target_word = 144 * (k / 8) + 3357 / 8;
    words[target_word] |= @as(u32, 1) << @intCast((3357 % 8) * 4);
    const groups = rows * (k / 64);
    const scales = try std.testing.allocator.alloc(u16, groups);
    defer std.testing.allocator.free(scales);
    const biases = try std.testing.allocator.alloc(u16, groups);
    defer std.testing.allocator.free(biases);
    @memset(scales, 0);
    @memset(biases, 0);
    const group_index = 144 * (k / 64) + 3357 / 64;
    scales[group_index] = 0xbb57; // -0.0032806396484375
    biases[group_index] = 0x3d14; //  0.0361328125

    var matrix = try convert(std.testing.allocator, rows, k, words, scales, biases);
    defer matrix.deinit();
    const col: usize = 3357;
    const lo = matrix.offsets[144];
    const hi = matrix.offsets[145];
    var found = false;
    for (lo..hi) |i| if (matrix.columns[i] == @as(u16, @intCast(col))) {
        found = true;
        const exact: f32 = @as(f32, bf16ToF32(scales[group_index])) + bf16ToF32(biases[group_index]);
        try std.testing.expectEqual(@as(u32, @bitCast(exact)), @as(u32, @bitCast(@as(f32, matrix.high[144 * k + col]) + matrix.deltas[i])));
        try std.testing.expectEqual(@as(u16, 15623), @import("affine4_host.zig").bf16of(exact));
        try std.testing.expectEqual(@as(u16, 15622), @import("affine4_host.zig").bf16of(@as(f32, matrix.high[144 * k + col])));
    };
    try std.testing.expect(found);
}

test "signed zero coefficient preserves its raw FP32 sign with an explicit signed-zero residual" {
    const k = 64;
    const words: [8]u32 = @splat(0);
    const scales = [_]u16{0x8000};
    const biases = [_]u16{0x8000};
    var matrix = try convert(std.testing.allocator, 1, k, &words, &scales, &biases);
    defer matrix.deinit();
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(@as(f32, matrix.high[0]))));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(matrix.deltas[0])));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(@as(f32, matrix.high[0]) + correctionAt(matrix, 0, 0))));
    try std.testing.expect(@as(u32, @bitCast(@as(f32, matrix.high[0]) + 0.0)) != 0x80000000);
    try std.testing.expectEqual(@as(u32, k), matrix.offsets[1]);
}

fn correctionAt(matrix: Matrix, row: usize, col: usize) f32 {
    for (matrix.offsets[row]..matrix.offsets[row + 1]) |i| if (matrix.columns[i] == col) return matrix.deltas[i];
    return 0;
}

test "reject non-finite and non-FP32-exact coefficients for caller fallback" {
    const k = 64;
    const words: [8]u32 = @splat(0);
    const scales = [_]u16{0x7f80};
    const biases = [_]u16{0};
    try std.testing.expectError(error.NonFiniteInput, convert(std.testing.allocator, 1, k, &words, &scales, &biases));
    var one_code = words;
    one_code[0] = 1;
    const finite_scale = [_]u16{0x3f80};
    const tiny_bias = [_]u16{0x3080}; // 2^-30: exact in BF16, lost by FP32 1 + bias
    try std.testing.expectError(error.CoefficientNotFP32Exact, convert(std.testing.allocator, 1, k, &one_code, &finite_scale, &tiny_bias));

    const far_bias = [_]u16{0x0001};
    const largest_scale = [_]u16{0x7f7f};
    try std.testing.expectError(error.CoefficientExponentSpan, convert(std.testing.allocator, 1, k, &one_code, &largest_scale, &far_bias));
}

test "reject malformed shape, FP16 overflow, and non-finite FP32 coefficients" {
    const k = 64;
    const words: [8]u32 = @splat(0);
    const scale = [_]u16{0x3f80};
    const bias = [_]u16{0};
    try std.testing.expectError(error.BadShape, convert(std.testing.allocator, 1, 65, &words, &scale, &bias));

    var one_code = words;
    one_code[0] = 1;
    const too_large_for_half = [_]u16{0x4780}; // 65536
    try std.testing.expectError(error.HalfOverflow, convert(std.testing.allocator, 1, k, &one_code, &too_large_for_half, &bias));

    one_code[0] = 15;
    const largest_finite = [_]u16{0x7f7f};
    try std.testing.expectError(error.NonFiniteCoefficient, convert(std.testing.allocator, 1, k, &one_code, &largest_finite, &bias));
}
