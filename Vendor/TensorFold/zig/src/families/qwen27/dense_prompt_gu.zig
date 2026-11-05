//! Prompt-only derived FP16 high coefficients and exact sparse FP32 residuals.
const std = @import("std");
const mtl = @import("metal");
const projection = @import("projection.zig");
const types = @import("dense_prompt_gu_types.zig");
const source = @import("qwen_runtime_sources");

pub const Weights = types.Weights;
const n = 34816;
const k = 5120;
const groups = k / 64;
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn bf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// One original unsigned code and original (unfolded) metadata pair.
pub fn coefficient(code: u4, scale: u16, bias: u16) !struct { high: f16, residual: f32 } {
    const exact = @mulAdd(f32, bf16(scale), @as(f32, @floatFromInt(code)), bf16(bias));
    if (!std.math.isFinite(exact)) return error.BadDenseCoefficient;
    const high: f16 = @floatCast(exact);
    const expanded: f32 = @floatCast(high);
    if (!std.math.isFinite(expanded)) return error.BadDenseCoefficient;
    const residual = exact - expanded;
    const reconstructed = expanded + residual;
    if (@as(u32, @bitCast(reconstructed)) != @as(u32, @bitCast(exact))) return error.BadDenseCoefficient;
    return .{ .high = high, .residual = residual };
}

/// Shared CPU writer; input words and metadata are one original checkpoint column.
fn packColumn(a: std.mem.Allocator, words: []const u32, scales: []const u16, biases: []const u16, high: []f16, indices: *std.ArrayList(u16), residual: *std.ArrayList(f32)) !void {
    if (words.len != k / 8 or scales.len != groups or biases.len != groups or high.len != k) return error.BadDenseShape;
    for (0..k) |j| {
        const code: u4 = @truncate(words[j / 8] >> @as(u5, @intCast((j % 8) * 4)));
        const value = try coefficient(code, scales[j / 64], biases[j / 64]);
        high[j] = value.high;
        if (value.residual != 0) {
            try indices.append(a, @intCast(j));
            try residual.append(a, value.residual);
        }
    }
}

/// ORIGINAL checkpoint triples, gate then up. Nothing from the regpacked path is read.
/// The caller takes ownership of all four returned buffers together.
pub fn prepare(a: std.mem.Allocator, device: mtl.Device, parts: []const projection.Triple, layer: u32) !Weights {
    if (parts.len != 2) return error.BadDenseShape;
    for (parts) |p| {
        if (p.bits != 4 or p.group != 64 or p.weight.dtype != .u32 or p.weight.rank != 2 or p.weight.shape[0] != n / 2 or p.weight.shape[1] != k / 8 or p.weight.bytes != n / 2 * k / 8 * 4) return error.BadDenseShape;
        if (p.scales.dtype != .bf16 or p.biases.dtype != .bf16 or p.scales.rank != 2 or p.biases.rank != 2 or p.scales.shape[0] != n / 2 or p.biases.shape[0] != n / 2 or p.scales.shape[1] != groups or p.biases.shape[1] != groups or p.scales.bytes != n / 2 * groups * 2 or p.biases.bytes != n / 2 * groups * 2) return error.BadDenseShape;
    }
    const high = try device.buffer(n * k * 2, options);
    errdefer high.deinit();
    const offsets = try device.buffer((n + 1) * 4, options);
    errdefer offsets.deinit();
    const dst = high.slice(f16, n * k);
    const off = offsets.slice(u32, n + 1);
    var indices: std.ArrayList(u16) = .empty;
    defer indices.deinit(a);
    var residual: std.ArrayList(f32) = .empty;
    defer residual.deinit(a);
    var column: usize = 0;
    for (parts) |p| {
        const words = p.weight.host(u32);
        const scales = p.scales.host(u16);
        const biases = p.biases.host(u16);
        for (0..n / 2) |c| {
            off[column] = @intCast(indices.items.len);
            try packColumn(a, words[c * (k / 8) ..][0 .. k / 8], scales[c * groups ..][0..groups], biases[c * groups ..][0..groups], dst[column * k ..][0..k], &indices, &residual);
            column += 1;
        }
    }
    off[n] = @intCast(indices.items.len);
    const ix = try device.buffer(@max(1, indices.items.len) * 2, options);
    errdefer ix.deinit();
    const delta = try device.buffer(@max(1, residual.items.len) * 4, options);
    errdefer delta.deinit();
    @memset(ix.contents()[0..ix.length()], 0);
    @memset(delta.contents()[0..delta.length()], 0);
    @memcpy(ix.slice(u16, indices.items.len), indices.items);
    @memcpy(delta.slice(f32, residual.items.len), residual.items);
    return .{ .high = high, .offsets = offsets, .indices = ix, .residual = delta, .count = @intCast(indices.items.len), .n = n, .k = k, .layer = layer };
}

pub fn deinitWeights(w: Weights) void {
    w.high.deinit();
    w.offsets.deinit();
    w.indices.deinit();
    w.residual.deinit();
}

pub fn bytes(w: Weights) usize {
    return w.high.length() + w.offsets.length() + w.indices.length() + w.residual.length();
}

pub const Program = struct {
    pipe: mtl.Pipeline,

    pub fn init(a: std.mem.Allocator, device: mtl.Device) !Program {
        _ = a;
        const lib = try mtl.Library.fromSource(device, source.dense_prompt_gu, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "q27_dense_prompt_gu", false);
        errdefer pipe.deinit();
        if (pipe.simdWidth() != 32 or pipe.maxThreads() < 256) return error.UnsupportedDensePipeline;
        return .{ .pipe = pipe };
    }

    pub fn deinit(p: Program) void {
        p.pipe.deinit();
    }

    /// The caller orders producers and consumers, and keeps all buffers alive until GPU completion.
    pub fn encode(p: Program, e: mtl.ComputeEncoder, w: Weights, x: projection.Ref, y: projection.Ref, rows: u32) !void {
        if (rows == 0 or rows % 128 != 0 or w.n != n or w.k != k or w.count > n * k) return error.BadDenseShape;
        if (x.offset % 16 != 0 or y.offset % 16 != 0) return error.BadDenseAlignment;
        const xb = try std.math.mul(usize, rows, k * 2);
        const yb = try std.math.mul(usize, rows, n * 2);
        if (x.offset > x.buffer.length() or xb > x.buffer.length() - x.offset or y.offset > y.buffer.length() or yb > y.buffer.length() - y.offset) return error.BadDenseStorage;
        if (w.high.length() < n * k * 2 or w.offsets.length() < (n + 1) * 4 or w.indices.length() < @as(usize, w.count) * 2 or w.residual.length() < @as(usize, w.count) * 4) return error.BadDenseStorage;
        e.setPipeline(p.pipe);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(w.high, 0, 1);
        e.setBuffer(w.offsets, 0, 2);
        e.setBuffer(w.indices, 0, 3);
        e.setBuffer(w.residual, 0, 4);
        e.setBuffer(y.buffer, y.offset, 5);
        e.setValue(rows, 6);
        e.dispatchGroups(.{ .width = n / 64, .height = rows / 128, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
    }
};

test "exact coefficient residual reconstructs an FP16 exception" {
    // Original BF16 scale and bias whose unsigned code requires more than 11 significand bits.
    const scale: u16 = 0x3b6d;
    const bias: u16 = 0xbd14;
    const pair = try coefficient(1, scale, bias);
    const exact = @mulAdd(f32, bf16(scale), 1, bf16(bias));
    const high: f32 = @floatCast(pair.high);
    try std.testing.expect(pair.residual != 0);
    try std.testing.expectEqual(@as(u32, @bitCast(exact)), @as(u32, @bitCast(high + pair.residual)));
    try std.testing.expect(@as(u32, @bitCast(high)) != @as(u32, @bitCast(exact)));
}

test "dense coefficient rejects nonfinite and FP16 overflow" {
    try std.testing.expectError(error.BadDenseCoefficient, coefficient(1, 0x7f80, 0));
    try std.testing.expectError(error.BadDenseCoefficient, coefficient(15, 0x4780, 0));
}

test "original checkpoint column packing reconstructs every nibble and sorted CSR residual" {
    const a = std.testing.allocator;
    var words: [k / 8]u32 = undefined;
    var scales: [groups]u16 = undefined;
    var biases: [groups]u16 = undefined;
    var high: [k]f16 = undefined;
    for (&words, 0..) |*word, wi| {
        word.* = 0;
        for (0..8) |j| word.* |= @as(u32, @intCast((wi * 3 + j * 5) % 16)) << @as(u5, @intCast(j * 4));
    }
    for (&scales, &biases, 0..) |*scale, *bias, i| {
        scale.* = if (i % 2 == 0) 0x3b6d else 0x3c81;
        bias.* = if (i % 3 == 0) 0xbd14 else 0xbb80;
    }
    var indices: std.ArrayList(u16) = .empty;
    defer indices.deinit(a);
    var residual: std.ArrayList(f32) = .empty;
    defer residual.deinit(a);
    try packColumn(a, &words, &scales, &biases, &high, &indices, &residual);
    try std.testing.expect(indices.items.len > 0);
    try std.testing.expectEqual(indices.items.len, residual.items.len);
    var next: usize = 0;
    var omission_caught = false;
    for (0..k) |j| {
        const code = (j / 8 * 3 + j % 8 * 5) % 16;
        // FP64 arithmetic is an independent oracle for these exactly representable coefficients.
        const oracle: f32 = @floatCast(@as(f64, bf16(scales[j / 64])) * @as(f64, @floatFromInt(code)) + @as(f64, bf16(biases[j / 64])));
        const expanded: f32 = @floatCast(high[j]);
        var reconstructed = expanded;
        if (next < indices.items.len and indices.items[next] == j) {
            if (next > 0) try std.testing.expect(indices.items[next - 1] < indices.items[next]);
            reconstructed += residual.items[next];
            omission_caught = omission_caught or @as(u32, @bitCast(expanded)) != @as(u32, @bitCast(oracle));
            next += 1;
        }
        try std.testing.expectEqual(@as(u32, @bitCast(oracle)), @as(u32, @bitCast(reconstructed)));
    }
    try std.testing.expectEqual(indices.items.len, next);
    try std.testing.expect(omission_caught);
}
