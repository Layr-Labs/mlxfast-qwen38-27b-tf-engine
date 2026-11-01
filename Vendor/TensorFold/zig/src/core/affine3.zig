//! Three-bit drafter preparation, derived from the public48d clipped affine rule; target q4 preparation is unchanged.
const std=@import("std");
const affine4=@import("affine4.zig");
const Prepared=affine4.Prepared;
const asFloat=affine4.asFloat;
const asBf16=affine4.asBf16;
const Error=error{NonFiniteAffineInput,NonFiniteAffineRange};
const max_workers=16;
pub fn prepare(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize) !Prepared {
    const bits:u8=3;
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

fn run(out:*Error!void,weights:[]const u16,words:[]u32,scales:[]u16,biases:[]u16,first:usize,last:usize,levels:u32)void{
 out.*=groups(weights,words,scales,biases,first,last,levels);
}
fn groups(weights:[]const u16,words:[]u32,scales:[]u16,biases:[]u16,first:usize,last:usize,levels:u32)Error!void{
 for(first..last)|g|{
  const values=weights[g*64..][0..64];var lower:f32=std.math.inf(f32);var upper:f32=0;
  for(values)|bits|{const v=asFloat(bits);if(!std.math.isFinite(v))return error.NonFiniteAffineInput;lower=@min(lower,v);upper=@max(upper,v);}
  try clipped(values,words[g*8..][0..8],&scales[g],&biases[g],lower,upper,@floatFromInt(levels));
 }
}
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
/// Fingerprint the quantizer, packed-code mapping and BF16 helpers used by this prepared copy.
pub fn signature()[32]u8{
 var hash=std.crypto.hash.sha2.Sha256.init(.{});hash.update(@embedFile("affine3.zig"));hash.update(@embedFile("reg3.zig"));hash.update(@embedFile("affine4.zig"));return hash.finalResult();
}
