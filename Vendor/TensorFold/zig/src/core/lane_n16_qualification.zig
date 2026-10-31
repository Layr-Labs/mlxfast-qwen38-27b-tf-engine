//! Device-local startup qualification for the smaller register projection operand layout and arithmetic.
//! Synthetic buffers only; the caller caches the answer once per device and falls back when false.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const patterns = 8;
const rows = 16;
const width = 256;
const result_words = 16;

pub fn qualify(device: mtl.Device, _: std.mem.Allocator) !bool {
    if (!device.tensorUnits()) return false;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const library = mtl.Library.fromSource(device, sources.core_lane_n16_qualification, mtl.CompileOptions.mlx()) catch |err| {
        std.log.info("qwen27: N16 startup qualification unavailable ({t}); keeping N32", .{err});
        return false;
    };
    defer library.deinit();
    const pipeline = mtl.Pipeline.init(device, library, "tf_lane_n16_qualify", false) catch |err| {
        std.log.info("qwen27: N16 startup qualification pipeline unavailable ({t}); keeping N32", .{err});
        return false;
    };
    defer pipeline.deinit();
    if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < 32) return false;
    const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const input = try device.buffer(patterns * rows * width * 2, options);
    defer input.deinit();
    const output = try device.buffer(patterns * 32 * result_words * @sizeOf(u32), options);
    defer output.deinit();
    const words = input.slice(u16, patterns * rows * width);
    for (words, 0..) |*word, i| {
        const pattern = i / (rows * width);
        const index = i % (rows * width);
        const hash: u32 = @as(u32, @intCast(index)) *% 1664525 +% 1013904223;
        const sign: u16 = @intCast((hash >> 16) & 0x8000);
        const mantissa: u16 = @intCast((hash >> 9) & 127);
        word.* = switch (pattern) {
            0 => 0x3f00 | mantissa,
            1 => sign | (@as(u16, @intCast(120 + index % 14)) << 7) | mantissa,
            2 => (if (index % 2 == 0) @as(u16, 0) else 0x8000) | 0x3f80,
            3 => sign | @as(u16, @intCast(1 + index % 127)),
            4 => sign | (if (index % 3 == 0) @as(u16, 0x5a00) else 0x2500) | mantissa,
            5 => if (index % 2 == 0) 0 else sign | 0x4000 | mantissa,
            6 => sign | (@as(u16, @intCast(96 + index % 65)) << 7) | mantissa,
            else => sign | (@as(u16, @intCast(110 + index % 34)) << 7) | mantissa,
        };
    }
    @memset(output.slice(u32, patterns * 32 * result_words), std.math.maxInt(u32));
    const queue = try device.queue();
    defer queue.deinit();
    const cb = queue.commandBuffer();
    const encoder = cb.compute(.serial);
    encoder.setPipeline(pipeline);
    encoder.setBuffer(input, 0, 0);
    encoder.setBuffer(output, 0, 1);
    encoder.dispatchGroups(.{ .width = patterns }, .{ .width = 32 });
    encoder.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.log.info("qwen27: N16 startup qualification GPU failure ({s}); keeping N32", .{message});
        return false;
    }
    const results = output.slice(u32, patterns * 32 * result_words);
    var failures: u32 = 0;
    var first_failure: usize = 0;
    for (0..patterns * 32) |i| {
        const flags = results[i * result_words];
        if (failures == 0 and flags != 0) first_failure = i * result_words;
        failures |= flags;
    }
    const diagnostics = results[first_failure..][0..result_words];
    std.log.info("qwen27: N16 capacities Bsmall={d} Bwide={d} Psmall={d} Pwide={d}; first kind={d} half={d} index={d} wide_index={d} small=({d},{d}) wide=({d},{d}) rows={d} bits=0x{x}/0x{x}", .{ diagnostics[1], diagnostics[2], diagnostics[3], diagnostics[4], diagnostics[5], diagnostics[6], diagnostics[7], diagnostics[8], @as(i32, @bitCast(diagnostics[9])), @as(i32, @bitCast(diagnostics[10])), @as(i32, @bitCast(diagnostics[11])), @as(i32, @bitCast(diagnostics[12])), diagnostics[13], diagnostics[14], diagnostics[15] });
    if (failures != 0) {
        std.log.info("qwen27: N16 startup qualification differs (flags=0x{x}); keeping N32", .{failures});
        return false;
    }
    std.log.info("qwen27: N16 startup qualification passed layout and synthetic FP32 products", .{});
    return true;
}
