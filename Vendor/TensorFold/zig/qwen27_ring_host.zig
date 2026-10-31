//! BF16 address and actual production binding checks. No model or GPU execution.
const std = @import("std");
const mtl = @import("metal");
const shared = @import("src/core/shared_attention.zig");
const context = @import("src/families/qwen27/dflash/runtime_context.zig");
const runtime = @import("src/families/qwen27/dflash/runtime_backend.zig");
const op = @import("src/families/qwen27/dflash/operators.zig");
const a = std.testing.allocator;

fn pipeline(id: usize) mtl.Pipeline {
    return .{ .id = @ptrFromInt(id) };
}
fn attention(part: mtl.Buffer, ring: bool) shared.Attention {
    return .{ .shape = .{ .query_heads = 32, .kv_heads = 8, .head_dim = 128, .rotary_dim = 128, .capacity = 2064, .output = .bf16, .split_chunk = 64, .ring_split = ring }, .attend = pipeline(1), .decode = pipeline(2), .rope = pipeline(3), .append = pipeline(4), .split = pipeline(5), .merge = pipeline(6), .split_ring = if (ring) pipeline(7) else null, .part = part };
}
fn words(comptime n: usize, pass: anytype, i: usize) [n]u32 {
    return @bitCast(pass.values[i][0 .. n * 4].*);
}

test "qwen27 ring: direct BF16 sequences equal gather plus concat across ring and chunk boundaries" {
    const width = 64;
    const capacity = 2064;
    const ring = try a.alloc(u16, capacity * width);
    defer a.free(ring);
    const own = try a.alloc(u16, 16 * width);
    defer a.free(own);
    const merged = try a.alloc(u16, capacity * width);
    defer a.free(merged);
    for (ring, 0..) |*v, i| v.* = @truncate((i / width) *% 173 +% (i % width) *% 191 +% 0x3f80);
    for (own, 0..) |*v, i| v.* = @truncate((i / width) *% 97 +% (i % width) *% 157 +% 0xbe80);
    for ([_]usize{ 64, 2048 }) |window| {
        for ([_]usize{ 0, 1, window / 2, window - 1 }) |prefix| {
            for ([_]usize{ 0, 1, window - 2, window - 1 }) |begin| {
                for ([_]usize{ 1, 2, 16 }) |rows| {
                    const keys = prefix + rows;
                    // The old gather is two contiguous bit copies at most, followed by current rows.
                    const first = @min(prefix, window - begin);
                    @memcpy(merged[0 .. first * width], ring[begin * width ..][0 .. first * width]);
                    @memcpy(merged[first * width ..][0 .. (prefix - first) * width], ring[0 .. (prefix - first) * width]);
                    @memcpy(merged[prefix * width ..][0 .. rows * width], own[0 .. rows * width]);
                    const mask = try a.alloc(u8, rows * keys);
                    defer a.free(mask);
                    try context.fillMask(mask, @intCast(prefix), @intCast(rows), @intCast(window), 8);
                    for ([_]usize{ 64, 256 }) |chunk_size| for (0..rows) |query| {
                        // The same logical addresses hold for both supported chunk sizes.
                        var chunk: usize = 0;
                        while (chunk < keys) : (chunk += chunk_size) {
                            for (chunk..@min(keys, chunk + chunk_size)) |key| {
                                const visible = (key >= prefix or prefix + query < key + window) and !(query < 8 and key >= prefix + 8);
                                try std.testing.expectEqual(@as(u8, @intFromBool(visible)), mask[query * keys + key]);
                                const direct = if (key < prefix) ring[((begin + key) % window) * width ..][0..width] else own[(key - prefix) * width ..][0..width];
                                try std.testing.expectEqualSlices(u16, merged[key * width ..][0..width], direct);
                            }
                        }
                    };
                }
            }
        }
    }
}

test "qwen27 ring: actual runtime binds ring and current rows while fallback keeps copies" {
    for ([_]bool{ false, true }) |enabled| {
        for ([_]u32{ 0, 1000, 1023, 1065, 2047 }) |prefix| {
            for ([_]u32{ 0, 1, 2047 }) |begin| {
                for ([_]u32{ 1, 2, 16 }) |rows| {
                    for ([_]bool{ false, true }) |batch| {
                        for ([_]bool{ false, true }) |padded| {
                            var dispatches: mtl.DispatchRecord = .{};
                            var record: mtl.Record = .{ .gpa = a, .dispatches = &dispatches };
                            defer record.deinit();
                            const device = record.device();
                            const part = try device.buffer(16 * 32 * 33 * 130 * 4, 0);
                            defer part.deinit();
                            var caches = [_]runtime.Cache{.{ .keys = try device.buffer(2064 * 1024 * 2, 0), .values = try device.buffer(2064 * 1024 * 2, 0) }};
                            defer caches[0].keys.deinit();
                            defer caches[0].values.deinit();
                            var target: @import("src/families/qwen27/model.zig").Model = undefined;
                            target.glue = .{ .embed = pipeline(8), .head = pipeline(8), .slice = pipeline(8), .add = pipeline(8), .gather = pipeline(8), .take = pipeline(8) };
                            var backend: runtime.Backend = undefined;
                            backend.config = .{};
                            backend.target = &target;
                            backend.failed = false;
                            backend.transaction = null;
                            backend.attention = attention(part, enabled);
                            backend.caches = &caches;
                            backend.batch = &.{&caches};
                            backend.context = .{ .cache = 1, .begin = begin, .end = @as(u64, begin) + prefix };
                            backend.frame = .{ .allocator = a, .device = device, .queue = record.queue() };
                            defer backend.frame.deinit();
                            const epoch = try backend.frame.begin();
                            const query = try backend.frame.alloc(rows, 4096, .bf16);
                            const k_storage = try backend.frame.ref(try backend.frame.alloc(rows * 2 + 1, 1024, .bf16));
                            const v_storage = try backend.frame.ref(try backend.frame.alloc(rows * 2 + 1, 1024, .bf16));
                            const k = try backend.frame.add(.{ .buf = k_storage.buf, .off = 2048 }, rows, 1024, if (padded) 2048 else 1024, .bf16);
                            const v = try backend.frame.add(.{ .buf = v_storage.buf, .off = 2048 }, rows, 1024, if (padded) 2048 else 1024, .bf16);
                            const kr = try backend.frame.ref(k);
                            const vr = try backend.frame.ref(v);
                            const contexts = [_]op.Context{backend.context};
                            const out = try context.attention(&backend, epoch, .{ .context = backend.context, .config = backend.config, .layer = 0, .query = query, .keys = k, .values = v, .positions = &.{}, .streams = if (batch) &contexts else &.{}, .block = rows });
                            const contiguous = prefix + rows <= 2048 - begin;
                            const direct = enabled and !batch and !padded and prefix + rows > 1024 and !contiguous;
                            const prefix_pieces: usize = if (prefix == 0) 0 else if (prefix > 2048 - begin) 2 else 1;
                            const copies: usize = if (direct) 0 else if (contiguous) 2 else 2 + 2 * prefix_pieces;
                            try std.testing.expectEqual(copies + (if (prefix + rows > 1024) @as(usize, 2) else 1), dispatches.n);
                            const split = dispatches.passes[copies];
                            try std.testing.expectEqual(@as(u32, if (direct) 7 else if (prefix + rows > 1024) 5 else 1), split.pipe);
                            const args = words(5, split, 4);
                            try std.testing.expectEqualSlices(u32, &.{ rows, prefix + rows, prefix + rows, 0 }, args[0..4]);
                            if (direct) {
                                try std.testing.expectEqual(@intFromPtr(kr.buf.id), split.buffers[1].id);
                                try std.testing.expectEqual(@intFromPtr(vr.buf.id), split.buffers[2].id);
                                try std.testing.expectEqual(@as(usize, 2048), split.buffers[1].off);
                                try std.testing.expectEqual(@as(usize, 2048), split.buffers[2].off);
                                try std.testing.expectEqual(@intFromPtr(part.id), split.buffers[5].id);
                                try std.testing.expectEqual(@as(usize, 0), split.buffers[5].off);
                                try std.testing.expectEqual(@as(usize, 16 * 32 * 33 * 130 * 4), part.length());
                                try std.testing.expectEqual(@intFromPtr(caches[0].keys.id), split.buffers[6].id);
                                try std.testing.expectEqual(@intFromPtr(caches[0].values.id), split.buffers[7].id);
                                try std.testing.expectEqualSlices(u32, &.{ prefix, begin, 2048 }, &words(3, split, 8));
                                try std.testing.expectEqual(@as(usize, 8), split.grid.width);
                                try std.testing.expectEqual(@as(usize, rows), split.grid.height);
                                try std.testing.expectEqual((prefix + rows + 63) / 64, split.grid.depth);
                                try std.testing.expectEqual(@as(usize, 5), backend.frame.cursor); // Q, current K/V, output, mask only
                            } else {
                                try std.testing.expectEqual(@as(u32, 8), dispatches.passes[0].pipe);
                                try std.testing.expectEqual(if (contiguous) @as(usize, 5) else 7, backend.frame.cursor);
                                if (contiguous) {
                                    try std.testing.expectEqual(@intFromPtr(caches[0].keys.id), split.buffers[1].id);
                                    try std.testing.expectEqual(@intFromPtr(caches[0].values.id), split.buffers[2].id);
                                    try std.testing.expectEqual(@as(usize, begin) * 1024 * 2, split.buffers[1].off);
                                    try std.testing.expectEqual(@as(usize, begin) * 1024 * 2, split.buffers[2].off);
                                    for (dispatches.passes[0..2], [_]mtl.Buffer{ caches[0].keys, caches[0].values }) |copy, cache| {
                                        try std.testing.expectEqual(@intFromPtr(cache.id), copy.buffers[2].id);
                                        try std.testing.expectEqual(@as(usize, begin + prefix) * 1024 * 2, copy.buffers[2].off);
                                    }
                                } else {
                                    try std.testing.expect(split.buffers[1].id != @intFromPtr(caches[0].keys.id));
                                    try std.testing.expect(split.buffers[2].id != @intFromPtr(caches[0].values.id));
                                }
                            }
                            try std.testing.expectEqual(if (prefix + rows > 1024) @as(usize, 1) else 0, record.barriers);
                            try std.testing.expectEqual(mtl.DispatchType.serial, record.last_dispatch.?);
                            const output = try backend.frame.ref(out);
                            const last = dispatches.passes[dispatches.n - 1];
                            try std.testing.expectEqual(@intFromPtr(output.buf.id), last.buffers[if (prefix + rows > 1024) @as(usize, 2) else 5].id);
                            if (prefix + rows > 1024) {
                                try std.testing.expectEqual(@as(u32, 6), last.pipe);
                                try std.testing.expectEqual(@as(usize, 32), last.grid.width);
                                try std.testing.expectEqual(@as(usize, rows), last.grid.height);
                                try std.testing.expectEqual(@as(usize, 1), last.grid.depth);
                                try std.testing.expectEqual(@intFromPtr(part.id), last.buffers[0].id);
                            }
                            try backend.frame.finish();
                            backend.frame.release();
                            try record.check();
                        }
                    }
                }
            }
        }
    }
}

test "qwen27 ring: guards reject bad extents capacity and aliases before dispatch" {
    var dispatches: mtl.DispatchRecord = .{};
    var record: mtl.Record = .{ .gpa = a, .dispatches = &dispatches };
    defer record.deinit();
    const device = record.device();
    const buffer = try device.buffer(2064 * 8192, 0);
    defer buffer.deinit();
    const out = try device.buffer(2064 * 8192, 0);
    defer out.deinit();
    var p = attention(buffer, true);
    const e: mtl.ComputeEncoder = .{ .record = &dispatches };
    var ring = shared.Ring{ .k = .{ .buf = buffer }, .v = .{ .buf = buffer }, .prefix = 2047, .begin = 2047, .window = 2048 };
    const input = shared.Ref{ .buf = buffer };
    for (0..7) |case| {
        p.split_ring = pipeline(7);
        var bad = ring;
        switch (case) {
            0 => bad.window = 0,
            1 => bad.prefix = 2048,
            2 => bad.begin = 2048,
            3 => bad.prefix = 2046,
            4 => bad.window = 2065,
            5 => p.split_ring = null,
            6 => bad.prefix = 2064,
            else => unreachable,
        }
        try std.testing.expectError(error.InvalidAttentionView, p.encodeSplitRing(e, 16, 2063, 2063, 1, input, input, input, input, .{ .buf = out }, bad));
    }
    p.split_ring = pipeline(7);
    ring.k.off = buffer.length() - 2;
    try std.testing.expectError(error.AttentionBufferTooSmall, p.encodeSplitRing(e, 16, 2063, 2063, 1, input, input, input, input, .{ .buf = out }, ring));
    ring.k.off = 1;
    try std.testing.expectError(error.MisalignedAttentionBuffer, p.encodeSplitRing(e, 16, 2063, 2063, 1, input, input, input, input, .{ .buf = out }, ring));
    ring.k.off = 0;
    try std.testing.expectError(error.AliasedAttentionBuffer, p.encodeSplitRing(e, 16, 2063, 2063, 1, input, input, input, input, input, ring));
    const short = try device.buffer(16 * 1024 * 2 - 2, 0);
    defer short.deinit();
    try std.testing.expectError(error.AttentionBufferTooSmall, p.encodeSplitRing(e, 16, 2063, 2063, 1, input, .{ .buf = short }, input, input, .{ .buf = out }, ring));
    try std.testing.expectError(error.InvalidAttentionView, p.encodeSplitRing(e, 16, 2064, 2064, 1, input, input, input, input, .{ .buf = out }, ring));
    try std.testing.expectError(error.InvalidAttentionView, p.encodeSplitRing(e, 17, 2064, 2064, 1, input, input, input, input, .{ .buf = out }, ring));
    try std.testing.expectError(error.InvalidAttentionView, p.encodeSplitRing(e, 16, 2065, 2065, 1, input, input, input, input, .{ .buf = out }, ring));
    try std.testing.expectEqual(@as(usize, 0), dispatches.n);
    // Capacity 2064 is allocated; valid full context plus 16 rows uses 2063 keys.
    try p.encodeSplitRing(e, 16, 2063, 2063, 1, input, input, input, input, .{ .buf = out }, ring);
    try std.testing.expectEqual(@as(usize, 2), dispatches.n);
}

test "qwen27 ring: source enables the optional entry only when requested" {
    var shape = attention(undefined, false).shape;
    const control = try shared.source(a, shape);
    defer a.free(control);
    try std.testing.expect(std.mem.indexOf(u8, control, "#define TF_RING_SPLIT 0\n") != null);
    shape.ring_split = true;
    const direct = try shared.source(a, shape);
    defer a.free(direct);
    try std.testing.expect(std.mem.indexOf(u8, direct, "#define TF_RING_SPLIT 1\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, direct, "#define TF_CHUNK 64\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, direct, "tf_attention_split<true>") != null);
    try std.testing.expect(std.mem.indexOf(u8, direct, "tf_attention_split<false>") != null);
}
