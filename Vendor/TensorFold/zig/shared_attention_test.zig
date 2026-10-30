//! Actual attention source, allocation and dispatch paths with a recording Metal boundary.
const std = @import("std");
const attention = @import("src/core/shared_attention.zig");
const mtl = @import("metal");
const a = std.testing.allocator;
const shape = attention.Shape{ .query_heads = 32, .kv_heads = 8, .head_dim = 128, .rotary_dim = 128, .capacity = 2064, .output = .bf16 };

test "shared attention: default and draft chunk source agree with actual scratch allocation" {
    for ([_]usize{ 256, 64 }, [_]usize{ 2396160, 8785920 }) |chunk, bytes| {
        var s = shape;
        if (chunk == 64) s.split_chunk = chunk;
        const text = try attention.source(a, s);
        defer a.free(text);
        const define = if (chunk == 64) "#define TF_CHUNK 64\n" else "#define TF_CHUNK 256\n";
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, "#define TF_CHUNK "));
        try std.testing.expect(std.mem.indexOf(u8, text, define) != null);
        try std.testing.expect(std.mem.indexOf(u8, text, "#define TF_TILE 4\n") != null);
        var allocated: usize = 0;
        const p = try attention.Attention.init(a, .{ .bytes = &allocated }, s);
        defer p.deinit();
        try std.testing.expectEqual(bytes, allocated);
        try std.testing.expectEqual(bytes, p.part.length());
        try std.testing.expectEqual(chunk, p.shape.split_chunk);
    }
}

test "shared attention: split dispatch and scratch cover chunk boundaries and capacity" {
    const keys = [_]usize{ 64, 65, 256, 257, 1024, 1025, 1067, 2048, 2049, 2064 };
    const chunks256 = [_]usize{ 1, 1, 1, 2, 4, 5, 5, 8, 9, 9 };
    const chunks64 = [_]usize{ 1, 2, 4, 5, 16, 17, 17, 32, 33, 33 };
    for ([_]usize{ 256, 64 }) |chunk| {
        var s = shape;
        s.split_chunk = chunk;
        var allocated: usize = 0;
        const p = try attention.Attention.init(a, .{ .bytes = &allocated }, s);
        defer p.deinit();
        const q = attention.Ref{ .buf = .{ .id = 1, .bytes = 16 * 32 * 128 * 2 } };
        const k = attention.Ref{ .buf = .{ .id = 2, .bytes = 2064 * 8 * 128 * 2 } };
        const v = attention.Ref{ .buf = .{ .id = 3, .bytes = 2064 * 8 * 128 * 2 } };
        const mask = attention.Ref{ .buf = .{ .id = 4, .bytes = 16 * 2064 } };
        const out = attention.Ref{ .buf = .{ .id = 5, .bytes = 16 * 32 * 128 * 2 } };
        for (keys, if (chunk == 64) chunks64 else chunks256) |n, chunks| for ([_]usize{ 2, 16 }) |rows| {
            var record: mtl.Record = .{};
            const e = mtl.ComputeEncoder{ .r = &record };
            try p.encodeSplit(e, rows, n, n, 0.125, q, k, v, mask, out);
            try std.testing.expectEqual(@as(usize, 2), record.n);
            try std.testing.expectEqual(@as(u32, 1), record.passes[0].pipe);
            try std.testing.expectEqual(mtl.Size.of(8, rows, chunks), record.passes[0].grid);
            try std.testing.expectEqual(mtl.Size.of(32, 1, 1), record.passes[0].threads);
            try std.testing.expectEqual(@as(u32, 2), record.passes[1].pipe);
            try std.testing.expectEqual(mtl.Size.of(32, rows, 1), record.passes[1].grid);
            try std.testing.expectEqual(p.part.id, record.passes[0].buffers[5].id);
            try std.testing.expectEqual(p.part.id, record.passes[1].buffers[0].id);
            try std.testing.expectEqualSlices(u8, record.passes[0].values[4][0..20], record.passes[1].values[1][0..20]);
            try std.testing.expect(rows * 32 * chunks * 130 * 4 <= allocated);
        };
        var record: mtl.Record = .{};
        const e = mtl.ComputeEncoder{ .r = &record };
        try std.testing.expectError(error.InvalidAttentionView, p.encodeSplit(e, 17, 2064, 2064, 0.125, q, k, v, mask, out));
        try std.testing.expectError(error.InvalidAttentionView, p.encodeSplit(e, 2, 2065, 2065, 0.125, q, k, v, mask, out));
        try std.testing.expectEqual(@as(usize, 0), record.n);
    }
}

test "shared attention: invalid chunks fail before allocation" {
    for ([_]usize{ 0, 1, 32, 128, 255, 257, std.math.maxInt(usize) }) |chunk| {
        var s = shape;
        s.split_chunk = chunk;
        var allocated: usize = 0;
        try std.testing.expectError(error.InvalidAttentionShape, attention.Attention.init(a, .{ .bytes = &allocated }, s));
        try std.testing.expectEqual(@as(usize, 0), allocated);
    }
}
