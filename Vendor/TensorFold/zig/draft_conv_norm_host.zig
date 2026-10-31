//! Real primitive admission and dispatch, plus the actual five-layer host schedule.
const std = @import("std");
const mtl = @import("metal");
const draft = @import("src/core/draft_ops.zig");
fn params(rows: u32) draft.ConvNorm {
    return .{ .conv = .{ .rows = rows, .width = 5120, .branch = 1, .residual = 1, .block = 2, .x_stride = 5128, .dynamic_stride = 1296, .y_stride = 5136 }, .norm = .{ .rows = rows, .heads = 1, .dim = 5120, .x_stride = 5136, .y_stride = 5144 } };
}
fn ref(id: usize, rows: u32, stride: u32, width: u32) draft.Ref {
    return .{ .buf = .{ .id = id, .bytes = ((@as(usize, rows) - 1) * stride + width) * 2 + id * 16 }, .off = id * 16 };
}
test "drafter conv-norm: original argument blocks and strided BF16 buffers feed one 1024-thread row group" {
    const ops = draft.Ops{ .pipes = undefined, .conv_norm = .{ .id = 5 } };
    for ([_]u32{ 1, 2, 8, 16, 128 }) |rows| {
        const p = params(rows);
        const refs = [_]draft.Ref{ ref(1, rows, p.conv.x_stride, 5120), ref(2, rows, p.conv.dynamic_stride, 1280), ref(3, 1, 20480, 20480), ref(4, rows, p.conv.x_stride, 5120), ref(5, 1, 5120, 5120), ref(6, rows, p.conv.y_stride, 5120), ref(7, rows, p.norm.y_stride, 5120) };
        var record: mtl.Record = .{};
        try ops.convNorm(.{ .r = &record }, refs[0], refs[1], refs[2], refs[3], refs[4], refs[5], refs[6], p);
        try std.testing.expectEqual(@as(usize, 1), record.n);
        const pass = record.passes[0];
        try std.testing.expectEqual(@as(u32, 5), pass.pipe);
        try std.testing.expectEqual(mtl.Size.of(1, rows, 1), pass.grid);
        try std.testing.expectEqual(mtl.Size.of(1024, 1, 1), pass.threads);
        try std.testing.expect(!pass.direct);
        for (refs, [_]usize{ 0, 1, 2, 3, 4, 7, 8 }) |r, slot| {
            try std.testing.expectEqual(r.buf.id, pass.buffers[slot].id);
            try std.testing.expectEqual(r.off, pass.buffers[slot].off);
        }
        try std.testing.expectEqual(@sizeOf(draft.Conv), pass.lengths[5]);
        try std.testing.expectEqual(@sizeOf(draft.Norm), pass.lengths[6]);
        try std.testing.expectEqualDeep(p.conv, std.mem.bytesToValue(draft.Conv, pass.values[5][0..@sizeOf(draft.Conv)]));
        try std.testing.expectEqualDeep(p.norm, std.mem.bytesToValue(draft.Norm, pass.values[6][0..@sizeOf(draft.Norm)]));
    }
}
test "drafter conv-norm: unsupported shape, rounding, pipeline and aliases fail before dispatch" {
    const r = [_]draft.Ref{ ref(1, 2, 5128, 5120), ref(2, 2, 1296, 1280), ref(3, 1, 20480, 20480), ref(4, 2, 5128, 5120), ref(5, 1, 5120, 5120), ref(6, 2, 5136, 5120), ref(7, 2, 5144, 5120) };
    for (0..8) |case| {
        var p = params(2);
        var ops = draft.Ops{ .pipes = undefined, .conv_norm = .{ .id = 5 } };
        var raw = r[5];
        const want: anyerror = switch (case) {
            0 => blk: {
                p.conv.branch = 0;
                break :blk error.BadDraftShape;
            },
            1 => blk: {
                p.conv.residual = 0;
                break :blk error.BadDraftShape;
            },
            2 => blk: {
                p.norm.dim = 2560;
                break :blk error.BadDraftShape;
            },
            3 => blk: {
                p.norm.x_stride += 1;
                break :blk error.BadDraftShape;
            },
            4 => blk: {
                p.norm.round_normalized = 0;
                break :blk error.BadDraftShape;
            },
            5 => blk: {
                raw = r[3];
                raw.buf.bytes += 32;
                break :blk error.DraftAlias;
            },
            6 => blk: {
                raw.off += 1;
                break :blk error.BadDraftBuffer;
            },
            7 => blk: {
                ops.conv_norm = null;
                break :blk error.DraftConvNormUnsupported;
            },
            else => unreachable,
        };
        var record: mtl.Record = .{};
        try std.testing.expectError(want, ops.convNorm(.{ .r = &record }, r[0], r[1], r[2], r[3], r[4], raw, r[6], p));
        try std.testing.expectEqual(@as(usize, 0), record.n);
    }
}
test {
    _ = @import("src/families/qwen27/dflash/execution_test.zig");
}
