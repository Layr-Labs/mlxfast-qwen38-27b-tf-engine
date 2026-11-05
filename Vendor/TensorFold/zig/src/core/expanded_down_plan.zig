//! Row-chunk selection and byte offsets shared by the exact-half DOWN fast path and its host tests.
const std = @import("std");

pub const chunk_rows: u32 = 128;

pub const Plan = struct {
    native_chunks: u32,
    tail_rows: u32,

    pub fn nativeRows(selected: Plan) u32 {
        return selected.native_chunks * chunk_rows;
    }
};

pub fn plan(rows: u32) Plan {
    return .{ .native_chunks = rows / chunk_rows, .tail_rows = rows % chunk_rows };
}

pub fn rowByteOffset(base: usize, row: u32, width: u32) usize {
    return base + @as(usize, row) * width * @sizeOf(u16);
}

pub fn sumsByteOffset(base: usize, row: u32) usize {
    return base + @as(usize, row) * @sizeOf(f32);
}

pub fn rangeFits(length: usize, offset: usize, bytes: usize) bool {
    return offset <= length and bytes <= length - offset;
}

pub fn recordNativeEncode(counter: *std.atomic.Value(u64)) void {
    _ = counter.fetchAdd(1, .monotonic);
}

test "ranked prompt segment lengths select native 128-row chunks and exact Q4 tails" {
    const cases = [_]struct { rows: u32, chunks: u32, tail: u32 }{
        .{ .rows = 320, .chunks = 2, .tail = 64 },
        .{ .rows = 384, .chunks = 3, .tail = 0 },
        .{ .rows = 361, .chunks = 2, .tail = 105 },
        .{ .rows = 383, .chunks = 2, .tail = 127 },
        .{ .rows = 1065, .chunks = 8, .tail = 41 },
    };
    var recorded: std.atomic.Value(u64) = .init(0);
    for (cases) |case| {
        const selected = plan(case.rows);
        try std.testing.expectEqual(case.chunks, selected.native_chunks);
        try std.testing.expectEqual(case.tail, selected.tail_rows);
        try std.testing.expectEqual(case.rows, selected.nativeRows() + selected.tail_rows);
        if (case.chunks > 0) try std.testing.expect(selected.nativeRows() >= chunk_rows);
        for (0..selected.native_chunks) |_| recordNativeEncode(&recorded);
        const first_tail = selected.nativeRows();
        try std.testing.expectEqual(@as(usize, first_tail) * 17408 * 2, rowByteOffset(0, first_tail, 17408));
        try std.testing.expectEqual(@as(usize, first_tail) * 5120 * 2, rowByteOffset(0, first_tail, 5120));
        try std.testing.expectEqual(@as(usize, first_tail) * 4, sumsByteOffset(0, first_tail));
    }
    try std.testing.expectEqual(@as(u64, 17), recorded.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), plan(127).native_chunks);
    try std.testing.expectEqual(@as(u32, 1), plan(128).native_chunks);
}
