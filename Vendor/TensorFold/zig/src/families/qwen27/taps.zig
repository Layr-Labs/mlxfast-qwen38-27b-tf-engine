//! Tap planes store native post-block residuals until accepted rows and their successor ids are known.
const std = @import("std");
const mtl = @import("metal");
const Ref = @import("projection.zig").Ref;
const Glue = @import("glue_gpu.zig").Kernels;

pub const layers = [_]usize{ 5, 19, 33, 47, 61 };

pub const Taps = struct {
    buffer: mtl.Buffer,
    capacity: u32,
    width: u32,
    seen: u8 = 0,
    ids: [5]usize = layers,
    rows: u32 = 0,
    /// A prompt pass's taps go here instead of the planes (Generation.promptChunk): row r of plane j into row r of the
    /// drafter's ring at column j * width, where the tap gather would copy it.
    into: ?Into = null,

    pub const Into = struct { buffer: mtl.Buffer, offset: usize, stride: usize };

    pub fn init(device: mtl.Device, capacity: u32, width: u32) !Taps {
        if (capacity == 0 or capacity > @import("gpu_frame.zig").max_rows or width == 0) return error.BadTapShape;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, capacity, width), 2 * layers.len);
        return .{ .buffer = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked), .capacity = capacity, .width = width };
    }

    pub fn deinit(t: Taps) void {
        t.buffer.deinit();
    }

    pub fn begin(t: *Taps) void {
        t.seen = 0;
        t.rows = 0;
    }

    pub fn record(t: *Taps, glue: Glue, e: mtl.ComputeEncoder, layer: usize, hidden: Ref, rows: u32) !void {
        if (rows == 0 or rows > t.capacity) return error.BadTapShape;
        const index = std.mem.indexOfScalar(usize, &t.ids, layer) orelse return;
        if (t.seen != 0 and rows != t.rows) return error.BadTapShape;
        const bit = @as(u8, 1) << @intCast(index);
        if (t.seen & bit != 0) return error.DuplicateTap;
        // unfenced: the copy runs beside the mixer's projections, whose barrier orders it before any reader of the plane
        // and before the next write of `hidden` (two norms on)
        try t.copy(glue, e, index, hidden, rows, 0);
        t.seen |= bit;
        t.rows = rows;
    }

    /// `record` for one segment of a staggered prompt pass (decode_round.staggered): its `rows` go at row `first` of
    /// each plane, the pass's `total` rows in all. The segment ending at `total` marks the plane recorded.
    pub fn recordAt(t: *Taps, glue: Glue, e: mtl.ComputeEncoder, layer: usize, hidden: Ref, rows: u32, first: u32, total: u32) !void {
        if (rows == 0 or total > t.capacity or rows > total - @min(first, total) or first >= total) return error.BadTapShape;
        const index = std.mem.indexOfScalar(usize, &t.ids, layer) orelse return;
        // unfenced, as record's: the copy runs beside the mixer's projections in the segment's own lane
        try t.copy(glue, e, index, hidden, rows, first);
        if (first + rows == total) t.seen |= @as(u8, 1) << @intCast(index);
        t.rows = total;
    }

    /// Rows [first, first + rows) of plane `index`: `hidden`'s rows into the planes, or into the ring rows `into` names.
    fn copy(t: *const Taps, glue: Glue, e: mtl.ComputeEncoder, index: usize, hidden: Ref, rows: u32, first: u32) !void {
        const p = @import("glue_gpu.zig").Slice{ .rows = rows, .width = t.width, .stride = t.width, .offset = 0 };
        if (t.into) |to| {
            const row_bytes = @as(usize, t.width) * 2;
            const at = to.offset + @as(usize, first) * to.stride + index * row_bytes;
            if (to.stride % 2 != 0 or to.stride < row_bytes * layers.len or at + (@as(usize, rows) - 1) * to.stride + row_bytes > to.buffer.length()) return error.BadTapShape;
            return glue.unstackRows(e, hidden, .{ .buffer = to.buffer, .offset = at }, p, @intCast(to.stride / 2));
        }
        var at = try t.plane(index);
        at.offset += @as(usize, first) * t.width * 2;
        try glue.unstack(e, hidden, at, p);
    }

    pub fn plane(t: Taps, index: usize) !Ref {
        if (index >= layers.len) return error.BadTapIndex;
        return .{ .buffer = t.buffer, .offset = index * @as(usize, t.capacity) * t.width * 2 };
    }

    pub fn complete(t: Taps) !void {
        if (t.seen != (1 << layers.len) - 1) return error.IncompleteTaps;
    }

    pub fn accepted(t: Taps, glue: Glue, e: mtl.ComputeEncoder, path: []const u32, kept: Ref, out: Ref) !void {
        try t.complete();
        if (path.len == 0) return;
        if (path.len > t.rows) return error.BadTapGather;
        for (path) |row| if (row >= t.rows) return error.BadTapGather;
        try glue.tapRows(e, .{ .buffer = t.buffer }, kept, out, .{ .rows = @intCast(path.len), .width = t.width, .capacity = t.capacity, .planes = layers.len });
        e.barrier();
    }
};

test "DFlash tap ids name completed target blocks" {
    try std.testing.expectEqualSlices(usize, &.{ 5, 19, 33, 47, 61 }, &layers);
}
