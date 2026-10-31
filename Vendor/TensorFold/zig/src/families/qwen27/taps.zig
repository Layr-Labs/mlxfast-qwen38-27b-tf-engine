//! Tap planes store native post-block residuals until accepted rows and their successor ids are known.
const std = @import("std");
const mtl = @import("metal");
const Ref = @import("projection.zig").Ref;
const Glue = @import("glue_gpu.zig").Kernels;

pub const layers = [_]usize{ 5, 19, 33, 47, 61 };
/// Parts a pass's rows come in at most (a split prompt chunk's segments, core/stagger.zig's MAX).
pub const max_parts = 4;

pub const Taps = struct {
    buffer: mtl.Buffer,
    capacity: u32,
    width: u32,
    /// Each part's recorded taps, a bit a tap.
    seen: [max_parts]u8 = @splat(0),
    parts: u8 = 1,
    ids: [5]usize = layers,
    rows: u32 = 0,

    pub fn init(device: mtl.Device, capacity: u32, width: u32) !Taps {
        if (capacity == 0 or capacity > @import("gpu_frame.zig").max_rows or width == 0) return error.BadTapShape;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, capacity, width), 2 * layers.len);
        return .{ .buffer = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked), .capacity = capacity, .width = width };
    }

    pub fn deinit(t: Taps) void {
        t.buffer.deinit();
    }

    /// A pass of `rows` rows in `parts` parts, each recording its own rows of every tap.
    pub fn begin(t: *Taps, rows: u32, parts: u8) !void {
        if (rows == 0 or rows > t.capacity or parts == 0 or parts > max_parts) return error.BadTapShape;
        t.seen = @splat(0);
        t.parts = parts;
        t.rows = rows;
    }

    /// Rows [first, first + rows) of the pass, part `part`'s, from `hidden` (their first row).
    pub fn record(t: *Taps, glue: Glue, e: mtl.ComputeEncoder, layer: usize, hidden: Ref, first: u32, rows: u32, part: u8) !void {
        if (rows == 0 or part >= t.parts or first > t.rows or rows > t.rows - first) return error.BadTapShape;
        const index = std.mem.indexOfScalar(usize, &t.ids, layer) orelse return;
        const bit = @as(u8, 1) << @intCast(index);
        if (t.seen[part] & bit != 0) return error.DuplicateTap;
        var at = try t.plane(index);
        at.offset += @as(usize, first) * t.width * 2;
        // unfenced: the copy runs beside the mixer's projections, whose barrier orders it before any reader of the plane
        // and before the next write of `hidden` (two norms on)
        try glue.unstack(e, hidden, at, .{ .rows = rows, .width = t.width, .stride = t.width, .offset = 0 });
        t.seen[part] |= bit;
    }

    pub fn plane(t: Taps, index: usize) !Ref {
        if (index >= layers.len) return error.BadTapIndex;
        return .{ .buffer = t.buffer, .offset = index * @as(usize, t.capacity) * t.width * 2 };
    }

    pub fn complete(t: Taps) !void {
        for (t.seen[0..t.parts]) |seen| if (seen != (1 << layers.len) - 1) return error.IncompleteTaps;
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
