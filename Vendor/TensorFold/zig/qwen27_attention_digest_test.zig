//! Exercise production upload and encode validation with host buffers and no GPU.
const std = @import("std");
const mtl = @import("metal");
const q = @import("src/families/qwen27/attention.zig");

const Case = struct {
    plan: q.Plan,
    scratch: q.Scratch,
    binding: q.Binding,
    buffers: [6]mtl.Buffer,
    caches: [1]q.Cache,
    op: q.Attention = .{ .prefix = @splat(.{}), .io = @splat(.{}), .prompt = null, .units = true },
    fn init() !Case {
        const device: mtl.Device = .{};
        var plan = try q.Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0, 0 }, .start = 63, .capacity = 512 }}, .{});
        errdefer plan.deinit();
        const limits = q.Limits{ .rows = 3, .streams = 1, .keys = 512 };
        var scratch = try q.Scratch.init(device, limits);
        errdefer scratch.deinit();
        var binding = try q.Binding.init(device, limits);
        errdefer binding.deinit();
        var buffers: [6]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (buffers[0..made]) |buffer| buffer.deinit();
        for ([_]usize{ 512 * 1024 * 2, 512 * 1024 * 2, 3 * 6144 * 2, 3 * 1024 * 2, 3 * 1024 * 2, 3 * 6144 * 2 }, &buffers) |bytes, *buffer| {
            buffer.* = try device.buffer(bytes, 0);
            made += 1;
        }
        var c = Case{ .plan = plan, .scratch = scratch, .binding = binding, .buffers = buffers, .caches = .{.{ .keys = .{ .buffer = buffers[0] }, .values = .{ .buffer = buffers[1] }, .capacity = 512 }} };
        try c.scratch.upload(c.plan);
        try c.binding.upload(c.plan, &c.caches);
        return c;
    }
    fn deinit(c: *Case) void {
        for (c.buffers) |buffer| buffer.deinit();
        c.binding.deinit();
        c.scratch.deinit();
        c.plan.deinit();
    }
    fn encode(c: *Case, plan: q.Plan, caches: []const q.Cache, record: *mtl.Record) !void {
        try c.op.encode(.{ .record = record }, &c.scratch, &c.binding, plan, .{ .buffer = c.buffers[2] }, .{ .buffer = c.buffers[3] }, .{ .buffer = c.buffers[4] }, .{ .buffer = c.buffers[5] }, caches, null, false);
    }
    fn expectChanged(c: *Case) !void {
        try std.testing.expectError(error.AttentionNotUploaded, c.scratch.check(c.plan));
        try std.testing.expectError(error.AttentionNotUploaded, c.binding.check(c.plan, &c.caches));
        try std.testing.expectError(error.AttentionNotUploaded, c.binding.admitDeviceKeep(c.plan, &c.caches, 3));
        try std.testing.expectError(error.AttentionNotUploaded, c.binding.uploadKeep(c.plan, &c.caches, &.{&.{0}}));
        var record: mtl.Record = .{};
        try std.testing.expectError(error.AttentionNotUploaded, c.encode(c.plan, &c.caches, &record));
        try std.testing.expectEqual(@as(usize, 0), record.dispatches);
    }
};

test "qwen27 digest: every encode validates current plan slices and metadata" {
    var c = try Case.init();
    defer c.deinit();
    var record: mtl.Record = .{};
    try c.encode(c.plan, &c.caches, &record);
    try std.testing.expect(record.dispatches > 0);
    inline for (.{ "tile_stream", "q_rows", "nodes", "paths", "positions" }) |name| {
        const values = @field(c.plan, name);
        const at = if (comptime std.mem.eql(u8, name, "paths")) 2 * q.planning.path_width + 1 else 0;
        const before = values[at];
        values[at] ^= 1;
        try c.expectChanged();
        values[at] = before;
    }
    c.plan.meta[12] += 1;
    try c.expectChanged();
    c.plan.meta[12] -= 1;
    record = .{};
    try c.encode(c.plan, &c.caches, &record);
    try std.testing.expect(record.dispatches > 0);
}

test "qwen27 digest: changed parents and mismatched upload identities still fail" {
    var c = try Case.init();
    defer c.deinit();
    var other = try q.Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0, 1 }, .start = 63, .capacity = 512 }}, .{});
    defer other.deinit();
    var record: mtl.Record = .{};
    try std.testing.expectError(error.AttentionNotUploaded, c.encode(other, &c.caches, &record));
    try c.scratch.upload(other);
    try c.scratch.check(other);
    try std.testing.expectError(error.AttentionNotUploaded, c.encode(other, &c.caches, &record));
    try std.testing.expectEqual(@as(usize, 0), record.dispatches);
    try c.binding.upload(other, &c.caches);
    try c.encode(other, &c.caches, &record);
    try std.testing.expect(record.dispatches > 0);
}

test "qwen27 digest: cache address stride capacity and missing uploads still fail" {
    var c = try Case.init();
    defer c.deinit();
    for (0..4) |i| {
        var caches = c.caches;
        const want: anyerror = switch (i) {
            0 => blk: {
                caches[0].keys.offset = 16;
                break :blk error.AttentionNotUploaded;
            },
            1 => blk: {
                caches[0].key_stride = 512 * 256 + 16;
                break :blk error.AttentionNotUploaded;
            },
            2 => blk: {
                caches[0].capacity = 64;
                caches[0].key_stride = 512 * 256;
                caches[0].value_stride = 512 * 256;
                break :blk error.AttentionCapacity;
            },
            3 => blk: {
                caches[0].capacity = 0;
                break :blk error.AttentionNotUploaded;
            },
            else => unreachable,
        };
        try std.testing.expectError(want, c.binding.check(c.plan, &caches));
        var record: mtl.Record = .{};
        try std.testing.expectError(want, c.encode(c.plan, &caches, &record));
        try std.testing.expectEqual(@as(usize, 0), record.dispatches);
    }
    var record: mtl.Record = .{};
    const uploaded = c.binding.uploaded;
    c.binding.uploaded = null;
    try std.testing.expectError(error.AttentionNotUploaded, c.binding.check(c.plan, &c.caches));
    try std.testing.expectError(error.AttentionNotUploaded, c.encode(c.plan, &c.caches, &record));
    c.binding.uploaded = uploaded;
    c.scratch.uploaded = null;
    try std.testing.expectError(error.AttentionNotUploaded, c.encode(c.plan, &c.caches, &record));
    try std.testing.expectEqual(@as(usize, 0), record.dispatches);
}
