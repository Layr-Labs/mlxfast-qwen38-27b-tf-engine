//! Host guards for prompt parts, tap rings and constructor allocation failures.
const std = @import("std");
const taps = @import("native/qwen27_prompt_taps.zig");
const planning = @import("families/qwen27/attention_plan.zig");
const caches = @import("families/qwen27/dflash/caches.zig");
const Ring = @import("families/qwen27/tap_ring.zig").Ring;

const Gather = struct {
    end: usize,
    seen: usize = 0,
    scratch: [130]u32 = @splat(0xfeed),
    ring: [257]u32 = @splat(0),
    pub fn gather(g: *Gather, first: usize, rows: usize) !void {
        if (rows > 128) return error.ScratchOverflow;
        for (0..rows) |i| g.scratch[i + 1] = @intCast(first + i);
        for (g.scratch[1..][0..rows]) |row| {
            try std.testing.expectEqual(@as(u32, @intCast(g.seen)), row);
            g.ring[g.end % g.ring.len] = row;
            g.end += 1;
            g.seen += 1;
        }
        try std.testing.expectEqual(@as(u32, 0xfeed), g.scratch[0]);
        try std.testing.expectEqual(@as(u32, 0xfeed), g.scratch[129]);
    }
};

test "qwen27 speed: secondary tap chunks preserve source indices, guards, tails and restored rings" {
    for ([_]usize{ 1, 127, 128, 129, 256, 511, 512, 513, 2048 }) |rows| {
        for ([_]usize{ 0, 19, 256, 2041 }) |restored| {
            var g = Gather{ .end = restored };
            try taps.absorb(&g, rows);
            try std.testing.expectEqual(rows, g.seen);
            try std.testing.expectEqual(restored + rows, g.end);
            for (rows - @min(rows, g.ring.len)..rows) |i| {
                try std.testing.expectEqual(@as(u32, @intCast(i)), g.ring[(restored + i) % g.ring.len]);
            }
        }
    }
}

test "qwen27 speed: prompt parts fit the actual fused or unfused attention plan" {
    var parents: [2048]i32 = undefined;
    for (&parents, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
    for ([_]bool{ true, false }) |tensor| for ([_]bool{ true, false }) |fuse| {
        const geometry = planning.Geometry{ .shared_prefix = tensor, .fuse_chain = fuse };
        for ([_]u32{ 16, 128, 129, 512, 2048 }) |requested| {
            const part = geometry.promptRows(requested);
            const want = @min(requested, @as(u32, if (tensor and fuse) @import("families/qwen27/gpu_frame.zig").max_rows else planning.max_rows));
            try std.testing.expectEqual(want, part);
            for ([_]u32{ 128, 129, 512, 2051 }) |length| {
                var at: u32 = 0;
                while (at < length) {
                    const n = @min(part, length - at);
                    var plan = try planning.Plan.init(std.testing.allocator, &.{.{
                        .parents = parents[0..n],
                        .start = at,
                        .capacity = 4096,
                    }}, geometry);
                    defer plan.deinit();
                    try std.testing.expectEqual(n, plan.rows);
                    at += n;
                }
            }
        }
    };
}

const Device = struct {
    fail: usize,
    calls: usize = 0,
    alive: usize = 0,
    pub const Buffer = struct {
        owner: *Device,
        pub fn deinit(b: Buffer) void {
            b.owner.alive -= 1;
        }
    };
    pub fn buffer(d: *Device, _: usize, _: usize) !Buffer {
        const at = d.calls;
        d.calls += 1;
        if (at == d.fail) return error.NoBuffer;
        d.alive += 1;
        return .{ .owner = d };
    }
};

const Allocation = struct {
    backing: [1024]u8 align(16) = undefined,
    allocated: usize = 0,
    freed: usize = 0,
    wrong: bool = false,
    fn allocator(a: *Allocation) std.mem.Allocator {
        return .{ .ptr = a, .vtable = &.{ .alloc = alloc, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap, .free = free } };
    }
    fn alloc(ptr: *anyopaque, n: usize, _: std.mem.Alignment, _: usize) ?[*]u8 {
        const a: *Allocation = @ptrCast(@alignCast(ptr));
        if (n > a.backing.len) return null;
        a.allocated = n;
        return &a.backing;
    }
    fn free(ptr: *anyopaque, buf: []u8, _: std.mem.Alignment, _: usize) void {
        const a: *Allocation = @ptrCast(@alignCast(ptr));
        a.wrong = a.wrong or buf.len != a.allocated or buf.ptr != &a.backing;
        a.freed += buf.len;
    }
};

test "qwen27 speed: every cache buffer failure frees the complete CPU allocation and all device owners" {
    for (0..11) |failure| {
        var d = Device{ .fail = failure };
        var a: Allocation = .{};
        if (caches.newCaches(Device.Buffer, a.allocator(), &d, .{}, 5)) |made| {
            caches.freeCaches(Device.Buffer, a.allocator(), made);
        } else |err| try std.testing.expectEqual(error.NoBuffer, err);
        try std.testing.expectEqual(@as(usize, 0), d.alive);
        try std.testing.expect(!a.wrong);
        try std.testing.expectEqual(a.allocated, a.freed);
    }
}


test "qwen27 speed: incomplete counted normalization cannot reach host token selection" {
    const guard = @import("families/qwen27/counted_norm_status.zig");
    const Mock = struct {
        picked: bool = false,
        fn finish(m: *@This(), completed_failure_word: u32) !void {
            try guard.check(completed_failure_word);
            m.picked = true;
        }
    };
    var ready = Mock{};
    try ready.finish(0);
    try std.testing.expect(ready.picked);
    for ([_]u32{ 1, 2, 0xffffffff }) |failed| {
        var incomplete = Mock{};
        try std.testing.expectError(error.QwenNormDependencyIncomplete, incomplete.finish(failed));
        try std.testing.expect(!incomplete.picked);
    }
}

test "qwen27 speed: all counted dependency records are checked before host picks" {
    const guard = @import("families/qwen27/counted_norm_status.zig");
    const Mock = struct {
        picked: bool = false,
        fn finish(m: *@This(), norm: []const u32, gu: []const u32, attention: []const u32, scan: []const u32) !void {
            for ([_][]const u32{ norm, gu, attention, scan }) |bank| try guard.checkBank(bank);
            m.picked = true;
        }
    };
    var good: [6 * 4]u32 = @splat(0);
    good[0] = 99; good[1] = 1; good[3] = 7; // producer, arrival and reserved words are not failure flags
    var ready = Mock{};
    try ready.finish(good[0..4], good[4..8], good[8..20], good[20..24]);
    try std.testing.expect(ready.picked);
    for (0..6) |record| {
        var words = good;
        words[4 * record + 2] = 1;
        var bad = Mock{};
        try std.testing.expectError(error.QwenNormDependencyIncomplete, bad.finish(words[0..4], words[4..8], words[8..20], words[20..24]));
        try std.testing.expect(!bad.picked);
    }
    try std.testing.expectError(error.BadQwenCountedStatus, guard.checkBank(&.{}));
    try std.testing.expectError(error.BadQwenCountedStatus, guard.checkBank(&.{0,0,0}));
}
