//! Production CUDA phase drains with fake device resources.
const std = @import("std");
const cuda = @import("cuda");
const pages = @import("cuda_pages.zig");
const E = @import("cuda_engine.zig").Engine;
const D = cuda.paging_test_device.Device;
const t = std.testing;

fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = D.stream };
    e.kv8 = false;
    e.n_taps = 0;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 64, .linear = @splat(false) };
    e.g.linear[0] = true;
    return e;
}

pub fn drains(comptime closePhaseCaches: anytype, comptime closeUploadSpares: anytype) !void {
    const EmptyCaches = struct {
        page_pool: ?*pages.Pages = null,
        d: *const cuda.Driver,
        context: @TypeOf(cuda.copy_source.current()),

        pub fn closeWindows(_: *@This()) u64 {
            return 0;
        }
        pub fn closePages(c: *@This()) u64 {
            return if (c.page_pool) |p| p.drain() else 0;
        }
        pub fn closeRecurrent(_: *@This()) u64 {
            return 0;
        }
        pub fn closeDrafts(_: *@This()) u64 {
            return 0;
        }
        pub fn closeUploads(c: *@This()) u64 {
            return closeUploadSpares(c.d, c.context);
        }
    };
    var d = D.driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream, "the phase-drain test driver must drain before it leaves");
    const layout = pages.layout(&e, 64, 16, 0, 0);
    const p = try pages.Pages.init(t.allocator, &e, layout, 64, 0, 1, layout.need(64, 0, pages.window_rows));
    defer p.deinit();
    const context = @as(@TypeOf(cuda.copy_source.current()), @ptrFromInt(41));
    var caches: EmptyCaches = .{ .page_pool = p, .d = &d, .context = context };

    const cache = try p.open(1, 0);
    const held = p.cacheBytes();
    try t.expect(held > 0);
    try t.expectEqual(held, closePhaseCaches(&caches));
    try t.expect(p.close(cache));
    try t.expectEqual(@as(u64, 0), closePhaseCaches(&caches));
    try t.expectEqual(@as(u64, 0), p.cacheBytes());

    cuda.copy_source.setCurrent(context);
    defer cuda.copy_source.setCurrent(null);
    cuda.copy_source.setSparesForTest(.{ .count = 1, .bytes = 1 << 20 });
    defer {
        cuda.copy_source.dropSpares(&d, context);
        cuda.copy_source.setSparesForTest(.{ .count = 0, .bytes = 0 });
    }
    try cuda.copy_source.enqueue(&d, 256, &@as([16]u8, @splat(7)), e.stream);
    try cuda.copy_source.synchronizeStream(&d, e.stream.handle);
    try t.expect(cuda.copy_source.spareBytes(&d, context) > 0);
    try t.expectEqual(@as(u64, 0), closePhaseCaches(&caches));
    try t.expectEqual(@as(u64, 0), cuda.copy_source.spareBytes(&d, context));
}

pub fn firstWindow(comptime takeFirstWindow: anytype, comptime FirstWindow: type) !void {
    const Request = struct {
        allocator: std.mem.Allocator,

        fn takeWindow(r: *@This()) anyerror!*u8 {
            const byte = try r.allocator.create(u8);
            byte.* = 7;
            return byte;
        }

        fn takePages(_: *@This(), _: usize, _: usize) anyerror!*u8 {
            return error.NoPageLane;
        }
    };
    var failing = std.testing.FailingAllocator.init(t.allocator, .{});
    failing.fail_index = failing.alloc_index;
    var request: Request = .{ .allocator = failing.allocator() };
    var window: FirstWindow = .{};
    try t.expectError(error.OutOfMemory, takeFirstWindow(&window, true, &request, 1, 1, Request.takeWindow, Request.takePages));
    try t.expect(!window.active);

    failing.fail_index = std.math.maxInt(usize);
    const byte = try takeFirstWindow(&window, true, &request, 1, 1, Request.takeWindow, Request.takePages);
    try t.expect(window.active);
    try t.expectEqual(@as(u8, 7), byte.*);
    window.release();
    t.allocator.destroy(byte);
}
