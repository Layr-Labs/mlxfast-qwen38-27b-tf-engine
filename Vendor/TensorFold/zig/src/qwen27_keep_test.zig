//! The real pool round and shared keep encode against a recording device with byte sentinels.
const std = @import("std");
const metal = @import("metal");
const gpu = @import("families/qwen27/gdn_pool_gpu.zig");
const abi = @import("families/qwen27/gdn_contract.zig");
const gdn = @import("families/qwen27/gdn.zig");
const Plan = @import("families/qwen27/gdn_plan.zig").Plan;
const Window = @import("families/qwen27/state.zig").Window;
const Config = @import("families/qwen27/config.zig").Config;

test {
    _ = @import("families/qwen27/gdn_test.zig");
    _ = @import("families/qwen27/attention_plan.zig");
}

test "qwen27 speed: shared in-place verify then empty and full keeps restore state and retain conv histories" {
    const a = std.testing.allocator;
    var c: Config = undefined;
    c.k_heads = 1;
    c.v_heads = 1;
    c.dk = 128;
    c.dv = 128;
    c.conv_kernel = 2;
    c.layers = 1;
    c.eps = 1e-6;
    const shape = try abi.Shape.init(c);
    const kinds = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
    var window = try Window.init(a, 9, &.{-1}, 2, 64);
    defer window.deinit();
    for ([_]bool{ false, true }) |empty_second| {
        var pool: gpu.Pool = undefined;
        pool.budget = try @import("families/qwen27/gdn_pool.zig").Budget.init(shape, 1, 2, 2);
        pool.before = null;
        pool.before_slots = 0;
        pool.profiler = null;
        pool.tree_mode = .auto;
        pool.lazy = true;
        pool.stash = null;
        pool.pending = null;
        pool.lazy_keep = false;
        pool.consume = false;
        for (&pool.buffers, 0..) |*b, i| {
            b.* = .{ .id = i + 1, .data = try a.alignedAlloc(u8, .@"16", 2 * shape.state * 4) };
            @memset(b.data, 0);
        }
        defer for (pool.buffers) |b| a.free(b.data);
        pool.kernels = undefined;
        pool.kernels.pipelines = .{ .pre = .{ .id = 0 }, .tree = .{ .id = 1 }, .replay = .{ .id = 2 }, .post = .{ .id = 3 }, .conv_commit = .{ .id = 4 }, .tree_slots = .{ .{ .id = 10 }, .{ .id = 11 }, .{ .id = 12 }, .{ .id = 13 }, .{ .id = 14 } } };
        pool.kernels.publish_conv = .{ .id = 8 };
        const bytes = 2 * shape.state * 4;
        var metadata: gpu.Metadata = undefined;
        inline for (@typeInfo(gpu.Metadata).@"struct".field_names, 0..) |field, i| {
            const buffer = metal.Buffer{ .id = 100 + i, .data = try a.alignedAlloc(u8, .@"16", bytes) };
            @field(metadata, field) = .{ .buffer = buffer };
        }
        defer inline for (@typeInfo(gpu.Metadata).@"struct".field_names) |field| a.free(@field(metadata, field).buffer.data);
        const state = try pool.committed(0, 0, true);
        @memset(@constCast(state), 0x12);
        const other = try pool.committed(0, 1, true);
        @memset(@constCast(other), 0x34);
        const original = try a.dupe(u8, state);
        defer a.free(original);
        const second = try a.dupe(u8, other);
        defer a.free(second);
        const history = try pool.committed(0, 0, false);
        @memset(@constCast(history), 0x56);
        const other_history = try pool.committed(0, 1, false);
        @memset(@constCast(other_history), 0x78);
        const old_history = try a.dupe(u8, history);
        defer a.free(old_history);
        var plan = try Plan.init(a, try shape.params(2, 2, kinds), &.{
            .{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{0} },
            .{ .window = window, .state_slot = 1, .next_slot = 1, .path = &.{0} },
        });
        defer plan.deinit();
        const scratch = metal.Buffer{ .id = 999, .data = try a.alignedAlloc(u8, .@"16", bytes) };
        defer a.free(scratch.data);
        const output = metal.Buffer{ .id = 1000, .data = try a.alignedAlloc(u8, .@"16", bytes) };
        defer a.free(output.data);
        const span = gdn.Span{ .buffer = scratch, .bytes = scratch.length() };
        const refs = gpu.Refs{ .z = .{ .buffer = scratch }, .a = .{ .buffer = scratch }, .b = .{ .buffer = scratch }, .output = .{ .buffer = output }, .scalars = .{ .conv = span, .a_log = span, .dt = span, .norm = span, .kinds = kinds } };
        var record: metal.Recorder = .{};
        const e = metal.ComputeEncoder{ .record = &record };
        var allocations: metal.Allocations = .{};
        defer pool.releaseBefore();
        try pool.prepare(.{ .allocations = &allocations }, plan, false);
        try pool.forward(0, .{ .host = plan, .metadata = metadata }, refs, e);
        try std.testing.expect(!std.mem.eql(u8, original, state));
        var keep = try Plan.init(a, try shape.params(2, 2, kinds), &.{
            .{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{} },
            .{ .window = window, .state_slot = 1, .next_slot = 1, .path = if (empty_second) &.{} else &.{0} },
        });
        defer keep.deinit();
        try pool.commit(0, .{ .host = keep, .metadata = metadata }, e);
        try std.testing.expectEqualSlices(u8, original, state);
        try std.testing.expectEqual(empty_second, std.mem.eql(u8, second, other));
        try std.testing.expectEqualSlices(u8, old_history, history);
        for (other_history) |byte| try std.testing.expectEqual(@as(u8, if (empty_second) 0x78 else 0), byte);
        try std.testing.expectEqual(@as(u32, 9), (try window.keep(&.{})).length);
        try std.testing.expectEqual(@as(u32, 10), (try window.keep(&.{0})).length);
        try std.testing.expectEqual(@as(usize, if (empty_second) 2 else 1), record.restores);
    }
}

test "qwen27 speed: a device keep stashes its rows, the next verify's slot scan steps them and a flush replays them" {
    const a = std.testing.allocator;
    var c: Config = undefined;
    c.k_heads = 1;
    c.v_heads = 1;
    c.dk = 128;
    c.dv = 128;
    c.conv_kernel = 2;
    c.layers = 1;
    c.eps = 1e-6;
    const shape = try abi.Shape.init(c);
    const kinds = abi.WeightKinds{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 };
    var window = try Window.init(a, 9, &.{ -1, 0 }, 2, 64);
    defer window.deinit();
    var pool: gpu.Pool = undefined;
    pool.budget = try @import("families/qwen27/gdn_pool.zig").Budget.init(shape, 1, 2, 2);
    pool.before = null;
    pool.before_slots = 0;
    pool.profiler = null;
    pool.tree_mode = .auto;
    pool.residency = null;
    pool.lazy = true;
    pool.stash = null;
    pool.pending = null;
    pool.lazy_keep = false;
    pool.consume = false;
    for (&pool.buffers, 0..) |*b, i| {
        b.* = .{ .id = i + 1, .data = try a.alignedAlloc(u8, .@"16", 2 * shape.state * 4) };
        @memset(b.data, 0);
    }
    defer for (pool.buffers) |b| a.free(b.data);
    pool.kernels = undefined;
    pool.kernels.pipelines = .{ .pre = .{ .id = 0 }, .tree = .{ .id = 1 }, .replay = .{ .id = 2 }, .post = .{ .id = 3 }, .conv_commit = .{ .id = 4 }, .tree_slots = .{ .{ .id = 10 }, .{ .id = 11 }, .{ .id = 12 }, .{ .id = 13 }, .{ .id = 14 } } };
    pool.kernels.stash = .{ .id = 20 };
    const bytes = 2 * shape.state * 4;
    var metadata: gpu.Metadata = undefined;
    inline for (@typeInfo(gpu.Metadata).@"struct".field_names, 0..) |field, i| {
        const buffer = metal.Buffer{ .id = 100 + i, .data = try a.alignedAlloc(u8, .@"16", bytes) };
        @memset(buffer.data, 0);
        @field(metadata, field) = .{ .buffer = buffer };
    }
    defer inline for (@typeInfo(gpu.Metadata).@"struct".field_names) |field| a.free(@field(metadata, field).buffer.data);
    // one window, both rows kept: what a depth-1 round's device keep sees
    var plan = try Plan.init(a, try shape.params(2, 2, kinds), &.{.{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{ 0, 1 } }});
    defer plan.deinit();
    try std.testing.expect(pool.lazyFits(plan));
    const scratch = metal.Buffer{ .id = 999, .data = try a.alignedAlloc(u8, .@"16", bytes) };
    defer a.free(scratch.data);
    const output = metal.Buffer{ .id = 1000, .data = try a.alignedAlloc(u8, .@"16", bytes) };
    defer a.free(output.data);
    const span = gdn.Span{ .buffer = scratch, .bytes = scratch.length() };
    const refs = gpu.Refs{ .z = .{ .buffer = scratch }, .a = .{ .buffer = scratch }, .b = .{ .buffer = scratch }, .output = .{ .buffer = output }, .scalars = .{ .conv = span, .a_log = span, .dt = span, .norm = span, .kinds = kinds } };
    var record: metal.Recorder = .{ .segments = 1 };
    const e = metal.ComputeEncoder{ .record = &record };
    var allocations: metal.Allocations = .{};
    try pool.prepare(.{ .allocations = &allocations }, plan, false);
    const stash = pool.stash orelse return error.NoStash;
    defer stash.deinit();
    const state_id = pool.buffers[0].id;
    // the keep: the stash in place of the replay, then the conv window
    pool.lazy_keep = true;
    try pool.commitScan(0, .{ .host = plan, .metadata = metadata }, e);
    pool.lazy_keep = false;
    pool.pending = 0;
    try std.testing.expectEqual(@as(usize, 2), record.event_count);
    try std.testing.expectEqual(@as(usize, 20), record.events[0].pipeline);
    try std.testing.expectEqual(@as(usize, 4), record.events[1].pipeline);
    // the next verify: its slot scan binds the stash and turns the prologue on
    try std.testing.expect(pool.consumable(plan));
    record.event_count = 0;
    pool.consume = true;
    try pool.forward(0, .{ .host = plan, .metadata = metadata }, refs, e);
    pool.consume = false;
    var scans: usize = 0;
    for (record.events[0..record.event_count]) |event| if (event.pipeline >= 10 and event.pipeline <= 14) {
        scans += 1;
        try std.testing.expectEqual(@as(?usize, stash.id), event.stash);
        try std.testing.expectEqual(@as(u32, 1), event.lazy_on);
    };
    try std.testing.expectEqual(@as(usize, 1), scans);
    // a verify not consuming: the prologue stays off
    record.event_count = 0;
    try pool.forward(0, .{ .host = plan, .metadata = metadata }, refs, e);
    for (record.events[0..record.event_count]) |event| if (event.pipeline >= 10 and event.pipeline <= 14) try std.testing.expectEqual(@as(u32, 0), event.lazy_on);
    // a flush: the stash replayed into the slot's state, then nothing pending
    record.event_count = 0;
    try pool.flush(e);
    try std.testing.expect(pool.pending == null);
    try std.testing.expectEqual(@as(usize, 1), record.event_count);
    try std.testing.expectEqual(@as(usize, 2), record.events[0].pipeline);
    try std.testing.expectEqual(@as(?usize, state_id), record.events[0].state);
    record.event_count = 0;
    try pool.flush(e);
    try std.testing.expectEqual(@as(usize, 0), record.event_count);
    // a shared round (two windows) never stashes, and a pool with lazy keeps off never does
    var shared = try Plan.init(a, try shape.params(4, 2, kinds), &.{
        .{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{0} },
        .{ .window = window, .state_slot = 1, .next_slot = 1, .path = &.{0} },
    });
    defer shared.deinit();
    try std.testing.expect(!pool.lazyFits(shared));
    pool.lazy = false;
    try std.testing.expect(!pool.lazyFits(plan));
}
