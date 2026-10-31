//! Native flags, byte counts and non-prefix replay are checked without a Metal device.
const std = @import("std");
const Config = @import("config.zig").Config;
const abi = @import("gdn_contract.zig");
const plan_mod = @import("gdn_plan.zig");
const Window = @import("state.zig").Window;

fn config() Config {
    return .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 };
}

const native = abi.WeightKinds{ .conv = .bf16, .a_log = .f32, .dt = .bf16, .norm = .f32 };

test "GDN Metal records have fixed offsets without enum padding" {
    try std.testing.expectEqual(@as(usize, 44), @sizeOf(abi.Params));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(abi.Params, "eps"));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(abi.Segment));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(abi.Keep));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(@import("core").tree_slots.Step));
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(@import("core").tree_slots.Step, "read_slot"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(@import("core").tree_slots.Step, "write_slot"));
}

test "compact GDN schedules isolate unequal streams and overflow keeps the snapshot plan valid" {
    const a = std.testing.allocator;
    const slots = @import("core").tree_slots;
    var parents: [33]i32 = undefined;
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast((row - 1) / 2);
    var bush = try Window.init(a, 100, &parents, 4, 128);
    defer bush.deinit();
    var chain = try Window.init(a, 200, &.{ -1, 0, 1, 2, 3 }, 4, 512);
    defer chain.deinit();
    const shape = try abi.Shape.init(config());
    const p = try shape.params(38, 4, native);
    const inputs = [_]plan_mod.Input{ .{ .window = bush, .state_slot = 3, .next_slot = 0, .path = &.{ 0, 2, 5 } }, .{ .window = chain, .state_slot = 2, .next_slot = 1, .path = &.{} } };
    var plan = try plan_mod.Plan.init(a, p, &inputs);
    defer plan.deinit();
    try std.testing.expect(plan.compact);
    try @import("gdn_pool.zig").validatePlan(shape, 4, 64, plan);
    try slots.validate(&parents, plan.slot_steps[0..33], 0, 16);
    try slots.validate(chain.parents, plan.slot_steps[33..], 33, 16);
    try std.testing.expectEqual(@as(i32, -1), plan.slot_steps[33].read_slot);
    const saved = plan.slot_steps[1];
    plan.slot_steps[1].read_slot = 15;
    try std.testing.expectError(error.BadGdnPoolPlan, @import("gdn_pool.zig").validatePlan(shape, 4, 64, plan));
    plan.slot_steps[1] = saved;
    var fallback = try plan_mod.Plan.initSlots(a, p, &inputs, 1);
    defer fallback.deinit();
    try std.testing.expect(!fallback.compact);
    try @import("gdn_pool.zig").validatePlan(shape, 4, 64, fallback);
    try std.testing.expectEqualSlices(i32, plan.parents, fallback.parents);
    try std.testing.expectEqualSlices(u32, plan.kept_rows, fallback.kept_rows);
    try std.testing.expectError(error.BadSlotPlan, plan_mod.Plan.initSlots(a, p, &inputs, 0));
    try std.testing.expectError(error.BadSlotPlan, plan_mod.Plan.initSlots(a, p, &inputs, 17));
}

test "native storage keeps all sixteen combinations and refuses an implicit F16 cast" {
    for (0..16) |flags| {
        const k = abi.WeightKinds{ .conv = @fromBackingInt(@intCast(flags & 1)), .a_log = @fromBackingInt(@intCast((flags >> 1) & 1)), .dt = @fromBackingInt(@intCast((flags >> 2) & 1)), .norm = @fromBackingInt(@intCast((flags >> 3) & 1)) };
        try std.testing.expectEqual(@as(u32, @intCast(flags)), k.flags());
    }
    try std.testing.expectEqual(abi.Storage.f32, try abi.storage(.f32));
    try std.testing.expectEqual(abi.Storage.bf16, try abi.storage(.bf16));
    try std.testing.expectError(error.UnsupportedGdnStorage, abi.storage(.f16));
}

test "actual target widths and snapshot bytes preserve FP32 state" {
    const shape = try abi.Shape.init(config());
    try std.testing.expectEqual(@as(usize, 10240), shape.qkv);
    try std.testing.expectEqual(@as(usize, 6144), shape.value);
    try std.testing.expectEqual(@as(usize, 786432), shape.state);
    const sizes = try abi.Bytes.init(shape, try shape.params(8, 4, native), native);
    try std.testing.expectEqual(@as(usize, 8 * 786432 * 4), sizes.snapshots);
    try std.testing.expectEqual(@as(usize, 4 * 786432 * 4), sizes.state);
    try std.testing.expectEqual(@as(usize, 48 * 4), sizes.a_log);
    try std.testing.expectEqual(@as(usize, 48 * 2), sizes.dt);
}

test "wrong shape flags and dimensions refuse before an encoder exists" {
    var c = config();
    c.dk = 127;
    try std.testing.expectError(error.UnsupportedGdnShape, abi.Shape.init(c));
    c = config();
    c.v_heads = 47;
    try std.testing.expectError(error.UnsupportedGdnShape, abi.Shape.init(c));
    const shape = try abi.Shape.init(config());
    try std.testing.expectError(error.BadGdnDispatch, shape.params(0, 4, native));
    try std.testing.expectError(error.BadGdnDispatch, shape.params(8, 129, native));
    var p = try shape.params(8, 4, native);
    p.dk = 64;
    try std.testing.expectError(error.BadGdnDispatch, abi.Bytes.init(shape, p, native));
    try std.testing.expectError(error.Overflow, abi.countBytes(f32, std.math.maxInt(usize), 2));
}

test "unequal windows flatten conv rows without shifting history or local parents" {
    const a = std.testing.allocator;
    var w0 = try Window.init(a, 20, &.{ -1, 0, 0, 1, 2 }, 4, 128);
    defer w0.deinit();
    var w1 = try Window.init(a, 40, &.{ -1, 0, 1 }, 4, 128);
    defer w1.deinit();
    const shape = try abi.Shape.init(config());
    var plan = try plan_mod.Plan.init(a, try shape.params(8, 4, native), &.{ .{ .window = w0, .state_slot = 2, .next_slot = 0, .path = &.{ 0, 2, 4 } }, .{ .window = w1, .state_slot = 3, .next_slot = 1, .path = &.{ 0, 1 } } });
    defer plan.deinit();
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 1, 2, -1, 0, 1 }, plan.parents);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 8 }, plan.windows[20..24]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 8, 9, 10 }, plan.windows[28..32]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4, 5, 6 }, plan.kept_rows);
    try std.testing.expectEqual(@as(u32, 5), plan.segments[1].first);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2, 2, 2, 2, 3, 3, 3 }, plan.row_slots);
}

test "empty keep retains the base and duplicate destination slots refuse" {
    const a = std.testing.allocator;
    var w = try Window.init(a, 20, &.{ -1, 0 }, 4, 128);
    defer w.deinit();
    const shape = try abi.Shape.init(config());
    const p = try shape.params(2, 2, native);
    var plan = try plan_mod.Plan.init(a, p, &.{.{ .window = w, .state_slot = 1, .next_slot = 0, .path = &.{} }});
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 0), plan.kept_rows.len);
    try std.testing.expectEqual(@as(u32, 0), plan.keeps[0].rows);
    const inputs = [_]plan_mod.Input{ .{ .window = w, .state_slot = 0, .next_slot = 1, .path = &.{0} }, .{ .window = w, .state_slot = 1, .next_slot = 1, .path = &.{0} } };
    try std.testing.expectError(error.OverlappingGdnCommit, plan_mod.Plan.init(a, try shape.params(4, 2, native), &inputs));
}

test "a bad accepted path or input extent cannot become replay indices" {
    const a = std.testing.allocator;
    var w = try Window.init(a, 20, &.{ -1, 0, 0 }, 4, 128);
    defer w.deinit();
    const shape = try abi.Shape.init(config());
    const bad = [_]plan_mod.Input{.{ .window = w, .state_slot = 0, .next_slot = 1, .path = &.{ 0, 1, 2 } }};
    try std.testing.expectError(error.BadPath, plan_mod.Plan.init(a, try shape.params(3, 2, native), &bad));
    const good = [_]plan_mod.Input{.{ .window = w, .state_slot = 0, .next_slot = 1, .path = &.{ 0, 2 } }};
    try std.testing.expectError(error.BadGdnPlan, plan_mod.Plan.init(a, try shape.params(2, 2, native), &good));
}

test "snapshot scratch scales with the aggregate round instead of layers or slots" {
    const Budget = @import("gdn_pool.zig").Budget;
    const shape = try abi.Shape.init(config());
    const one = try Budget.init(shape, 48, 1, 32);
    const four = try Budget.init(shape, 48, 4, 32);
    try std.testing.expectEqual(@as(usize, 96 * 1024 * 1024), one.snapshots);
    try std.testing.expectEqual(one.snapshots, four.snapshots);
    try std.testing.expectEqual(one.logBytes(), four.logBytes());
    try std.testing.expectEqual(@as(usize, 30 * 1024 * 1024), one.projections);
    try std.testing.expectEqual(@as(usize, 144 * 1024 * 1024), one.state);
    try std.testing.expect(one.logBytes() < 31 * 1024 * 1024);
    try std.testing.expectEqual(@as(usize, 786432 * 4), try four.stateOffset(0, 1));
    try std.testing.expectEqual(@as(usize, 48 * 786432 * 4 * 4), four.state);
    try std.testing.expectError(error.BadGdnPoolIndex, four.stateOffset(48, 0));
    try std.testing.expectError(error.BadGdnPool, Budget.init(shape, 48, 4, @import("gpu_frame.zig").max_rows + 1));
    // a prompt chain's rows past a tree's keep no snapshots: the slab stays at the tree's
    const tree = try Budget.init(shape, 48, 1, 128);
    const chain = try Budget.init(shape, 48, 1, 512);
    try std.testing.expectEqual(tree.snapshots, chain.snapshots);
    try std.testing.expectEqual(4 * tree.projections, chain.projections);
}

test "native byte spans accept BF16 alignment and reject unsafe range arithmetic" {
    try abi.checkSpan(1024, 2, 1022, 512, 2);
    try std.testing.expectError(error.BadGdnBuffer, abi.checkSpan(1024, 2, 1022, 512, 4));
    try std.testing.expectError(error.BadGdnBuffer, abi.checkSpan(1024, 1025, 0, 0, 2));
    try std.testing.expectError(error.BadGdnBuffer, abi.checkSpan(1024, 0, 1025, 512, 2));
    try std.testing.expectError(error.BadGdnBuffer, abi.checkSpan(1024, 0, 512, 1024, 2));
}

test "a plan from a larger slot pool cannot index the committed state bank" {
    const a = std.testing.allocator;
    var w = try Window.init(a, 0, &.{ -1, 0 }, 4, 128);
    defer w.deinit();
    const shape = try abi.Shape.init(config());
    var plan = try plan_mod.Plan.init(a, try shape.params(2, 8, native), &.{.{ .window = w, .state_slot = 4, .next_slot = 5, .path = &.{ 0, 1 } }});
    defer plan.deinit();
    try @import("gdn_pool.zig").validatePlan(shape, 8, 32, plan);
    try std.testing.expectError(error.BadGdnPoolPlan, @import("gdn_pool.zig").validatePlan(shape, 4, 32, plan));
}

test "cross-stream conv rows and a broken retained path refuse before encoding" {
    const a = std.testing.allocator;
    var w = try Window.init(a, 0, &.{ -1, 0, 0 }, 4, 128);
    defer w.deinit();
    const shape = try abi.Shape.init(config());
    var plan = try plan_mod.Plan.init(a, try shape.params(6, 2, native), &.{ .{ .window = w, .state_slot = 0, .next_slot = 0, .path = &.{ 0, 2 } }, .{ .window = w, .state_slot = 1, .next_slot = 1, .path = &.{ 0, 1 } } });
    defer plan.deinit();
    try @import("gdn_pool.zig").validatePlan(shape, 2, 32, plan);
    plan.windows[12] = 3;
    try std.testing.expectError(error.BadGdnPoolPlan, @import("gdn_pool.zig").validatePlan(shape, 2, 32, plan));
    plan.windows[12] = 0;
    plan.kept_rows[0] = 1;
    try std.testing.expectError(error.BadGdnPoolPlan, @import("gdn_pool.zig").validatePlan(shape, 2, 32, plan));
}

test "depth-one stars store only the root's state and every tree replays leaves without storing them" {
    const a = std.testing.allocator;
    const slots = @import("core").tree_slots;
    const shape = try abi.Shape.init(config());
    for (2..9) |rows| {
        var parents: [8]i32 = undefined;
        for (parents[0..rows], 0..) |*parent, row| parent.* = if (row == 0) -1 else 0;
        var window = try Window.init(a, 10, parents[0..rows], 4, 64);
        defer window.deinit();
        var plan = try plan_mod.Plan.init(a, try shape.params(@intCast(rows), 1, native), &.{.{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{0} }});
        defer plan.deinit();
        try std.testing.expect(plan.compact);
        try std.testing.expectEqual(@as(usize, 1), plan.slot_peak);
        try slots.validate(parents[0..rows], plan.slot_steps, 0, 1);
        for (plan.slot_steps) |step| try std.testing.expectEqual(@as(i32, if (step.row == 0) 0 else -1), step.write_slot);
    }
    // kernel semantics on random trees: a row reads its parent's state from a slot, and a leaf stores none
    var random: u64 = 1729;
    for ([_]usize{ 2, 5, 8, 17, 64 }) |width| for (0..32) |_| {
        var parents: [64]i32 = undefined;
        var expected: [64]u64 = undefined;
        var actual: [64]u64 = undefined;
        for (parents[0..width], 0..) |*parent, row| {
            random = random *% 6364136223846793005 +% 1442695040888963407;
            parent.* = if (row == 0) -1 else @intCast(random % row);
            const value = if (parent.* < 0) @as(u64, 23) else expected[@intCast(parent.*)];
            expected[row] = value *% 65537 +% row +% 1;
        }
        var window = try Window.init(a, 10, parents[0..width], 4, 128);
        defer window.deinit();
        var plan = try plan_mod.Plan.init(a, try shape.params(@intCast(width), 1, native), &.{.{ .window = window, .state_slot = 0, .next_slot = 0, .path = &.{0} }});
        defer plan.deinit();
        try std.testing.expect(plan.compact and plan.slot_peak <= slots.max_slots);
        try slots.validate(parents[0..width], plan.slot_steps, 0, @import("core_delta").slot_kernels[@import("core_delta").slotKernel(plan.slot_peak)]);
        var registers: [slots.max_slots]u64 = @splat(0);
        for (plan.slot_steps) |step| {
            const value = if (step.read_slot < 0) @as(u64, 23) else registers[@intCast(step.read_slot)];
            actual[step.row] = value *% 65537 +% step.row +% 1;
            if (step.write_slot >= 0) registers[@intCast(step.write_slot)] = actual[step.row];
        }
        try std.testing.expectEqualSlices(u64, expected[0..width], actual[0..width]);
    };
}

test "depth one siblings keep none first or second draft with matching state and audit maps" {
    const a = std.testing.allocator;
    const matcher = @import("core").tree_round;
    const audit = @import("dflash/audit.zig");
    const parents = [_]i32{ -1, 0, 0 };
    const tokens = [_]u32{ 1, 2, 3 };
    const shape = try abi.Shape.init(config());
    var window = try Window.init(a, 10, &parents, 4, 64);
    defer window.deinit();
    var attention = try @import("attention_plan.zig").Plan.init(a, &.{.{ .parents = &parents, .start = 10, .capacity = 64 }}, .{});
    defer attention.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 11 }, attention.positions);
    for ([_]u32{ 4, 2, 3 }, 0..) |root_pick, which| {
        const picks = [_]matcher.Pick{ .{ .token = root_pick }, .{ .token = 5 }, .{ .token = 6 } };
        const match = matcher.Match{ .rows = 3, .vocab = 8, .budget = 2 };
        const raw = try matcher.oracle(match, &tokens, &parents, &picks, &.{});
        const accepted = try matcher.decode(&raw, match);
        const want_path: []const u32 = switch (which) {
            0 => &.{0},
            1 => &.{ 0, 1 },
            else => &.{ 0, 2 },
        };
        const want_tokens: []const u32 = switch (which) {
            0 => &.{4},
            1 => &.{ 2, 5 },
            else => &.{ 3, 6 },
        };
        try std.testing.expectEqualSlices(u32, want_path, accepted.path);
        try std.testing.expectEqualSlices(u32, want_tokens, accepted.tokens);
        var recurrent = try plan_mod.Plan.init(a, try shape.params(3, 1, native), &.{.{ .window = window, .state_slot = 0, .next_slot = 0, .path = accepted.path }});
        defer recurrent.deinit();
        try std.testing.expectEqualSlices(u32, want_path, recurrent.kept_rows);
        // the siblings are leaves and store no state: the root's is the only one held
        try std.testing.expectEqual(@as(usize, 1), recurrent.slot_peak);
        try @import("core").tree_slots.validate(&parents, recurrent.slot_steps, 0, 1);
        const kept = try window.keep(accepted.path);
        try std.testing.expectEqual(@as(u32, @intCast(10 + want_path.len)), kept.length);
        try std.testing.expectEqual(@as(?u32, want_path[want_path.len - 1]), kept.conv_tail_row);
        const map = try attention.keepRows(a, &.{accepted.path});
        defer a.free(map);
        var cache = [_]u32{ 101, 202, 303 };
        var gathered: [2]u32 = undefined;
        for (map) |row| gathered[row.row] = cache[row.source];
        for (map) |row| cache[row.destination] = gathered[row.row];
        try std.testing.expectEqual(@as(u32, 101), cache[0]);
        if (which != 0) try std.testing.expectEqual(@as(u32, if (which == 1) 202 else 303), cache[1]);
        var journal: audit.Journal = .{};
        defer journal.deinit(a);
        try journal.record(a, .{ .committed = @intCast(accepted.tokens.len), .drafted = 2, .accepted = accepted.matched });
        try std.testing.expectEqual(@as(u64, 2), journal.drafted_total);
        try std.testing.expectEqual(@as(u64, if (which == 0) 0 else 1), journal.accepted_total);
        try std.testing.expectEqual(@as(u64, want_tokens.len), journal.committed_total);
        try std.testing.expect(journal.committed_total <= 2);
    }
}

test "depth one sibling terminal and remaining-one commits retain only the root state" {
    const a = std.testing.allocator;
    const matcher = @import("core").tree_round;
    var window = try Window.init(a, 10, &.{ -1, 0, 0 }, 4, 64);
    defer window.deinit();
    const shape = try abi.Shape.init(config());
    for ([_]u32{ 2, 3 }) |pick| for ([_]bool{ false, true }) |eos| {
        const match = matcher.Match{ .rows = 3, .vocab = 8, .budget = if (eos) 2 else 1, .eos_count = @intFromBool(eos) };
        const stop = [_]u32{pick};
        const raw = try matcher.oracle(match, &.{ 1, 2, 3 }, &.{ -1, 0, 0 }, &.{ .{ .token = pick }, .{ .token = 5 }, .{ .token = 6 } }, if (eos) &stop else &.{});
        const accepted = try matcher.decode(&raw, match);
        try std.testing.expectEqualSlices(u32, &.{0}, accepted.path);
        try std.testing.expectEqualSlices(u32, &stop, accepted.tokens);
        var plan = try plan_mod.Plan.init(a, try shape.params(3, 1, native), &.{.{ .window = window, .state_slot = 0, .next_slot = 0, .path = accepted.path }});
        defer plan.deinit();
        try std.testing.expectEqualSlices(u32, &.{0}, plan.kept_rows);
        try std.testing.expectEqual(@as(u32, 11), (try window.keep(accepted.path)).length);
    };
}
