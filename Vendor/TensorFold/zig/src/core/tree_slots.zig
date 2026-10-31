//! Depth-first trees recycle a parent's register slot at its last child and preserve original output rows.
const std = @import("std");

pub const max_slots = 16;
pub const Step = extern struct { row: u32, read_slot: i32, write_slot: i32 };

pub fn build(gpa: std.mem.Allocator, parents: []const i32, steps: []Step, first: u32, limit: usize) !usize {
    return buildImpl(gpa, parents, steps, first, limit, false);
}

/// A tree scan consumes leaf outputs but never reads their resulting states.
pub fn buildForOutputs(gpa: std.mem.Allocator, parents: []const i32, steps: []Step, first: u32, limit: usize) !usize {
    return buildImpl(gpa, parents, steps, first, limit, true);
}

fn buildImpl(gpa: std.mem.Allocator, parents: []const i32, steps: []Step, first: u32, limit: usize, discard_leaves: bool) !usize {
    if (parents.len == 0 or parents.len != steps.len or limit == 0 or limit > max_slots or parents.len > std.math.maxInt(u32) - first) return error.BadSlotPlan;
    const sizes = try gpa.alloc(usize, parents.len);
    defer gpa.free(sizes);
    @memset(sizes, 1);
    for (parents, 0..) |parent, row| if (parent < -1 or parent >= row) return error.BadSlotParents;
    var remaining = parents.len;
    while (remaining > 0) {
        remaining -= 1;
        if (parents[remaining] >= 0) sizes[@intCast(parents[remaining])] += sizes[remaining];
    }
    var planner = Builder{ .parents = parents, .sizes = sizes, .steps = steps, .first = first, .limit = limit, .discard_leaves = discard_leaves };
    for (parents, 0..) |parent, row| if (parent == -1) try planner.visit(row, -1, false);
    std.debug.assert(planner.at == steps.len and planner.live == 0);
    return if (discard_leaves) @max(@as(usize, 1), planner.peak) else planner.peak;
}

const Builder = struct {
    parents: []const i32,
    sizes: []const usize,
    steps: []Step,
    first: u32,
    limit: usize,
    discard_leaves: bool,
    used: [max_slots]bool = @splat(false),
    live: usize = 0,
    peak: usize = 0,
    at: usize = 0,

    fn take(b: *Builder) !i32 {
        for (b.used[0..b.limit], 0..) |*used, slot| if (!used.*) {
            used.* = true;
            b.live += 1;
            b.peak = @max(b.peak, b.live);
            return @intCast(slot);
        };
        return error.SlotOverflow;
    }

    fn visit(b: *Builder, row: usize, read: i32, reuse: bool) error{SlotOverflow}!void {
        var last: ?usize = null;
        for (b.parents, 0..) |parent, child| if (parent == row) {
            if (last == null or b.sizes[child] >= b.sizes[last.?]) last = child;
        };
        if (b.discard_leaves and last == null) {
            b.steps[b.at] = .{ .row = b.first + @as(u32, @intCast(row)), .read_slot = read, .write_slot = -1 };
            b.at += 1;
            if (reuse) {
                std.debug.assert(read >= 0 and b.used[@intCast(read)]);
                b.used[@intCast(read)] = false;
                b.live -= 1;
            }
            return;
        }
        const write = if (reuse) read else try b.take();
        b.steps[b.at] = .{ .row = b.first + @as(u32, @intCast(row)), .read_slot = read, .write_slot = write };
        b.at += 1;
        if (last) |heavy| {
            for (b.parents, 0..) |parent, child| if (parent == row and child != heavy) try b.visit(child, write, false);
            try b.visit(heavy, write, true);
        } else {
            b.used[@intCast(write)] = false;
            b.live -= 1;
        }
    }
};

pub fn validate(parents: []const i32, steps: []const Step, first: u32, limit: usize) !void {
    return validateImpl(parents, steps, first, limit, false);
}

/// Negative writes are valid only for childless nodes and an output-only consumer.
pub fn validateForOutputs(parents: []const i32, steps: []const Step, first: u32, limit: usize) !void {
    return validateImpl(parents, steps, first, limit, true);
}

fn validateImpl(parents: []const i32, steps: []const Step, first: u32, limit: usize, discard_leaves: bool) !void {
    if (parents.len == 0 or parents.len != steps.len or limit == 0 or limit > max_slots) return error.BadSlotPlan;
    var owners: [max_slots]i32 = @splat(-2);
    for (steps, 0..) |step, at| {
        if (step.row < first or step.row - first >= parents.len or step.read_slot < -1 or step.read_slot >= limit or step.write_slot < (if (discard_leaves) @as(i32, -1) else 0) or step.write_slot >= limit) return error.BadSlotPlan;
        const row = step.row - first;
        const parent = parents[row];
        if (parent < -1 or parent >= row) return error.BadSlotParents;
        if (step.read_slot < 0) {
            if (parent != -1) return error.BadSlotPlan;
        } else if (owners[@intCast(step.read_slot)] != parent) return error.BadSlotPlan;
        for (steps[0..at]) |prior| if (prior.row == step.row) return error.BadSlotPlan;
        if (step.write_slot < 0) {
            for (parents) |child_parent| if (child_parent == row) return error.BadSlotPlan;
        } else owners[@intCast(step.write_slot)] = @intCast(row);
    }
}

test "slot schedules retain each parent's state while moving the largest child last" {
    const parents = [_]i32{ -1, 0, 0, 1, 1, 3, 2, 6, 6 };
    var steps: [parents.len]Step = undefined;
    try std.testing.expectEqual(@as(usize, 3), try build(std.testing.allocator, &parents, &steps, 23, 16));
    try validate(&parents, &steps, 23, 16);
    try std.testing.expectEqualSlices(u32, &.{ 23, 24, 27, 26, 28, 25, 29, 30, 31 }, &rowsOf(steps));
    steps[2].read_slot = 15;
    try std.testing.expectError(error.BadSlotPlan, validate(&parents, &steps, 23, 16));
}

fn rowsOf(steps: anytype) [steps.len]u32 {
    var rows: [steps.len]u32 = undefined;
    for (steps, &rows) |step, *row| row.* = step.row;
    return rows;
}

test "long chains need one slot and stars need two independent of depth or width" {
    var parents: [512]i32 = undefined;
    var steps: [parents.len]Step = undefined;
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast(row - 1);
    try std.testing.expectEqual(@as(usize, 1), try build(std.testing.allocator, &parents, &steps, 0, 1));
    try validate(&parents, &steps, 0, 1);
    @memset(parents[1..], 0);
    try std.testing.expectError(error.SlotOverflow, build(std.testing.allocator, &parents, &steps, 0, 1));
    try std.testing.expectEqual(@as(usize, 2), try build(std.testing.allocator, &parents, &steps, 0, 2));
    try validate(&parents, &steps, 0, 2);
}

test "balanced trees reuse ancestor slots and an explicit slot budget refuses overflow" {
    var parents: [127]i32 = undefined;
    var steps: [parents.len]Step = undefined;
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast((row - 1) / 2);
    try std.testing.expectEqual(@as(usize, 7), try build(std.testing.allocator, &parents, &steps, 0, 16));
    try validate(&parents, &steps, 0, 16);
    try std.testing.expectError(error.SlotOverflow, build(std.testing.allocator, &parents, &steps, 0, 6));
}

test "forests reset to committed state and bad parents never make a schedule" {
    const parents = [_]i32{ -1, 0, -1, 2, 2, -1 };
    var steps: [parents.len]Step = undefined;
    _ = try build(std.testing.allocator, &parents, &steps, 73, 16);
    try validate(&parents, &steps, 73, 16);
    for ([_][]const i32{ &.{ -1, 1 }, &.{ -2, 0 }, &.{ -1, 2 } }) |bad| {
        try std.testing.expectError(error.BadSlotParents, build(std.testing.allocator, bad, steps[0..bad.len], 0, 16));
    }
}

test "irregular trees compute every row from the same parent as an independent topological replay" {
    var random: u64 = 1729;
    for ([_]usize{ 17, 32, 64, 128 }) |width| {
        var parents: [128]i32 = undefined;
        var steps: [128]Step = undefined;
        var expected: [128]u64 = undefined;
        var actual: [128]u64 = undefined;
        for (0..32) |_| {
            for (parents[0..width], 0..) |*parent, row| {
                random = random *% 6364136223846793005 +% 1442695040888963407;
                parent.* = if (row == 0) -1 else @intCast(random % row);
                const value = if (parent.* < 0) @as(u64, 23) else expected[@intCast(parent.*)];
                expected[row] = value *% 65537 +% row +% 1;
            }
            _ = try build(std.testing.allocator, parents[0..width], steps[0..width], 11, 16);
            try validate(parents[0..width], steps[0..width], 11, 16);
            var registers: [max_slots]u64 = @splat(0);
            for (steps[0..width]) |step| {
                const row = step.row - 11;
                const value = if (step.read_slot < 0) @as(u64, 23) else registers[@intCast(step.read_slot)];
                actual[row] = value *% 65537 +% row +% 1;
                registers[@intCast(step.write_slot)] = actual[row];
            }
            try std.testing.expectEqualSlices(u64, expected[0..width], actual[0..width]);
        }
    }
}

test "output-only star scans retain one parent register and discard every unread leaf state" {
    var parents: [512]i32 = @splat(0);
    parents[0] = -1;
    var steps: [parents.len]Step = undefined;
    try std.testing.expectEqual(@as(usize, 1), try buildForOutputs(std.testing.allocator, &parents, &steps, 17, 1));
    try validateForOutputs(&parents, &steps, 17, 1);
    try std.testing.expectEqual(@as(i32, 0), steps[0].write_slot);
    for (steps[1..]) |step| {
        try std.testing.expectEqual(@as(i32, 0), step.read_slot);
        try std.testing.expectEqual(@as(i32, -1), step.write_slot);
    }
    try std.testing.expectError(error.BadSlotPlan, validate(&parents, &steps, 17, 1));
    try std.testing.expectError(error.SlotOverflow, build(std.testing.allocator, &parents, &steps, 17, 1));
}

test "output-only singleton forests and chains keep the existing nonempty kernel minimum" {
    var parents: [64]i32 = @splat(-1);
    var steps: [parents.len]Step = undefined;
    try std.testing.expectEqual(@as(usize, 1), try buildForOutputs(std.testing.allocator, &parents, &steps, 0, 1));
    try validateForOutputs(&parents, &steps, 0, 1);
    for (steps) |step| try std.testing.expectEqual(@as(i32, -1), step.write_slot);
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast(row - 1);
    try std.testing.expectEqual(@as(usize, 1), try buildForOutputs(std.testing.allocator, &parents, &steps, 0, 1));
    try validateForOutputs(&parents, &steps, 0, 1);
    try std.testing.expectEqual(@as(i32, -1), steps[steps.len - 1].write_slot);
}

test "output-only balanced plans lower the exact slot bound without changing visit order" {
    var parents: [127]i32 = undefined;
    var kept: [parents.len]Step = undefined;
    var transient: [parents.len]Step = undefined;
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast((row - 1) / 2);
    try std.testing.expectEqual(@as(usize, 7), try build(std.testing.allocator, &parents, &kept, 0, 16));
    try std.testing.expectEqual(@as(usize, 6), try buildForOutputs(std.testing.allocator, &parents, &transient, 0, 6));
    try validateForOutputs(&parents, &transient, 0, 6);
    for (kept, transient) |old, replacement| {
        try std.testing.expectEqual(old.row, replacement.row);
        try std.testing.expectEqual(old.read_slot, replacement.read_slot);
        if (replacement.write_slot >= 0) try std.testing.expectEqual(old.write_slot, replacement.write_slot);
    }
    try std.testing.expectError(error.SlotOverflow, buildForOutputs(std.testing.allocator, &parents, &transient, 0, 5));
}

test "output-only leaf omission preserves every row against an independent parent-state oracle" {
    var seed: u64 = 8191;
    for ([_]usize{ 1, 2, 3, 5, 16, 33, 128 }) |count| for (0..32) |_| {
        var parents: [128]i32 = undefined;
        var steps: [128]Step = undefined;
        var expected: [128]u64 = undefined;
        var actual: [128]u64 = undefined;
        for (parents[0..count], 0..) |*parent, row| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            parent.* = if (row % 11 == 0) -1 else @intCast(seed % row);
            const base = if (parent.* < 0) @as(u64, 23) else expected[@intCast(parent.*)];
            expected[row] = base *% 65537 +% row +% 1;
        }
        _ = try buildForOutputs(std.testing.allocator, parents[0..count], steps[0..count], 11, 16);
        try validateForOutputs(parents[0..count], steps[0..count], 11, 16);
        var registers: [max_slots]u64 = @splat(0);
        for (steps[0..count]) |step| {
            const row = step.row - 11;
            const base = if (step.read_slot < 0) @as(u64, 23) else registers[@intCast(step.read_slot)];
            actual[row] = base *% 65537 +% row +% 1;
            if (step.write_slot >= 0) registers[@intCast(step.write_slot)] = actual[row];
        }
        try std.testing.expectEqualSlices(u64, expected[0..count], actual[0..count]);
    };
}

test "output-only validation refuses omitted internal state and live-parent clobbering" {
    const parents = [_]i32{ -1, 0, 0, 0, 0 };
    var steps: [parents.len]Step = undefined;
    _ = try buildForOutputs(std.testing.allocator, &parents, &steps, 0, 1);
    const original = steps;
    steps[0].write_slot = -1;
    try std.testing.expectError(error.BadSlotPlan, validateForOutputs(&parents, &steps, 0, 1));
    steps = original;
    steps[1].write_slot = 0;
    try std.testing.expectError(error.BadSlotPlan, validateForOutputs(&parents, &steps, 0, 1));
    steps = original;
    steps[steps.len - 1].write_slot = -2;
    try std.testing.expectError(error.BadSlotPlan, validateForOutputs(&parents, &steps, 0, 1));
}
