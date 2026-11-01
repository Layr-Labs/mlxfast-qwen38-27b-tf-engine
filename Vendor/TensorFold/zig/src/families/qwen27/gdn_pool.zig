//! State persists per stream and layer; round logs persist per layer while one snapshot slab is reused.
const std = @import("std");
const abi = @import("gdn_contract.zig");
const Plan = @import("gdn_plan.zig").Plan;

pub fn validatePlan(shape: abi.Shape, slots: usize, capacity: usize, plan: Plan) !void {
    const rows = plan.parents.len;
    const taps = shape.c.conv_kernel;
    if (rows == 0 or rows > capacity or slots == 0 or slots > 128 or plan.windows.len != rows * taps or plan.row_slots.len != rows or plan.segments.len == 0 or plan.segments.len > slots or plan.keeps.len != plan.segments.len) return error.BadGdnPoolPlan;
    if (plan.compact and plan.slot_steps.len != rows) return error.BadGdnPoolPlan;
    var first: usize = 0;
    var kept: usize = 0;
    var destinations: u128 = 0;
    for (plan.segments, plan.keeps) |segment, keep| {
        if (segment.first != first or segment.rows == 0 or segment.rows > rows - first or segment.state_slot >= slots or segment.next_slot >= slots or keep.state_slot != segment.state_slot or keep.next_slot != segment.next_slot or keep.first != kept or keep.rows > segment.rows or keep.rows > plan.kept_rows.len - kept) return error.BadGdnPoolPlan;
        const bit = @as(u128, 1) << @intCast(segment.next_slot);
        if (destinations & bit != 0) return error.BadGdnPoolPlan;
        destinations |= bit;
        for (0..segment.rows) |local| {
            const row = first + local;
            const parent = plan.parents[row];
            if ((local == 0 and parent != -1) or (local > 0 and (parent < 0 or parent >= local)) or plan.row_slots[row] != segment.state_slot) return error.BadGdnPoolPlan;
            for (plan.windows[row * taps ..][0..taps], 0..) |source, tap| {
                const behind = taps - 1 - tap;
                var ancestor: i32 = @intCast(local);
                var hops: usize = 0;
                while (hops < behind and ancestor >= 0) : (hops += 1) ancestor = plan.parents[first + @as(usize, @intCast(ancestor))];
                const expected: usize = if (ancestor >= 0) taps - 1 + first + @as(usize, @intCast(ancestor)) else taps - 1 - (behind - hops + 1);
                if (source != expected) return error.BadGdnPoolPlan;
            }
        }
        var parent: i32 = -1;
        for (plan.kept_rows[kept..][0..keep.rows]) |global| {
            if (global < first or global >= first + segment.rows or plan.parents[global] != parent) return error.BadGdnPoolPlan;
            parent = @intCast(global - first);
        }
        if (plan.compact) @import("core").tree_slots.validate(plan.parents[first..][0..segment.rows], plan.slot_steps[first..][0..segment.rows], segment.first, 16) catch return error.BadGdnPoolPlan;
        first += segment.rows;
        kept += keep.rows;
    }
    if (first != rows or kept != plan.kept_rows.len) return error.BadGdnPoolPlan;
}

/// Depth-one star rounds whose keeps defer their GDN replays (gdn_pool_gpu.Pool.stash): every `defer_rounds`-th star scan
/// writes the state the deferred rows leave back over the committed one.
pub const defer_rounds = 4;
/// Rows a depth-one star keeps at most: its root and one child.
pub const star_kept = 2;
/// Pending rows a layer holds: `defer_rounds` keeps of a depth-one star.
pub const pending_rows = star_kept * defer_rounds;

/// Only one-row windows advanced by a verify need rollback words, indexed by their original slot.
pub fn beforeSlots(plan: Plan, advance: bool) usize {
    if (!advance) return 0;
    var slots: usize = 0;
    for (plan.segments) |segment| if (segment.rows == 1) {
        slots = @max(slots, @as(usize, segment.state_slot) + 1);
    };
    return slots;
}

pub const Budget = struct {
    shape: abi.Shape,
    layers: usize,
    slots: usize,
    rows: usize,
    state: usize,
    history: usize,
    projections: usize,
    query: usize,
    key: usize,
    value: usize,
    decay: usize,
    mixing: usize,
    snapshots: usize,
    conv_tails: usize,
    scratch: usize,
    /// The pending rows' key, value, decay and mixing words, `pending_rows` a layer (gdn_pool_gpu.Pool.stash).
    pending: [4]usize,

    pub fn init(shape: abi.Shape, layers: usize, slots: usize, rows: usize) !Budget {
        if (layers == 0 or layers > shape.c.layers or slots == 0 or slots > 128 or rows == 0 or rows > @import("gpu_frame.zig").max_rows) return error.BadGdnPool;
        const layer_slots = try std.math.mul(usize, layers, slots);
        const layer_rows = try std.math.mul(usize, layers, rows);
        const query = try abi.countBytes(u16, shape.c.k_heads * shape.c.dk, layer_rows);
        const value = try abi.countBytes(u16, shape.value, layer_rows);
        const decay = try abi.countBytes(f32, shape.c.v_heads, layer_rows);
        const mixing = try abi.countBytes(u16, shape.c.v_heads, layer_rows);
        const snapshot = try abi.countBytes(f32, shape.state, @min(rows, abi.snapshot_rows)); // trees only: a chain keeps no snapshots
        const tails = try abi.countBytes(u16, shape.conv, rows);
        const outputs = try abi.countBytes(u16, shape.value, rows * 2);
        const pending = try std.math.mul(usize, layers, pending_rows);
        return .{
            .shape = shape,
            .layers = layers,
            .slots = slots,
            .rows = rows,
            .state = try abi.countBytes(f32, shape.state, layer_slots),
            .history = try abi.countBytes(u16, shape.conv, layer_slots),
            .projections = try abi.countBytes(u16, shape.qkv, layer_rows),
            .query = query,
            .key = query,
            .value = value,
            .decay = decay,
            .mixing = mixing,
            .snapshots = snapshot,
            .conv_tails = tails,
            .scratch = try std.math.add(usize, try std.math.add(usize, snapshot, tails), outputs),
            .pending = .{ try abi.countBytes(u16, shape.c.k_heads * shape.c.dk, pending), try abi.countBytes(u16, shape.value, pending), try abi.countBytes(f32, shape.c.v_heads, pending), try abi.countBytes(u16, shape.c.v_heads, pending) },
        };
    }

    pub fn allocationSizes(b: Budget) [16]usize {
        return .{ b.state, b.history, b.history, b.projections, b.query, b.key, b.value, b.decay, b.mixing, b.snapshots, b.conv_tails, b.rows * b.shape.value * 2, b.pending[0], b.pending[1], b.pending[2], b.pending[3] };
    }

    pub fn stateOffset(b: Budget, layer: usize, slot: usize) !usize {
        if (layer >= b.layers or slot >= b.slots) return error.BadGdnPoolIndex;
        return abi.countBytes(f32, b.shape.state, layer * b.slots + slot);
    }

    pub fn historyOffset(b: Budget, layer: usize, slot: usize) !usize {
        if (layer >= b.layers or slot >= b.slots) return error.BadGdnPoolIndex;
        return abi.countBytes(u16, b.shape.conv, layer * b.slots + slot);
    }

    pub fn logBytes(b: Budget) usize {
        return b.query + b.key + b.value + b.decay + b.mixing;
    }

    pub fn totalBytes(b: Budget) usize {
        return 2 * b.state + 2 * b.history + b.projections + b.logBytes() + b.scratch + b.pending[0] + b.pending[1] + b.pending[2] + b.pending[3];
    }
};
