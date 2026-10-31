//! QKV projects straight into the pool's log and ZBA into the frame's mixed buffer, side by side from the same input.
//! A prompt pass on the reg path runs only b and a (the prepare's) beside QKV: z, which only the gate reads, runs beside
//! the scan, whose sequential rows leave the GPU's tensor units idle.
const metal = @import("metal");
const Pool = @import("gdn_pool_gpu.zig").Pool;
const pool_gpu = @import("gdn_pool_gpu.zig");
const Plan = @import("gdn_plan.zig").Plan;
const Frame = @import("gpu_frame.zig").Frame;
const Weights = @import("gpu_weights.zig");
const Quant = @import("quant_gpu.zig").Kernels;
const gdn = @import("gdn.zig");
const Linear = @import("projection.zig").Linear;
const Glue = @import("glue_gpu.zig").Kernels;

pub const Hook = struct {
    pool: *Pool,
    quant: *Quant,
    glue: Glue,
    plan: *const Plan,
    linear_index: []const u32,
    accepted_chain: bool = false,

    pub fn forward(ptr: *anyopaque, e: metal.ComputeEncoder, block: usize, weight: Weights.Gdn, frame: *Frame, rows: u32) !void {
        const hook: *Hook = @ptrCast(@alignCast(ptr));
        if (block >= hook.linear_index.len or hook.linear_index[block] >= hook.pool.budget.layers or rows != hook.plan.parents.len) return error.BadGdnHook;
        const layer = hook.linear_index[block];
        const sums = @import("forward.zig").sums(frame, .sums, rows);
        try hook.quant.quant(e, weight.qkv, frame.get(.input), sums, frame.get(.dims), try hook.pool.projected(layer), rows);
        var z = Late{ .quant = hook.quant, .zba = weight.zba, .frame = frame, .sums = sums, .rows = rows, .tiles = zTiles(weight.zba, frame, hook.quant.prompt) };
        if (z.tiles > 0) try hook.quant.quantRange(e, weight.zba, frame.get(.input), sums, frame.get(.dims), frame.get(.mixed), rows, .{ z.tiles, weight.zba.n / 32 - z.tiles }) else try hook.quant.quant(e, weight.zba, frame.get(.input), sums, frame.get(.dims), frame.get(.mixed), rows);
        e.barrier();
        const beside: ?gdn.Beside = if (z.tiles > 0) .{ .ptr = &z, .encode = Late.encode } else null;
        // the gate writes the out projection's group sums into `.sums` once qkv and zba have read the norm's
        var gated = try refs(weight, frame);
        const xs = frame.get(.sums);
        frame.mixer_sums = false;
        gated.sums = .{ .span = .{ .buffer = xs.buffer, .offset = xs.offset, .bytes = xs.buffer.length() - xs.offset }, .stride = sums.stride, .written = &frame.mixer_sums };
        if (hook.accepted_chain) try hook.pool.forwardChain(layer, devicePlan(hook.plan.*, frame), gated, e, hook.quant.prompt, beside) else try hook.pool.forward(layer, devicePlan(hook.plan.*, frame), gated, e, beside);
    }

    /// The layer's prompt chain split at row `first` (forward.Stages.split_first), its gate writing the out projection's
    /// group sums as `forward`'s does; null where the layer takes `forward` whole.
    pub fn halves(hook: *Hook, block: usize, weight: Weights.Gdn, frame: *Frame, rows: u32, first: u32) !?pool_gpu.Halves {
        if (block >= hook.linear_index.len or hook.linear_index[block] >= hook.pool.budget.layers or rows != hook.plan.parents.len) return error.BadGdnHook;
        if (!hook.accepted_chain or !hook.quant.prompt) return null;
        var gated = try refs(weight, frame);
        if (weight.zba.n != gated.zba_stride) return null;
        const xs = frame.get(.sums);
        frame.mixer_sums = false;
        gated.sums = .{ .span = .{ .buffer = xs.buffer, .offset = xs.offset, .bytes = xs.buffer.length() - xs.offset }, .stride = @import("forward.zig").sums(frame, .sums, rows).stride, .written = &frame.mixer_sums };
        return hook.pool.chainHalves(hook.linear_index[block], devicePlan(hook.plan.*, frame), gated, first);
    }

    /// `forward`'s qkv and zba projections over rows [first, first + count) alone, without a barrier.
    pub fn project(hook: *Hook, e: metal.ComputeEncoder, block: usize, weight: Weights.Gdn, frame: *Frame, rows: u32, first: u32, count: u32) !void {
        if (block >= hook.linear_index.len or hook.linear_index[block] >= hook.pool.budget.layers or first + count > rows) return error.BadGdnHook;
        var sums = @import("forward.zig").sums(frame, .sums, rows);
        sums.ref.offset += @as(usize, first) * 4;
        var input = frame.get(.input);
        input.offset += @as(usize, first) * weight.qkv.k * 2;
        var qkv = try hook.pool.projected(hook.linear_index[block]);
        qkv.offset += @as(usize, first) * weight.qkv.n * 2;
        var zba = frame.get(.mixed);
        zba.offset += @as(usize, first) * weight.zba.n * 2;
        try hook.quant.quant(e, weight.qkv, input, sums, frame.get(.dims), qkv, count);
        try hook.quant.quant(e, weight.zba, input, sums, frame.get(.dims), zba, count);
    }

    pub fn commit(hook: *Hook, e: metal.ComputeEncoder, block: usize, frame: *Frame) !void {
        if (block >= hook.linear_index.len or hook.linear_index[block] >= hook.pool.budget.layers) return error.BadGdnHook;
        try hook.pool.commit(hook.linear_index[block], devicePlan(hook.plan.*, frame), e);
    }
};

/// The z tiles of the fused zba rows, encoded beside the scan: they read the norm's rows and sums, which nothing writes
/// before the gate, and write only z's columns, which only the gate reads.
const Late = struct {
    quant: *Quant,
    zba: Linear,
    frame: *Frame,
    sums: @import("quant_gpu.zig").Sums,
    rows: u32,
    tiles: u32,
    fn encode(ptr: *anyopaque, e: metal.ComputeEncoder) anyerror!void {
        const z: *Late = @ptrCast(@alignCast(ptr));
        try z.quant.quantRange(e, z.zba, z.frame.get(.input), z.sums, z.frame.get(.dims), z.frame.get(.mixed), z.rows, .{ 0, z.tiles });
    }
};

/// z's 32-column tiles for a prompt pass whose zba takes ranged launches, z ending on a tile boundary before b and a; 0
/// keeps one launch (decoded rows: their short scan already keeps the GPU busy).
fn zTiles(zba: Linear, frame: *Frame, prompt: bool) u32 {
    const width: u32 = @intCast(frame.config.gdnValueDim());
    if (!prompt or !Quant.ranged(zba) or width % 32 != 0 or zba.n % 32 != 0 or zba.n <= width) return 0;
    return width / 32;
}

/// z, b and a read in place from the fused zba rows.
fn refs(weight: Weights.Gdn, frame: *Frame) !pool_gpu.Refs {
    const width = frame.config.gdnValueDim();
    const heads = frame.config.v_heads;
    const mixed = frame.get(.mixed);
    return .{ .z = mixed, .b = .{ .buffer = mixed.buffer, .offset = mixed.offset + width * 2 }, .a = .{ .buffer = mixed.buffer, .offset = mixed.offset + (width + heads) * 2 }, .output = frame.get(.mixer_out), .scalars = try pool_gpu.Scalars.from(weight), .zba_stride = @intCast(width + 2 * heads) };
}

fn devicePlan(plan: Plan, frame: *Frame) pool_gpu.DevicePlan {
    return .{ .host = plan, .metadata = .{ .parents = frame.get(.parents), .windows = frame.get(.windows), .row_slots = frame.get(.row_slots), .segments = frame.get(.segments), .keeps = frame.get(.keeps), .kept_rows = frame.get(.kept_rows), .slot_steps = frame.get(.slot_steps) } };
}
