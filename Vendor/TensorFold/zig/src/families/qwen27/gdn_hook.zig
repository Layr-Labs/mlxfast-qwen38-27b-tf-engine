//! QKV projects straight into the pool's log and ZBA into the frame's mixed buffer, side by side from the same input.
//! A prompt pass on the reg path runs only b and a (the prepare's) beside QKV: z, which only the gate reads, runs beside
//! the scan, whose sequential rows leave the GPU's tensor units idle.
const metal = @import("metal");
const Pool = @import("gdn_pool_gpu.zig").Pool;
const pool_gpu = @import("gdn_pool_gpu.zig");
const Plan = @import("gdn_plan.zig").Plan;
const View = @import("gpu_frame.zig").View;
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
    /// Rows of the pool's row scratch before this pass's rows (a split chunk's later segment).
    row: u32 = 0,

    pub fn forward(ptr: *anyopaque, e: metal.ComputeEncoder, block: usize, weight: Weights.Gdn, view: *View, rows: u32) !void {
        const hook: *Hook = @ptrCast(@alignCast(ptr));
        if (block >= hook.linear_index.len or hook.linear_index[block] >= hook.pool.budget.layers or rows != hook.plan.parents.len) return error.BadGdnHook;
        const layer = hook.linear_index[block];
        const sums = @import("forward.zig").sums(view, .sums, rows);
        try hook.quant.quant(e, weight.qkv, view.get(.input), sums, view.get(.dims), try hook.pool.projected(layer), rows);
        var z = Late{ .quant = hook.quant, .zba = weight.zba, .view = view, .sums = sums, .rows = rows, .tiles = zTiles(weight.zba, view, hook.quant.prompt) };
        if (z.tiles > 0) try hook.quant.quantRange(e, weight.zba, view.get(.input), sums, view.get(.dims), view.get(.mixed), rows, .{ z.tiles, weight.zba.n / 32 - z.tiles }) else try hook.quant.quant(e, weight.zba, view.get(.input), sums, view.get(.dims), view.get(.mixed), rows);
        e.barrier();
        const beside: ?gdn.Beside = if (z.tiles > 0) .{ .ptr = &z, .encode = Late.encode } else null;
        // the gate writes the out projection's group sums into `.sums` once qkv and zba have read the norm's
        var gated = try refs(weight, view);
        const xs = view.get(.sums);
        view.mixer_sums = false;
        gated.sums = .{ .span = .{ .buffer = xs.buffer, .offset = xs.offset, .bytes = xs.buffer.length() - xs.offset }, .stride = sums.stride, .written = &view.mixer_sums };
        gated.row = hook.row;
        if (hook.accepted_chain) try hook.pool.forwardChain(layer, devicePlan(hook.plan.*, view), gated, e, hook.quant.prompt, beside) else try hook.pool.forward(layer, devicePlan(hook.plan.*, view), gated, e, beside);
    }

    pub fn commit(hook: *Hook, e: metal.ComputeEncoder, block: usize, view: *View) !void {
        if (block >= hook.linear_index.len or hook.linear_index[block] >= hook.pool.budget.layers) return error.BadGdnHook;
        try hook.pool.commit(hook.linear_index[block], devicePlan(hook.plan.*, view), e);
    }
};

/// The z tiles of the fused zba rows, encoded beside the scan: they read the norm's rows and sums, which nothing writes
/// before the gate, and write only z's columns, which only the gate reads.
const Late = struct {
    quant: *Quant,
    zba: Linear,
    view: *View,
    sums: @import("quant_gpu.zig").Sums,
    rows: u32,
    tiles: u32,
    fn encode(ptr: *anyopaque, e: metal.ComputeEncoder) anyerror!void {
        const z: *Late = @ptrCast(@alignCast(ptr));
        try z.quant.quantRange(e, z.zba, z.view.get(.input), z.sums, z.view.get(.dims), z.view.get(.mixed), z.rows, .{ 0, z.tiles });
    }
};

/// z's 32-column tiles for a prompt pass whose zba takes ranged launches, z ending on a tile boundary before b and a; 0
/// keeps one launch (decoded rows: their short scan already keeps the GPU busy).
fn zTiles(zba: Linear, view: *const View, prompt: bool) u32 {
    const width: u32 = @intCast(view.frame.config.gdnValueDim());
    if (!prompt or !Quant.ranged(zba) or width % 32 != 0 or zba.n % 32 != 0 or zba.n <= width) return 0;
    return width / 32;
}

/// z, b and a read in place from the fused zba rows.
fn refs(weight: Weights.Gdn, view: *const View) !pool_gpu.Refs {
    const width = view.frame.config.gdnValueDim();
    const heads = view.frame.config.v_heads;
    const mixed = view.get(.mixed);
    return .{ .z = mixed, .b = .{ .buffer = mixed.buffer, .offset = mixed.offset + width * 2 }, .a = .{ .buffer = mixed.buffer, .offset = mixed.offset + (width + heads) * 2 }, .output = view.get(.mixer_out), .scalars = try pool_gpu.Scalars.from(weight), .zba_stride = @intCast(width + 2 * heads) };
}

fn devicePlan(plan: Plan, view: *const View) pool_gpu.DevicePlan {
    return .{ .host = plan, .metadata = .{ .parents = view.get(.parents), .windows = view.get(.windows), .row_slots = view.get(.row_slots), .segments = view.get(.segments), .keeps = view.get(.keeps), .kept_rows = view.get(.kept_rows), .slot_steps = view.get(.slot_steps) } };
}
