//! The Metal graph borrows native scalar tensors and owns affine layouts and optional dense coefficients.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("core").checkpoint_metal;
const Config = @import("config.zig").Config;
const projection = @import("projection.zig");
const lane = @import("core").lane_projection;
const widths = @import("widths.zig");
const Width = widths.Width;
const Ref = projection.Ref;
const Linear = projection.Linear;
const dense_prompt_gu = @import("dense_prompt_gu.zig");
pub const Tensor = cp.Tensor;

/// Auxiliary coefficients use a bounded share of the device's advertised working set.
pub fn defaultAuxBudget(max_working_set: u64) usize {
    return if (max_working_set >= @as(u64, 64) << 30) @as(usize, 8) << 30 else @as(usize, 7) << 28;
}

pub const LoadOptions = struct {
    dense_prompt_gu: bool = false,
    max_aux_bytes: usize = 7 << 28,
};

const dense_n: usize = 34816;
const dense_k: usize = 5120;
// CSR keeps one dummy u16 index and f32 residual even when its count is zero.
const dense_min_bytes: usize = dense_n * dense_k * @sizeOf(u16) + (dense_n + 1) * @sizeOf(u32) + @sizeOf(u16) + @sizeOf(f32);
const DenseBudget = struct {
    limit: usize,
    used: usize = 0,

    fn remaining(b: DenseBudget) usize {
        return b.limit -| b.used;
    }
};

fn denseLayerEligible(layer: usize, total: usize, hidden: usize, intermediate: usize, gate: Width, up: Width) bool {
    return layer < total -| 1 and hidden == dense_k and intermediate == dense_n / 2 and
        widths.isUniform(gate) and widths.isUniform(up);
}

/// Keep the last minimum-budget-eligible layers; callers mark terminal/mixed layers false first.
fn selectDenseLayers(eligible: []bool, max_aux_bytes: usize) void {
    var count = max_aux_bytes / dense_min_bytes;
    var i = eligible.len;
    while (i > 0) {
        i -= 1;
        const take = eligible[i] and count > 0;
        eligible[i] = take;
        if (take) count -= 1;
    }
}

/// All size/budget checks precede the four infallible ownership transfers.
fn admitDense(owned: anytype, budget: *DenseBudget, w: anytype) !bool {
    const buffers = .{ w.high, w.offsets, w.indices, w.residual };
    var actual: usize = 0;
    inline for (buffers) |buffer| actual = try std.math.add(usize, actual, buffer.length());
    if (actual > budget.remaining()) return false;
    std.debug.assert(owned.capacity - owned.items.len >= 4);
    inline for (buffers) |buffer| owned.appendAssumeCapacity(buffer);
    budget.used += actual;
    return true;
}

pub const Gdn = struct { qkv: Linear, zba: Linear, out: Linear, conv: Tensor, a_log: Tensor, dt: Tensor, norm: Tensor };
pub const Attention = struct { q: Linear, kv: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Layer = struct { norm: Tensor, post_norm: Tensor, gu: Linear, down: Linear, mixer: union(@import("config.zig").Kind) { linear: Gdn, attention: Attention } };

pub const Weights = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    embed: projection.Triple,
    norm: Tensor,
    head: Linear,
    /// The same matrix with eight adjacent tiles interleaved: a decoded round's head projection reads this one, and
    /// a prompt's last row (and the drafter's span slice) the unbatched `head`.
    head_decode: Linear,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    runs: std.ArrayList([]projection.Run) = .empty,
    /// These buffers are already included in `owned` and its residency/memory inventory.
    dense_prompt_gu_bytes: usize = 0,

    pub fn hasDensePromptGu(w: *const Weights) bool {
        for (w.layers) |l| if (l.gu.prompt_dense_gu != null) return true;
        return false;
    }

    pub fn deinit(w: *Weights) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(w.allocator);
        for (w.runs.items) |r| w.allocator.free(r);
        w.runs.deinit(w.allocator);
        w.allocator.free(w.layers);
    }

    /// Whether every MLP projection is 4-bit g64 in one matrix (the Neural Engine share's format).
    pub fn uniformMlp(w: *const Weights) bool {
        for (w.layers) |l| for ([_]Linear{ l.gu, l.down }) |p| if (!uniform(p)) return false;
        return true;
    }
};

fn uniform(p: Linear) bool {
    return p.bits == 4 and p.group == 64 and p.runs.len == 0;
}

/// Modules the graph reads only through repacked copies: their tensors stay in the files until load stages them.
const repacked_modules = [_][]const u8{ ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.in_proj_b", ".linear_attn.in_proj_a", ".linear_attn.out_proj", ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj" };

/// Whether `name` is a weight, scales or biases tensor of a repacked module or the LM head.
pub fn repacked(name: []const u8, _: usize) bool {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return false;
    const leaf = name[dot + 1 ..];
    if (!std.mem.eql(u8, leaf, "weight") and !std.mem.eql(u8, leaf, "scales") and !std.mem.eql(u8, leaf, "biases")) return false;
    const module = name[0..dot];
    if (std.mem.eql(u8, module, "lm_head") or std.mem.endsWith(u8, module, ".lm_head")) return true;
    for (repacked_modules) |m| if (std.mem.endsWith(u8, module, m)) return true;
    return false;
}

const Builder = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    checkpoint: *const cp.Checkpoint,
    prefix: []const u8,
    owned: *std.ArrayList(mtl.Buffer),
    stage: *cp.Stage,
    runs: *std.ArrayList([]projection.Run),
    widths: *const widths.Widths,
    dense_layers: []const bool = &.{},
    dense_budget: ?*DenseBudget = null,

    fn get(b: Builder, suffix: []const u8) !Tensor {
        const name = try std.mem.concat(b.allocator, u8, &.{ b.prefix, suffix });
        defer b.allocator.free(name);
        return b.checkpoint.get(name);
    }

    fn tensor(b: Builder, suffix: []const u8, dtype: cp.DType, shape: []const usize) !Tensor {
        return checked(try b.get(suffix), dtype, shape);
    }

    fn quant(b: Builder, module: []const u8, n: usize, k: usize, w: Width) !projection.Triple {
        if (k * w.bits % 32 != 0 or k % w.group != 0) return error.BadNativeTensor;
        const suffixes = [_][]const u8{ ".weight", ".scales", ".biases" };
        var tensors: [3]Tensor = undefined;
        for (suffixes, &tensors, 0..) |suffix, *t, i| {
            const name = try std.mem.concat(b.allocator, u8, &.{ module, suffix });
            defer b.allocator.free(name);
            t.* = try b.tensor(name, if (i == 0) .u32 else .bf16, &.{ n, if (i == 0) k * w.bits / 32 else k / w.group });
        }
        return .{ .weight = tensors[0], .scales = tensors[1], .biases = tensors[2], .bits = w.bits, .group = w.group };
    }

    /// `parts` (stacked in order, `n` rows of `k` each, at their widths) staged from the files and repacked.
    fn linear(b: Builder, parts: []const Part, k: usize) !Linear {
        return b.linearForced(parts, k, null);
    }

    /// The same matrix prepared with `forced` adjacent tiles interleaved (see projection.prepareForced).
    fn linearForced(b: Builder, parts: []const Part, k: usize, forced: ?u32) !Linear {
        return b.linearWithDense(parts, k, forced, null);
    }

    fn linearGu(b: Builder, parts: []const Part, k: usize, layer_index: usize) !Linear {
        const selected = layer_index < b.dense_layers.len and b.dense_layers[layer_index];
        return b.linearWithDense(parts, k, null, if (selected) @as(u32, @intCast(layer_index)) else null);
    }

    fn linearWithDense(b: Builder, parts: []const Part, k: usize, forced: ?u32, dense_layer: ?u32) !Linear {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const names = try arena.allocator().alloc([]const u8, parts.len * 3);
        for (parts, 0..) |p, i| for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suffix, j| {
            names[i * 3 + j] = try std.mem.concat(arena.allocator(), u8, &.{ b.prefix, p.module, suffix });
        };
        try b.stage.reserve(b.checkpoint, names);
        defer b.stage.reset();
        var triples: [3]projection.Triple = undefined;
        if (parts.len > triples.len) return error.BadProjectionShape;
        for (parts, triples[0..parts.len], 0..) |p, *t, i| {
            if (k * p.w.bits % 32 != 0 or k % p.w.group != 0) return error.BadNativeTensor;
            t.* = .{
                .weight = try checked(try b.stage.read(b.checkpoint, names[i * 3]), .u32, &.{ p.n, k * p.w.bits / 32 }),
                .scales = try checked(try b.stage.read(b.checkpoint, names[i * 3 + 1]), .bf16, &.{ p.n, k / p.w.group }),
                .biases = try checked(try b.stage.read(b.checkpoint, names[i * 3 + 2]), .bf16, &.{ p.n, k / p.w.group }),
                .bits = p.w.bits,
                .group = p.w.group,
            };
        }
        var result = try b.stackForced(triples[0..parts.len], forced);
        if (dense_layer) |layer_index| {
            const budget = b.dense_budget orelse return error.MissingDenseBudget;
            if (budget.remaining() < dense_min_bytes) return result;
            // Capacity is secured before prepare owns any of its four buffers.
            try b.owned.ensureUnusedCapacity(b.allocator, 4);
            const weights = try dense_prompt_gu.prepare(b.allocator, b.device, triples[0..parts.len], layer_index);
            var transferred = false;
            defer if (!transferred) dense_prompt_gu.deinitWeights(weights);
            const before = budget.used;
            if (try admitDense(b.owned, budget, weights)) {
                result.prompt_dense_gu = weights;
                transferred = true;
                std.log.info("qwen27: dense prompt GU layer={d} aux_bytes={d} residual_count={d}", .{ layer_index, budget.used - before, weights.count });
            } else {
                std.log.info("qwen27: dense prompt GU layer={d} rejected by remaining auxiliary budget", .{layer_index});
            }
        }
        return result;
    }

    /// Fused members of one width take one matrix as before; a mixed stack becomes one matrix a member.
    fn stack(b: Builder, parts: []const projection.Triple) !Linear {
        return b.stackForced(parts, null);
    }

    fn stackForced(b: Builder, parts: []const projection.Triple, forced: ?u32) !Linear {
        var ws: [3]Width = undefined;
        for (parts, 0..) |p, i| ws[i] = .{ .bits = p.bits, .group = @intCast(p.group) };
        var buf: [3][2]usize = undefined;
        const list = widths.runs(ws[0..parts.len], &buf);
        if (list.len == 1) return projection.prepareForced(b.device, b.allocator, b.owned, parts, 32, forced);
        const runs = try b.allocator.alloc(projection.Run, list.len);
        b.runs.append(b.allocator, runs) catch |err| {
            b.allocator.free(runs);
            return err;
        };
        var first: u32 = 0;
        for (list, runs) |r, *run| {
            run.* = .{ .linear = try projection.prepareForced(b.device, b.allocator, b.owned, parts[r[0]..][0..r[1]], 32, forced), .first = first };
            first += run.linear.n;
        }
        var fused = runs[0].linear;
        fused.n = first;
        fused.raw = null;
        fused.runs = runs;
        return fused;
    }

    fn layer(b: Builder, c: Config, i: usize) !Layer {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        const Name = struct {
            fn full(allocator: std.mem.Allocator, p: []const u8, leaf: []const u8) ![]const u8 {
                return std.mem.concat(allocator, u8, &.{ p, leaf });
            }
        };
        const w = b.widths.layers[i];
        const M = widths.Member;
        const mixer: @FieldType(Layer, "mixer") = switch (c.kind(i)) {
            .linear => .{ .linear = .{
                .qkv = try b.linear(&.{.{ .module = try Name.full(a, prefix, "linear_attn.in_proj_qkv"), .n = c.gdnQkvDim(), .w = w[@backingInt(M.qkv)] }}, c.hidden),
                .zba = try b.linear(&.{ .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_z"), .n = c.gdnValueDim(), .w = w[@backingInt(M.z)] }, .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_b"), .n = c.v_heads, .w = w[@backingInt(M.b)] }, .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_a"), .n = c.v_heads, .w = w[@backingInt(M.a)] } }, c.hidden),
                .out = try b.linear(&.{.{ .module = try Name.full(a, prefix, "linear_attn.out_proj"), .n = c.hidden, .w = w[@backingInt(M.out)] }}, c.gdnValueDim()),
                .conv = try b.tensor(try Name.full(a, prefix, "linear_attn.conv1d.weight"), .bf16, &.{ c.gdnQkvDim(), c.conv_kernel, 1 }),
                .a_log = try b.tensor(try Name.full(a, prefix, "linear_attn.A_log"), .bf16, &.{c.v_heads}),
                .dt = try b.tensor(try Name.full(a, prefix, "linear_attn.dt_bias"), .bf16, &.{c.v_heads}),
                .norm = try b.tensor(try Name.full(a, prefix, "linear_attn.norm.weight"), .bf16, &.{c.dv}),
            } },
            .attention => .{ .attention = .{
                .q = try b.linear(&.{.{ .module = try Name.full(a, prefix, "self_attn.q_proj"), .n = c.qDim(), .w = w[@backingInt(M.q)] }}, c.hidden),
                .kv = try b.linear(&.{ .{ .module = try Name.full(a, prefix, "self_attn.k_proj"), .n = c.kvDim(), .w = w[@backingInt(M.k)] }, .{ .module = try Name.full(a, prefix, "self_attn.v_proj"), .n = c.kvDim(), .w = w[@backingInt(M.v)] } }, c.hidden),
                .out = try b.linear(&.{.{ .module = try Name.full(a, prefix, "self_attn.o_proj"), .n = c.hidden, .w = w[@backingInt(M.o)] }}, c.heads * c.head_dim),
                .q_norm = try b.tensor(try Name.full(a, prefix, "self_attn.q_norm.weight"), .bf16, &.{c.head_dim}),
                .k_norm = try b.tensor(try Name.full(a, prefix, "self_attn.k_norm.weight"), .bf16, &.{c.head_dim}),
            } },
        };
        const gu = try b.linearGu(&.{ .{ .module = try Name.full(a, prefix, "mlp.gate_proj"), .n = c.intermediate, .w = w[@backingInt(M.gate)] }, .{ .module = try Name.full(a, prefix, "mlp.up_proj"), .n = c.intermediate, .w = w[@backingInt(M.up)] } }, c.hidden, i);
        const down = try b.linear(&.{.{ .module = try Name.full(a, prefix, "mlp.down_proj"), .n = c.hidden, .w = w[@backingInt(M.down)] }}, c.intermediate);
        return .{ .norm = try b.tensor(try Name.full(a, prefix, "input_layernorm.weight"), .bf16, &.{c.hidden}), .post_norm = try b.tensor(try Name.full(a, prefix, "post_attention_layernorm.weight"), .bf16, &.{c.hidden}), .gu = gu, .down = down, .mixer = mixer };
    }
};

const Part = struct { module: []const u8, n: usize, w: Width };

fn checked(t: Tensor, dtype: cp.DType, shape: []const usize) !Tensor {
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.BadNativeTensor;
    return t;
}

/// The graph from `checkpoint`, whose `repacked` tensors stay in their files: each Linear is staged, repacked, dropped.
pub fn load(allocator: std.mem.Allocator, device: mtl.Device, checkpoint: *const cp.Checkpoint, c: Config, ws: *const widths.Widths) !Weights {
    return loadWithOptions(allocator, device, checkpoint, c, ws, .{});
}

pub fn loadWithOptions(allocator: std.mem.Allocator, device: mtl.Device, checkpoint: *const cp.Checkpoint, c: Config, ws: *const widths.Widths, options: LoadOptions) !Weights {
    const prefix: []const u8 = if (checkpoint.has("language_model.model.embed_tokens.weight")) "language_model." else if (checkpoint.has("model.embed_tokens.weight")) "" else return error.MissingTextNamespace;
    if (checkpoint.has("language_model.model.embed_tokens.weight") and checkpoint.has("model.embed_tokens.weight")) return error.AmbiguousTextNamespace;
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        for (owned.items) |b| b.deinit();
        owned.deinit(allocator);
    }
    var runs: std.ArrayList([]projection.Run) = .empty;
    errdefer {
        for (runs.items) |r| allocator.free(r);
        runs.deinit(allocator);
    }
    const layers = try allocator.alloc(Layer, c.layers);
    errdefer allocator.free(layers);
    var stage = cp.Stage{ .device = device };
    defer stage.deinit();
    const selected: []bool = if (options.dense_prompt_gu) try allocator.alloc(bool, c.layers) else &.{};
    defer if (options.dense_prompt_gu) allocator.free(selected);
    if (options.dense_prompt_gu) {
        for (selected, 0..) |*take, i| {
            const w = ws.layers[i];
            take.* = denseLayerEligible(i, c.layers, c.hidden, c.intermediate, w[@backingInt(widths.Member.gate)], w[@backingInt(widths.Member.up)]);
        }
        selectDenseLayers(selected, options.max_aux_bytes);
    }
    var dense_budget = DenseBudget{ .limit = options.max_aux_bytes };
    const b = Builder{ .allocator = allocator, .device = device, .checkpoint = checkpoint, .prefix = prefix, .owned = &owned, .stage = &stage, .runs = &runs, .widths = ws, .dense_layers = selected, .dense_budget = &dense_budget };
    for (layers, 0..) |*l, i| l.* = try b.layer(c, i);
    const embed = try b.quant("model.embed_tokens", c.vocab, c.hidden, ws.embed);
    const norm = try b.tensor("model.norm.weight", .bf16, &.{c.hidden});
    const head = try b.linear(&.{.{ .module = "lm_head", .n = c.vocab, .w = ws.head }}, c.hidden);
    // A decoded round's head is the largest single matrix the round streams and the only decode shape `tileBatch`
    // leaves unbatched. It takes its own copy with eight adjacent tiles interleaved, the order every other decode
    // shape reads, while `head` keeps the row order the drafter slices 98,592 of its rows in (runtime_head.zig).
    const batched_head = device.tensorUnits() and ws.head.bits == 4 and ws.head.group == 64 and c.vocab % 32 == 0;
    const head_decode = if (batched_head) try b.linearForced(&.{.{ .module = "lm_head", .n = c.vocab, .w = ws.head }}, c.hidden, 8) else head;
    var extra: ?Linear = null;
    if (batched_head) extra = head_decode;
    if (device.tensorUnits()) try laneOrder(allocator, device, layers, head, extra);
    return .{ .allocator = allocator, .layers = layers, .embed = embed, .norm = norm, .head = head, .head_decode = head_decode, .owned = owned, .runs = runs, .dense_prompt_gu_bytes = dense_budget.used };
}

/// Tensor-unit projections leave load in lane_projection.reg_layout, the order quant_gpu's reg kernel reads.
fn laneOrder(a: std.mem.Allocator, device: mtl.Device, layers: []const Layer, head: Linear, extra: ?Linear) !void {
    var jobs: std.ArrayList(lane.Order) = .empty;
    defer jobs.deinit(a);
    for (layers) |l| {
        const list = switch (l.mixer) {
            .linear => |g| [_]Linear{ g.qkv, g.zba, g.out, l.gu, l.down },
            .attention => |t| [_]Linear{ t.q, t.kv, t.out, l.gu, l.down },
        };
        for (list) |p| try regJobs(a, &jobs, p);
    }
    try regJobs(a, &jobs, head);
    if (extra) |p| try regJobs(a, &jobs, p);
    try lane.regOrder(device, jobs.items);
}

/// Only 4-bit g64 matrices take the reg kernel; other widths keep lane_projection.weight_layout.
fn regJobs(a: std.mem.Allocator, jobs: *std.ArrayList(lane.Order), p: Linear) std.mem.Allocator.Error!void {
    for (p.runs) |r| try regJobs(a, jobs, r.linear);
    if (p.runs.len == 0 and p.bits == 4 and p.group == 64 and p.n % 32 == 0) try jobs.append(a, .{ .w = .{ .buf = p.words.buffer, .off = p.words.offset }, .n = projection.storedColumns(p.n, p.batch), .k = p.k });
}

pub fn ref(t: Tensor) Ref {
    return .{ .buffer = t.buffer, .offset = t.offset };
}

test "repacked names the modules the graph reads only through owned copies" {
    try std.testing.expect(repacked("language_model.model.layers.3.mlp.gate_proj.weight", 2));
    try std.testing.expect(repacked("model.layers.3.self_attn.o_proj.scales", 2));
    try std.testing.expect(repacked("language_model.model.layers.0.linear_attn.out_proj.biases", 2));
    try std.testing.expect(repacked("language_model.lm_head.weight", 2));
    try std.testing.expect(repacked("lm_head.scales", 2));
    try std.testing.expect(!repacked("language_model.model.embed_tokens.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.0.linear_attn.conv1d.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.3.self_attn.q_norm.weight", 2));
    try std.testing.expect(!repacked("language_model.model.norm.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.0.linear_attn.A_log", 2));
}

test "dense prompt GU chooses the last budget eligible nonterminal layers" {
    try std.testing.expectEqual(@as(usize, 356655114), dense_min_bytes);
    try std.testing.expect(!(LoadOptions{}).dense_prompt_gu);
    try std.testing.expectEqual(@as(usize, 7 << 28), (LoadOptions{}).max_aux_bytes);
    var selected: [64]bool = undefined;
    for (&selected, 0..) |*take, i| take.* = denseLayerEligible(i, 64, 5120, 17408, widths.uniform, widths.uniform);
    selectDenseLayers(&selected, (LoadOptions{}).max_aux_bytes);
    for (selected, 0..) |take, i| try std.testing.expectEqual(i >= 58 and i <= 62, take);
    try std.testing.expect(!denseLayerEligible(63, 64, 5120, 17408, widths.uniform, widths.uniform));
    try std.testing.expect(!denseLayerEligible(62, 64, 2560, 17408, widths.uniform, widths.uniform));
    try std.testing.expect(!denseLayerEligible(62, 64, 5120, 17408, .{ .bits = 2, .group = 64 }, widths.uniform));
    try std.testing.expect(!denseLayerEligible(62, 64, 5120, 17408, widths.uniform, .{ .bits = 4, .group = 128 }));
    for (&selected, 0..) |*take, i| take.* = i < 63 and i != 60;
    selectDenseLayers(&selected, (LoadOptions{}).max_aux_bytes);
    for (selected, 0..) |take, i| try std.testing.expectEqual(i >= 57 and i <= 62 and i != 60, take);
    @memset(&selected, true);
    selectDenseLayers(&selected, dense_min_bytes - 1);
    for (selected) |take| try std.testing.expect(!take);
}

test "dense prompt GU actual buffer budget rejects atomically then transfers exactly four" {
    const Buffer = struct {
        id: u32,
        size: usize,
        fn length(b: @This()) usize { return b.size; }
    };
    const prepared = .{
        .high = Buffer{ .id = 1, .size = 80 },
        .offsets = Buffer{ .id = 2, .size = 8 },
        .indices = Buffer{ .id = 3, .size = 2 },
        .residual = Buffer{ .id = 4, .size = 4 },
    };
    var owned: std.ArrayList(Buffer) = .empty;
    defer owned.deinit(std.testing.allocator);
    try owned.ensureUnusedCapacity(std.testing.allocator, 4);
    var budget = DenseBudget{ .limit = 93 };
    try std.testing.expect(!try admitDense(&owned, &budget, prepared));
    try std.testing.expectEqual(@as(usize, 0), owned.items.len);
    try std.testing.expectEqual(@as(usize, 0), budget.used);
    budget.limit = 94;
    try std.testing.expect(try admitDense(&owned, &budget, prepared));
    try std.testing.expectEqual(@as(usize, 4), owned.items.len);
    try std.testing.expectEqual(@as(usize, 94), budget.used);
    for (owned.items, 1..) |buffer, id| try std.testing.expectEqual(@as(u32, @intCast(id)), buffer.id);
    try std.testing.expect(!try admitDense(&owned, &budget, prepared));
    try std.testing.expectEqual(@as(usize, 4), owned.items.len);
    try std.testing.expectEqual(@as(usize, 94), budget.used);
}

test "dense prompt GU actual byte sum overflow transfers no ownership" {
    const Buffer = struct {
        size: usize,
        fn length(b: @This()) usize { return b.size; }
    };
    const prepared = .{
        .high = Buffer{ .size = std.math.maxInt(usize) },
        .offsets = Buffer{ .size = 1 },
        .indices = Buffer{ .size = 2 },
        .residual = Buffer{ .size = 4 },
    };
    var owned: std.ArrayList(Buffer) = .empty;
    defer owned.deinit(std.testing.allocator);
    try owned.ensureUnusedCapacity(std.testing.allocator, 4);
    var budget = DenseBudget{ .limit = std.math.maxInt(usize) };
    try std.testing.expectError(error.Overflow, admitDense(&owned, &budget, prepared));
    try std.testing.expectEqual(@as(usize, 0), owned.items.len);
    try std.testing.expectEqual(@as(usize, 0), budget.used);
}

test "dense auxiliary default budget follows working set boundaries and remains capped" {
    const low: usize = 7 << 28;
    const high: usize = 8 << 30;
    const threshold: u64 = @as(u64, 64) << 30;
    try std.testing.expectEqual(low, defaultAuxBudget(0));
    try std.testing.expectEqual(low, defaultAuxBudget(@as(u64, 36) << 30));
    try std.testing.expectEqual(low, defaultAuxBudget(threshold - 1));
    try std.testing.expectEqual(high, defaultAuxBudget(threshold));
    try std.testing.expectEqual(high, defaultAuxBudget(threshold + 1));
    try std.testing.expectEqual(high, defaultAuxBudget(@as(u64, 128) << 30));
    try std.testing.expectEqual(high, defaultAuxBudget(std.math.maxInt(u64)));
    try std.testing.expectEqual(low, (LoadOptions{}).max_aux_bytes);
}

test "working set budget chooses five or twenty four eligible sidecars" {
    var selected: [64]bool = undefined;
    for ([_]u64{ @as(u64, 36) << 30, @as(u64, 64) << 30, std.math.maxInt(u64) }) |working_set| {
        for (&selected, 0..) |*take, i| take.* = denseLayerEligible(i, 64, 5120, 17408, widths.uniform, widths.uniform);
        selectDenseLayers(&selected, defaultAuxBudget(working_set));
        const first: usize = if (working_set < @as(u64, 64) << 30) 58 else 39;
        var count: usize = 0;
        for (selected, 0..) |take, i| {
            try std.testing.expectEqual(i >= first and i <= 62, take);
            if (take) count += 1;
        }
        try std.testing.expectEqual(@as(usize, if (first == 58) 5 else 24), count);
    }
}
