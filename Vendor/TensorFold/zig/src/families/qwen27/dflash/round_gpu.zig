//! Target picks and tree acceptance run on the GPU with one bounded result readback.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const contract = core.tree_round;
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
/// The verify's picks spread each row over tree_round_gpu.argmax_spans threadgroups and merge them (same picks);
/// off, each row's whole vocabulary runs through one threadgroup.
const span_picks = true;
pub const Matcher = struct {
    ops: core.tree_round_gpu.Ops,
    buffers: [6]mtl.Buffer,
    pub fn init(device: mtl.Device) !Matcher {
        const ops = try core.tree_round_gpu.Ops.init(device);
        errdefer ops.deinit();
        var buffers: [6]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (buffers[0..made]) |buffer| buffer.deinit();
        for (&buffers, [_]usize{ contract.max_rows * 4, contract.max_rows * 4, contract.max_rows * @sizeOf(contract.Pick), 16 * 4, @sizeOf(contract.Result), contract.max_rows * core.tree_round_gpu.argmax_spans * @sizeOf(core.tree_round_gpu.ArgmaxPart) }) |*buffer, bytes| {
            buffer.* = try device.buffer(bytes, options);
            made += 1;
        }
        return .{ .ops = ops, .buffers = buffers };
    }
    pub fn deinit(m: Matcher) void {
        m.ops.deinit();
        for (m.buffers) |buffer| buffer.deinit();
    }
    pub fn encode(m: *Matcher, e: mtl.ComputeEncoder, logits: core.tree_round_gpu.Ref, ids: []const u32, parents: []const i32, vocab: u32, budget: u32, eos: []const u32) !contract.Match {
        const shape = try m.stage(ids, parents, vocab, budget, eos);
        if (span_picks)
            try m.ops.argmaxSpans(e, logits, .{ .buf = m.buffers[5] }, .{ .buf = m.buffers[2] }, .{ .rows = shape.rows, .vocab = vocab, .stride = vocab })
        else
            try m.ops.argmax(e, logits, .{ .buf = m.buffers[2] }, .{ .rows = shape.rows, .vocab = vocab, .stride = vocab });
        e.barrier();
        return m.matchStaged(e, shape);
    }
    /// `encode` on picks an earlier dispatch in `e` wrote into `picks()` (a barrier between): the head sieve's.
    pub fn encodePicked(m: *Matcher, e: mtl.ComputeEncoder, ids: []const u32, parents: []const i32, vocab: u32, budget: u32, eos: []const u32) !contract.Match {
        return m.matchStaged(e, try m.stage(ids, parents, vocab, budget, eos));
    }
    /// The rows' picks (tree_round.Pick a row) the match reads.
    pub fn picks(m: *const Matcher) mtl.Buffer {
        return m.buffers[2];
    }
    /// Row `row`'s pick of the last round, once its command buffer ran.
    pub fn pickAt(m: *const Matcher, row: u32) contract.Pick {
        return m.buffers[2].slice(contract.Pick, row + 1)[row];
    }
    fn stage(m: *Matcher, ids: []const u32, parents: []const i32, vocab: u32, budget: u32, eos: []const u32) !contract.Match {
        if (ids.len == 0 or ids.len > contract.max_rows or parents.len != ids.len or eos.len > 16) return error.BadRoundShape;
        @memcpy(m.buffers[0].slice(u32, ids.len), ids);
        @memcpy(m.buffers[1].slice(i32, parents.len), parents);
        @memcpy(m.buffers[3].slice(u32, eos.len), eos);
        return .{ .rows = @intCast(ids.len), .vocab = vocab, .budget = budget, .eos_count = @intCast(eos.len) };
    }
    fn matchStaged(m: *Matcher, e: mtl.ComputeEncoder, shape: contract.Match) !contract.Match {
        try m.ops.match(e, .{ .buf = m.buffers[0] }, .{ .buf = m.buffers[1] }, .{ .buf = m.buffers[2] }, if (shape.eos_count == 0) null else .{ .buf = m.buffers[3] }, .{ .buf = m.buffers[4] }, shape);
        e.barrier();
        return shape;
    }
    pub fn read(m: *Matcher, shape: contract.Match) !contract.Decoded {
        const result: *const contract.Result = @ptrCast(@alignCast(m.buffers[4].contents()));
        const decoded = try contract.decode(result, shape);
        if (decoded.status != .ok) return error.InvalidTargetDraftRound;
        return decoded;
    }
};
