//! Parent-conditioned DFlash tree search leaves all target verification and keyed draws authoritative.
const std = @import("std");
const weights = @import("weights.zig");
const op = @import("operators.zig");
pub const Noise = struct { ptr: *anyopaque, draw: *const fn (*anyopaque, u64, u32) anyerror!f64 };
/// rows: lattice depth rows the search may use (0: all); a depth-1 tree drafted over a wider block reads only row 0.
pub const Options = struct { nodes: u32 = 15, children: u32 = 4, rows: u32 = 0, tau: f64 = 1.5, edge: f64 = 0.6, noise_weight: f64 = 0.7, temperature: f64 = 1, noise: ?Noise = null };
/// Candidate count and maximum draft-parent path length are independent.
pub const Policy = struct {
    nodes: usize,
    max_depth: ?u32 = null,
    /// Siblings a node may offer (Options.children): a depth-one tree of `nodes` drafts needs as many.
    children: u32 = 4,

    pub fn block(p: Policy, cap: u32) !u32 {
        if (p.nodes > 15 or cap == 0 or cap > 16) return error.BadDraftOptions;
        if (p.max_depth) |depth| if (depth == 0 or depth > 15) return error.BadDraftDepth;
        return @min(cap, @min(@as(u32, @intCast(p.nodes)) + 1, (p.max_depth orelse 15) + 1));
    }

    pub fn checkTree(p: Policy, parents: []const i32) !void {
        const limit = p.max_depth orelse return;
        if (parents.len > p.nodes or parents.len > 15) return error.BadDraftWidth;
        var depths: [15]u32 = undefined;
        for (parents, 0..) |parent, row| {
            if (parent < -1 or parent >= row) return error.BadDraftParents;
            depths[row] = if (parent < 0) 1 else depths[@intCast(parent)] + 1;
            if (depths[row] > limit) return error.BadDraftDepth;
        }
    }

    pub fn budget(p: Policy, remaining: u32) u32 {
        return if (p.max_depth) |depth| @min(remaining, depth + 1) else remaining;
    }
};
pub const Tree = struct {
    gpa: std.mem.Allocator,
    tokens: []u32,
    parents: []i32,
    scores: []f64,
    pub fn deinit(t: *Tree) void {
        t.gpa.free(t.tokens);
        t.gpa.free(t.parents);
        t.gpa.free(t.scores);
        t.* = undefined;
    }
    pub fn verifyParents(t: Tree, gpa: std.mem.Allocator) ![]i32 {
        const parents = try gpa.alloc(i32, t.parents.len + 1);
        parents[0] = -1;
        for (t.parents, 0..) |p, i| {
            if (p < -1 or p >= i) {
                gpa.free(parents);
                return error.BadDraftParents;
            }
            parents[i + 1] = p + 1;
        }
        return parents;
    }
};
const Node = struct { score: f64, parent: i32, token: u32, depth: u32 };
fn before(a: Node, b: Node) bool {
    return a.score > b.score or (a.score == b.score and (a.parent < b.parent or (a.parent == b.parent and a.token < b.token)));
}
fn native(t: weights.Tensor, row: u32, column: u32) !f64 {
    const at = (try std.math.add(usize, try std.math.mul(usize, row, t.shape[1]), column)) * 2;
    if (at + 2 > t.bytes.len) return error.DraftCodebookShape;
    const bits = std.mem.readInt(u16, t.bytes[at..][0..2], .little);
    const value: f32 = @bitCast(@as(u32, bits) << 16);
    if (!std.math.isFinite(value)) return error.DraftCodebookFinite;
    return value;
}
fn edges(g: *const weights.Graph, lattice: op.Lattice, parent: u32, token: u32, depth: u32) !f64 {
    @setFloatMode(.strict);
    var sum: f64 = 0;
    for (0..g.config.rank) |r| sum += (try native(g.predecessor, parent, @intCast(r))) * @as(f64, lattice.projected[@as(usize, depth) * lattice.rank + r]) * (try native(g.successor, token, @intCast(r)));
    return sum;
}
fn expand(g: *const weights.Graph, lattice: op.Lattice, options: Options, first: u64, parent_token: u32, parent: i32, depth: u32, cumulative: f64, frontier: *std.ArrayList(Node)) !void {
    @setFloatMode(.strict);
    var scores: [16]f64 = undefined;
    var order: [16]u32 = undefined;
    const temperature = if (options.noise != null) @max(options.temperature, 1e-6) else 1;
    const row = @as(usize, depth) * lattice.topk;
    var maximum: f64 = -std.math.inf(f64);
    for (0..lattice.topk) |i| {
        const token = lattice.candidates[row + i];
        var score = @as(f64, lattice.unary[row + i]) / temperature + options.edge * try edges(g, lattice, parent_token, token, depth) / temperature;
        if (options.noise) |n| score += options.noise_weight * try n.draw(n.ptr, try std.math.add(u64, first, depth), token);
        score /= options.tau;
        if (!std.math.isFinite(score)) return error.DraftScoreFinite;
        scores[i] = score;
        order[i] = @intCast(i);
        maximum = @max(maximum, score);
    }
    var denominator: f64 = 0;
    for (scores[0..lattice.topk]) |*score| {
        score.* -= maximum;
        denominator += @exp(score.*);
    }
    const normalizer = @log(denominator);
    for (scores[0..lattice.topk]) |*score| score.* -= normalizer;
    const Rank = struct {
        scores: []const f64,
        fn less(x: @This(), a: u32, b: u32) bool {
            return x.scores[a] > x.scores[b] or (x.scores[a] == x.scores[b] and a < b);
        }
    };
    std.mem.sort(u32, order[0..lattice.topk], Rank{ .scores = scores[0..lattice.topk] }, Rank.less);
    for (order[0..@min(options.children, lattice.topk)]) |i| try frontier.append(lattice.gpa, .{ .score = cumulative + scores[i], .parent = parent, .token = lattice.candidates[row + i], .depth = depth });
}
fn shallow(l: op.Lattice, rows: u32) op.Lattice {
    if (rows == 0 or rows >= l.depth) return l;
    var out = l;
    out.depth = rows;
    out.candidates = l.candidates[0 .. @as(usize, rows) * l.topk];
    out.unary = l.unary[0 .. @as(usize, rows) * l.topk];
    out.projected = l.projected[0 .. @as(usize, rows) * l.rank];
    return out;
}
pub fn propose(gpa: std.mem.Allocator, g: *const weights.Graph, whole: op.Lattice, anchor: u32, first_position: u64, options: Options) !Tree {
    try whole.check(g.config);
    const lattice = shallow(whole, options.rows);
    try g.config.check();
    for ([_]weights.Tensor{ g.predecessor, g.successor }) |t| if (t.dtype != .bf16 or t.rank != 2 or t.shape[0] != g.config.vocab or t.shape[1] != g.config.rank) return error.DraftCodebookShape;
    try lattice.check(g.config);
    if (anchor >= g.config.vocab or options.nodes > 32 or options.children == 0 or options.children > 16 or lattice.topk > 16 or !std.math.isFinite(options.tau) or options.tau <= 0 or !std.math.isFinite(options.edge) or !std.math.isFinite(options.noise_weight) or !std.math.isFinite(options.temperature) or options.temperature <= 0) return error.BadDraftSelector;
    var frontier: std.ArrayList(Node) = .empty;
    defer frontier.deinit(lattice.gpa);
    const tokens = try gpa.alloc(u32, options.nodes);
    defer gpa.free(tokens);
    const parents = try gpa.alloc(i32, options.nodes);
    defer gpa.free(parents);
    const scores = try gpa.alloc(f64, options.nodes);
    defer gpa.free(scores);
    if (options.nodes > 0) try expand(g, lattice, options, first_position, anchor, -1, 0, 0, &frontier);
    var count: usize = 0;
    while (frontier.items.len > 0 and count < options.nodes) {
        var best: usize = 0;
        for (frontier.items[1..], 1..) |node, i| if (before(node, frontier.items[best])) {
            best = i;
        };
        const node = frontier.swapRemove(best);
        tokens[count] = node.token;
        parents[count] = node.parent;
        scores[count] = node.score;
        const me = count;
        count += 1;
        if (node.depth + 1 < lattice.depth) try expand(g, lattice, options, first_position, node.token, @intCast(me), node.depth + 1, node.score, &frontier);
    }
    const out_tokens = try gpa.dupe(u32, tokens[0..count]);
    errdefer gpa.free(out_tokens);
    const out_parents = try gpa.dupe(i32, parents[0..count]);
    errdefer gpa.free(out_parents);
    return .{ .gpa = gpa, .tokens = out_tokens, .parents = out_parents, .scores = try gpa.dupe(f64, scores[0..count]) };
}

test "draft roots shift to pending row while descendants retain topology" {
    const a = std.testing.allocator;
    const t = Tree{ .gpa = a, .tokens = &.{}, .parents = @constCast(&[_]i32{ -1, -1, 0, 2 }), .scores = &.{} };
    const parents = try t.verifyParents(a);
    defer a.free(parents);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 1, 3 }, parents);
}

test "proposal width and parent depth have separate bounds" {
    const p = Policy{ .nodes = 2, .max_depth = 1 };
    try std.testing.expectEqual(@as(u32, 2), try p.block(16));
    try std.testing.expectEqual(@as(u32, 3), try (Policy{ .nodes = 2 }).block(16));
    try std.testing.expectEqual(@as(u32, 1), try (Policy{ .nodes = 0, .max_depth = 1 }).block(16));
    try p.checkTree(&.{ -1, -1 });
    try p.checkTree(&.{});
    try std.testing.expectError(error.BadDraftDepth, p.checkTree(&.{ -1, 0 }));
    try std.testing.expectError(error.BadDraftWidth, p.checkTree(&.{ -1, -1, -1 }));
    try std.testing.expectError(error.BadDraftParents, p.checkTree(&.{ -1, 1 }));
    try std.testing.expectError(error.BadDraftDepth, (Policy{ .nodes = 2, .max_depth = 0 }).block(16));
    try std.testing.expectEqual(@as(u32, 2), p.budget(128));
    try std.testing.expectEqual(@as(u32, 1), p.budget(1));
}
