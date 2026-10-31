//! Prepared affine q4 words use the existing shared lane packing and encode contract.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
pub const cpu = @import("affine4.zig");
pub const Adapter = struct {
    projection: lane.Projection,
    words: mtl.Buffer,
    metadata: mtl.Buffer,
    /// With `order`, the words are rewritten into reg_layout and encode on the reg kernel (same output words), and the
    /// metadata takes lane_projection.indexed_layout when the matrix has at most lane.sb_table distinct pairs.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        var layout = lane.Layout{ .n = prepared.n, .k = prepared.k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = true, .cooperative = order == null, .reg = order != null, .reg_lean = order != null };
        try layout.validate();
        const words = try device.buffer(layout.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer words.deinit();
        const pairs = try a.alloc(u16, layout.metadataElements());
        defer a.free(pairs);
        try lane.pack(layout, prepared.words, prepared.scales, prepared.biases, words.slice(u32, layout.weightWords()), pairs);
        if (order) |o| try o.run(&.{.{ .w = .{ .buf = words }, .n = prepared.n, .k = prepared.k }});
        const metadata = indexed: {
            if (order != null) {
                const sb = try device.buffer(lane.indexedBytes(layout.n, layout.k, layout.format.group), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                if (index(a, pairs, sb.contents()[0..lane.indexedBytes(layout.n, layout.k, layout.format.group)]) catch |err| {
                    sb.deinit();
                    return err;
                }) {
                    layout.indexed = true;
                    break :indexed sb;
                }
                sb.deinit();
            }
            const sb = try device.buffer(pairs.len * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            @memcpy(sb.slice(u16, pairs.len), pairs);
            break :indexed sb;
        };
        errdefer metadata.deinit();
        const projection = try lane.Projection.init(a, device, layout);
        return .{ .projection = projection, .words = words, .metadata = metadata };
    }
    pub fn deinit(adapter: Adapter) void {
        adapter.projection.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try adapter.projection.encode(e, x, .{ .buf = adapter.words }, .{ .buf = adapter.metadata }, y, rows);
    }
};

/// lane_projection.indexed_layout of packed pairs ([K/64][N][scale, bias]) into `out`: the distinct (scale | bias << 16)
/// words, most used first (ties in first-seen order), then each group's u16 index. False past lane.sb_table words. Every
/// index names its group's own word, so the reg kernel reads the values the pairs hold.
fn index(a: std.mem.Allocator, pairs: []const u16, out: []u8) !bool {
    const cap = lane.sb_table;
    const groups = pairs.len / 2;
    if (out.len != cap * 4 + groups * 2) return error.BadIndexedMetadata;
    var seen = std.AutoHashMap(u32, u32).init(a);
    defer seen.deinit();
    try seen.ensureTotalCapacity(2 * cap);
    const firsts = try a.alloc(u32, cap);
    defer a.free(firsts);
    const counts = try a.alloc(u32, cap);
    defer a.free(counts);
    @memset(counts, 0);
    const ids = try a.alloc(u16, groups);
    defer a.free(ids);
    for (ids, 0..) |*id, i| {
        const word = @as(u32, pairs[2 * i]) | (@as(u32, pairs[2 * i + 1]) << 16);
        const got = seen.getOrPutAssumeCapacity(word);
        if (!got.found_existing) {
            if (seen.count() > cap) return false;
            got.value_ptr.* = seen.count() - 1;
            firsts[got.value_ptr.*] = word;
        }
        counts[got.value_ptr.*] += 1;
        id.* = @intCast(got.value_ptr.*);
    }
    // most used first, ties by first-seen id: key = (max - count) << 16 | id, ascending
    const distinct = seen.count();
    const keys = try a.alloc(u64, distinct);
    defer a.free(keys);
    for (keys, 0..) |*key, id| key.* = (@as(u64, std.math.maxInt(u32) - counts[id]) << 16) | @as(u64, id);
    std.mem.sort(u64, keys, {}, std.sort.asc(u64));
    const rank = try a.alloc(u16, distinct);
    defer a.free(rank);
    const table = std.mem.bytesAsSlice(u32, out[0 .. cap * 4]);
    @memset(table, 0);
    for (keys, 0..) |key, r| {
        const id: usize = @intCast(key & 0xFFFF);
        rank[id] = @intCast(r);
        table[r] = firsts[id];
    }
    const indices = std.mem.bytesAsSlice(u16, out[cap * 4 ..]);
    for (indices, ids) |*at, id| at.* = rank[id];
    return true;
}

test "an indexed copy names every group's own pair" {
    const a = std.testing.allocator;
    const groups = 96;
    var pairs: [2 * groups]u16 = undefined;
    for (0..groups) |i| {
        pairs[2 * i] = @intCast(0x3c00 + (i * 7) % 5);
        pairs[2 * i + 1] = @intCast(0xb800 + (i * 3) % 4);
    }
    const out = try a.alloc(u8, lane.sb_table * 4 + groups * 2);
    defer a.free(out);
    try std.testing.expect(try index(a, &pairs, out));
    const table = std.mem.bytesAsSlice(u32, out[0 .. lane.sb_table * 4]);
    const indices = std.mem.bytesAsSlice(u16, out[lane.sb_table * 4 ..]);
    for (0..groups) |i| try std.testing.expectEqual(@as(u32, pairs[2 * i]) | (@as(u32, pairs[2 * i + 1]) << 16), table[indices[i]]);
}
