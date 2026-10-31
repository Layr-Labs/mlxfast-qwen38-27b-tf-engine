//! Prepared affine q4 words use the existing shared lane packing and encode contract.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
pub const cpu = @import("affine4.zig");
pub const Adapter = struct {
    projection: lane.Projection,
    small: ?lane.Projection = null,
    words: mtl.Buffer,
    metadata: mtl.Buffer,
    /// Reg adapters' (scale, bias) pairs as lane_projection.indexed_layout, when they fit its table (`meta`).
    index: ?mtl.Buffer = null,
    /// With `order`, the words are rewritten into reg_layout and encode on the reg kernel (same output words).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        return initWithSmall(a, device, prepared, sk, order, false);
    }
    /// Optional row-bounded pipeline shares the same prepared weight and metadata buffers.
    pub fn initWithSmall(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder, small_rows: bool) !Adapter {
        var layout = lane.Layout{
            .n = prepared.n,
            .k = prepared.k,
            .format = .{ .bits = 4, .group = 64 },
            .sk = sk,
            .precompute_sums = true,
            .cooperative = order == null,
            .reg = order != null,
            .reg_lean = order != null, // codes-only prefetch, as the target's decoded rows take
        };
        try layout.validate();
        const words = try device.buffer(layout.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer words.deinit();
        const metadata = try device.buffer(layout.metadataElements() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer metadata.deinit();
        try lane.pack(layout, prepared.words, prepared.scales, prepared.biases, words.slice(u32, layout.weightWords()), metadata.slice(u16, layout.metadataElements()));
        if (order) |o| try o.run(&.{.{ .w = .{ .buf = words }, .n = prepared.n, .k = prepared.k }});
        // reg adapters read the pairs through an index (the same words, fewer bytes a group) when they fit its table;
        // every pipeline below is built after this, so the rows8 one reads the same index
        const index: ?mtl.Buffer = if (order != null) try indexPairs(device, a, metadata.slice(u16, layout.metadataElements()), prepared.n, prepared.k) else null;
        errdefer if (index) |b| b.deinit();
        layout.indexed = index != null;
        const projection = try lane.Projection.init(a, device, layout);
        errdefer projection.deinit();
        const small: ?lane.Projection = if (small_rows and layout.reg) blk: {
            var narrow = layout;
            narrow.rows8 = true;
            break :blk try lane.Projection.init(a, device, narrow);
        } else null;
        return .{ .projection = projection, .small = small, .words = words, .metadata = metadata, .index = index };
    }
    pub fn deinit(adapter: Adapter) void {
        if (adapter.small) |p| p.deinit();
        adapter.projection.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
        if (adapter.index) |b| b.deinit();
    }
    /// The metadata the adapter's launches bind: the index where it has one, else the pairs.
    pub fn meta(adapter: Adapter) lane.Ref {
        return .{ .buf = adapter.index orelse adapter.metadata };
    }
    pub fn forRows(adapter: Adapter, rows: u32) lane.Projection {
        return if (rows <= 8) adapter.small orelse adapter.projection else adapter.projection;
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try adapter.forRows(rows).encode(e, x, .{ .buf = adapter.words }, adapter.meta(), y, rows);
    }
};

/// lane_projection.indexed_layout of packed pairs ([K/64][N][scale, bias]): the distinct (scale | bias << 16) words in
/// first-seen order, then each group's u16 index, as the target's projections build it. Null past lane_projection.sb_table
/// distinct pairs (the pairs are then read directly).
fn indexPairs(device: mtl.Device, allocator: std.mem.Allocator, pair: []const u16, n: usize, k: usize) !?mtl.Buffer {
    const cap = lane.sb_table;
    var seen = std.AutoHashMap(u32, u16).init(allocator);
    defer seen.deinit();
    try seen.ensureTotalCapacity(2 * cap);
    const sb = try device.buffer(lane.indexedBytes(n, k, 64), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    var keep = false;
    defer if (!keep) sb.deinit();
    const table = sb.slice(u32, cap);
    @memset(table, 0);
    const index = @as([*]u16, @ptrCast(@alignCast(sb.contents() + cap * 4)))[0 .. n * (k / 64)];
    for (index, 0..) |*at, i| {
        const word = @as(u32, pair[2 * i]) | (@as(u32, pair[2 * i + 1]) << 16);
        const got = seen.getOrPutAssumeCapacity(word);
        if (!got.found_existing) {
            if (seen.count() > cap) return null;
            got.value_ptr.* = @intCast(seen.count() - 1);
            table[got.value_ptr.*] = word;
        }
        at.* = got.value_ptr.*;
    }
    keep = true;
    return sb;
}
