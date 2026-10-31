//! Prepared affine q4 words use the existing shared lane packing and encode contract.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
pub const cpu = @import("affine4.zig");
pub const Adapter = struct {
    projection: lane.Projection,
    words: mtl.Buffer,
    metadata: mtl.Buffer,
    /// Reg adapters' (scale, bias) pairs through lane_projection.indexMetadata, when they fit its table.
    index: ?mtl.Buffer = null,
    /// With `order`, the words are rewritten into reg_layout and encode on the reg kernel (same output words).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        var layout = lane.Layout{ .n = prepared.n, .k = prepared.k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = true, .cooperative = order == null, .reg = order != null, .reg_lean = order != null };
        try layout.validate();
        const words = try device.buffer(layout.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer words.deinit();
        const metadata = try device.buffer(layout.metadataElements() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer metadata.deinit();
        try lane.pack(layout, prepared.words, prepared.scales, prepared.biases, words.slice(u32, layout.weightWords()), metadata.slice(u16, layout.metadataElements()));
        if (order) |o| try o.run(&.{.{ .w = .{ .buf = words }, .n = prepared.n, .k = prepared.k }});
        // reg adapters read the pairs through an index (the same words, fewer bytes a group) when they fit its table
        var index: ?mtl.Buffer = null;
        errdefer if (index) |b| b.deinit();
        if (order != null) if (try lane.indexMetadata(device, a, metadata.slice(u16, layout.metadataElements()), prepared.n, prepared.k)) |x| {
            index = x.buffer;
            layout.indexed = true;
            layout.indexed12 = x.twelve;
        };
        const projection = try lane.Projection.init(a, device, layout);
        return .{ .projection = projection, .words = words, .metadata = metadata, .index = index };
    }
    pub fn deinit(adapter: Adapter) void {
        adapter.projection.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
        if (adapter.index) |b| b.deinit();
    }
    /// The metadata the adapter's launches bind: the index where it has one, else the pairs.
    pub fn meta(adapter: Adapter) lane.Ref {
        return .{ .buf = adapter.index orelse adapter.metadata };
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try adapter.projection.encode(e, x, .{ .buf = adapter.words }, adapter.meta(), y, rows);
    }
};
