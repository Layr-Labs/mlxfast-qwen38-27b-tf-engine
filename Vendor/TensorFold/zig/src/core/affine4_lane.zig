//! Prepared affine q4 words use the existing shared lane packing and encode contract.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
pub const cpu = @import("affine4.zig");
pub const Adapter = struct {
    projection: lane.Projection,
    /// The same words on the rows8 pipeline: a draft pass stores at most eight rows, so its lane drops the padding ones.
    projection8: ?lane.Projection,
    words: mtl.Buffer,
    metadata: mtl.Buffer,
    /// With `order`, the words are rewritten into reg_layout and encode on the reg kernel (same output words).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        const layout = lane.Layout{ .n = prepared.n, .k = prepared.k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = true, .cooperative = order == null, .reg = order != null };
        try layout.validate();
        const words = try device.buffer(layout.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer words.deinit();
        const metadata = try device.buffer(layout.metadataElements() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer metadata.deinit();
        try lane.pack(layout, prepared.words, prepared.scales, prepared.biases, words.slice(u32, layout.weightWords()), metadata.slice(u16, layout.metadataElements()));
        if (order) |o| try o.run(&.{.{ .w = .{ .buf = words }, .n = prepared.n, .k = prepared.k }});
        const projection = try lane.Projection.init(a, device, layout);
        errdefer projection.deinit();
        var narrow = layout;
        narrow.rows8 = true;
        const projection8 = if (order != null) try lane.Projection.init(a, device, narrow) else null;
        return .{ .projection = projection, .projection8 = projection8, .words = words, .metadata = metadata };
    }
    pub fn deinit(adapter: Adapter) void {
        adapter.projection.deinit();
        if (adapter.projection8) |p| p.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        const p = if (rows <= 8) adapter.projection8 orelse adapter.projection else adapter.projection;
        try p.encode(e, x, .{ .buf = adapter.words }, .{ .buf = adapter.metadata }, y, rows);
    }
};
