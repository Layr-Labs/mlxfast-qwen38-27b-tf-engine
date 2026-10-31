//! Prepared affine q4 words use the existing shared lane packing and encode contract.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
pub const cpu = @import("affine4.zig");
pub const Adapter = struct {
    projection: lane.Projection,
    /// The same weights for decoded rows (at most 8): ROWS8 and the lean prefetch (reg adapters only).
    small: ?lane.Projection = null,
    words: mtl.Buffer,
    metadata: mtl.Buffer,
    /// With `order`, the words are rewritten into reg_layout and encode on the reg kernel (same output words).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        // reg adapters take the lean prefetch (codes only in flight; metadata and sums load in their own group)
        const layout = lane.Layout{ .n = prepared.n, .k = prepared.k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = true, .cooperative = order == null, .reg = order != null, .reg_lean = order != null };
        try layout.validate();
        const words = try device.buffer(layout.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer words.deinit();
        const metadata = try device.buffer(layout.metadataElements() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer metadata.deinit();
        try lane.pack(layout, prepared.words, prepared.scales, prepared.biases, words.slice(u32, layout.weightWords()), metadata.slice(u16, layout.metadataElements()));
        if (order) |o| try o.run(&.{.{ .w = .{ .buf = words }, .n = prepared.n, .k = prepared.k }});
        const projection = try lane.Projection.init(a, device, layout);
        errdefer projection.deinit();
        var rows8 = layout;
        rows8.rows8 = true;
        const small: ?lane.Projection = if (layout.reg) try lane.Projection.init(a, device, rows8) else null;
        return .{ .projection = projection, .small = small, .words = words, .metadata = metadata };
    }
    /// The pipeline for `rows` rows: decoded rows (at most 8) take `small` when there is one.
    pub fn forRows(adapter: Adapter, rows: u32) lane.Projection {
        return if (rows <= 8) adapter.small orelse adapter.projection else adapter.projection;
    }
    pub fn deinit(adapter: Adapter) void {
        if (adapter.small) |p| p.deinit();
        adapter.projection.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try adapter.projection.encode(e, x, .{ .buf = adapter.words }, .{ .buf = adapter.metadata }, y, rows);
    }
};
