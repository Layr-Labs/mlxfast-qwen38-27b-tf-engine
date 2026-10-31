//! Prepared affine q4 words use the existing shared lane packing and encode contract.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
pub const cpu = @import("affine4.zig");
pub const Adapter = struct {
    projection: lane.Projection,
    compact_projection: ?lane.Projection = null,
    words: mtl.Buffer,
    metadata: mtl.Buffer,
    /// With `order`, the words are rewritten into reg_layout and encode on the reg kernel (same output words).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        return initImpl(a, device, prepared, sk, order, false);
    }
    /// Drafter SK8 views of at most eight rows may use compact partials with the same packed weights.
    pub fn initLivePartials8(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        return initImpl(a, device, prepared, sk, order, true);
    }
    fn initImpl(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder, compact: bool) !Adapter {
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
        var compact_projection: ?lane.Projection = null;
        if (compact and order != null and sk == 8) {
            var live = layout;
            live.live_partials8 = true;
            compact_projection = try lane.Projection.init(a, device, live);
        }
        return .{ .projection = projection, .compact_projection = compact_projection, .words = words, .metadata = metadata };
    }
    pub fn deinit(adapter: Adapter) void {
        if (adapter.compact_projection) |p| p.deinit();
        adapter.projection.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
    }
    pub fn select(adapter: Adapter, rows: u32) lane.Projection {
        if (rows >= 1 and rows <= 8) if (adapter.compact_projection) |p| return p;
        return adapter.projection;
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try adapter.projection.encode(e, x, .{ .buf = adapter.words }, .{ .buf = adapter.metadata }, y, rows);
    }
};
