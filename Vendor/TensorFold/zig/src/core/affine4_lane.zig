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
    /// Reg adapters' rows8 build of `projection` on the same words and metadata: launches of at most 8 rows (the
    /// drafter's decode blocks) skip each lane's padding row.
    small: ?lane.Projection = null,
    /// The layout `projection` was built from.
    layout: lane.Layout,
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
        errdefer projection.deinit();
        var small: ?lane.Projection = null;
        if (layout.reg) {
            var l8 = layout;
            l8.rows8 = true;
            small = try lane.Projection.init(a, device, l8);
        }
        return .{ .projection = projection, .words = words, .metadata = metadata, .index = index, .small = small, .layout = layout };
    }
    /// The projection a launch of `rows` takes: the rows8 build for at most 8 rows where there is one.
    pub fn forRows(adapter: Adapter, rows: u32) lane.Projection {
        if (rows <= 8) if (adapter.small) |s| return s;
        return adapter.projection;
    }
    pub fn deinit(adapter: Adapter) void {
        adapter.projection.deinit();
        if (adapter.small) |s| s.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
        if (adapter.index) |b| b.deinit();
    }
    /// The metadata the adapter's launches bind: the index where it has one, else the pairs.
    pub fn meta(adapter: Adapter) lane.Ref {
        return .{ .buf = adapter.index orelse adapter.metadata };
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try adapter.encodeOn(adapter.forRows(rows), e, x, y, rows);
    }
    /// The adapter's words through `p`, a projection of the adapter's layout (Prompts.get's).
    pub fn encodeOn(adapter: Adapter, p: lane.Projection, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try p.encode(e, x, .{ .buf = adapter.words }, adapter.meta(), y, rows);
    }
};

/// Prompt-row pipelines for reg adapters: each adapter layout's kernel at `lane.max_reg_rows` rows a launch, one pipeline
/// shared by every adapter of that layout (callers order launches on its owned sums buffer).
pub const Prompts = struct {
    const Key = struct { n: usize, k: usize, sk: usize, indexed: bool, indexed12: bool };
    map: std.AutoHashMapUnmanaged(Key, lane.Projection) = .empty,

    fn key(adapter: Adapter) Key {
        const l = adapter.layout;
        return .{ .n = l.n, .k = l.k, .sk = l.sk, .indexed = l.indexed, .indexed12 = l.indexed12 };
    }
    /// The adapter layout's prompt kernel (compiled once a layout); nothing for adapters off the reg kernel.
    pub fn add(ps: *Prompts, a: std.mem.Allocator, device: mtl.Device, adapter: Adapter) !void {
        if (!adapter.layout.reg) return;
        const got = try ps.map.getOrPut(a, key(adapter));
        if (got.found_existing) return;
        errdefer ps.map.removeByPtr(got.key_ptr);
        var l = adapter.layout;
        l.max_rows = lane.max_reg_rows;
        got.value_ptr.* = try lane.Projection.init(a, device, l);
    }
    pub fn get(ps: Prompts, adapter: Adapter) ?lane.Projection {
        return ps.map.get(key(adapter));
    }
    pub fn deinit(ps: *Prompts, a: std.mem.Allocator) void {
        var it = ps.map.valueIterator();
        while (it.next()) |p| p.deinit();
        ps.map.deinit(a);
    }
};
