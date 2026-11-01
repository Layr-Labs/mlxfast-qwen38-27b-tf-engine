//! Prepared affine q4 words use the existing shared lane packing and encode contract.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
pub const cpu = @import("affine4.zig");
/// Draft adapters' decoded rows (at most 8) on a 2-bit re-quantization of their q4 words (Adapter.words2).
pub const draft_reg2 = true;
pub const Adapter = struct {
    projection: lane.Projection,
    words: mtl.Buffer,
    metadata: mtl.Buffer,
    /// Reg adapters' (scale, bias) pairs through lane_projection.indexMetadata, when they fit its table.
    index: ?mtl.Buffer = null,
    /// Reg adapters' rows8 build of `projection` on the same words and metadata: launches of at most 8 rows (the
    /// drafter's decode blocks) skip each lane's padding row.
    small: ?lane.Projection = null,
    /// The rows8 build's own 2-bit words (reg2_layout), pairs and index (draft_reg2): `small` reads these, not `words`.
    words2: ?mtl.Buffer = null,
    metadata2: ?mtl.Buffer = null,
    index2: ?mtl.Buffer = null,
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
        // the reg kernel's scales carry 2^24 (lane.foldScale); the cooperative kernel reads them as packed
        if (order != null) try lane.foldScales(metadata.slice(u16, layout.metadataElements()));
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
        var words2: ?mtl.Buffer = null;
        errdefer if (words2) |b| b.deinit();
        var metadata2: ?mtl.Buffer = null;
        errdefer if (metadata2) |b| b.deinit();
        var index2: ?mtl.Buffer = null;
        errdefer if (index2) |b| b.deinit();
        if (layout.reg and draft_reg2) {
            // decoded rows (at most 8) read a 2-bit re-quantization: half the code bytes; the drafter only proposes
            var p2 = try cpu.requant2(a, prepared);
            defer p2.deinit();
            var l2 = lane.Layout{ .n = prepared.n, .k = prepared.k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = true, .reg = true, .reg_lean = true };
            const tmp = try device.buffer(l2.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            defer tmp.deinit();
            metadata2 = try device.buffer(l2.metadataElements() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            try lane.pack(l2, p2.words, p2.scales, p2.biases, tmp.slice(u32, l2.weightWords()), metadata2.?.slice(u16, l2.metadataElements()));
            try order.?.run(&.{.{ .w = .{ .buf = tmp }, .n = prepared.n, .k = prepared.k }});
            try lane.foldScales(metadata2.?.slice(u16, l2.metadataElements()));
            if (try lane.indexMetadata(device, a, metadata2.?.slice(u16, l2.metadataElements()), prepared.n, prepared.k)) |x| {
                index2 = x.buffer;
                l2.indexed = true;
                l2.indexed12 = x.twelve;
            }
            const four = l2.weightWords();
            l2.rows8 = true;
            l2.reg2 = true;
            try l2.validate();
            words2 = try device.buffer(l2.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            try lane.packReg2(tmp.slice(u32, four), words2.?.slice(u32, l2.weightWords()), prepared.n, prepared.k);
            small = try lane.Projection.init(a, device, l2);
        } else if (layout.reg) {
            var l8 = layout;
            l8.rows8 = true;
            small = try lane.Projection.init(a, device, l8);
        }
        return .{ .projection = projection, .words = words, .metadata = metadata, .index = index, .small = small, .words2 = words2, .metadata2 = metadata2, .index2 = index2 };
    }
    /// The words a launch of `rows` binds: the 2-bit build's for at most 8 rows where there is one.
    pub fn wordsFor(adapter: Adapter, rows: u32) mtl.Buffer {
        if (rows <= 8) if (adapter.words2) |w| return w;
        return adapter.words;
    }
    /// The metadata a launch of `rows` binds (as `meta`, for the build `forRows` picks).
    pub fn metaFor(adapter: Adapter, rows: u32) lane.Ref {
        if (rows <= 8) if (adapter.metadata2) |m| return .{ .buf = adapter.index2 orelse m };
        return adapter.meta();
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
        if (adapter.words2) |b| b.deinit();
        if (adapter.metadata2) |b| b.deinit();
        if (adapter.index2) |b| b.deinit();
    }
    /// The metadata the adapter's launches bind: the index where it has one, else the pairs.
    pub fn meta(adapter: Adapter) lane.Ref {
        return .{ .buf = adapter.index orelse adapter.metadata };
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try adapter.forRows(rows).encode(e, x, .{ .buf = adapter.wordsFor(rows) }, adapter.metaFor(rows), y, rows);
    }
};
