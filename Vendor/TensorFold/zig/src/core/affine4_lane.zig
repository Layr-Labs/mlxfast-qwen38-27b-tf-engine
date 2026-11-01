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
    /// A gate|up adapter's fused build for decode blocks (at most 8 rows): serial split-K, the target's SwiGLU epilogue
    /// (TF_GU_TURNS: gate and up tiles on two SIMD groups, silu(gate) * up from registers) writing the activated rows and
    /// their 64-column group sums, on the same words and indexed metadata.
    swiglu: ?lane.Projection = null,
    /// With `order`, the words are rewritten into reg_layout and encode on the reg kernel (same output words).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        if (prepared.nibbles) {
            // codes below 4 bits on the nibble grid: the reg kernel's 2-bit words (halveReg), no other path
            if (order == null or prepared.bits != 2 or prepared.group != 64) return error.UnsupportedNibbles;
        } else if (prepared.bits != 4 or prepared.group != 64) return initPlain(a, device, prepared, sk);
        var layout = lane.Layout{ .n = prepared.n, .k = prepared.k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = true, .cooperative = order == null, .reg = order != null, .reg_lean = order != null };
        try layout.validate();
        var words = try device.buffer(layout.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer words.deinit();
        const metadata = try device.buffer(layout.metadataElements() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer metadata.deinit();
        try lane.pack(layout, prepared.words, prepared.scales, prepared.biases, words.slice(u32, layout.weightWords()), metadata.slice(u16, layout.metadataElements()));
        if (order) |o| try o.run(&.{.{ .w = .{ .buf = words }, .n = prepared.n, .k = prepared.k }});
        // the reg kernel's scales carry 2^24 (lane.foldScale); the cooperative kernel reads them as packed
        if (order != null) try lane.foldScales(metadata.slice(u16, layout.metadataElements()));
        if (prepared.nibbles) {
            var two = layout;
            two.format.bits = 2;
            try two.validate();
            const half = try device.buffer(two.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            lane.halveReg(words.slice(u32, layout.weightWords()), half.slice(u32, two.weightWords())) catch |err| {
                half.deinit();
                return err;
            };
            words.deinit();
            words = half;
            layout = two;
        }
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
        errdefer if (small) |s| s.deinit();
        var swiglu: ?lane.Projection = null;
        if (layout.reg and layout.format.bits == 4 and layout.indexed and prepared.n % 128 == 0) {
            var ls = layout;
            ls.rows8 = true;
            ls.serial_sk2 = true;
            ls.sk = 2;
            ls.swiglu = true;
            if (ls.validate()) |_| {
                swiglu = try lane.Projection.init(a, device, ls);
            } else |_| {}
        }
        return .{ .projection = projection, .words = words, .metadata = metadata, .index = index, .small = small, .swiglu = swiglu };
    }
    /// A `bits` != 4 copy on the plain lane kernel (MLX bit stream codes, pairs metadata, precomputed row sums).
    fn initPlain(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize) !Adapter {
        var layout = lane.Layout{ .n = prepared.n, .k = prepared.k, .format = .{ .bits = prepared.bits, .group = prepared.group }, .sk = sk, .precompute_sums = true };
        // wider groups take more threadgroup memory a slice: fewer K slices where the split would not fit
        while (layout.sk > 1 and layout.threadgroupBytes() > 32768) layout.sk /= 2;
        try layout.validate();
        const words = try device.buffer(layout.weightWords() * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer words.deinit();
        const metadata = try device.buffer(layout.metadataElements() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer metadata.deinit();
        try lane.pack(layout, prepared.words, prepared.scales, prepared.biases, words.slice(u32, layout.weightWords()), metadata.slice(u16, layout.metadataElements()));
        const projection = try lane.Projection.init(a, device, layout);
        return .{ .projection = projection, .words = words, .metadata = metadata };
    }
    /// The projection a launch of `rows` takes: the rows8 build for at most 8 rows where there is one.
    pub fn forRows(adapter: Adapter, rows: u32) lane.Projection {
        if (rows <= 8) if (adapter.small) |s| return s;
        return adapter.projection;
    }
    pub fn deinit(adapter: Adapter) void {
        adapter.projection.deinit();
        if (adapter.small) |s| s.deinit();
        if (adapter.swiglu) |s| s.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
        if (adapter.index) |b| b.deinit();
    }
    /// The metadata the adapter's launches bind: the index where it has one, else the pairs.
    pub fn meta(adapter: Adapter) lane.Ref {
        return .{ .buf = adapter.index orelse adapter.metadata };
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        try adapter.forRows(rows).encode(e, x, .{ .buf = adapter.words }, adapter.meta(), y, rows);
    }
};
