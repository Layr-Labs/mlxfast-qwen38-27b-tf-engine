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
    /// `small` built with Layout.count: the same launch, each threadgroup adding one to a counter word once its rows are
    /// stored, so a launch that reads them can share its barrier interval (lane.Projection.encodeCounted).
    small_count: ?lane.Projection = null,
    /// Selected 2-bit and q4 down adapters' rows8 build on an 8-row tensor op (Layout.m8), taken in place of `small`
    /// only after its words equal `small`'s on the GPU at init (agrees8).
    small8: ?lane.Projection = null,
    /// Reg adapters' prompt build on the same words and the (folded) pairs: launches of more than 16 rows (a prompt's
    /// absorb) take the target's prompt loop (prompt_sr: pointer-stepped codes, pairs and sums), the same words out.
    prompt: ?lane.Projection = null,
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
        var small_count: ?lane.Projection = null;
        if (layout.reg) {
            var l8 = layout;
            l8.rows8 = true;
            small = try lane.Projection.init(a, device, l8);
            var lc = l8;
            lc.count = true;
            // a layout the counted build refuses keeps the uncounted launch only
            small_count = lane.Projection.init(a, device, lc) catch null;
        }
        errdefer if (small) |s| s.deinit();
        errdefer if (small_count) |s| s.deinit();
        var small8: ?lane.Projection = null;
        // The stock drafter's q4 down projection is 5120 x 17408. Restrict extra probes to this shape;
        // other q4 adapters and all existing 2-bit behavior retain their previous paths.
        const q4_down = !prepared.nibbles and prepared.bits == 4 and prepared.n == 5120 and prepared.k == 17408;
        if (layout.reg and (prepared.nibbles or q4_down)) if (small) |s| {
            var l8 = layout;
            l8.rows8 = true;
            l8.m8 = true;
            if (lane.Projection.init(a, device, l8)) |p8| {
                if (agrees8(device, s, p8, words, if (index) |ix| ix else metadata, layout.k) catch false) {
                    small8 = p8;
                    // Keep counted and uncounted outputs on the same verified descriptor when a counted
                    // m8 build is supported; otherwise leave the existing counted pipeline untouched.
                    if (q4_down and small_count != null) {
                        var lc = l8;
                        lc.count = true;
                        if (lane.Projection.init(a, device, lc)) |count8| {
                            small_count.?.deinit();
                            small_count = count8;
                        } else |_| {}
                    }
                } else p8.deinit();
            } else |_| {}
        };
        errdefer if (small8) |s| s.deinit();
        var prompt: ?lane.Projection = null;
        if (layout.reg and layout.format.bits == 4) {
            var lp = layout;
            lp.indexed = false;
            lp.indexed12 = false;
            lp.prompt_sr = true;
            prompt = try lane.Projection.init(a, device, lp);
        }
        return .{ .projection = projection, .words = words, .metadata = metadata, .index = index, .small = small, .small_count = small_count, .small8 = small8, .prompt = prompt };
    }
    /// Whether `p8` writes `small`'s words bit for bit on these weights: both launched on the same rows (deterministic
    /// pseudo-random bf16 activations in [-2, 2)) at every row count a rows8 launch takes, 1 to 8, their outputs compared
    /// on the host.
    fn agrees8(device: mtl.Device, small: lane.Projection, p8: lane.Projection, words: mtl.Buffer, meta_buf: mtl.Buffer, k: usize) !bool {
        const max_rows: usize = 8;
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const x = try device.buffer(max_rows * k * 2, options);
        defer x.deinit();
        const y_bytes = max_rows * small.y_stride * small.output_bytes;
        const y16 = try device.buffer(y_bytes, options);
        defer y16.deinit();
        const y8 = try device.buffer(y_bytes, options);
        defer y8.deinit();
        var state: u32 = 0x9E3779B9;
        for (x.slice(u16, max_rows * k)) |*v| {
            state = state *% 1664525 +% 1013904223;
            const f = (@as(f32, @floatFromInt(state >> 8)) / 16777216.0) * 4.0 - 2.0;
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        const queue = try device.queue();
        defer queue.deinit();
        for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 }) |rows| {
            @memset(y16.contents()[0..y_bytes], 0xA5);
            @memset(y8.contents()[0..y_bytes], 0x5A);
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            try small.encode(e, .{ .buf = x }, .{ .buf = words }, .{ .buf = meta_buf }, .{ .buf = y16 }, rows);
            e.barrier();
            try p8.encode(e, .{ .buf = x }, .{ .buf = words }, .{ .buf = meta_buf }, .{ .buf = y8 }, rows);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.status() != .completed) return false;
            const used = ((rows - 1) * small.y_stride + small.n) * small.output_bytes;
            if (!std.mem.eql(u8, y16.contents()[0..used], y8.contents()[0..used])) return false;
        }
        return true;
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
        if (rows <= 8) if (adapter.small8 orelse adapter.small) |s| return s;
        return adapter.projection;
    }
    pub fn deinit(adapter: Adapter) void {
        adapter.projection.deinit();
        if (adapter.small) |s| s.deinit();
        if (adapter.small_count) |s| s.deinit();
        if (adapter.small8) |s| s.deinit();
        if (adapter.prompt) |p| p.deinit();
        adapter.words.deinit();
        adapter.metadata.deinit();
        if (adapter.index) |b| b.deinit();
    }
    /// The metadata the adapter's launches bind: the index where it has one, else the pairs.
    pub fn meta(adapter: Adapter) lane.Ref {
        return .{ .buf = adapter.index orelse adapter.metadata };
    }
    pub fn encode(adapter: Adapter, e: mtl.ComputeEncoder, x: lane.Ref, y: lane.Ref, rows: u32) !void {
        // prompt-sized launches: the prompt loop on the pairs (its words are the indexed launch's)
        if (rows > 16) if (adapter.prompt) |p| return p.encode(e, x, .{ .buf = adapter.words }, .{ .buf = adapter.metadata }, y, rows);
        try adapter.forRows(rows).encode(e, x, .{ .buf = adapter.words }, adapter.meta(), y, rows);
    }
};
