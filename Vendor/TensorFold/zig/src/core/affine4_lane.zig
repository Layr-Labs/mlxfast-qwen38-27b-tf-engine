//! Prepared affine q4 words use the existing shared lane packing and encode contract.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
pub const cpu = @import("affine4.zig");
/// Per-shape-class verdicts for the sequential metadata layout: true adopts it, false keeps native.
/// Key: n | k << 32 | sk << 64 | bits << 72 | nibbles << 80 | form << 88
/// (form: 0 pairs, 1 indexed, 2 indexed12). One per Backend.prepare, shared by every linear it prepares.
pub const SeqCache = std.AutoHashMapUnmanaged(u128, bool);
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
    /// 2-bit reg adapters' rows8 build on an 8-row tensor op (Layout.m8), taken in place of `small` only after its words
    /// equal `small`'s on the GPU at init (agrees8): the 16-row op spends 16 rows of tensor work a code on these
    /// tensor-bound launches of 1 to 8 rows.
    small8: ?lane.Projection = null,
    /// Reg adapters' prompt build on the same words and the (folded) pairs: launches of more than 16 rows (a prompt's
    /// absorb) take the target's prompt loop (prompt_sr: pointer-stepped codes, pairs and sums), the same words out.
    prompt: ?lane.Projection = null,
    /// True when the metadata (pairs and index alike) is the sequential tile-major repack and every build above reads
    /// it as such (Layout.seq_meta); false is the native layout. Set only by initStream after bit-verification.
    seq: bool = false,
    /// With `order`, the words are rewritten into reg_layout and encode on the reg kernel (same output words).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Adapter {
        if (prepared.nibbles) {
            // codes below 4 bits on the nibble grid: the reg kernel's 2-bit words (halveReg), no other path
            if (order == null or prepared.bits != 2 or prepared.group != 64) return error.UnsupportedNibbles;
        } else if (prepared.bits != 4 or prepared.group != 64) return initPlain(a, device, prepared, sk);
        return assembleNative(a, device, try packPrepared(a, device, prepared, sk, order), prepared.nibbles);
    }

    /// Native packed buffers: the words every launch reads plus the metadata (pairs, and the index where one fits).
    const Packed = struct {
        layout: lane.Layout,
        words: mtl.Buffer,
        metadata: mtl.Buffer,
        index: ?mtl.Buffer = null,
        fn meta(p: Packed) mtl.Buffer {
            return p.index orelse p.metadata;
        }
        fn deinit(p: Packed) void {
            p.words.deinit();
            p.metadata.deinit();
            if (p.index) |ix| ix.deinit();
        }
    };

    /// Native weights into Metal buffers: packed codes, the reg rewrite, folded scales, the halved 2-bit words and the
    /// index where one fits. The words are metadata-order-free, so native and sequential builds share them.
    fn packPrepared(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder) !Packed {
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
        return .{ .layout = layout, .words = words, .metadata = metadata, .index = index };
    }

    /// An adapter's launches for one metadata order: the decode, counted and prompt builds on the same words.
    const Built = struct {
        projection: lane.Projection,
        small: ?lane.Projection = null,
        small_count: ?lane.Projection = null,
        small8: ?lane.Projection = null,
        prompt: ?lane.Projection = null,
        fn deinit(b: Built) void {
            if (b.prompt) |p| p.deinit();
            if (b.small8) |s| s.deinit();
            if (b.small_count) |s| s.deinit();
            if (b.small) |s| s.deinit();
            b.projection.deinit();
        }
    };

    /// Every launch for `layout` on `words` and `meta` (the index where the layout has one, else the pairs): the same
    /// builds init always made, now shared by the native and sequential variants of one adapter. `native_meta` null
    /// is a native layout (the m8 verdict comes from agrees8 on its own builds, init's behavior); set, it is a
    /// sequential layout whose m8 verdict comes from a native pair instead (small8NativeOk), so the decision
    /// launches verified kernels only.
    fn buildAll(a: std.mem.Allocator, device: mtl.Device, layout: lane.Layout, words: mtl.Buffer, meta_buf: mtl.Buffer, nibbles: bool, native_meta: ?mtl.Buffer) !Built {
        var b: Built = .{ .projection = try lane.Projection.init(a, device, layout) };
        errdefer b.deinit();
        if (layout.reg) {
            var l8 = layout;
            l8.rows8 = true;
            b.small = try lane.Projection.init(a, device, l8);
            var lc = l8;
            lc.count = true;
            // a layout the counted build refuses keeps the uncounted launch only
            b.small_count = lane.Projection.init(a, device, lc) catch null;
        }
        if (layout.reg and nibbles) if (b.small) |s| {
            var l8 = layout;
            l8.rows8 = true;
            l8.m8 = true;
            if (native_meta) |nm| {
                if (small8NativeOk(a, device, layout, words, nm)) {
                    if (lane.Projection.init(a, device, l8)) |p8| b.small8 = p8 else |_| {}
                }
            } else if (lane.Projection.init(a, device, l8)) |p8| {
                if (agrees8(device, s, p8, words, meta_buf, layout.k) catch false) b.small8 = p8 else p8.deinit();
            } else |_| {}
        };
        if (layout.reg and layout.format.bits == 4) {
            var lp = layout;
            lp.indexed = false;
            lp.indexed12 = false;
            lp.prompt_sr = true;
            b.prompt = try lane.Projection.init(a, device, lp);
        }
        return b;
    }

    fn assemble(words: mtl.Buffer, metadata: mtl.Buffer, index: ?mtl.Buffer, built: Built, seq: bool) Adapter {
        return .{ .projection = built.projection, .words = words, .metadata = metadata, .index = index, .small = built.small, .small_count = built.small_count, .small8 = built.small8, .prompt = built.prompt, .seq = seq };
    }

    /// Packed buffers plus their native launches: init's result, and initStream's fallback on any mismatch.
    fn assembleNative(a: std.mem.Allocator, device: mtl.Device, src: Packed, nibbles: bool) !Adapter {
        errdefer src.deinit();
        const built = try buildAll(a, device, src.layout, src.words, src.meta(), nibbles, null);
        return assemble(src.words, src.metadata, src.index, built, false);
    }

    /// Whether a 2-bit `layout`'s rows8 build takes the m8 build: agrees8 on a NATIVE small/m8 pair over
    /// `native_meta` (init's own launches, on verified kernels). The verdict carries to a sequential layout's
    /// own pair: m8 and rows8 differ only in their tensor op, never in metadata addressing, so their
    /// equivalence is metadata-order-free — and no unverified kernel launches to decide it.
    fn small8NativeOk(a: std.mem.Allocator, device: mtl.Device, layout: lane.Layout, words: mtl.Buffer, native_meta: mtl.Buffer) bool {
        var nl = layout;
        nl.seq_meta = false;
        nl.rows8 = true;
        const ns = lane.Projection.init(a, device, nl) catch return false;
        defer ns.deinit();
        var nm8l = nl;
        nm8l.m8 = true;
        const nm8 = lane.Projection.init(a, device, nm8l) catch return false;
        defer nm8.deinit();
        return agrees8(device, ns, nm8, words, native_meta, layout.k) catch false;
    }

    fn seqKey(n: usize, k: usize, sk: usize, bits: u8, nibbles: bool, form: u2) u128 {
        return @as(u128, @intCast(n)) | (@as(u128, @intCast(k)) << 32) | (@as(u128, @intCast(sk)) << 64) | (@as(u128, bits) << 72) | (@as(u128, @intFromBool(nibbles)) << 80) | (@as(u128, form) << 88);
    }

    fn seqFormName(form: u2) []const u8 {
        return if (form == 2) "indexed12" else if (form == 1) "indexed" else "pairs";
    }

    /// Class-A shapes: reg-path adapters (4-bit g64, or 2-bit nibbles g64) whose every launch shares the sequential
    /// addressing. Anything else (plain, cooperative, SIMD tiles) keeps its incumbent path untouched.
    fn seqClass(prepared: cpu.Prepared, order: ?lane.RegOrder) bool {
        if (order == null) return false;
        if (prepared.nibbles) return prepared.bits == 2 and prepared.group == 64;
        return prepared.bits == 4 and prepared.group == 64;
    }

    /// Sequential metadata buffers: tile-major pairs, and the tile-major index where the native one has one.
    const SeqBufs = struct {
        pairs: mtl.Buffer,
        index: ?mtl.Buffer = null,
        fn meta(sb: SeqBufs) mtl.Buffer {
            return sb.index orelse sb.pairs;
        }
        fn deinit(sb: SeqBufs) void {
            sb.pairs.deinit();
            if (sb.index) |ix| ix.deinit();
        }
    };

    /// An index buffer's body past its table (mirror of lane.indexMetadata's slicing).
    fn indexBody16(buf: mtl.Buffer, cap: usize, count: usize) []u16 {
        return @as([*]u16, @ptrCast(@alignCast(buf.contents() + cap * 4)))[0..count];
    }
    fn indexBody12(buf: mtl.Buffer, cap: usize, count: usize) []u32 {
        return @as([*]u32, @ptrCast(@alignCast(buf.contents() + cap * 4)))[0..count];
    }

    /// init with a load-time repack into the sequential metadata layout (lane.seq_metadata_layout): class-A adapters
    /// transpose their metadata here and adopt it only when the transpose rebuilds every weight bit-for-bit
    /// (seqRebuild), so the new buffers are an exact permutation of the native ones. The check is CPU-only by
    /// design: verification must never take down the engine, and no unverified kernel launches before adoption
    /// (the 2-bit m8 verdict comes from a native pair instead: small8NativeOk). Any failure keeps the native
    /// layout for the shape class. The whole attempt runs at load, outside all timed windows, and changes only
    /// address arithmetic on the hot path, so it cannot add serialized work by construction.
    pub fn initStream(a: std.mem.Allocator, device: mtl.Device, prepared: cpu.Prepared, sk: usize, order: ?lane.RegOrder, cache: *SeqCache) !Adapter {
        if (!seqClass(prepared, order)) return init(a, device, prepared, sk, order);
        const src = try packPrepared(a, device, prepared, sk, order);
        const form: u2 = if (src.layout.indexed12) 2 else if (src.layout.indexed) 1 else 0;
        const key = seqKey(prepared.n, prepared.k, sk, src.layout.format.bits, prepared.nibbles, form);
        if (cache.get(key)) |verdict| if (!verdict) return assembleNative(a, device, src, prepared.nibbles);
        var seq_bufs: ?SeqBufs = seqTranspose(device, src) catch null;
        if (seq_bufs) |sb| {
            const rebuilt = seqRebuild(a, src, sb) catch false;
            if (!rebuilt) {
                sb.deinit();
                seq_bufs = null;
            }
        }
        const sb = seq_bufs orelse {
            cache.put(a, key, false) catch {};
            std.log.warn("r33: [{d}, {d}] {s} rebuild mismatch, native metadata kept", .{ prepared.n, prepared.k, seqFormName(form) });
            return assembleNative(a, device, src, prepared.nibbles);
        };
        var seq_layout = src.layout;
        seq_layout.seq_meta = true;
        const built = buildAll(a, device, seq_layout, src.words, sb.meta(), prepared.nibbles, src.meta()) catch {
            sb.deinit();
            cache.put(a, key, false) catch {};
            std.log.warn("r33: [{d}, {d}] {s} sequential build failed, native metadata kept", .{ prepared.n, prepared.k, seqFormName(form) });
            return assembleNative(a, device, src, prepared.nibbles);
        };
        src.metadata.deinit();
        if (src.index) |ix| ix.deinit();
        cache.put(a, key, true) catch {};
        std.log.info("r33: [{d}, {d}] {s} sequential metadata adopted", .{ prepared.n, prepared.k, seqFormName(form) });
        return assemble(src.words, sb.pairs, sb.index, built, true);
    }

    /// Native metadata into sequential buffers: pairs transposed, the index's table copied verbatim and its body
    /// transposed. Same byte sizes; a permutation, verified by seqRebuild before anything reads it.
    fn seqTranspose(device: mtl.Device, src: Packed) !SeqBufs {
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const n = src.layout.n;
        const k = src.layout.k;
        const group = src.layout.format.group;
        const pairs = try device.buffer(src.metadata.length(), options);
        errdefer pairs.deinit();
        try lane.seqPairs(n, k, group, src.metadata.slice(u16, src.layout.metadataElements()), pairs.slice(u16, src.layout.metadataElements()));
        var index: ?mtl.Buffer = null;
        errdefer if (index) |ix| ix.deinit();
        if (src.index) |native| {
            const seq = try device.buffer(native.length(), options);
            if (src.layout.indexed12) {
                if (native.length() != lane.indexed12Bytes(n, k, group)) return error.InvalidStorage;
                @memcpy(seq.slice(u32, lane.sb_table12), native.slice(u32, lane.sb_table12));
                const count = (k / group) * (n / 32) * 12;
                try lane.seqIndex12(n, k, group, indexBody12(native, lane.sb_table12, count), indexBody12(seq, lane.sb_table12, count));
            } else {
                if (!src.layout.indexed or native.length() != lane.indexedBytes(n, k, group)) return error.InvalidStorage;
                @memcpy(seq.slice(u32, lane.sb_table), native.slice(u32, lane.sb_table));
                const count = n * (k / group);
                try lane.seqIndex(n, k, group, indexBody16(native, lane.sb_table, count), indexBody16(seq, lane.sb_table, count));
            }
            index = seq;
        }
        return .{ .pairs = pairs, .index = index };
    }

    /// Every weight rebuilt bit-for-bit: the sequential buffers transpose back into native order and compare equal,
    /// tables included. False keeps the native layout for this shape.
    fn seqRebuild(a: std.mem.Allocator, src: Packed, seq: SeqBufs) !bool {
        const n = src.layout.n;
        const k = src.layout.k;
        const group = src.layout.format.group;
        const native_p = src.metadata.slice(u16, src.layout.metadataElements());
        const back = try a.alloc(u16, native_p.len);
        defer a.free(back);
        try lane.unseqPairs(n, k, group, seq.pairs.slice(u16, native_p.len), back);
        if (!std.mem.eql(u16, back, native_p)) return false;
        if (src.index) |native| {
            const six = seq.index orelse return false;
            if (src.layout.indexed12) {
                if (!std.mem.eql(u32, six.slice(u32, lane.sb_table12), native.slice(u32, lane.sb_table12))) return false;
                const count = (k / group) * (n / 32) * 12;
                const tmp = try a.alloc(u32, count);
                defer a.free(tmp);
                try lane.unseqIndex12(n, k, group, indexBody12(six, lane.sb_table12, count), tmp);
                if (!std.mem.eql(u32, tmp, indexBody12(native, lane.sb_table12, count))) return false;
            } else {
                if (!std.mem.eql(u32, six.slice(u32, lane.sb_table), native.slice(u32, lane.sb_table))) return false;
                const count = n * (k / group);
                const tmp = try a.alloc(u16, count);
                defer a.free(tmp);
                try lane.unseqIndex(n, k, group, indexBody16(six, lane.sb_table, count), tmp);
                if (!std.mem.eql(u16, tmp, indexBody16(native, lane.sb_table, count))) return false;
            }
        } else if (seq.index != null) return false;
        return true;
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
