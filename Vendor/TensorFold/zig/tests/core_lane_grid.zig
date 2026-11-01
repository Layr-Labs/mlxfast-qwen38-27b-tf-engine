//! Whole-frame prompt reg launches (lane.max_reg_rows rows a launch) against 128-row launches: every output byte of the
//! prompt kernels (one tile, serial split-K, tile ranges and SwiGLU; on tile-batched codes also TN 2 and 4 with one or two
//! K splits as SIMD groups, TN 2 on tile ranges and the TN 2 SwiGLU), own or producer sums, and of the drafter adapters'
//! lean kernel (K split up to 8; packed, indexed or indexed12 metadata).
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core_lane");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

const Bufs = struct { x: mtl.Buffer, w: mtl.Buffer, sb: mtl.Buffer, given: mtl.Buffer, stride: u32, k: usize, n: usize };

/// `rows` in launches of `step`, as quant_gpu's prompt loops encode them.
fn run(queue: mtl.Queue, p: lane.Projection, b: Bufs, external: bool, y: mtl.Buffer, hs: ?mtl.Buffer, rows: u32, step: u32) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    var first: u32 = 0;
    while (first < rows) : (first += step) {
        const n: u32 = @min(rows - first, step);
        const xr = lane.Ref{ .buf = b.x, .off = @as(usize, first) * b.k * 2 };
        const sums = lane.Sums{ .ref = .{ .buf = b.given, .off = @as(usize, first) * 4 }, .stride = b.stride };
        if (hs) |h| {
            try p.encodeSwiglu(e, xr, .{ .buf = b.w }, .{ .buf = b.sb }, sums, .{ .buf = y, .off = @as(usize, first) * (b.n / 2) * 2 }, .{ .buf = h, .off = @as(usize, first) * 4 }, n);
        } else {
            const yr = lane.Ref{ .buf = y, .off = @as(usize, first) * b.n * 2 };
            if (external) try p.encodeSums(e, xr, .{ .buf = b.w }, .{ .buf = b.sb }, sums, yr, n) else try p.encode(e, xr, .{ .buf = b.w }, .{ .buf = b.sb }, yr, n);
        }
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
}

/// Reg-ordered codes u32[N/32][K/64][256] into tile batches u32[N/32/B][K/64][B][256], as projection.prepareForced stores them.
fn batch(a: std.mem.Allocator, words: []u32, n: usize, k: usize, tb: usize) !void {
    const plain = try a.dupe(u32, words);
    const kg = k / 64;
    for (0..n / 32) |tile| for (0..kg) |g| {
        const to = ((tile / tb * kg + g) * tb + tile % tb) * 256;
        @memcpy(words[to .. to + 256], plain[(tile * kg + g) * 256 ..][0..256]);
    };
}

/// Metadata the weights carry: random pairs, or pairs from 16 scales and 16 biases (few enough for indexed12).
const Pairs = enum { random, few };

/// Every layout of `layouts` (sharing n, k and format) at max_reg_rows a launch against its 128-row launches; `index` binds the pairs
/// through lane_projection.indexMetadata as Adapter.init does.
fn check(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, order: lane.RegOrder, rng: std.Random, layouts: []const lane.Layout, pairs: Pairs, index: bool) !usize {
    const base = layouts[0];
    const n = base.n;
    const k = base.k;
    const max_rows = lane.max_reg_rows + 52; // past a frame, so a second long launch starts at a row offset
    const raw = try a.alloc(u32, base.weightWords());
    for (raw) |*v| v.* = rng.int(u32);
    const scales = try a.alloc(u16, n * (k / 64));
    const biases = try a.alloc(u16, n * (k / 64));
    for (scales, biases) |*s, *b| switch (pairs) {
        .random => {
            s.* = bf16(0.002 + rng.float(f32) * 0.02);
            b.* = bf16(-rng.float(f32) * 0.15);
        },
        .few => {
            s.* = bf16(0.002 + @as(f32, @floatFromInt(rng.uintLessThan(u32, 16))) * 0.00125);
            b.* = bf16(-@as(f32, @floatFromInt(rng.uintLessThan(u32, 16))) * 0.01);
        },
    };
    const w = try device.buffer(base.weightWords() * 4, opts);
    defer w.deinit();
    const packed_sb = try device.buffer(base.metadataElements() * 2, opts);
    defer packed_sb.deinit();
    try lane.pack(base, raw, scales, biases, w.slice(u32, raw.len), packed_sb.slice(u16, base.metadataElements()));
    try order.run(&.{.{ .w = .{ .buf = w }, .n = n, .k = k }});
    if (base.tile_batch != 1) try batch(a, w.slice(u32, raw.len), n, k, base.tile_batch);
    var twelve = false;
    var sb = packed_sb;
    var indexed: ?mtl.Buffer = null;
    defer if (indexed) |b| b.deinit();
    if (index) {
        const got = (try lane.indexMetadata(device, a, packed_sb.slice(u16, base.metadataElements()), n, k)) orelse return error.NotIndexed;
        indexed = got.buffer;
        sb = got.buffer;
        twelve = got.twelve;
        if (twelve != (pairs == .few)) return error.UnexpectedIndexForm;
    }
    const x = try device.buffer(max_rows * k * 2, opts);
    defer x.deinit();
    const xs = x.slice(u16, max_rows * k);
    for (xs) |*v| v.* = bf16((rng.float(f32) - 0.5) * 4);
    const stride: u32 = 16 * ((max_rows + 15) / 16);
    const given = try device.buffer((k / 64) * stride * 4, opts);
    defer given.deinit();
    const g = given.slice(f32, (k / 64) * stride);
    @memset(g, 0);
    for (0..max_rows) |m| for (0..k / 64) |grp| {
        var acc: f32 = 0;
        for (0..64) |i| acc += @as(f32, @bitCast(@as(u32, xs[m * k + grp * 64 + i]) << 16));
        g[grp * stride + m] = acc;
    };
    const bufs = Bufs{ .x = x, .w = w, .sb = sb, .given = given, .stride = stride, .k = k, .n = n };
    const y_bytes = max_rows * n * 2;
    const hs_bytes = (n / 128) * stride * 4;
    const want = try device.buffer(y_bytes, opts);
    defer want.deinit();
    const got = try device.buffer(y_bytes, opts);
    defer got.deinit();
    const want_hs = try device.buffer(hs_bytes, opts);
    defer want_hs.deinit();
    const got_hs = try device.buffer(hs_bytes, opts);
    defer got_hs.deinit();
    var cases: usize = 0;
    for (layouts, 0..) |given_layout, which| {
        var layout = given_layout;
        layout.indexed = index;
        layout.indexed12 = twelve;
        const is_swiglu = layout.swiglu;
        const ref = try lane.Projection.init(a, device, layout);
        defer ref.deinit();
        var long_layout = layout;
        long_layout.max_rows = lane.max_reg_rows;
        const long = try lane.Projection.init(a, device, long_layout);
        defer long.deinit();
        for ([_]u32{ 1, 17, 128, 300, 1065, max_rows }) |rows| for ([_]bool{ false, true }) |external| {
            if (is_swiglu and !external) continue;
            @memset(want.contents()[0..y_bytes], 0xa5);
            @memset(want_hs.contents()[0..hs_bytes], 0xa5);
            try run(queue, ref, bufs, external, want, if (is_swiglu) want_hs else null, rows, lane.reg_rows);
            @memset(got.contents()[0..y_bytes], 0xa5);
            @memset(got_hs.contents()[0..hs_bytes], 0xa5);
            try run(queue, long, bufs, external, got, if (is_swiglu) got_hs else null, rows, lane.max_reg_rows);
            const same = std.mem.eql(u8, got.contents()[0..y_bytes], want.contents()[0..y_bytes]) and
                (!is_swiglu or std.mem.eql(u8, got_hs.contents()[0..hs_bytes], want_hs.contents()[0..hs_bytes]));
            if (!same) {
                std.debug.print("DIFFER n={d} k={d} layout={d} index={} twelve={} rows={d} external={}\n", .{ n, k, which, index, twelve, rows, external });
                return error.GridBytesDiffer;
            }
            cases += 1;
        };
    }
    return cases;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.TensorUnitsRequired;
    const queue = try device.queue();
    defer queue.deinit();
    const order = try lane.RegOrder.init(device);
    defer order.deinit();
    var prng = std.Random.DefaultPrng.init(0xa1);
    const rng = prng.random();
    // qwen27's prompt kernels on unbatched codes: one tile, K split 2 (parallel and serial), tile ranges, the 128-thread SwiGLU
    const plain = lane.Layout{ .n = 512, .k = 1024, .format = .{ .bits = 4, .group = 64 }, .precompute_sums = true, .reg = true, .reg_lean = true, .prompt_sr = true };
    var sk2 = plain;
    sk2.sk = 2;
    var serial = sk2;
    serial.serial_sk2 = true;
    var ranged = plain;
    ranged.ranges = &.{ .{ 2, 5 }, .{ 9, 4 } };
    var swiglu = plain;
    swiglu.swiglu = true;
    var cases = try check(a, device, queue, order, rng, &.{ plain, sk2, serial, ranged, swiglu }, .random, false);
    // on 8-tile code batches (quant_gpu.promptTiles' layouts): TN 2 and 4, one K split or two as SIMD groups of one
    // threadgroup, TN 2 on one or two tile ranges, the 64-thread TN 2 SwiGLU, and the one-tile kernels beside them
    var b8 = plain;
    b8.tile_batch = 8;
    var b8_serial = serial;
    b8_serial.tile_batch = 8;
    var tn2 = b8;
    tn2.prompt_tn = 2;
    var tn2_sk2 = tn2;
    tn2_sk2.sk = 2;
    var tn4_sk2 = tn2_sk2;
    tn4_sk2.prompt_tn = 4;
    var tn2_range = tn2;
    tn2_range.ranges = &.{.{ 4, 8 }};
    var tn2_ranges = tn2_sk2;
    tn2_ranges.ranges = &.{ .{ 2, 6 }, .{ 10, 4 } };
    var b8_swiglu = b8;
    b8_swiglu.swiglu = true;
    var tn2_swiglu = b8_swiglu;
    tn2_swiglu.prompt_tn = 2;
    cases += try check(a, device, queue, order, rng, &.{ b8, b8_serial, tn2, tn2_sk2, tn4_sk2, tn2_range, tn2_ranges, b8_swiglu, tn2_swiglu }, .random, false);
    // the drafter adapters' kernel (affine4_lane.Adapter.init's layout): K split up to 8, packed or indexed pairs
    const adapter = lane.Layout{ .n = 512, .k = 4096, .format = .{ .bits = 4, .group = 64 }, .precompute_sums = true, .reg = true, .reg_lean = true };
    var adapters: [4]lane.Layout = undefined;
    for (&adapters, [_]usize{ 1, 2, 4, 8 }) |*l, sk| {
        l.* = adapter;
        l.sk = sk;
    }
    cases += try check(a, device, queue, order, rng, &adapters, .random, false);
    cases += try check(a, device, queue, order, rng, &adapters, .random, true);
    cases += try check(a, device, queue, order, rng, &adapters, .few, true);
    std.debug.print("GRID CHECK PASS cases={d}: every output byte equals the 128-row launches\n", .{cases});
}
