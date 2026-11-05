//! Load-time comparison of complete target DOWN weight streams, after output qualification.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
const policy = @import("decode_m8_policy.zig");

pub const matrices = 64;
const rows = 8;
const sums_stride = 16;
const n = 5120;
const k = 17408;
const guard_bytes = 32;
const payload_bytes = rows * n * 2;
const block_bytes = payload_bytes + 2 * guard_bytes;

/// Borrowed pipelines and immutable model buffers. The caller retains their ownership.
pub const Program = struct {
    layer: usize,
    baseline: lane.Projection,
    candidate: lane.Projection,
    words: lane.Ref,
    metadata: lane.Ref,
};

pub const Evidence = struct {
    decision: policy.Decision,
    m16: [policy.cohort_pairs]f64,
    m8: [policy.cohort_pairs]f64,
    m16_null: [policy.null_pairs_per_arm]policy.NullPair,
    m8_null: [policy.null_pairs_per_arm]policy.NullPair,
};

pub fn run(device: mtl.Device, queue: mtl.Queue, programs: []const Program, live_rows: u32) !Evidence {
    if (live_rows == 0 or live_rows > rows) return error.InvalidM8Cohort;
    if (programs.len != matrices) return error.IncompleteM8Cohort;
    for (programs, 0..) |p, i| {
        if (p.layer != i or p.baseline.m8 or !p.candidate.m8) return error.InvalidM8Cohort;
        for ([_]lane.Projection{ p.baseline, p.candidate }) |v| {
            if (v.n != n or v.k != k or v.y_stride != n or v.output_bytes != 2 or !v.reg or !v.rows8 or !v.count or !v.wait)
                return error.InvalidM8Cohort;
        }
        if (p.baseline.weight_bytes != p.candidate.weight_bytes or p.baseline.metadata_bytes != p.candidate.metadata_bytes or
            p.baseline.tile_count != p.candidate.tile_count or p.baseline.sk != p.candidate.sk) return error.InvalidM8Cohort;
        if (p.words.off > p.words.buf.length() or p.baseline.weight_bytes > p.words.buf.length() - p.words.off)
            return error.InvalidM8Cohort;
        for (programs[0..i]) |previous| {
            if (p.words.buf.id == previous.words.buf.id) {
                if (p.words.off >= previous.words.off) {
                    if (p.words.off - previous.words.off < previous.baseline.weight_bytes) return error.DuplicateM8Cohort;
                } else if (previous.words.off - p.words.off < p.baseline.weight_bytes) return error.DuplicateM8Cohort;
            }
        }
    }
    const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const x = try device.buffer(rows * k * 2, options);
    defer x.deinit();
    const xs = try device.buffer((k / 64) * sums_stride * 4, options);
    defer xs.deinit();
    const y = try device.buffer(matrices * block_bytes, options);
    defer y.deinit();
    const counted = try device.buffer(matrices * 8, options);
    defer counted.deinit();
    const waiting = try device.buffer(matrices * 8, options);
    defer waiting.deinit();
    const probe = Probe{ .queue = queue, .programs = programs, .x = x, .xs = xs, .y = y, .counted = counted, .waiting = waiting, .live_rows = live_rows };
    var expected: [2][32]u8 = undefined;
    // Both complete streams warm equally. Every invocation still checks every output block.
    for (0..2) |seed| {
        probe.inputs(@intCast(seed));
        for (0..3) |warm| {
            const base = try probe.measure(false);
            if (warm == 0) expected[seed] = base.hash else if (!std.mem.eql(u8, &expected[seed], &base.hash)) return error.M8CalibrationIdentity;
            const candidate = try probe.measure(true);
            if (!std.mem.eql(u8, &expected[seed], &candidate.hash)) return error.M8CalibrationIdentity;
        }
    }
    var result: Evidence = undefined;
    for (0..policy.null_pairs_per_arm) |i| {
        const seed = i % 2;
        probe.inputs(@intCast(seed));
        // Balance which complete null series runs first, in addition to the primary AB/BA order.
        for (0..2) |turn| {
            const candidate = (turn + i) % 2 != 0;
            const first = try probe.checked(candidate, expected[seed]);
            const second = try probe.checked(candidate, expected[seed]);
            const pair: policy.NullPair = .{ .first_seconds = first, .second_seconds = second };
            if (candidate) result.m8_null[i] = pair else result.m16_null[i] = pair;
            std.log.info("qwen27: M8 null arm={s} pair={d} seconds={d:.9},{d:.9}", .{ if (candidate) "M8" else "M16", i, first, second });
        }
    }
    for (0..policy.cohort_pairs) |i| {
        const seed = i % 2;
        probe.inputs(@intCast(seed));
        for (0..2) |turn| {
            const candidate = (turn + i) % 2 != 0;
            const seconds = try probe.checked(candidate, expected[seed]);
            if (candidate) result.m8[i] = seconds else result.m16[i] = seconds;
        }
        std.log.info("qwen27: M8 cohort pair={d} M16={d:.9} M8={d:.9}", .{ i, result.m16[i], result.m8[i] });
    }
    result.decision = policy.decide(true, &result.m16, &result.m8, &result.m16_null, &result.m8_null);
    return result;
}

const Measurement = struct { seconds: f64, hash: [32]u8 };
const Probe = struct {
    queue: mtl.Queue,
    programs: []const Program,
    x: mtl.Buffer,
    xs: mtl.Buffer,
    y: mtl.Buffer,
    counted: mtl.Buffer,
    waiting: mtl.Buffer,
    live_rows: u32,

    fn inputs(p: Probe, seed: u32) void {
        @setFloatMode(.strict);
        var state: u32 = 0x9E3779B9 ^ (seed *% 0x85EBCA6B);
        const values = p.x.slice(u16, rows * k);
        for (values) |*v| {
            state = state *% 1664525 +% 1013904223;
            const f = (@as(f32, @floatFromInt(state >> 8)) / 16777216.0) * 2.0 - 1.0;
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        const sums = p.xs.slice(f32, (k / 64) * sums_stride);
        @memset(sums, 0);
        for (0..k / 64) |g| for (0..rows) |r| {
            var sum: f32 = 0;
            for (0..64) |j| sum += @as(f32, @bitCast(@as(u32, values[r * k + g * 64 + j]) << 16));
            sums[g * sums_stride + r] = sum;
        };
    }

    fn checked(p: Probe, candidate: bool, expected: [32]u8) !f64 {
        const value = try p.measure(candidate);
        if (!std.mem.eql(u8, &expected, &value.hash)) return error.M8CalibrationIdentity;
        return value.seconds;
    }

    fn measure(p: Probe, candidate: bool) !Measurement {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        @memset(p.y.contents()[0 .. matrices * block_bytes], 0xFF);
        for (0..matrices) |i| {
            const base = i * block_bytes;
            @memset(p.y.contents()[base..][0..guard_bytes], 0xD3);
            @memset(p.y.contents()[base + guard_bytes + payload_bytes ..][0..guard_bytes], 0xD3);
        }
        @memset(p.counted.contents()[0 .. matrices * 8], 0);
        const ready = p.waiting.slice(u32, matrices * 2);
        @memset(ready, 0);
        for (0..matrices) |i| ready[2 * i] = 1;
        const cb = p.queue.commandBuffer();
        const e = cb.compute(.serial);
        var closed = false;
        defer if (!closed) e.end();
        for (p.programs, 0..) |program, i| {
            const projection = if (candidate) program.candidate else program.baseline;
            try projection.encodeCountedAfter(e, .{ .buf = p.x }, program.words, program.metadata, .{ .ref = .{ .buf = p.xs }, .stride = sums_stride }, .{ .buf = p.y, .off = i * block_bytes + guard_bytes }, p.live_rows, .{ .buf = p.counted, .off = i * 8 }, .{ .buf = p.waiting, .off = i * 8 }, 1);
        }
        e.end();
        closed = true;
        cb.commit();
        cb.wait();
        if (cb.status() != .completed) return error.M8CalibrationGpu;
        const count = p.counted.slice(u32, matrices * 2);
        const wait = p.waiting.slice(u32, matrices * 2);
        for (p.programs, 0..) |program, i| {
            const projection = if (candidate) program.candidate else program.baseline;
            if (count[2 * i] != projection.tile_count or count[2 * i + 1] != 0 or wait[2 * i] != 0 or wait[2 * i + 1] != 0)
                return error.M8CalibrationCounter;
        }
        for (0..matrices) |i| {
            const base = i * block_bytes;
            for (p.y.contents()[base..][0..guard_bytes]) |byte| if (byte != 0xD3) return error.M8CalibrationGuard;
            for (p.y.contents()[base + guard_bytes + payload_bytes ..][0..guard_bytes]) |byte| if (byte != 0xD3) return error.M8CalibrationGuard;
            const at = (base + guard_bytes) / 2;
            for (p.y.slice(u16, matrices * block_bytes / 2)[at..][0 .. p.live_rows * n]) |word|
                if ((word & 0x7F80) == 0x7F80) return error.M8CalibrationNonfinite;
            for (p.y.contents()[base + guard_bytes + p.live_rows * n * 2 .. base + guard_bytes + payload_bytes]) |byte|
                if (byte != 0xFF) return error.M8CalibrationGuard;
        }
        const start = cb.gpuStart();
        const end = cb.gpuEnd();
        if (!std.math.isFinite(start) or !std.math.isFinite(end) or start <= 0 or end <= start) return error.M8CalibrationTime;
        const seconds = end - start;
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(p.y.contents()[0 .. matrices * block_bytes], &hash, .{});
        return .{ .seconds = seconds, .hash = hash };
    }
};
