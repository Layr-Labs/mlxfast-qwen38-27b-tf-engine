//! Borrowed real matrices and disposable synthetic buffers, before any model frame exists.
//! Command-buffer/hash/guard scaffold adapted from team d8f76fd4 / R8 08949fe.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
const projection = @import("projection.zig");
const native = @import("native_down.zig");
const policy = @import("native_down_startup_policy.zig");
const matrices = 64;
const rows = policy.rows;
const n = native.n;
const k = native.k;
const half = n;
const guard = 32;
const y_bytes = rows * half * 2;
const y_block = y_bytes + 2 * guard;
pub const scratch_bytes: usize = matrices * y_block + rows * k * 2 + rows * (k / 64) * 4 + rows * n * 2;
pub const max_cohorts = policy.max_cohorts;
const budget_seconds: f64 = 120;

pub fn refsDistinct(p: anytype) bool {
    const words = p.native_down orelse return false;
    const pairs = p.native_down_pairs orelse return false;
    return words.buffer.id != p.words.buffer.id and words.buffer.id != p.pairs.buffer.id and
        pairs.buffer.id != p.words.buffer.id and pairs.buffer.id != p.pairs.buffer.id and
        words.buffer.id != pairs.buffer.id;
}

pub fn eligible(p: projection.Linear) bool {
    return p.n == n and p.k == k and p.bits == 4 and p.group == 64 and p.tile == 32 and p.batch == 8 and p.runs.len == 0 and p.raw == null and refsDistinct(p) and (projection.promptSplit(p.n, p.k) orelse p.slices) == 2;
}

test "native weights and original-scale pairs cannot alias original folded buffers" {
    const R = struct { buffer: struct { id: u32 } };
    const P = struct { words: R, pairs: R, native_down: ?R, native_down_pairs: ?R };
    const good = P{ .words = .{ .buffer = .{ .id = 1 } }, .pairs = .{ .buffer = .{ .id = 2 } }, .native_down = .{ .buffer = .{ .id = 3 } }, .native_down_pairs = .{ .buffer = .{ .id = 4 } } };
    try std.testing.expect(refsDistinct(good));
    var bad = good;
    bad.native_down_pairs = null;
    try std.testing.expect(!refsDistinct(bad));
    bad = good;
    bad.native_down = null;
    try std.testing.expect(!refsDistinct(bad));
    for ([_]u32{ 1, 2, 3 }) |id| {
        bad = good;
        bad.native_down_pairs = .{ .buffer = .{ .id = id } };
        try std.testing.expect(!refsDistinct(bad));
    }
    for ([_]u32{ 1, 2, 4 }) |id| {
        bad = good;
        bad.native_down = .{ .buffer = .{ .id = id } };
        try std.testing.expect(!refsDistinct(bad));
    }
}
fn fits(ref: projection.Ref, bytes: usize) bool {
    return ref.offset <= ref.buffer.length() and bytes <= ref.buffer.length() - ref.offset;
}
pub fn run(kernels: anytype, queue: mtl.Queue, baseline: lane.Projection, layers: []const projection.Linear, passes: *u16, mode: policy.Mode) !policy.Decision {
    if (mode == .off) return .{ .status = .off };
    if (layers.len != matrices or !kernels.device.tensorUnits()) return error.NativeDownShape;
    if (!policy.budgetFits(kernels.device.allocated(), kernels.device.maxWorkingSet(), scratch_bytes)) return error.NativeDownBudget;
    if (baseline.n != n or baseline.k != k or !baseline.residual or !baseline.reg or baseline.rows8 or baseline.sk != 2 or baseline.group_count != 272 or baseline.pipe.simdWidth() != 32 or baseline.pipe.maxThreads() < 64) return error.NativeDownShape;
    for (layers, 0..) |p, i| {
        if (!eligible(p) or !fits(p.words, baseline.weight_bytes) or !fits(p.pairs, baseline.metadata_bytes) or !fits(p.native_down.?, native.bytes) or !fits(p.native_down_pairs.?, native.metadata_bytes)) return error.NativeDownShape;
        for (layers[0..i]) |other| {
            if (p.words.buffer.id == other.words.buffer.id) {
                const distance = if (p.words.offset >= other.words.offset) p.words.offset - other.words.offset else other.words.offset - p.words.offset;
                if (distance < baseline.weight_bytes) return error.NativeDownShape;
            }
        }
    }
    const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const x = try kernels.device.buffer(rows * k * 2, options);
    defer x.deinit();
    const xs = try kernels.device.buffer(rows * (k / 64) * 4, options);
    defer xs.deinit();
    const y = try kernels.device.buffer(matrices * y_block, options);
    defer y.deinit();
    const hidden = try kernels.device.buffer(rows * n * 2, options);
    defer hidden.deinit();
    const P = Probe(@TypeOf(kernels));
    var probe = P{ .kernels = kernels, .queue = queue, .baseline = baseline, .layers = layers, .x = x, .xs = xs, .y = y, .hidden = hidden, .passes = passes, .started = mtl.clock.seconds(), .limit = policy.cohortLimit(mode) };
    return policy.admit(&probe, mode);
}

fn Probe(comptime K: type) type {
    return struct {
        kernels: K,
        queue: mtl.Queue,
        baseline: lane.Projection,
        layers: []const projection.Linear,
        x: mtl.Buffer,
        xs: mtl.Buffer,
        y: mtl.Buffer,
        hidden: mtl.Buffer,
        passes: *u16,
        started: f64,
        limit: usize,
        input_hash: policy.Hash = undefined,
        active_rows: u32 = rows,
        const Self = @This();
        fn budget(p: *Self) !void {
            const now = mtl.clock.seconds();
            if (!std.math.isFinite(now) or !std.math.isFinite(p.started) or now < p.started or now - p.started > budget_seconds or p.passes.* >= p.limit) return error.NativeDownBudget;
        }
        fn hashInputs(p: *Self) policy.Hash {
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            h.update(p.x.contents()[0 .. rows * k * 2]);
            h.update(p.xs.contents()[0 .. rows * (k / 64) * 4]);
            h.update(p.hidden.contents()[0 .. rows * n * 2]);
            var out: policy.Hash = undefined;
            h.final(&out);
            return out;
        }
        pub fn inputs(p: *Self, seed: u32) !void {
            try p.budget();
            try policy.inputs(p.x.slice(u16, rows * k), p.xs.slice(f32, rows * (k / 64)), seed);
            std.mem.copyForwards(u16, p.hidden.slice(u16, rows * n), p.x.slice(u16, rows * k)[0 .. rows * n]);
            try policy.checkInputSums(p.x.slice(u16, rows * k), p.xs.slice(f32, rows * (k / 64)));
            p.input_hash = p.hashInputs();
        }
        pub fn original(p: *Self, seed: u32, active: u32) !policy.Hash {
            p.active_rows = active;
            const measured = try p.measure(false, seed != 0);
            try p.checkSums();
            return measured.hash;
        }
        pub fn candidate(p: *Self, seed: u32, active: u32, expected: policy.Hash) !void {
            p.active_rows = active;
            _ = try p.checked(true, expected, seed != 0);
            try p.checkSums();
        }
        pub fn beginWarm(p: *Self) !void {
            p.active_rows = rows;
            try p.inputs(0);
        }
        pub fn warm(p: *Self, use_native: bool, expected: policy.Hash) !void {
            _ = try p.checked(use_native, expected, false);
        }
        pub fn nulls(p: *Self, expected: [2]policy.Hash, holdout: bool) ![policy.null_count]policy.Pair {
            var result: [policy.null_count]policy.Pair = undefined;
            for (&result, 0..) |*out, i| {
                const seed = i % 2;
                try p.inputs(@intCast(seed));
                var a: f64 = 0;
                var b: f64 = 0;
                for (0..4) |turn| {
                    const inert = policy.candidateTurn(i, turn, holdout);
                    // Both labels use the EXACT same original pipeline, bindings and matrix order.
                    const value = try p.checked(false, expected[seed], seed != 0);
                    if (inert) b += value else a += value;
                }
                out.* = .{ .baseline = a / 2, .candidate = b / 2 };
            }
            return result;
        }
        pub fn pair(p: *Self, i: usize, expected: [2]policy.Hash, holdout: bool) !policy.Pair {
            const seed = i % 2;
            try p.inputs(@intCast(seed));
            var a: f64 = 0;
            var b: f64 = 0;
            for (0..4) |turn| {
                const use_native = policy.candidateTurn(i, turn, holdout);
                const value = try p.checked(use_native, expected[seed], seed != 0);
                if (use_native) b += value else a += value;
            }
            return .{ .baseline = a / 2, .candidate = b / 2 };
        }
        fn checked(p: *Self, use_native: bool, expected: policy.Hash, reverse: bool) !f64 {
            const value = try p.measure(use_native, reverse);
            try policy.requireIdentity(expected, value.hash);
            return value.seconds;
        }
        fn measure(p: *Self, use_native: bool, reverse: bool) !struct { seconds: f64, hash: policy.Hash } {
            try p.budget();
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            @memset(p.y.contents()[0 .. matrices * y_block], 0xFF);
            for (0..matrices) |i| for ([_]struct { b: mtl.Buffer, block: usize, bytes: usize }{.{ .b = p.y, .block = y_block, .bytes = y_bytes }}) |v| {
                @memset(v.b.contents()[i * v.block ..][0..guard], 0xD3);
                @memset(v.b.contents()[i * v.block + guard + v.bytes ..][0..guard], 0xD3);
            };
            const cb = p.queue.commandBuffer();
            const e = cb.compute(.serial);
            var closed = false;
            defer if (!closed) {
                e.end();
                cb.commit();
                cb.wait();
                p.passes.* += 1;
            };
            for (0..matrices) |step| {
                const i = if (reverse) matrices - 1 - step else step;
                const layer = p.layers[i];
                if (use_native) {
                    _ = try p.kernels.encodeNativeStartup(e, layer, .{ .buffer = p.x }, .{ .ref = .{ .buffer = p.xs }, .stride = rows }, .{ .buffer = p.hidden }, .{ .buffer = p.y, .offset = i * y_block + guard }, p.active_rows);
                } else try p.baseline.encodeResid(e, .{ .buf = p.x }, .{ .buf = layer.words.buffer, .off = layer.words.offset }, .{ .buf = layer.pairs.buffer, .off = layer.pairs.offset }, .{ .ref = .{ .buf = p.xs }, .stride = rows }, .{ .buf = p.hidden }, .{ .buf = p.y, .off = i * y_block + guard }, p.active_rows);
            }
            e.end();
            closed = true;
            cb.commit();
            cb.wait();
            p.passes.* += 1;
            if (cb.status() != .completed) return error.NativeDownGpu;
            for (0..matrices) |i| {
                for ([_]struct { b: mtl.Buffer, block: usize, bytes: usize }{.{ .b = p.y, .block = y_block, .bytes = y_bytes }}) |v| {
                    for (v.b.contents()[i * v.block ..][0..guard]) |byte| if (byte != 0xD3) return error.NativeDownGuard;
                    for (v.b.contents()[i * v.block + guard + v.bytes ..][0..guard]) |byte| if (byte != 0xD3) return error.NativeDownGuard;
                }
                const yw: [*]const u16 = @ptrCast(@alignCast(p.y.contents() + i * y_block + guard));
                try policy.checkRows(yw[0 .. rows * half], half, rows, p.active_rows);
            }
            try policy.requireIdentity(p.input_hash, p.hashInputs());
            const start = cb.gpuStart();
            const end = cb.gpuEnd();
            const seconds = end - start;
            if (!std.math.isFinite(start) or !std.math.isFinite(end) or !std.math.isFinite(seconds) or start <= 0 or seconds <= 0) return error.NativeDownTime;
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            h.update(p.y.contents()[0 .. matrices * y_block]);
            var hash: policy.Hash = undefined;
            h.final(&hash);
            if (mtl.clock.seconds() - p.started > budget_seconds) return error.NativeDownBudget;
            return .{ .seconds = seconds, .hash = hash };
        }
        fn checkSums(p: *Self) !void {
            try policy.checkInputSums(p.x.slice(u16, rows * k), p.xs.slice(f32, rows * (k / 64)));
        }
    };
}
