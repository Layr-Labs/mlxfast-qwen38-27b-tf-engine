//! Pure synthetic activations and borrowed immutable model matrices, before any frame exists.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
const projection = @import("projection.zig");
const policy = @import("prompt_gu_walk_policy.zig");
const matrices = 64;
const rows = 128;
const n = 34816;
const k = 5120;
const half = n / 2;
const guard = 32;
const y_bytes = rows * half * 2;
const hs_bytes = rows * (half / 64) * 4;
const y_block = y_bytes + 2 * guard;
const hs_block = hs_bytes + 2 * guard;

pub fn eligible(p: projection.Linear) bool {
    return p.n == n and p.k == k and p.bits == 4 and p.group == 64 and p.tile == 32 and p.batch == 8 and p.runs.len == 0 and p.raw == null and (projection.promptSplit(p.n, p.k) orelse p.slices) == 1;
}

pub fn run(device: mtl.Device, queue: mtl.Queue, programs: [4]lane.Projection, layers: []const projection.Linear) !u8 {
    if (layers.len != matrices) return error.PromptWalkShape;
    for (layers) |p| if (!eligible(p)) return error.PromptWalkShape;
    for (programs) |p| {
        if (p.n != n or p.k != k or !p.swiglu or !p.reg or p.rows8 or p.swiglu_threads != 64 or p.group_count != 80) return error.PromptWalkShape;
        if (p.pipe.simdWidth() != 32 or p.pipe.maxThreads() < 64 or mtl.objc.msg(usize, p.pipe.id, "staticThreadgroupMemoryLength", .{}) != 8192) return error.PromptWalkPipeline;
    }
    for (layers, 0..) |p, i| {
        if (p.words.offset > p.words.buffer.length() or programs[0].weight_bytes > p.words.buffer.length() - p.words.offset or p.pairs.offset > p.pairs.buffer.length() or programs[0].metadata_bytes > p.pairs.buffer.length() - p.pairs.offset) return error.PromptWalkStorage;
        for (layers[0..i]) |previous| {
            if (p.words.buffer.id == previous.words.buffer.id) {
                const distance = if (p.words.offset >= previous.words.offset) p.words.offset - previous.words.offset else previous.words.offset - p.words.offset;
                if (distance < programs[0].weight_bytes) return error.PromptWalkDuplicateMatrix;
            }
        }
    }
    const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const x = try device.buffer(rows * k * 2, options);
    defer x.deinit();
    const xs = try device.buffer(rows * (k / 64) * 4, options);
    defer xs.deinit();
    const y = try device.buffer(matrices * y_block, options);
    defer y.deinit();
    const hs = try device.buffer(matrices * hs_block, options);
    defer hs.deinit();
    const probe = Probe{ .queue = queue, .programs = programs, .layers = layers, .x = x, .xs = xs, .y = y, .hs = hs };
    var expected: [2][32]u8 = undefined;
    for (0..2) |seed| {
        probe.inputs(@intCast(seed));
        const first = try probe.measure(0, seed != 0);
        expected[seed] = first.hash;
        try probe.checkSums();
        for (1..4) |arm| _ = try probe.checked(arm, expected[seed], seed != 0);
    }
    return policy.select(probe, expected);
}

const Measurement = struct { seconds: f64, hash: [32]u8 };
const Probe = struct {
    queue: mtl.Queue,
    programs: [4]lane.Projection,
    layers: []const projection.Linear,
    x: mtl.Buffer,
    xs: mtl.Buffer,
    y: mtl.Buffer,
    hs: mtl.Buffer,

    fn inputs(p: Probe, seed: u32) void {
        @setFloatMode(.strict);
        var state: u32 = 0x9E3779B9 ^ (seed *% 0x85EBCA6B);
        const values = p.x.slice(u16, rows * k);
        for (values) |*v| {
            state = state *% 1664525 +% 1013904223;
            // Bounded dyadic BF16 values, no prompt/tokenizer/model-state source.
            const f = @as(f32, @floatFromInt(@as(i32, @intCast((state >> 24) & 31)) - 16)) / 16.0;
            v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        const sums = p.xs.slice(f32, rows * (k / 64));
        for (0..k / 64) |g| for (0..rows) |r| {
            var sum: f32 = 0;
            for (0..64) |j| sum += @as(f32, @bitCast(@as(u32, values[r * k + g * 64 + j]) << 16));
            sums[g * rows + r] = sum;
        };
    }

    pub fn nulls(p: Probe, arm: usize, expected: [2][32]u8) ![2 * policy.null_pairs]policy.Pair {
        var result: [2 * policy.null_pairs]policy.Pair = undefined;
        for (0..policy.null_pairs) |i| {
            const seed = i % 2;
            p.inputs(@intCast(seed));
            for (0..2) |turn| {
                const choice = if ((turn + i) % 2 == 0) @as(usize, 0) else arm;
                const first = try p.checked(choice, expected[seed], seed != 0);
                const second = try p.checked(choice, expected[seed], seed != 0);
                result[2 * i + turn] = .{ .baseline = first, .candidate = second };
            }
        }
        return result;
    }

    pub fn pairs(p: Probe, arm: usize, expected: [2][32]u8) ![policy.pair_count]policy.Pair {
        var result: [policy.pair_count]policy.Pair = undefined;
        for (0..policy.pair_count) |i| {
            const seed = i % 2;
            p.inputs(@intCast(seed));
            var a: f64 = 0;
            var b: f64 = 0;
            // ABBA/BAAB paired blocks: both arms occupy both inner/outer positions.
            for (0..4) |turn| {
                const candidate = ((turn == 1 or turn == 2) != ((i / 2) % 2 != 0));
                const value = try p.checked(if (candidate) arm else 0, expected[seed], seed != 0);
                if (candidate) b += value else a += value;
            }
            result[i] = .{ .baseline = a / 2, .candidate = b / 2 };
        }
        return result;
    }

    fn checked(p: Probe, arm: usize, expected: [32]u8, reverse: bool) !f64 {
        const value = try p.measure(arm, reverse);
        try policy.requireIdentity(expected, value.hash);
        return value.seconds;
    }

    fn measure(p: Probe, arm: usize, reverse: bool) !Measurement {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        @memset(p.y.contents()[0 .. matrices * y_block], 0xFF);
        @memset(p.hs.contents()[0 .. matrices * hs_block], 0xFF);
        for (0..matrices) |i| {
            for ([_]struct { b: mtl.Buffer, block: usize, bytes: usize }{ .{ .b = p.y, .block = y_block, .bytes = y_bytes }, .{ .b = p.hs, .block = hs_block, .bytes = hs_bytes } }) |v| {
                @memset(v.b.contents()[i * v.block ..][0..guard], 0xD3);
                @memset(v.b.contents()[i * v.block + guard + v.bytes ..][0..guard], 0xD3);
            }
        }
        const cb = p.queue.commandBuffer();
        const e = cb.compute(.serial);
        var closed = false;
        // Drain a partially encoded command buffer on every pre-commit error before scratch can be freed.
        defer if (!closed) {
            e.end();
            cb.commit();
            cb.wait();
        };
        for (0..matrices) |step| {
            const i = if (reverse) matrices - 1 - step else step;
            const layer = p.layers[i];
            try p.programs[arm].encodeSwiglu(e, .{ .buf = p.x }, .{ .buf = layer.words.buffer, .off = layer.words.offset }, .{ .buf = layer.pairs.buffer, .off = layer.pairs.offset }, .{ .ref = .{ .buf = p.xs }, .stride = rows }, .{ .buf = p.y, .off = i * y_block + guard }, .{ .buf = p.hs, .off = i * hs_block + guard }, rows);
        }
        e.end();
        closed = true;
        cb.commit();
        cb.wait();
        if (cb.status() != .completed) return error.PromptWalkGpu;
        for (0..matrices) |i| {
            for ([_]struct { b: mtl.Buffer, block: usize, bytes: usize }{ .{ .b = p.y, .block = y_block, .bytes = y_bytes }, .{ .b = p.hs, .block = hs_block, .bytes = hs_bytes } }) |v| {
                for (v.b.contents()[i * v.block ..][0..guard]) |byte| if (byte != 0xD3) return error.PromptWalkGuard;
                for (v.b.contents()[i * v.block + guard + v.bytes ..][0..guard]) |byte| if (byte != 0xD3) return error.PromptWalkGuard;
            }
            const yw: [*]const u16 = @ptrCast(@alignCast(p.y.contents() + i * y_block + guard));
            for (yw[0 .. rows * half]) |word| if (word & 0x7F80 == 0x7F80) return error.PromptWalkNonfinite;
            const sw: [*]const f32 = @ptrCast(@alignCast(p.hs.contents() + i * hs_block + guard));
            for (sw[0 .. rows * (half / 64)]) |value| if (!std.math.isFinite(value)) return error.PromptWalkNonfinite;
        }
        const seconds = cb.gpuSeconds();
        if (!std.math.isFinite(seconds) or seconds <= 0 or cb.gpuStart() <= 0) return error.PromptWalkTime;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(p.y.contents()[0 .. matrices * y_block]);
        h.update(p.hs.contents()[0 .. matrices * hs_block]);
        var hash: [32]u8 = undefined;
        h.final(&hash);
        return .{ .seconds = seconds, .hash = hash };
    }

    fn checkSums(p: Probe) !void {
        @setFloatMode(.strict);
        for (0..matrices) |i| {
            const yw: [*]const u16 = @ptrCast(@alignCast(p.y.contents() + i * y_block + guard));
            const sw: [*]const f32 = @ptrCast(@alignCast(p.hs.contents() + i * hs_block + guard));
            for (0..half / 64) |g| for (0..rows) |r| {
                var sum: f32 = 0;
                for (0..64) |j| sum += @as(f32, @bitCast(@as(u32, yw[r * half + g * 64 + j]) << 16));
                if (@as(u32, @bitCast(sum)) != @as(u32, @bitCast(sw[g * rows + r]))) return error.PromptWalkSums;
            };
        }
    }
};
