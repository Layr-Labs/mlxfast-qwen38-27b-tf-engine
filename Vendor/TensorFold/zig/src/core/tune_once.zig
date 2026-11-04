//! Load-time device-timed variant adoption (tune-once): measure once at load, outside timed
//! windows, on synthetic seeded data; adopt a fixed winner per boot only when the candidate is
//! bit-identical and device-timed faster, else the incumbent stays. The trial holds no prompt or
//! decode content, so the adopted pipeline and flag are setup the Engine-state rules allow at
//! engine open or first use, kept across phases; nothing is written anywhere, so no state crosses
//! jobs. Every helper that touches the GPU returns an error on any miss; callers keep the
//! incumbent on any error.
const std = @import("std");
const mtl = @import("metal");

/// The 32-bit narrowing this framework offers pilots: `size_t` becomes `uint` in kernel text.
pub const narrow_from = "size_t";
pub const narrow_to = "uint";

/// Non-overlapping occurrences of `needle` in `haystack`.
pub fn countOccurrences(haystack: []const u8, needle: []const u8) usize {
    @setEvalBranchQuota(1000000);
    if (needle.len == 0) return 0;
    var n: usize = 0;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) {
        if (std.mem.eql(u8, haystack[i..][0..needle.len], needle)) {
            n += 1;
            i += needle.len;
        } else i += 1;
    }
    return n;
}

/// Lines of `text` carrying at least one `needle`.
pub fn countLinesWith(text: []const u8, needle: []const u8) usize {
    @setEvalBranchQuota(1000000);
    var n: usize = 0;
    var hit = false;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '\n') {
            if (hit) n += 1;
            hit = false;
            i += 1;
        } else if (i + needle.len <= text.len and std.mem.eql(u8, text[i..][0..needle.len], needle)) {
            hit = true;
            i += needle.len;
        } else i += 1;
    }
    if (hit) n += 1;
    return n;
}

/// `text` with every `size_t` narrowed to `uint` (a 32-bit-index twin of the launched text).
pub fn narrow(comptime text: []const u8) [text.len - countOccurrences(text, narrow_from) * (narrow_from.len - narrow_to.len)]u8 {
    @setEvalBranchQuota(1000000);
    comptime {
        var out: [text.len - countOccurrences(text, narrow_from) * (narrow_from.len - narrow_to.len)]u8 = undefined;
        var i: usize = 0;
        var o: usize = 0;
        while (i < text.len) {
            if (i + narrow_from.len <= text.len and std.mem.eql(u8, text[i..][0..narrow_from.len], narrow_from)) {
                @memcpy(out[o..][0..narrow_to.len], narrow_to);
                o += narrow_to.len;
                i += narrow_from.len;
            } else {
                out[o] = text[i];
                o += 1;
                i += 1;
            }
        }
        return out;
    }
}

/// `MLXFAST_TUNE=0` keeps every incumbent; tuning is on by default.
pub fn tuneEnabled() bool {
    const v = std.c.getenv("MLXFAST_TUNE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// `MLXFAST_TUNE_FORCE=1` adopts a bit-identical twin without timing it.
pub fn tuneForced() bool {
    const v = std.c.getenv("MLXFAST_TUNE_FORCE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Edge bf16 words the synthetic trial data always covers (signed zeros, infinities, NaNs, the
/// tiniest subnormal, the largest finite values, one).
pub const edge_bf16 = [_]u16{ 0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80, 0x3C00, 0xBC00 };

/// Deterministic synthetic bf16 words from `seed`: small magnitudes (exponents 2^-4 to 2^1) with
/// every 64th word cycling the edge table. The same seed always fills the same words.
pub fn fillBf16(words: []u16, seed: u64) void {
    var state: u64 = seed *% 0x9E3779B97F4A7C15 +% 0x243F6A8885A308D3;
    if (state == 0) state = 0x853C49E6748FEA9B;
    for (words, 0..) |*w, i| {
        if (i % 64 == 63) {
            w.* = edge_bf16[(i / 64) % edge_bf16.len];
            continue;
        }
        var x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        const r: u64 = x *% 0x2545F4914F6CDD1D;
        const exp: u16 = @intCast(123 + (r % 6));
        const sign: u16 = @as(u16, @truncate(r >> 48)) & 0x8000;
        const mantissa: u16 = @as(u16, @truncate(r >> 16)) & 0x007F;
        w.* = sign | (exp << 7) | mantissa;
    }
}

/// The median of `values` (sorted in place); at least one value.
pub fn medianOf(values: []f64) f64 {
    std.debug.assert(values.len > 0);
    std.mem.sort(f64, values, {}, std.sort.asc(f64));
    const n = values.len;
    return if (n % 2 == 1) values[n / 2] else (values[n / 2 - 1] + values[n / 2]) / 2;
}

/// Which candidate runs first in `cycle`: rotate by cycle / 2, reverse on odd cycles, so over
/// each pair of cycles the two candidates take mirrored positions and run in both orders.
pub fn pairOrder(cycle: usize, order: *[2]usize) void {
    var o = [2]usize{ 0, 1 };
    if ((cycle / 2) % 2 == 1) o = .{ 1, 0 };
    if (cycle % 2 == 1) std.mem.swap(usize, &o[0], &o[1]);
    order.* = o;
}

/// The adoption bar: the candidate's median at least 3% below the incumbent's.
pub const median_bar: f64 = 0.97;

/// Timed cycles per duel; every duel times at least 16.
pub const min_cycles: usize = 16;
pub const max_cycles: usize = 64;

/// Adopt the candidate only when both hold: its median clears the bar, and it beat the incumbent
/// in at least 75% of the cycles, each cycle's two samples compared.
pub fn decide(inc_med: f64, cand_med: f64, wins: usize, cycles: usize) bool {
    if (cycles == 0) return false;
    if (!(cand_med <= inc_med * median_bar)) return false;
    return wins * 4 >= cycles * 3;
}

/// One candidate's sample: what the trial encodes into a serial compute encoder.
pub const EncodeFn = *const fn (e: mtl.ComputeEncoder, ctx: *anyopaque) void;

/// Encode, commit, wait, and return the command buffer's GPU span in seconds (the device's own
/// clock). Any Metal miss is an error, so the caller keeps the incumbent.
pub fn launchSeconds(queue: mtl.Queue, enc: EncodeFn, ctx: *anyopaque) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    enc(e, ctx);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.status() != .completed) return error.TuneLaunchFailed;
    const s = cb.gpuSeconds();
    if (!(s > 0.0) or !std.math.isFinite(s)) return error.TuneLaunchFailed;
    return s;
}

pub const Outcome = struct {
    adopted: bool,
    inc_med: f64,
    cand_med: f64,
    wins: usize,
    cycles: usize,
};

/// Time `warmup` untimed cycles, then `cycles` timed cycles of incumbent versus candidate round
/// robin in mirrored order, and judge the medians with the shared rule.
pub fn timeDuel(queue: mtl.Queue, inc: EncodeFn, inc_ctx: *anyopaque, cand: EncodeFn, cand_ctx: *anyopaque, cycles: usize, warmup: usize) !Outcome {
    if (cycles == 0 or cycles > max_cycles) return error.TuneBadCycles;
    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        _ = try launchSeconds(queue, inc, inc_ctx);
        _ = try launchSeconds(queue, cand, cand_ctx);
    }
    var inc_s: [max_cycles]f64 = undefined;
    var cand_s: [max_cycles]f64 = undefined;
    var wins: usize = 0;
    for (0..cycles) |c| {
        var o: [2]usize = undefined;
        pairOrder(c, &o);
        var samp: [2]f64 = undefined;
        for (o) |which| {
            samp[which] = try launchSeconds(queue, if (which == 0) inc else cand, if (which == 0) inc_ctx else cand_ctx);
        }
        inc_s[c] = samp[0];
        cand_s[c] = samp[1];
        if (samp[1] < samp[0]) wins += 1;
    }
    const inc_med = medianOf(inc_s[0..cycles]);
    const cand_med = medianOf(cand_s[0..cycles]);
    return .{ .adopted = decide(inc_med, cand_med, wins, cycles), .inc_med = inc_med, .cand_med = cand_med, .wins = wins, .cycles = cycles };
}

test "narrow swaps every size_t for uint" {
    const sample = "X[size_t(a)*b+c] and size_t base=size_t(x);";
    try std.testing.expectEqual(@as(usize, 3), countOccurrences(sample, narrow_from));
    try std.testing.expectEqual(@as(usize, 2), countLinesWith("a size_t b\nnope\nsize_t x size_t y\n", narrow_from));
    try std.testing.expectEqual(@as(usize, 0), countOccurrences("nothing here", narrow_from));
    const out = comptime narrow(sample);
    try std.testing.expectEqualStrings("X[uint(a)*b+c] and uint base=uint(x);", &out);
    try std.testing.expectEqual(@as(usize, 0), countOccurrences(&out, narrow_from));
}

test "median of odd and even samples" {
    var a = [_]f64{ 3, 1, 2 };
    try std.testing.expectEqual(@as(f64, 2), medianOf(&a));
    var b = [_]f64{ 4, 1, 3, 2 };
    try std.testing.expectEqual(@as(f64, 2.5), medianOf(&b));
}

test "pair order mirrors over each cycle pair" {
    var o: [2]usize = undefined;
    pairOrder(0, &o);
    try std.testing.expectEqual([2]usize{ 0, 1 }, o);
    pairOrder(1, &o);
    try std.testing.expectEqual([2]usize{ 1, 0 }, o);
    pairOrder(2, &o);
    try std.testing.expectEqual([2]usize{ 1, 0 }, o);
    pairOrder(3, &o);
    try std.testing.expectEqual([2]usize{ 0, 1 }, o);
}

test "adoption needs the bar and the wins" {
    try std.testing.expect(decide(1.0, 0.96, 12, 16));
    try std.testing.expect(decide(1.0, 0.97, 12, 16));
    try std.testing.expect(!decide(1.0, 0.98, 16, 16));
    try std.testing.expect(!decide(1.0, 0.90, 11, 16));
    try std.testing.expect(!decide(1.0, 1.01, 0, 16));
    try std.testing.expect(!decide(1.0, 0.50, 0, 0));
}

test "synthetic bf16 is seeded and covers the edges" {
    var a: [1024]u16 = undefined;
    var b: [1024]u16 = undefined;
    fillBf16(&a, 12345);
    fillBf16(&b, 12345);
    try std.testing.expectEqualSlices(u16, &a, &b);
    fillBf16(&b, 999);
    try std.testing.expect(!std.mem.eql(u16, &a, &b));
    for (edge_bf16) |e| {
        var found = false;
        for (a) |w| if (w == e) {
            found = true;
            break;
        };
        try std.testing.expect(found);
    }
}
