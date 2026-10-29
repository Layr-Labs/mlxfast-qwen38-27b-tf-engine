// Modified by Layr Labs, Inc. in 2026; see THIRD_PARTY_NOTICES.md.
//! A committed token's log probability under the target's raw distribution, with its best tokens (OpenAI logprobs).
const std = @import("std");

/// The most alternatives a row carries (OpenAI's `top_logprobs` limit).
pub const max_top = 20;

/// The words a backend writes for one row: the log normalizer, the pick's logit, the best ids, their logits, the rank.
pub const words = 3 + 2 * max_top;

pub const Row = struct {
    token: u32, // the committed token
    logprob: f32, // its log probability; NaN when it was forced and is not among the best `count`
    normalizer: f32 = 0, // logsumexp subtracted from logprobs; zero for synthetic rows that already carry raw values
    raw_logits: [max_top]f32 = undefined,
    has_raw: bool = false,
    count: u8 = 0,
    ids: [max_top]u32 = undefined, // logit descending, ties by the lower id
    logprobs: [max_top]f32 = undefined,
    rank: u32 = 0, // vLLM's: the vocabulary's tokens at or above it, itself included, when the backend was asked for it

    /// One row from a backend's words: `pick` is the token whose logit the backend gathered, `count` best ids follow.
    pub fn fromWords(w: []const u32, pick: u32, count: u8) Row {
        std.debug.assert(count <= max_top and w.len >= words);
        const lse: f32 = @bitCast(w[0]);
        var r: Row = .{ .token = pick, .logprob = @as(f32, @bitCast(w[1])) - lse, .normalizer = lse, .has_raw = true, .count = count, .rank = w[2 + 2 * max_top] };
        for (0..count) |i| {
            r.ids[i] = w[2 + i];
            r.raw_logits[i] = @bitCast(w[2 + max_top + i]);
            r.logprobs[i] = r.raw_logits[i] - lse;
        }
        return r;
    }

    /// The backend's best raw logits, in its value-descending and lower-id tie order.
    pub fn rawTop(r: Row, out: []Raw) error{Unsupported}!usize {
        if (!r.has_raw) return error.Unsupported;
        const n = @min(@min(out.len, r.count), max_top);
        for (out[0..n], r.ids[0..n], r.raw_logits[0..n]) |*entry, id, value| {
            entry.* = .{ .token = id, .logit = value };
        }
        var i: usize = 1;
        while (i < n) : (i += 1) {
            const item = out[i];
            var j = i;
            while (j > 0 and rawBefore(item, out[j - 1])) : (j -= 1) out[j] = out[j - 1];
            out[j] = item;
        }
        return n;
    }

    /// One bf16 vocabulary row, preserving its raw top logits beside normalized log probabilities.
    pub fn fromBf16(logits: []const u16, pick: u32, count: u8) !Row {
        if (logits.len == 0 or pick >= logits.len or count > max_top) return error.InvalidLogitRow;
        var most: f64 = -std.math.inf(f64);
        for (logits) |word| {
            const value = bf16(word);
            if (!std.math.isFinite(value)) return error.NonfiniteLogits;
            most = @max(most, value);
        }
        var sum: f64 = 0;
        for (logits) |word| sum += @exp(@as(f64, bf16(word)) - most);
        const lse: f32 = @floatCast(most + @log(sum));
        var r: Row = .{
            .token = pick,
            .logprob = bf16(logits[pick]) - lse,
            .normalizer = lse,
            .has_raw = true,
            .count = count,
        };
        if (count == 0) return r;
        var filled: usize = 0;
        for (logits, 0..) |word, id| {
            const value = bf16(word);
            if (filled == count and value <= r.raw_logits[count - 1]) continue;
            var at = @min(filled, count - 1);
            while (at > 0 and value > r.raw_logits[at - 1]) : (at -= 1) {
                r.ids[at] = r.ids[at - 1];
                r.raw_logits[at] = r.raw_logits[at - 1];
            }
            r.ids[at] = @intCast(id);
            r.raw_logits[at] = value;
            filled = @min(filled + 1, count);
        }
        for (r.logprobs[0..count], r.raw_logits[0..count]) |*value, raw| value.* = raw - lse;
        return r;
    }

    /// A forced token's row at this position: its log probability among the best, else NaN.
    pub fn forToken(r: Row, token: u32) Row {
        if (token == r.token) return r;
        var out = r;
        out.token = token;
        out.logprob = std.math.nan(f32);
        for (r.ids[0..r.count], r.logprobs[0..r.count]) |id, lp| {
            if (id == token) out.logprob = lp;
        }
        return out;
    }
};

pub const Raw = struct { token: u32, logit: f32 };

fn rawBefore(a: Raw, b: Raw) bool {
    return a.logit > b.logit or (a.logit == b.logit and a.token < b.token);
}

fn bf16(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

test "a row reads the normalizer, the pick and the best ids from the backend's words" {
    var w: [words]u32 = @splat(0);
    w[0] = @bitCast(@as(f32, 2.5));
    w[1] = @bitCast(@as(f32, 2.0));
    w[2] = 7;
    w[3] = 9;
    w[2 + max_top] = @bitCast(@as(f32, 2.0));
    w[3 + max_top] = @bitCast(@as(f32, 1.0));
    const r = Row.fromWords(&w, 7, 2);
    try std.testing.expectEqual(@as(f32, -0.5), r.logprob);
    try std.testing.expectEqualSlices(u32, &.{ 7, 9 }, r.ids[0..2]);
    try std.testing.expectEqual(@as(f32, -1.5), r.logprobs[1]);
    var raw: [2]Raw = undefined;
    try std.testing.expectEqual(@as(usize, 2), try r.rawTop(&raw));
    try std.testing.expectEqualDeep(Raw{ .token = 7, .logit = 2.0 }, raw[0]);
    try std.testing.expectEqualDeep(Raw{ .token = 9, .logit = 1.0 }, raw[1]);
    try std.testing.expectEqual(@as(f32, -1.5), r.forToken(9).logprob);
    try std.testing.expect(std.math.isNan(r.forToken(3).logprob));
}
