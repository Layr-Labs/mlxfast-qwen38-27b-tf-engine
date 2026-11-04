//! Host-side accounting for Qwen27's monotonic B2 norm-publisher counter.
const std = @import("std");

pub const Epoch = struct {
    value: u32 = 0,
    enabled: bool = true,

    /// Reserve enough headroom for any publisher in the whole upcoming forward interval.
    /// Once disabled, publication cannot be re-enabled for this model lifetime.
    pub fn preflight(e: *Epoch, rows: u32, layers: u32) bool {
        if (!e.enabled or rows == 0 or layers == 0) return false;
        const advance = std.math.mul(u32, rows, layers) catch {
            e.enabled = false;
            return false;
        };
        _ = std.math.add(u32, e.value, advance) catch {
            e.enabled = false;
            return false;
        };
        return true;
    }

    /// Peek the cumulative threshold for one publisher. Overflow permanently disables the fast path.
    pub fn next(e: *Epoch, rows: u32) ?u32 {
        if (!e.enabled or rows == 0) return null;
        return std.math.add(u32, e.value, rows) catch blk: {
            e.enabled = false;
            break :blk null;
        };
    }

    /// Called only after the corresponding norm dispatch was encoded successfully.
    pub fn commit(e: *Epoch, threshold: u32) void {
        e.value = threshold;
    }
};

test "publication epoch thresholds advance cumulatively and commit after encoding" {
    var e: Epoch = .{};
    try std.testing.expect(e.preflight(8, 256));
    const first = e.next(8).?;
    try std.testing.expectEqual(@as(u32, 0), e.value);
    e.commit(first);
    try std.testing.expectEqual(@as(u32, 8), e.value);
    try std.testing.expectEqual(@as(?u32, 16), e.next(8));
    e.commit(e.next(8).?);
    try std.testing.expectEqual(@as(u32, 16), e.value);
}

test "whole interval preflight disables before a possible u32 wrap" {
    var e = Epoch{ .value = std.math.maxInt(u32) - 7 };
    try std.testing.expect(!e.preflight(1, 8));
    try std.testing.expect(!e.enabled);
    try std.testing.expectEqual(@as(?u32, null), e.next(1));
    // Sabotage the stored value after rejection: the disabled state must stay sticky.
    e.value = 0;
    try std.testing.expect(!e.preflight(1, 1));
    try std.testing.expectEqual(@as(?u32, null), e.next(1));
}

test "single publisher overflow is fail-closed and sticky" {
    var e = Epoch{ .value = std.math.maxInt(u32) - 7 };
    try std.testing.expectEqual(@as(?u32, null), e.next(8));
    try std.testing.expect(!e.enabled);
    e.value = 0;
    try std.testing.expectEqual(@as(?u32, null), e.next(1));
    try std.testing.expect(!e.preflight(1, 1));
}

test "publication epoch rejects zero and multiplication overflow" {
    var e: Epoch = .{};
    try std.testing.expect(!e.preflight(0, 1));
    try std.testing.expect(!e.preflight(1, 0));
    try std.testing.expectEqual(@as(?u32, null), e.next(0));
    try std.testing.expect(!e.preflight(8, std.math.maxInt(u32)));
    try std.testing.expect(!e.enabled);
}
