const std = @import("std");
const bench_session = @import("bench_session");
const dflash_audit = @import("dflash_audit");

const FakeRound = struct { tokens: []const u32, drafted: u32, accepted: u32 };
const Stats = struct { rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0 };

fn drive(rounds: []const FakeRound, journal: *dflash_audit.Journal) !struct { stats: Stats, emitted: u64 } {
    var stats: Stats = .{};
    var emitted: u64 = 0;
    for (rounds) |round| {
        try journal.record(std.testing.allocator, .{
            .committed = @intCast(round.tokens.len),
            .drafted = round.drafted,
            .accepted = round.accepted,
        });
        stats.rounds += 1;
        stats.drafted += round.drafted;
        stats.accepted += round.accepted;
        emitted += round.tokens.len;
    }
    return .{ .stats = stats, .emitted = emitted };
}

test "N fake rounds produce N rectangular audit entries and matching totals" {
    var journal: dflash_audit.Journal = .{};
    defer journal.deinit(std.testing.allocator);
    const result = try drive(&.{
        .{ .tokens = &.{ 10, 11, 12 }, .drafted = 4, .accepted = 2 },
        .{ .tokens = &.{13}, .drafted = 3, .accepted = 0 },
        .{ .tokens = &.{ 14, 15 }, .drafted = 2, .accepted = 1 },
    }, &journal);

    try std.testing.expectEqual(result.stats.rounds, @as(u64, @intCast(journal.rounds.items.len)));
    try std.testing.expectEqual(result.emitted, journal.committed_total);
    try std.testing.expectEqual(result.stats.drafted, journal.drafted_total);
    try std.testing.expectEqual(result.stats.accepted, journal.accepted_total);
    for (journal.rounds.items) |round| try std.testing.expectEqual(dflash_audit.VerificationShape.rectangular, round.verification_shape);

    var audit = bench_session.DflashAudit.init();
    defer audit.deinit(std.testing.allocator);
    try dflash_audit.appendTo(&journal, &audit, std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1, 2 }, audit.acceptance_lengths.items);
    try std.testing.expectEqual(journal.committed_total, audit.committed_total);
    try std.testing.expectEqual(journal.drafted_total, audit.drafted_total);
    try std.testing.expectEqual(journal.accepted_total, audit.accepted_total);
    try audit.check(@intCast(result.emitted));
}

test "a round that accepts no drafts commits one token" {
    var journal: dflash_audit.Journal = .{};
    defer journal.deinit(std.testing.allocator);
    _ = try drive(&.{.{ .tokens = &.{42}, .drafted = 5, .accepted = 0 }}, &journal);
    try std.testing.expectEqual(@as(u32, 1), journal.rounds.items[0].committed);
    try std.testing.expectEqual(@as(u32, 0), journal.rounds.items[0].accepted);
}

test "depth one drafts at most one token per round" {
    var journal: dflash_audit.Journal = .{};
    defer journal.deinit(std.testing.allocator);
    const result = try drive(&.{
        .{ .tokens = &.{ 1, 2 }, .drafted = 1, .accepted = 1 },
        .{ .tokens = &.{3}, .drafted = 1, .accepted = 0 },
        .{ .tokens = &.{ 4, 5 }, .drafted = 1, .accepted = 1 },
    }, &journal);
    for (journal.rounds.items) |round| try std.testing.expect(round.drafted <= 1);
    try std.testing.expect(result.stats.drafted <= result.stats.rounds);
}
