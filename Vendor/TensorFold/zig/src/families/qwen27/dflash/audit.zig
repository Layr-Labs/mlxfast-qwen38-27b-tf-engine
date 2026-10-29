const std = @import("std");

pub const VerificationShape = enum { rectangular };

pub const Round = struct {
    committed: u32,
    drafted: u32,
    accepted: u32,
    verification_shape: VerificationShape = .rectangular,
};

pub const Journal = struct {
    rounds: std.ArrayListUnmanaged(Round) = .empty,
    drafted_total: u64 = 0,
    accepted_total: u64 = 0,
    committed_total: u64 = 0,

    pub fn deinit(journal: *Journal, alloc: std.mem.Allocator) void {
        journal.rounds.deinit(alloc);
        journal.* = undefined;
    }

    pub fn clearRetainingCapacity(journal: *Journal) void {
        journal.rounds.clearRetainingCapacity();
        journal.drafted_total = 0;
        journal.accepted_total = 0;
        journal.committed_total = 0;
    }

    pub fn reserve(journal: *Journal, alloc: std.mem.Allocator, count: usize) !void {
        try journal.rounds.ensureUnusedCapacity(alloc, count);
    }

    pub fn record(journal: *Journal, alloc: std.mem.Allocator, round: Round) !void {
        try check(round);
        try journal.reserve(alloc, 1);
        journal.recordAssumeCapacity(round);
    }

    pub fn recordAssumeCapacity(journal: *Journal, round: Round) void {
        std.debug.assert(valid(round));
        journal.rounds.appendAssumeCapacity(round);
        journal.drafted_total += round.drafted;
        journal.accepted_total += round.accepted;
        journal.committed_total += round.committed;
    }
};

fn valid(round: Round) bool {
    return round.committed > 0 and round.accepted <= round.drafted and round.accepted <= round.committed;
}

fn check(round: Round) error{BadAuditRound}!void {
    if (!valid(round)) return error.BadAuditRound;
}

pub fn appendTo(journal: *const Journal, audit: anytype, alloc: std.mem.Allocator) !void {
    try audit.acceptance_lengths.ensureUnusedCapacity(alloc, journal.rounds.items.len);
    for (journal.rounds.items) |round| {
        std.debug.assert(round.verification_shape == .rectangular);
        try audit.appendRound(alloc, round.committed, round.drafted, round.accepted);
    }
    audit.verification_mode = .rectangular;
}
