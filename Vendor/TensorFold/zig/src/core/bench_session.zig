const std = @import("std");

pub const TopLogit = struct { token: u32, logit: f32 };

pub const Step = struct {
    token: u32,
    top: [8]TopLogit,
    has_top: bool,
};

pub const PhaseDiag = struct {
    work_tokens: u64,
    peak_ram_bytes: ?u64,
    cache_bytes: u64,
};

pub const FreeSpec = union(enum) {
    serial,
    dflash: struct { depth: u8 },
};

pub const DflashAudit = struct {
    acceptance_lengths: std.ArrayListUnmanaged(u32),
    drafted_total: u64,
    accepted_total: u64,
    committed_total: u64,
    verification_mode: enum { rectangular },

    pub fn init() DflashAudit {
        return .{
            .acceptance_lengths = .empty,
            .drafted_total = 0,
            .accepted_total = 0,
            .committed_total = 0,
            .verification_mode = .rectangular,
        };
    }

    pub fn deinit(audit: *DflashAudit, alloc: std.mem.Allocator) void {
        audit.acceptance_lengths.deinit(alloc);
        audit.* = undefined;
    }

    pub fn appendRound(
        audit: *DflashAudit,
        alloc: std.mem.Allocator,
        committed: u32,
        drafted: u32,
        accepted: u32,
    ) !void {
        try audit.acceptance_lengths.append(alloc, committed);
        audit.drafted_total += drafted;
        audit.accepted_total += accepted;
        audit.committed_total += committed;
    }

    pub fn check(audit: DflashAudit, count: u32) error{AuditMismatch}!void {
        var total: u64 = 0;
        for (audit.acceptance_lengths.items) |length| {
            total = std.math.add(u64, total, length) catch return error.AuditMismatch;
        }
        if (total != count or audit.committed_total != count) return error.AuditMismatch;
    }
};

pub const BenchSession = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        prefill: *const fn (ptr: *anyopaque, tokens: []const u32) anyerror!u32,
        generate: *const fn (ptr: *anyopaque, tokens: []const u32, steps: u32, out: []u32) anyerror!void,
        seed: *const fn (ptr: *anyopaque, tokens: []const u32, want_top: bool) anyerror!Step,
        force: *const fn (ptr: *anyopaque, token: u32, want_top: bool) anyerror!Step,
        seedFree: *const fn (ptr: *anyopaque, tokens: []const u32, spec: FreeSpec) anyerror!u32,
        runFree: *const fn (ptr: *anyopaque, count: u32, out: []u32, audit: *DflashAudit) anyerror!void,
        closePhase: *const fn (ptr: *anyopaque) anyerror!PhaseDiag,
        /// Samples the same counters as closePhase without draining or dropping retained state.
        inspectPhase: *const fn (ptr: *anyopaque) anyerror!PhaseDiag,
        reset: *const fn (ptr: *anyopaque) void,
    };

    pub fn prefill(session: BenchSession, tokens: []const u32) anyerror!u32 {
        return session.vtable.prefill(session.ptr, tokens);
    }

    pub fn generate(session: BenchSession, tokens: []const u32, steps: u32, out: []u32) anyerror!void {
        return session.vtable.generate(session.ptr, tokens, steps, out);
    }

    pub fn seed(session: BenchSession, tokens: []const u32, want_top: bool) anyerror!Step {
        return session.vtable.seed(session.ptr, tokens, want_top);
    }

    pub fn force(session: BenchSession, token: u32, want_top: bool) anyerror!Step {
        return session.vtable.force(session.ptr, token, want_top);
    }

    pub fn seedFree(session: BenchSession, tokens: []const u32, spec: FreeSpec) anyerror!u32 {
        return session.vtable.seedFree(session.ptr, tokens, spec);
    }

    pub fn runFree(session: BenchSession, count: u32, out: []u32, audit: *DflashAudit) anyerror!void {
        return session.vtable.runFree(session.ptr, count, out, audit);
    }

    pub fn closePhase(session: BenchSession) anyerror!PhaseDiag {
        return session.vtable.closePhase(session.ptr);
    }

    pub fn inspectPhase(session: BenchSession) anyerror!PhaseDiag {
        return session.vtable.inspectPhase(session.ptr);
    }

    pub fn reset(session: BenchSession) void {
        session.vtable.reset(session.ptr);
    }
};

pub const MockSession = struct {
    pub const vocab: u32 = 151936;

    allocator: std.mem.Allocator,
    work_tokens: u64 = 0,
    peak_ram_bytes: ?u64 = null,
    free_spec: FreeSpec = .serial,
    free_pending: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator) MockSession {
        return .{ .allocator = allocator };
    }

    pub fn session(mock: *MockSession) BenchSession {
        return .{ .ptr = mock, .vtable = &vtable };
    }

    fn next(token: u32) u32 {
        return @intCast((@as(u64, token) * 31 + 7) % vocab);
    }

    fn step(token: u32, want_top: bool) Step {
        const result = next(token);
        var top: [8]TopLogit = undefined;
        for (&top, 0..) |*entry, i| {
            entry.* = .{
                .token = @intCast((@as(u64, result) + i) % vocab),
                .logit = @floatFromInt(8 - i),
            };
        }
        return .{ .token = result, .top = top, .has_top = want_top };
    }

    fn from(ptr: *anyopaque) *MockSession {
        return @ptrCast(@alignCast(ptr));
    }

    fn promptLast(tokens: []const u32) !u32 {
        if (tokens.len == 0) return error.EmptyPrompt;
        return tokens[tokens.len - 1];
    }

    fn prefillFn(ptr: *anyopaque, tokens: []const u32) anyerror!u32 {
        const mock = from(ptr);
        const result = next(try promptLast(tokens));
        mock.work_tokens += 1;
        return result;
    }

    fn generateFn(ptr: *anyopaque, tokens: []const u32, steps: u32, out: []u32) anyerror!void {
        const mock = from(ptr);
        if (out.len < steps) return error.OutputTooSmall;
        var token = try promptLast(tokens);
        for (out[0..steps]) |*value| {
            token = next(token);
            value.* = token;
        }
        mock.work_tokens += steps;
    }

    fn seedFn(ptr: *anyopaque, tokens: []const u32, want_top: bool) anyerror!Step {
        const mock = from(ptr);
        const result = step(try promptLast(tokens), want_top);
        mock.work_tokens += 1;
        return result;
    }

    fn forceFn(ptr: *anyopaque, token: u32, want_top: bool) anyerror!Step {
        const mock = from(ptr);
        const result = step(token, want_top);
        mock.work_tokens += 1;
        return result;
    }

    fn seedFreeFn(ptr: *anyopaque, tokens: []const u32, spec: FreeSpec) anyerror!u32 {
        const mock = from(ptr);
        const result = next(try promptLast(tokens));
        mock.free_spec = spec;
        mock.free_pending = result;
        mock.work_tokens += 1;
        return result;
    }

    fn runFreeFn(ptr: *anyopaque, count: u32, out: []u32, audit: *DflashAudit) anyerror!void {
        const mock = from(ptr);
        if (out.len < count) return error.OutputTooSmall;
        var token = mock.free_pending orelse return error.SessionNotSeeded;
        var written: u32 = 0;
        while (written < count) {
            const left = count - written;
            const committed: u32 = switch (mock.free_spec) {
                .serial => 1,
                .dflash => |spec| @min(@as(u32, spec.depth) + 1, left),
            };
            const drafted: u32 = switch (mock.free_spec) {
                .serial => 0,
                .dflash => |spec| spec.depth,
            };
            const accepted = committed - 1;
            try audit.appendRound(mock.allocator, committed, drafted, accepted);
            for (out[written..][0..committed]) |*value| {
                token = next(token);
                value.* = token;
            }
            written += committed;
            mock.work_tokens += 1;
        }
        mock.free_pending = token;
    }

    fn closePhaseFn(ptr: *anyopaque) anyerror!PhaseDiag {
        const mock = from(ptr);
        const diag = PhaseDiag{
            .work_tokens = mock.work_tokens,
            .peak_ram_bytes = mock.peak_ram_bytes,
            .cache_bytes = 0,
        };
        mock.resetState();
        return diag;
    }

    fn inspectPhaseFn(ptr: *anyopaque) anyerror!PhaseDiag {
        const mock = from(ptr);
        return .{ .work_tokens = mock.work_tokens, .peak_ram_bytes = mock.peak_ram_bytes, .cache_bytes = 0 };
    }

    fn resetFn(ptr: *anyopaque) void {
        from(ptr).resetState();
    }

    fn resetState(mock: *MockSession) void {
        mock.work_tokens = 0;
        mock.free_spec = .serial;
        mock.free_pending = null;
    }

    pub const vtable: BenchSession.VTable = .{
        .prefill = prefillFn,
        .generate = generateFn,
        .seed = seedFn,
        .force = forceFn,
        .seedFree = seedFreeFn,
        .runFree = runFreeFn,
        .closePhase = closePhaseFn,
        .inspectPhase = inspectPhaseFn,
        .reset = resetFn,
    };
};

test "DFlash audit checks committed lengths and totals" {
    var audit = DflashAudit.init();
    defer audit.deinit(std.testing.allocator);
    try audit.appendRound(std.testing.allocator, 2, 3, 1);
    try audit.appendRound(std.testing.allocator, 3, 3, 2);
    try audit.check(5);
    try std.testing.expectError(error.AuditMismatch, audit.check(4));
    audit.committed_total += 1;
    try std.testing.expectError(error.AuditMismatch, audit.check(5));
}

test "mock session keeps deterministic steps and DFlash rounds" {
    var mock = MockSession.init(std.testing.allocator);
    const session = mock.session();
    const seeded = try session.seed(&.{2}, true);
    try std.testing.expectEqual(@as(u32, 69), seeded.token);
    try std.testing.expect(seeded.has_top);
    try std.testing.expectEqual(seeded.token, seeded.top[0].token);
    try std.testing.expectEqual(@as(f32, 8), seeded.top[0].logit);
    const forced = try session.force(seeded.token, false);
    try std.testing.expectEqual(MockSession.next(seeded.token), forced.token);
    try std.testing.expect(!forced.has_top);

    session.reset();
    const free_seed = try session.seedFree(&.{4}, .{ .dflash = .{ .depth = 2 } });
    try std.testing.expectEqual(@as(u32, 131), free_seed);
    var out: [7]u32 = undefined;
    var audit = DflashAudit.init();
    defer audit.deinit(std.testing.allocator);
    try session.runFree(out.len, &out, &audit);
    try std.testing.expectEqualSlices(u32, &.{ 4068, 126115, 111172, 103747, 25508, 31075, 51716 }, &out);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3, 1 }, audit.acceptance_lengths.items);
    try std.testing.expectEqual(@as(u64, 6), audit.drafted_total);
    try std.testing.expectEqual(@as(u64, 4), audit.accepted_total);
    try audit.check(out.len);
    const diag = try session.closePhase();
    try std.testing.expectEqual(@as(u64, 4), diag.work_tokens);
    try std.testing.expectEqual(@as(u64, 0), diag.cache_bytes);
}
