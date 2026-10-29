const std = @import("std");
const bench = @import("bench_session");
const adapter_mod = @import("bench_session_engine");
const api = @import("engine_api");
const lanes = @import("lanes");
const process_memory = @import("process_memory");

const Fixture = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    backend_table: lanes.backend.Backend.VTable = undefined,
    fail_first_draft: bool = false,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: api.LaneHost,
    adapter: adapter_mod.Adapter,

    fn init() !*Fixture {
        return initModel(.{ .exact_width = 1, .gpu_tokens = true, .hidden_rows = true });
    }

    fn initDrafted() !*Fixture {
        return initDraftedFailure(false, false);
    }

    fn initDraftedFailure(fail_draw: bool, fail_draft: bool) !*Fixture {
        const f = try initModelWithBackend(
            .{ .exact_width = 2, .gpu_tokens = true, .hidden_rows = true, .mtp = true, .speculate = true, .drafts = 1, .batch_rows = 2, .draft_streams = true },
            fail_draft,
        );
        if (fail_draw) f.target.fail_read = 0;
        return f;
    }

    fn initModel(model: lanes.Model) !*Fixture {
        return initModelWithBackend(model, false);
    }

    fn initModelWithBackend(model: lanes.Model, fail_draft: bool) !*Fixture {
        const a = std.testing.allocator;
        const f = try a.create(Fixture);
        errdefer a.destroy(f);
        f.cfg = try lanes.Config.init(a, model, model.exact_width, model.drafts);
        errdefer f.cfg.deinit(a);
        f.target = .{ .gpa = a };
        errdefer f.target.deinit();
        f.fail_first_draft = fail_draft;
        f.core = lanes.Engine.init(a, &f.cfg, if (fail_draft) f.failingBackend() else f.target.backend(), f.clock.clock());
        errdefer f.core.deinit();
        f.host = api.LaneHost.init(a, std.testing.io, &f.core, .{ .lanes = 1, .context_window = 32, .logprobs = true });
        f.adapter = adapter_mod.Adapter.init(a, f.host.engine(), null);
        return f;
    }

    fn failingBackend(f: *Fixture) lanes.backend.Backend {
        f.backend_table = f.target.backend().vtable.*;
        f.backend_table.draft = failDraft;
        return .{ .ptr = &f.target, .vtable = &f.backend_table };
    }

    fn failDraft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) anyerror!void {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const f: *Fixture = @alignCast(@fieldParentPtr("target", target));
        if (f.fail_first_draft) {
            f.fail_first_draft = false;
            return error.InjectedDraftFailure;
        }
        return target.backend().draft(requests);
    }

    fn deinit(f: *Fixture) void {
        f.core.deinit();
        f.target.deinit();
        f.cfg.deinit(std.testing.allocator);
        std.testing.allocator.destroy(f);
    }
};

test "retained seed and forced steps keep one lane state" {
    const f = try Fixture.init();
    defer f.deinit();
    var cache_probe: CacheMock = .{ .spare_bytes = 64 };
    var store = api.prompt_cache.Store.init(std.testing.allocator, cache_probe.snapshots(), .{}, 64);
    defer store.deinit();
    f.host.cache = &store;
    const session = f.adapter.session();
    const prompt = [_]u32{ 2, 3 };
    const seeded = try session.seed(&prompt, false);
    try std.testing.expectEqual(@as(u32, 41), seeded.token);
    const first = try session.force(11, false);
    try std.testing.expectEqual(@as(u32, 24), first.token);
    const second = try session.force(7, false);
    try std.testing.expectEqual(@as(u32, 76), second.token);
    const diag = try session.closePhase();
    try std.testing.expectEqual(@as(u64, 3), diag.work_tokens);
    try std.testing.expectEqual(@as(u64, 0), diag.cache_bytes);
    try std.testing.expectEqual(@as(usize, 0), f.target.lanes.count());
    try std.testing.expectEqual(@as(u32, 1), cache_probe.trim_calls);
}

test "retained seed rejects a prompt that leaves no context row" {
    const f = try Fixture.init();
    defer f.deinit();
    const full: [32]u32 = @splat(1);
    try std.testing.expectError(error.ContextFull, f.adapter.session().seed(&full, false));
    try std.testing.expectEqual(@as(usize, 0), f.target.lanes.count());
}

test "retained top rejects a producer with normalized rows" {
    const f = try Fixture.init();
    defer f.deinit();
    try std.testing.expectError(error.Unsupported, f.adapter.session().seed(&.{ 2, 3 }, true));
    try std.testing.expectEqual(@as(usize, 0), f.target.lanes.count());
}

const CacheMock = struct {
    spare_bytes: u64 = 0,
    trim_calls: u32 = 0,

    fn self(ptr: *anyopaque) *CacheMock {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 0;
    }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!*anyopaque {
        return error.Unsupported;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.Unsupported;
    }
    fn drop(_: *anyopaque, _: *anyopaque) void {}
    fn spare(ptr: *anyopaque) u64 {
        return self(ptr).spare_bytes;
    }
    fn trim(ptr: *anyopaque, room: u64) void {
        const mock = self(ptr);
        mock.spare_bytes = @min(mock.spare_bytes, room);
        mock.trim_calls += 1;
    }

    fn snapshots(mock: *CacheMock) api.prompt_cache.Snapshots {
        return .{ .ptr = mock, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .spare = spare, .trim = trim } };
    }
};

test "reset drops retained state" {
    const f = try Fixture.init();
    defer f.deinit();
    const session = f.adapter.session();
    var cache_probe: CacheMock = .{};
    var store = api.prompt_cache.Store.init(std.testing.allocator, cache_probe.snapshots(), .{}, 0);
    defer store.deinit();
    try store.shared_keys.append(std.testing.allocator, 17);
    f.host.cache = &store;
    _ = try session.seed(&.{ 5, 8 }, false);
    session.reset();
    try std.testing.expectError(error.SessionNotSeeded, session.force(13, false));
    try std.testing.expectEqual(@as(usize, 0), f.target.lanes.count());
    try std.testing.expectEqual(@as(usize, 0), store.shared_keys.items.len);
}

test "prefill and generate use plain greedy submissions" {
    const f = try Fixture.init();
    defer f.deinit();
    try f.host.start();
    defer f.host.stop();
    const session = f.adapter.session();
    const prompt = [_]u32{ 4, 6, 8 };
    try std.testing.expectEqual(lanes.fake.next(&prompt, null, prompt.len), try session.prefill(&prompt));
    var got: [4]u32 = undefined;
    try session.generate(&prompt, got.len, &got);
    var history = std.ArrayList(u32).empty;
    defer history.deinit(std.testing.allocator);
    try history.appendSlice(std.testing.allocator, &prompt);
    for (got) |token| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), token);
        try history.append(std.testing.allocator, token);
    }
    const diag = try session.closePhase();
    try std.testing.expectEqual(@as(u64, 0), diag.work_tokens);
    try std.testing.expectEqual(@as(u64, 0), diag.cache_bytes);
}

test "raw top eight keeps values and lower token tie order" {
    const lp = lanes.logprob;
    var words: [lp.words]u32 = @splat(0);
    words[0] = @bitCast(@as(f32, 9.5));
    for (0..8) |i| {
        words[2 + i] = @intCast(20 + i);
        words[2 + lp.max_top + i] = @bitCast(@as(f32, @floatFromInt(8 - i / 2)));
    }
    words[2] = 21;
    words[3] = 20;
    const row = lp.Row.fromWords(&words, 20, 8);
    var raw: [8]lp.Raw = undefined;
    try std.testing.expectEqual(@as(usize, 8), try row.rawTop(&raw));
    try std.testing.expectEqual(@as(f32, 8), raw[0].logit);
    try std.testing.expectEqual(@as(f32, 8), raw[1].logit);
    try std.testing.expect(raw[0].token < raw[1].token);
    try std.testing.expectEqual(@as(f32, -1.5), row.logprobs[0]);
    var bf16: [10]u16 = undefined;
    for (&bf16, 0..) |*word, i| {
        const value: f32 = if (i == 2 or i == 3) 7 else @floatFromInt(i);
        word.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    const backend_row = try lp.Row.fromBf16(&bf16, 9, 8);
    try std.testing.expectEqual(@as(usize, 8), try backend_row.rawTop(&raw));
    try std.testing.expectEqual(@as(u32, 9), raw[0].token);
    try std.testing.expectEqual(@as(u32, 8), raw[1].token);
    try std.testing.expectEqual(@as(u32, 2), raw[2].token);
    try std.testing.expectEqual(@as(u32, 3), raw[3].token);
    var normalized: lp.Row = .{ .token = 1, .logprob = -0.5, .count = 1 };
    normalized.ids[0] = 1;
    normalized.logprobs[0] = -0.5;
    try std.testing.expectError(error.Unsupported, normalized.rawTop(&raw));
}

test "Linux VmHWM parser returns bytes" {
    const sample = "Name:\tbench-worker\nVmHWM:\t  123456 kB\nVmRSS:\t   65432 kB\nThreads:\t4\n";
    const got = process_memory.parseLinuxStatus(sample).?;
    try std.testing.expectEqual(@as(u64, 65_432 * 1024), got.physical_footprint_bytes);
    try std.testing.expectEqual(@as(u64, 123_456 * 1024), got.lifetime_peak_physical_footprint_bytes);
}

const DraftMock = struct {
    next: u32 = 100,
    resets: u32 = 0,

    fn source(d: *DraftMock) adapter_mod.DflashSource {
        return .{ .ctx = d, .seed = seed, .round = round, .reset = reset };
    }

    fn seed(ptr: *anyopaque, tokens: []const u32, _: u8) anyerror!u32 {
        const d: *DraftMock = @ptrCast(@alignCast(ptr));
        if (tokens.len == 0) return error.EmptyPrompt;
        d.next = tokens[tokens.len - 1] + 1;
        return d.next;
    }

    fn round(
        ptr: *anyopaque,
        remaining: u32,
        out: []u32,
        audit: *bench.DflashAudit,
        alloc: std.mem.Allocator,
    ) anyerror!u32 {
        const d: *DraftMock = @ptrCast(@alignCast(ptr));
        const committed = @min(remaining, 3);
        for (out[0..committed]) |*token| {
            d.next += 1;
            token.* = d.next;
        }
        try audit.appendRound(alloc, committed, 1, @min(committed - 1, 1));
        return committed;
    }

    fn reset(ptr: *anyopaque) anyerror!void {
        const d: *DraftMock = @ptrCast(@alignCast(ptr));
        d.resets += 1;
    }
};

test "free-run audit lengths sum to count" {
    const f = try Fixture.init();
    defer f.deinit();
    var draft: DraftMock = .{};
    f.adapter.dflash = draft.source();
    const session = f.adapter.session();
    try std.testing.expectEqual(@as(u32, 10), try session.seedFree(&.{ 3, 9 }, .{ .dflash = .{ .depth = 1 } }));
    var audit = bench.DflashAudit.init();
    defer audit.deinit(std.testing.allocator);
    var out: [8]u32 = undefined;
    try session.runFree(out.len, &out, &audit);
    try audit.check(out.len);
    var sum: u64 = 0;
    for (audit.acceptance_lengths.items) |length| sum += length;
    try std.testing.expectEqual(@as(u64, out.len), sum);
    try std.testing.expectEqual(@as(u64, out.len), audit.committed_total);
    try std.testing.expectEqual(@as(u64, 3), audit.drafted_total);
    const diag = try session.closePhase();
    try std.testing.expectEqual(@as(u64, 4), diag.work_tokens);
}

test "Qwen resident open passes CUDA backend refusal with MTP controls unset" {
    const drafted = adapter_mod.residentOpenRequest("target", "qwen3_5", "drafter");
    try std.testing.expectEqualStrings("qwen3_5", drafted.model_type);
    try std.testing.expectEqualStrings("drafter", drafted.drafter.?);
    try std.testing.expect(drafted.mtp_drafts == null);
    try std.testing.expect(drafted.mtp_confidence == null);
    const serial = adapter_mod.residentOpenRequest("target", "qwen3_5", null);
    try std.testing.expect(serial.drafter == null);
    try std.testing.expect(serial.mtp_drafts == null);
    try std.testing.expect(serial.mtp_confidence == null);
}

test "public DFlash stream stays idle between seed and run" {
    const delayed = try Fixture.init();
    defer delayed.deinit();
    try delayed.host.start();
    defer delayed.host.stop();
    var delayed_binding = adapter_mod.EngineDflashBinding.init(std.testing.allocator, delayed.host.engine());
    defer delayed_binding.deinit();
    delayed.adapter.dflash = delayed_binding.source();
    const delayed_session = delayed.adapter.session();
    _ = try delayed_session.seedFree(&.{ 3, 9 }, .{ .dflash = .{ .depth = 1 } });
    try std.Io.sleep(std.testing.io, .fromMilliseconds(200), .awake);
    try std.testing.expectEqual(@as(usize, 1), delayed.core.bench_stream.?.emitted().len);
    var delayed_audit = bench.DflashAudit.init();
    defer delayed_audit.deinit(std.testing.allocator);
    var delayed_out: [4]u32 = undefined;
    try delayed_session.runFree(delayed_out.len, &delayed_out, &delayed_audit);
    _ = try delayed_session.closePhase();

    const immediate = try Fixture.init();
    defer immediate.deinit();
    try immediate.host.start();
    defer immediate.host.stop();
    var immediate_binding = adapter_mod.EngineDflashBinding.init(std.testing.allocator, immediate.host.engine());
    defer immediate_binding.deinit();
    immediate.adapter.dflash = immediate_binding.source();
    const immediate_session = immediate.adapter.session();
    _ = try immediate_session.seedFree(&.{ 3, 9 }, .{ .dflash = .{ .depth = 1 } });
    var immediate_audit = bench.DflashAudit.init();
    defer immediate_audit.deinit(std.testing.allocator);
    var immediate_out: [4]u32 = undefined;
    try immediate_session.runFree(immediate_out.len, &immediate_out, &immediate_audit);
    try std.testing.expectEqualSlices(u32, &immediate_out, &delayed_out);
    _ = try immediate_session.closePhase();
}

fn expectDflashSeedFailureRecovers(f: *Fixture, expected: anyerror) !void {
    defer f.deinit();
    try f.host.start();
    defer f.host.stop();
    var binding = adapter_mod.EngineDflashBinding.init(std.testing.allocator, f.host.engine());
    defer binding.deinit();
    f.adapter.dflash = binding.source();
    const session = f.adapter.session();
    const prompt = [_]u32{ 3, 9 };

    try std.testing.expectError(expected, session.seedFree(&prompt, .{ .dflash = .{ .depth = 1 } }));
    try std.testing.expectEqual(@as(usize, 0), f.target.lanes.count());
    _ = try session.prefill(&prompt);
    var correctness: [2]u32 = undefined;
    try session.generate(&prompt, correctness.len, &correctness);
}

test "DFlash first draw failure releases its lane" {
    try expectDflashSeedFailureRecovers(try Fixture.initDraftedFailure(true, false), error.GpuFailed);
}

test "DFlash first draft failure releases its lane" {
    try expectDflashSeedFailureRecovers(try Fixture.initDraftedFailure(false, true), error.InjectedDraftFailure);
}

test "CUDA binding serves correctness and DFlash through one fake lane engine" {
    const f = try Fixture.initDrafted();
    defer f.deinit();
    try f.host.start();
    defer f.host.stop();
    var binding = adapter_mod.EngineDflashBinding.init(std.testing.allocator, f.host.engine());
    defer binding.deinit();
    f.adapter.dflash = adapter_mod.platformDflash(.linux, f.host.engine(), true, &binding);
    try std.testing.expect(f.adapter.dflash != null);
    f.target.raw_rows = true;
    const session = f.adapter.session();
    const prompt = [_]u32{ 3, 9 };

    _ = try session.prefill(&prompt);
    try std.testing.expectEqual(@as(usize, 0), f.target.drafts_calls);
    try std.testing.expectEqual(@as(usize, 0), f.target.feature_calls);

    session.reset();
    const first = try session.seed(&prompt, true);
    const second = try session.force(first.token, true);
    try std.testing.expect(first.has_top);
    try std.testing.expect(second.has_top);
    try std.testing.expectEqual(@as(usize, 0), f.target.drafts_calls);
    try std.testing.expectEqual(@as(usize, 0), f.target.feature_calls);

    session.reset();
    _ = try session.seedFree(&prompt, .{ .dflash = .{ .depth = 1 } });
    var drafted_audit = bench.DflashAudit.init();
    defer drafted_audit.deinit(std.testing.allocator);
    var drafted: [5]u32 = undefined;
    try session.runFree(drafted.len, &drafted, &drafted_audit);
    try drafted_audit.check(drafted.len);
    try std.testing.expect(f.target.drafts_calls > 0);

    session.reset();
    _ = try session.seedFree(&prompt, .serial);
    var serial_audit = bench.DflashAudit.init();
    defer serial_audit.deinit(std.testing.allocator);
    var serial: [5]u32 = undefined;
    try session.runFree(serial.len, &serial, &serial_audit);
    try serial_audit.check(serial.len);
    try std.testing.expectEqualSlices(u32, &serial, &drafted);
}

const ThreeRoundMock = struct {
    next: u32 = 20,

    fn source(mock: *ThreeRoundMock) adapter_mod.DflashSource {
        return .{ .ctx = mock, .seed = seed, .round = round, .reset = reset };
    }

    fn seed(ptr: *anyopaque, tokens: []const u32, _: u8) anyerror!u32 {
        const mock: *ThreeRoundMock = @ptrCast(@alignCast(ptr));
        if (tokens.len == 0) return error.EmptyPrompt;
        mock.next = tokens[tokens.len - 1] + 1;
        return mock.next;
    }

    fn round(
        ptr: *anyopaque,
        remaining: u32,
        out: []u32,
        audit: *bench.DflashAudit,
        alloc: std.mem.Allocator,
    ) anyerror!u32 {
        const mock: *ThreeRoundMock = @ptrCast(@alignCast(ptr));
        if (remaining != 3 or out.len < remaining) return error.BadDraftRound;
        for (out[0..remaining]) |*token| {
            mock.next += 1;
            token.* = mock.next;
            try audit.appendRound(alloc, 1, 1, 1);
        }
        return remaining;
    }

    fn reset(_: *anyopaque) anyerror!void {}
};

test "free-run work counts each source round" {
    const f = try Fixture.init();
    defer f.deinit();
    var mock: ThreeRoundMock = .{};
    f.adapter.dflash = mock.source();
    const session = f.adapter.session();
    _ = try session.seedFree(&.{ 3, 9 }, .{ .dflash = .{ .depth = 1 } });
    var audit = bench.DflashAudit.init();
    defer audit.deinit(std.testing.allocator);
    var out: [3]u32 = undefined;
    try session.runFree(out.len, &out, &audit);
    const diag = try session.closePhase();
    try std.testing.expectEqual(@as(u64, 4), diag.work_tokens);
}

test "local iterate phase work matches benchd for serial and DFlash" {
    const f = try Fixture.init();
    defer f.deinit();
    try f.host.start();
    defer f.host.stop();
    var mock: ThreeRoundMock = .{};
    f.adapter.dflash = mock.source();
    const session = f.adapter.session();

    session.reset();
    _ = try session.seed(&.{ 3, 9 }, false);
    _ = try session.force(10, false);
    _ = try session.force(11, false);
    try std.testing.expectEqual(@as(u64, 3), (try session.closePhase()).work_tokens);

    session.reset();
    _ = try session.prefill(&.{ 3, 9 });
    session.reset();
    _ = try session.prefill(&.{ 3, 9 });
    try std.testing.expectEqual(@as(u64, 0), (try session.closePhase()).work_tokens);

    session.reset();
    _ = try session.seed(&.{ 3, 9 }, false);
    _ = try session.force(10, false);
    _ = try session.force(11, false);
    try std.testing.expectEqual(@as(u64, 3), (try session.closePhase()).work_tokens);

    const specs = [_]bench.FreeSpec{ .serial, .{ .dflash = .{ .depth = 1 } } };
    for (specs) |spec| {
        session.reset();
        _ = try session.seedFree(&.{ 3, 9 }, spec);
        var audit = bench.DflashAudit.init();
        defer audit.deinit(std.testing.allocator);
        var out: [3]u32 = undefined;
        try session.runFree(out.len, &out, &audit);
        const expected = audit.acceptance_lengths.items.len + 1;
        try std.testing.expectEqual(@as(u64, @intCast(expected)), (try session.closePhase()).work_tokens);
    }

    session.reset();
    var out: [3]u32 = undefined;
    try session.generate(&.{ 3, 9 }, out.len, &out);
    try std.testing.expectEqual(@as(u64, 0), (try session.closePhase()).work_tokens);
}

test "Linux selects the public engine DFlash source" {
    try std.testing.expectEqual(.cuda, adapter_mod.platformDflashBackend(.linux, true, true));
    try std.testing.expectEqual(.metal, adapter_mod.platformDflashBackend(.macos, true, true));
    try std.testing.expectEqual(.none, adapter_mod.platformDflashBackend(.linux, false, true));
    try std.testing.expectEqual(.none, adapter_mod.platformDflashBackend(.linux, true, false));
}

test "a step without top logits is initialized" {
    const step = adapter_mod.testStep(.{ .token = 7, .top = undefined, .has_top = false });
    try std.testing.expect(!step.has_top);
    for (step.top) |entry| {
        try std.testing.expectEqual(@as(u32, 0), entry.token);
        try std.testing.expectEqual(@as(f32, 0), entry.logit);
    }
}
