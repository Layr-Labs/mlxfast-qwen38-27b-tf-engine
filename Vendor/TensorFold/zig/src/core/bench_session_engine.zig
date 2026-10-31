//! Engine Protocol bench sessions over the public engine and its retained-step seam.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const bench = @import("bench_session");

const RUsage = extern struct {
    uuid: [16]u8,
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    padding: [20]u64,
    lifetime_max_phys_footprint: u64,
    trailing: [6]u64,
};

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;

fn processPeak() ?u64 {
    if (builtin.os.tag == .macos) {
        var info: RUsage = undefined;
        if (proc_pid_rusage(std.c.getpid(), 4, &info) != 0) return null;
        return info.lifetime_max_phys_footprint;
    }
    if (builtin.os.tag != .linux) return null;
    var buf: [8192]u8 = undefined;
    const fd = std.c.open("/proc/self/status", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    var lines = std.mem.splitScalar(u8, buf[0..@intCast(n)], '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "VmHWM:")) continue;
        var fields = std.mem.tokenizeAny(u8, line[6..], " \t");
        const kib = std.fmt.parseInt(u64, fields.next() orelse return null, 10) catch return null;
        if (!std.mem.eql(u8, fields.next() orelse return null, "kB")) return null;
        return std.math.mul(u64, kib, 1024) catch null;
    }
    return null;
}

/// A platform owner adapts its DFlash Generation here. One call must be one Generation.runRound.
pub const DflashSource = struct {
    ctx: *anyopaque,
    seed: *const fn (ctx: *anyopaque, tokens: []const u32, depth: u8) anyerror!u32,
    round: *const fn (ctx: *anyopaque, remaining: u32, out: []u32, audit: *bench.DflashAudit, alloc: std.mem.Allocator) anyerror!u32,
    reset: *const fn (ctx: *anyopaque) anyerror!void,
};

pub const DflashBackend = enum { none, metal, cuda };

pub fn platformDflashBackend(os: std.Target.Os.Tag, enabled: bool, lane_host: bool) DflashBackend {
    if (!enabled or !lane_host) return .none;
    return switch (os) {
        .macos => .metal,
        .linux => .cuda,
        else => .none,
    };
}

pub fn platformDflash(
    os: std.Target.Os.Tag,
    engine: api.Engine,
    enabled: bool,
    binding: *EngineDflashBinding,
) ?DflashSource {
    return switch (platformDflashBackend(os, enabled, engine.info().lane_host)) {
        .metal, .cuda => binding.source(),
        .none => null,
    };
}

pub fn residentOpenRequest(dir: []const u8, model_type: []const u8, drafter: ?[]const u8) api.Open {
    return .{
        .dir = dir,
        .model_type = model_type,
        .lanes = 1,
        .lanes_fixed = true,
        .drafter = drafter,
        .prompt_cache_gib = 0,
    };
}

/// Run the loaded drafter through the public lane request API.
pub const EngineDflashBinding = struct {
    allocator: std.mem.Allocator,
    engine: api.Engine,
    host: *api.LaneHost,
    owned: bool = false,
    opened: bool = false,
    previous_most_drafts: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, engine: api.Engine) EngineDflashBinding {
        return .{ .allocator = allocator, .engine = engine, .host = @ptrCast(@alignCast(engine.ctx)) };
    }

    pub fn deinit(binding: *EngineDflashBinding) void {
        binding.host.lock();
        defer binding.host.unlock();
        binding.resetNow();
    }

    pub fn source(binding: *EngineDflashBinding) DflashSource {
        return .{ .ctx = binding, .seed = seed, .round = round, .reset = reset };
    }

    fn seed(ptr: *anyopaque, tokens: []const u32, depth: u8) anyerror!u32 {
        const binding: *EngineDflashBinding = @ptrCast(@alignCast(ptr));
        if (depth != 1) return error.BadDraftDepth;
        const h = binding.host;
        h.lock();
        defer h.unlock();
        try binding.ready(false);
        binding.resetNow();
        binding.previous_most_drafts = h.core.rule.most_drafts;
        h.core.rule.most_drafts = depth;
        errdefer binding.resetNow();
        const context = binding.engine.info().context_window;
        if (tokens.len == 0) return error.EmptyPrompt;
        if (tokens.len >= context) return error.ContextFull;
        const Stream = @TypeOf(h.core.bench_stream.?);
        h.core.bench_stream = try Stream.init(binding.allocator, .{
            .id = "bench",
            .prompt = tokens,
            .max_new = context - @as(u32, @intCast(tokens.len)),
            .drafts = true,
        });
        binding.owned = true;
        binding.opened = true;
        try h.core.addStream(&h.core.bench_stream.?);
        if (!removeLive(h, &h.core.bench_stream.?)) return error.BadDraftRound;
        const emitted = h.core.bench_stream.?.emitted();
        if (emitted.len != 1) return error.BadDraftRound;
        return emitted[0];
    }

    fn round(ptr: *anyopaque, remaining: u32, out: []u32, audit: *bench.DflashAudit, alloc: std.mem.Allocator) anyerror!u32 {
        const binding: *EngineDflashBinding = @ptrCast(@alignCast(ptr));
        if (remaining == 0 or remaining > out.len) return error.BadDraftRound;
        const h = binding.host;
        h.lock();
        defer h.unlock();
        try binding.ready(true);
        errdefer binding.resetNow();
        const s = if (h.core.bench_stream) |*stream| stream else return error.SessionNotSeeded;
        const target = std.math.add(u32, @intCast(s.emitted().len), remaining) catch return error.ContextFull;
        if (target > s.max_new) return error.ContextFull;
        s.max_new = target;
        var written: usize = 0;
        while (written < remaining) {
            const before = s.emitted().len;
            const drafted = s.drafted;
            const accepted = s.accepted;
            try h.core.live.append(binding.allocator, s);
            try h.core.step();
            _ = removeLive(h, s);
            if (s.finished) binding.opened = false;
            const emitted = s.emitted();
            if (emitted.len <= before or emitted.len - before > remaining - written) return error.BadDraftRound;
            const committed: u32 = @intCast(emitted.len - before);
            @memcpy(out[written..][0..committed], emitted[before..]);
            try audit.appendRound(
                alloc,
                committed,
                std.math.cast(u32, s.drafted - drafted) orelse return error.BadDraftAudit,
                std.math.cast(u32, s.accepted - accepted) orelse return error.BadDraftAudit,
            );
            written += committed;
        }
        if (!s.finished) return error.BadDraftRound;
        binding.opened = false;
        s.deinit(binding.allocator);
        h.core.bench_stream = null;
        binding.owned = false;
        binding.restoreDepth();
        return remaining;
    }

    fn reset(ptr: *anyopaque) anyerror!void {
        const binding: *EngineDflashBinding = @ptrCast(@alignCast(ptr));
        binding.host.lock();
        defer binding.host.unlock();
        binding.resetNow();
    }

    fn ready(binding: *EngineDflashBinding, active: bool) !void {
        const h = binding.host;
        if (h.closing) return error.Closed;
        if (h.queued.items.len != 0 or h.admitted.items.len != 0 or h.filling.items.len != 0) return error.Busy;
        if (h.core.activeCount() != 0) return error.Busy;
        if (active != binding.owned) return error.SessionNotSeeded;
        if (h.on_thread) |start| try start.run(start.ctx);
    }

    fn removeLive(h: *api.LaneHost, s: *@TypeOf(h.core.bench_stream.?)) bool {
        for (h.core.live.items, 0..) |stream, i| {
            if (stream != s) continue;
            _ = h.core.live.orderedRemove(i);
            return true;
        }
        return false;
    }

    fn resetNow(binding: *EngineDflashBinding) void {
        const h = binding.host;
        if (binding.owned) if (h.core.bench_stream) |*s| {
            _ = removeLive(h, s);
            if (binding.opened) h.core.release(s);
            s.deinit(binding.allocator);
            h.core.bench_stream = null;
        };
        binding.owned = false;
        binding.opened = false;
        binding.restoreDepth();
    }

    fn restoreDepth(binding: *EngineDflashBinding) void {
        if (binding.previous_most_drafts) |value| binding.host.core.rule.most_drafts = value;
        binding.previous_most_drafts = null;
    }
};

/// Depth-one siblings a DFlash round verifies: the pending row and 4 drafts make 5 verify rows in the 8-row decode projections, whose
/// weight reads and tensor-op tiles do not grow with the rows (rows8), so each further sibling only adds its chance to land.
pub const dflash_width = 4;
/// A sibling the drafter gives less than this probability is left out: its verify row (GDN tree step, attention row,
/// argmax) costs more of the round than its chance to land returns. The round's best draft is always verified.
/// Siblings at 0.7-1% land about 2% of the time, a full round each, against about 0.13 ms a row; below 0.7% they
/// almost never land.
pub const dflash_floor = 0.007;

/// Bind the Metal DFlash Generation without importing a platform module into host or Linux builds.
pub fn GenerationBinding(comptime generation: type) type {
    return struct {
        const Generation = generation.Generation;
        generation: *Generation,
        chunk: usize,
        host: ?*api.LaneHost = null,
        depth: u8 = 0,

        pub fn source(binding: *@This()) DflashSource {
            return .{ .ctx = binding, .seed = seed, .round = round, .reset = reset };
        }

        fn seed(ptr: *anyopaque, tokens: []const u32, depth: u8) anyerror!u32 {
            const binding: *@This() = @ptrCast(@alignCast(ptr));
            if (depth != 1) return error.BadDraftDepth;
            if (binding.host) |host| host.lock();
            defer if (binding.host) |host| host.unlock();
            try binding.ready();
            try binding.resetNow();
            binding.depth = depth;
            try binding.generation.prefill(tokens, binding.chunk);
            const result = try binding.generation.runRound(1, .{ .nodes = dflash_width, .children = dflash_width, .floor = dflash_floor, .max_depth = depth }, &.{}, false, null, 0);
            defer binding.generation.allocator.free(result.tokens);
            if (result.tokens.len != 1) return error.BadDraftRound;
            return result.tokens[0];
        }

        fn round(ptr: *anyopaque, remaining: u32, out: []u32, audit: *bench.DflashAudit, alloc: std.mem.Allocator) anyerror!u32 {
            const binding: *@This() = @ptrCast(@alignCast(ptr));
            if (binding.host) |host| host.lock();
            defer if (binding.host) |host| host.unlock();
            try binding.ready();
            if (binding.depth != 1) return error.SessionNotSeeded;
            if (binding.generation.journal != null) return error.AuditAlreadyAttached;
            var journal: generation.AuditJournal = .{};
            defer journal.deinit(binding.generation.allocator);
            binding.generation.journal = &journal;
            defer binding.generation.journal = null;
            const result = try binding.generation.runRound(remaining, .{ .nodes = dflash_width, .children = dflash_width, .floor = dflash_floor, .max_depth = binding.depth }, &.{}, true, null, 0);
            defer binding.generation.allocator.free(result.tokens);
            if (result.tokens.len == 0 or result.tokens.len > out.len) return error.BadDraftRound;
            if (result.rounds != 1) return error.BadDraftRound;
            @memcpy(out[0..result.tokens.len], result.tokens);
            try generation.appendTo(&journal, audit, alloc);
            return @intCast(result.tokens.len);
        }

        fn reset(ptr: *anyopaque) anyerror!void {
            const binding: *@This() = @ptrCast(@alignCast(ptr));
            if (binding.host) |host| host.lock();
            defer if (binding.host) |host| host.unlock();
            try binding.resetNow();
        }

        /// The lane host idle, as EngineDflashBinding.ready requires, and its thread hook run.
        fn ready(binding: *@This()) !void {
            const h = binding.host orelse return;
            if (h.closing) return error.Closed;
            if (h.queued.items.len != 0 or h.admitted.items.len != 0 or h.filling.items.len != 0) return error.Busy;
            if (h.core.activeCount() != 0) return error.Busy;
            if (h.on_thread) |start| try start.run(start.ctx);
        }

        fn resetNow(binding: *@This()) !void {
            try binding.generation.settle();
            try binding.generation.runner.reset(0);
            try binding.generation.draft.reset();
            binding.depth = 0;
        }
    };
}

pub const Adapter = struct {
    allocator: std.mem.Allocator,
    engine: api.Engine,
    dflash: ?DflashSource = null,
    next_id: api.Id = 1,
    retained_work: u64 = 0,
    free_work: u64 = 0,
    free_spec: bench.FreeSpec = .serial,
    free_pending: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, engine: api.Engine, dflash: ?DflashSource) Adapter {
        return .{ .allocator = allocator, .engine = engine, .dflash = dflash };
    }

    pub fn session(a: *Adapter) bench.BenchSession {
        return fromEngine(a);
    }
};

pub fn fromEngine(adapter: *Adapter) bench.BenchSession {
    return .{ .ptr = adapter, .vtable = &vtable };
}

fn self(ptr: *anyopaque) *Adapter {
    return @ptrCast(@alignCast(ptr));
}

const Run = struct {
    out: []u32,
    written: usize = 0,
    failed: bool = false,
    done: std.atomic.Value(bool) = .init(false),

    fn event(ptr: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const run: *Run = @ptrCast(@alignCast(ptr));
        switch (ev.*) {
            .tokens => |tokens| {
                if (tokens.len > run.out.len -| run.written) {
                    run.failed = true;
                } else {
                    @memcpy(run.out[run.written..][0..tokens.len], tokens);
                    run.written += tokens.len;
                }
            },
            .finished => |finished| {
                run.failed = run.failed or finished.reason == .failed or finished.reason == .cancelled;
                run.done.store(true, .release);
            },
            else => {},
        }
    }

    fn wait(run: *Run) !void {
        while (!run.done.load(.acquire)) std.Thread.yield() catch {};
        if (run.failed) return error.EngineFailed;
    }
};

fn submit(a: *Adapter, tokens: []const u32, steps: u32, out: []u32) !void {
    if (out.len < steps) return error.OutputTooSmall;
    if (steps == 0) return;
    var run: Run = .{ .out = out[0..steps] };
    const request: api.Request = .{ .prompt = tokens, .max_tokens = steps, .drafts = false };
    const id = a.next_id;
    a.next_id +%= 1;
    try a.engine.submit(id, &request, .{ .ctx = &run, .event = Run.event });
    try run.wait();
    if (run.written != steps) return error.ShortGeneration;
}

fn prefillFn(ptr: *anyopaque, tokens: []const u32) anyerror!u32 {
    var out: [1]u32 = undefined;
    try submit(self(ptr), tokens, 1, &out);
    return out[0];
}

fn generateFn(ptr: *anyopaque, tokens: []const u32, steps: u32, out: []u32) anyerror!void {
    return submit(self(ptr), tokens, steps, out);
}

fn toStep(step: api.RetainedStep) bench.Step {
    var top: [8]bench.TopLogit = @splat(.{ .token = 0, .logit = 0 });
    if (step.has_top) {
        for (&top, step.top) |*to, from| to.* = .{ .token = from.token, .logit = from.logit };
    }
    return .{ .token = step.token, .top = top, .has_top = step.has_top };
}

pub fn testStep(step: api.RetainedStep) bench.Step {
    return toStep(step);
}

fn seedFn(ptr: *anyopaque, tokens: []const u32, want_top: bool) anyerror!bench.Step {
    const a = self(ptr);
    const result = try a.engine.retainedSeed(tokens, want_top);
    a.retained_work += 1;
    return toStep(result);
}

fn forceFn(ptr: *anyopaque, token: u32, want_top: bool) anyerror!bench.Step {
    const a = self(ptr);
    const result = try a.engine.retainedForce(token, want_top);
    a.retained_work += 1;
    return toStep(result);
}

fn seedFreeFn(ptr: *anyopaque, tokens: []const u32, spec: bench.FreeSpec) anyerror!u32 {
    const a = self(ptr);
    errdefer resetFn(ptr);
    const token = switch (spec) {
        .serial => blk: {
            const result = try a.engine.retainedSeed(tokens, false);
            a.retained_work += 1;
            break :blk result.token;
        },
        .dflash => |d| blk: {
            if (d.depth != 1) return error.BadDraftDepth;
            const source = a.dflash orelse return error.Unsupported;
            const result = try source.seed(source.ctx, tokens, d.depth);
            a.free_work = 1;
            break :blk result;
        },
    };
    a.free_spec = spec;
    a.free_pending = token;
    return token;
}

fn runFreeFn(ptr: *anyopaque, count: u32, out: []u32, audit: *bench.DflashAudit) anyerror!void {
    const a = self(ptr);
    errdefer resetFn(ptr);
    if (out.len < count) return error.OutputTooSmall;
    var pending = a.free_pending orelse return error.SessionNotSeeded;
    var written: u32 = 0;
    while (written < count) switch (a.free_spec) {
        .serial => {
            const next = try a.engine.retainedForce(pending, false);
            a.retained_work += 1;
            pending = next.token;
            out[written] = pending;
            written += 1;
            try audit.appendRound(a.allocator, 1, 0, 0);
        },
        .dflash => {
            const source = a.dflash orelse return error.Unsupported;
            const rounds_before = audit.acceptance_lengths.items.len;
            const committed = try source.round(source.ctx, count - written, out[written..count], audit, a.allocator);
            if (committed == 0 or committed > count - written) return error.BadDraftRound;
            const rounds = audit.acceptance_lengths.items.len - rounds_before;
            if (rounds == 0) return error.BadDraftAudit;
            a.free_work += rounds;
            written += committed;
            pending = out[written - 1];
        },
    };
    a.free_pending = pending;
    try audit.check(count);
}

fn closePhaseFn(ptr: *anyopaque) anyerror!bench.PhaseDiag {
    const a = self(ptr);
    if (a.dflash) |source| try source.reset(source.ctx);
    const retained = try a.engine.retainedClosePhase();
    const peak = processPeak();
    const result: bench.PhaseDiag = .{
        .work_tokens = retained.work_tokens + a.free_work,
        .peak_ram_bytes = peak,
        .cache_bytes = retained.cache_bytes,
    };
    a.free_work = 0;
    a.retained_work = 0;
    a.free_spec = .serial;
    a.free_pending = null;
    return result;
}

fn inspectPhaseFn(ptr: *anyopaque) anyerror!bench.PhaseDiag {
    const a = self(ptr);
    const memory = a.engine.memory(false);
    const peak = processPeak();
    return .{
        .work_tokens = a.retained_work + a.free_work,
        .peak_ram_bytes = peak,
        .cache_bytes = if (memory) |m| m.cache else 0,
    };
}

fn resetFn(ptr: *anyopaque) void {
    const a = self(ptr);
    if (a.dflash) |source| source.reset(source.ctx) catch {};
    a.engine.retainedReset() catch {};
    a.free_work = 0;
    a.retained_work = 0;
    a.free_spec = .serial;
    a.free_pending = null;
}

const vtable: bench.BenchSession.VTable = .{
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
