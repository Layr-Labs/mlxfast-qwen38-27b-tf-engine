const std = @import("std");
const builtin = @import("builtin");
const bench = @import("bench_session");
const protocol = @import("protocol.zig");
const limits = @import("resident_client.zig");

pub const Identity = struct {
    backend: []const u8,
    device: []const u8,
    manifest_digest: ?[]const u8 = null,
    capabilities: []const []const u8 = &.{},
    runner_id: []const u8 = "tensorfold/qwen3.8-27b",
    model_type: []const u8 = "qwen3_5",
    build: []const u8 = "tensorfold-1.1.6",
    failure: ?FailureSource = null,

    pub fn fromResident(
        hello: *const limits.ResidentHello,
        capabilities: []const []const u8,
        failure: ?FailureSource,
    ) Identity {
        return .{
            .backend = hello.backend,
            .device = hello.device,
            .manifest_digest = &hello.manifest_digest,
            .capabilities = capabilities,
            .failure = failure,
        };
    }
};

pub const FailureSource = struct {
    ctx: *anyopaque,
    reason: *const fn (ctx: *anyopaque, err: anyerror) []const u8,
};

const Phase = enum { none, prefill, decode, correctness, correctness_anchor, free_decode, free_done };
pub const max_request_line_bytes: usize = 16 * 1024 * 1024;
const serial_spec_modes = [_][]const u8{"serial"};
const free_spec_modes = [_][]const u8{ "serial", "dflash" };

pub fn run(
    alloc: std.mem.Allocator,
    session: bench.BenchSession,
    ident: Identity,
    reader: *std.Io.Reader,
    writer: *std.Io.Writer,
) !void {
    return runReporting(alloc, session, ident, reader, writer, null);
}

pub fn runReporting(
    alloc: std.mem.Allocator,
    session: bench.BenchSession,
    ident: Identity,
    reader: *std.Io.Reader,
    writer: *std.Io.Writer,
    failure_writer: ?*std.Io.Writer,
) !void {
    try validateIdentity(ident);
    var nonce_bytes: [16]u8 = undefined;
    try randomBytes(&nonce_bytes);
    const nonce_hex = std.fmt.bytesToHex(nonce_bytes, .lower);
    const nonce = nonce_hex[0..];
    const free_run_enabled = hasCapability(ident.capabilities, "free_run_decode");

    var hello = protocol.Response.success(0, nonce);
    hello.expert_stats = .{};
    hello.protocol_version = protocol.protocol_version;
    hello.backend = ident.backend;
    hello.device = ident.device;
    hello.capabilities = ident.capabilities;
    hello.spec_modes = if (free_run_enabled) &free_spec_modes else &serial_spec_modes;
    if (ident.manifest_digest) |digest| {
        hello.runner = .{
            .id = ident.runner_id,
            .model_type = ident.model_type,
            .manifest_sha256 = digest,
            .build = ident.build,
        };
    }
    try protocol.writeResponse(writer, hello);

    var phase: Phase = .none;
    var free_spec: bench.FreeSpec = .serial;
    defer session.reset();
    while (true) {
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const request_alloc = arena.allocator();
        const framed = try readRequestLine(request_alloc, reader) orelse break;
        const line = switch (framed) {
            .line => |value| value,
            .invalid => |can_continue| {
                session.reset();
                phase = .none;
                free_spec = .serial;
                try protocol.writeResponse(writer, protocol.Response.failure(-1, nonce, "invalid request"));
                if (can_continue) continue else break;
            },
        };
        var parsed = protocol.parseRequest(request_alloc, line) catch {
            session.reset();
            phase = .none;
            free_spec = .serial;
            try protocol.writeResponse(writer, protocol.Response.failure(-1, nonce, "invalid request"));
            continue;
        };
        defer parsed.deinit();
        const request = parsed.value;

        var engine_failure: ?anyerror = null;
        const response = dispatch(request_alloc, alloc, session, &phase, &free_spec, nonce, free_run_enabled, request, &engine_failure) catch |err| blk: {
            const failure = engine_failure orelse err;
            const reason = if (ident.failure) |source| source.reason(source.ctx, failure) else @errorName(failure);
            reportFailure(failure_writer, request.kind, reason);
            session.reset();
            phase = .none;
            free_spec = .serial;
            break :blk protocol.Response.failure(request.id, nonce, @errorName(err));
        };
        try protocol.writeResponse(writer, response);
    }
}

fn reportFailure(writer: ?*std.Io.Writer, op: []const u8, reason: []const u8) void {
    if (writer) |out| {
        out.print("bench-worker: {s} failed: {s}\n", .{ op, reason }) catch return;
        out.flush() catch {};
        return;
    }
    std.debug.print("bench-worker: {s} failed: {s}\n", .{ op, reason });
}

const FramedLine = union(enum) {
    line: []const u8,
    invalid: bool,
};

fn readRequestLine(alloc: std.mem.Allocator, reader: *std.Io.Reader) !?FramedLine {
    var line: std.Io.Writer.Allocating = .init(alloc);
    defer line.deinit();
    _ = reader.streamDelimiterLimit(&line.writer, '\n', .limited(max_request_line_bytes + 1)) catch |err| switch (err) {
        error.StreamTooLong => {
            const can_continue = if (reader.discardDelimiterInclusive('\n')) |_| true else |discard_err| switch (discard_err) {
                error.EndOfStream => false,
                else => return discard_err,
            };
            return .{ .invalid = can_continue };
        },
        else => return err,
    };
    _ = reader.takeByte() catch |err| switch (err) {
        error.EndOfStream => {
            if (line.written().len == 0) return null;
            return .{ .invalid = false };
        },
        else => return err,
    };
    return .{ .line = try line.toOwnedSlice() };
}

fn randomBytes(bytes: []u8) !void {
    if (comptime @TypeOf(std.c.arc4random_buf) != void) {
        std.c.arc4random_buf(bytes.ptr, bytes.len);
        return;
    }
    if (comptime builtin.os.tag == .linux) {
        var written: usize = 0;
        while (written < bytes.len) {
            const rc = std.os.linux.getrandom(bytes[written..].ptr, bytes.len - written, 0);
            switch (std.os.linux.errno(rc)) {
                .SUCCESS => written += rc,
                .INTR => continue,
                else => return error.EntropyUnavailable,
            }
        }
        return;
    }
    return error.EntropyUnavailable;
}

fn dispatch(
    alloc: std.mem.Allocator,
    session_alloc: std.mem.Allocator,
    session: bench.BenchSession,
    phase: *Phase,
    free_spec: *bench.FreeSpec,
    nonce: []const u8,
    free_run_enabled: bool,
    request: protocol.Request,
    engine_failure: *?anyerror,
) !protocol.Response {
    const kind = try request.resolvedKind();
    try validateFields(request, kind);
    var response = protocol.Response.success(request.id, nonce);

    switch (kind) {
        .prefill => {
            session.reset();
            const tokens = try tokenSlice(alloc, request.prompt_tokens.?);
            defer alloc.free(tokens);
            response.token = try engineCall(u32, engine_failure, session.prefill(tokens));
            phase.* = .prefill;
        },
        .decode_begin => {
            session.reset();
            if (request.spec) |spec| if (!protocol.isSerial(spec)) return error.UnsupportedSpec;
            const tokens = try tokenSlice(alloc, request.seed_tokens.?);
            defer alloc.free(tokens);
            const step = try engineCall(bench.Step, engine_failure, session.seed(tokens, false));
            response.seed_token = step.token;
            if (request.spec != null) response.effective_spec = protocol.EffectiveSpec.serial();
            phase.* = .decode;
        },
        .decode_step => {
            if (phase.* != .decode) return error.WrongPhase;
            const step = try engineCall(bench.Step, engine_failure, session.force(try tokenId(request.token.?), false));
            response.token = step.token;
        },
        .correctness => {
            session.reset();
            const steps = try boundedCount(request.steps.?, limits.max_correctness_steps);
            const prompt = try tokenSlice(alloc, request.prompt_tokens.?);
            defer alloc.free(prompt);
            const out = try alloc.alloc(u32, steps);
            try engineCall(void, engine_failure, session.generate(prompt, steps, out));
            response.tokens = out;
            response.peak_ram_gb = peakRamGb(try engineCall(bench.PhaseDiag, engine_failure, session.inspectPhase()));
            phase.* = .correctness;
        },
        .correctness_begin => {
            session.reset();
            const tokens = try tokenSlice(alloc, request.prompt_tokens.?);
            defer alloc.free(tokens);
            const step = try engineCall(bench.Step, engine_failure, session.seed(tokens, true));
            try correctnessResponse(alloc, session, step, &response, engine_failure);
            phase.* = .correctness_anchor;
        },
        .correctness_step => {
            if (phase.* != .correctness_anchor) return error.WrongPhase;
            const step = try engineCall(bench.Step, engine_failure, session.force(try tokenId(request.token.?), true));
            try correctnessResponse(alloc, session, step, &response, engine_failure);
        },
        .phase_diagnostics => {
            if (phase.* == .none) return error.NoOpenPhase;
            const diag = try engineCall(bench.PhaseDiag, engine_failure, session.closePhase());
            response.expert_stats = .{};
            response.peak_ram_gb = peakRamGb(diag);
            response.completed_work = try protocol.checkedDiagnostic(diag.work_tokens);
            response.cache_memory = try protocol.checkedDiagnostic(diag.cache_bytes);
            phase.* = .none;
        },
        .free_decode_begin => {
            if (!free_run_enabled) return error.CapabilityNotAdvertised;
            session.reset();
            const resolved = try resolveFreeSpec(request.spec);
            const tokens = try tokenSlice(alloc, request.seed_tokens.?);
            defer alloc.free(tokens);
            response.seed_token = try engineCall(u32, engine_failure, session.seedFree(tokens, resolved.session));
            response.effective_spec = resolved.wire;
            free_spec.* = resolved.session;
            phase.* = .free_decode;
        },
        .free_decode_run => {
            if (!free_run_enabled) return error.CapabilityNotAdvertised;
            if (phase.* != .free_decode) return error.WrongPhase;
            const count = request.count.?;
            if (count > limits.max_free_count) return error.CountTooLarge;
            const out = try alloc.alloc(u32, count);
            var audit = bench.DflashAudit.init();
            defer audit.deinit(session_alloc);
            // Organizer pre-launch test: a fixed 70 ms pause inside the timed
            // decode run. It slows the run on purpose and leaves every token unchanged.
            var delay: std.c.timespec = .{ .sec = 0, .nsec = 70 * std.time.ns_per_ms };
            while (std.c.nanosleep(&delay, &delay) != 0) {}
            try engineCall(void, engine_failure, session.runFree(count, out, &audit));
            try audit.check(count);
            response.tokens = out;
            response.acceptance_lengths = try alloc.dupe(u32, audit.acceptance_lengths.items);
            response.drafted_total = audit.drafted_total;
            response.accepted_total = audit.accepted_total;
            response.committed_total = audit.committed_total;
            switch (free_spec.*) {
                .serial => {
                    response.spec_decoder = "serial";
                    response.verification_mode = "serial";
                },
                .dflash => {
                    response.spec_decoder = "dflash";
                    response.verification_mode = "rectangular";
                },
            }
            phase.* = .free_done;
        },
    }
    return response;
}

fn hasCapability(capabilities: []const []const u8, wanted: []const u8) bool {
    for (capabilities) |capability| if (std.mem.eql(u8, capability, wanted)) return true;
    return false;
}

fn validateIdentity(ident: Identity) !void {
    if (ident.manifest_digest) |digest| {
        if (digest.len != 64) return error.InvalidManifestDigest;
        for (digest) |byte| if (!std.ascii.isDigit(byte) and !(byte >= 'a' and byte <= 'f')) {
            return error.InvalidManifestDigest;
        };
    }
    for (ident.capabilities) |capability| {
        if (std.mem.eql(u8, capability, "batched_free_run_decode") or
            std.mem.eql(u8, capability, "per_stream_timing") or
            std.mem.eql(u8, capability, "cohort_reference_replay"))
        {
            return error.UnsupportedCapability;
        }
    }
}

fn validateFields(request: protocol.Request, kind: protocol.Kind) !void {
    const prompt = request.prompt_tokens != null;
    const token = request.token != null;
    const seed = request.seed_tokens != null;
    const steps = request.steps != null;
    const count = request.count != null;
    const spec = request.spec != null;
    const valid = switch (kind) {
        .prefill, .correctness_begin => prompt and !token and !seed and !steps and !count and !spec,
        .correctness => prompt and !token and !seed and steps and !count and !spec,
        .decode_begin, .free_decode_begin => !prompt and !token and seed and !steps and !count,
        .decode_step, .correctness_step => !prompt and token and !seed and !steps and !count and !spec,
        .phase_diagnostics => !prompt and !token and !seed and !steps and !count and !spec,
        .free_decode_run => !prompt and !token and !seed and !steps and count and !spec,
    };
    if (!valid) return error.InvalidFields;
}

fn tokenId(value: i64) !u32 {
    if (value < 0 or value > std.math.maxInt(u32)) return error.InvalidToken;
    return @intCast(value);
}

fn tokenSlice(alloc: std.mem.Allocator, values: []const i64) ![]u32 {
    if (values.len == 0) return error.EmptyTokens;
    const result = try alloc.alloc(u32, values.len);
    errdefer alloc.free(result);
    for (values, result) |value, *out| out.* = try tokenId(value);
    return result;
}

fn boundedCount(value: i64, max: u32) !u32 {
    if (value < 0 or value > max) return error.InvalidCount;
    return @intCast(value);
}

fn peakRamGb(diag: bench.PhaseDiag) ?f64 {
    const bytes = diag.peak_ram_bytes orelse return null;
    return @as(f64, @floatFromInt(bytes)) / 1_000_000_000.0;
}

fn correctnessResponse(
    alloc: std.mem.Allocator,
    session: bench.BenchSession,
    step: bench.Step,
    response: *protocol.Response,
    engine_failure: *?anyerror,
) !void {
    if (!step.has_top) return error.MissingTopLogits;
    const top = try alloc.alloc(protocol.TopLogit, protocol.top_logits_k);
    for (step.top, top) |entry, *out| {
        if (!std.math.isFinite(entry.logit)) return error.InvalidLogit;
        out.* = .{ .token = entry.token, .logit = entry.logit };
    }
    response.token = step.token;
    response.top_logits = top;
    response.expert_stats = .{};
    response.peak_ram_gb = peakRamGb(try engineCall(bench.PhaseDiag, engine_failure, session.inspectPhase()));
}

fn engineCall(comptime T: type, failure: *?anyerror, result: anyerror!T) anyerror!T {
    return result catch |err| {
        failure.* = err;
        return err;
    };
}

const ResolvedSpec = struct { session: bench.FreeSpec, wire: protocol.EffectiveSpec };

fn resolveFreeSpec(spec: ?protocol.SpecConfig) !ResolvedSpec {
    const value = spec orelse return .{ .session = .serial, .wire = protocol.EffectiveSpec.serial() };
    if (protocol.isSerial(value)) return .{ .session = .serial, .wire = protocol.EffectiveSpec.serial() };
    const depth = try protocol.parseDflash(value);
    return .{
        .session = .{ .dflash = .{ .depth = depth } },
        .wire = protocol.EffectiveSpec.withDflash(depth),
    };
}
