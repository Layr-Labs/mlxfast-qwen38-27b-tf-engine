const std = @import("std");
const bench = @import("bench_session");
const protocol = @import("protocol.zig");
const resident_client = @import("resident_client.zig");
const worker = @import("worker.zig");

const caps = [_][]const u8{"free_run_decode"};
const manifest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn identity() worker.Identity {
    return .{
        .backend = "mock",
        .device = "host",
        .manifest_digest = manifest,
        .capabilities = &caps,
        .runner_id = "tensorfold/mock",
        .model_type = "qwen3_5",
        .build = "test",
    };
}

fn runText(mock: *bench.MockSession, input: []const u8) ![]u8 {
    return runTextWithIdentity(mock, identity(), input);
}

fn runTextWithIdentity(mock: *bench.MockSession, ident: worker.Identity, input: []const u8) ![]u8 {
    var reader = std.Io.Reader.fixed(input);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer output.deinit();
    var failures: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer failures.deinit();
    try worker.runReporting(std.testing.allocator, mock.session(), ident, &reader, &output.writer, &failures.writer);
    return output.toOwnedSlice();
}

fn lineAt(text: []const u8, index: usize) !std.json.Parsed(std.json.Value) {
    var lines = std.mem.splitScalar(u8, text, '\n');
    var at: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (at == index) return std.json.parseFromSlice(std.json.Value, std.testing.allocator, line, .{});
        at += 1;
    }
    return error.MissingLine;
}

fn expectKeys(object: std.json.ObjectMap, expected: []const []const u8) !void {
    try std.testing.expectEqual(expected.len, object.count());
    for (expected) |key| try std.testing.expect(object.get(key) != null);
}

fn expectCommon(object: std.json.ObjectMap, id: i64, ok: bool) !void {
    try std.testing.expectEqual(id, object.get("id").?.integer);
    try std.testing.expectEqual(ok, object.get("ok").?.bool);
    try std.testing.expect(object.get("nonce").?.string.len == 32);
}

fn arrayIntegers(value: std.json.Value, expected: []const i64) !void {
    try std.testing.expect(value == .array);
    try std.testing.expectEqual(expected.len, value.array.items.len);
    for (value.array.items, expected) |got, want| try std.testing.expectEqual(want, got.integer);
}

fn number(value: std.json.Value) f64 {
    return switch (value) {
        .integer => |integer| @floatFromInt(integer),
        .float => |float| float,
        else => unreachable,
    };
}

test "hello response has v1 identity, no resident, and no cohort capability" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(&mock, "");
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 0);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{
        "id",           "nonce",      "ok",     "expert_stats", "protocol_version", "backend", "device",
        "capabilities", "spec_modes", "runner",
    });
    try expectCommon(object, 0, true);
    try std.testing.expectEqual(@as(i64, 1), object.get("protocol_version").?.integer);
    try std.testing.expectEqualStrings("mock", object.get("backend").?.string);
    try std.testing.expectEqualStrings("host", object.get("device").?.string);
    try std.testing.expectEqualStrings(manifest, object.get("runner").?.object.get("manifest_sha256").?.string);
    try std.testing.expect(object.get("resident") == null);
    try std.testing.expectEqualStrings("free_run_decode", object.get("capabilities").?.array.items[0].string);
    try std.testing.expectEqual(@as(usize, 2), object.get("spec_modes").?.array.items.len);
    try expectKeys(object.get("expert_stats").?.object, &.{
        "expert_cache_hits", "expert_cache_misses", "expert_cache_evictions",
        "expert_bytes_read", "expert_read_seconds", "expert_peak_cached_tensors",
    });
    try expectKeys(object.get("runner").?.object, &.{ "id", "model_type", "manifest_sha256", "build" });
}

test "outer hello omits changing resident identities" {
    const residents = [_]protocol.ResidentIdentity{
        .{ .pid = 57, .load_epoch = 100 },
        .{ .pid = 57, .load_epoch = 200 },
    };
    for (residents) |resident_identity| {
        var backend = "mock".*;
        var device = "host".*;
        var revision = "revision".*;
        var resident_hello: resident_client.ResidentHello = .{
            .backend = &backend,
            .device = &device,
            .manifest_digest = manifest.*,
            .resident = resident_identity,
            .weights_revision = &revision,
        };
        var mock = bench.MockSession.init(std.testing.allocator);
        const text = try runTextWithIdentity(
            &mock,
            .fromResident(&resident_hello, &caps, null),
            "",
        );
        defer std.testing.allocator.free(text);
        var parsed = try lineAt(text, 0);
        defer parsed.deinit();
        const object = parsed.value.object;
        try std.testing.expectEqual(resident_identity, resident_hello.resident);
        try std.testing.expect(object.get("resident") == null);
        try std.testing.expectEqualStrings("mock", object.get("backend").?.string);
        try std.testing.expectEqualStrings("host", object.get("device").?.string);
        try std.testing.expectEqualStrings(manifest, object.get("runner").?.object.get("manifest_sha256").?.string);
        try std.testing.expectEqualStrings("free_run_decode", object.get("capabilities").?.array.items[0].string);
    }
}

test "prefill request returns one materialized token" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(&mock, "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n");
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 1);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{ "id", "nonce", "ok", "token" });
    try expectCommon(object, 1, true);
    try std.testing.expectEqual(@as(i64, 69), object.get("token").?.integer);
}

test "decode_begin accepts the serial spec and echoes it" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":2,\"kind\":\"decode_begin\",\"seed_tokens\":[2],\"spec\":{\"mode\":\"serial\"}}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 1);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{ "id", "nonce", "ok", "seed_token", "effective_spec" });
    try expectCommon(object, 2, true);
    try std.testing.expectEqual(@as(i64, 69), object.get("seed_token").?.integer);
    try expectKeys(object.get("effective_spec").?.object, &.{"mode"});
}

test "decode_step uses the retained decode session" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"decode_begin\",\"seed_tokens\":[2]}\n" ++
            "{\"id\":2,\"kind\":\"decode_step\",\"token\":69}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 2);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{ "id", "nonce", "ok", "token" });
    try expectCommon(object, 2, true);
    try std.testing.expectEqual(@as(i64, 2146), object.get("token").?.integer);
}

test "correctness request returns the requested token count and peak RAM" {
    var mock = bench.MockSession.init(std.testing.allocator);
    mock.peak_ram_bytes = 2_000_000_000;
    const text = try runText(
        &mock,
        "{\"id\":3,\"kind\":\"correctness\",\"prompt_tokens\":[2],\"steps\":3}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 1);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{ "id", "nonce", "ok", "tokens", "peak_ram_gb" });
    try expectCommon(object, 3, true);
    try arrayIntegers(object.get("tokens").?, &.{ 69, 2146, 66533 });
    try std.testing.expectEqual(@as(f64, 2), number(object.get("peak_ram_gb").?));
}

test "correctness_begin returns eight raw logits and zero expert stats" {
    var mock = bench.MockSession.init(std.testing.allocator);
    mock.peak_ram_bytes = 1_000_000_000;
    const text = try runText(
        &mock,
        "{\"id\":4,\"kind\":\"correctness_begin\",\"prompt_tokens\":[2]}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 1);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{ "id", "nonce", "ok", "token", "top_logits", "expert_stats", "peak_ram_gb" });
    try expectCommon(object, 4, true);
    try std.testing.expectEqual(@as(usize, 8), object.get("top_logits").?.array.items.len);
    const first = object.get("top_logits").?.array.items[0].object;
    try expectKeys(first, &.{ "token", "logit" });
    try std.testing.expectEqual(@as(i64, 69), first.get("token").?.integer);
    try std.testing.expectEqual(@as(f64, 8), number(first.get("logit").?));
    try std.testing.expectEqual(@as(i64, 0), object.get("expert_stats").?.object.get("expert_cache_hits").?.integer);
}

test "correctness_step returns eight raw logits" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"correctness_begin\",\"prompt_tokens\":[2]}\n" ++
            "{\"id\":2,\"kind\":\"correctness_step\",\"token\":69}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 2);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{ "id", "nonce", "ok", "token", "top_logits", "expert_stats" });
    try expectCommon(object, 2, true);
    try std.testing.expectEqual(@as(usize, 8), object.get("top_logits").?.array.items.len);
    try std.testing.expectEqual(@as(i64, 2146), object.get("token").?.integer);
}

test "phase_diagnostics closes the phase and reports the drain" {
    var mock = bench.MockSession.init(std.testing.allocator);
    mock.peak_ram_bytes = 3_000_000_000;
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n" ++
            "{\"id\":2,\"kind\":\"phase_diagnostics\"}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 2);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{
        "id", "nonce", "ok", "expert_stats", "peak_ram_gb", "completed_work", "cache_memory",
    });
    try expectCommon(object, 2, true);
    try std.testing.expectEqual(@as(i64, 1), object.get("completed_work").?.integer);
    try std.testing.expectEqual(@as(i64, 0), object.get("cache_memory").?.integer);
}

test "free_decode_begin echoes the resolved DFlash spec" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":5,\"kind\":\"free_decode_begin\",\"seed_tokens\":[4]," ++
            "\"spec\":{\"mode\":\"dflash\",\"dflash\":{\"depth\":1}}}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 1);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{ "id", "nonce", "ok", "seed_token", "effective_spec" });
    try expectCommon(object, 5, true);
    try std.testing.expectEqual(@as(i64, 131), object.get("seed_token").?.integer);
    const effective = object.get("effective_spec").?.object;
    try std.testing.expectEqualStrings("dflash", effective.get("mode").?.string);
    try std.testing.expectEqual(@as(i64, 1), effective.get("dflash").?.object.get("depth").?.integer);
}

test "free_decode_run returns tokens and DFlash audit fields" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[4]," ++
            "\"spec\":{\"mode\":\"dflash\",\"dflash\":{\"depth\":2}}}\n" ++
            "{\"id\":2,\"kind\":\"free_decode_run\",\"count\":7}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 2);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{
        "id",              "nonce",        "ok",                "tokens", "acceptance_lengths", "drafted_total", "accepted_total",
        "committed_total", "spec_decoder", "verification_mode",
    });
    try expectCommon(object, 2, true);
    try arrayIntegers(object.get("acceptance_lengths").?, &.{ 3, 3, 1 });
    try std.testing.expectEqual(@as(i64, 7), object.get("committed_total").?.integer);
    try std.testing.expectEqualStrings("dflash", object.get("spec_decoder").?.string);
    try std.testing.expectEqualStrings("rectangular", object.get("verification_mode").?.string);
    try std.testing.expect(object.get("verify_replay_disagreements") == null);
}

test "unknown request field is refused by the closed wire" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":8,\"kind\":\"prefill\",\"prompt_tokens\":[2],\"extra\":true}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 1);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectKeys(object, &.{ "id", "nonce", "ok", "error" });
    try expectCommon(object, -1, false);
}

test "nonce stays stable for every response in one run" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n" ++
            "{\"id\":2,\"kind\":\"phase_diagnostics\"}\n",
    );
    defer std.testing.allocator.free(text);
    var hello = try lineAt(text, 0);
    defer hello.deinit();
    var first = try lineAt(text, 1);
    defer first.deinit();
    var second = try lineAt(text, 2);
    defer second.deinit();
    const nonce = hello.value.object.get("nonce").?.string;
    try std.testing.expectEqualStrings(nonce, first.value.object.get("nonce").?.string);
    try std.testing.expectEqualStrings(nonce, second.value.object.get("nonce").?.string);
}

test "an error resets state before the next request" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"decode_begin\",\"seed_tokens\":[2]}\n" ++
            "{\"id\":2,\"kind\":\"correctness_step\",\"token\":69}\n" ++
            "{\"id\":3,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n" ++
            "{\"id\":4,\"kind\":\"phase_diagnostics\"}\n",
    );
    defer std.testing.allocator.free(text);
    var failure = try lineAt(text, 2);
    defer failure.deinit();
    try expectCommon(failure.value.object, 2, false);
    var diag = try lineAt(text, 4);
    defer diag.deinit();
    try std.testing.expectEqual(@as(i64, 1), diag.value.object.get("completed_work").?.integer);
}

test "free-run acceptance lengths sum to count" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[4]," ++
            "\"spec\":{\"mode\":\"dflash\",\"dflash\":{\"depth\":2}}}\n" ++
            "{\"id\":2,\"kind\":\"free_decode_run\",\"count\":7}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 2);
    defer parsed.deinit();
    var sum: i64 = 0;
    for (parsed.value.object.get("acceptance_lengths").?.array.items) |value| sum += value.integer;
    try std.testing.expectEqual(@as(i64, 7), sum);
    try std.testing.expectEqual(@as(usize, 7), parsed.value.object.get("tokens").?.array.items.len);
}

test "full prefill and decode transcript has four completed forwards" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n" ++
            "{\"id\":2,\"kind\":\"decode_begin\",\"seed_tokens\":[2]}\n" ++
            "{\"id\":3,\"kind\":\"decode_step\",\"token\":69}\n" ++
            "{\"id\":4,\"kind\":\"decode_step\",\"token\":2146}\n" ++
            "{\"id\":5,\"kind\":\"decode_step\",\"token\":66533}\n" ++
            "{\"id\":6,\"kind\":\"phase_diagnostics\"}\n",
    );
    defer std.testing.allocator.free(text);
    var diag = try lineAt(text, 6);
    defer diag.deinit();
    try expectCommon(diag.value.object, 6, true);
    try std.testing.expectEqual(@as(i64, 4), diag.value.object.get("completed_work").?.integer);
    try std.testing.expectEqual(@as(i64, 0), diag.value.object.get("cache_memory").?.integer);
}

const CountingWriter = struct {
    interface: std.Io.Writer,
    buffer: [4096]u8,
    flush_count: usize,

    fn init(out: *CountingWriter) void {
        out.interface = .{ .vtable = &vtable, .buffer = &out.buffer };
        out.flush_count = 0;
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        _ = data;
        _ = splat;
        w.end = 0;
        return 0;
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const out: *CountingWriter = @alignCast(@fieldParentPtr("interface", w));
        out.flush_count += 1;
        w.end = 0;
    }

    const vtable: std.Io.Writer.VTable = .{ .drain = drain, .flush = flush };
};

const DiagnosticSession = struct {
    diag: bench.PhaseDiag,

    fn session(value: *DiagnosticSession) bench.BenchSession {
        return .{ .ptr = value, .vtable = &vtable };
    }

    fn from(ptr: *anyopaque) *DiagnosticSession {
        return @ptrCast(@alignCast(ptr));
    }

    fn prefill(_: *anyopaque, _: []const u32) anyerror!u32 {
        return 1;
    }

    fn generate(_: *anyopaque, _: []const u32, _: u32, _: []u32) anyerror!void {
        return error.Unused;
    }

    fn seed(_: *anyopaque, _: []const u32, _: bool) anyerror!bench.Step {
        return error.Unused;
    }

    fn force(_: *anyopaque, _: u32, _: bool) anyerror!bench.Step {
        return error.Unused;
    }

    fn seedFree(_: *anyopaque, _: []const u32, _: bench.FreeSpec) anyerror!u32 {
        return error.Unused;
    }

    fn runFree(_: *anyopaque, _: u32, _: []u32, _: *bench.DflashAudit) anyerror!void {
        return error.Unused;
    }

    fn closePhase(ptr: *anyopaque) anyerror!bench.PhaseDiag {
        return from(ptr).diag;
    }

    fn inspectPhase(ptr: *anyopaque) anyerror!bench.PhaseDiag {
        return from(ptr).diag;
    }

    fn reset(_: *anyopaque) void {}

    const vtable: bench.BenchSession.VTable = .{
        .prefill = prefill,
        .generate = generate,
        .seed = seed,
        .force = force,
        .seedFree = seedFree,
        .runFree = runFree,
        .closePhase = closePhase,
        .inspectPhase = inspectPhase,
        .reset = reset,
    };
};

fn runDiagnostic(diag: bench.PhaseDiag) ![]u8 {
    var session = DiagnosticSession{ .diag = diag };
    var reader = std.Io.Reader.fixed(
        "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n" ++
            "{\"id\":2,\"kind\":\"phase_diagnostics\"}\n",
    );
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer output.deinit();
    var failures: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer failures.deinit();
    try worker.runReporting(std.testing.allocator, session.session(), identity(), &reader, &output.writer, &failures.writer);
    return output.toOwnedSlice();
}

test "flushes hello and every request response" {
    var mock = bench.MockSession.init(std.testing.allocator);
    var reader = std.Io.Reader.fixed("{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n");
    var output: CountingWriter = undefined;
    output.init();
    try worker.run(std.testing.allocator, mock.session(), identity(), &reader, &output.interface);
    try std.testing.expectEqual(@as(usize, 2), output.flush_count);
}

test "reads across boundaries and rejects a line over 16 MiB" {
    const request = "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n";
    var reader_buffer: [8]u8 = undefined;
    var source: std.testing.Reader = .init(&reader_buffer, &.{.{ .buffer = request }});
    source.artificial_limit = .limited(3);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var boundary_mock = bench.MockSession.init(std.testing.allocator);
    try worker.run(std.testing.allocator, boundary_mock.session(), identity(), &source.interface, &output.writer);
    var boundary_response = try lineAt(output.written(), 1);
    defer boundary_response.deinit();
    try expectCommon(boundary_response.value.object, 1, true);

    const valid = "{\"id\":2,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n";
    const long_len = worker.max_request_line_bytes + 1;
    const input = try std.testing.allocator.alloc(u8, long_len + 1 + valid.len);
    defer std.testing.allocator.free(input);
    @memset(input[0..long_len], 'x');
    input[long_len] = '\n';
    @memcpy(input[long_len + 1 ..], valid);
    var limit_mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(&limit_mock, input);
    defer std.testing.allocator.free(text);
    var too_long = try lineAt(text, 1);
    defer too_long.deinit();
    try expectCommon(too_long.value.object, -1, false);
    var recovered = try lineAt(text, 2);
    defer recovered.deinit();
    try expectCommon(recovered.value.object, 2, true);
}

test "rejects numeric strings and byte-array strings" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":\"1\",\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n" ++
            "{\"id\":2,\"kind\":[112,114,101,102,105,108,108],\"prompt_tokens\":[2]}\n" ++
            "{\"id\":3,\"kind\":\"prefill\",\"prompt_tokens\":[\"2\"]}\n" ++
            "{\"id\":4,\"kind\":\"free_decode_begin\",\"seed_tokens\":[2]," ++
            "\"spec\":{\"mode\":[100,102,108,97,115,104],\"dflash\":{\"depth\":1}}}\n",
    );
    defer std.testing.allocator.free(text);
    for (1..5) |index| {
        var parsed = try lineAt(text, index);
        defer parsed.deinit();
        try expectCommon(parsed.value.object, -1, false);
    }
}

test "rejects unknown DFlash block keys" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[2]," ++
            "\"spec\":{\"mode\":\"dflash\",\"dflash\":{\"depth\":1,\"extra\":true}}}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 1);
    defer parsed.deinit();
    try expectCommon(parsed.value.object, 1, false);
}

test "defaults omitted DFlash depth to one" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[2]," ++
            "\"spec\":{\"mode\":\"dflash\",\"dflash\":{}}}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 1);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectCommon(object, 1, true);
    try std.testing.expectEqual(@as(i64, 1), object.get("effective_spec").?.object.get("dflash").?.object.get("depth").?.integer);
}

test "reports serial audit identity for serial free decode" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(
        &mock,
        "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[2],\"spec\":{\"mode\":\"serial\"}}\n" ++
            "{\"id\":2,\"kind\":\"free_decode_run\",\"count\":2}\n",
    );
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 2);
    defer parsed.deinit();
    const object = parsed.value.object;
    try expectCommon(object, 2, true);
    try std.testing.expectEqualStrings("serial", object.get("spec_decoder").?.string);
    try std.testing.expectEqualStrings("serial", object.get("verification_mode").?.string);
}

test "rejects blank and unterminated request lines" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runText(&mock, "\n{\"id\":2,\"kind\":\"prefill\",\"prompt_tokens\":[2]}");
    defer std.testing.allocator.free(text);
    var blank = try lineAt(text, 1);
    defer blank.deinit();
    try expectCommon(blank.value.object, -1, false);
    var unterminated = try lineAt(text, 2);
    defer unterminated.deinit();
    try expectCommon(unterminated.value.object, -1, false);
}

test "advertises DFlash only with free-run capability" {
    var mock = bench.MockSession.init(std.testing.allocator);
    const text = try runTextWithIdentity(&mock, .{ .backend = "mock", .device = "host" }, "");
    defer std.testing.allocator.free(text);
    var parsed = try lineAt(text, 0);
    defer parsed.deinit();
    const modes = parsed.value.object.get("spec_modes").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), modes.len);
    try std.testing.expectEqualStrings("serial", modes[0].string);
}

test "rejects diagnostic values outside the Rust i64 range" {
    const too_large = @as(u64, std.math.maxInt(i64)) + 1;
    const work_text = try runDiagnostic(.{ .work_tokens = too_large, .peak_ram_bytes = null, .cache_bytes = 0 });
    defer std.testing.allocator.free(work_text);
    var work = try lineAt(work_text, 2);
    defer work.deinit();
    try expectKeys(work.value.object, &.{ "id", "nonce", "ok", "error" });
    try expectCommon(work.value.object, 2, false);

    const cache_text = try runDiagnostic(.{ .work_tokens = 0, .peak_ram_bytes = null, .cache_bytes = too_large });
    defer std.testing.allocator.free(cache_text);
    var cache = try lineAt(cache_text, 2);
    defer cache.deinit();
    try expectKeys(cache.value.object, &.{ "id", "nonce", "ok", "error" });
    try expectCommon(cache.value.object, 2, false);

    try std.testing.expectEqual(std.math.maxInt(i64), try protocol.checkedDiagnostic(std.math.maxInt(i64)));
}

test {
    _ = @import("worker_dflash_test.zig");
}
