const std = @import("std");
const bench = @import("bench_session");
const worker = @import("worker.zig");

const caps = [_][]const u8{"free_run_decode"};

fn identity() worker.Identity {
    return .{ .backend = "mock", .device = "host", .capabilities = &caps };
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

const DflashOnlySession = struct {
    fail: bool = false,
    saw_request: bool = false,

    fn session(value: *DflashOnlySession) bench.BenchSession {
        return .{ .ptr = value, .vtable = &vtable };
    }

    fn from(ptr: *anyopaque) *DflashOnlySession {
        return @ptrCast(@alignCast(ptr));
    }

    fn prefill(_: *anyopaque, _: []const u32) anyerror!u32 {
        return error.GpuPathRequired;
    }

    fn generate(_: *anyopaque, _: []const u32, _: u32, _: []u32) anyerror!void {
        return error.GpuPathRequired;
    }

    fn seed(_: *anyopaque, _: []const u32, _: bool) anyerror!bench.Step {
        return error.GpuPathRequired;
    }

    fn force(_: *anyopaque, _: u32, _: bool) anyerror!bench.Step {
        return error.GpuPathRequired;
    }

    fn seedFree(ptr: *anyopaque, tokens: []const u32, spec: bench.FreeSpec) anyerror!u32 {
        const fake = from(ptr);
        if (fake.fail) return error.ForcedDflashFailure;
        if (!std.mem.eql(u32, tokens, &.{ 11, 12, 13 })) return error.WrongSeedTokens;
        switch (spec) {
            .serial => return error.WrongSpec,
            .dflash => |d| if (d.depth != 1) return error.WrongDepth,
        }
        fake.saw_request = true;
        return 77;
    }

    fn runFree(_: *anyopaque, _: u32, _: []u32, _: *bench.DflashAudit) anyerror!void {
        return error.GpuPathRequired;
    }

    fn diag(_: *anyopaque) anyerror!bench.PhaseDiag {
        return .{ .work_tokens = 0, .peak_ram_bytes = null, .cache_bytes = 0 };
    }

    fn reset(_: *anyopaque) void {}

    const vtable: bench.BenchSession.VTable = .{
        .prefill = prefill,
        .generate = generate,
        .seed = seed,
        .force = force,
        .seedFree = seedFree,
        .runFree = runFree,
        .closePhase = diag,
        .inspectPhase = diag,
        .reset = reset,
    };
};

test "recorder DFlash depth one request reaches the DFlash engine path" {
    var fake: DflashOnlySession = .{};
    var reader = std.Io.Reader.fixed(
        "{\"id\":0,\"kind\":\"free_decode_begin\",\"seed_tokens\":[11,12,13]," ++
            "\"spec\":{\"mode\":\"dflash\",\"dflash\":{\"depth\":1}}}\n",
    );
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try worker.run(std.testing.allocator, fake.session(), identity(), &reader, &output.writer);
    try std.testing.expect(fake.saw_request);
    var response = try lineAt(output.written(), 1);
    defer response.deinit();
    try std.testing.expectEqual(@as(i64, 0), response.value.object.get("id").?.integer);
    try std.testing.expect(response.value.object.get("ok").?.bool);
    try std.testing.expectEqual(@as(i64, 77), response.value.object.get("seed_token").?.integer);
}

test "engine failure writes one stderr line" {
    var fake: DflashOnlySession = .{ .fail = true };
    var reader = std.Io.Reader.fixed(
        "{\"id\":9,\"kind\":\"free_decode_begin\",\"seed_tokens\":[11,12,13]," ++
            "\"spec\":{\"mode\":\"dflash\",\"dflash\":{\"depth\":1}}}\n",
    );
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var failures: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer failures.deinit();
    try worker.runReporting(std.testing.allocator, fake.session(), identity(), &reader, &output.writer, &failures.writer);
    try std.testing.expectEqualStrings(
        "bench-worker: free_decode_begin failed: ForcedDflashFailure\n",
        failures.written(),
    );
    var response = try lineAt(output.written(), 1);
    defer response.deinit();
    try std.testing.expectEqual(@as(i64, 9), response.value.object.get("id").?.integer);
    try std.testing.expect(!response.value.object.get("ok").?.bool);
}
