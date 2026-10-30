//! TensorFold Engine Protocol v1 resident and stdio worker.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const bench = @import("bench_session");
const adapter_mod = @import("bench_session_engine");
const engines = @import("server_engines");
const worker = @import("worker.zig");
const resident = @import("resident.zig");
const resident_client = @import("resident_client.zig");

pub const serves = engines.serves;
pub const std_options = engines.std_options;

pub const manifest_json =
    "{\"schema_version\":1,\"runner_id\":\"tensorfold/qwen3.8-27b\",\"model_type\":\"qwen3_5\"," ++
    "\"backends\":[\"metal\",\"cuda\"],\"spec_modes\":[\"serial\",\"dflash\"],\"dflash_depths\":[1]}";
const capabilities = [_][]const u8{"free_run_decode"};
const self_test_socket_path_limit = 104;
const usage =
    "usage: bench-worker resident --weights <dir> [--drafter <dir>] --socket <path>\n" ++
    "       bench-worker runtime-worker --weights <dir> [extra args]\n" ++
    "       bench-worker self-test\n" ++
    "       bench-worker manifest --digest\n";

const ResidentArgs = struct { weights: []const u8, drafter: ?[]const u8, socket: []const u8 };

pub fn main(init: std.process.Init) !u8 {
    const alloc = init.arena.allocator();
    const raw = try init.minimal.args.toSlice(alloc);
    const argv = try alloc.alloc([]const u8, raw.len);
    for (argv, raw) |*to, from| to.* = from;
    if (argv.len == 2 and std.mem.eql(u8, argv[1], "self-test")) return selfTest(init);
    if (argv.len == 3 and std.mem.eql(u8, argv[1], "manifest") and std.mem.eql(u8, argv[2], "--digest")) {
        const digest = manifestDigest();
        try std.Io.File.stdout().writeStreamingAll(init.io, &digest);
        try std.Io.File.stdout().writeStreamingAll(init.io, "\n");
        return 0;
    }
    if (argv.len >= 2 and std.mem.eql(u8, argv[1], "runtime-worker")) {
        const weights = parseRuntime(argv[2..]) orelse return usageError(init.io);
        return runtimeWorker(init, weights);
    }
    if (argv.len >= 2 and std.mem.eql(u8, argv[1], "resident")) {
        const args = parseResident(argv[2..]) orelse return usageError(init.io);
        return residentMain(init, args);
    }
    return usageError(init.io);
}

fn usageError(io: std.Io) u8 {
    std.Io.File.stderr().writeStreamingAll(io, usage) catch {};
    return 2;
}

fn parseRuntime(args: []const []const u8) ?[]const u8 {
    var weights: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--weights")) {
            i += 1;
            if (i == args.len or weights != null) return null;
            weights = args[i];
        }
    }
    return weights;
}

fn parseResident(args: []const []const u8) ?ResidentArgs {
    var weights: ?[]const u8 = null;
    var drafter: ?[]const u8 = null;
    var socket: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const target = if (std.mem.eql(u8, args[i], "--weights")) &weights else if (std.mem.eql(u8, args[i], "--drafter")) &drafter else if (std.mem.eql(u8, args[i], "--socket")) &socket else return null;
        i += 1;
        if (i == args.len or target.* != null) return null;
        target.* = args[i];
    }
    return .{ .weights = weights orelse return null, .drafter = drafter, .socket = socket orelse return null };
}

fn runtimeWorker(init: std.process.Init, weights: []const u8) !u8 {
    const socket = init.environ_map.get("BENCH_WORKER_RESIDENT_SOCKET") orelse {
        std.debug.print("bench-worker: BENCH_WORKER_RESIDENT_SOCKET is not set\n", .{});
        return 1;
    };
    var client = resident_client.Client.connect(init.gpa, init.io, socket, weights) catch |err| {
        std.debug.print("bench-worker: resident connection failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer client.deinit();
    var in_buffer: [16 << 10]u8 = undefined;
    var out_buffer: [16 << 10]u8 = undefined;
    var err_buffer: [1024]u8 = undefined;
    var input = std.Io.File.stdin().readerStreaming(init.io, &in_buffer);
    var output = std.Io.File.stdout().writerStreaming(init.io, &out_buffer);
    var failure_output = std.Io.File.stderr().writerStreaming(init.io, &err_buffer);
    defer output.interface.flush() catch {};
    defer failure_output.interface.flush() catch {};
    worker.runReporting(init.gpa, client.session(), identityOf(&client), &input.interface, &output.interface, &failure_output.interface) catch |err| {
        std.debug.print("bench-worker: protocol worker failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

fn identityOf(client_: *resident_client.Client) worker.Identity {
    return .fromResident(
        &client_.hello,
        &capabilities,
        .{ .ctx = client_, .reason = residentFailureReason },
    );
}

fn residentFailureReason(ptr: *anyopaque, err: anyerror) []const u8 {
    const client_: *resident_client.Client = @ptrCast(@alignCast(ptr));
    return client_.failureReason(err);
}

fn residentMain(init: std.process.Init, args: ResidentArgs) !u8 {
    const a = init.arena.allocator();
    var problem: []const u8 = "";
    const model_type = try modelType(a, init.io, args.weights, &problem);
    const request = adapter_mod.residentOpenRequest(args.weights, model_type, args.drafter);
    const opened = (engines.open(a, init.gpa, init.io, request.dir, request.model_type, .{
        .model = request.dir,
        .drafter = request.drafter,
        .parallel = "1",
        .prompt_cache_gib = request.prompt_cache_gib,
    }, &problem) catch |err| {
        std.debug.print("bench-worker: engine open failed: {s}\n", .{@errorName(err)});
        return 1;
    }) orelse {
        std.debug.print("bench-worker: engine open refused: {s}\n", .{problem});
        return 1;
    };
    defer opened.close(opened.ctx);

    var binding = adapter_mod.EngineDflashBinding.init(init.gpa, opened.engine);
    defer binding.deinit();
    const dflash = platformDflash(opened.engine, args.drafter != null, &binding);
    // Match server.Config.keep_warm_s / the serve CLI default, independent of requests.
    const keep_warm_s: i64 = 900;
    const keepalive: ?*api.keepalive.Keepalive = if (opened.engine.keepaliveTarget()) |target|
        api.keepalive.Keepalive.start(init.gpa, init.io, target, keep_warm_s * std.time.ns_per_s) catch |err| blk: {
            std.debug.print("bench-worker: idle keepalive off: {s}\n", .{@errorName(err)});
            break :blk null;
        }
    else
        null;
    // Join before binding cleanup and engine close: both can release resident buffers.
    defer if (keepalive) |k| k.stop();
    var adapter = adapter_mod.Adapter.init(init.gpa, opened.engine, dflash);
    const built = engines.identity(a, .{ .model = args.weights, .drafter = args.drafter, .parallel = "1", .prompt_cache_gib = 0 });
    const digest = manifestDigest();
    const loaded_at = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const epoch: u64 = @intCast(@max(0, loaded_at));
    resident.serve(init.gpa, adapter.session(), .{
        .io = init.io,
        .socket_path = args.socket,
        .activity = if (keepalive) |k| .{ .ctx = k, .begin = keepaliveBegin, .end = keepaliveEnd } else null,
        .weights = args.weights,
        .identity = .{
            .backend = built.backend,
            .device = built.device,
            .manifest_digest = &digest,
            .pid = @intCast(std.c.getpid()),
            .load_epoch = epoch,
            .weights_revision = weightsRevision(args.weights),
        },
    }) catch |err| {
        std.debug.print("bench-worker: resident failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

fn keepaliveBegin(ctx: *anyopaque) void {
    const k: *api.keepalive.Keepalive = @ptrCast(@alignCast(ctx));
    k.begin();
}

fn keepaliveEnd(ctx: *anyopaque) void {
    const k: *api.keepalive.Keepalive = @ptrCast(@alignCast(ctx));
    k.end();
}

fn modelType(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch |err| {
        problem.* = "could not read checkpoint config.json";
        return err;
    };
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch {
        problem.* = "checkpoint config.json is not valid JSON";
        return error.BadModelConfig;
    };
    if (doc != .object) return error.BadModelConfig;
    if (doc.object.get("model_type")) |value| if (value == .string) return value.string;
    if (doc.object.get("text_config")) |text| if (text == .object) if (text.object.get("model_type")) |value| if (value == .string) return value.string;
    return error.BadModelConfig;
}

fn weightsRevision(path: []const u8) []const u8 {
    return std.fs.path.basename(std.mem.trimEnd(u8, path, std.fs.path.sep_str));
}

pub fn manifestDigest() [64]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(manifest_json, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

fn platformDflash(engine: api.Engine, enabled: bool, binding: *adapter_mod.EngineDflashBinding) ?adapter_mod.DflashSource {
    return adapter_mod.platformDflash(builtin.os.tag, engine, enabled, binding);
}

const MockServer = struct {
    alloc: std.mem.Allocator,
    mock: *bench.MockSession,
    config: resident.Config,
    failure: ?anyerror = null,
    stopped: std.atomic.Value(bool) = .init(false),

    fn run(s: *MockServer) void {
        defer s.stopped.store(true, .release);
        resident.serve(s.alloc, s.mock.session(), s.config) catch |err| {
            s.failure = err;
        };
    }
};

const SelfTestPaths = struct {
    alloc: std.mem.Allocator,
    dir: []u8,
    socket: []u8,
    checkpoint: []u8,
    dir_created: bool = false,

    pub fn deinit(paths: *SelfTestPaths) void {
        paths.alloc.free(paths.checkpoint);
        paths.alloc.free(paths.socket);
        paths.alloc.free(paths.dir);
    }
};

fn validateSelfTestDir(stat: std.Io.File.Stat) !void {
    if (stat.kind == .sym_link) return error.SelfTestDirSymlink;
    if (stat.kind != .directory) return error.SelfTestDirNotDirectory;
}

fn prepareSelfTestPaths(
    alloc: std.mem.Allocator,
    io: std.Io,
    requested_dir: ?[]const u8,
    tmp_root_env: ?[]const u8,
    pid: c_int,
    stamp: i128,
) !SelfTestPaths {
    const requested = if (requested_dir) |value| if (value.len != 0) value else null else null;
    const dir = if (requested) |value|
        try alloc.dupe(u8, value)
    else blk: {
        const raw_root = if (tmp_root_env) |value| if (value.len == 0) "/tmp" else value else "/tmp";
        const trimmed = std.mem.trimEnd(u8, raw_root, "/");
        const root = if (trimmed.len == 0) "/" else trimmed;
        const name = try std.fmt.allocPrint(alloc, "tfbw-{d}-{d}", .{ pid, stamp });
        defer alloc.free(name);
        break :blk try std.fs.path.join(alloc, &.{ root, name });
    };
    errdefer alloc.free(dir);
    const socket = try std.fs.path.join(alloc, &.{ dir, "resident.sock" });
    errdefer alloc.free(socket);
    const checkpoint = try std.fs.path.join(alloc, &.{ dir, "revision" });
    errdefer alloc.free(checkpoint);
    if (socket.len >= self_test_socket_path_limit) return error.SocketPathTooLong;

    const cwd = std.Io.Dir.cwd();
    var status: std.Io.Dir.CreatePathStatus = .existed;
    var dir_created = false;
    errdefer if (dir_created) cwd.deleteDir(io, dir) catch {};
    if (cwd.statFile(io, dir, .{ .follow_symlinks = false })) |stat| {
        try validateSelfTestDir(stat);
    } else |err| switch (err) {
        error.FileNotFound => {
            status = try cwd.createDirPathStatus(io, dir, .fromMode(0o700));
            dir_created = status == .created;
            try validateSelfTestDir(try cwd.statFile(io, dir, .{ .follow_symlinks = false }));
        },
        else => return err,
    }
    if (status == .created) {
        var opened = try cwd.openDir(io, dir, .{ .iterate = true, .follow_symlinks = false });
        defer opened.close(io);
        try opened.setPermissions(io, .fromMode(0o700));
    }
    if (cwd.statFile(io, socket, .{ .follow_symlinks = false })) |_| {
        return error.SelfTestSocketExists;
    } else |err| if (err != error.FileNotFound) return err;
    return .{ .alloc = alloc, .dir = dir, .socket = socket, .checkpoint = checkpoint, .dir_created = dir_created };
}

fn printUnexpectedSelfTestPathError(
    err: anyerror,
    requested_dir: ?[]const u8,
    tmp_root_env: ?[]const u8,
    pid: c_int,
    stamp: i128,
) void {
    const requested = if (requested_dir) |value| if (value.len != 0) value else null else null;
    if (requested) |dir| {
        std.debug.print("bench-worker: self-test failed with {s} for directory {s}\n", .{ @errorName(err), dir });
        return;
    }
    const raw_root = if (tmp_root_env) |value| if (value.len == 0) "/tmp" else value else "/tmp";
    const trimmed = std.mem.trimEnd(u8, raw_root, "/");
    const root = if (trimmed.len == 0) "/" else trimmed;
    const separator = if (std.mem.eql(u8, root, "/")) "" else "/";
    std.debug.print(
        "bench-worker: self-test failed with {s} for directory {s}{s}tfbw-{d}-{d}\n",
        .{ @errorName(err), root, separator, pid, stamp },
    );
}

fn selfTest(init: std.process.Init) !u8 {
    const a = init.gpa;
    const digest = manifestDigest();
    const requested_dir = init.environ_map.get("BENCH_WORKER_SELF_TEST_DIR");
    const tmp_root_env = init.environ_map.get("TMPDIR");
    const pid = std.c.getpid();
    const stamp = std.Io.Clock.awake.now(init.io).toNanoseconds();
    var paths = prepareSelfTestPaths(
        a,
        init.io,
        requested_dir,
        tmp_root_env,
        pid,
        stamp,
    ) catch |err| {
        switch (err) {
            error.SelfTestDirSymlink => std.debug.print("bench-worker: self-test directory is a symlink\n", .{}),
            error.SelfTestDirNotDirectory => std.debug.print("bench-worker: self-test path is not a directory\n", .{}),
            error.SelfTestSocketExists => std.debug.print("bench-worker: self-test socket path already exists\n", .{}),
            error.SocketPathTooLong => std.debug.print("bench-worker: self-test socket path must be shorter than 104 bytes\n", .{}),
            else => printUnexpectedSelfTestPathError(err, requested_dir, tmp_root_env, pid, stamp),
        }
        return 1;
    };
    defer paths.deinit();
    defer if (paths.dir_created) std.Io.Dir.cwd().deleteDir(init.io, paths.dir) catch {};
    try std.Io.Dir.cwd().createDir(init.io, paths.checkpoint, .default_dir);
    defer std.Io.Dir.cwd().deleteDir(init.io, paths.checkpoint) catch {};
    defer std.Io.Dir.cwd().deleteFile(init.io, paths.socket) catch {};
    var mock = bench.MockSession.init(a);
    var ready = std.atomic.Value(bool).init(false);
    const config: resident.Config = .{
        .io = init.io,
        .socket_path = paths.socket,
        .weights = paths.checkpoint,
        .identity = .{
            .backend = "mock",
            .device = "self-test",
            .manifest_digest = &digest,
            .pid = 42,
            .load_epoch = 77,
            .weights_revision = "revision",
        },
        .max_clients = 2,
        .ready = &ready,
    };
    var server: MockServer = .{ .alloc = a, .mock = &mock, .config = config };
    const server_thread = try std.Thread.spawn(.{}, MockServer.run, .{&server});
    while (!ready.load(.acquire) and !server.stopped.load(.acquire))
        try std.Io.sleep(init.io, .fromMilliseconds(1), .awake);
    if (!ready.load(.acquire)) {
        server_thread.join();
        return selfTestFailed();
    }
    const socket_stat = try std.Io.Dir.cwd().statFile(init.io, paths.socket, .{ .follow_symlinks = false });
    const socket_ok = socket_stat.kind == .unix_domain_socket and socket_stat.permissions.toMode() & 0o777 == 0o600;
    const transcript =
        "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[2]}\n" ++
        "{\"id\":2,\"kind\":\"phase_diagnostics\"}\n" ++
        "{\"id\":3,\"kind\":\"decode_begin\",\"seed_tokens\":[2]}\n" ++
        "{\"id\":4,\"kind\":\"decode_step\",\"token\":69}\n" ++
        "{\"id\":5,\"kind\":\"phase_diagnostics\"}\n" ++
        "{\"id\":6,\"kind\":\"correctness\",\"prompt_tokens\":[2],\"steps\":2}\n" ++
        "{\"id\":7,\"kind\":\"phase_diagnostics\"}\n" ++
        "{\"id\":8,\"kind\":\"correctness_begin\",\"prompt_tokens\":[2]}\n" ++
        "{\"id\":9,\"kind\":\"correctness_step\",\"token\":69}\n" ++
        "{\"id\":10,\"kind\":\"phase_diagnostics\"}\n" ++
        "{\"id\":11,\"kind\":\"free_decode_begin\",\"seed_tokens\":[4],\"spec\":{\"mode\":\"serial\"}}\n" ++
        "{\"id\":12,\"kind\":\"free_decode_run\",\"count\":3}\n" ++
        "{\"id\":13,\"kind\":\"phase_diagnostics\"}\n" ++
        "{\"id\":14,\"kind\":\"free_decode_begin\",\"seed_tokens\":[4],\"spec\":{\"mode\":\"dflash\",\"dflash\":{\"depth\":1}}}\n" ++
        "{\"id\":15,\"kind\":\"free_decode_run\",\"count\":3}\n" ++
        "{\"id\":16,\"kind\":\"phase_diagnostics\"}\n" ++
        "{\"id\":17,\"kind\":\"free_decode_begin\",\"seed_tokens\":[9],\"spec\":{\"mode\":\"serial\"}}\n";
    const first = try mockWorker(a, init.io, config, transcript);
    defer a.free(first.output);
    var second = try resident_client.Client.connect(a, init.io, paths.socket, paths.checkpoint);
    var joined = false;
    defer if (!joined) server_thread.join();
    var second_open = true;
    defer if (second_open) second.deinit();
    if (!socket_ok or std.mem.indexOf(u8, first.output, "\"ok\":false") != null or
        std.mem.indexOf(u8, first.output, "\"spec_decoder\":\"serial\"") == null or
        std.mem.indexOf(u8, first.output, "\"spec_decoder\":\"dflash\"") == null or
        !std.meta.eql(first.identity, second.hello.resident)) return selfTestFailed();
    var reset_out: [1]u32 = undefined;
    var reset_audit = bench.DflashAudit.init();
    defer reset_audit.deinit(a);
    if (second.session().runFree(1, &reset_out, &reset_audit)) |_| return selfTestFailed() else |_| {}
    second.deinit();
    second_open = false;
    server_thread.join();
    joined = true;
    if (server.failure != null) return selfTestFailed();
    try std.Io.File.stdout().writeStreamingAll(init.io, "engine-protocol-v1 self-test: ok\n");
    return 0;
}

const MockResult = struct { output: []u8, identity: @import("protocol.zig").ResidentIdentity };

fn mockWorker(alloc: std.mem.Allocator, io: std.Io, config: resident.Config, input_text: []const u8) !MockResult {
    var client_ = try resident_client.Client.connect(alloc, io, config.socket_path, config.weights);
    defer client_.deinit();
    const ident = client_.hello.resident;
    var input = std.Io.Reader.fixed(input_text);
    var output: std.Io.Writer.Allocating = .init(alloc);
    errdefer output.deinit();
    try worker.run(alloc, client_.session(), identityOf(&client_), &input, &output.writer);
    return .{ .output = try output.toOwnedSlice(), .identity = ident };
}

fn selfTestFailed() u8 {
    std.debug.print("bench-worker: self-test failed\n", .{});
    return 1;
}

test {
    _ = @import("self_test_test.zig").Cases(SelfTestPaths, prepareSelfTestPaths);
}
