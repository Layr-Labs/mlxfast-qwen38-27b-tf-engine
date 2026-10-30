const std = @import("std");
const bench = @import("bench_session");
const resident = @import("resident.zig");
const resident_client = @import("resident_client.zig");
const digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const ShortSocketDir = struct {
    buffer: [64]u8 = undefined,
    len: usize = 0,

    fn init() !ShortSocketDir {
        var result: ShortSocketDir = .{};
        for (0..8) |_| {
            var random: [6]u8 = undefined;
            std.testing.io.random(&random);
            var encoded: [std.base64.url_safe.Encoder.calcSize(random.len)]u8 = undefined;
            _ = std.base64.url_safe.Encoder.encode(&encoded, &random);
            const root = try std.fmt.bufPrint(&result.buffer, "/tmp/tfb-{s}", .{encoded});
            std.Io.Dir.createDirAbsolute(std.testing.io, root, @fromBackingInt(@intCast(0o700))) catch |err| {
                if (err == error.PathAlreadyExists) continue;
                return err;
            };
            result.len = root.len;
            return result;
        }
        return error.PathAlreadyExists;
    }

    fn cleanup(dir: *ShortSocketDir) void {
        std.Io.Dir.cwd().deleteTree(std.testing.io, dir.buffer[0..dir.len]) catch {};
    }

    fn path(dir: *const ShortSocketDir, name: []const u8) ![]u8 {
        return std.fmt.allocPrint(std.testing.allocator, "{s}/{s}", .{ dir.buffer[0..dir.len], name });
    }
};
const ActivityProbe = struct {
    begins: usize = 0,
    ends: usize = 0,
    active: bool = false,
    bad_order: bool = false,

    fn hooks(p: *ActivityProbe) resident.Activity {
        return .{ .ctx = p, .begin = begin, .end = end };
    }

    fn begin(ctx: *anyopaque) void {
        const p: *ActivityProbe = @ptrCast(@alignCast(ctx));
        p.bad_order = p.bad_order or p.active;
        p.active = true;
        p.begins += 1;
    }

    fn end(ctx: *anyopaque) void {
        const p: *ActivityProbe = @ptrCast(@alignCast(ctx));
        p.bad_order = p.bad_order or !p.active;
        p.active = false;
        p.ends += 1;
    }

    // Read only after the server thread joins.
    fn expectBalanced(p: *const ActivityProbe, count: usize) !void {
        try std.testing.expectEqual(count, p.begins);
        try std.testing.expectEqual(count, p.ends);
        try std.testing.expect(!p.active and !p.bad_order);
    }
};
const Server = struct {
    mock: *bench.MockSession,
    config: resident.Config,
    fd: std.posix.fd_t,
    failure: ?anyerror = null,
    fn run(s: *Server) void {
        defer _ = std.c.close(s.fd);
        resident.serveConnected(std.testing.allocator, s.fd, s.mock.session(), s.config) catch |err| {
            s.failure = err;
        };
    }
};
fn socketPair() ![2]std.posix.fd_t {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPairFailed;
    return .{ pair[0], pair[1] };
}
fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
const TricklePeer = struct {
    fd: std.posix.fd_t,
    failure: ?anyerror = null,
    fn run(peer: *TricklePeer) void {
        const header = [_]u8{ 8, 0, 0, 0 };
        for (header) |byte| {
            var delay: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
            while (std.c.nanosleep(&delay, &delay) != 0) {}
            const one = [1]u8{byte};
            const rc = std.posix.system.write(peer.fd, &one, one.len);
            if (std.posix.errno(rc) != .SUCCESS or rc != one.len) {
                peer.failure = error.SocketWriteFailed;
                return;
            }
        }
    }
};
test "frame deadline stops a trickle peer" {
    const pair = try socketPair();
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    var peer: TricklePeer = .{ .fd = pair[0] };
    const thread = try std.Thread.spawn(.{}, TricklePeer.run, .{&peer});
    const start = monotonicNs();
    try std.testing.expectError(error.Timeout, resident_client.receiveFrameWithTimeout(std.testing.allocator, pair[1], 300));
    const elapsed = monotonicNs() - start;
    thread.join();
    try std.testing.expect(peer.failure == null);
    try std.testing.expect(elapsed < std.time.ns_per_s);
}
const SlowSession = struct {
    delay_seconds: u64,
    fn session(s: *SlowSession) bench.BenchSession {
        return .{ .ptr = s, .vtable = &vtable };
    }
    fn from(ptr: *anyopaque) *SlowSession {
        return @ptrCast(@alignCast(ptr));
    }
    fn prefill(ptr: *anyopaque, _: []const u32) anyerror!u32 {
        var delay: std.c.timespec = .{ .sec = @intCast(from(ptr).delay_seconds), .nsec = 0 };
        while (std.c.nanosleep(&delay, &delay) != 0) {}
        return 69;
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
const SlowServer = struct {
    slow: *SlowSession,
    config: resident.Config,
    fd: std.posix.fd_t,
    failure: ?anyerror = null,
    fn run(s: *SlowServer) void {
        defer _ = std.c.close(s.fd);
        resident.serveConnected(std.testing.allocator, s.fd, s.slow.session(), s.config) catch |err| {
            s.failure = err;
        };
    }
};
test "inference may take longer than the frame timeout" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    const pair = try socketPair();
    var slow: SlowSession = .{ .delay_seconds = 40 };
    var server: SlowServer = .{ .slow = &slow, .fd = pair[0], .config = .{
        .io = std.testing.io,
        .socket_path = "unused",
        .weights = weights,
        .identity = .{ .backend = "mock", .device = "test", .manifest_digest = digest, .pid = 1, .load_epoch = 2, .weights_revision = "weights" },
    } };
    const thread = try std.Thread.spawn(.{}, SlowServer.run, .{&server});
    var connected = try resident_client.Client.attach(std.testing.allocator, std.testing.io, pair[1], weights);
    try std.testing.expectEqual(@as(u32, 69), try connected.session().prefill(&.{2}));
    connected.deinit();
    thread.join();
    try std.testing.expect(server.failure == null);
}
fn sleepMs(ms: u64) void {
    var delay: std.c.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * std.time.ns_per_ms) };
    while (std.c.nanosleep(&delay, &delay) != 0) {}
}
fn idleRun(idle_timeout_ms: u32) !bool {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    var mock = bench.MockSession.init(std.testing.allocator);
    const pair = try socketPair();
    var server: Server = .{ .mock = &mock, .fd = pair[0], .config = .{
        .io = std.testing.io,
        .socket_path = "unused",
        .weights = weights,
        .identity = .{ .backend = "mock", .device = "test", .manifest_digest = digest, .pid = 1, .load_epoch = 2, .weights_revision = "weights" },
        .timeout_ms = 200,
        .idle_timeout_ms = idle_timeout_ms,
    } };
    const thread = try std.Thread.spawn(.{}, Server.run, .{&server});
    var connected = try resident_client.Client.attach(std.testing.allocator, std.testing.io, pair[1], weights);
    try std.testing.expectEqual(@as(u32, 69), try connected.session().prefill(&.{2}));
    // benchd idles the attached connection during its cool gate (for example a 40 s gate with a 30 s frame timeout).
    sleepMs(700);
    const second_ok = if (connected.session().prefill(&.{2})) |token| token == 69 else |_| false;
    connected.deinit();
    thread.join();
    return second_ok;
}
test "attached client may idle longer than the frame timeout" {
    try std.testing.expect(try idleRun(5_000));
}
test "idle limit still closes a client that idles past it" {
    try std.testing.expect(!(try idleRun(300)));
}
test "connect handshake has one deadline" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    var sockets = try ShortSocketDir.init();
    defer sockets.cleanup();
    const path = try sockets.path("unaccepted.sock");
    defer std.testing.allocator.free(path);
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    var addr = try resident_client.unixAddress(path);
    const listener = std.c.socket(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0);
    if (listener < 0) return error.SkipZigTest;
    defer _ = std.c.close(listener);
    if (std.c.bind(listener, @ptrCast(&addr), @sizeOf(@TypeOf(addr))) != 0) return error.SkipZigTest;
    if (std.c.listen(listener, 1) != 0) return error.ListenFailed;
    const start = monotonicNs();
    try std.testing.expectError(
        error.Timeout,
        resident_client.Client.connectWithTimeout(std.testing.allocator, std.testing.io, path, weights, 300),
    );
    try std.testing.expect(monotonicNs() - start < std.time.ns_per_s);
}

test "transport accepts exact worker limits and refuses larger counts before allocation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    var mock = bench.MockSession.init(std.testing.allocator);
    const pair = try socketPair();
    var server: Server = .{
        .mock = &mock,
        .fd = pair[0],
        .config = .{
            .io = std.testing.io,
            .socket_path = "unused",
            .weights = weights,
            .identity = .{
                .backend = "mock",
                .device = "test",
                .manifest_digest = digest,
                .pid = 42,
                .load_epoch = 99,
                .weights_revision = "revision",
            },
        },
    };
    const thread = try std.Thread.spawn(.{}, Server.run, .{&server});
    var connected = try resident_client.Client.attach(std.testing.allocator, std.testing.io, pair[1], weights);
    errdefer {
        connected.deinit();
        thread.join();
    }
    const session = connected.session();

    const correctness = try std.testing.allocator.alloc(u32, resident_client.max_correctness_steps + 1);
    defer std.testing.allocator.free(correctness);
    try session.generate(&.{2}, resident_client.max_correctness_steps, correctness);
    _ = try session.seedFree(&.{2}, .serial);
    const free = try std.testing.allocator.alloc(u32, resident_client.max_free_count + 1);
    defer std.testing.allocator.free(free);
    var audit = bench.DflashAudit.init();
    defer audit.deinit(std.testing.allocator);
    try session.runFree(resident_client.max_free_count, free, &audit);

    const work_before_refusal = mock.work_tokens;
    connected.allocator = std.testing.failing_allocator;
    try std.testing.expectError(
        error.RequestTooLarge,
        session.generate(&.{2}, resident_client.max_correctness_steps + 1, correctness),
    );
    try std.testing.expectError(
        error.RequestTooLarge,
        session.runFree(resident_client.max_free_count + 1, free, &audit),
    );
    connected.allocator = std.testing.allocator;
    try std.testing.expectEqual(work_before_refusal, mock.work_tokens);
    connected.deinit();
    thread.join();
    try std.testing.expect(server.failure == null);
}

test "socket round trip, stable identity, errors, and same-uid check" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var sockets = try ShortSocketDir.init();
    defer sockets.cleanup();
    const path = try sockets.path("resident.sock");
    defer std.testing.allocator.free(path);
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    var mock = bench.MockSession.init(std.testing.allocator);
    var failures: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer failures.deinit();
    var activity: ActivityProbe = .{};
    const first_pair = try socketPair();
    var server: Server = .{
        .mock = &mock,
        .fd = first_pair[0],
        .config = .{
            .io = std.testing.io,
            .socket_path = path,
            .weights = weights,
            .identity = .{
                .backend = "mock",
                .device = "test",
                .manifest_digest = digest,
                .pid = 42,
                .load_epoch = 99,
                .weights_revision = "revision",
            },
            .failure_writer = &failures.writer,
            .activity = activity.hooks(),
        },
    };
    const first_thread = try std.Thread.spawn(.{}, Server.run, .{&server});

    var first = try resident_client.Client.attach(std.testing.allocator, std.testing.io, first_pair[1], weights);
    const first_identity = first.hello.resident;
    try std.testing.expectEqualStrings(digest, &first.hello.manifest_digest);
    const session = first.session();
    try std.testing.expectEqual(@as(u32, 69), try session.prefill(&.{2}));
    const no_top = try session.seed(&.{2}, false);
    try std.testing.expect(!no_top.has_top);
    const seeded = try session.seed(&.{2}, true);
    try std.testing.expect(seeded.has_top);
    try std.testing.expectEqual(@as(u32, 69), seeded.token);
    const before_bad_frame = mock.work_tokens;
    var bad: resident_client.Encoder = .{};
    defer bad.deinit(std.testing.allocator);
    try bad.byte(std.testing.allocator, @backingInt(resident_client.Op.prefill));
    try bad.int(std.testing.allocator, u32, std.math.maxInt(u32));
    try resident_client.sendFrame(first.fd, bad.bytes.items);
    const refused = (try resident_client.receiveFrame(std.testing.allocator, first.fd)).?;
    defer std.testing.allocator.free(refused);
    try std.testing.expectEqual(@as(u8, 1), refused[0]);
    try std.testing.expectEqual(before_bad_frame, mock.work_tokens);
    var bad_generate: resident_client.Encoder = .{};
    defer bad_generate.deinit(std.testing.allocator);
    try bad_generate.byte(std.testing.allocator, @backingInt(resident_client.Op.generate));
    try bad_generate.tokens(std.testing.allocator, &.{2});
    try bad_generate.int(std.testing.allocator, u32, std.math.maxInt(u32));
    try resident_client.sendFrame(first.fd, bad_generate.bytes.items);
    const generate_refused = (try resident_client.receiveFrame(std.testing.allocator, first.fd)).?;
    defer std.testing.allocator.free(generate_refused);
    try std.testing.expectEqual(@as(u8, 1), generate_refused[0]);
    try std.testing.expectEqual(before_bad_frame, mock.work_tokens);
    var failed_audit = bench.DflashAudit.init();
    defer failed_audit.deinit(std.testing.allocator);
    var failed_out: [1]u32 = undefined;
    try std.testing.expectError(error.ResidentRequestFailed, session.runFree(1, &failed_out, &failed_audit));
    try std.testing.expectEqualStrings("SessionNotSeeded", first.failureReason(error.ResidentRequestFailed));
    session.reset();
    first.deinit();
    first_thread.join();
    // Two connection resets plus seven dispatches (including an explicit reset), including malformed and engine-error requests.
    try activity.expectBalanced(9);
    if (server.failure) |err| return err;

    const second_pair = try socketPair();
    server.fd = second_pair[0];
    server.failure = null;
    const second_thread = try std.Thread.spawn(.{}, Server.run, .{&server});
    var second = try resident_client.Client.attach(std.testing.allocator, std.testing.io, second_pair[1], weights);
    try std.testing.expectEqual(first_identity, second.hello.resident);
    const dflash = try second.session().seedFree(&.{4}, .{ .dflash = .{ .depth = 1 } });
    try std.testing.expectEqual(@as(u32, 131), dflash);
    var out: [3]u32 = undefined;
    var audit = bench.DflashAudit.init();
    defer audit.deinit(std.testing.allocator);
    try second.session().runFree(out.len, &out, &audit);
    try audit.check(out.len);
    second.deinit();
    second_thread.join();
    // A reconnect adds two resets and two successful dispatches.
    try activity.expectBalanced(13);
    try std.testing.expectEqualStrings(
        "bench-worker resident: prefill failed: TooManyTokens\n" ++
            "bench-worker resident: generate failed: RequestTooLarge\n" ++
            "bench-worker resident: run_free failed: SessionNotSeeded\n",
        failures.written(),
    );

    try std.testing.expect(resident.uidAllowed(1000, 1000));
    try std.testing.expect(!resident.uidAllowed(1000, 1001));
    if (server.failure) |err| return err;
}

test "inner token count is checked before allocation" {
    var frame = [_]u8{ @backingInt(resident_client.Op.prefill), 0xff, 0xff, 0xff, 0xff };
    var decoder: resident_client.Decoder = .{ .bytes = frame[1..] };
    try std.testing.expectError(error.TooManyTokens, decoder.tokensAlloc(std.testing.failing_allocator));
}

const ListenServer = struct {
    mock: *bench.MockSession,
    config: resident.Config,
    failure: ?anyerror = null,
    stopped: std.atomic.Value(bool) = .init(false),

    fn run(s: *ListenServer) void {
        defer s.stopped.store(true, .release);
        resident.serve(std.testing.allocator, s.mock.session(), s.config) catch |err| {
            s.failure = err;
        };
    }
};

fn rawConnect(path: []const u8) !std.posix.fd_t {
    var addr = try resident_client.unixAddress(path);
    const rc = std.posix.system.socket(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0);
    if (std.posix.errno(rc) != .SUCCESS) return error.SocketOpenFailed;
    const fd: std.posix.fd_t = @intCast(rc);
    errdefer _ = std.c.close(fd);
    if (std.posix.errno(std.posix.system.connect(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr)))) != .SUCCESS)
        return error.ConnectFailed;
    return fd;
}

fn readHelloLine(alloc: std.mem.Allocator, fd: std.posix.fd_t, timeout_ms: u32) ![]u8 {
    try resident_client.setSocketTimeout(fd, timeout_ms);
    var line: std.ArrayListUnmanaged(u8) = .empty;
    errdefer line.deinit(alloc);
    while (line.items.len < resident_client.max_hello_bytes) {
        var byte: [1]u8 = undefined;
        const rc = std.posix.system.read(fd, &byte, byte.len);
        if (std.posix.errno(rc) != .SUCCESS or rc == 0) return error.NoResidentHello;
        if (byte[0] == '\n') return line.toOwnedSlice(alloc);
        try line.append(alloc, byte[0]);
    }
    return error.ResidentHelloTooLarge;
}

test "resident speaks first and serves after a probe disconnect" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    var sockets = try ShortSocketDir.init();
    defer sockets.cleanup();
    const path = try sockets.path("hello.sock");
    defer std.testing.allocator.free(path);
    var ready = std.atomic.Value(bool).init(false);
    var mock = bench.MockSession.init(std.testing.allocator);
    const pid: u32 = @intCast(std.c.getpid());
    var server: ListenServer = .{ .mock = &mock, .config = .{
        .io = std.testing.io,
        .socket_path = path,
        .weights = weights,
        .identity = .{ .backend = "mock", .device = "test", .manifest_digest = digest, .pid = pid, .load_epoch = 2, .weights_revision = "weights" },
        .max_clients = 2,
        .ready = &ready,
    } };
    const thread = try std.Thread.spawn(.{}, ListenServer.run, .{&server});
    while (!ready.load(.acquire) and !server.stopped.load(.acquire))
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    if (!ready.load(.acquire)) {
        thread.join();
        if (server.failure) |err| {
            if (err == error.BindFailed) return error.SkipZigTest;
            return err;
        }
        return error.ServerStopped;
    }

    const probe = try rawConnect(path);
    const hello_line = try readHelloLine(std.testing.allocator, probe, 1000);
    defer std.testing.allocator.free(hello_line);
    var hello = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, hello_line, .{});
    defer hello.deinit();
    try std.testing.expectEqual(@as(i64, 0), hello.value.object.get("id").?.integer);
    try std.testing.expect(hello.value.object.get("ok").?.bool);
    try std.testing.expectEqual(@as(i64, 1), hello.value.object.get("protocol_version").?.integer);
    try std.testing.expectEqual(@as(i64, pid), hello.value.object.get("resident").?.object.get("pid").?.integer);
    _ = std.c.close(probe);

    var connected = try resident_client.Client.connect(std.testing.allocator, std.testing.io, path, weights);
    try std.testing.expectEqual(@as(u32, 69), try connected.session().prefill(&.{2}));
    connected.deinit();
    thread.join();
    try std.testing.expect(server.failure == null);
}

test "probe_resident body accepts the unsolicited hello" {
    const attempts = 50;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    var sockets = try ShortSocketDir.init();
    defer sockets.cleanup();
    const path = try sockets.path("python-probe.sock");
    defer std.testing.allocator.free(path);
    var ready = std.atomic.Value(bool).init(false);
    var mock = bench.MockSession.init(std.testing.allocator);
    var server: ListenServer = .{ .mock = &mock, .config = .{
        .io = std.testing.io,
        .socket_path = path,
        .weights = weights,
        .identity = .{ .backend = "mock", .device = "test", .manifest_digest = digest, .pid = @intCast(std.c.getpid()), .load_epoch = 2, .weights_revision = "weights" },
        .max_clients = attempts,
        .ready = &ready,
    } };
    const thread = try std.Thread.spawn(.{}, ListenServer.run, .{&server});
    while (!ready.load(.acquire) and !server.stopped.load(.acquire))
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    if (!ready.load(.acquire)) {
        thread.join();
        if (server.failure) |err| {
            if (err == error.BindFailed) return error.SkipZigTest;
            return err;
        }
        return error.ServerStopped;
    }

    const probe_body =
        \\import json, socket, sys, time
        \\MAX_HELLO_BYTES = 1024 * 1024
        \\PROBE_DEADLINE_SECONDS = 30.0
        \\deadline = time.monotonic() + PROBE_DEADLINE_SECONDS
        \\with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        \\    sock.settimeout(deadline - time.monotonic())
        \\    sock.connect(sys.argv[1])
        \\    line = bytearray()
        \\    while not line.endswith(b"\\n"):
        \\        sock.settimeout(deadline - time.monotonic())
        \\        chunk = sock.recv(65536)
        \\        if not chunk:
        \\            break
        \\        if len(line) + len(chunk) > MAX_HELLO_BYTES:
        \\            raise RuntimeError("hello overflow")
        \\        line.extend(chunk)
        \\reply = json.loads(line.decode())
        \\if reply.get("ok") is not True:
        \\    sys.exit(1)
        \\print(json.dumps({k: reply.get(k) for k in ("backend", "device", "protocol_version", "spec_modes", "runner", "resident")}))
    ;
    var all_ok = true;
    for (0..attempts) |_| {
        var child = try std.process.spawn(std.testing.io, .{
            .argv = &.{ "/usr/bin/env", "python3", "-c", probe_body, path },
            .stdin = .ignore,
            .stdout = .ignore,
            .stderr = .inherit,
        });
        const term = try child.wait(std.testing.io);
        all_ok = all_ok and term == .exited and term.exited == 0;
    }
    var waited: u32 = 0;
    while (!server.stopped.load(.acquire) and waited < 1000) : (waited += 1)
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    while (!server.stopped.load(.acquire)) {
        const fallback = try rawConnect(path);
        _ = std.c.close(fallback);
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    thread.join();
    try std.testing.expect(all_ok);
    try std.testing.expect(server.failure == null);
}

test "silent client does not block the next hello" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    var sockets = try ShortSocketDir.init();
    defer sockets.cleanup();
    const path = try sockets.path("silent.sock");
    defer std.testing.allocator.free(path);
    var ready = std.atomic.Value(bool).init(false);
    var mock = bench.MockSession.init(std.testing.allocator);
    var server: ListenServer = .{ .mock = &mock, .config = .{
        .io = std.testing.io,
        .socket_path = path,
        .weights = weights,
        .identity = .{ .backend = "mock", .device = "test", .manifest_digest = digest, .pid = 1, .load_epoch = 2, .weights_revision = "weights" },
        .max_clients = 2,
        .ready = &ready,
    } };
    const thread = try std.Thread.spawn(.{}, ListenServer.run, .{&server});
    while (!ready.load(.acquire) and !server.stopped.load(.acquire))
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    if (!ready.load(.acquire)) {
        thread.join();
        if (server.failure) |err| {
            if (err == error.BindFailed) return error.SkipZigTest;
            return err;
        }
        return error.ServerStopped;
    }

    const silent = try rawConnect(path);
    defer _ = std.c.close(silent);
    const start = monotonicNs();
    const probe = try rawConnect(path);
    defer _ = std.c.close(probe);
    const hello_line = try readHelloLine(std.testing.allocator, probe, 5000);
    defer std.testing.allocator.free(hello_line);
    try std.testing.expect(monotonicNs() - start < 5 * std.time.ns_per_s);
    thread.join();
    try std.testing.expect(server.failure == null);
}

test "client refuses wrong resident_weights" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "held", .default_dir);
    try tmp.dir.createDir(std.testing.io, "asked", .default_dir);
    const held = try tmp.dir.realPathFileAlloc(std.testing.io, "held", std.testing.allocator);
    defer std.testing.allocator.free(held);
    const asked = try tmp.dir.realPathFileAlloc(std.testing.io, "asked", std.testing.allocator);
    defer std.testing.allocator.free(asked);
    var mock = bench.MockSession.init(std.testing.allocator);
    var activity: ActivityProbe = .{};
    const pair = try socketPair();
    var server: Server = .{ .mock = &mock, .fd = pair[0], .config = .{
        .io = std.testing.io,
        .socket_path = "unused",
        .activity = activity.hooks(),
        .weights = held,
        .identity = .{ .backend = "mock", .device = "test", .manifest_digest = digest, .pid = 1, .load_epoch = 2, .weights_revision = "held" },
    } };
    const thread = try std.Thread.spawn(.{}, Server.run, .{&server});
    try std.testing.expectError(
        error.WeightsMismatch,
        resident_client.Client.attach(std.testing.allocator, std.testing.io, pair[1], asked),
    );
    thread.join();
    // Failed handshake still balances both connection resets without dispatching.
    try activity.expectBalanced(2);
    try std.testing.expectEqual(error.MissingHandshake, server.failure.?);
}

test "stalled and partial handshakes do not block the next client" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(std.testing.io, "weights", .default_dir);
    const weights = try tmp.dir.realPathFileAlloc(std.testing.io, "weights", std.testing.allocator);
    defer std.testing.allocator.free(weights);
    var sockets = try ShortSocketDir.init();
    defer sockets.cleanup();
    const path = try sockets.path("deadline.sock");
    defer std.testing.allocator.free(path);
    var ready = std.atomic.Value(bool).init(false);
    var mock = bench.MockSession.init(std.testing.allocator);
    var server: ListenServer = .{ .mock = &mock, .config = .{
        .io = std.testing.io,
        .socket_path = path,
        .weights = weights,
        .identity = .{ .backend = "mock", .device = "test", .manifest_digest = digest, .pid = 1, .load_epoch = 2, .weights_revision = "weights" },
        .max_clients = 3,
        .ready = &ready,
        .timeout_ms = 50,
    } };
    const thread = try std.Thread.spawn(.{}, ListenServer.run, .{&server});
    while (!ready.load(.acquire) and !server.stopped.load(.acquire))
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    if (!ready.load(.acquire)) {
        thread.join();
        if (server.failure) |err| {
            if (err == error.BindFailed) return error.SkipZigTest;
            return err;
        }
        return error.ServerStopped;
    }

    const stalled = try rawConnect(path);
    try std.Io.sleep(std.testing.io, .fromMilliseconds(80), .awake);
    _ = std.c.close(stalled);

    const partial = try rawConnect(path);
    const partial_header = [_]u8{ 5, 0 };
    if (std.posix.errno(std.posix.system.write(partial, &partial_header, partial_header.len)) != .SUCCESS)
        return error.SocketWriteFailed;
    try std.Io.sleep(std.testing.io, .fromMilliseconds(80), .awake);
    _ = std.c.close(partial);

    var final = try resident_client.Client.connect(std.testing.allocator, std.testing.io, path, weights);
    try std.testing.expectEqual(@as(u32, 69), try final.session().prefill(&.{2}));
    final.deinit();
    thread.join();
    try std.testing.expect(server.failure == null);
}

test "directory identity separates equal relative paths under different bases" {
    var one = std.testing.tmpDir(.{});
    defer one.cleanup();
    var two = std.testing.tmpDir(.{});
    defer two.cleanup();
    try one.dir.createDir(std.testing.io, "weights", .default_dir);
    try two.dir.createDir(std.testing.io, "weights", .default_dir);
    const a = try resident.directoryIdentity(one.dir, std.testing.io, "weights");
    const b = try resident.directoryIdentity(two.dir, std.testing.io, "weights");
    try std.testing.expect(!std.meta.eql(a, b));
}

test "resident refuses an existing non-socket path" {
    var sockets = try ShortSocketDir.init();
    defer sockets.cleanup();
    const path = try sockets.path("blocked");
    defer std.testing.allocator.free(path);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = "x" });
    var mock = bench.MockSession.init(std.testing.allocator);
    try std.testing.expectError(error.ExistingPathNotSocket, resident.serve(std.testing.allocator, mock.session(), .{
        .io = std.testing.io,
        .socket_path = path,
        .weights = "weights",
        .identity = .{
            .backend = "mock",
            .device = "test",
            .manifest_digest = digest,
            .pid = 1,
            .load_epoch = 1,
            .weights_revision = "revision",
        },
        .max_clients = 0,
    }));
}
