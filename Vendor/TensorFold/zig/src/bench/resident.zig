//! One loaded bench session behind a same-uid Unix socket.
const std = @import("std");
const builtin = @import("builtin");
const bench = @import("bench_session");
const client = @import("resident_client.zig");
const protocol = @import("protocol.zig");

const capabilities = [_][]const u8{"free_run_decode"};
const spec_modes = [_][]const u8{ "serial", "dflash" };
const handshake_timeout_ms: u32 = 2_000;

pub const Identity = struct {
    backend: []const u8,
    device: []const u8,
    manifest_digest: []const u8,
    pid: u32,
    load_epoch: u64,
    weights_revision: []const u8,
};

pub const Config = struct {
    io: std.Io,
    socket_path: []const u8,
    weights: []const u8,
    identity: Identity,
    max_clients: ?usize = null,
    ready: ?*std.atomic.Value(bool) = null,
    timeout_ms: u32 = client.default_timeout_ms,
    // Wait between two requests of one attached client. benchd idles the connection during its quiescence and cool
    // gates (cool gate budget 900 s), so this must be longer than timeout_ms; a started frame still has timeout_ms.
    idle_timeout_ms: u32 = default_idle_timeout_ms,
    weights_identity: ?DirectoryIdentity = null,
    failure_writer: ?*std.Io.Writer = null,
};

pub const default_idle_timeout_ms: u32 = 3_600_000;

pub const DirectoryIdentity = struct { device_major: u32, device_minor: u32, inode: u64 };

pub fn serve(alloc: std.mem.Allocator, session: bench.BenchSession, config: Config) !void {
    const listener = try listen(config.io, config.socket_path);
    defer _ = std.c.close(listener);
    defer std.Io.Dir.cwd().deleteFile(config.io, config.socket_path) catch {};
    var resolved = config;
    resolved.weights_identity = try directoryIdentity(std.Io.Dir.cwd(), config.io, config.weights);
    if (config.ready) |ready| ready.store(true, .release);
    var served: usize = 0;
    while (config.max_clients == null or served < config.max_clients.?) {
        const rc = std.c.accept(listener, null, null);
        if (rc < 0) switch (std.posix.errno(rc)) {
            .INTR => continue,
            else => return error.AcceptFailed,
        };
        const fd: std.posix.fd_t = @intCast(rc);
        {
            defer _ = std.c.close(fd);
            try client.setSocketTimeout(fd, config.timeout_ms);
            if (!try sameUid(fd)) continue;
            session.reset();
            serveClient(alloc, fd, session, resolved) catch {};
            session.reset();
        }
        served += 1;
    }
}

pub fn serveConnected(alloc: std.mem.Allocator, fd: std.posix.fd_t, session: bench.BenchSession, config: Config) !void {
    var resolved = config;
    resolved.weights_identity = try directoryIdentity(std.Io.Dir.cwd(), config.io, config.weights);
    try client.setSocketTimeout(fd, config.timeout_ms);
    if (!try sameUid(fd)) return error.PeerUidMismatch;
    session.reset();
    defer session.reset();
    return serveClient(alloc, fd, session, resolved);
}

fn serveClient(alloc: std.mem.Allocator, fd: std.posix.fd_t, session: bench.BenchSession, config: Config) !void {
    try sendHello(alloc, fd, config);
    const first_timeout_ms = @min(config.timeout_ms, handshake_timeout_ms);
    const first = (try client.receiveFrameWithTransferTimeout(alloc, fd, first_timeout_ms, config.timeout_ms)) orelse
        return error.MissingHandshake;
    defer alloc.free(first);
    var hello: client.Decoder = .{ .bytes = first };
    if (try hello.byte() != @backingInt(client.Op.hello)) return error.MissingHandshake;
    const weights = try hello.string();
    try hello.done();
    const requested = directoryIdentity(std.Io.Dir.cwd(), config.io, weights) catch null;
    if (requested == null or !std.meta.eql(requested.?, config.weights_identity.?)) {
        try sendFailure(alloc, fd, "WeightsMismatch");
        return;
    }
    var response: client.Encoder = .{};
    defer response.deinit(alloc);
    try response.byte(alloc, 0);
    try response.string(alloc, config.identity.backend);
    try response.string(alloc, config.identity.device);
    try response.string(alloc, config.identity.manifest_digest);
    try response.int(alloc, u32, config.identity.pid);
    try response.int(alloc, u64, config.identity.load_epoch);
    try response.string(alloc, config.identity.weights_revision);
    try client.sendFrame(fd, response.bytes.items);

    while (true) {
        const frame = (try client.receiveFrameWithTransferTimeout(alloc, fd, config.idle_timeout_ms, config.timeout_ms)) orelse return;
        defer alloc.free(frame);
        var reply = dispatch(alloc, session, frame) catch |err| {
            reportFailure(config.failure_writer, operationName(frame), @errorName(err));
            try sendFailure(alloc, fd, @errorName(err));
            continue;
        };
        defer reply.deinit(alloc);
        try client.sendFrame(fd, reply.bytes.items);
    }
}

fn operationName(frame: []const u8) []const u8 {
    if (frame.len == 0) return "unknown";
    const op = std.enums.fromInt(client.Op, frame[0]) orelse return "unknown";
    return @tagName(op);
}

fn reportFailure(writer: ?*std.Io.Writer, op: []const u8, reason: []const u8) void {
    if (writer) |out| {
        out.print("bench-worker resident: {s} failed: {s}\n", .{ op, reason }) catch return;
        out.flush() catch {};
        return;
    }
    std.debug.print("bench-worker resident: {s} failed: {s}\n", .{ op, reason });
}

const WireHello = struct {
    id: i64 = 0,
    nonce: []const u8,
    ok: bool = true,
    expert_stats: protocol.ExpertStats = .{},
    protocol_version: u32 = protocol.protocol_version,
    backend: []const u8,
    device: []const u8,
    capabilities: []const []const u8 = &capabilities,
    spec_modes: []const []const u8 = &spec_modes,
    runner: protocol.RunnerIdentity,
    resident: protocol.ResidentIdentity,
    resident_weights: []const u8,
};

fn sendHello(alloc: std.mem.Allocator, fd: std.posix.fd_t, config: Config) !void {
    var nonce_bytes: [16]u8 = undefined;
    try randomBytes(&nonce_bytes);
    const nonce_hex = std.fmt.bytesToHex(nonce_bytes, .lower);
    const resolved = try std.Io.Dir.cwd().realPathFileAlloc(config.io, config.weights, alloc);
    defer alloc.free(resolved);
    var line: std.Io.Writer.Allocating = .init(alloc);
    defer line.deinit();
    try std.json.Stringify.value(WireHello{
        .nonce = &nonce_hex,
        .backend = config.identity.backend,
        .device = config.identity.device,
        .runner = .{
            .id = "tensorfold/qwen3.8-27b",
            .model_type = "qwen3_5",
            .manifest_sha256 = config.identity.manifest_digest,
            .build = "tensorfold-1.1.6",
        },
        .resident = .{ .pid = config.identity.pid, .load_epoch = config.identity.load_epoch },
        .resident_weights = resolved,
    }, .{}, &line.writer);
    try line.writer.writeByte('\n');
    try client.writeAll(fd, line.written());
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

fn dispatch(alloc: std.mem.Allocator, session: bench.BenchSession, frame: []const u8) !client.Encoder {
    var d: client.Decoder = .{ .bytes = frame };
    const op = std.enums.fromInt(client.Op, try d.byte()) orelse return error.UnknownOperation;
    if (op == .hello) return error.UnexpectedHandshake;
    var out: client.Encoder = .{};
    errdefer out.deinit(alloc);
    try out.byte(alloc, 0);
    switch (op) {
        .hello => unreachable,
        .prefill => {
            const tokens = try d.tokensAlloc(alloc);
            defer alloc.free(tokens);
            try d.done();
            try out.int(alloc, u32, try session.prefill(tokens));
        },
        .generate => {
            var check = d;
            try check.skipTokens();
            const checked_steps = try check.int(u32);
            try check.done();
            if (checked_steps > client.max_correctness_steps) return error.RequestTooLarge;
            const tokens = try d.tokensAlloc(alloc);
            defer alloc.free(tokens);
            const steps = try d.int(u32);
            try d.done();
            const generated = try alloc.alloc(u32, steps);
            defer alloc.free(generated);
            try session.generate(tokens, steps, generated);
            try out.tokensWithLimit(alloc, generated, client.max_correctness_steps);
        },
        .seed => {
            const tokens = try d.tokensAlloc(alloc);
            defer alloc.free(tokens);
            const want_top = (try d.byte()) != 0;
            try d.done();
            try encodeStep(alloc, &out, try session.seed(tokens, want_top));
        },
        .force => {
            const token = try d.int(u32);
            const want_top = (try d.byte()) != 0;
            try d.done();
            try encodeStep(alloc, &out, try session.force(token, want_top));
        },
        .seed_free => {
            const tokens = try d.tokensAlloc(alloc);
            defer alloc.free(tokens);
            const spec: bench.FreeSpec = switch (try d.byte()) {
                0 => .serial,
                1 => .{ .dflash = .{ .depth = try d.byte() } },
                else => return error.InvalidFreeSpec,
            };
            try d.done();
            try out.int(alloc, u32, try session.seedFree(tokens, spec));
        },
        .run_free => {
            const count = try d.int(u32);
            try d.done();
            if (count > client.max_free_count) return error.RequestTooLarge;
            const generated = try alloc.alloc(u32, count);
            defer alloc.free(generated);
            var audit = bench.DflashAudit.init();
            defer audit.deinit(alloc);
            try session.runFree(count, generated, &audit);
            try audit.check(count);
            try out.tokensWithLimit(alloc, generated, client.max_free_count);
            try out.int(alloc, u32, @intCast(audit.acceptance_lengths.items.len));
            for (audit.acceptance_lengths.items) |length| try out.int(alloc, u32, length);
            try out.int(alloc, u64, audit.drafted_total);
            try out.int(alloc, u64, audit.accepted_total);
            try out.int(alloc, u64, audit.committed_total);
        },
        .close_phase => {
            try d.done();
            try encodeDiag(alloc, &out, try session.closePhase());
        },
        .inspect_phase => {
            try d.done();
            try encodeDiag(alloc, &out, try session.inspectPhase());
        },
        .reset => {
            try d.done();
            session.reset();
        },
    }
    return out;
}

fn encodeStep(alloc: std.mem.Allocator, out: *client.Encoder, step: bench.Step) !void {
    try out.int(alloc, u32, step.token);
    try out.byte(alloc, @intFromBool(step.has_top));
    if (step.has_top) {
        for (step.top) |entry| {
            try out.int(alloc, u32, entry.token);
            try out.float(alloc, entry.logit);
        }
    }
}

pub fn directoryIdentity(base: std.Io.Dir, io: std.Io, path: []const u8) !DirectoryIdentity {
    const resolved = try base.realPathFileAlloc(io, path, std.heap.page_allocator);
    defer std.heap.page_allocator.free(resolved);
    const resolved_z = try std.fmt.allocPrintSentinel(std.heap.page_allocator, "{s}", .{resolved}, 0);
    defer std.heap.page_allocator.free(resolved_z);
    if (comptime builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var statx = std.mem.zeroes(linux.Statx);
        switch (linux.errno(linux.statx(linux.AT.FDCWD, resolved_z.ptr, 0, .{ .INO = true }, &statx))) {
            .SUCCESS => return .{ .device_major = statx.dev_major, .device_minor = statx.dev_minor, .inode = statx.ino },
            else => return error.StatFailed,
        }
    }
    if (comptime builtin.os.tag == .macos) {
        var stat: std.c.Stat = undefined;
        if (std.c.stat(resolved_z.ptr, &stat) != 0) return error.StatFailed;
        return .{ .device_major = 0, .device_minor = @bitCast(stat.dev), .inode = stat.ino };
    }
    return error.UnsupportedPlatform;
}

fn encodeDiag(alloc: std.mem.Allocator, out: *client.Encoder, diag: bench.PhaseDiag) !void {
    try out.int(alloc, u64, diag.work_tokens);
    try out.byte(alloc, @intFromBool(diag.peak_ram_bytes != null));
    if (diag.peak_ram_bytes) |peak| try out.int(alloc, u64, peak);
    try out.int(alloc, u64, diag.cache_bytes);
}

fn sendFailure(alloc: std.mem.Allocator, fd: std.posix.fd_t, message: []const u8) !void {
    var out: client.Encoder = .{};
    defer out.deinit(alloc);
    try out.byte(alloc, 1);
    try out.string(alloc, message);
    try client.sendFrame(fd, out.bytes.items);
}

fn listen(io: std.Io, path: []const u8) !std.posix.fd_t {
    const cwd = std.Io.Dir.cwd();
    if (cwd.statFile(io, path, .{ .follow_symlinks = false })) |stat| {
        if (stat.kind != .unix_domain_socket) return error.ExistingPathNotSocket;
        try cwd.deleteFile(io, path);
    } else |err| if (err != error.FileNotFound) return err;
    var addr = try client.unixAddress(path);
    const rc = std.c.socket(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0);
    if (rc < 0) return error.SocketOpenFailed;
    const fd: std.posix.fd_t = @intCast(rc);
    errdefer _ = std.c.close(fd);
    if (std.c.bind(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr))) != 0) return error.BindFailed;
    const path_z = try std.posix.toPosixPath(path);
    if (std.c.chmod(&path_z, 0o600) != 0) return error.ChmodFailed;
    if (std.c.listen(fd, 16) != 0) return error.ListenFailed;
    return fd;
}

pub fn uidAllowed(expected: std.c.uid_t, actual: std.c.uid_t) bool {
    return expected == actual;
}

fn sameUid(fd: std.posix.fd_t) !bool {
    const expected = std.c.getuid();
    if (comptime builtin.os.tag == .macos) {
        var uid: std.c.uid_t = undefined;
        var gid: std.c.gid_t = undefined;
        if (getpeereid(fd, &uid, &gid) != 0) return error.PeerCredentialsFailed;
        return uidAllowed(expected, uid);
    }
    if (comptime builtin.os.tag == .linux) {
        const Ucred = extern struct { pid: c_int, uid: std.c.uid_t, gid: std.c.gid_t };
        var cred: Ucred = undefined;
        var len: std.c.socklen_t = @sizeOf(Ucred);
        if (std.c.getsockopt(fd, std.c.SOL.SOCKET, std.c.SO.PEERCRED, &cred, &len) != 0) return error.PeerCredentialsFailed;
        return len == @sizeOf(Ucred) and uidAllowed(expected, cred.uid);
    }
    return error.PeerCredentialsUnsupported;
}

extern "c" fn getpeereid(fd: c_int, uid: *std.c.uid_t, gid: *std.c.gid_t) c_int;
