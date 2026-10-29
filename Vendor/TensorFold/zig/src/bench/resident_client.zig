//! BenchSession proxy for one resident Unix socket.
const std = @import("std");
const bench = @import("bench_session");
const protocol = @import("protocol.zig");
pub const max_frame_bytes: usize = 64 * 1024 * 1024;
pub const max_hello_bytes: usize = 1024 * 1024;
pub const max_prompt_tokens: usize = 2048;
pub const max_correctness_steps: u32 = 8192;
pub const max_free_count: u32 = 1536;
pub const default_timeout_ms: u32 = 30_000;
const minimum_inference_timeout_ms: u32 = 60_000;
const inference_timeout_per_token_ms: u32 = 1_000;
pub const Op = enum(u8) {
    hello,
    prefill,
    generate,
    seed,
    force,
    seed_free,
    run_free,
    close_phase,
    inspect_phase,
    reset,
};
pub const ResidentHello = struct {
    backend: []u8,
    device: []u8,
    manifest_digest: [64]u8,
    resident: protocol.ResidentIdentity,
    weights_revision: []u8,
};
pub const Encoder = struct {
    bytes: std.ArrayListUnmanaged(u8) = .empty,

    pub fn deinit(e: *Encoder, alloc: std.mem.Allocator) void {
        e.bytes.deinit(alloc);
    }
    pub fn byte(e: *Encoder, alloc: std.mem.Allocator, value: u8) !void {
        try e.bytes.append(alloc, value);
    }

    pub fn int(e: *Encoder, alloc: std.mem.Allocator, comptime T: type, value: T) !void {
        var raw: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &raw, value, .little);
        try e.bytes.appendSlice(alloc, &raw);
    }
    pub fn float(e: *Encoder, alloc: std.mem.Allocator, value: f32) !void {
        try e.int(alloc, u32, @bitCast(value));
    }

    pub fn string(e: *Encoder, alloc: std.mem.Allocator, value: []const u8) !void {
        try e.int(alloc, u32, std.math.cast(u32, value.len) orelse return error.FrameTooLarge);
        try e.bytes.appendSlice(alloc, value);
    }
    pub fn tokens(e: *Encoder, alloc: std.mem.Allocator, values: []const u32) !void {
        return e.tokensWithLimit(alloc, values, max_prompt_tokens);
    }

    pub fn tokensWithLimit(e: *Encoder, alloc: std.mem.Allocator, values: []const u32, limit: usize) !void {
        if (values.len > limit) return error.TooManyTokens;
        try e.int(alloc, u32, std.math.cast(u32, values.len) orelse return error.FrameTooLarge);
        for (values) |value| try e.int(alloc, u32, value);
    }
};
pub const Decoder = struct {
    bytes: []const u8,
    at: usize = 0,

    pub fn byte(d: *Decoder) !u8 {
        if (d.at == d.bytes.len) return error.TruncatedFrame;
        defer d.at += 1;
        return d.bytes[d.at];
    }
    pub fn int(d: *Decoder, comptime T: type) !T {
        const end = std.math.add(usize, d.at, @sizeOf(T)) catch return error.TruncatedFrame;
        if (end > d.bytes.len) return error.TruncatedFrame;
        defer d.at = end;
        const raw: *const [@sizeOf(T)]u8 = @ptrCast(d.bytes[d.at..end].ptr);
        return std.mem.readInt(T, raw, .little);
    }

    pub fn float(d: *Decoder) !f32 {
        return @bitCast(try d.int(u32));
    }
    pub fn string(d: *Decoder) ![]const u8 {
        const len = try d.int(u32);
        const end = std.math.add(usize, d.at, len) catch return error.TruncatedFrame;
        if (end > d.bytes.len) return error.TruncatedFrame;
        defer d.at = end;
        return d.bytes[d.at..end];
    }

    pub fn tokensAlloc(d: *Decoder, alloc: std.mem.Allocator) ![]u32 {
        const len = try d.tokensLength();
        const start = d.at;
        d.at += len * @sizeOf(u32);
        const out = try alloc.alloc(u32, len);
        errdefer alloc.free(out);
        var values: Decoder = .{ .bytes = d.bytes[start..d.at] };
        for (out) |*value| value.* = try values.int(u32);
        return out;
    }
    pub fn skipTokens(d: *Decoder) !void {
        const len = try d.tokensLength();
        d.at += len * @sizeOf(u32);
    }

    fn tokensLength(d: *Decoder) !usize {
        const len: usize = try d.int(u32);
        if (len > max_prompt_tokens) return error.TooManyTokens;
        const bytes = std.math.mul(usize, len, @sizeOf(u32)) catch return error.TruncatedFrame;
        if (bytes > d.bytes.len - d.at) return error.TruncatedFrame;
        return len;
    }

    pub fn done(d: Decoder) !void {
        if (d.at != d.bytes.len) return error.TrailingFrameData;
    }
};
pub fn sendFrame(fd: std.posix.fd_t, bytes: []const u8) !void {
    if (bytes.len > max_frame_bytes) return error.FrameTooLarge;
    var header: [4]u8 = undefined;
    std.mem.writeInt(u32, &header, @intCast(bytes.len), .little);
    try writeAll(fd, &header);
    try writeAll(fd, bytes);
}
pub fn receiveFrame(alloc: std.mem.Allocator, fd: std.posix.fd_t) !?[]u8 {
    return receiveFrameWithTimeout(alloc, fd, default_timeout_ms);
}

pub fn receiveFrameWithTimeout(alloc: std.mem.Allocator, fd: std.posix.fd_t, timeout_ms: u32) !?[]u8 {
    var deadline = try Deadline.init(timeout_ms);
    return receiveFrameBefore(alloc, fd, &deadline, null);
}
pub fn receiveFrameWithTransferTimeout(alloc: std.mem.Allocator, fd: std.posix.fd_t, wait_timeout_ms: u32, transfer_timeout_ms: u32) !?[]u8 {
    var deadline = try Deadline.init(wait_timeout_ms);
    return receiveFrameBefore(alloc, fd, &deadline, transfer_timeout_ms);
}
fn receiveFrameBefore(alloc: std.mem.Allocator, fd: std.posix.fd_t, deadline: *const Deadline, transfer_timeout_ms: ?u32) !?[]u8 {
    var header: [4]u8 = undefined;
    const first = try readSomeBefore(fd, &header, deadline);
    if (first == 0) return null;
    var transfer_deadline = if (transfer_timeout_ms) |timeout| try Deadline.init(timeout) else undefined;
    const frame_deadline = if (transfer_timeout_ms != null) &transfer_deadline else deadline;
    try readRestBefore(fd, header[first..], frame_deadline);
    const len = std.mem.readInt(u32, &header, .little);
    if (len > max_frame_bytes) return error.FrameTooLarge;
    const bytes = try alloc.alloc(u8, len);
    errdefer alloc.free(bytes);
    try readRestBefore(fd, bytes, frame_deadline);
    return bytes;
}

fn readSomeBefore(fd: std.posix.fd_t, out: []u8, deadline: *const Deadline) !usize {
    try waitReadable(fd, deadline);
    while (true) {
        const rc = std.posix.system.read(fd, out.ptr, out.len);
        switch (std.posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => continue,
            .AGAIN => return error.Timeout,
            .CONNRESET => return error.ConnectionResetByPeer,
            else => return error.SocketReadFailed,
        }
    }
}

fn waitReadable(fd: std.posix.fd_t, deadline: *const Deadline) !void {
    while (true) {
        var fds = [_]std.c.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const rc = std.c.poll(&fds, 1, @intCast(try deadline.remainingMs()));
        if (rc > 0) return;
        if (rc == 0) return error.Timeout;
        if (std.posix.errno(rc) != .INTR) return error.SocketPollFailed;
    }
}

fn readRestBefore(fd: std.posix.fd_t, out: []u8, deadline: *const Deadline) !void {
    var at: usize = 0;
    while (at < out.len) {
        const count = try readSomeBefore(fd, out[at..], deadline);
        if (count == 0) return error.EndOfStream;
        at += count;
    }
}

pub fn writeAll(fd: std.posix.fd_t, bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = std.posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        switch (std.posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.EndOfStream;
                at += @intCast(rc);
            },
            .INTR => continue,
            .AGAIN => return error.SocketTimeout,
            .PIPE, .CONNRESET => return error.ConnectionResetByPeer,
            else => return error.SocketWriteFailed,
        }
    }
}

const WireHello = struct {
    id: i64,
    nonce: []const u8,
    ok: bool,
    expert_stats: protocol.ExpertStats,
    protocol_version: u32,
    backend: []const u8,
    device: []const u8,
    capabilities: []const []const u8,
    spec_modes: []const []const u8,
    runner: protocol.RunnerIdentity,
    resident: protocol.ResidentIdentity,
    resident_weights: []const u8,
};

fn receiveHelloBefore(alloc: std.mem.Allocator, fd: std.posix.fd_t, deadline: *const Deadline) ![]u8 {
    var line: std.ArrayListUnmanaged(u8) = .empty;
    errdefer line.deinit(alloc);
    while (line.items.len < max_hello_bytes) {
        var byte: [1]u8 = undefined;
        if (try readSomeBefore(fd, &byte, deadline) == 0) return error.NoResidentHello;
        if (byte[0] == '\n') return line.toOwnedSlice(alloc);
        try line.append(alloc, byte[0]);
    }
    return error.ResidentHelloTooLarge;
}

pub const Client = struct {
    allocator: std.mem.Allocator,
    fd: std.posix.fd_t,
    hello: ResidentHello,
    timeout_ms: u32,
    last_failure: [128]u8 = undefined,
    last_failure_len: usize = 0,

    pub fn connect(alloc: std.mem.Allocator, io: std.Io, path: []const u8, weights: []const u8) !Client {
        return connectWithTimeout(alloc, io, path, weights, default_timeout_ms);
    }
    pub fn connectWithTimeout(alloc: std.mem.Allocator, io: std.Io, path: []const u8, weights: []const u8, timeout_ms: u32) !Client {
        var deadline = try Deadline.init(timeout_ms);
        const fd = try connectSocket(path, &deadline);
        return attachBefore(alloc, io, fd, weights, timeout_ms, &deadline);
    }
    pub fn attach(alloc: std.mem.Allocator, io: std.Io, fd: std.posix.fd_t, weights: []const u8) !Client {
        return attachWithTimeout(alloc, io, fd, weights, default_timeout_ms);
    }

    pub fn attachWithTimeout(alloc: std.mem.Allocator, io: std.Io, fd: std.posix.fd_t, weights: []const u8, timeout_ms: u32) !Client {
        var deadline = try Deadline.init(timeout_ms);
        return attachBefore(alloc, io, fd, weights, timeout_ms, &deadline);
    }
    fn attachBefore(alloc: std.mem.Allocator, io: std.Io, fd: std.posix.fd_t, weights: []const u8, timeout_ms: u32, deadline: *const Deadline) !Client {
        errdefer _ = std.c.close(fd);
        try setSocketTimeout(fd, try deadline.remainingMs());
        const hello_line = try receiveHelloBefore(alloc, fd, deadline);
        defer alloc.free(hello_line);
        var parsed = std.json.parseFromSlice(WireHello, alloc, hello_line, .{
            .allocate = .alloc_always,
            .duplicate_field_behavior = .@"error",
        }) catch return error.InvalidResidentHello;
        defer parsed.deinit();
        const wire = parsed.value;
        if (wire.id != 0 or !wire.ok or wire.nonce.len == 0 or wire.protocol_version != protocol.protocol_version)
            return error.InvalidResidentHello;
        var hello_request: Encoder = .{};
        defer hello_request.deinit(alloc);
        try hello_request.byte(alloc, @backingInt(Op.hello));
        const resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, weights, alloc);
        defer alloc.free(resolved);
        const resident_weights = std.Io.Dir.cwd().realPathFileAlloc(io, wire.resident_weights, alloc) catch
            return error.WeightsMismatch;
        defer alloc.free(resident_weights);
        if (!std.mem.eql(u8, resolved, resident_weights)) return error.WeightsMismatch;
        try hello_request.string(alloc, resolved);
        try setSocketTimeout(fd, try deadline.remainingMs());
        try sendFrame(fd, hello_request.bytes.items);
        const response = (try receiveFrameBefore(alloc, fd, deadline, null)) orelse return error.NoResidentHello;
        defer alloc.free(response);
        var d: Decoder = .{ .bytes = response };
        try responseOk(&d);
        const backend = try alloc.dupe(u8, try d.string());
        errdefer alloc.free(backend);
        const device = try alloc.dupe(u8, try d.string());
        errdefer alloc.free(device);
        const digest = try d.string();
        if (digest.len != 64) return error.InvalidResidentHello;
        const pid = try d.int(u32);
        const load_epoch = try d.int(u64);
        const revision = try alloc.dupe(u8, try d.string());
        errdefer alloc.free(revision);
        try d.done();
        return .{
            .allocator = alloc,
            .fd = fd,
            .timeout_ms = timeout_ms,
            .hello = .{
                .backend = backend,
                .device = device,
                .manifest_digest = digest[0..64].*,
                .resident = .{ .pid = pid, .load_epoch = load_epoch },
                .weights_revision = revision,
            },
        };
    }

    pub fn deinit(c: *Client) void {
        _ = std.c.close(c.fd);
        c.allocator.free(c.hello.backend);
        c.allocator.free(c.hello.device);
        c.allocator.free(c.hello.weights_revision);
        c.* = undefined;
    }
    pub fn session(c: *Client) bench.BenchSession {
        return .{ .ptr = c, .vtable = &vtable };
    }

    fn self(ptr: *anyopaque) *Client {
        return @ptrCast(@alignCast(ptr));
    }

    fn call(c: *Client, encoded: *Encoder, wait_timeout_ms: u32) !struct { bytes: []u8, decoder: Decoder } {
        c.last_failure_len = 0;
        try setSocketTimeout(c.fd, c.timeout_ms);
        try sendFrame(c.fd, encoded.bytes.items);
        var wait_deadline = try Deadline.init(wait_timeout_ms);
        const bytes = (try receiveFrameBefore(c.allocator, c.fd, &wait_deadline, c.timeout_ms)) orelse
            return error.ResidentClosed;
        var d: Decoder = .{ .bytes = bytes };
        responseOk(&d) catch |err| {
            if (err == error.ResidentRequestFailed) {
                var failure: Decoder = .{ .bytes = bytes };
                _ = failure.byte() catch {};
                const reason = failure.string() catch "";
                const len = @min(reason.len, c.last_failure.len);
                @memcpy(c.last_failure[0..len], reason[0..len]);
                c.last_failure_len = len;
            }
            c.allocator.free(bytes);
            return err;
        };
        return .{ .bytes = bytes, .decoder = d };
    }

    fn inferenceTimeout(c: *const Client, tokens: u32) u32 {
        const work_ms = std.math.mul(u32, @max(tokens, 1), inference_timeout_per_token_ms) catch
            std.math.maxInt(u32);
        const budget = std.math.add(u32, minimum_inference_timeout_ms, work_ms) catch std.math.maxInt(u32);
        return @max(c.timeout_ms, budget);
    }

    pub fn failureReason(c: *const Client, err: anyerror) []const u8 {
        return if (err == error.ResidentRequestFailed and c.last_failure_len != 0) c.last_failure[0..c.last_failure_len] else @errorName(err);
    }
    fn request(c: *Client, op: Op) !Encoder {
        var e: Encoder = .{};
        errdefer e.deinit(c.allocator);
        try e.byte(c.allocator, @backingInt(op));
        return e;
    }

    fn prefill(ptr: *anyopaque, tokens: []const u32) anyerror!u32 {
        const c = self(ptr);
        var e = try c.request(.prefill);
        defer e.deinit(c.allocator);
        try e.tokens(c.allocator, tokens);
        var r = try c.call(&e, c.inferenceTimeout(@intCast(tokens.len)));
        defer c.allocator.free(r.bytes);
        const value = try r.decoder.int(u32);
        try r.decoder.done();
        return value;
    }

    fn generate(ptr: *anyopaque, tokens: []const u32, steps: u32, out: []u32) anyerror!void {
        const c = self(ptr);
        if (tokens.len > max_prompt_tokens or steps > max_correctness_steps) return error.RequestTooLarge;
        if (out.len < steps) return error.OutputTooSmall;
        var e = try c.request(.generate);
        defer e.deinit(c.allocator);
        try e.tokens(c.allocator, tokens);
        try e.int(c.allocator, u32, steps);
        var r = try c.call(&e, c.inferenceTimeout(steps));
        defer c.allocator.free(r.bytes);
        const count = try r.decoder.int(u32);
        if (count != steps) return error.BadResidentResponse;
        for (out[0..steps]) |*value| value.* = try r.decoder.int(u32);
        try r.decoder.done();
    }

    fn decodeStep(d: *Decoder) !bench.Step {
        var result: bench.Step = .{ .token = try d.int(u32), .top = undefined, .has_top = (try d.byte()) != 0 };
        if (result.has_top) {
            for (&result.top) |*entry| entry.* = .{ .token = try d.int(u32), .logit = try d.float() };
        }
        return result;
    }

    fn seed(ptr: *anyopaque, tokens: []const u32, want_top: bool) anyerror!bench.Step {
        const c = self(ptr);
        var e = try c.request(.seed);
        defer e.deinit(c.allocator);
        try e.tokens(c.allocator, tokens);
        try e.byte(c.allocator, @intFromBool(want_top));
        var r = try c.call(&e, c.inferenceTimeout(@intCast(tokens.len)));
        defer c.allocator.free(r.bytes);
        const result = try decodeStep(&r.decoder);
        try r.decoder.done();
        return result;
    }

    fn force(ptr: *anyopaque, token: u32, want_top: bool) anyerror!bench.Step {
        const c = self(ptr);
        var e = try c.request(.force);
        defer e.deinit(c.allocator);
        try e.int(c.allocator, u32, token);
        try e.byte(c.allocator, @intFromBool(want_top));
        var r = try c.call(&e, c.inferenceTimeout(1));
        defer c.allocator.free(r.bytes);
        const result = try decodeStep(&r.decoder);
        try r.decoder.done();
        return result;
    }

    fn seedFree(ptr: *anyopaque, tokens: []const u32, spec: bench.FreeSpec) anyerror!u32 {
        const c = self(ptr);
        var e = try c.request(.seed_free);
        defer e.deinit(c.allocator);
        try e.tokens(c.allocator, tokens);
        switch (spec) {
            .serial => try e.byte(c.allocator, 0),
            .dflash => |d| {
                try e.byte(c.allocator, 1);
                try e.byte(c.allocator, d.depth);
            },
        }
        var r = try c.call(&e, c.inferenceTimeout(@intCast(tokens.len)));
        defer c.allocator.free(r.bytes);
        const result = try r.decoder.int(u32);
        try r.decoder.done();
        return result;
    }

    fn runFree(ptr: *anyopaque, count: u32, out: []u32, audit: *bench.DflashAudit) anyerror!void {
        const c = self(ptr);
        if (count > max_free_count) return error.RequestTooLarge;
        if (out.len < count) return error.OutputTooSmall;
        var e = try c.request(.run_free);
        defer e.deinit(c.allocator);
        try e.int(c.allocator, u32, count);
        var r = try c.call(&e, c.inferenceTimeout(count));
        defer c.allocator.free(r.bytes);
        if (try r.decoder.int(u32) != count) return error.BadResidentResponse;
        for (out[0..count]) |*value| value.* = try r.decoder.int(u32);
        const rounds = try r.decoder.int(u32);
        if (rounds > max_free_count) return error.BadResidentResponse;
        const audit_bytes = std.math.mul(usize, rounds, @sizeOf(u32)) catch return error.BadResidentResponse;
        if (audit_bytes + 3 * @sizeOf(u64) > r.decoder.bytes.len - r.decoder.at) return error.BadResidentResponse;
        try audit.acceptance_lengths.ensureUnusedCapacity(c.allocator, rounds);
        for (0..rounds) |_| audit.acceptance_lengths.appendAssumeCapacity(try r.decoder.int(u32));
        audit.drafted_total = try r.decoder.int(u64);
        audit.accepted_total = try r.decoder.int(u64);
        audit.committed_total = try r.decoder.int(u64);
        try r.decoder.done();
    }

    fn decodeDiag(d: *Decoder) !bench.PhaseDiag {
        const work = try d.int(u64);
        const has_peak = try d.byte();
        const peak = if (has_peak != 0) try d.int(u64) else null;
        return .{ .work_tokens = work, .peak_ram_bytes = peak, .cache_bytes = try d.int(u64) };
    }

    fn diag(ptr: *anyopaque, op: Op) anyerror!bench.PhaseDiag {
        const c = self(ptr);
        var e = try c.request(op);
        defer e.deinit(c.allocator);
        var r = try c.call(&e, c.timeout_ms);
        defer c.allocator.free(r.bytes);
        const result = try decodeDiag(&r.decoder);
        try r.decoder.done();
        return result;
    }

    fn closePhase(ptr: *anyopaque) anyerror!bench.PhaseDiag {
        return diag(ptr, .close_phase);
    }

    fn inspectPhase(ptr: *anyopaque) anyerror!bench.PhaseDiag {
        return diag(ptr, .inspect_phase);
    }

    fn reset(ptr: *anyopaque) void {
        const c = self(ptr);
        var e = c.request(.reset) catch return;
        defer e.deinit(c.allocator);
        var r = c.call(&e, c.timeout_ms) catch return;
        defer c.allocator.free(r.bytes);
        r.decoder.done() catch {};
    }

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

fn responseOk(d: *Decoder) !void {
    if (try d.byte() == 0) return;
    _ = try d.string();
    return error.ResidentRequestFailed;
}

const Deadline = struct {
    expires_ns: u64,

    fn init(timeout_ms: u32) !Deadline {
        const now = try monotonicNs();
        return .{ .expires_ns = now + @as(u64, timeout_ms) * std.time.ns_per_ms };
    }

    fn remainingMs(d: Deadline) !u32 {
        const now = try monotonicNs();
        if (now >= d.expires_ns) return error.Timeout;
        return @intCast((d.expires_ns - now + std.time.ns_per_ms - 1) / std.time.ns_per_ms);
    }
};

fn monotonicNs() !u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0 or ts.sec < 0 or ts.nsec < 0 or ts.nsec >= std.time.ns_per_s)
        return error.ClockUnavailable;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn connectSocket(path: []const u8, deadline: *const Deadline) !std.posix.fd_t {
    var addr = try unixAddress(path);
    const rc = std.posix.system.socket(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0);
    if (std.posix.errno(rc) != .SUCCESS) return error.SocketOpenFailed;
    const fd: std.posix.fd_t = @intCast(rc);
    errdefer _ = std.c.close(fd);
    const flags = std.c.fcntl(fd, std.c.F.GETFL);
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.SocketModeFailed;
    const connect_rc = std.c.connect(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr)));
    if (connect_rc != 0) {
        if (std.posix.errno(connect_rc) != .INPROGRESS) return error.ResidentUnavailable;
        try waitWritable(fd, deadline);
        var socket_error: c_int = 0;
        var error_len: std.c.socklen_t = @sizeOf(c_int);
        if (std.c.getsockopt(fd, std.c.SOL.SOCKET, std.c.SO.ERROR, &socket_error, &error_len) != 0 or socket_error != 0)
            return error.ResidentUnavailable;
    }
    if (std.c.fcntl(fd, std.c.F.SETFL, flags) < 0) return error.SocketModeFailed;
    return fd;
}

fn waitWritable(fd: std.posix.fd_t, deadline: *const Deadline) !void {
    while (true) {
        var fds = [_]std.c.pollfd{.{ .fd = fd, .events = std.posix.POLL.OUT, .revents = 0 }};
        const rc = std.c.poll(&fds, 1, @intCast(try deadline.remainingMs()));
        if (rc > 0) return;
        if (rc == 0) return error.Timeout;
        if (std.posix.errno(rc) != .INTR) return error.ConnectPollFailed;
    }
}

fn setReceiveTimeout(fd: std.posix.fd_t, timeout_ms: u32) !void {
    const tv: std.c.timeval = .{
        .sec = @intCast(timeout_ms / 1000),
        .usec = @intCast((timeout_ms % 1000) * 1000),
    };
    while (std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.RCVTIMEO, &tv, @sizeOf(std.c.timeval)) != 0) {
        if (std.posix.errno(-1) != .INTR) return error.SetSocketTimeoutFailed;
    }
}

pub fn setSocketTimeout(fd: std.posix.fd_t, timeout_ms: u32) !void {
    const tv: std.c.timeval = .{
        .sec = @intCast(timeout_ms / 1000),
        .usec = @intCast((timeout_ms % 1000) * 1000),
    };
    try setReceiveTimeout(fd, timeout_ms);
    while (std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.SNDTIMEO, &tv, @sizeOf(std.c.timeval)) != 0) {
        if (std.posix.errno(-1) != .INTR) return error.SetSocketTimeoutFailed;
    }
}

pub fn unixAddress(path: []const u8) !std.c.sockaddr.un {
    var addr = std.mem.zeroes(std.c.sockaddr.un);
    if (path.len == 0 or path.len >= addr.path.len) return error.SocketPathTooLong;
    if (@hasField(std.c.sockaddr.un, "len")) addr.len = @sizeOf(std.c.sockaddr.un);
    addr.family = std.posix.AF.UNIX;
    @memcpy(addr.path[0..path.len], path);
    return addr;
}
