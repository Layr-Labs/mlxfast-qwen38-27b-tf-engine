const std = @import("std");

pub const protocol_version: u32 = 1;
pub const top_logits_k: usize = 8;

pub const Kind = enum {
    prefill,
    decode_begin,
    decode_step,
    correctness,
    correctness_begin,
    correctness_step,
    phase_diagnostics,
    free_decode_begin,
    free_decode_run,
};

pub const SpecConfig = struct {
    mode: []const u8,
    mtp: ?std.json.Value = null,
    dflash: ?std.json.Value = null,
    dspark: ?std.json.Value = null,
};

pub const Request = struct {
    id: i64,
    kind: []const u8,
    prompt_tokens: ?[]const i64 = null,
    token: ?i64 = null,
    seed_tokens: ?[]const i64 = null,
    steps: ?i64 = null,
    count: ?u32 = null,
    spec: ?SpecConfig = null,

    pub fn resolvedKind(request: Request) error{UnknownKind}!Kind {
        return std.meta.stringToEnum(Kind, request.kind) orelse error.UnknownKind;
    }
};

pub const TopLogit = struct {
    token: i64,
    logit: f64,
};

pub const ExpertStats = struct {
    expert_cache_hits: u64 = 0,
    expert_cache_misses: u64 = 0,
    expert_cache_evictions: u64 = 0,
    expert_bytes_read: u64 = 0,
    expert_read_seconds: f64 = 0,
    expert_peak_cached_tensors: u64 = 0,
};

pub const EffectiveDflash = struct {
    depth: u8,
};

pub const EffectiveSpec = struct {
    mode: []const u8,
    dflash: ?EffectiveDflash = null,

    pub fn serial() EffectiveSpec {
        return .{ .mode = "serial" };
    }

    pub fn withDflash(depth: u8) EffectiveSpec {
        return .{ .mode = "dflash", .dflash = .{ .depth = depth } };
    }
};

pub const RunnerIdentity = struct {
    id: []const u8,
    model_type: []const u8,
    manifest_sha256: []const u8,
    build: []const u8,
};

pub const ResidentIdentity = struct {
    pid: u32,
    load_epoch: u64,
};

pub const Response = struct {
    id: i64,
    nonce: ?[]const u8 = null,
    ok: bool,
    @"error": ?[]const u8 = null,
    token: ?i64 = null,
    top_logits: ?[]const TopLogit = null,
    seed_token: ?i64 = null,
    tokens: ?[]const u32 = null,
    expert_stats: ?ExpertStats = null,
    peak_ram_gb: ?f64 = null,
    protocol_version: ?u32 = null,
    backend: ?[]const u8 = null,
    device: ?[]const u8 = null,
    completed_work: ?i64 = null,
    cache_memory: ?i64 = null,
    capabilities: ?[]const []const u8 = null,
    acceptance_lengths: ?[]const u32 = null,
    drafted_total: ?u64 = null,
    accepted_total: ?u64 = null,
    committed_total: ?u64 = null,
    effective_spec: ?EffectiveSpec = null,
    spec_modes: ?[]const []const u8 = null,
    spec_decoder: ?[]const u8 = null,
    verify_replay_disagreements: ?u64 = null,
    verification_mode: ?[]const u8 = null,
    rectangular_verification_rounds: ?u64 = null,
    serial_verification_rounds: ?u64 = null,
    runner: ?RunnerIdentity = null,
    resident: ?ResidentIdentity = null,

    pub fn success(id: i64, nonce: []const u8) Response {
        return .{ .id = id, .nonce = nonce, .ok = true };
    }

    pub fn failure(id: i64, nonce: []const u8, message: []const u8) Response {
        return .{ .id = id, .nonce = nonce, .ok = false, .@"error" = message };
    }
};

pub fn parseRequest(alloc: std.mem.Allocator, line: []const u8) !std.json.Parsed(Request) {
    var raw = try std.json.parseFromSlice(std.json.Value, alloc, line, .{
        .duplicate_field_behavior = .@"error",
    });
    defer raw.deinit();
    if (raw.value != .object) return error.UnexpectedToken;
    try validateRawRequest(raw.value.object);
    for (raw.value.object.values()) |value| if (value == .null) return error.UnexpectedToken;
    if (raw.value.object.get("spec")) |spec| {
        if (spec != .object) return error.UnexpectedToken;
        for (spec.object.values()) |value| if (value == .null) return error.UnexpectedToken;
    }
    return std.json.parseFromSlice(Request, alloc, line, .{
        .allocate = .alloc_always,
        .duplicate_field_behavior = .@"error",
    });
}

pub fn writeResponse(writer: *std.Io.Writer, response: Response) !void {
    try std.json.Stringify.value(response, .{ .emit_null_optional_fields = false }, writer);
    try writer.writeByte('\n');
    try writer.flush();
}

pub fn parseDflash(spec: SpecConfig) error{InvalidSpec}!u8 {
    if (!std.mem.eql(u8, spec.mode, "dflash") or spec.mtp != null or spec.dspark != null) return error.InvalidSpec;
    const block = spec.dflash orelse return error.InvalidSpec;
    if (block != .object) return error.InvalidSpec;
    for (block.object.keys()) |key| if (!std.mem.eql(u8, key, "depth")) return error.InvalidSpec;
    const depth = block.object.get("depth") orelse return 1;
    if (depth != .integer or depth.integer < 0 or depth.integer > std.math.maxInt(u8)) return error.InvalidSpec;
    return @intCast(depth.integer);
}

pub fn isSerial(spec: SpecConfig) bool {
    return std.mem.eql(u8, spec.mode, "serial") and spec.mtp == null and spec.dflash == null and spec.dspark == null;
}

pub fn checkedDiagnostic(value: u64) error{DiagnosticOutOfRange}!i64 {
    return std.math.cast(i64, value) orelse error.DiagnosticOutOfRange;
}

fn validateRawRequest(object: std.json.ObjectMap) !void {
    const integer_fields = [_][]const u8{ "id", "token", "steps", "count" };
    for (integer_fields) |key| if (object.get(key)) |value| {
        if (value != .integer) return error.UnexpectedToken;
    };

    if (object.get("kind")) |value| if (value != .string) return error.UnexpectedToken;
    const token_fields = [_][]const u8{ "prompt_tokens", "seed_tokens" };
    for (token_fields) |key| if (object.get(key)) |value| {
        if (value != .array) return error.UnexpectedToken;
        for (value.array.items) |item| if (item != .integer) return error.UnexpectedToken;
    };

    if (object.get("spec")) |value| {
        if (value != .object) return error.UnexpectedToken;
        if (value.object.get("mode")) |mode| if (mode != .string) return error.UnexpectedToken;
    }
}
