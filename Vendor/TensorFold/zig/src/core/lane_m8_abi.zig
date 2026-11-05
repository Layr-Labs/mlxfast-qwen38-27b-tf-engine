//! Host-side validator for one-time M8/M16 cooperative-fragment ABI probes.
//! This module is pure host code; it does not import Metal or dispatch work.
const std = @import("std");

pub const lanes: usize = 32;
pub const record_words: usize = 494;
pub const header_words: usize = 4;
pub const map_base: usize = 4;
pub const map_words: usize = 32;
pub const records_base: usize = 36;
pub const guard_words: usize = 8;

pub const mode_run: u32 = 0;
pub const mode_probe: u32 = 1;
pub const magic: u32 = 0x4D384142;
pub const version: u32 = 1;
pub const dead: u32 = 0xDEADDEAD;
pub const guard: u32 = 0xD3D3D3D3;

const b8_base: usize = 10;
const b16_base: usize = 203;
const p8_base: usize = 396;
const p16_base: usize = 445;
const u32_max: u32 = std.math.maxInt(u32);

pub const Error = error{
    BadScopes,
    BadLength,
    BadRawLength,
    BadHeader,
    BadMode,
    BadCanary,
    BadSentinel,
    BadRecord,
    BadCapacity,
    BadStorage,
    BadCoordinate,
    BadCoverage,
    BadRhsMap,
    BadDestinationMap,
    BadModeMap,
};

/// Number of u32 words for the ABI probe configuration, including the result guard.
pub fn configWords(scopes: usize) Error!usize {
    if (scopes == 0 or scopes > lanes) return error.BadScopes;
    const records = std.math.mul(usize, scopes, lanes) catch return error.BadLength;
    const record_size = std.math.mul(usize, records, record_words) catch return error.BadLength;
    const with_records = std.math.add(usize, records_base, record_size) catch return error.BadLength;
    return std.math.add(usize, with_records, guard_words) catch return error.BadLength;
}

/// Fill the complete probe buffer with poison, then install its header and suffix canary.
pub fn initialize(words: []u32, scopes: usize) Error!void {
    const count = try configWords(scopes);
    if (words.len != count) return error.BadLength;
    @memset(words, dead);
    words[0] = mode_probe;
    words[1] = @intCast(scopes);
    words[2] = magic;
    words[3] = version;
    @memset(words[count - guard_words ..], guard);
}

/// Convert the original raw query output (little-endian 494-word records, no global header)
/// into the guarded config consumed by the candidate query mode.
pub fn initializeFromM4Raw(words: []u32, scopes: usize, raw: []const u8) Error!void {
    const count = try configWords(scopes);
    if (words.len != count) return error.BadLength;
    const records = std.math.mul(usize, scopes, lanes) catch return error.BadLength;
    const raw_words = std.math.mul(usize, records, record_words) catch return error.BadLength;
    const raw_bytes = std.math.mul(usize, raw_words, @sizeOf(u32)) catch return error.BadRawLength;
    if (raw.len != raw_bytes) return error.BadRawLength;
    try initialize(words, scopes);
    for (0..raw_words) |i| {
        words[records_base + i] = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
    }
}

/// Validate all lane/slot records and return one two-bit-per-output-slot map word per lane.
/// In probe mode the map area must still be poison. In run mode it must equal this validated map.
/// This function never modifies the supplied buffer.
pub fn validate(words: []const u32, scopes: usize) Error![lanes]u32 {
    const count = try configWords(scopes);
    if (words.len != count) return error.BadLength;
    if (words[1] != scopes or words[2] != magic or words[3] != version) return error.BadHeader;
    if (words[0] != mode_probe and words[0] != mode_run) return error.BadMode;
    for (words[count - guard_words ..]) |v| if (v != guard) return error.BadCanary;
    if (words[0] == mode_probe) {
        for (words[map_base..records_base]) |v| if (v != dead) return error.BadSentinel;
    }

    var lane_maps: [lanes]u32 = undefined;
    var first_header: ?[8]u32 = null;
    var first_records: [lanes][record_words]u32 = undefined;
    for (0..scopes) |sg| {
        var rhs8_seen: [2][64 * 32]bool = @splat(@splat(false));
        var rhs16_seen: [2][64 * 32]bool = @splat(@splat(false));
        var rhs8_duplicate: [2]bool = @splat(false);
        var rhs16_duplicate: [2]bool = @splat(false);
        for (0..lanes) |lane| {
            const rec_index = sg * lanes + lane;
            const base = records_base + rec_index * record_words;
            const rec = words[base..][0..record_words];
            if (rec[0] != sg or rec[1] != lane) return error.BadRecord;

            const header: [8]u32 = rec[2..10].*;
            if (first_header) |expected| {
                if (!std.mem.eql(u32, &header, &expected)) return error.BadCapacity;
            } else first_header = header;
            if (header[0] != 64 or header[1] != 64) return error.BadCapacity;
            if (header[2] != 128 or header[3] != 128) return error.BadStorage;
            if (header[4] < 8 or header[5] < 12)
                return error.BadCapacity;
            const p8_min_bytes = std.math.mul(u32, header[4], 4) catch return error.BadStorage;
            const p16_min_bytes = std.math.mul(u32, header[5], 4) catch return error.BadStorage;
            if (header[6] < p8_min_bytes or header[7] < p16_min_bytes) return error.BadStorage;

            try validateSlots(rec, b8_base, 64, header[0], .rhs, &rhs8_seen, &rhs8_duplicate);
            try validateSlots(rec, b16_base, 64, header[1], .rhs, &rhs16_seen, &rhs16_duplicate);
            try validateSlots(rec, p8_base, 16, header[4], .p8, null, null);
            try validateSlots(rec, p16_base, 16, header[5], .p16, null, null);
            if (!rhsEqual(rec)) return error.BadRhsMap;

            var encoded: u32 = 0;
            for (0..4) |j| {
                const mode = try destinationMode(rec, j);
                encoded |= @as(u32, mode) << @intCast(j * 2);
            }
            if (sg == 0) {
                lane_maps[lane] = encoded;
                first_records[lane] = rec[0..record_words].*;
            } else {
                // Every record field must agree across SIMD groups except its SG index.
                if (!std.mem.eql(u32, rec[1..], first_records[lane][1..])) return error.BadRhsMap;
                if (encoded != lane_maps[lane]) return error.BadDestinationMap;
            }
        }
        if (!completeRhs(&rhs8_seen, &rhs8_duplicate) or !completeRhs(&rhs16_seen, &rhs16_duplicate)) return error.BadCoverage;
    }
    if (words[0] == mode_run) {
        if (!std.mem.eql(u32, words[map_base..records_base], &lane_maps)) return error.BadModeMap;
    }
    return lane_maps;
}

/// Validate the entire probe image first; only then publish maps and switch it to run mode.
pub fn activate(words: []u32, scopes: usize) Error!void {
    const result = try validate(words, scopes);
    @memcpy(words[map_base..records_base], &result);
    words[0] = mode_run;
}

const SlotKind = enum { rhs, p8, p16 };

fn validateSlots(
    rec: []const u32,
    offset: usize,
    limit: usize,
    expected_cap: u32,
    kind: SlotKind,
    seen: ?* [2][64 * 32]bool,
    duplicate: ?* [2]bool,
) Error!void {
    if (rec[offset] != expected_cap) return error.BadCapacity;
    for (0..limit) |i| {
        const at = offset + 1 + 3 * i;
        const valid = rec[at];
        const x = rec[at + 1];
        const y = rec[at + 2];
        if (valid > 1) return error.BadRecord;
        if (valid == 0) {
            if (x != u32_max or y != u32_max) return error.BadSentinel;
            continue;
        }
        if (i >= expected_cap) return error.BadCoordinate;
        switch (kind) {
            .rhs => {
                // Require a coordinate in at least one allowed full-tile orientation.
                if (!((x < 64 and y < 32) or (x < 32 and y < 64))) return error.BadCoordinate;
                const seen_map = seen orelse return error.BadCoverage;
                const duplicate_map = duplicate orelse return error.BadCoverage;
                if (x < 64 and y < 32) {
                    const index: usize = @as(usize, y) * 64 + x;
                    if (seen_map[0][index]) duplicate_map[0] = true;
                    seen_map[0][index] = true;
                }
                if (x < 32 and y < 64) {
                    const index: usize = @as(usize, y) * 32 + x;
                    if (seen_map[1][index]) duplicate_map[1] = true;
                    seen_map[1][index] = true;
                }
            },
            .p8 => if (x >= 32 or y >= 8) return error.BadCoordinate,
            .p16 => if (x >= 32 or y >= 16) return error.BadCoordinate,
        }
    }
}

fn completeTile(seen: *const [64 * 32]bool) bool {
    for (seen) |v| if (!v) return false;
    return true;
}

fn completeRhs(seen: *const [2][64 * 32]bool, duplicate: *const [2]bool) bool {
    return (!duplicate[0] and completeTile(&seen[0])) or (!duplicate[1] and completeTile(&seen[1]));
}

fn rhsEqual(rec: []const u32) bool {
    for (0..64) |i| {
        const a = b8_base + 1 + i * 3;
        const b = b16_base + 1 + i * 3;
        if (rec[a] != 1 or rec[b] != 1 or rec[a + 1] != rec[b + 1] or rec[a + 2] != rec[b + 2]) return false;
    }
    return true;
}

fn destinationMode(rec: []const u32, j: usize) Error!u32 {
    const lo8 = p8_base + 1 + j * 3;
    const lo16 = p16_base + 1 + j * 3;
    const hi16 = p16_base + 1 + (8 + j) * 3;
    if (rec[lo8] != 1 or rec[lo16] != 1 or rec[lo8 + 1] != rec[lo16 + 1] or rec[lo8 + 2] != rec[lo16 + 2])
        return error.BadDestinationMap;
    if (rec[hi16] != 1) return error.BadDestinationMap;
    const hx = rec[hi16 + 1];
    const hy = rec[hi16 + 2];
    if (hy >= 8) return 0; // padding in the M8 destination
    const compact = p8_base + 1 + (4 + j) * 3;
    if (compact < p8_base + 1 + rec[p8_base] * 3 and rec[compact] == 1 and
        rec[compact + 1] == hx and rec[compact + 2] == hy) return 1;
    const legacy = p8_base + 1 + (8 + j) * 3;
    if (8 + j < 16 and 8 + j < rec[p8_base] and rec[legacy] == 1 and
        rec[legacy + 1] == hx and rec[legacy + 2] == hy) return 2;
    return error.BadDestinationMap;
}

const HighMode = enum { padded, compact, legacy };

fn putSlot(rec: []u32, base: usize, index: usize, valid: bool, x: u32, y: u32) void {
    const at = base + 1 + index * 3;
    rec[at] = @intFromBool(valid);
    rec[at + 1] = if (valid) x else u32_max;
    rec[at + 2] = if (valid) y else u32_max;
}

fn makeRecords(words: []u32, scopes: usize, modes: [4]HighMode, transpose_rhs: bool) !void {
    try initialize(words, scopes);
    for (0..scopes) |sg| for (0..lanes) |lane| {
        const base = records_base + (sg * lanes + lane) * record_words;
        const rec = words[base..][0..record_words];
        rec[0] = @intCast(sg);
        rec[1] = @intCast(lane);
        rec[2] = 64;
        rec[3] = 64;
        rec[4] = 128;
        rec[5] = 128;
        rec[6] = 16;
        rec[7] = 16;
        rec[8] = 64;
        rec[9] = 64;
        rec[b8_base] = 64;
        rec[b16_base] = 64;
        rec[p8_base] = 16;
        rec[p16_base] = 16;
        for (0..64) |i| {
            const global = lane * 64 + i;
            const k: u32 = @intCast(global / 32);
            const n: u32 = @intCast(global % 32);
            const x = if (transpose_rhs) n else k;
            const y = if (transpose_rhs) k else n;
            putSlot(rec, b8_base, i, true, x, y);
            putSlot(rec, b16_base, i, true, x, y);
        }
        for (0..16) |i| {
            putSlot(rec, p8_base, i, true, @intCast(i), 0);
            putSlot(rec, p16_base, i, true, @intCast(i), 0);
        }
        for (0..4) |j| {
            switch (modes[j]) {
                .padded => putSlot(rec, p16_base, 8 + j, true, @intCast(j), @intCast(8 + j)),
                .compact => putSlot(rec, p16_base, 8 + j, true, @intCast(4 + j), 0),
                .legacy => putSlot(rec, p16_base, 8 + j, true, @intCast(8 + j), 0),
            }
        }
    };
}

fn configAlloc(a: std.mem.Allocator, scopes: usize) ![]u32 {
    return try a.alloc(u32, try configWords(scopes));
}

test "config arithmetic, initialization header, poison and exact guard" {
    try std.testing.expectEqual(@as(usize, 36 + 32 * 494 + 8), try configWords(1));
    try std.testing.expectEqual(@as(usize, 36 + 32 * 32 * 494 + 8), try configWords(32));
    try std.testing.expectError(error.BadScopes, configWords(0));
    try std.testing.expectError(error.BadScopes, configWords(33));
    const a = std.testing.allocator;
    const words = try configAlloc(a, 4);
    defer a.free(words);
    try initialize(words, 4);
    try std.testing.expectEqual(mode_probe, words[0]);
    try std.testing.expectEqual(@as(u32, 4), words[1]);
    try std.testing.expectEqual(magic, words[2]);
    try std.testing.expectEqual(version, words[3]);
    for (words[map_base..records_base]) |v| try std.testing.expectEqual(dead, v);
    for (words[words.len - guard_words ..]) |v| try std.testing.expectEqual(guard, v);
    try std.testing.expectError(error.BadLength, initialize(words[0 .. words.len - 1], 4));
}

test "M4-shaped raw record import and padded map activate only after validation" {
    const a = std.testing.allocator;
    const source = try configAlloc(a, 4);
    defer a.free(source);
    try makeRecords(source, 4, .{ .padded, .padded, .padded, .padded }, false);
    const raw_words = 4 * lanes * record_words;
    const raw = try a.alloc(u8, raw_words * 4);
    defer a.free(raw);
    for (0..raw_words) |i| std.mem.writeInt(u32, raw[i * 4 ..][0..4], source[records_base + i], .little);

    const imported = try configAlloc(a, 4);
    defer a.free(imported);
    try initializeFromM4Raw(imported, 4, raw);
    const before = try a.dupe(u32, imported);
    defer a.free(before);
    const map = try validate(imported, 4);
    for (map) |v| try std.testing.expectEqual(@as(u32, 0), v);
    try std.testing.expectEqualSlices(u32, before, imported); // validate is read-only
    try activate(imported, 4);
    try std.testing.expectEqual(mode_run, imported[0]);
    try std.testing.expectEqualSlices(u32, &map, imported[map_base..records_base]);
    try std.testing.expectEqualSlices(u32, &map, &try validate(imported, 4));
    try std.testing.expectError(error.BadRawLength, initializeFromM4Raw(imported, 4, raw[0 .. raw.len - 1]));
}

test "compact and legacy destinations encode two-bit modes per lane" {
    const a = std.testing.allocator;
    const words = try configAlloc(a, 4);
    defer a.free(words);
    try makeRecords(words, 4, .{ .padded, .compact, .legacy, .padded }, true);
    const map = try validate(words, 4);
    const expected: u32 = 0 | (1 << 2) | (2 << 4) | (0 << 6);
    for (map) |v| try std.testing.expectEqual(expected, v);
}

test "capacity, storage, sentinels, coordinates, coverage and SIMDgroup mutations fail closed" {
    const a = std.testing.allocator;
    const words = try configAlloc(a, 4);
    defer a.free(words);
    try makeRecords(words, 4, .{ .padded, .padded, .padded, .padded }, false);
    const base = records_base;
    try std.testing.expectError(error.BadCapacity, mutateAndValidate(words, 4, base + 2, 32));
    try std.testing.expectError(error.BadStorage, mutateAndValidate(words, 4, base + 4, 127));
    try std.testing.expectError(error.BadCapacity, mutateAndValidate(words, 4, base + b8_base, 32));
    try std.testing.expectError(error.BadSentinel, mutateAndValidate(words, 4, base + b8_base + 1, 0));
    try std.testing.expectError(error.BadCoordinate, mutateAndValidate(words, 4, base + b8_base + 2, 64));
    try std.testing.expectError(error.BadCoverage, mutatePairAndValidate(words, 4, base + b8_base + 1 + 3 + 2, base + b16_base + 1 + 3 + 2, 0));
    // A corrupted map in the second SIMD group violates the byte-identical per-lane map rule.
    const sg1lane0 = records_base + lanes * record_words;
    try std.testing.expectError(error.BadRhsMap, mutateAndValidate(words, 4, sg1lane0 + b8_base + 1 + 2, 1));
}

fn mutateAndValidate(words: []u32, scopes: usize, index: usize, value: u32) Error!void {
    const old = words[index];
    words[index] = value;
    defer words[index] = old;
    _ = try validate(words, scopes);
}

fn mutatePairAndValidate(words: []u32, scopes: usize, first_index: usize, second_index: usize, value: u32) Error!void {
    const first_saved = words[first_index];
    const second_saved = words[second_index];
    words[first_index] = value;
    words[second_index] = value;
    defer {
        words[first_index] = first_saved;
        words[second_index] = second_saved;
    }
    _ = try validate(words, scopes);
}

test "unknown M8 destination relation, header, mode, and canary are rejected without activation" {
    const a = std.testing.allocator;
    const words = try configAlloc(a, 4);
    defer a.free(words);
    try makeRecords(words, 4, .{ .padded, .padded, .padded, .padded }, false);
    const bad_slot = records_base + p16_base + 1 + 8 * 3 + 2; // old high row, made live but unmapped
    const old = words[bad_slot];
    words[bad_slot] = 0;
    const before = try a.dupe(u32, words);
    defer a.free(before);
    try std.testing.expectError(error.BadDestinationMap, activate(words, 4));
    try std.testing.expectEqualSlices(u32, before, words); // failed activate must not publish or flip
    words[bad_slot] = old;
    try std.testing.expectError(error.BadHeader, mutateAndValidate(words, 4, 2, 0));
    try std.testing.expectError(error.BadMode, mutateAndValidate(words, 4, 0, 9));
    try std.testing.expectError(error.BadCanary, mutateAndValidate(words, 4, words.len - 1, 0));
    try activate(words, 4);
    try std.testing.expectError(error.BadModeMap, mutateAndValidate(words, 4, map_base, 1));
}
