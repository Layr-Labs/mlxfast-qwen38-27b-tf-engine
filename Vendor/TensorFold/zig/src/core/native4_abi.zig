//! Fail-closed CPU parser for the native4 cooperative P-fragment ABI query.
const std = @import("std");

pub const lanes = 32;
pub const record_words = 36;
pub const header_words = 4;
pub const records_base = header_words;
pub const guard_words = 8;
pub const mode_run: u32 = 0;
pub const mode_query: u32 = 1;
pub const magic: u32 = 0x4E345041;
pub const version: u32 = 1;
pub const guard: u32 = 0xD3D3D3D3;
pub const poison: u32 = 0xDEADDEAD;
pub const invalid_coordinate: u32 = std.math.maxInt(u32);
pub const max_slots = 16;
const logical_rows = 8;
const logical_columns = 32;
const logical_elements = logical_rows * logical_columns;

pub fn configWords(scopes: usize) !usize {
    if (scopes == 0 or scopes > 32) return error.InvalidScopeCount;
    const records = try std.math.mul(usize, scopes, lanes);
    const body = try std.math.mul(usize, records, record_words);
    return try std.math.add(usize, try std.math.add(usize, header_words, body), guard_words);
}

/// Initialize a query buffer. Caller uploads it before dispatching the metadata-only query kernel.
pub fn initialize(words: []u32, scopes: usize) !void {
    if (words.len != try configWords(scopes)) return error.InvalidStorage;
    @memset(words, poison);
    words[0] = mode_query;
    words[1] = @intCast(scopes);
    words[2] = magic;
    words[3] = version;
    @memset(words[words.len - guard_words ..], guard);
}

/// Validate the complete query without modifying it. Different threadgroups may use different valid layouts.
pub fn validate(words: []const u32, scopes: usize) !void {
    if (words.len != try configWords(scopes)) return error.InvalidStorage;
    if (words[0] != mode_query or words[1] != @as(u32, @intCast(scopes)) or words[2] != magic or words[3] != version) return error.InvalidHeader;
    for (words[words.len - guard_words ..]) |v| if (v != guard) return error.CorruptGuard;

    for (0..scopes) |scope| {
        var seen: [logical_elements]bool = @splat(false);
        var covered: usize = 0;
        for (0..lanes) |lane| {
            const rec = record(words, scope, lane);
            if (rec[0] != @as(u32, @intCast(scope)) or rec[1] != @as(u32, @intCast(lane))) return error.InvalidRecordIdentity;
            const capacity = rec[2];
            const valid_mask = rec[3];
            if (capacity == poison or valid_mask == poison) return error.IncompleteRecord;
            if (valid_mask & ~@as(u32, 0xFFFF) != 0) return error.InvalidMask;
            const queried = @min(capacity, max_slots);
            const available_mask: u32 = if (queried == 0) 0 else if (queried == max_slots) 0xFFFF else (@as(u32, 1) << @intCast(queried)) - 1;
            if (valid_mask & ~available_mask != 0) return error.InvalidMask;
            for (0..max_slots) |slot| {
                const x = rec[4 + 2 * slot];
                const y = rec[5 + 2 * slot];
                if (slot >= queried or (valid_mask & (@as(u32, 1) << @intCast(slot))) == 0) {
                    if (x != invalid_coordinate or y != invalid_coordinate) return error.UnexpectedSlotData;
                    continue;
                }
                if (x == poison or y == poison or x == invalid_coordinate or y == invalid_coordinate) return error.IncompleteSlot;
                // Coordinates outside the logical 8x32 shape describe masked padding and do not cover outputs.
                if (x >= logical_columns or y >= logical_rows) continue;
                const logical = y * logical_columns + x;
                if (seen[logical]) return error.DuplicateCoordinate;
                seen[logical] = true;
                covered += 1;
            }
        }
        if (covered != logical_elements) return error.IncompleteCoverage;
    }
}

/// Normalize query records for the hot kernel only after every scope, slot and guard has passed validation.
/// x becomes the flattened logical M8xN32 output index; y is retained for the native output projection.
pub fn activate(words: []u32, scopes: usize) !void {
    try validate(words, scopes);
    for (0..scopes) |scope| for (0..lanes) |lane| {
        const rec = recordMut(words, scope, lane);
        const capacity = rec[2];
        const queried = @min(capacity, max_slots);
        const valid_mask = rec[3];
        var live_mask: u32 = 0;
        for (0..queried) |slot| {
            const bit = @as(u32, 1) << @intCast(slot);
            if (valid_mask & bit == 0) continue;
            const x = rec[4 + 2 * slot];
            const y = rec[5 + 2 * slot];
            if (x >= logical_columns or y >= logical_rows) continue;
            rec[4 + 2 * slot] = y * logical_columns + x;
            live_mask |= bit;
        }
        rec[3] = live_mask;
    };
    // Publish the hot mode only after all records are normalized.
    words[0] = mode_run;
}

fn record(words: []const u32, scope: usize, lane: usize) []const u32 {
    const base = records_base + (scope * lanes + lane) * record_words;
    return words[base..][0..record_words];
}

fn recordMut(words: []u32, scope: usize, lane: usize) []u32 {
    const base = records_base + (scope * lanes + lane) * record_words;
    return words[base..][0..record_words];
}

test "native4 ABI validates independent complete scope mappings including inactive and oversized capacity" {
    const n = try configWords(2);
    const words = try std.testing.allocator.alloc(u32, n);
    defer std.testing.allocator.free(words);
    try initialize(words, 2);
    writeCoverage(words, 0, false);
    writeCoverage(words, 1, true);
    try validate(words, 2);
    try activate(words, 2);
    try std.testing.expectEqual(mode_run, words[0]);
    for (0..2) |scope| for (0..lanes) |lane| {
        const rec = record(words, scope, lane);
        const capacity = rec[2];
        const queried = @min(capacity, max_slots);
        for (0..queried) |slot| {
            const bit = @as(u32, 1) << @intCast(slot);
            if (rec[3] & bit == 0) continue;
            const linear_index = rec[4 + 2 * slot];
            const y = rec[5 + 2 * slot];
            try std.testing.expect(linear_index < logical_elements);
            try std.testing.expectEqual(linear_index / logical_columns, y);
        }
    };
}

test "native4 ABI rejects malformed records without mutating query data" {
    const n = try configWords(1);
    const words = try std.testing.allocator.alloc(u32, n);
    defer std.testing.allocator.free(words);
    try initialize(words, 1);
    writeCoverage(words, 0, false);
    words[records_base + 2] = 4;
    const invalid = try std.testing.allocator.dupe(u32, words);
    defer std.testing.allocator.free(invalid);
    try std.testing.expectError(error.InvalidMask, validate(words, 1));
    try std.testing.expectEqualSlices(u32, invalid, words); // validation itself is read-only
    try std.testing.expectError(error.InvalidMask, activate(words, 1));
    try std.testing.expectEqualSlices(u32, invalid, words);
}

test "native4 ABI sabotage cases catch wrong identities, duplicate or missing coordinates and guard damage" {
    const n = try configWords(1);
    const words = try std.testing.allocator.alloc(u32, n);
    defer std.testing.allocator.free(words);

    try initialize(words, 1);
    writeCoverage(words, 0, false);
    recordMut(words, 0, 7)[1] = 9;
    try std.testing.expectError(error.InvalidRecordIdentity, validate(words, 1));

    try initialize(words, 1);
    writeCoverage(words, 0, false);
    const duplicate = recordMut(words, 0, 1);
    const first = record(words, 0, 0);
    duplicate[4] = first[4];
    duplicate[5] = first[5];
    try std.testing.expectError(error.DuplicateCoordinate, validate(words, 1));

    try initialize(words, 1);
    writeCoverage(words, 0, false);
    recordMut(words, 0, 0)[3] &= ~@as(u32, 1);
    try std.testing.expectError(error.UnexpectedSlotData, validate(words, 1));

    try initialize(words, 1);
    writeCoverage(words, 0, false);
    const partial = recordMut(words, 0, 0);
    partial[3] &= ~@as(u32, 1);
    partial[4] = invalid_coordinate;
    partial[5] = invalid_coordinate;
    try std.testing.expectError(error.IncompleteCoverage, validate(words, 1));

    try initialize(words, 1);
    writeCoverage(words, 0, false);
    recordMut(words, 0, 0)[4 + 2 * 8] = poison;
    try std.testing.expectError(error.UnexpectedSlotData, validate(words, 1));

    try initialize(words, 1);
    writeCoverage(words, 0, false);
    words[words.len - 1] = 0;
    try std.testing.expectError(error.CorruptGuard, validate(words, 1));
}

test "native4 ABI rejects bad extents and malformed headers" {
    try std.testing.expectError(error.InvalidScopeCount, configWords(0));
    try std.testing.expectError(error.InvalidScopeCount, configWords(33));
    const words = try std.testing.allocator.alloc(u32, try configWords(1));
    defer std.testing.allocator.free(words);
    try std.testing.expectError(error.InvalidStorage, initialize(words[0 .. words.len - 1], 1));
    try initialize(words, 1);
    writeCoverage(words, 0, false);
    words[2] ^= 1;
    try std.testing.expectError(error.InvalidHeader, validate(words, 1));
}

fn writeCoverage(words: []u32, scope: usize, oversized: bool) void {
    for (0..lanes) |lane| {
        const rec = recordMut(words, scope, lane);
        rec[0] = @intCast(scope);
        rec[1] = @intCast(lane);
        for (0..max_slots) |slot| {
            rec[4 + 2 * slot] = invalid_coordinate;
            rec[5 + 2 * slot] = invalid_coordinate;
        }
        const active_lanes: usize = if (oversized) 16 else 32;
        if (lane >= active_lanes) {
            rec[2] = 0;
            rec[3] = 0;
            continue;
        }
        const per_lane: usize = if (oversized) 16 else 8;
        rec[2] = if (oversized) 32 else 16;
        rec[3] = if (oversized) 0xFFFF else 0xFF;
        for (0..per_lane) |slot| {
            const logical = lane * per_lane + slot;
            rec[4 + 2 * slot] = @intCast(logical % logical_columns);
            rec[5 + 2 * slot] = @intCast(logical / logical_columns);
        }
    }
}
