//! Pure policy, canonical half-carrier packing and byte offsets for the optional prompt GU.
const std = @import("std");
pub const Mode = @import("native_gu_startup_policy.zig").Mode;
pub const n: u32 = 34816;
pub const k: u32 = 5120;
pub const columns: u32 = n / 2;
pub const groups: u32 = k / 64;
pub const bytes: usize = @as(usize, n) * k * 2;
pub const up_word_offset: usize = bytes / 2;
pub const up_pair_offset: usize = @as(usize, columns) * 4;
pub const Part = struct { words: []const u32, rows: usize };
pub const Params = extern struct { rows: u32, x_stride: u32, sb_stride: u32, y_stride: u32, sum_stride: u32, tiles: u32 };

pub fn enabled(text: ?[]const u8, disabled: bool) bool {
    return !disabled and if (text) |v| std.mem.eql(u8, v, "1") else true;
}
/// Explicit local opt-out and unsupported models/hardware retain the original path.
pub fn requestedMode(text: ?[]const u8, disabled: bool, units: bool, target: bool) Mode {
    return if (units and target and enabled(text, disabled)) .adaptive else .off;
}
pub fn mustRun(mode: Mode, units: bool, out: u32, inner: u32) bool {
    return mode == .required and units and out == n and inner == k;
}
pub fn requireAdmission(mode: Mode, admitted: bool) !void {
    if (mode == .required and !admitted) return error.NativeGuRequired;
}
pub fn requireRoute(mode: Mode, units: bool, out: u32, inner: u32, work: bool, encoded: bool) !void {
    if (work and mustRun(mode, units, out, inner) and !encoded) return error.NativeGuRequiredRoute;
}
pub fn requireEngaged(mode: Mode, work: bool, before: u64, after: u64) !void {
    if (mode == .required and work and after <= before) return error.NativeGuNotEngaged;
}

pub fn admit(requested: bool, units: bool, out: usize, inner: usize, bits: u8, group: usize, equal_halves: bool) bool {
    return requested and units and out == n and inner == k and bits == 4 and group == 64 and equal_halves;
}
pub fn route(present: bool, units: bool, rows: u32, out: u32, inner: u32, bits: u8, group: u32, tile: u32, runs: usize, raw: bool) bool {
    return present and units and rows > 0 and out == n and inner == k and bits == 4 and group == 64 and tile == 32 and runs == 0 and !raw;
}

/// [tile64][group64][column64][64 raw half words]. qbits=0..15 represents q*2^-24.
/// Read original checkpoint U4 words directly. Never cast q numerically to f16.
pub fn pack(out: []u8, parts: []const Part, cols: usize, inner: usize) !void {
    if (cols == 0 or inner == 0 or cols % 64 != 0 or inner % 64 != 0) return error.BadNativeGUShape;
    const elements = try std.math.mul(usize, cols, inner);
    if (out.len != try std.math.mul(usize, elements, 2)) return error.BadNativeGUShape;
    var total: usize = 0;
    for (parts) |part| {
        if (part.words.len != try std.math.mul(usize, part.rows, inner / 8)) return error.BadNativeGUShape;
        total = try std.math.add(usize, total, part.rows);
    }
    if (total != cols) return error.BadNativeGUShape;
    var base: usize = 0;
    for (parts) |part| {
        for (0..part.rows) |r| for (0..inner / 64) |g| {
            const row = base + r;
            const to = (((row / 64) * (inner / 64) + g) * 64 + row % 64) * 64 * 2;
            for (0..64) |j| {
                const word = part.words[r * (inner / 8) + g * 8 + j / 8];
                const q: u16 = @intCast((word >> @as(u5, @intCast(4 * (j % 8)))) & 15);
                std.mem.writeInt(u16, out[to + 2 * j ..][0..2], q, .little);
            }
        };
        base += part.rows;
    }
}
pub fn promptParams(chunk_rows: u32, original_stride: u32) Params {
    return .{ .rows = chunk_rows, .x_stride = k, .sb_stride = n, .y_stride = columns, .sum_stride = original_stride, .tiles = n / 128 };
}
pub const Chunk = struct { rows: u32, x: usize, y: usize, sums: usize };
pub fn chunk(first: u32, rows: u32) Chunk {
    return .{ .rows = @min(rows -| first, 128), .x = @as(usize, first) * k * 2, .y = @as(usize, first) * columns * 2, .sums = @as(usize, first) * 4 };
}

test "native GU requests device qualification by default and preserves explicit opt-out" {
    const t = std.testing;
    try t.expect(enabled(null, false));
    try t.expect(!enabled(null, true));
    try t.expect(enabled("1", false));
    try t.expect(!enabled("1", true));
    for ([_][]const u8{ "0", "", "true", "2" }) |v| try t.expect(!enabled(v, false));
    try t.expect(admit(true, true, n, k, 4, 64, true));
    try t.expect(!admit(false, true, n, k, 4, 64, true));
    try t.expect(!admit(true, false, n, k, 4, 64, true));
    try t.expect(!admit(true, true, n, k, 4, 64, false));
    try t.expect(!admit(true, true, n - 1, k, 4, 64, true));
    try t.expect(!admit(true, true, n, k - 64, 4, 64, true));
    try t.expect(!admit(true, true, n, k, 2, 64, true));
    try t.expect(!admit(true, true, n, k, 4, 32, true));
    try t.expect(route(true, true, 361, n, k, 4, 64, 32, 0, false));
    try t.expect(!route(false, true, 361, n, k, 4, 64, 32, 0, false));
    try t.expect(!route(true, false, 361, n, k, 4, 64, 32, 0, false));
    try t.expect(!route(true, true, 0, n, k, 4, 64, 32, 0, false));
    try t.expect(!route(true, true, 361, n, k, 4, 64, 32, 1, false));
    try t.expect(!route(true, true, 361, n, k, 4, 64, 32, 0, true));
}

test "HALF GU pack preserves every raw carrier nibble across tiles groups and stack seam" {
    const a = std.testing.allocator;
    const cols = 128;
    const inner = 5120;
    const source = try a.alloc(u32, cols * inner / 8);
    defer a.free(source);
    @memset(source, 0);
    for (0..cols) |row| for (0..inner) |col| {
        const code: u32 = @intCast((row * 7 + col * 11 + row * col % 13) & 15);
        source[row * (inner / 8) + col / 8] |= code << @as(u5, @intCast((col % 8) * 4));
    };
    const storage = try a.alloc(u8, cols * inner * 2 + 32);
    defer a.free(storage);
    @memset(storage, 0xa5);
    const dst = storage[16..][0 .. cols * inner * 2];
    try pack(dst, &.{ .{ .words = source[0 .. 48 * inner / 8], .rows = 48 }, .{ .words = source[48 * inner / 8 ..], .rows = 80 } }, cols, inner);
    var codes: u16 = 0;
    // Independent inverse of flat half storage; compare RAW u16, not f16 equality.
    for (0..cols * inner) |at| {
        const row = at / (64 * 64 * (inner / 64)) * 64 + (at / 64) % 64;
        const col = (at / (64 * 64)) % (inner / 64) * 64 + at % 64;
        const expected: u16 = @intCast((row * 7 + col * 11 + row * col % 13) & 15);
        const got = std.mem.readInt(u16, dst[at * 2 ..][0..2], .little);
        try std.testing.expectEqual(expected, got);
        codes |= @as(u16, 1) << @as(u4, @intCast(got));
    }
    try std.testing.expectEqual(@as(u16, 0xFFFF), codes);
    for (storage[0..16]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
    for (storage[storage.len - 16 ..]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
    try std.testing.expectError(error.BadNativeGUShape, pack(dst[1..], &.{.{ .words = source, .rows = cols }}, cols, inner));
    try std.testing.expectError(error.BadNativeGUShape, pack(dst, &.{.{ .words = source[1..], .rows = cols }}, cols, inner));
    try std.testing.expectEqual(@as(usize, 356515840), bytes);
    try std.testing.expectEqual(@as(usize, 178257920), up_word_offset);
    try std.testing.expectError(error.Overflow, pack(&.{}, &.{}, std.math.maxInt(usize) - 63, 64));
}

test "native GU prompt chunk offsets retain global sums stride and tail bounds" {
    const t = std.testing;
    const middle = chunk(128, 361);
    const tail = chunk(256, 361);
    try t.expectEqual(@as(u32, 128), middle.rows);
    try t.expectEqual(@as(usize, 128 * 5120 * 2), middle.x);
    try t.expectEqual(@as(usize, 128 * 17408 * 2), middle.y);
    try t.expectEqual(@as(usize, 128 * 4), middle.sums);
    try t.expectEqual(@as(u32, 105), tail.rows);
    try t.expectEqual(@as(usize, 256 * 4), tail.sums);
    try t.expectEqual(@as(u32, 0), chunk(384, 361).rows);
    const params = promptParams(tail.rows, 361);
    try t.expectEqual(@as(u32, 105), params.rows);
    try t.expectEqual(@as(u32, 361), params.sum_stride);
    try t.expectEqual(@as(u32, 272), params.tiles);
    try t.expectEqual(@as(usize, 24), @sizeOf(Params));
    try t.expectEqual(@as(usize, 178257920), up_word_offset);
    try t.expectEqual(@as(usize, 69632), up_pair_offset);
}

/// Only before frames/residency exist and after every startup command buffer is drained.
/// Duplicate optional references and duplicate owned IDs are cleared/released exactly once.
pub fn releaseOwned(layers: anytype, owned: anytype) void {
    for (layers) |*layer| {
        const ref = layer.gu.native_gu orelse continue;
        const id = ref.buffer.id;
        var original = false;
        for (layers) |*other| {
            original = original or other.gu.words.buffer.id == id or other.gu.pairs.buffer.id == id;
            if (other.gu.native_gu) |value| if (value.buffer.id == id) {
                other.gu.native_gu = null;
            };
        }
        // A malformed optional alias must never release original matrix/metadata storage.
        if (original) continue;
        var released = false;
        var i: usize = 0;
        while (i < owned.items.len) {
            if (owned.items[i].id == id) {
                const buffer = owned.orderedRemove(i);
                if (!released) {
                    buffer.deinit();
                    released = true;
                }
            } else i += 1;
        }
    }
}

test "native GU actual optional ownership release clears aliases once and preserves originals" {
    const B = struct {
        id: u32,
        calls: *[4]u32,
        count: *usize,
        fn deinit(b: @This()) void {
            b.calls[b.count.*] = b.id;
            b.count.* += 1;
        }
    };
    const R = struct { buffer: B };
    const L = struct { gu: struct { words: R, pairs: R, native_gu: ?R } };
    var calls: [4]u32 = @splat(0);
    var count: usize = 0;
    const side = B{ .id = 10, .calls = &calls, .count = &count };
    const words = B{ .id = 20, .calls = &calls, .count = &count };
    const pairs = B{ .id = 30, .calls = &calls, .count = &count };
    const unrelated = B{ .id = 40, .calls = &calls, .count = &count };
    const borrowed = B{ .id = 99, .calls = &calls, .count = &count };
    var layers = [_]L{
        .{ .gu = .{ .words = .{ .buffer = words }, .pairs = .{ .buffer = pairs }, .native_gu = .{ .buffer = side } } },
        .{ .gu = .{ .words = .{ .buffer = words }, .pairs = .{ .buffer = pairs }, .native_gu = .{ .buffer = side } } },
        .{ .gu = .{ .words = .{ .buffer = words }, .pairs = .{ .buffer = pairs }, .native_gu = .{ .buffer = borrowed } } },
        .{ .gu = .{ .words = .{ .buffer = words }, .pairs = .{ .buffer = pairs }, .native_gu = .{ .buffer = words } } },
    };
    var owned: std.ArrayList(B) = .empty;
    defer owned.deinit(std.testing.allocator);
    try owned.appendSlice(std.testing.allocator, &.{ side, words, side, pairs, unrelated });
    releaseOwned(&layers, &owned);
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expectEqual(@as(u32, 10), calls[0]);
    try std.testing.expectEqual(@as(usize, 3), owned.items.len);
    for (layers) |l| try std.testing.expect(l.gu.native_gu == null);
    for (owned.items, [_]u32{ 20, 30, 40 }) |value, id| try std.testing.expectEqual(id, value.id);
    releaseOwned(&layers, &owned);
    try std.testing.expectEqual(@as(usize, 1), count);
}

test "native GU optional load request preserves guards and admitted mode requires routes" {
    try std.testing.expectEqual(Mode.adaptive, requestedMode(null, false, true, true));
    try std.testing.expectEqual(Mode.adaptive, requestedMode("1", false, true, true));
    try std.testing.expectEqual(Mode.off, requestedMode("0", false, true, true));
    try std.testing.expectEqual(Mode.off, requestedMode(null, true, true, true));
    try std.testing.expectEqual(Mode.off, requestedMode(null, false, false, true));
    try std.testing.expectEqual(Mode.off, requestedMode(null, false, true, false));
    try requireAdmission(.required, true);
    try std.testing.expectError(error.NativeGuRequired, requireAdmission(.required, false));
    try requireAdmission(.adaptive, false);
    try requireRoute(.required, true, n, k, true, true);
    try std.testing.expectError(error.NativeGuRequiredRoute, requireRoute(.required, true, n, k, true, false));
    try requireRoute(.required, false, n, k, true, false);
    try requireRoute(.off, true, n, k, true, false);
    try requireRoute(.adaptive, true, n, k, true, false);
    try requireRoute(.required, true, 5120, 5120, true, false);
    try requireRoute(.required, true, n, k, false, false);
}
test "native GU actual prompt engagement rejects missing saturated and wrapped counters" {
    try requireEngaged(.required, true, 7, 8);
    try std.testing.expectError(error.NativeGuNotEngaged, requireEngaged(.required, true, 7, 7));
    try std.testing.expectError(error.NativeGuNotEngaged, requireEngaged(.required, true, std.math.maxInt(u64), 0));
    try std.testing.expectError(error.NativeGuNotEngaged, requireEngaged(.required, true, std.math.maxInt(u64), std.math.maxInt(u64)));
    try requireEngaged(.off, true, 7, 7);
    try requireEngaged(.adaptive, true, 7, 7);
    try requireEngaged(.required, false, 7, 7);
}

/// Model-open finalization before glue/Frame/residency may borrow optional buffers.
/// A complete admission adopts required routing; rejection releases every sidecar.
/// Stock model/weight errors never pass through this optional-only transition.
pub fn finishAdmission(kernels: anytype, weights: anytype) void {
    if (kernels.native_gu_mode != .adaptive) return;
    const accepted = kernels.native_gu_status == .native and kernels.native_gu_pipe != null and kernels.native_gu_abi != null;
    if (accepted) {
        kernels.native_gu_mode = .required;
    } else {
        kernels.dropNativeGU();
        weights.releaseNativeGU();
        kernels.native_gu_mode = .off;
    }
}

const AdmissionMock = struct {
    native_gu_mode: Mode = .adaptive,
    native_gu_status: @import("native_gu_startup_policy.zig").Status = .selection,
    native_gu_pipe: ?u32 = 7,
    native_gu_abi: ?u32 = 8,
    pipeline_releases: usize = 0,
    abi_releases: usize = 0,
    fn dropNativeGU(m: *@This()) void {
        if (m.native_gu_pipe != null) m.pipeline_releases += 1;
        if (m.native_gu_abi != null) m.abi_releases += 1;
        m.native_gu_pipe = null;
        m.native_gu_abi = null;
    }
};
const AdmissionWeightsMock = struct {
    sidecars: usize = 64,
    releases: usize = 0,
    fn releaseNativeGU(m: *@This()) void {
        m.releases += m.sidecars;
        m.sidecars = 0;
    }
};

test "optional SG1 admission failures release every sidecar PSO and ABI before fallback" {
    const policy = @import("native_gu_startup_policy.zig");
    for (std.meta.tags(policy.Status)) |status| {
        if (status == .native) continue;
        var owner: AdmissionMock = .{ .native_gu_status = status };
        var weights: AdmissionWeightsMock = .{};
        finishAdmission(&owner, &weights);
        try std.testing.expectEqual(Mode.off, owner.native_gu_mode);
        try std.testing.expect(owner.native_gu_pipe == null and owner.native_gu_abi == null);
        try std.testing.expectEqual(@as(usize, 1), owner.pipeline_releases);
        try std.testing.expectEqual(@as(usize, 1), owner.abi_releases);
        try std.testing.expectEqual(@as(usize, 0), weights.sidecars);
        try std.testing.expectEqual(@as(usize, 64), weights.releases);
        finishAdmission(&owner, &weights);
        try std.testing.expectEqual(@as(usize, 1), owner.pipeline_releases);
        try std.testing.expectEqual(@as(usize, 1), owner.abi_releases);
        try std.testing.expectEqual(@as(usize, 64), weights.releases);
    }
}

test "optional SG1 prepare failure and partial sidecar allocations are discarded once" {
    // Prepare may fail before publishing either handle; shape/pack failure may
    // leave any prefix of the 64 sidecars. The same finalizer drains both states.
    for (0..65) |allocated| {
        var owner: AdmissionMock = .{ .native_gu_status = .shape };
        if (allocated == 0) {
            owner.native_gu_status = .unavailable;
            owner.native_gu_pipe = null;
            owner.native_gu_abi = null;
        }
        var weights: AdmissionWeightsMock = .{ .sidecars = allocated };
        finishAdmission(&owner, &weights);
        try std.testing.expectEqual(Mode.off, owner.native_gu_mode);
        try std.testing.expectEqual(@as(usize, 0), weights.sidecars);
        try std.testing.expectEqual(allocated, weights.releases);
        finishAdmission(&owner, &weights);
        try std.testing.expectEqual(allocated, weights.releases);
    }
}

test "optional SG1 missing ABI or pipeline cannot retain sidecars or activate native" {
    for ([_]bool{ false, true }) |missing_abi| {
        var owner: AdmissionMock = .{ .native_gu_status = .native };
        if (missing_abi) owner.native_gu_abi = null else owner.native_gu_pipe = null;
        var weights: AdmissionWeightsMock = .{};
        finishAdmission(&owner, &weights);
        try std.testing.expectEqual(Mode.off, owner.native_gu_mode);
        try std.testing.expect(owner.native_gu_pipe == null and owner.native_gu_abi == null);
        try std.testing.expectEqual(@as(usize, 0), weights.sidecars);
        try std.testing.expectEqual(@as(usize, 64), weights.releases);
    }
}

test "adopted SG1 path retains resources and rejects silent fallback and missing hits" {
    var owner: AdmissionMock = .{ .native_gu_status = .native };
    var weights: AdmissionWeightsMock = .{};
    finishAdmission(&owner, &weights);
    try std.testing.expectEqual(Mode.required, owner.native_gu_mode);
    try std.testing.expect(owner.native_gu_pipe != null and owner.native_gu_abi != null);
    try std.testing.expectEqual(@as(usize, 64), weights.sidecars);
    try std.testing.expectEqual(@as(usize, 0), weights.releases);
    finishAdmission(&owner, &weights);
    try std.testing.expectEqual(Mode.required, owner.native_gu_mode);
    try std.testing.expectEqual(@as(usize, 0), owner.pipeline_releases);
    try std.testing.expectEqual(@as(usize, 0), owner.abi_releases);
    try requireRoute(owner.native_gu_mode, true, n, k, true, true);
    try std.testing.expectError(error.NativeGuRequiredRoute, requireRoute(owner.native_gu_mode, true, n, k, true, false));
    try requireEngaged(owner.native_gu_mode, true, 10, 11);
    try std.testing.expectError(error.NativeGuNotEngaged, requireEngaged(owner.native_gu_mode, true, 10, 10));
}
