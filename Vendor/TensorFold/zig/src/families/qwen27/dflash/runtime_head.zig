//! Whole-tile draft vocabulary spans preserve the borrowed target head's arithmetic and map columns back to token IDs.
const std = @import("std");
const mtl = @import("metal");
const projection = @import("../projection.zig");
const lane = @import("core").lane_projection;
pub const spans = [_][2]u32{ .{ 0, 98304 }, .{ 248032, 248320 } };
pub const width = 98592;
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
pub const Head = struct {
    allocator: std.mem.Allocator,
    linear: projection.Linear,
    ids: ?mtl.Buffer = null,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, source: projection.Linear) !Head {
        var head = Head{ .allocator = a, .linear = source };
        errdefer head.deinit();
        if (source.n != 248320) return head;
        if (source.tile != 32 or source.k % source.group != 0 or source.runs.len > 0) return error.BadDraftHead;
        const k: usize = source.k;
        const row = k * source.bits / 8; // whole 32-row tiles in the lane layout, rows in the raw one
        const input_words = if (source.raw) |raw| raw.w else source.words.buffer;
        const input_offset = if (source.raw) |raw| raw.w_off else source.words.offset;
        const groups = k / source.group;
        // The borrowed head's rows land in a copy of their own, and a draft round's one row reads it once a round: it is
        // the largest decode matrix left in the unbatched order, so its code blocks are stored in the same eight-tile
        // interleave every other decoded shape reads (projection.prepareForced) — whole tiles of one group at a time, a
        // block a tile's 32 lanes of one group's codes. The raw (row-major) copy keeps its rows.
        const lane_layout = source.raw == null;
        const batch: usize = if (lane_layout) 8 else 1;
        const sb: usize = source.batch;
        const block = @as(usize, source.tile) * 8 * 4;
        const stored = if (lane_layout) projection.storedColumns(width, @intCast(batch)) else width;
        const words = try head.allocate(device, stored * row);
        var destination: usize = 0;
        if (lane_layout) {
            var at: usize = 0; // the copy's own tile index, in span order
            for (spans) |span| {
                for (0..(span[1] - span[0]) / source.tile) |i| {
                    const st: usize = span[0] / source.tile + i;
                    for (0..groups) |g| {
                        const from = input_offset + (((st / sb) * groups + g) * sb + st % sb) * block;
                        const to = (((at / batch) * groups + g) * batch + at % batch) * block;
                        @memcpy(words.contents()[to..][0..block], input_words.contents()[from..][0..block]);
                    }
                    at += 1;
                }
            }
        } else for (spans) |span| {
            const rows = span[1] - span[0];
            const first = input_offset + @as(usize, span[0]) * row;
            @memcpy(words.contents()[destination * row ..][0 .. @as(usize, rows) * row], input_words.contents()[first..][0 .. @as(usize, rows) * row]);
            destination += rows;
        }
        head.linear.n = width;
        head.linear.index = null;
        head.linear.pairs32 = null; // the source's covers its whole vocabulary; the copy's pairs are rebuilt below
        head.linear.batch = @intCast(batch);
        head.linear.words = .{ .buffer = words };
        if (source.raw) |raw| {
            const scales = try head.allocate(device, width * groups * 2);
            const biases = try head.allocate(device, width * groups * 2);
            destination = 0;
            for (spans) |span| {
                const bytes = @as(usize, span[1] - span[0]) * groups * 2;
                const first = @as(usize, span[0]) * groups * 2;
                @memcpy(scales.contents()[destination * groups * 2 ..][0..bytes], raw.scales.contents()[raw.s_off + first ..][0..bytes]);
                @memcpy(biases.contents()[destination * groups * 2 ..][0..bytes], raw.biases.contents()[raw.b_off + first ..][0..bytes]);
                destination += span[1] - span[0];
            }
            var selected = raw;
            selected.w = words;
            selected.scales = scales;
            selected.biases = biases;
            selected.w_off = 0;
            selected.s_off = 0;
            selected.b_off = 0;
            selected.n = width;
            try selected.validate();
            head.linear.raw = selected;
            head.linear.pairs = .{ .buffer = scales };
        } else {
            const pairs = try head.allocate(device, width * groups * 4);
            for (0..groups) |g| {
                destination = 0;
                for (spans) |span| {
                    const bytes = @as(usize, span[1] - span[0]) * 4;
                    const first = source.pairs.offset + (g * source.n + span[0]) * 4;
                    @memcpy(pairs.contents()[(g * width + destination) * 4 ..][0..bytes], source.pairs.buffer.contents()[first..][0..bytes]);
                    destination += span[1] - span[0];
                }
            }
            head.linear.pairs = .{ .buffer = pairs };
            // the source's indexed copy, trimmed to the spans' columns: the same table, each group's indices in span order
            if (source.index) |index| {
                const table = lane.sb_table * 4;
                const trimmed = try head.allocate(device, lane.indexedBytes(width, k, source.group));
                @memcpy(trimmed.contents()[0..table], index.buffer.contents()[index.offset..][0..table]);
                for (0..groups) |g| {
                    destination = 0;
                    for (spans) |span| {
                        const bytes = @as(usize, span[1] - span[0]) * 2;
                        const first = index.offset + table + (g * source.n + span[0]) * 2;
                        @memcpy(trimmed.contents()[table + (g * width + destination) * 2 ..][0..bytes], index.buffer.contents()[first..][0..bytes]);
                        destination += span[1] - span[0];
                    }
                }
                head.linear.index = .{ .buffer = trimmed };
            }
        }
        head.ids = try head.allocate(device, width * 4);
        const ids = head.ids.?.slice(u32, width);
        destination = 0;
        for (spans) |span| for (span[0]..span[1]) |id| {
            ids[destination] = @intCast(id);
            destination += 1;
        };
        return head;
    }
    fn allocate(h: *Head, device: mtl.Device, bytes: usize) !mtl.Buffer {
        const buffer = try device.buffer(bytes, options);
        h.owned.append(h.allocator, buffer) catch |err| {
            buffer.deinit();
            return err;
        };
        return buffer;
    }
    pub fn deinit(h: *Head) void {
        for (h.owned.items) |buffer| buffer.deinit();
        h.owned.deinit(h.allocator);
    }
};
test "draft vocabulary includes both stock spans and excludes their gap" {
    try std.testing.expectEqual(width, spans[0][1] - spans[0][0] + spans[1][1] - spans[1][0]);
    for (spans) |span| {
        try std.testing.expect(span[0] < span[1]);
        try std.testing.expectEqual(@as(u32, 0), span[0] % 32);
        try std.testing.expectEqual(@as(u32, 0), span[1] % 32);
    }
}
