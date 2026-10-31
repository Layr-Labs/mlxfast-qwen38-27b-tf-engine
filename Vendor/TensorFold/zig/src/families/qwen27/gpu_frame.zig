//! One aggregate row frame owns activation, group sums and metadata independently of stream caches.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const Ref = @import("projection.zig").Ref;

pub const Field = enum { ids, dims, positions, hidden, residual, input, sums, mixed, mixer_out, output, z, a, b, query, key, value, norm_query, norm_key, q_gate, gate, up, activated, act_sums, logits, drawn, eps, parents, windows, row_slots, segments, keeps, kept_rows, rotated_key, takes, slot_steps, last_dims };

/// A prompt pass's rows at most (a chain); rounds with trees and every logits row stay within `logit_rows`.
pub const max_rows = 2048;
pub const logit_rows = 128;

pub fn allocationSizes(config: Config, capacity: u32) [std.enums.values(Field).len]usize {
    const mp = 16 * ((capacity + 15) / 16);
    return .{
        capacity * 4,                                                                       8 * 4,                                                                                                                                  3 * capacity * 4,
        capacity * config.hidden * 2,                                                       capacity * config.hidden * 2,                                                                                                           capacity * config.hidden * 2,
        mp * @max(config.hidden, @max(config.intermediate, config.gdnValueDim())) / 64 * 4, capacity * @max(config.gdnQkvDim(), @max(config.qDim(), @max(2 * config.intermediate, config.gdnValueDim() + 2 * config.v_heads))) * 2, capacity * config.gdnValueDim() * 2,
        capacity * config.hidden * 2,                                                       capacity * config.gdnValueDim() * 2,                                                                                                    capacity * config.v_heads * 2,
        capacity * config.v_heads * 2,                                                      capacity * config.heads * config.head_dim * 2,                                                                                          capacity * config.kvDim() * 2,
        capacity * config.kvDim() * 2,                                                      capacity * config.heads * config.head_dim * 2,                                                                                          capacity * config.kvDim() * 2,
        capacity * config.qDim() * 2,                                                       capacity * config.intermediate * 2,                                                                                                     capacity * config.intermediate * 2,
        capacity * config.intermediate * 2,                                                 mp * config.intermediate / 64 * 4,                                                                                                      @min(capacity, logit_rows) * config.vocab * 2,
        capacity * 4,                                                                       4,                                                                                                                                      capacity * 4,
        capacity * config.conv_kernel * 4,                                                  capacity * 4,                                                                                                                           capacity * 16,
        capacity * 16,                                                                      capacity * 4,                                                                                                                           capacity * config.kvDim() * 2,
        capacity * 4,                                                                       capacity * @sizeOf(@import("core").tree_slots.Step), 8 * 4,
    };
}

pub const small_rows = 512;
const scratch_fields = [_]Field{ .hidden, .residual, .input, .sums, .mixed, .mixer_out, .output, .z, .a, .b, .query, .key, .value, .norm_query, .norm_key, .q_gate, .gate, .up, .activated, .act_sums };

pub fn decodeSizes(config: Config, capacity: u32) [scratch_fields.len]usize {
    const widths = allocationSizes(config, @min(capacity, small_rows));
    var result: [scratch_fields.len]usize = undefined;
    for (scratch_fields, &result) |field, *bytes| bytes.* = widths[@backingInt(field)];
    return result;
}

pub const Frame = struct {
    buffers: [std.enums.values(Field).len]mtl.Buffer,
    capacity: u32,
    config: Config,
    standby: ?[scratch_fields.len]mtl.Buffer = null,
    small: bool = false,
    /// A GDN block's gate or a decoded-row attention merge wrote its out projection's group sums into `.sums` (read
    /// and cleared by that projection).
    mixer_sums: bool = false,

    pub fn init(device: mtl.Device, config: Config, capacity: u32) !Frame {
        if (capacity == 0 or capacity > max_rows) return error.BadFrameRows;
        const widths = allocationSizes(config, capacity);
        comptime std.debug.assert(widths.len == std.enums.values(Field).len);
        var buffers: [widths.len]mtl.Buffer = undefined;
        var count: usize = 0;
        errdefer for (buffers[0..count]) |b| b.deinit();
        for (widths, &buffers) |bytes, *b| {
            b.* = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            count += 1;
        }
        var standby: ?[scratch_fields.len]mtl.Buffer = null;
        if (capacity > small_rows) {
            var bank: [scratch_fields.len]mtl.Buffer = undefined;
            var made: usize = 0;
            errdefer for (bank[0..made]) |buffer| buffer.deinit();
            for (decodeSizes(config, capacity), &bank) |bytes, *buffer| {
                buffer.* = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                made += 1;
            }
            standby = bank;
        }
        buffers[@backingInt(Field.eps)].slice(f32, 1)[0] = config.eps;
        const last_dims = buffers[@backingInt(Field.last_dims)].slice(i32, 8);
        @memset(last_dims, 0);
        last_dims[0] = 1;
        last_dims[1] = 16;
        return .{ .buffers = buffers, .capacity = capacity, .config = config, .standby = standby };
    }

    pub fn deinit(frame: *Frame) void {
        for (frame.buffers) |b| b.deinit();
        if (frame.standby) |bank| for (bank) |b| b.deinit();
    }

    /// Both banks keep their residency even when the next completed round changes the active views.
    pub fn appendBuffers(frame: *const Frame, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        try list.appendSlice(a, &frame.buffers);
        if (frame.standby) |bank| try list.appendSlice(a, &bank);
    }

    pub fn get(frame: *const Frame, field: Field) Ref {
        return .{ .buffer = frame.buffers[@backingInt(field)] };
    }

    /// The frame's first `rows` rows as one pass sees them, on the frame's own dims.
    pub fn whole(frame: *Frame, rows: u32) View {
        return View.init(frame, 0, rows, frame.get(.dims), 0, 0);
    }

    /// Called between completed rounds; metadata, position planes and outputs retain their original owners.
    pub fn setRows(frame: *Frame, ids: []const u32, positions: []const i32) !void {
        if (ids.len == 0 or ids.len > frame.capacity or positions.len != ids.len) return error.BadFrameRows;
        if (frame.standby) |*bank| {
            const small = ids.len <= small_rows;
            if (small != frame.small) {
                for (scratch_fields, bank) |field, *buffer| std.mem.swap(mtl.Buffer, &frame.buffers[@backingInt(field)], buffer);
                frame.small = small;
            }
        }
        @memcpy(frame.get(.ids).buffer.slice(u32, ids.len), ids);
        @memcpy(frame.get(.positions).buffer.slice(i32, positions.len), positions);
        @memset(frame.get(.dims).buffer.slice(i32, 8), 0);
        frame.get(.dims).buffer.slice(i32, 8)[0] = @intCast(ids.len);
        frame.get(.dims).buffer.slice(i32, 8)[1] = @intCast(16 * ((ids.len + 15) / 16));
    }
};

/// Rows [start, start + rows) of a frame as one pass over them sees them: every row field from its first row, the
/// group sums (XS[group * stride + row], `stride` its padded rows) in a region of their own, dims of their own and the
/// window descriptors of its part. A split prompt chunk's segments run on two views of one frame at once.
pub const View = struct {
    frame: *Frame,
    start: u32,
    rows: u32,
    /// [rows, padded rows, 0...] as setRows writes the frame's.
    dims: Ref,
    /// Padded rows before this view's group-sums region.
    sums_row: u32,
    /// Whose segment and keep descriptors (16 bytes each) the view's plan uploads.
    part: u32,
    row_bytes: [std.enums.values(Field).len]usize,
    /// A GDN block's gate or a decoded-row attention merge wrote its out projection's group sums into `.sums` (read
    /// and cleared by that projection).
    mixer_sums: bool = false,

    pub fn init(frame: *Frame, start: u32, rows: u32, dims: Ref, sums_row: u32, part: u32) View {
        var row_bytes = allocationSizes(frame.config, 1); // one row of each row field
        row_bytes[@backingInt(Field.positions)] = 4; // three planes `capacity` rows apart: the first plane's row
        return .{ .frame = frame, .start = start, .rows = rows, .dims = dims, .sums_row = sums_row, .part = part, .row_bytes = row_bytes };
    }

    pub fn get(v: *const View, field: Field) Ref {
        var r = v.frame.get(field);
        r.offset += switch (field) {
            .dims => return v.dims,
            .eps, .logits, .last_dims => 0,
            // group-major: a group's 16 padded rows at capacity 1
            .sums, .act_sums => @as(usize, v.sums_row) * (v.row_bytes[@backingInt(field)] / 16),
            .segments, .keeps => @as(usize, v.part) * 16,
            else => @as(usize, v.start) * v.row_bytes[@backingInt(field)],
        };
        return r;
    }

    /// The view's dims, written where `dims` points.
    pub fn setDims(v: *const View) void {
        const words = @as([*]i32, @ptrCast(@alignCast(v.dims.buffer.contents() + v.dims.offset)))[0..8];
        @memset(words, 0);
        words[0] = @intCast(v.rows);
        words[1] = @intCast(16 * ((v.rows + 15) / 16));
    }
};
