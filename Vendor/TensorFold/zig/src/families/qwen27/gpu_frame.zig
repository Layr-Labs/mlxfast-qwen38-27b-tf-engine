//! One aggregate row frame owns activation, group sums and metadata independently of stream caches.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const Ref = @import("projection.zig").Ref;

pub const Field = enum { ids, dims, positions, hidden, residual, input, sums, mixed, mixer_out, output, z, a, b, query, key, value, norm_query, norm_key, q_gate, gate, up, activated, act_sums, logits, drawn, eps, parents, windows, row_slots, segments, keeps, kept_rows, rotated_key, takes, slot_steps, last_dims, out_sums };

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
        // team-m5-08q: out_sums, a waiting out projection's group sums (Frame.mixer_xs): every out projection's k
        mp * @max(config.gdnValueDim(), config.heads * config.head_dim) / 64 * 4,
    };
}

pub const small_rows = 512;
const scratch_fields = [_]Field{ .hidden, .residual, .input, .sums, .mixed, .mixer_out, .output, .z, .a, .b, .query, .key, .value, .norm_query, .norm_key, .q_gate, .gate, .up, .activated, .act_sums, .out_sums };

pub fn decodeSizes(config: Config, capacity: u32) [scratch_fields.len]usize {
    const widths = allocationSizes(config, @min(capacity, small_rows));
    var result: [scratch_fields.len]usize = undefined;
    for (scratch_fields, &result) |field, *bytes| bytes.* = widths[@backingInt(field)];
    return result;
}

/// team-m5-08q: a count the out projection waits on: `wait` threadgroups count themselves on the u32 word at `offset` of
/// `words`, past a device fence; the next word takes the out projection's arrivals, and its last threadgroup leaves both
/// zero.
pub const MixerCount = struct { words: mtl.Buffer, offset: usize, wait: u32 };

pub const Frame = struct {
    buffers: [std.enums.values(Field).len]mtl.Buffer,
    capacity: u32,
    config: Config,
    standby: ?[scratch_fields.len]mtl.Buffer = null,
    small: bool = false,
    /// A GDN block's gate or a decoded-row attention merge wrote its out projection's group sums into `.sums` (read
    /// and cleared by that projection).
    mixer_sums: bool = false,
    /// team-m5-08q: set by forward.Stages.mixer for its hook: these rows' out projection can take the waiting counted
    /// launch (quant_gpu.Kernels.outWaits), so the mixer's last kernel may count its threadgroups for it instead of ending
    /// on a barrier.
    out_waits: bool = false,
    /// team-m5-08q: the mixer's last kernel (the GDN gate, or the decoded attention merge) counted its threadgroups and no
    /// barrier follows it: forward.post's out projection waits on that count (read and cleared there).
    mixer_count: ?MixerCount = null,
    /// team-m5-08q: the field the mixer's last kernel wrote its out projection's group sums into: `.out_sums` on rows
    /// whose out projection waits (out_waits), since that projection then shares a barrier interval with the mixer's
    /// counted projections, which read the norm's `.sums` in it. A counted consumer reads only what nothing earlier in
    /// its interval read: a threadgroup on a core whose cache still holds lines of the earlier read would see them
    /// rather than the counted writes. Read and reset to `.sums` by forward.post.
    mixer_xs: Field = .sums,

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
