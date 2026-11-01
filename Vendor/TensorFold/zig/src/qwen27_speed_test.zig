//! Host guards for prompt parts, tap rings and constructor allocation failures.
const std = @import("std");
const taps = @import("native/qwen27_prompt_taps.zig");
const planning = @import("families/qwen27/attention_plan.zig");
const caches = @import("families/qwen27/dflash/caches.zig");
const Ring = @import("families/qwen27/tap_ring.zig").Ring;

const Gather = struct {
    end: usize,
    seen: usize = 0,
    scratch: [130]u32 = @splat(0xfeed),
    ring: [257]u32 = @splat(0),
    pub fn gather(g: *Gather, first: usize, rows: usize) !void {
        if (rows > 128) return error.ScratchOverflow;
        for (0..rows) |i| g.scratch[i + 1] = @intCast(first + i);
        for (g.scratch[1..][0..rows]) |row| {
            try std.testing.expectEqual(@as(u32, @intCast(g.seen)), row);
            g.ring[g.end % g.ring.len] = row;
            g.end += 1;
            g.seen += 1;
        }
        try std.testing.expectEqual(@as(u32, 0xfeed), g.scratch[0]);
        try std.testing.expectEqual(@as(u32, 0xfeed), g.scratch[129]);
    }
};

test "qwen27 speed: secondary tap chunks preserve source indices, guards, tails and restored rings" {
    for ([_]usize{ 1, 127, 128, 129, 256, 511, 512, 513, 2048 }) |rows| {
        for ([_]usize{ 0, 19, 256, 2041 }) |restored| {
            var g = Gather{ .end = restored };
            try taps.absorb(&g, rows);
            try std.testing.expectEqual(rows, g.seen);
            try std.testing.expectEqual(restored + rows, g.end);
            for (rows - @min(rows, g.ring.len)..rows) |i| {
                try std.testing.expectEqual(@as(u32, @intCast(i)), g.ring[(restored + i) % g.ring.len]);
            }
        }
    }
}

test "qwen27 speed: prompt parts fit the actual fused or unfused attention plan" {
    var parents: [2048]i32 = undefined;
    for (&parents, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
    for ([_]bool{ true, false }) |tensor| for ([_]bool{ true, false }) |fuse| {
        const geometry = planning.Geometry{ .shared_prefix = tensor, .fuse_chain = fuse };
        for ([_]u32{ 16, 128, 129, 512, 2048 }) |requested| {
            const part = geometry.promptRows(requested);
            const want = @min(requested, @as(u32, if (tensor and fuse) @import("families/qwen27/gpu_frame.zig").max_rows else planning.max_rows));
            try std.testing.expectEqual(want, part);
            for ([_]u32{ 128, 129, 512, 2051 }) |length| {
                var at: u32 = 0;
                while (at < length) {
                    const n = @min(part, length - at);
                    var plan = try planning.Plan.init(std.testing.allocator, &.{.{
                        .parents = parents[0..n],
                        .start = at,
                        .capacity = 4096,
                    }}, geometry);
                    defer plan.deinit();
                    try std.testing.expectEqual(n, plan.rows);
                    at += n;
                }
            }
        }
    };
}

const Device = struct {
    fail: usize,
    calls: usize = 0,
    alive: usize = 0,
    pub const Buffer = struct {
        owner: *Device,
        pub fn deinit(b: Buffer) void {
            b.owner.alive -= 1;
        }
    };
    pub fn buffer(d: *Device, _: usize, _: usize) !Buffer {
        const at = d.calls;
        d.calls += 1;
        if (at == d.fail) return error.NoBuffer;
        d.alive += 1;
        return .{ .owner = d };
    }
};

const Allocation = struct {
    backing: [1024]u8 align(16) = undefined,
    allocated: usize = 0,
    freed: usize = 0,
    wrong: bool = false,
    fn allocator(a: *Allocation) std.mem.Allocator {
        return .{ .ptr = a, .vtable = &.{ .alloc = alloc, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap, .free = free } };
    }
    fn alloc(ptr: *anyopaque, n: usize, _: std.mem.Alignment, _: usize) ?[*]u8 {
        const a: *Allocation = @ptrCast(@alignCast(ptr));
        if (n > a.backing.len) return null;
        a.allocated = n;
        return &a.backing;
    }
    fn free(ptr: *anyopaque, buf: []u8, _: std.mem.Alignment, _: usize) void {
        const a: *Allocation = @ptrCast(@alignCast(ptr));
        a.wrong = a.wrong or buf.len != a.allocated or buf.ptr != &a.backing;
        a.freed += buf.len;
    }
};

test "qwen27 speed: every cache buffer failure frees the complete CPU allocation and all device owners" {
    for (0..11) |failure| {
        var d = Device{ .fail = failure };
        var a: Allocation = .{};
        if (caches.newCaches(Device.Buffer, a.allocator(), &d, .{}, 5)) |made| {
            caches.freeCaches(Device.Buffer, a.allocator(), made);
        } else |err| try std.testing.expectEqual(error.NoBuffer, err);
        try std.testing.expectEqual(@as(usize, 0), d.alive);
        try std.testing.expect(!a.wrong);
        try std.testing.expectEqual(a.allocated, a.freed);
    }
}


test "qwen27 speed: incomplete counted normalization cannot reach host token selection" {
    const guard = @import("families/qwen27/counted_norm_status.zig");
    const Mock = struct {
        picked: bool = false,
        fn finish(m: *@This(), completed_failure_word: u32) !void {
            try guard.check(completed_failure_word);
            m.picked = true;
        }
    };
    var ready = Mock{};
    try ready.finish(0);
    try std.testing.expect(ready.picked);
    for ([_]u32{ 1, 2, 0xffffffff }) |failed| {
        var incomplete = Mock{};
        try std.testing.expectError(error.QwenNormDependencyIncomplete, incomplete.finish(failed));
        try std.testing.expect(!incomplete.picked);
    }
}


test "qwen27 speed: reg3 storage roundtrip and invalid-code rejection" {
 const a=std.testing.allocator;const reg3=@import("core/reg3.zig");
 var state:u32=0x19374852;
 for([_]usize{1,2,7,80})|blocks|{
  const wide=try a.alloc(u32,blocks*256);defer a.free(wide);
  const compact=try a.alloc(u32,blocks*192);defer a.free(compact);
  const restored=try a.alloc(u32,wide.len);defer a.free(restored);
  for(wide)|*v|{state^=state<<13;state^=state>>17;state^=state<<5;v.*=state&0x77777777;}
  try reg3.pack(wide,compact);try reg3.unpack(compact,restored);try std.testing.expectEqualSlices(u32,wide,restored);
  wide[wide.len-1]|=0x80000000;@memset(compact,0xdeadbeef);try std.testing.expectError(error.WideCodes,reg3.pack(wide,compact));for(compact)|v|try std.testing.expectEqual(@as(u32,0xdeadbeef),v);
 }
 try std.testing.expectError(error.InvalidStorage,reg3.pack(&.{0},&.{}));
 try std.testing.expectError(error.InvalidStorage,reg3.unpack(&.{0},&.{}));
}
test "qwen27 speed: reg3 preparation codes are bounded and malformed data fails" {
 const quant=@import("core/affine3.zig");var w:[128]u16=undefined;
 for(&w,0..)|*v,i|{const x:f32=(@as(f32,@floatFromInt(i))-64)*0.001;v.*=@import("core/affine4.zig").asBf16(x);}
 var q=try quant.prepare(std.testing.allocator,&w,2,64);defer q.deinit();try std.testing.expectEqual(@as(u8,3),q.bits);try std.testing.expect(q.nibbles);
 for(q.words)|word|try std.testing.expect(word&0x88888888==0);
 try std.testing.expectError(error.BadAffineShape,quant.prepare(std.testing.allocator,&w,2,63));
 w[7]=0x7fc0;try std.testing.expectError(error.NonFiniteAffineInput,quant.prepare(std.testing.allocator,&w,2,64));
}

test "qwen27 speed: reg3 preparation identifies its own rule and excludes default q4 parity" {
 const op=@import("families/qwen27/dflash/operators.zig");
 const sig=@import("core/affine4.zig").threeSignature();
 try std.testing.expect(!std.mem.allEqual(u8,&sig,0));
 var preparation=op.Preparation{.mode=.prepared_reg3_reference,.quantization_verified=true,.source_sha256=sig};try preparation.check();
 preparation.default_proposal_parity=true;try std.testing.expectError(error.DraftPreparationUnqualified,preparation.check());
 preparation.default_proposal_parity=false;preparation.quantization_verified=false;try std.testing.expectError(error.DraftPreparationUnqualified,preparation.check());
 preparation.quantization_verified=true;preparation.source_sha256=@splat(0);try std.testing.expectError(error.DraftPreparationUnqualified,preparation.check());
}
