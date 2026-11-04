//! The tune-once pilot on synthetic buffers: build the draft Ops, run the linear32
//! load-time tune exactly as engine open does, and report the fixed winner.
const std = @import("std");
const mtl = @import("metal");
const ops = @import("draft_ops");

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var o = try ops.Ops.init(device);
    defer o.deinit();
    o.tuneLinear32(device);
    std.debug.print("linear32: tuned={} adopted={}\n", .{ o.tuned, o.use_linear32 });
}
