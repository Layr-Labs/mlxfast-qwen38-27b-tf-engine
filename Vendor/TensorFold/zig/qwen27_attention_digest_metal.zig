//! Host buffers and dispatch counts only; no device, shader or arithmetic execution.
const std = @import("std");
const types = @import("src/metal/types.zig");
const recording = @import("gemma4_head_metal.zig");
pub const ResourceOptions = types.ResourceOptions;
pub const ResourceUsage = types.ResourceUsage;
pub const Size = types.Size;
pub const Pipeline = recording.Pipeline;
pub const Library = recording.Library;
pub const CompileOptions = recording.CompileOptions;
pub const Buffer = struct {
    id: usize,
    data: []align(16) u8,
    pub fn length(b: Buffer) usize {
        return b.data.len;
    }
    pub fn gpuAddress(b: Buffer) u64 {
        return @intFromPtr(b.data.ptr);
    }
    pub fn slice(b: Buffer, comptime T: type, n: usize) []T {
        std.debug.assert(n * @sizeOf(T) <= b.data.len);
        return @as([*]T, @ptrCast(@alignCast(b.data.ptr)))[0..n];
    }
    pub fn deinit(b: Buffer) void {
        std.testing.allocator.free(b.data);
    }
};
pub const Device = struct {
    pub fn buffer(_: Device, bytes: usize, _: usize) !Buffer {
        const data = try std.testing.allocator.alignedAlloc(u8, .@"16", bytes);
        @memset(data, 0);
        return .{ .id = @intFromPtr(data.ptr), .data = data };
    }
};
pub const Record = struct { dispatches: usize = 0 };
pub const ComputeEncoder = struct {
    record: *Record,
    pub fn setPipeline(_: ComputeEncoder, _: Pipeline) void {}
    pub fn setBuffer(_: ComputeEncoder, _: Buffer, _: usize, _: usize) void {}
    pub fn setValue(_: ComputeEncoder, _: anytype, _: usize) void {}
    pub fn useResource(_: ComputeEncoder, _: Buffer, _: usize) void {}
    pub fn dispatchGroups(e: ComputeEncoder, _: Size, _: Size) void {
        e.record.dispatches += 1;
    }
    pub fn dispatchThreads(e: ComputeEncoder, _: Size, _: Size) void {
        e.record.dispatches += 1;
    }
    pub fn barrier(_: ComputeEncoder) void {}
    pub fn end(_: ComputeEncoder) void {}
};
pub const Queue = struct {
    pub fn commandBuffer(_: Queue) CommandBuffer {
        unreachable;
    }
};
pub const CommandBuffer = struct {
    pub fn compute(_: CommandBuffer, _: enum { serial }) ComputeEncoder {
        unreachable;
    }
    pub fn commit(_: CommandBuffer) void {
        unreachable;
    }
    pub fn wait(_: CommandBuffer) void {
        unreachable;
    }
    pub fn failure(_: CommandBuffer) ?[]const u8 {
        unreachable;
    }
    pub fn gpuSeconds(_: CommandBuffer) f64 {
        unreachable;
    }
};
