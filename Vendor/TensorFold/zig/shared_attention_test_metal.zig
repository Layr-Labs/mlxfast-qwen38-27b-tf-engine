//! Recording-only attention boundary; no device, shader compilation or arithmetic.
const std = @import("std");
const recording = @import("gemma4_recording_metal.zig");
pub const ResourceOptions = @import("src/metal/types.zig").ResourceOptions;
pub const Size = recording.Size;
pub const Record = recording.Record;
pub const Buffer = struct {
    id: usize,
    bytes: usize,
    pub fn length(b: Buffer) usize {
        return b.bytes;
    }
    pub fn deinit(_: Buffer) void {}
};
pub const Device = struct {
    bytes: *usize,
    pub fn buffer(d: Device, bytes: usize, _: usize) !Buffer {
        d.bytes.* = bytes;
        return .{ .id = 100, .bytes = bytes };
    }
};
pub const CompileOptions = struct {
    language: usize = 0,
    math_functions: enum { fast, precise } = .fast,
    pub fn mlx() CompileOptions {
        return .{};
    }
};
pub const Library = struct {
    pub fn fromSource(_: Device, _: []const u8, _: CompileOptions) !Library {
        return .{};
    }
    pub fn deinit(_: Library) void {}
};
pub const Pipeline = struct {
    id: u32,
    pub fn init(_: Device, _: Library, name: []const u8, _: bool) !Pipeline {
        return .{ .id = if (std.mem.endsWith(u8, name, "_split")) 1 else if (std.mem.endsWith(u8, name, "_merge")) 2 else 0 };
    }
    pub fn deinit(_: Pipeline) void {}
    pub fn simdWidth(_: Pipeline) usize {
        return 32;
    }
    pub fn maxThreads(_: Pipeline) usize {
        return 1024;
    }
};
pub const ComputeEncoder = struct {
    r: *Record,
    pub fn setPipeline(e: ComputeEncoder, p: Pipeline) void {
        (recording.ComputeEncoder{ .r = e.r }).setPipeline(p.id);
    }
    pub fn setBuffer(e: ComputeEncoder, b: Buffer, off: usize, i: usize) void {
        (recording.ComputeEncoder{ .r = e.r }).setBuffer(.{ .id = b.id, .bytes = b.bytes }, off, i);
    }
    pub fn setValue(e: ComputeEncoder, v: anytype, i: usize) void {
        (recording.ComputeEncoder{ .r = e.r }).setValue(v, i);
    }
    pub fn dispatchGroups(e: ComputeEncoder, g: Size, t: Size) void {
        (recording.ComputeEncoder{ .r = e.r }).dispatchGroups(g, t);
    }
    pub fn barrier(_: ComputeEncoder) void {}
};
