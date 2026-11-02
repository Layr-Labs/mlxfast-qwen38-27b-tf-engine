//! Command queues, command buffers and compute encoders (serial or concurrent dispatch).
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const Buffer = @import("device.zig").Buffer;
const Pipeline = @import("library.zig").Pipeline;
const SharedEvent = @import("sync.zig").SharedEvent;
const Event = @import("sync.zig").Event;
const IndirectCommandBuffer = @import("icb.zig").IndirectCommandBuffer;
const ResidencySet = @import("residency.zig").ResidencySet;

const Id = objc.Id;
const Size = types.Size;

pub const Queue = struct {
    id: Id,

    /// A command buffer that retains what it references (autoreleased: keep a pool around it).
    pub fn commandBuffer(self: Queue) CommandBuffer {
        return .{ .id = objc.msg(Id, self.id, "commandBuffer", .{}) };
    }

    /// A command buffer that does not retain its resources: the caller keeps them alive until it completes.
    pub fn commandBufferUnretained(self: Queue) CommandBuffer {
        return .{ .id = objc.msg(Id, self.id, "commandBufferWithUnretainedReferences", .{}) };
    }

    /// Make `set` resident for every command buffer of this queue.
    pub fn addResidencySet(self: Queue, set: ResidencySet) void {
        objc.msg(void, self.id, "addResidencySet:", .{set.id});
    }

    pub fn removeResidencySet(self: Queue, set: ResidencySet) void {
        objc.msg(void, self.id, "removeResidencySet:", .{set.id});
    }

    pub fn deinit(self: Queue) void {
        objc.release(self.id);
    }
};

pub const CommandBuffer = struct {
    id: Id,

    pub fn compute(self: CommandBuffer, dispatch: types.DispatchType) ComputeEncoder {
        const e: ComputeEncoder = .{ .id = objc.msg(Id, self.id, "computeCommandEncoderWithDispatchType:", .{@backingInt(dispatch)}) };
        bound.reset(e.id);
        return e;
    }

    /// A blit encoder on this buffer, for copies between buffers on the GPU.
    pub fn blit(self: CommandBuffer) BlitEncoder {
        return .{ .id = objc.msg(Id, self.id, "blitCommandEncoder", .{}) };
    }

    /// The GPU waits here until `event` reaches `value`.
    pub fn waitFor(self: CommandBuffer, event: SharedEvent, value: u64) void {
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    /// The GPU sets `event` to `value` once the work before this point completes.
    pub fn signal(self: CommandBuffer, event: SharedEvent, value: u64) void {
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    /// waitFor and signal with a GPU-only event.
    pub fn waitForEvent(self: CommandBuffer, event: Event, value: u64) void {
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    pub fn signalEvent(self: CommandBuffer, event: Event, value: u64) void {
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    pub fn useResidencySet(self: CommandBuffer, set: ResidencySet) void {
        objc.msg(void, self.id, "useResidencySet:", .{set.id});
    }

    pub fn commit(self: CommandBuffer) void {
        objc.msg(void, self.id, "commit", .{});
    }

    pub fn wait(self: CommandBuffer) void {
        objc.msg(void, self.id, "waitUntilCompleted", .{});
    }

    pub fn status(self: CommandBuffer) types.CommandBufferStatus {
        return @fromBackingInt(@intCast(objc.msg(usize, self.id, "status", .{})));
    }

    /// The error text when the command buffer failed, else null.
    pub fn failure(self: CommandBuffer) ?[*:0]const u8 {
        if (self.status() != .@"error") return null;
        return objc.errorText(objc.msg(?Id, self.id, "error", .{}));
    }

    /// GPU execution span in seconds (host clock), valid once completed.
    pub fn gpuSeconds(self: CommandBuffer) f64 {
        return self.gpuEnd() - self.gpuStart();
    }

    pub fn gpuStart(self: CommandBuffer) f64 {
        return objc.msg(f64, self.id, "GPUStartTime", .{});
    }

    pub fn gpuEnd(self: CommandBuffer) f64 {
        return objc.msg(f64, self.id, "GPUEndTime", .{});
    }

    pub fn retain(self: CommandBuffer) CommandBuffer {
        return .{ .id = objc.retain(self.id) };
    }

    pub fn release(self: CommandBuffer) void {
        objc.release(self.id);
    }
};

/// Copies between buffers on the GPU.
pub const BlitEncoder = struct {
    id: Id,

    pub fn copy(self: BlitEncoder, from: Buffer, from_offset: usize, to: Buffer, to_offset: usize, size: usize) void {
        objc.msg(void, self.id, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", .{ from.id, from_offset, to.id, to_offset, size });
    }

    pub fn end(self: BlitEncoder) void {
        objc.msg(void, self.id, "endEncoding", .{});
    }
};

/// The buffer bindings this thread last set on its current compute encoder, so a setBuffer that repeats one (same encoder,
/// index, buffer and offset) is not sent again. Metal keeps an encoder's buffer bindings across pipeline changes and
/// dispatches; a setBytes at an index replaces its buffer binding (the slot is forgotten), and an encoder's slots are
/// forgotten when it is created or ended (an ended encoder's id can be reused by the next one).
const Bound = struct {
    const slots = 32;
    enc: ?Id = null,
    buffer: [slots]?Id = @splat(null),
    offset: [slots]usize = @splat(0),
    fn reset(b: *Bound, e: ?Id) void {
        b.enc = e;
        b.buffer = @splat(null);
    }
    /// Whether `buffer` at `offset` is already bound at `index` on `e`; records it as bound either way.
    fn same(b: *Bound, e: Id, buffer: Id, offset: usize, index: usize) bool {
        if (b.enc == null or b.enc.? != e) b.reset(e);
        if (index >= slots) return false;
        if (b.buffer[index]) |held| if (held == buffer and b.offset[index] == offset) return true;
        b.buffer[index] = buffer;
        b.offset[index] = offset;
        return false;
    }
    fn forget(b: *Bound, e: Id, index: usize) void {
        if (b.enc == null or b.enc.? != e) b.reset(e);
        if (index < slots) b.buffer[index] = null;
    }
};
threadlocal var bound: Bound = .{};

pub const ComputeEncoder = struct {
    id: Id,

    pub fn setPipeline(self: ComputeEncoder, pipeline: Pipeline) void {
        objc.msg(void, self.id, "setComputePipelineState:", .{pipeline.id});
    }

    pub fn setBuffer(self: ComputeEncoder, buffer: Buffer, offset: usize, index: usize) void {
        if (bound.same(self.id, buffer.id, offset, index)) return;
        objc.msg(void, self.id, "setBuffer:offset:atIndex:", .{ buffer.id, offset, index });
    }

    /// Small constants copied into the command stream (at most 4 KB).
    pub fn setBytes(self: ComputeEncoder, bytes: []const u8, index: usize) void {
        bound.forget(self.id, index);
        objc.msg(void, self.id, "setBytes:length:atIndex:", .{ bytes.ptr, bytes.len, index });
    }

    pub fn setValue(self: ComputeEncoder, value: anytype, index: usize) void {
        self.setBytes(std.mem.asBytes(&value), index);
    }

    pub fn setThreadgroupMemory(self: ComputeEncoder, len: usize, index: usize) void {
        objc.msg(void, self.id, "setThreadgroupMemoryLength:atIndex:", .{ len, index });
    }

    /// Whole threadgroups: `groups` of `threads` each.
    pub fn dispatchGroups(self: ComputeEncoder, groups: Size, threads: Size) void {
        objc.msg(void, self.id, "dispatchThreadgroups:threadsPerThreadgroup:", .{ groups, threads });
    }

    /// Whole threadgroups of `threads` each, their counts (3 u32) read from `args` at `offset` when the dispatch runs.
    pub fn dispatchIndirect(self: ComputeEncoder, args: Buffer, offset: usize, threads: Size) void {
        objc.msg(void, self.id, "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:", .{ args.id, offset, threads });
    }

    /// A grid of `grid` threads in groups of `threads` (edge groups may be partial), as MLX dispatches.
    pub fn dispatchThreads(self: ComputeEncoder, grid: Size, threads: Size) void {
        objc.msg(void, self.id, "dispatchThreads:threadsPerThreadgroup:", .{ grid, threads });
    }

    /// Later dispatches see every buffer write before this point (concurrent encoders).
    pub fn barrier(self: ComputeEncoder) void {
        objc.msg(void, self.id, "memoryBarrierWithScope:", .{types.BarrierScope.buffers});
    }

    /// Declare a resource used through indirect commands or argument buffers.
    pub fn useResource(self: ComputeEncoder, buffer: Buffer, usage: usize) void {
        objc.msg(void, self.id, "useResource:usage:", .{ buffer.id, usage });
    }

    pub fn execute(self: ComputeEncoder, icb: IndirectCommandBuffer, first: usize, count: usize) void {
        bound.reset(self.id); // indirect commands may leave the encoder's bindings changed
        objc.msg(void, self.id, "executeCommandsInBuffer:withRange:", .{ icb.id, types.Range{ .location = first, .length = count } });
    }

    pub fn end(self: ComputeEncoder) void {
        bound.reset(null);
        objc.msg(void, self.id, "endEncoding", .{});
    }
};
