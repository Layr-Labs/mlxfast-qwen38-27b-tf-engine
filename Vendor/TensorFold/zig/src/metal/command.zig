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
const mtl4 = @import("mtl4.zig");

const Id = objc.Id;
const Size = types.Size;

pub const Queue = struct {
    id: Id,
    /// The Metal 4 path's state (mtl4.zig) when this queue is an MTL4CommandQueue; null on the classic path.
    m4: ?*mtl4.Queue = null,

    /// A command buffer that retains what it references (autoreleased: keep a pool around it).
    pub fn commandBuffer(self: Queue) CommandBuffer {
        if (self.m4) |q| {
            const s = q.slot();
            return .{ .id = s.allocator, .m4 = s, .generation = s.generation };
        }
        return .{ .id = objc.msg(Id, self.id, "commandBuffer", .{}) };
    }

    /// A command buffer that does not retain its resources: the caller keeps them alive until it completes.
    pub fn commandBufferUnretained(self: Queue) CommandBuffer {
        if (self.m4 != null) return self.commandBuffer(); // the Metal 4 slot retains what it binds until it completes
        return .{ .id = objc.msg(Id, self.id, "commandBufferWithUnretainedReferences", .{}) };
    }

    /// Make `set` resident for every command buffer of this queue.
    pub fn addResidencySet(self: Queue, set: ResidencySet) void {
        if (self.m4) |q| return q.addResidencySet(set.id);
        objc.msg(void, self.id, "addResidencySet:", .{set.id});
    }

    pub fn removeResidencySet(self: Queue, set: ResidencySet) void {
        if (self.m4) |q| return q.removeResidencySet(set.id);
        objc.msg(void, self.id, "removeResidencySet:", .{set.id});
    }

    pub fn deinit(self: Queue) void {
        if (self.m4) |q| return q.deinit();
        objc.release(self.id);
    }
};

pub const CommandBuffer = struct {
    id: Id,
    /// The Metal 4 slot this buffer records into (mtl4.zig), and the slot's generation when it was taken: a slot is
    /// reused only after its buffer completed, so a stale generation reads as completed.
    m4: ?*mtl4.Slot = null,
    generation: u32 = 0,

    pub fn compute(self: CommandBuffer, dispatch: types.DispatchType) ComputeEncoder {
        // Metal 4 encoders dispatch concurrently: a .serial one barriers before each dispatch, as the classic encoder orders them
        if (self.m4) |s| return .{ .id = s.encoder(), .m4 = s, .serial = dispatch == .serial };
        return .{ .id = objc.msg(Id, self.id, "computeCommandEncoderWithDispatchType:", .{@backingInt(dispatch)}) };
    }

    /// A blit encoder on this buffer, for copies between buffers on the GPU (on Metal 4, a compute encoder's copies).
    pub fn blit(self: CommandBuffer) BlitEncoder {
        if (self.m4) |s| return .{ .id = s.encoder(), .m4 = s };
        return .{ .id = objc.msg(Id, self.id, "blitCommandEncoder", .{}) };
    }

    /// The GPU waits here until `event` reaches `value`.
    pub fn waitFor(self: CommandBuffer, event: SharedEvent, value: u64) void {
        if (self.m4) |s| return s.waitFor(event.id, value);
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    /// The GPU sets `event` to `value` once the work before this point completes.
    pub fn signal(self: CommandBuffer, event: SharedEvent, value: u64) void {
        if (self.m4) |s| return s.signal(event.id, value);
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    /// waitFor and signal with a GPU-only event.
    pub fn waitForEvent(self: CommandBuffer, event: Event, value: u64) void {
        if (self.m4) |s| return s.waitFor(event.id, value);
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    pub fn signalEvent(self: CommandBuffer, event: Event, value: u64) void {
        if (self.m4) |s| return s.signal(event.id, value);
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    pub fn useResidencySet(self: CommandBuffer, set: ResidencySet) void {
        if (self.m4) |s| return s.q.addResidencySet(set.id); // a Metal 4 queue keeps the set for every buffer
        objc.msg(void, self.id, "useResidencySet:", .{set.id});
    }

    pub fn commit(self: CommandBuffer) void {
        if (self.m4) |s| return s.commit();
        objc.msg(void, self.id, "commit", .{});
    }

    pub fn wait(self: CommandBuffer) void {
        if (self.m4) |s| return s.wait(self.generation);
        objc.msg(void, self.id, "waitUntilCompleted", .{});
    }

    pub fn status(self: CommandBuffer) types.CommandBufferStatus {
        if (self.m4) |s| return s.status(self.generation);
        return @fromBackingInt(@intCast(objc.msg(usize, self.id, "status", .{})));
    }

    /// The error text when the command buffer failed, else null (never on Metal 4, which reports through feedback).
    pub fn failure(self: CommandBuffer) ?[*:0]const u8 {
        if (self.m4 != null) return null;
        if (self.status() != .@"error") return null;
        return objc.errorText(objc.msg(?Id, self.id, "error", .{}));
    }

    /// GPU execution span in seconds (host clock), valid once completed (0 on Metal 4).
    pub fn gpuSeconds(self: CommandBuffer) f64 {
        return self.gpuEnd() - self.gpuStart();
    }

    pub fn gpuStart(self: CommandBuffer) f64 {
        if (self.m4 != null) return 0;
        return objc.msg(f64, self.id, "GPUStartTime", .{});
    }

    pub fn gpuEnd(self: CommandBuffer) f64 {
        if (self.m4 != null) return 0;
        return objc.msg(f64, self.id, "GPUEndTime", .{});
    }

    pub fn retain(self: CommandBuffer) CommandBuffer {
        if (self.m4 != null) return self; // the slot outlives the buffer's completion
        return .{ .id = objc.retain(self.id) };
    }

    pub fn release(self: CommandBuffer) void {
        if (self.m4 != null) return;
        objc.release(self.id);
    }
};

/// Copies between buffers on the GPU.
pub const BlitEncoder = struct {
    id: Id,
    m4: ?*mtl4.Slot = null,

    pub fn copy(self: BlitEncoder, from: Buffer, from_offset: usize, to: Buffer, to_offset: usize, size: usize) void {
        if (self.m4) |s| {
            _ = s.keep(from.id);
            _ = s.keep(to.id);
            // copies of a classic blit encoder run in order: each waits for the work before it
            objc.msg(void, self.id, "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", .{ mtl4_stages_all, mtl4_stages_all, mtl4_visibility_device });
        }
        objc.msg(void, self.id, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", .{ from.id, from_offset, to.id, to_offset, size });
    }

    pub fn end(self: BlitEncoder) void {
        objc.msg(void, self.id, "endEncoding", .{});
    }
};

pub const ComputeEncoder = struct {
    id: Id,
    /// The Metal 4 slot whose argument table and upload chunks this encoder binds through; null on the classic path.
    m4: ?*mtl4.Slot = null,
    /// A Metal 4 encoder standing in for a classic serial one: every dispatch waits for the dispatches before it.
    serial: bool = false,

    pub fn setPipeline(self: ComputeEncoder, pipeline: Pipeline) void {
        objc.msg(void, self.id, "setComputePipelineState:", .{pipeline.id});
    }

    pub fn setBuffer(self: ComputeEncoder, buffer: Buffer, offset: usize, index: usize) void {
        if (self.m4) |s| return s.bind(buffer.id, offset, index);
        objc.msg(void, self.id, "setBuffer:offset:atIndex:", .{ buffer.id, offset, index });
    }

    /// Small constants copied into the command stream (at most 4 KB).
    pub fn setBytes(self: ComputeEncoder, bytes: []const u8, index: usize) void {
        if (self.m4) |s| return s.bytes(bytes, index);
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
        if (self.serial) self.barrier();
        objc.msg(void, self.id, "dispatchThreadgroups:threadsPerThreadgroup:", .{ groups, threads });
    }

    /// Whole threadgroups of `threads` each, their counts (3 u32) read from `args` at `offset` when the dispatch runs.
    pub fn dispatchIndirect(self: ComputeEncoder, args: Buffer, offset: usize, threads: Size) void {
        if (self.serial) self.barrier();
        if (self.m4) |s| return objc.msg(void, self.id, "dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:", .{ s.address(args.id, offset), threads });
        objc.msg(void, self.id, "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:", .{ args.id, offset, threads });
    }

    /// A grid of `grid` threads in groups of `threads` (edge groups may be partial), as MLX dispatches.
    pub fn dispatchThreads(self: ComputeEncoder, grid: Size, threads: Size) void {
        if (self.serial) self.barrier();
        objc.msg(void, self.id, "dispatchThreads:threadsPerThreadgroup:", .{ grid, threads });
    }

    /// Later dispatches see every buffer write before this point (concurrent encoders).
    pub fn barrier(self: ComputeEncoder) void {
        if (self.m4 != null) return objc.msg(void, self.id, "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", .{ mtl4_stages_all, mtl4_stages_all, mtl4_visibility_device });
        objc.msg(void, self.id, "memoryBarrierWithScope:", .{types.BarrierScope.buffers});
    }

    /// Declare a resource used through indirect commands or argument buffers.
    pub fn useResource(self: ComputeEncoder, buffer: Buffer, usage: usize) void {
        if (self.m4) |s| { // resident through the queue's set; Metal 4 has no useResource
            _ = s.keep(buffer.id);
            return;
        }
        objc.msg(void, self.id, "useResource:usage:", .{ buffer.id, usage });
    }

    pub fn execute(self: ComputeEncoder, icb: IndirectCommandBuffer, first: usize, count: usize) void {
        objc.msg(void, self.id, "executeCommandsInBuffer:withRange:", .{ icb.id, types.Range{ .location = first, .length = count } });
    }

    pub fn end(self: ComputeEncoder) void {
        objc.msg(void, self.id, "endEncoding", .{});
    }

    /// The encoder's dispatches start after the work that last updated `fence` (core/fence.zig).
    pub fn waitFence(self: ComputeEncoder, fence: Id) void {
        if (self.m4 != null) return objc.msg(void, self.id, "waitForFence:beforeEncoderStages:", .{ fence, mtl4_stages_all });
        objc.msg(void, self.id, "waitForFence:", .{fence});
    }

    /// Later waits on `fence` see this encoder's work complete.
    pub fn updateFence(self: ComputeEncoder, fence: Id) void {
        if (self.m4 != null) return objc.msg(void, self.id, "updateFence:afterEncoderStages:", .{ fence, mtl4_stages_all });
        objc.msg(void, self.id, "updateFence:", .{fence});
    }
};

/// MTLStageAll and MTL4VisibilityOptionDevice (mtl4.zig's).
const mtl4_stages_all: usize = std.math.maxInt(isize);
const mtl4_visibility_device: usize = 1 << 0;
