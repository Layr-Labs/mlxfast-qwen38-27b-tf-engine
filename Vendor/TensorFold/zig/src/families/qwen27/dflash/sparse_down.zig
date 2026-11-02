//! Conditionally read the twelve highest-magnitude BF16 SwiGLU channels in each quantization group.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const op = @import("operators.zig");

const n = 5120;
const k = 17408;
const groups = k / 64;
const slots = 16;
const splits = 32;
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const Weights = struct {
    codes: mtl.Buffer,
    metadata: mtl.Buffer,

    fn deinit(w: Weights) void {
        w.codes.deinit();
        w.metadata.deinit();
    }
};
const Pipelines = struct {
    select: mtl.Pipeline,
    project: mtl.Pipeline,
    finish: mtl.Pipeline,

    fn init(device: mtl.Device) !Pipelines {
        const library = try mtl.Library.fromSource(device, @embedFile("sparse_down.metal"), mtl.CompileOptions.mlx());
        defer library.deinit();
        const select = try mtl.Pipeline.init(device, library, "selectChannels", false);
        errdefer select.deinit();
        if (select.simdWidth() != 32) return error.DraftPreparationUnqualified;
        const project = try mtl.Pipeline.init(device, library, "sectorDown", false);
        errdefer project.deinit();
        return .{ .select = select, .project = project, .finish = try mtl.Pipeline.init(device, library, "finishSector", false) };
    }

    fn deinit(pipes: Pipelines) void {
        pipes.select.deinit();
        pipes.project.deinit();
        pipes.finish.deinit();
    }
};

pub const Set = struct {
    weights: std.AutoHashMapUnmanaged(usize, Weights) = .empty,
    pipes: ?Pipelines = null,

    pub fn deinit(set: *Set, a: std.mem.Allocator) void {
        var it = set.weights.valueIterator();
        while (it.next()) |w| w.deinit();
        set.weights.deinit(a);
        if (set.pipes) |pipes| pipes.deinit();
    }

    pub fn prepare(set: *Set, a: std.mem.Allocator, device: mtl.Device, key: usize, original: core.affine4.Prepared) !void {
        if (original.n != n or original.k != k or original.bits != 4 or original.group != 64 or original.nibbles or set.weights.contains(key)) return error.DraftPreparationUnqualified;
        if (set.pipes == null) set.pipes = try Pipelines.init(device);
        const codes = try device.buffer(n * k / 2, options);
        errdefer codes.deinit();
        const metadata = try device.buffer(n * groups * 4, options);
        errdefer metadata.deinit();
        const dst = codes.slice(u8, n * k / 2);
        const meta = metadata.slice(u32, n * groups);
        // Eight adjacent outputs per SIMD lane put a whole 128-byte code span for
        // a selected activation together. Affine pairs follow the same output order.
        for (0..n / 256) |tile| for (0..groups) |group| {
            const base = tile * groups + group;
            for (0..128) |pair| {
                const row = tile * 256 + pair * 2;
                const first = row * groups + group;
                for (0..2) |j| {
                    const at = first + j * groups;
                    meta[base * 256 + pair * 2 + j] = (@as(u32, original.biases[at]) << 16) | original.scales[at];
                }
                for (0..8) |i| {
                    const lo = original.words[first * 8 + i];
                    const hi = original.words[(first + groups) * 8 + i];
                    for (0..8) |j| {
                        const shift: u5 = @intCast(j * 4);
                        dst[(base * 64 + i * 8 + j) * 128 + pair] = @intCast(((lo >> shift) & 15) | (((hi >> shift) & 15) << 4));
                    }
                }
            }
        };
        try set.weights.put(a, key, .{ .codes = codes, .metadata = metadata });
    }

    pub fn encode(set: *const Set, b: anytype, encoder: mtl.ComputeEncoder, weights: Weights, gate: op.View, up: op.View, y: op.View) !void {
        const pipes = set.pipes orelse return error.DraftPreparationUnqualified;
        // Frame views own all transient storage. Two byte-sized indices fit per BF16 storage element.
        const ids = try b.frame.alloc(gate.rows, groups * slots / 2, .bf16);
        const vals = try b.frame.alloc(gate.rows, groups * slots, .f32);
        const parts = try b.frame.alloc(gate.rows, splits * n, .f32);
        const partials = try b.frame.ref(parts);
        const gate_ref = try b.frame.ref(gate);
        const up_ref = try b.frame.ref(up);
        const output = try b.frame.ref(y);
        const indices = try b.frame.ref(ids);
        const values = try b.frame.ref(vals);
        b.access(encoder, &.{ try b.span(gate), try b.span(up) }, &.{ try b.span(ids), try b.span(vals) });
        encoder.setPipeline(pipes.select);
        encoder.setBuffer(gate_ref.buf, gate_ref.off, 0);
        encoder.setBuffer(up_ref.buf, up_ref.off, 1);
        encoder.setBuffer(indices.buf, indices.off, 2);
        encoder.setBuffer(values.buf, values.off, 3);
        encoder.setValue([2]u32{ gate.stride, up.stride }, 4);
        encoder.dispatchGroups(mtl.Size.of(groups, gate.rows, 1), mtl.Size.of(32, 1, 1));

        b.access(encoder, &.{ try b.span(ids), try b.span(vals) }, &.{try b.span(parts)});
        encoder.setPipeline(pipes.project);
        encoder.setBuffer(weights.codes, 0, 0);
        encoder.setBuffer(weights.metadata, 0, 1);
        encoder.setBuffer(indices.buf, indices.off, 2);
        encoder.setBuffer(values.buf, values.off, 3);
        encoder.setBuffer(partials.buf, partials.off, 4);
        encoder.dispatchGroups(mtl.Size.of(n / 256, gate.rows, splits / 4), mtl.Size.of(128, 1, 1));

        b.access(encoder, &.{try b.span(parts)}, &.{try b.span(y)});
        encoder.setPipeline(pipes.finish);
        encoder.setBuffer(partials.buf, partials.off, 0);
        encoder.setBuffer(output.buf, output.off, 1);
        encoder.dispatchThreads(mtl.Size.of(n / 4, gate.rows, 1), mtl.Size.of(256, 1, 1));
    }
};
