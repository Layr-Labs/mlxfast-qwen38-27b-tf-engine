//! Conditionally read the twelve highest-magnitude BF16 SwiGLU channels in each quantization group.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const op = @import("operators.zig");

const n = 5120;
const k = 17408;
const groups = k / 64;
const slots = 16;
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

    fn init(device: mtl.Device) !Pipelines {
        const library = try mtl.Library.fromSource(device, @embedFile("sparse_down.metal"), mtl.CompileOptions.mlx());
        defer library.deinit();
        const select = try mtl.Pipeline.init(device, library, "selectChannels", false);
        errdefer select.deinit();
        return .{ .select = select, .project = try mtl.Pipeline.init(device, library, "selectedDown", false) };
    }

    fn deinit(pipes: Pipelines) void {
        pipes.select.deinit();
        pipes.project.deinit();
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
        for (0..n / 64) |tile| for (0..groups) |group| {
            const base = tile * groups + group;
            for (0..32) |lane| {
                const row0 = tile * 64 + lane * 2;
                const first = row0 * groups + group;
                for (0..2) |j| {
                    const index = first + j * groups;
                    meta[base * 64 + lane * 2 + j] = (@as(u32, original.biases[index]) << 16) | original.scales[index];
                }
                for (0..8) |i| {
                    const lo = original.words[first * 8 + i];
                    const hi = original.words[(first + groups) * 8 + i];
                    for (0..8) |j| {
                        const shift: u5 = @intCast(4 * j);
                        dst[(base * 64 + i * 8 + j) * 32 + lane] = @intCast(((lo >> shift) & 15) | (((hi >> shift) & 15) << 4));
                    }
                }
            }
        };
        try set.weights.put(a, key, .{ .codes = codes, .metadata = metadata });
    }

    pub fn encode(set: *const Set, b: anytype, encoder: mtl.ComputeEncoder, weights: Weights, x: op.View, y: op.View) !void {
        const pipes = set.pipes orelse return error.DraftPreparationUnqualified;
        // Frame views own all transient storage. Two byte-sized indices fit per BF16 storage element.
        const ids = try b.frame.alloc(x.rows, groups * slots / 2, .bf16);
        const vals = try b.frame.alloc(x.rows, groups * slots, .f32);
        const input = try b.frame.ref(x);
        const output = try b.frame.ref(y);
        const indices = try b.frame.ref(ids);
        const values = try b.frame.ref(vals);
        b.access(encoder, &.{try b.span(x)}, &.{ try b.span(ids), try b.span(vals) });
        encoder.setPipeline(pipes.select);
        encoder.setBuffer(input.buf, input.off, 0);
        encoder.setBuffer(indices.buf, indices.off, 1);
        encoder.setBuffer(values.buf, values.off, 2);
        encoder.dispatchGroups(mtl.Size.of(groups, x.rows, 1), mtl.Size.of(64, 1, 1));

        b.access(encoder, &.{ try b.span(ids), try b.span(vals) }, &.{try b.span(y)});
        encoder.setPipeline(pipes.project);
        encoder.setBuffer(weights.codes, 0, 0);
        encoder.setBuffer(weights.metadata, 0, 1);
        encoder.setBuffer(indices.buf, indices.off, 2);
        encoder.setBuffer(values.buf, values.off, 3);
        encoder.setBuffer(output.buf, output.off, 4);
        encoder.dispatchGroups(mtl.Size.of(n / 64, x.rows, 1), mtl.Size.of(256, 1, 1));
    }
};
