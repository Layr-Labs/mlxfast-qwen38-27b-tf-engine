//! Small operators use borrowed buffers and caller-owned encoders; copies never reinterpret activations.
const std = @import("std");
const mtl = @import("metal");
const source = @import("qwen_runtime_sources");
const Config = @import("config.zig").Config;
const abi = @import("glue.zig");
const projection = @import("projection.zig");
const profile = @import("core").gpu_profile;
const Ref = projection.Ref;

pub const Slice = extern struct { rows: u32, width: u32, stride: u32, offset: u32 };
const Add = extern struct { rows: u32, width: u32 };
pub const Gather = extern struct { rows: u32, width: u32, capacity: u32, planes: u32 };

pub const Kernels = struct {
    embed: mtl.Pipeline,
    head: mtl.Pipeline,
    slice: mtl.Pipeline,
    add: mtl.Pipeline,
    gather: mtl.Pipeline,
    take: mtl.Pipeline,
    touch: mtl.Pipeline,
    profiler: ?*profile.Trace = null,

    pub fn init(device: mtl.Device) !Kernels {
        const lib = try mtl.Library.fromSource(device, source.glue, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const copy = try mtl.Library.fromSource(device, source.copy, mtl.CompileOptions.mlx());
        defer copy.deinit();
        const embed = try mtl.Pipeline.init(device, lib, "q27_embed", false);
        errdefer embed.deinit();
        const head = try mtl.Pipeline.init(device, lib, "q27_head_norm", false);
        errdefer head.deinit();
        const slice = try mtl.Pipeline.init(device, copy, "q27_slice", false);
        errdefer slice.deinit();
        const add = try mtl.Pipeline.init(device, copy, "q27_add", false);
        errdefer add.deinit();
        const gather = try mtl.Pipeline.init(device, copy, "q27_tap_gather", false);
        errdefer gather.deinit();
        const take = try mtl.Pipeline.init(device, copy, "q27_take", false);
        errdefer take.deinit();
        const touch = try mtl.Pipeline.init(device, copy, "q27_touch", false);
        return .{ .embed = embed, .head = head, .slice = slice, .add = add, .gather = gather, .take = take, .touch = touch };
    }

    pub fn deinit(k: Kernels) void {
        inline for (.{ k.embed, k.head, k.slice, k.add, k.gather, k.take, k.touch }) |p| p.deinit();
    }

    /// Warms `w`'s leading bytes (codes, then scale/bias pairs) into the GPU's caches: encoded without a barrier beside
    /// a decoded latency-bound kernel (the GDN scan, attention), so the projection that follows it reads those lines from
    /// the system-level cache instead of DRAM. It writes nothing and nothing waits on it; up to touchBudget() bytes.
    pub fn warm(k: Kernels, e: mtl.ComputeEncoder, w: projection.Linear) void {
        var budget = touchBudget();
        if (budget == 0 or w.runs.len != 0) return;
        const words = @as(usize, w.n) * w.k * w.bits / 8;
        const pairs: usize = if (w.index == null) @as(usize, w.k / w.group) * w.n * 4 else 0;
        inline for (.{ .{ w.words, words }, .{ w.pairs, pairs } }) |span| {
            const bytes = @min(span[1], budget);
            budget -= bytes;
            const lines: u32 = @intCast(bytes / 128);
            if (lines != 0 and span[0].offset % 16 == 0) {
                e.setPipeline(k.touch);
                e.setBuffer(span[0].buffer, span[0].offset, 0);
                e.setValue([4]u32{ lines, 0x9E3779B9, 0xFFFFFFFF, 0 }, 1);
                e.setBuffer(span[0].buffer, span[0].offset, 2); // the never-taken store's target: never written
                e.dispatchThreads(mtl.Size.of(lines, 1, 1), mtl.Size.of(256, 1, 1));
            }
        }
    }

    /// Each row whose `takes` entry names an image row gets that row's bits from `images` over its embedding.
    pub fn imageRows(k: Kernels, e: mtl.ComputeEncoder, out: Ref, images: Ref, takes: Ref, width: u32, rows: u32) void {
        e.setPipeline(k.take);
        inline for (.{ out, images, takes }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue([2]u32{ rows, width }, 3);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn embedding(k: Kernels, input_encoder: mtl.ComputeEncoder, c: Config, p: projection.Triple, ids: Ref, out: Ref, rows: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .embedding);
        errdefer span.cancel();
        const e = span.encoder;

        const a = abi.Embed{ .width = @intCast(c.hidden), .rows = rows, .vocab = @intCast(c.vocab), .bits = p.bits, .group = p.group, .metadata = .bf16 };
        try abi.checkEmbed(a);
        e.setPipeline(k.embed);
        e.setBuffer(ids.buffer, ids.offset, 0);
        inline for (.{ p.weight, p.scales, p.biases }, 1..) |t, i| e.setBuffer(t.buffer, t.offset, i);
        e.setValue(a, 4);
        e.setBuffer(out.buffer, out.offset, 5);
        e.dispatchThreads(mtl.Size.of(c.hidden, rows, 1), mtl.Size.of(256, 1, 1));
        try span.end();
    }

    pub fn headNorm(k: Kernels, input_encoder: mtl.ComputeEncoder, x: Ref, weight: Ref, out: Ref, width: u32, rows: u32, eps: f32) !void {
        var span = profile.begin(k.profiler, input_encoder, .norm);
        errdefer span.cancel();
        const e = span.encoder;

        const a = abi.Head{ .width = width, .rows = rows, .stride = width, .activation = .bf16, .gain = .bf16, .eps = eps };
        e.setPipeline(k.head);
        bind(e, .{ x, weight });
        e.setValue(a, 2);
        e.setBuffer(out.buffer, out.offset, 3);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(try abi.headThreads(a), 1, 1));
        try span.end();
    }

    pub fn unstack(k: Kernels, input_encoder: mtl.ComputeEncoder, x: Ref, y: Ref, p: Slice) !void {
        var span = profile.begin(k.profiler, input_encoder, .copy);
        errdefer span.cancel();
        const e = span.encoder;

        if (p.rows == 0 or p.width == 0 or p.offset + p.width > p.stride) return error.BadSlice;
        e.setPipeline(k.slice);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setValue(p, 1);
        e.setBuffer(y.buffer, y.offset, 2);
        e.dispatchThreads(mtl.Size.of(p.width, p.rows, 1), mtl.Size.of(256, 1, 1));
        try span.end();
    }

    pub fn residual(k: Kernels, e: mtl.ComputeEncoder, x: Ref, r: Ref, out: Ref, width: u32, rows: u32) void {
        e.setPipeline(k.add);
        bind(e, .{ x, r });
        e.setValue(Add{ .width = width, .rows = rows }, 2);
        e.setBuffer(out.buffer, out.offset, 3);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn tapRows(k: Kernels, e: mtl.ComputeEncoder, x: Ref, kept: Ref, y: Ref, p: Gather) !void {
        if (p.rows == 0 or p.rows > p.capacity or p.planes != 5 or p.width == 0) return error.BadTapGather;
        if (x.offset > x.buffer.length() or @as(usize, p.capacity) * p.width * p.planes * 2 > x.buffer.length() - x.offset or kept.offset > kept.buffer.length() or @as(usize, p.rows) * 4 > kept.buffer.length() - kept.offset or y.offset > y.buffer.length() or @as(usize, p.rows) * p.width * p.planes * 2 > y.buffer.length() - y.offset) return error.BadTapGather;
        e.setPipeline(k.gather);
        bind(e, .{ x, kept });
        e.setValue(p, 2);
        e.setBuffer(y.buffer, y.offset, 3);
        e.dispatchThreads(mtl.Size.of(p.width * p.planes, p.rows, 1), mtl.Size.of(256, 1, 1));
    }
};

fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
}

/// Bytes a warm may touch: TF_Q27_TOUCH_MB (0 turns warming off), by default touch_mb.
fn touchBudget() usize {
    const mb = if (std.c.getenv("TF_Q27_TOUCH_MB")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch touch_mb else touch_mb;
    return mb << 20;
}
/// Default warm span: an M5 Max mixer output projection's codes (5120 x 6144 at 4 bits, 15 MiB) and most of its pairs.
const touch_mb = 24;
