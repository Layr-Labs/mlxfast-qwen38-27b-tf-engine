//! A one-draft round's draft token picked on the GPU at the end of the drafter's frame, so the verify queued behind it
//! starts without the host. The host still runs the selector and keeps nothing unless both picks agree.
const std = @import("std");
const mtl = @import("metal");
const topk = @import("core").bf16_topk;
pub const Ref = @import("core").draft_ops.Ref;
const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\struct Entry { uint token; float score; };
    \\struct Row { uint nonfinite; uint count; Entry entries[16]; };
    \\struct Params { uint anchor; uint rank; uint k; uint width; uint vocab; uint mapped; uint children; float edge; };
    \\static inline float bf(ushort x) { return as_type<float>(uint(x) << 16); }
    \\// One simdgroup a candidate: its unary score plus the edge weight times the rank-wide codebook product; then the
    \\// selector's order: the best score, ties in candidate order, the first `children` of those, the smallest token.
    \\kernel void tf_dflash_pick(device const Row* row [[buffer(0)]], device const uint* ids [[buffer(1)]],
    \\    device const ushort* projected [[buffer(2)]], device const ushort* predecessor [[buffer(3)]],
    \\    device const ushort* successor [[buffer(4)]], device uint* out [[buffer(5)]], constant Params& p [[buffer(6)]],
    \\    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    \\    threadgroup float keys[16];
    \\    threadgroup uint tokens[16];
    \\    const uint count = row->count;
    \\    const bool ok = row->nonfinite == 0 && count == p.k && p.k <= 16 && p.anchor < p.vocab;
    \\    if (sg < 16) {
    \\        float key = -INFINITY;
    \\        uint token = 0xffffffffu;
    \\        if (ok && sg < count) {
    \\            const uint column = row->entries[sg].token;
    \\            const uint id = column < p.width ? (p.mapped != 0 ? ids[column] : column) : 0xffffffffu;
    \\            if (id < p.vocab) {
    \\                device const ushort* a = predecessor + ulong(p.anchor) * p.rank;
    \\                device const ushort* c = successor + ulong(id) * p.rank;
    \\                float sum = 0.0f;
    \\                for (uint r = lane; r < p.rank; r += 32) sum += (bf(a[r]) * bf(projected[r])) * bf(c[r]);
    \\                sum = simd_sum(sum);
    \\                key = row->entries[sg].score + p.edge * sum;
    \\                token = id;
    \\            }
    \\        }
    \\        if (lane == 0) { keys[sg] = key; tokens[sg] = token; }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid != 0) return;
    \\    const uint n = ok ? count : 0;
    \\    bool any = false;
    \\    float top = -INFINITY;
    \\    for (uint i = 0; i < n; i++) if (tokens[i] != 0xffffffffu && isfinite(keys[i]) && (!any || keys[i] > top)) { top = keys[i]; any = true; }
    \\    uint pick = p.anchor;
    \\    if (any) {
    \\        uint seen = 0;
    \\        pick = 0xffffffffu;
    \\        for (uint i = 0; i < n && seen < p.children; i++) if (tokens[i] != 0xffffffffu && keys[i] == top) { pick = min(pick, tokens[i]); seen++; }
    \\    }
    \\    out[0] = pick;
    \\    out[1] = any ? 1u : 0u;
    \\}
    \\kernel void tf_dflash_pick_copy(device const uint* from [[buffer(0)]], device uint* to [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    \\    if (tid == 0) to[0] = from[0];
    \\}
;
pub const Params = extern struct { anchor: u32, rank: u32, k: u32, width: u32, vocab: u32, mapped: u32, children: u32, edge: f32 };
pub const Ops = struct {
    pick: mtl.Pipeline,
    copy: mtl.Pipeline,
    pub fn init(device: mtl.Device) !Ops {
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pick = try mtl.Pipeline.init(device, lib, "tf_dflash_pick", false);
        errdefer pick.deinit();
        return .{ .pick = pick, .copy = try mtl.Pipeline.init(device, lib, "tf_dflash_pick_copy", false) };
    }
    pub fn deinit(ops: Ops) void {
        ops.pick.deinit();
        ops.copy.deinit();
    }
    /// Row 0 of the top-k (`picks`), mapped through the head's token ids (null: columns are tokens), picked into out[0].
    pub fn encodePick(ops: Ops, e: mtl.ComputeEncoder, picks: Ref, ids: ?Ref, projected: Ref, predecessor: Ref, successor: Ref, out: Ref, p: Params) !void {
        if (p.k == 0 or p.k > topk.max_k or p.rank == 0 or p.anchor >= p.vocab or p.children == 0) return error.BadDraftPick;
        e.setPipeline(ops.pick);
        e.setBuffer(picks.buf, picks.off, 0);
        const map = ids orelse picks;
        e.setBuffer(map.buf, map.off, 1);
        e.setBuffer(projected.buf, projected.off, 2);
        e.setBuffer(predecessor.buf, predecessor.off, 3);
        e.setBuffer(successor.buf, successor.off, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.setValue(p, 6);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32 * topk.max_k, 1, 1));
    }
    /// One u32 from `from` to `to`, on the GPU.
    pub fn encodeCopy(ops: Ops, e: mtl.ComputeEncoder, from: Ref, to: Ref) void {
        e.setPipeline(ops.copy);
        e.setBuffer(from.buf, from.off, 0);
        e.setBuffer(to.buf, to.off, 1);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
    }
};
comptime {
    if (@sizeOf(topk.Row) != 8 + 8 * topk.max_k or topk.max_k != 16) @compileError("tf_dflash_pick's Row differs from bf16_topk.Row");
}
