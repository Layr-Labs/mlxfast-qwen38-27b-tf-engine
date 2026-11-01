//! Weight-only low-rank preparation for the draft's low-precision fusion projection.
const std = @import("std");
const core = @import("core");
const mtl = @import("metal");
const split = @import("../projection.zig").split;
const Adapter = core.affine4_lane.Adapter;

extern "c" fn cblas_dsyrk(c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, f64, [*]f64, c_int) void;
extern "c" fn cblas_sgemm(c_int, c_int, c_int, c_int, c_int, c_int, f32, [*]const f32, c_int, [*]const f32, c_int, f32, [*]f32, c_int) void;
extern "c" fn dsyevd_([*]const u8, [*]const u8, *const c_int, [*]f64, *const c_int, [*]f64, [*]f64, *c_int, [*]c_int, *c_int, *c_int) void;

pub const rank: usize = 2048;
fn expand(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}
fn rounded(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

pub const Factors = struct {
    key: usize,
    right: Adapter,
    left: Adapter,

    pub fn deinit(f: *Factors) void {
        f.left.deinit();
        f.right.deinit();
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, order: core.lane_projection.RegOrder, key: usize, source: []const u16, n: usize, k: usize) !Factors {
        if (n != 5120 or k != 25600 or source.len != n * k) return error.UnsupportedFusionShape;
        const left = try a.alloc(f32, n * rank);
        defer a.free(left);
        try basis(a, source, n, k, left);
        var lhs = try adapter(a, device, order, left, n, rank);
        errdefer lhs.deinit();
        // Project the original matrix onto the retained left singular vectors in K panels.
        const rhs = try a.alloc(f32, rank * k);
        defer a.free(rhs);
        const panel = try a.alloc(f32, n * 512);
        defer a.free(panel);
        for (0..k / 512) |block| {
            for (0..n) |r| for (0..512) |c| {
                panel[r * 512 + c] = expand(source[r * k + block * 512 + c]);
            };
            cblas_sgemm(101, 112, 111, rank, 512, @intCast(n), 1, left.ptr, rank, panel.ptr, 512, 0, rhs.ptr + block * 512, @intCast(k));
        }
        return .{ .key = key, .right = try adapter(a, device, order, rhs, rank, k), .left = lhs };
    }
};

fn adapter(a: std.mem.Allocator, device: mtl.Device, order: core.lane_projection.RegOrder, values: []const f32, n: usize, k: usize) !Adapter {
    const bf = try a.alloc(u16, values.len);
    defer a.free(bf);
    for (bf, values) |*out, value| {
        if (!std.math.isFinite(value)) return error.InvalidFusionFactor;
        out.* = rounded(value);
    }
    var prepared = try core.affine4.prepare(a, bf, n, k);
    defer prepared.deinit();
    return Adapter.init(a, device, prepared, try split(n, k), order);
}

fn basis(a: std.mem.Allocator, source: []const u16, n: usize, k: usize, left: []f32) !void {
    const dim: c_int = @intCast(n);
    const gram = try a.alloc(f64, n * n);
    defer a.free(gram);
    @memset(gram, 0);
    const panel = try a.alloc(f64, n * 512);
    defer a.free(panel);
    for (0..k / 512) |block| {
        for (0..n) |r| for (0..512) |c| {
            panel[r * 512 + c] = expand(source[r * k + block * 512 + c]);
        };
        cblas_dsyrk(101, 121, 111, dim, 512, 1, panel.ptr, 512, 1, gram.ptr, dim);
    }
    // The row-major upper triangle is the column-major lower triangle for LAPACK.
    const eig = try a.alloc(f64, n);
    defer a.free(eig);
    var lwork: c_int = -1;
    var liwork: c_int = -1;
    var info: c_int = 0;
    var wq: [1]f64 = undefined;
    var iq: [1]c_int = undefined;
    dsyevd_("V", "L", &dim, gram.ptr, &dim, eig.ptr, &wq, &lwork, &iq, &liwork, &info);
    if (info != 0 or !std.math.isFinite(wq[0]) or wq[0] < 1 or wq[0] > std.math.maxInt(c_int) or iq[0] < 1) return error.FusionEigensolver;
    lwork = @intFromFloat(@ceil(wq[0]));
    liwork = iq[0];
    const work = try a.alloc(f64, @intCast(lwork));
    defer a.free(work);
    const iwork = try a.alloc(c_int, @intCast(liwork));
    defer a.free(iwork);
    dsyevd_("V", "L", &dim, gram.ptr, &dim, eig.ptr, work.ptr, &lwork, iwork.ptr, &liwork, &info);
    if (info != 0) return error.FusionEigensolver;
    for (0..rank) |col| {
        const vec = gram[(n - 1 - col) * n ..][0..n];
        for (vec, 0..) |v, r| left[r * rank + col] = @floatCast(v);
    }
}
