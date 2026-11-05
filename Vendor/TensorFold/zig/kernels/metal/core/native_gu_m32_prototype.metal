// Standalone numerical premise only. Not registered in any production source factory.
// Default OFF. Compile/query/GPU qualification requires a separate root grant.
#ifndef TF_NATIVE_GU_M32_PROTOTYPE
#define TF_NATIVE_GU_M32_PROTOTYPE 0
#endif
#ifndef TF_NATIVE_GU_M32_TRACE_C
#define TF_NATIVE_GU_M32_TRACE_C 0
#endif

#if TF_NATIVE_GU_M32_PROTOTYPE
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Each W binding starts at the first requested 64-column tile. The canonical full
// sidecar is [Ntile64][G64=80][half=2][column=64][16 packed bytes], low nibble first.
// WG and WU may bind the same buffer at different offsets; no expanded weights.
// SG/SU point at original folded (scale,bias) uint32 pairs. Their group stride is
// sb_stride (34816 production, 64 for the compact single-tile numerical fixture).
struct tf_native_gu_m32_params {
  uint rows;
  uint x_stride;   // BF16 elements per input row; >=5120, <=INT32_MAX.
  uint sb_stride;  // packed (scale,bias) uint32 pairs per G64; >=tiles*64.
  uint y_stride;   // BF16 elements per activated row; >=tiles*64.
  uint sum_stride; // original common XS/HS row stride; >=rows.
  uint tiles;     // requested activated 64-column tiles; <=272.
};

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void tf_native_gu_m32_prototype(
    const device bfloat* X [[buffer(0)]],
    const device uchar* WG [[buffer(1)]],
    const device uchar* WU [[buffer(2)]],
    const device uint* SG [[buffer(3)]],
    const device uint* SU [[buffer(4)]],
    const device float* XS [[buffer(5)]],
    device bfloat* Y [[buffer(6)]],
    device float* HS [[buffer(7)]],
    constant tf_native_gu_m32_params& p [[buffer(8)]],
#if TF_NATIVE_GU_M32_TRACE_C
    device float* trace_gate [[buffer(9)]],
    device float* trace_up [[buffer(10)]],
#endif
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr uint K = 5120, GS = 64, GROUPS = K / GS;
  constexpr size_t HALF_BYTES = 64 * 16;
  constexpr size_t GROUP_BYTES = 2 * HALF_BYTES;
  constexpr size_t TILE_BYTES = GROUPS * GROUP_BYTES;
  const size_t r0 = size_t(tg.y) * 32;
  // Uniform across all 128 threads. No divergent return precedes the H barrier.
  if (tg.x >= p.tiles || r0 >= p.rows) return;
  const uint M = min(p.rows - uint(r0), 32u);
  const uint n0 = tg.x * 64;
  const array<int32_t, 2> xstr = {1, int32_t(p.x_stride)};

  constexpr auto dm = matmul2d_descriptor(32, 64, 32, false, true, false,
                                         matmul2d_descriptor::mode::multiply);
  constexpr auto da = matmul2d_descriptor(32, 64, 32, false, true, false,
                                         matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dm, execution_simdgroups<4>> opm;
  matmul2d<da, execution_simdgroups<4>> opa;
  using XT = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
  using WT = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>;
  // API-owned cooperative states, not float[16] arrays or a manual fragment ABI.
  auto P = opm.template get_destination_cooperative_tensor<XT, WT, float>();
  auto Cg = opm.template get_destination_cooperative_tensor<XT, WT, float>();
  auto Cu = opm.template get_destination_cooperative_tensor<XT, WT, float>();
  for (uint i = 0; i < Cg.get_capacity(); ++i) {
    const ushort slot = ushort(i);
    if (Cg.is_valid_element(slot)) Cg[slot] = 0.0f;
    if (Cu.is_valid_element(slot)) Cu[slot] = 0.0f;
  }

  // Every G64 keeps its own K32 multiply + K32 accumulate. No cross-group dot.
  for (uint g = 0; g < GROUPS; ++g) {
    const device bfloat* xg = X + r0 * p.x_stride + g * GS;
    XT a0((device bfloat*)xg, dextents<int32_t, 2>(32, int32_t(M)), xstr);
    XT a1((device bfloat*)(xg + 32), dextents<int32_t, 2>(32, int32_t(M)), xstr);
    const size_t wb = size_t(tg.x) * TILE_BYTES + g * GROUP_BYTES;
    WT bg0((device uchar*)(WG + wb), dextents<int32_t, 2>(32, 64));
    WT bg1((device uchar*)(WG + wb + HALF_BYTES), dextents<int32_t, 2>(32, 64));
    WT bu0((device uchar*)(WU + wb), dextents<int32_t, 2>(32, 64));
    WT bu1((device uchar*)(WU + wb + HALF_BYTES), dextents<int32_t, 2>(32, 64));

    opm.run(a0, bg0, P);
    opa.run(a1, bg1, P);
    for (uint i = 0; i < P.get_capacity(); ++i) {
      const ushort slot = ushort(i);
      if (!P.is_valid_element(slot)) continue;
      const auto xy = P.get_multidimensional_index(slot);
      const uint col = uint(xy[0]), m = uint(xy[1]);
      if (m >= M || col >= 64) continue;
      // Explicitly round the restored carrier P to FP32 before the original FMA.
      // Power-of-two restoration is not a proof of arbitrary-input/M5 parity.
      P[slot] *= 0x1p-24f;
      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(SG[size_t(g) * p.sb_stride + n0 + col]);
      const float xs = XS[size_t(g) * p.sum_stride + r0 + m];
      Cg[slot] = fma(float(sb.x), P[slot], fma(float(sb.y), xs, Cg[slot]));
    }

    opm.run(a0, bu0, P);
    opa.run(a1, bu1, P);
    for (uint i = 0; i < P.get_capacity(); ++i) {
      const ushort slot = ushort(i);
      if (!P.is_valid_element(slot)) continue;
      const auto xy = P.get_multidimensional_index(slot);
      const uint col = uint(xy[0]), m = uint(xy[1]);
      if (m >= M || col >= 64) continue;
      P[slot] *= 0x1p-24f;
      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(SU[size_t(g) * p.sb_stride + n0 + col]);
      const float xs = XS[size_t(g) * p.sum_stride + r0 + m];
      Cu[slot] = fma(float(sb.x), P[slot], fma(float(sb.y), xs, Cu[slot]));
    }
  }

  // A single 4 KiB BF16 H array. Each API-owned valid (column,row) is published
  // once, followed by a uniform barrier before rows read all 64 values in order.
  threadgroup bfloat H[32 * 64];
  for (uint i = 0; i < Cg.get_capacity(); ++i) {
    const ushort slot = ushort(i);
    if (!Cg.is_valid_element(slot)) continue;
    const auto xy = Cg.get_multidimensional_index(slot);
    const uint col = uint(xy[0]), m = uint(xy[1]);
    if (m >= M || col >= 64) continue;
    const bfloat gb = bfloat(Cg[slot]), ub = bfloat(Cu[slot]);
    const float gf = float(gb);
    const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * float(ub));
    Y[(r0 + m) * p.y_stride + n0 + col] = h;
    H[m * 64 + col] = h;
#if TF_NATIVE_GU_M32_TRACE_C
    trace_gate[(r0 + m) * p.y_stride + n0 + col] = Cg[slot];
    trace_up[(r0 + m) * p.y_stride + n0 + col] = Cu[slot];
#endif
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint t = sgi * 32 + lane;
  if (t < M) {
    float acc = 0.0f;
    for (uint col = 0; col < 64; ++col) acc += float(H[t * 64 + col]);
    HS[size_t(tg.x) * p.sum_stride + r0 + t] = acc;
  }
}
#endif // TF_NATIVE_GU_M32_PROTOTYPE
