// Standalone M32 preexpanded half-carrier RHS premise; default OFF and unregistered.
// Source basis a0481a: preserve literal16 state and geometry; change RHS representation only.
// REQUIRED HOST ADMISSION before ANY math dispatch: QUERY and RUN this EXACT
// pipeline instance via runtime abi[0], never a shadow or function-constant PSO.
// QUERY all128threads; requirecapacity16/all16valid/full2048unique32x64coords.
// Refuse RUN otherwise; query TRACE0/TRACE1/everymutant PSO separately.
// No production host guard/driver or M5 support is implemented here.
// Default OFF. Compile/query/GPU qualification requires a separate root grant.
#ifndef TF_NATIVE_GU_M32_HALFCARRIER_PROTOTYPE
#define TF_NATIVE_GU_M32_HALFCARRIER_PROTOTYPE 0
#endif
#ifndef TF_NATIVE_GU_M32_TRACE_C
#define TF_NATIVE_GU_M32_TRACE_C 0
#endif

#if TF_NATIVE_GU_M32_HALFCARRIER_PROTOTYPE
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Each W binding starts at the first requested 64-column tile. The canonical full
// sidecar is [Ntile64][G64=80][column=64][64 FP16 carrier words].
// Each half word has BIT PATTERN q (0..15), exactly q*2^-24, never numeric half(q).
// WG/WU bind expanded bytes, with production WU offset 178257920 bytes.
// Preparation remains weight-only and is outside the measured phase.
// SG/SU point at original folded (scale,bias) uint32 pairs. Their group stride is
// sb_stride (34816 production, 64 for the compact single-tile numerical fixture).
struct tf_native_gu_m32_halfcarrier_params {
  uint rows;
  uint x_stride;   // BF16 elements per input row; >=5120, <=INT32_MAX.
  uint sb_stride;  // packed (scale,bias) uint32 pairs per G64; >=tiles*64.
  uint y_stride;   // BF16 elements per activated row; >=tiles*64.
  uint sum_stride; // original common XS/HS row stride; >=rows.
  uint tiles;     // requested activated 64-column tiles; <=272.
};

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void tf_native_gu_m32_halfcarrier_prototype(
    const device bfloat* X [[buffer(0)]],
    const device uchar* WG [[buffer(1)]],
    const device uchar* WU [[buffer(2)]],
    const device uint* SG [[buffer(3)]],
    const device uint* SU [[buffer(4)]],
    const device float* XS [[buffer(5)]],
    device bfloat* Y [[buffer(6)]],
    device float* HS [[buffer(7)]],
    constant tf_native_gu_m32_halfcarrier_params& p [[buffer(8)]],
#if TF_NATIVE_GU_M32_TRACE_C
    device float* trace_gate [[buffer(9)]],
    device float* trace_up [[buffer(10)]],
#endif
    device uint* abi [[buffer(11)]],
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr uint K = 5120, GS = 64, GROUPS = K / GS;
  constexpr size_t GROUP_BYTES = 64 * 64 * sizeof(half);
  constexpr size_t TILE_BYTES = GROUPS * GROUP_BYTES;
  constexpr auto d64 = matmul2d_descriptor(32, 64, 64, false, true, false,
                                          matmul2d_descriptor::mode::multiply);
  matmul2d<d64, execution_simdgroups<4>> op;
  using XT = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
  using WT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
  // P stays API-owned. Cg/Cu are normal scalar arrays with ONLY literal indices.
  auto P = op.template get_destination_cooperative_tensor<XT, WT, float>();
  // Runtime ABI prefix, not a function constant. Same actual P in this PSO.
  // Host provides one disjoint guarded49668-u32 ABI buffer, sets mode1 QUERY,
  // validates all records, then changes ONLY mode to0 for RUN with same PSO.
  const uint abi_mode = abi[0];
  if (abi_mode == 1u) {
    constexpr uint ABI_VERSION = 1u, RECORD_WORDS = 4u + 128u * 3u;
    const uint tid = sgi * 32u + lane, base = 4u + tid * RECORD_WORDS;
    const uint capacity = uint(P.get_capacity());
    if (tid == 0u) { abi[1] = ABI_VERSION; abi[2] = 128u; abi[3] = RECORD_WORDS; }
    // Do not touch P values or X/W/SB/XS/Y/HS/trace/params in QUERY.
    abi[base] = capacity; abi[base + 1u] = sgi;
    abi[base + 2u] = lane; abi[base + 3u] = ABI_VERSION;
    for (uint i = 0; i < 128u; ++i) {
      const uint at = base + 4u + i * 3u;
      bool valid = false;
      if (i < capacity) valid = P.is_valid_element(ushort(i));
      if (valid) {
        const auto xy = P.get_multidimensional_index(ushort(i));
        abi[at] = 1u; abi[at + 1u] = uint(xy[0]); abi[at + 2u] = uint(xy[1]);
      } else {
        abi[at] = 0u; abi[at + 1u] = 0xffffffffu; abi[at + 2u] = 0xffffffffu;
      }
    }
    return; // Mode is TG-uniform; every thread returns before any math/barrier.
  }
  if (abi_mode != 0u) return; // Fail closed for unknown uniform modes.

  const size_t r0 = size_t(tg.y) * 32;
  // Uniform across all 128 threads. No divergent return precedes the H barrier.
  if (tg.x >= p.tiles || r0 >= p.rows) return;
  const uint M = min(p.rows - uint(r0), 32u);
  const uint n0 = tg.x * 64;
  const array<int32_t, 2> xstr = {1, int32_t(p.x_stride)};

  float Cg[16], Cu[16];
#define TF_GU_L16_APPLY(F) \
  F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) \
  F(8) F(9) F(10) F(11) F(12) F(13) F(14) F(15)
#define TF_GU_L16_ZERO(S) Cg[S] = 0.0f; Cu[S] = 0.0f;
  TF_GU_L16_APPLY(TF_GU_L16_ZERO)
#undef TF_GU_L16_ZERO

  // Each original G64 is one device-half-carrier K64 multiply. No cross-group dot.
  for (uint g = 0; g < GROUPS; ++g) {
    const device bfloat* xg = X + r0 * p.x_stride + g * GS;
    XT a((device bfloat*)xg, dextents<int32_t, 2>(64, int32_t(M)), xstr);
    const size_t wb = size_t(tg.x) * TILE_BYTES + g * GROUP_BYTES;
    WT bg((device half*)(WG + wb), dextents<int32_t, 2>(64, 64));
    WT bu((device half*)(WU + wb), dextents<int32_t, 2>(64, 64));

    op.run(a, bg, P);
#define TF_GU_L16_GATE(S) do { \
      constexpr ushort slot = ushort(S); \
      if (P.is_valid_element(slot)) { \
        const auto xy = P.get_multidimensional_index(slot); \
        const uint col = uint(xy[0]), m = uint(xy[1]); \
        if (m < M && col < 64) { \
          const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(SG[size_t(g) * p.sb_stride + n0 + col]); \
          const float xs = XS[size_t(g) * p.sum_stride + r0 + m]; \
          Cg[S] = fma(float(sb.x), P[slot], fma(float(sb.y), xs, Cg[S])); \
        } \
      } \
    } while (false);
    TF_GU_L16_APPLY(TF_GU_L16_GATE)
#undef TF_GU_L16_GATE

    op.run(a, bu, P);
#define TF_GU_L16_UP(S) do { \
      constexpr ushort slot = ushort(S); \
      if (P.is_valid_element(slot)) { \
        const auto xy = P.get_multidimensional_index(slot); \
        const uint col = uint(xy[0]), m = uint(xy[1]); \
        if (m < M && col < 64) { \
          const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(SU[size_t(g) * p.sb_stride + n0 + col]); \
          const float xs = XS[size_t(g) * p.sum_stride + r0 + m]; \
          Cu[S] = fma(float(sb.x), P[slot], fma(float(sb.y), xs, Cu[S])); \
        } \
      } \
    } while (false);
    TF_GU_L16_APPLY(TF_GU_L16_UP)
#undef TF_GU_L16_UP
  }

  // A single 4 KiB BF16 H array. Each API-owned valid (column,row) is published
  // once, followed by a uniform barrier before rows read all 64 values in order.
  threadgroup bfloat H[32 * 64];
#if TF_NATIVE_GU_M32_TRACE_C
#define TF_GU_L16_TRACE(S) \
    trace_gate[(r0 + m) * p.y_stride + n0 + col] = Cg[S]; \
    trace_up[(r0 + m) * p.y_stride + n0 + col] = Cu[S];
#else
#define TF_GU_L16_TRACE(S)
#endif
#define TF_GU_L16_PUBLISH(S) do { \
    constexpr ushort slot = ushort(S); \
    if (P.is_valid_element(slot)) { \
      const auto xy = P.get_multidimensional_index(slot); \
      const uint col = uint(xy[0]), m = uint(xy[1]); \
      if (m < M && col < 64) { \
        const bfloat gb = bfloat(Cg[S]), ub = bfloat(Cu[S]); \
        const float gf = float(gb); \
        const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * float(ub)); \
        Y[(r0 + m) * p.y_stride + n0 + col] = h; \
        H[m * 64 + col] = h; \
        TF_GU_L16_TRACE(S) \
      } \
    } \
  } while (false);
  TF_GU_L16_APPLY(TF_GU_L16_PUBLISH)
#undef TF_GU_L16_PUBLISH
#undef TF_GU_L16_TRACE
#undef TF_GU_L16_APPLY
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint t = sgi * 32 + lane;
  if (t < M) {
    float acc = 0.0f;
    for (uint col = 0; col < 64; ++col) acc += float(H[t * 64 + col]);
    HS[size_t(tg.x) * p.sum_stride + r0 + t] = acc;
  }
}
#endif // TF_NATIVE_GU_M32_HALFCARRIER_PROTOTYPE
