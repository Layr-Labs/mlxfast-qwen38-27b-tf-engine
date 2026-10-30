// mlp_xs for prompt rows: 16 columns a thread, the same words and in-order group sums as mlp_xs.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
[[kernel]] void custom_kernel_q27_mlp_xs_wide_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_float(
  const device bfloat16_t* GATE [[buffer(0)]],
  const device bfloat16_t* UP [[buffer(1)]],
  const device int32_t* dims [[buffer(2)]],
  device bfloat16_t* HOUT [[buffer(3)]],
  device float* XS [[buffer(4)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
Q27_SPECIALIZE

  // Each thread takes 16 consecutive columns; a group's 64 values are summed in order from zero, its four threads
  // taking turns (as the norm's sums are), so every word matches one thread a column and a serial sum per group.
  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  constexpr int E = 16;
  const int column = (int(threadgroup_position_in_grid.x) * 64 + int(t)) * E;
  const int g = column / 64;
  if (int(m) >= M) {
    if ((t & 3) == 0) XS[g * MP + m] = 0.0f;
    return;
  }
  const int e = int(m) * N + column;
  const int fused = int(m) * 2 * N + column;  // GATE and UP point into the fused gate_up rows
  float xv[E];
  for (int i = 0; i < E; i++) {
    const float gf = float(GATE[fused + i]);
    const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * float(UP[fused + i]));
    HOUT[e + i] = h;
    xv[i] = float(h);
  }
  float gs = 0.0f;
  for (uint turn = 0; turn < 4; turn++) {
    if ((t & 3) == turn) for (int i = 0; i < E; i++) gs += xv[i];
    gs = simd_shuffle(gs, ushort((thread_index_in_simdgroup & ~3u) | turn));
  }
  if ((t & 3) == 0) XS[g * MP + m] = gs;
}
