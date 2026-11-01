// Runtime constants preserve the Python operator's projection-shaped arithmetic.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
[[kernel]] void custom_kernel_q27_norm_xs_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float(
  const device bfloat16_t* H [[buffer(0)]],
  const device bfloat16_t* R [[buffer(1)]],
  const device bfloat16_t* Wt [[buffer(2)]],
  const constant float* eps [[buffer(3)]],
  const device int32_t* dims [[buffer(4)]],
  device bfloat16_t* HO [[buffer(5)]],
  device bfloat16_t* XO [[buffer(6)]],
  device float* XS [[buffer(7)]],
  device atomic_uint* NC [[buffer(8)]],
  constant uint& wait_for [[buffer(9)]],
  device atomic_uint* PC [[buffer(10)]],
  constant uint& count_out [[buffer(11)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
Q27_SPECIALIZE

  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  constexpr int E = 16;
  constexpr int TPG = K / E;
  threadgroup float red[TPG / 32];
  if (int(m) >= M) return;
  // wait_for above 0: this norm shares its barrier interval with the projection that writes R, whose threadgroups count
  // themselves on NC[0] past a device-scope fence; a row's threadgroup reads R once the count reaches wait_for (bounded),
  // past its own fence, and the last row's threadgroup leaves both words zero for the next pair
  if (wait_for != 0u) {
    threadgroup uint dependency_ready;
    if (t == 0) {
      for (uint spin = 0; spin < (1u << 24) && atomic_load_explicit(&NC[0], memory_order_relaxed) < wait_for; spin++) {}
      dependency_ready = uint(atomic_load_explicit(&NC[0], memory_order_relaxed) >= wait_for);
      if (dependency_ready == 0u) {
        atomic_store_explicit(&NC[2], 1u, memory_order_relaxed);
      } else {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (atomic_fetch_add_explicit(&NC[1], 1u, memory_order_relaxed) + 1u == uint(M)) {
          atomic_store_explicit(&NC[0], 0u, memory_order_relaxed);
          atomic_store_explicit(&NC[1], 0u, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dependency_ready == 0u) return;
  }
  // the grid covers the real rows: row 0's threadgroup zeroes every padded row's group sums, as mlp_xs does
  if (int(m) == 0) {
    const int pad = MP - M;
    if (pad > 0) for (int p = int(t); p < (K / 64) * pad; p += TPG) XS[(p / pad) * MP + M + (p % pad)] = 0.0f;
  }
  const int base = int(m) * K + int(t) * E;
  // Every load before the first store, in 16-byte words (Kernels.norm binds 16-byte aligned rows and weights; a thread's
  // words start 32 bytes apart): HO, XO and XS may alias H, R and Wt as far as the compiler knows, so the scalar loop's
  // loads each waited for the store before them. The arithmetic is that loop's, word for word and in the same order.
  bfloat4 h4[E / 4], r4[E / 4], w4[E / 4];
  for (int j = 0; j < E / 8; j++) {
    const uint4 hw = ((const device uint4*)(H + base))[j], rw = ((const device uint4*)(R + base))[j];
    h4[2 * j] = as_type<bfloat4>(hw.xy);
    h4[2 * j + 1] = as_type<bfloat4>(hw.zw);
    r4[2 * j] = as_type<bfloat4>(rw.xy);
    r4[2 * j + 1] = as_type<bfloat4>(rw.zw);
  }
  for (int j = 0; j < E / 8; j++) {
    const uint4 ww = ((const device uint4*)(Wt + int(t) * E))[j];
    w4[2 * j] = as_type<bfloat4>(ww.xy);
    w4[2 * j + 1] = as_type<bfloat4>(ww.zw);
  }
  float hv[E];
  bfloat4 o4[E / 4];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    const bfloat h = bfloat(float(h4[i / 4][i % 4]) + float(r4[i / 4][i % 4]));
    o4[i / 4][i % 4] = h;
    hv[i] = float(h);
    ss = fma(hv[i], hv[i], ss);
  }
  for (int j = 0; j < E / 8; j++) ((device uint4*)(HO + base))[j] = uint4(as_type<uint2>(o4[2 * j]), as_type<uint2>(o4[2 * j + 1]));
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += red[i];
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  float xv[E];
  bfloat4 x4[E / 4];
  for (int i = 0; i < E; i++) {
    const bfloat x = bfloat(float(w4[i / 4][i % 4]) * (hv[i] * inv));
    x4[i / 4][i % 4] = x;
    xv[i] = float(x);
  }
  for (int j = 0; j < E / 8; j++) ((device uint4*)(XO + base))[j] = uint4(as_type<uint2>(x4[2 * j]), as_type<uint2>(x4[2 * j + 1]));
  // a group's 64 values summed in order from zero, as the lane projection's sums are: its four threads take turns
  float gs = 0.0f;
  for (uint turn = 0; turn < 4; turn++) {
    if ((t & 3) == turn) for (int i = 0; i < E; i++) gs += xv[i];
    gs = simd_shuffle(gs, ushort((thread_index_in_simdgroup & ~3u) | turn));
  }
  if ((t & 3) == 0) XS[(t >> 2) * MP + m] = gs;
  if (count_out != 0u) {
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(&PC[0], 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(&PC[2], 1u, memory_order_relaxed);
    }
  }

}
