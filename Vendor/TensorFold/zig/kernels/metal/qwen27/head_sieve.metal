// The greedy verify head's sieve (head_sieve.zig): a 2-bit copy of the vocabulary head's codes (code >> 2, the same
// scales and biases) gives each (row, column) an approximate logit a and the spread s of the dropped bits' term (the
// square root of 1.25 times the sum over groups of scale^2 times the group's sum of squares: 1.25 is the variance of a
// uniform 2-bit residual about its middle 1.5). Each row's first tile (the 32-column tile of its best a) is recomputed
// exactly by the head's own tile-list launch; the row's threshold is the lower of that tile's exact best and its best a,
// and every tile whose a + k s reaches some row's threshold is recomputed the same way. Each row's token is the best of
// those exact words: the highest bf16 logit, ties to the lowest id, tf_round_bf16_argmax's rule. A row's true best u
// escapes only if its dropped bits' term exceeds both k s_u and that of the first tile's best-a column by k s_u.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// head_sieve.zig prepends the head's shape and metadata form; the defaults are the 27B head's (12-bit indices).
#ifndef TF_SIEVE_SHAPED
#define TF_N 248320
#define TF_K 5120
#define TF_TB 8
#define TF_SBI 4096
#define TF_SBI12 1
#define TF_SIEVE_KAPPA 10.0f
#endif

constant constexpr int NT = 32, GS = 64, KG = TF_K / GS, TILES = TF_N / NT;
constant constexpr float SLACK_REL = 1.0f / 128.0f; // a bf16 logit's spacing at most, relative to its size
constant constexpr float SLACK_ABS = 1e-2f;         // the f32 sums' rounding, generously

// One 4-bit (tile, group) block of the head's reg words (TF_TB tiles interleaved by group: a lane's words 0-3, then
// its words 4-7 32 lanes on) into the 2-bit reg block of tile_batch 1 (lane_projection.halveReg's form): word j keeps
// codes >> 2 of word j in bits 0-1 of each nibble and those of word j + 4 in bits 2-3, so the 2-bit kernel's two masks
// give back each word's codes >> 2 in place. One thread a lane's uint4 of a block: tiles * KG * 32 threads.
kernel void tf_head_sieve_pack(const device uint4* W [[buffer(0)]], device uint4* W2 [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t >= uint(TILES * KG * 32)) return;
  const uint block = t / 32, lane = t % 32, tile = block / KG, g = block % KG;
  const size_t wide = ((size_t)(tile / TF_TB) * KG + g) * TF_TB + tile % TF_TB;
  const uint4 w0 = W[wide * 64 + lane], w1 = W[wide * 64 + 32 + lane];
  W2[t] = ((w0 >> 2) & 0x33333333u) | (w1 & 0xCCCCCCCCu);
}

// Each group's sums of squares of the rows (XQ[g * 16 + row], zero past `rows`, the stride of the norm's sums for at
// most 8 rows), each group's values added in order from zero; the candidate list and its dispatch words cleared.
kernel void tf_head_sieve_prep(const device bfloat* X [[buffer(0)]], constant int& rows [[buffer(1)]], device float* XQ [[buffer(2)]],
    device uint* list [[buffer(3)]], device uint* args [[buffer(4)]], uint t [[thread_position_in_grid]]) {
  if (t == 0) {
    list[0] = 0;
    args[0] = 0;
    args[1] = 1;
    args[2] = 1;
  }
  if (t >= uint(KG * 16)) return;
  const int g = int(t) / 16, row = int(t) % 16;
  float acc = 0.0f;
  if (row < rows) {
    const device bfloat* x = X + (size_t)row * TF_K + g * GS;
    for (int i = 0; i < GS; i++) {
      const float v = float(x[i]);
      acc = fma(v, v, acc);
    }
  }
  XQ[t] = acc;
}

#if defined(TF_SBI)
inline uint4 tf_sieve_look(const device uint* tab, uint2 ix) {
  return uint4(tab[ix.x & 0xFFFFu], tab[ix.x >> 16], tab[ix.y & 0xFFFFu], tab[ix.y >> 16]);
}
#endif
#if defined(TF_SBI) && !defined(TF_SBI12)
// The scan's table for 16-bit indices: each metadata pair (s, b) as float2(4 s, fma(1.5, t, b)), t = s / 2^24, the
// epilogue's first products formed by the expressions the epilogue used to form per column and group. head_sieve.zig
// fills it once, a thread an entry.
kernel void tf_head_sieve_table(const device uint* SB [[buffer(0)]], device float2* T [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t >= uint(TF_SBI)) return;
  const vec<bfloat, 2> v = as_type<vec<bfloat, 2>>(SB[t]);
  const float s0 = float(v.x), b0 = float(v.y), t0 = s0 * (1.0f / 16777216.0f);
  T[t] = float2(4.0f * s0, fma(1.5f, t0, b0));
}
inline void tf_sieve_lookf(const device float2* tab, uint2 ix, thread float2* q) {
  q[0] = tab[ix.x & 0xFFFFu];
  q[1] = tab[ix.x >> 16];
  q[2] = tab[ix.y & 0xFFFFu];
  q[3] = tab[ix.y >> 16];
}
#endif
#if defined(TF_SBI12)
inline uint4 tf_sieve_look12a(const device uint* tab, uint3 ix) {
  return uint4(tab[ix.x & 0xFFFu], tab[(ix.x >> 12) & 0xFFFu], tab[(ix.x >> 24) | ((ix.y & 0xFu) << 8)], tab[(ix.y >> 4) & 0xFFFu]);
}
inline uint4 tf_sieve_look12b(const device uint* tab, uint3 ix) {
  return uint4(tab[(ix.y >> 16) & 0xFFFu], tab[(ix.y >> 28) | ((ix.z & 0xFFu) << 4)], tab[(ix.z >> 8) & 0xFFFu], tab[ix.z >> 20]);
}
#endif

// The sieve: a 32-column tile a threadgroup (one SIMD group), every group of the 2-bit copy through tf_lane_reg's
// decoded-row loop (rows8, one K split, codes as fp16 subnormals in the tensor op's right input, the head's own
// metadata words), with a = 4 s * (codes >> 2 product) + (b + 1.5 s) * row sum and s^2 * the row's sum of squares beside
// it. Each (tile, row)'s best a + k s + slack into HM[tile * 16 + row] and its best a into HM[tile * 16 + 8 + row].
// dims: rows, the sums' stride (16). Grid TILES threadgroups of 32.
[[max_total_threads_per_threadgroup(32)]]
[[kernel]] void tf_head_sieve_scan(const device bfloat* X [[buffer(0)]], const device uint4* W [[buffer(1)]], const device bfloat* SB [[buffer(2)]],
    constant int2& dims [[buffer(3)]], device float* HM [[buffer(4)]], const device float* XS [[buffer(5)]], const device float* XQ [[buffer(6)]], const device float2* TF2 [[buffer(7)]],
    uint lanei [[thread_index_in_simdgroup]], uint tg [[threadgroup_position_in_grid]]) {
  const int lane = int(lanei), M = min(dims.x, 8), stride = dims.y, tile = int(tg), n0 = tile * NT;
  if (tile >= TILES || M <= 0) return;
  const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)X, dextents<int32_t, 2>(TF_K, M));
  auto a0 = a.slice(0, 0);
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto P = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a0)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  thread half2* b2 = (thread half2*)&cb[0];
  const device uint4* sbv = (const device uint4*)SB;
  // a lane's columns nb + j (C[j], V[j]) and nb + 16 + j (C[4 + j], V[4 + j]) of row mb
  float C[8], V[8];
  for (int i = 0; i < 8; i++) {
    C[i] = 0.0f;
    V[i] = 0.0f;
  }
  const device uint4* wp = W + (size_t)tile * KG * 32 + lane;
  uint4 nw = wp[0];
#if defined(TF_SBI12)
  const device uint* tab = (const device uint*)SB;
  const device packed_uint3* ixs = (const device packed_uint3*)(tab + TF_SBI) + n0 / 8 + nb / 4;
  uint3 ni = ixs[0];
#elif defined(TF_SBI)
  const device uint* tab = (const device uint*)SB;
  const device uint2* ixs = (const device uint2*)(tab + TF_SBI) + (n0 + nb) / 4;
  uint2 ni0 = ixs[0], ni1 = ixs[4];
#endif
  const array<int32_t, 2> xstr = {1, TF_K};
  for (int g = 0; g < KG; g++) {
#if defined(TF_SBI12)
    const uint4 q0 = tf_sieve_look12a(tab, ni), q1 = tf_sieve_look12b(tab, ni);
#elif defined(TF_SBI)
    // 16-bit indices: the column's (4 s, fma(1.5, t, b)) from the scan's table (tf_head_sieve_table)
    float2 f0[4], f1[4];
    tf_sieve_lookf(TF2, ni0, f0);
    tf_sieve_lookf(TF2, ni1, f1);
#else
    const uint4 q0 = sbv[((size_t)g * TF_N + n0 + nb) / 4], q1 = sbv[((size_t)g * TF_N + n0 + nb + 16) / 4];
#endif
    const uint4 cw = nw;
    const float x0 = XS[g * stride + mb], xq = XQ[g * 16 + mb];
    for (int w = 0; w < 8; w++) {
      // codes >> 2 as fp16 subnormals (bits = code, value code * 2^-24); the scales carry the 2^24 back
      const uint word = w < 4 ? (cw[w] & 0x33333333u) : ((cw[w - 4] >> 2) & 0x33333333u);
      const int e = 16 * (w & 1) + 2 * (w >> 1);
      b2[e] = as_type<half2>(word & 0x000F000Fu);
      b2[e + 1] = as_type<half2>((word >> 4) & 0x000F000Fu);
      b2[e + 8] = as_type<half2>((word >> 8) & 0x000F000Fu);
      b2[e + 9] = as_type<half2>((word >> 12) & 0x000F000Fu);
    }
    // the next group's codes and indices under this one's tensor op (the last group reloads its own, unread)
    {
      const int next = min(g + 1, KG - 1);
      nw = wp[(size_t)next * 32];
#if defined(TF_SBI12)
      ni = ixs[(size_t)next * (TF_N / 8)];
#elif defined(TF_SBI)
      ni0 = ixs[(size_t)next * (TF_N / 4)];
      ni1 = ixs[(size_t)next * (TF_N / 4) + 4];
#endif
    }
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> x((device bfloat*)X + g * GS, dextents<int32_t, 2>(GS, M), xstr);
    op.run(x, cb, P);
#if defined(TF_SBI) && !defined(TF_SBI12)
    // the same words as the per-column form below: C from the table's products; V gathers (4 s)^2 = 2^52 t^2, an exact
    // power-of-two multiple of each term, scaled back after the loop
    for (int j = 0; j < 4; j++) {
      C[j] = fma(f0[j].x, P[j], fma(f0[j].y, x0, C[j]));
      C[4 + j] = fma(f1[j].x, P[8 + j], fma(f1[j].y, x0, C[4 + j]));
      V[j] = fma(f0[j].x * f0[j].x, xq, V[j]);
      V[4 + j] = fma(f1[j].x * f1[j].x, xq, V[4 + j]);
    }
#else
    for (int j = 0; j < 4; j++) {
      const vec<bfloat, 2> v0 = as_type<vec<bfloat, 2>>(q0[j]), v1 = as_type<vec<bfloat, 2>>(q1[j]);
      const float s0 = float(v0.x), b0 = float(v0.y), s1 = float(v1.x), b1 = float(v1.y); // scales stored times 2^24
      const float t0 = s0 * (1.0f / 16777216.0f), t1 = s1 * (1.0f / 16777216.0f);
      // code = 4 (code >> 2) + 1.5 + r: scale 4 s on the 2-bit product, bias b + 1.5 s on the row sum
      C[j] = fma(4.0f * s0, P[j], fma(fma(1.5f, t0, b0), x0, C[j]));
      C[4 + j] = fma(4.0f * s1, P[8 + j], fma(fma(1.5f, t1, b1), x0, C[4 + j]));
      V[j] = fma(t0 * t0, xq, V[j]);
      V[4 + j] = fma(t1 * t1, xq, V[4 + j]);
    }
#endif
  }
#if defined(TF_SBI) && !defined(TF_SBI12)
  for (int i = 0; i < 8; i++) V[i] *= 0x1p-52f;
#endif
  // fmax drops a NaN operand: a row of NaN logits keeps -inf here, its first tile 0 and every tile listed
  float hi = -INFINITY, mid = -INFINITY;
  for (int i = 0; i < 8; i++) {
    const float v = C[i], ks = TF_SIEVE_KAPPA * sqrt(1.25f * V[i]);
    hi = fmax(hi, v + ks + fma(fabs(v) + ks, SLACK_REL, SLACK_ABS));
    mid = fmax(mid, v);
  }
  // the four lanes of row mb differ in lane bits 0 and 3
  hi = fmax(hi, simd_shuffle_xor(hi, 1));
  hi = fmax(hi, simd_shuffle_xor(hi, 8));
  mid = fmax(mid, simd_shuffle_xor(mid, 1));
  mid = fmax(mid, simd_shuffle_xor(mid, 8));
  if ((lane & 9) == 0 && mb < M) {
    HM[(size_t)tile * 16 + mb] = hi;
    HM[(size_t)tile * 16 + 8 + mb] = mid;
  }
}

// Each row's first tile: the tile of its highest best a (ties: the lower tile; none finite: tile 0) into
// first[1 + row], with first[0] = rows (the tile list the head's tile-list launch reads). One threadgroup of 1024.
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void tf_head_sieve_top(const device float* HM [[buffer(0)]], constant int& rows [[buffer(1)]], device uint* first [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float bv[32][8];
  threadgroup uint bt[32][8];
  float best[8];
  uint at[8];
  for (int r = 0; r < 8; r++) {
    best[r] = -INFINITY;
    at[r] = 0xffffffffu;
  }
  // each thread's tiles in increasing order: a later equal value never displaces an earlier tile
  for (uint t = tid; t < uint(TILES); t += 1024)
    for (int r = 0; r < 8; r++) {
      const float v = HM[(size_t)t * 16 + 8 + r];
      if (r < rows && v > best[r]) {
        best[r] = v;
        at[r] = t;
      }
    }
  for (int r = 0; r < 8; r++) {
    const float m = simd_max(best[r]);
    const uint t = simd_min(best[r] == m ? at[r] : 0xffffffffu);
    if (lane == 0) {
      bv[sg][r] = m;
      bt[sg][r] = t;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < uint(rows) && tid < 8) {
    float m = bv[0][tid];
    uint t = bt[0][tid];
    for (int s = 1; s < 32; s++) {
      const float o = bv[s][tid];
      const uint ot = bt[s][tid];
      if (o > m || (o == m && ot < t)) {
        m = o;
        t = ot;
      }
    }
    first[1 + tid] = t == 0xffffffffu ? 0u : t;
  }
  if (tid == 0) first[0] = uint(rows);
}

// Each row's threshold: the lower of its first tile's exact best (Y1: the tile-list launch's words, row r's first
// tile at slot r of rows 256 apart; a bound of the row's top logit from below) and that tile's best a; then every tile
// some row's best a + k s + slack reaches into list (list[0] its count, then the tiles in any order), counted on the
// indirect dispatch's first word as well. A row's first tile always lands (its best a + k s is its best a or more).
// Grid TILES / 256 threadgroups of 256.
kernel void tf_head_sieve_select(const device float* HM [[buffer(0)]], const device ushort* Y1 [[buffer(1)]], const device uint* first [[buffer(2)]],
    constant int& rows [[buffer(3)]], device atomic_uint* list [[buffer(4)]], device atomic_uint* args [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint group [[threadgroup_position_in_grid]]) {
  threadgroup float thr[8];
  if (tid < 8) {
    float e = -INFINITY;
    if (int(tid) < rows) {
      const device ushort* y = Y1 + tid * 256 + tid * 32;
      for (int c = 0; c < 32; c++) {
        const float v = as_type<float>(uint(y[c]) << 16);
        if (v > e) e = v;
      }
      e = fmin(e, HM[(size_t)first[1 + tid] * 16 + 8 + tid]);
    }
    thr[tid] = e;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint t = group * 256 + tid;
  bool hit = false;
  if (t < uint(TILES))
    for (int r = 0; r < 8; r++) hit = hit || (r < rows && HM[(size_t)t * 16 + r] >= thr[r]);
  const uint n = simd_sum(uint(hit)), at = simd_prefix_exclusive_sum(uint(hit));
  uint base = 0;
  if (lane == 0 && n != 0) {
    base = atomic_fetch_add_explicit(list, n, memory_order_relaxed);
    atomic_fetch_add_explicit(args, n, memory_order_relaxed);
  }
  base = simd_shuffle(base, 0);
  if (hit) atomic_store_explicit(list + 1 + base + at, t, memory_order_relaxed);
}

// Each row's token from the listed tiles' exact words (Y: the tile-list launch's, slot j's 32 columns at 32 j of rows
// TF_N apart, slot j holding tile list[1 + j]): the highest finite value, ties to the lowest id, and the nonfinite
// mask (1 NaN, 2 +inf, 4 -inf) of the words read, as tf_round_bf16_argmax picks a full row. A threadgroup of 1024 a row.
[[max_total_threads_per_threadgroup(1024)]]
[[kernel]] void tf_head_sieve_pick(const device ushort* Y [[buffer(0)]], const device uint* list [[buffer(1)]], device uint2* picks [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32], bm[32];
  const uint cells = min(list[0], uint(TILES)) * 32;
  float best = -INFINITY;
  uint id = 0xffffffffu, mask = 0;
  for (uint cell = tid; cell < cells; cell += 1024) {
    const float value = as_type<float>(uint(Y[(size_t)row * TF_N + cell]) << 16);
    const uint token = list[1 + cell / 32] * 32 + cell % 32;
    if (!isfinite(value)) mask |= isnan(value) ? 1u : value > 0 ? 2u : 4u;
    else if (id == 0xffffffffu || value > best || (value == best && token < id)) {
      best = value;
      id = token;
    }
  }
  // a finite value is above -inf, so best == m keeps only real candidates (or none: id stays invalid)
  {
    const float m = simd_max(best);
    const uint t = simd_min(best == m ? id : 0xffffffffu);
    const uint k = simd_or(mask);
    if (lane == 0) {
      bv[sg] = m;
      bi[sg] = t;
      bm[sg] = k;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float m = bv[0];
    uint t = bi[0], k = bm[0];
    for (int s = 1; s < 32; s++) {
      if (bi[s] != 0xffffffffu && (t == 0xffffffffu || bv[s] > m || (bv[s] == m && bi[s] < t))) {
        m = bv[s];
        t = bi[s];
      }
      k |= bm[s];
    }
    picks[row] = uint2(k != 0 ? 0xffffffffu : t, k);
  }
}
