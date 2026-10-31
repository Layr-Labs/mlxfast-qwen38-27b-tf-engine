#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// A decoded rows' MLP block (at most 8 rows) in one dispatch: the fused gate/up projection with SwiGLU in its epilogue
// (tf_lane_reg's TF_SWIGLU, TF_SERIAL_SK2, TF_ROWS8, TF_REG_LEAN, TF_TB 8, TF_SBI(12) path for 34816 x 5120), then the
// down projection (tf_lane_reg's TF_ROWS8, TF_REG_LEAN, TF_TB 8, TF_SBI(12) path, four K chains, for 5120 x 17408), without the
// barrier between them. Each threadgroup takes a ticket when it starts: tickets [0, 272) are gate/up threadgroups (each
// writes one 64-column group of activations and that group's row sums, then publishes the group's flag to every down
// tile), tickets [272, 432) are down tiles, whose four chains wait for their groups' flags before they read the groups'
// sums and activations. Every gate/up ticket is taken before any down ticket, by a threadgroup that is already running, so a
// waiting chain only waits for running threadgroups. Every product, FMA and sum is tf_lane_reg's, in the same order;
// only the order in which threadgroups produce the groups differs (ticket t makes group (t % 4) * 68 + t / 4, so the
// four chains' groups arrive together).
//
// sync: [0] the ticket counter, [1] the finished count, then from word MF_FLAGS a flag per down tile and group
// ([tile][group]: each chain polls words no other tile polls, rather than every chain one hot word). The last threadgroup
// to finish clears both counters for the next dispatch; a flag holds the epoch of the dispatch that published its group.
// MF_GU12 / MF_DN12: the gate/up / down metadata in lane_projection.indexed12_layout (1) or indexed_layout (0).
// MF_PAUSE: integer steps a failed poll waits before the next.
// MF_CH: groups a down chain waits for at once.

#ifndef MF_GU12
#define MF_GU12 0
#endif
#ifndef MF_DN12
#define MF_DN12 0
#endif
#ifndef MF_CH
#define MF_CH 4
#endif
#ifndef MF_PAUSE
#define MF_PAUSE 0
#endif

constant constexpr int NT = 32, GS = 64, TB = 8;
constant constexpr int GU_N = 34816, GU_K = 5120, GU_KG = GU_K / GS, GU_GROUPS = GU_N / 128; // 272 threadgroups
constant constexpr int DN_N = 5120, DN_K = 17408, DN_KG = DN_K / GS, DN_SK = 4, DN_TILES = DN_N / NT; // 160 tiles
constant constexpr int MF_GROUPS = GU_GROUPS + DN_TILES, MF_FLAGS = 32; // the counters' own 128 bytes
static_assert(GU_GROUPS == DN_KG, "each gate/up threadgroup writes one group of the down projection's K");
static_assert(DN_KG % DN_SK == 0, "the down chains split K evenly");

inline uint4 mf_look(const device uint* tab, uint2 ix) {
  return uint4(tab[ix.x & 0xFFFFu], tab[ix.x >> 16], tab[ix.y & 0xFFFFu], tab[ix.y >> 16]);
}
inline uint4 mf_look12a(const device uint* tab, uint3 ix) {
  return uint4(tab[ix.x & 0xFFFu], tab[(ix.x >> 12) & 0xFFFu], tab[(ix.x >> 24) | ((ix.y & 0xFu) << 8)], tab[(ix.y >> 4) & 0xFFFu]);
}
inline uint4 mf_look12b(const device uint* tab, uint3 ix) {
  return uint4(tab[(ix.y >> 16) & 0xFFFu], tab[(ix.y >> 28) | ((ix.z & 0xFFu) << 4)], tab[(ix.z >> 8) & 0xFFFu], tab[ix.z >> 20]);
}

// A down chain's wait for groups [g0, g1): every lane polls each flag with a read-modify-write that adds nothing, then a
// device fence orders the groups' sums and activations after them. A relaxed atomic load can keep returning a stale
// cached word (polls stalled for seconds on Apple GPUs); the read-modify-write always reads the current one. A failed
// poll first runs MF_PAUSE dependent integer steps.
inline void mf_wait(device atomic_uint* flags, int g0, int g1, uint epoch, threadgroup float* sink) {
  uint spin = epoch;
  for (int g = g0; g < g1; g++)
    while (atomic_fetch_or_explicit(flags + g, 0u, memory_order_relaxed) != epoch)
      for (int i = 0; i < MF_PAUSE; i++) spin = spin * 1664525u + 1013904223u;
  if (spin == 0x9E3779B9u && epoch == 0u) sink[0] = 0.0f; // never true (epochs start at 1): keeps the pause
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

// One reg chain over groups [first, last) of a 4-bit g64 tile, decoded rows: tf_lane_reg's TF_REG_LEAN TF_ROWS8 loop,
// TF_SBI12 (lane_projection.indexed12_layout, a 4096-word table) when TWELVE, else TF_SBI (indexed_layout, 8192 words).
// C[i] for the live halves (i & 4) == 0. WAIT: a down chain, which waits for its groups' flags, MF_CH groups at a time,
// before reading their row sums and activations.
template <int N, int K, bool TWELVE, bool WAIT>
inline void mf_chain(thread float* C, const device bfloat* X, const device uint4* W, const device uint* SB,
                     const device float* XS, int stride, int M, int tile, int first, int last, int lane,
                     device atomic_uint* flags, uint epoch, threadgroup float* sink) {
  constexpr int KG = K / GS;
  const int n0 = tile * NT;
  const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)X, dextents<int32_t, 2>(K, M));
  auto a0 = a.slice(0, 0);
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto P = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a0)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  thread half2* b2 = (thread half2*)&cb[0];
  const device uint4* wl = W + ((size_t)(tile / TB) * KG * TB + tile % TB) * 64 + lane;
  const half2 bias = half2(1024.0h);
  for (int i = 0; i < 16; i++) C[i] = 0.0f;
  uint4 nw0 = 0, nw1 = 0;
  if (first < last) { nw0 = wl[(size_t)first * TB * 64]; nw1 = wl[(size_t)first * TB * 64 + 32]; }
  const device uint* tab = SB;
  const device packed_uint3* ix12 = (const device packed_uint3*)(tab + 4096) + n0 / 8 + nb / 4;
  const device uint2* ix16 = (const device uint2*)(tab + 8192) + (n0 + nb) / 4;
  uint3 ni = 0;
  uint2 ni0 = 0, ni1 = 0;
  if (first < last) {
    if (TWELVE) ni = ix12[(size_t)first * (N / 8)];
    else { ni0 = ix16[(size_t)first * (N / 4)]; ni1 = ix16[(size_t)first * (N / 4) + 4]; }
  }
  const device uint4* wp = wl + (size_t)first * TB * 64;
  for (int g = first; g < last; g++) {
    const uint4 q0 = TWELVE ? mf_look12a(tab, ni) : mf_look(tab, ni0), q1 = TWELVE ? mf_look12b(tab, ni) : mf_look(tab, ni1);
    if (WAIT && (g - first) % MF_CH == 0) mf_wait(flags, g, min(g + MF_CH, last), epoch, sink);
    // a down chain's row sums with an atomic load: 16 rows of a group's sums share a cache line with the next group's,
    // which may not be published yet
    const float x0 = WAIT ? as_type<float>(atomic_load_explicit((device atomic_uint*)XS + g * stride + mb, memory_order_relaxed)) : XS[g * stride + mb];
    const uint4 w0 = nw0, w1 = nw1;
    for (int w = 0; w < 8; w++) {
      // 1024 + code as fp16 bits, codes (0,4) (1,5) (2,6) (3,7) of a word; subtracting 1024 is exact
      const uint word = w < 4 ? w0[w] : w1[w - 4], lo = word & 0x0F0F0F0Fu, hi = (word >> 4) & 0x0F0F0F0Fu;
      const int e = 16 * (w & 1) + 2 * (w >> 1);
      b2[e] = as_type<half2>((lo & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 1] = as_type<half2>((hi & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 8] = as_type<half2>(((lo >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 9] = as_type<half2>(((hi >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
    }
    // no branch around the prefetch: a chain's last group reloads its own codes, which nothing reads
    { const int stay = int(g + 1 >= last); wp += (1 - stay) * TB * 64; nw0 = wp[0]; nw1 = wp[32]; }
    if (TWELVE) ni = ix12[(size_t)min(g + 1, last - 1) * (N / 8)];
    else { const size_t gi = (size_t)min(g + 1, last - 1) * (N / 4); ni0 = ix16[gi]; ni1 = ix16[gi + 4]; }
    const array<int32_t, 2> xstr = {1, K};
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> x((device bfloat*)X + g * GS, dextents<int32_t, 2>(GS, M), xstr);
    op.run(x, cb, P);
    for (int j = 0; j < 4; j++) {
      const vec<bfloat, 2> v0 = as_type<vec<bfloat, 2>>(q0[j]), v1 = as_type<vec<bfloat, 2>>(q1[j]);
      const float s0 = float(v0.x), b0 = float(v0.y), s1 = float(v1.x), b1 = float(v1.y);
      C[j] = fma(s0, P[j], fma(b0, x0, C[j]));
      C[8 + j] = fma(s1, P[8 + j], fma(b1, x0, C[8 + j]));
    }
  }
}

#define MF_LIVE(i) ((((i) >> 2) & 1) == 0)
#define MF_HALF(i) (((i) & 3) | (((i) >> 3) << 2))

[[max_total_threads_per_threadgroup(128)]]
[[kernel]] void tf_mlp_flow(const device bfloat* X [[buffer(0)]], const device uint4* WG [[buffer(1)]], const device uint* SG [[buffer(2)]],
    const device int* mdims [[buffer(3)]], device bfloat* H [[buffer(4)]], const device float* XS [[buffer(5)]],
    device float* HS [[buffer(6)]], const device uint4* WD [[buffer(7)]], const device uint* SD [[buffer(8)]],
    device bfloat* Y [[buffer(9)]], device atomic_uint* sync [[buffer(10)]], constant uint& epoch [[buffer(11)]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint ticket_tg;
  // one 3 KiB block for either role (as much as each of the two launches uses): gate/up's bf16 tile words and
  // activations, or down's chain partials
  threadgroup float4 shared_tg[192];
  threadgroup bfloat* gu = (threadgroup bfloat*)shared_tg; // [4 * 8 * 32]
  threadgroup bfloat* hb = gu + 4 * 8 * 32;                // [8 * 64]
  threadgroup float* part = (threadgroup float*)shared_tg;  // [(DN_SK - 1) * 8 * 32]
  static_assert(sizeof(float4) * 192 == 2 * (4 * 8 * 32 + 8 * 64) && sizeof(float4) * 192 == 4 * (DN_SK - 1) * 8 * 32, "the roles' blocks");
  device atomic_uint* flags = sync + MF_FLAGS;
  if (tid == 0) ticket_tg = atomic_fetch_add_explicit(sync, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ticket = int(ticket_tg);
  const int lane = int(lanei), ks = int(sgi), M = min(mdims[0], 8), stride = mdims[1];
  const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  float C[16];
  if (ticket < GU_GROUPS) {
    // gate/up: a threadgroup's four tiles are 64 gate columns and their up columns, each on the serial split of 2
    const int gx = (ticket % DN_SK) * (DN_KG / DN_SK) + ticket / DN_SK;
    const int tile = 2 * gx + (ks & 1) + (ks >> 1) * (GU_N / 64);
    float C0[16];
    mf_chain<GU_N, GU_K, MF_GU12 != 0, false>(C0, X, WG, SG, XS, stride, M, tile, 0, GU_KG / 2, lane, flags, epoch, part);
    mf_chain<GU_N, GU_K, MF_GU12 != 0, false>(C, X, WG, SG, XS, stride, M, tile, GU_KG / 2, GU_KG, lane, flags, epoch, part);
    for (int i = 0; i < 16; i++) if (MF_LIVE(i)) C[i] = C0[i] + C[i];
    for (int i = 0; i < 16; i++) {
      if (!MF_LIVE(i)) continue;
      gu[(ks * 8 + mb) * 32 + nb + 16 * (i >> 3) + (i & 3)] = static_cast<bfloat>(C[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int t = ks * 32 + lane;
    // the activations two columns a store, tf_lane_reg's values, and the group's row sums, all with atomic stores (which
    // other threadgroups of this dispatch see; plain stores need not be)
    device atomic_uint* H2 = (device atomic_uint*)H;
    for (int at = 2 * t; at < 8 * 64; at += 256) {
      const int m = at / 64, c = at % 64;
      if (m >= M) continue;
      bfloat hpair[2];
      for (int d = 0; d < 2; d++) {
        const float gf = float(gu[(((c + d) >> 5) * 8 + m) * 32 + ((c + d) & 31)]);
        hpair[d] = bfloat(gf / (1.0f + metal::exp(-gf)) * float(gu[((2 + ((c + d) >> 5)) * 8 + m) * 32 + ((c + d) & 31)]));
        hb[at + d] = hpair[d];
      }
      const uint word = uint(as_type<ushort>(hpair[0])) | (uint(as_type<ushort>(hpair[1])) << 16);
      atomic_store_explicit(H2 + ((size_t)m * (GU_N / 2) + gx * 64 + c) / 2, word, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_uint* HS2 = (device atomic_uint*)HS;
    for (int m = t; m < stride; m += 128) {
      float acc = 0.0f;
      if (m < M)
        for (int i = 0; i < 64; i++) acc += float(hb[m * 64 + i]);
      atomic_store_explicit(HS2 + gx * stride + m, as_type<uint>(acc), memory_order_relaxed);
    }
    // every thread's stores before the flag: a fence each, then the threadgroup's barrier
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    for (int t2 = tid; t2 < DN_TILES; t2 += 128) atomic_store_explicit(flags + t2 * GU_GROUPS + gx, epoch, memory_order_relaxed);
  } else {
    // down: four K chains a tile, each waiting for its groups in order, reduced as tf_lane_reg's rows8 reduction
    const int tile = ticket - GU_GROUPS;
    const int first = (ks * DN_KG) / DN_SK, last = ((ks + 1) * DN_KG) / DN_SK;
    mf_chain<DN_N, DN_K, MF_DN12 != 0, true>(C, H, WD, SD, HS, stride, M, tile, first, last, lane, flags + tile * GU_GROUPS, epoch, part);
    if (ks > 0) for (int i = 0; i < 16; i++) if (MF_LIVE(i)) part[((ks - 1) * 8 + MF_HALF(i)) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) {
      for (int s2 = 1; s2 < DN_SK; s2++) for (int i = 0; i < 16; i++) if (MF_LIVE(i)) C[i] += part[((s2 - 1) * 8 + MF_HALF(i)) * 32 + lane];
      const int n0 = tile * NT;
      for (int i = 0; i < 16; i++) {
        if (!MF_LIVE(i)) continue;
        const int m = mb + 8 * ((i >> 2) & 1), n = n0 + nb + 16 * (i >> 3) + (i & 3);
        if (m < M && n < DN_N) Y[(size_t)m * DN_N + n] = static_cast<bfloat>(C[i]);
      }
    }
  }
  // the last threadgroup to finish clears the counters: every ticket is taken by then
  if (tid == 0 && atomic_fetch_add_explicit(sync + 1, 1u, memory_order_relaxed) == uint(MF_GROUPS - 1)) {
    atomic_store_explicit(sync, 0u, memory_order_relaxed);
    atomic_store_explicit(sync + 1, 0u, memory_order_relaxed);
  }
}
