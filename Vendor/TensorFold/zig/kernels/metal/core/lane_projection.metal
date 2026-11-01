#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// 1-16 bf16 rows share each group's weight read; every row's sums and split-K order hold at every width.

#ifndef TF_SIMD_FRAGS
#define TF_SIMD_FRAGS 0
#endif

#ifndef TF_PRECOMPUTE_SUMS
#define TF_PRECOMPUTE_SUMS 0
#endif
#ifndef TF_YS
#define TF_YS TF_N
#endif
// Output rows lie TF_YS elements apart, so a member of a fused stack writes its columns of the fused rows.
#define TF_ODD_WORDS (TF_GROUP * TF_BITS / 32 % 2)

constant constexpr int NT = 32, GS = TF_GROUP, KG = TF_K / GS, NF = NT / 16;
constant constexpr int WP = GS * TF_BITS / 32;

#if TF_ODD_WORDS
struct tf_raw { uint w[WP]; uint4 s[NF]; }; // 3- and 5-bit groups of 32 are an odd word count, loaded a word at a time
#else
struct tf_raw { uint2 w[WP / 2]; uint4 s[NF]; }; // a lane's packed code words and its scale/bias quads for one group
#endif

inline tf_raw tf_lane_fetch(const device uint* W, const device uint4* SB, int tile, int g, int g_end, int n0, int lane,
                            short fn) {
  tf_raw raw;
  const bool ok = g < g_end;
  const device uint2* src = (const device uint2*)(W + (((size_t)tile * KG + (ok ? g : 0)) * NT + lane) * WP);
#if TF_ODD_WORDS
  for (int i = 0; i < WP; i++) raw.w[i] = ok && n0 + lane < TF_N ? ((const device uint*)src)[i] : 0u;
#elif TF_BITS != 4
  for (int i = 0; i < WP / 2; i++) raw.w[i] = ok && n0 + lane < TF_N ? src[i] : uint2(0);
#else
  for (int i = 0; i < WP / 2; i++) raw.w[i] = uint2(0);
#endif
#if TF_SIMD_FRAGS
  const device uint2* sb2 = (const device uint2*)SB;
  for (int f = 0; f < NF; f++) {
    const int col = n0 + f * 16 + fn;
    raw.s[f] = uint4(ok && col < TF_N ? sb2[((size_t)g * TF_N + col) / 2] : uint2(0),
                     ok && col + 8 < TF_N ? sb2[((size_t)g * TF_N + col + 8) / 2] : uint2(0));
  }
#else
  // Four lanes load the metadata quads and broadcast them; all 32 lanes fetch, even for inactive rows.
  const ushort leader = ushort((fn & 8) | ((fn >> 2) & 1));
  for (int f = 0; f < NF; f++) {
    const uint4 sb = lane == int(leader) && ok && n0 + f * 16 + fn < TF_N
      ? SB[((size_t)(ok ? g : 0) * TF_N + n0 + f * 16 + fn) / 4] : uint4(0);
    raw.s[f] = simd_shuffle(sb, leader);
  }
#endif
  return raw;
}

inline float tf_lane_xsum(const device bfloat* X, int m, int g, int M) {
  float acc = 0.0f;
  if (m < M) for (int i = 0; i < GS; i++) acc += float(X[(size_t)m * TF_K + g * GS + i]);
  return acc;
}

// Codes: 4-bit straight to the uint4 op; 2/3-bit widened to nibbles for it, 5/6/8-bit to bytes for the uint8 op.
#if TF_BITS == 4
#define TF_STAGE_CODES(raw)
#define TF_GROUP_PRODUCT(g) \
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b( \
      (device uchar*)W + ((size_t)tile * KG + (g)) * (NT * GS / 2), dextents<int32_t, 2>(GS, NT)); \
    auto a = tA.slice((g) * GS, 0); \
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>(); \
    op.run(a, b, P);
#else
#if TF_BITS < 4
inline void tf_stage_codes(threadgroup uint* stage, thread const tf_raw& raw, int lane) {
    uint words[WP + 1];
#if TF_ODD_WORDS
    for (int i = 0; i < WP; i++) words[i] = raw.w[i];
#else
    for (int i = 0; i < WP / 2; i++) { words[2*i] = raw.w[i].x; words[2*i+1] = raw.w[i].y; }
#endif
    words[WP] = 0u;
#if TF_BITS == 3
    for (int c = 0; c < GS / 8; c++) {
      const int bit = 24 * c, i = bit >> 5, sh = bit & 31;
      uint pack = words[i] >> sh;
      if (sh > 8) pack |= words[i + 1] << (32 - sh);
      uint nib = 0u;
      for (int j = 0; j < 8; j++) nib |= ((pack >> (3 * j)) & 7u) << (4 * j);
      stage[lane * (GS / 8) + c] = nib;
    }
#else
    for (int c = 0; c < GS / 8; c++) { // word c/2's half c%2: 8 codes of 2 bits to 8 nibbles
      uint v = (words[c >> 1] >> (16 * (c & 1))) & 0xFFFFu;
      v = (v | (v << 8)) & 0x00FF00FFu;
      v = (v | (v << 4)) & 0x0F0F0F0Fu;
      v = (v | (v << 2)) & 0x33333333u;
      stage[lane * (GS / 8) + c] = v;
    }
#endif
}
#else
inline void tf_stage_codes(threadgroup uint* stage, thread const tf_raw& raw, int lane) {
    uint words[WP + 1];
#if TF_ODD_WORDS
    for (int i = 0; i < WP; i++) words[i] = raw.w[i];
#else
    for (int i = 0; i < WP / 2; i++) { words[2*i] = raw.w[i].x; words[2*i+1] = raw.w[i].y; }
#endif
    words[WP] = 0u;
    for (int c = 0; c < GS / 4; c++) {
      uint word = 0u;
      for (int j = 0; j < 4; j++) {
        const int bit = (4*c + j) * TF_BITS, i = bit >> 5, sh = bit & 31;
        uint code = words[i] >> sh;
        if (sh + TF_BITS > 32) code |= words[i+1] << (32-sh);
        word |= (code & ((1u << TF_BITS)-1u)) << (8*j);
      }
      stage[lane * (GS / 4) + c] = word;
    }
}
#endif
#define TF_STAGE_CODES(raw) tf_stage_codes(stage, raw, lane);
#define TF_GROUP_PRODUCT(g) \
    simdgroup_barrier(mem_flags::mem_threadgroup); \
    auto a = tA.slice((g) * GS, 0); \
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>(); \
    op.run(a, b, P); \
    simdgroup_barrier(mem_flags::mem_threadgroup);
#endif

#if TF_PRECOMPUTE_SUMS
#define TF_SUM_VALUE(row, g) XS[(g) * 16 + (row)]
#else
#define TF_SUM_VALUE(row, g) tf_lane_xsum(X, row, g, M)
#endif

// Each group loads packed codes or stages widened codes, performs the tensor product, then applies the scale/bias FMAs.
#define TF_LANE_GROUP(g, raw)                                                                                  \
  {                                                                                                            \
    TF_STAGE_CODES(raw) \
    float s[NF][4], bb[NF][4];                                                                                 \
    for (int f = 0; f < NF; f++) {                                                                             \
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(raw.s[f]);                                              \
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }               \
    }                                                                                                          \
    raw = tf_lane_fetch(W, sbv, tile, (g) + TF_PF, g_end, n0, lane, fn);                                       \
    TF_GROUP_PRODUCT(g) \
    const float xs0 = TF_SUM_VALUE(fm, g); \
    const float xs1 = TF_SUM_VALUE(fm + 8, g); \
    for (int f = 0; f < NF; f++)                                                                               \
      for (int r = 0; r < 2; r++)                                                                              \
        for (int j = 0; j < 4; j++) {                                                                          \
          const int i = f * 8 + r * 4 + j;                                                                     \
          C[i] = fma(s[f][j], P[TF_SIMD_FRAGS ? r * 8 + f * 4 + j : i],                                         \
                     fma(bb[f][j], r ? xs1 : xs0, C[i]));                                       \
        }                                                                                                      \
  }

[[kernel]] void tf_lane(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* SBt [[buffer(2)]], const device int* mdims [[buffer(3)]], device TF_OUT* Y [[buffer(4)]],
#if TF_PRECOMPUTE_SUMS
    const device float* XS [[buffer(5)]],
#endif
    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]],
    uint tgx [[threadgroup_position_in_grid]]) {
  const int lane = int(lanei), sg = int(sgi);
  const short qid = short(lane) >> 2;
  const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
#if TF_SIMD_FRAGS
  const short fn = ((short(lane) & 8) >> 1) | ((short(lane) & 1) << 1);
#else
  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
#endif
  const int M = mdims[0];
  const int tile = tf_tile(int(tgx));
  const int n0 = tile * NT;
  const int g_begin = TF_G0 + (sg * TF_GN) / TF_SK, g_end = TF_G0 + ((sg + 1) * TF_GN) / TF_SK;
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(TF_K, M));
#if TF_BITS < 4
  threadgroup uint stage_all[TF_SK * NT * (GS / 8)];
  threadgroup uint* stage = stage_all + sg * NT * (GS / 8);
  tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> b((threadgroup uchar*)stage, dextents<int32_t, 2>(GS, NT));
#elif TF_BITS != 4
  threadgroup uint stage_all[TF_SK * NT * (GS / 4)];
  threadgroup uint* stage = stage_all + sg * NT * (GS / 4);
  tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage, dextents<int32_t, 2>(GS, NT));
#endif
  const device uint4* sbv = (const device uint4*)SBt;
  float C[NF * 8];
  for (int i = 0; i < NF * 8; i++) C[i] = 0.0f;
  tf_raw r0 = tf_lane_fetch(W, sbv, tile, g_begin, g_end, n0, lane, fn);
#if TF_PF == 2
  tf_raw r1 = tf_lane_fetch(W, sbv, tile, g_begin + 1, g_end, n0, lane, fn);
  for (int g = g_begin; g < g_end; g += 2) {
    TF_LANE_GROUP(g, r0)
    if (g + 1 < g_end) TF_LANE_GROUP(g + 1, r1)
  }
#else
  for (int g = g_begin; g < g_end; g++) TF_LANE_GROUP(g, r0)
#endif
  threadgroup float part[(TF_SK > 1 ? TF_SK - 1 : 1) * NF * 8 * 32];
  if (TF_SK > 1) {
    if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0)
      for (int s2 = 1; s2 < TF_SK; s2++) for (int i = 0; i < NF * 8; i++) C[i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
  }
  if (sg == 0)
    for (int f = 0; f < NF; f++)
      for (int r = 0; r < 2; r++) {
        const int m = fm + 8 * r;
        const int nn = n0 + f * 16 + fn;
        if (m < M && nn < TF_N)
          for (int j = 0; j < 4; j++) {
            const int col = nn + (TF_SIMD_FRAGS ? (j & 1) + 8 * (j >> 1) : j);
            if (col < TF_N) Y[(size_t)m * TF_YS + col] = static_cast<TF_OUT>(C[f * 8 + r * 4 + j]);
          }
      }
}

#if TF_PRECOMPUTE_SUMS
kernel void tf_lane_sums(const device bfloat* X [[buffer(0)]],constant int& rows [[buffer(1)]],device float* XS [[buffer(2)]],constant int& stride [[buffer(3)]],uint index [[thread_position_in_grid]]) {
  const int row=int(index)%stride,group=int(index)/stride;
  if(group>=KG)return;
  XS[index]=tf_lane_xsum(X,row,group,rows);
}
#endif

#if TF_BITS == 4
kernel void tf_lane_pair(const device bfloat* X [[buffer(0)]],const device uint* W [[buffer(1)]],const device bfloat* SB [[buffer(2)]],const device int* mdims [[buffer(3)]],device TF_OUT* Y [[buffer(4)]],
#if TF_PRECOMPUTE_SUMS
const device float* XS [[buffer(5)]],
#endif
uint group_id [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint thread_id [[thread_position_in_threadgroup]],uint tile_id [[threadgroup_position_in_grid]]) {
  const uint sg=group_id/2,tip=thread_id-sg*64;
  const int M=mdims[0],tile=tf_tile(int(tile_id)),n0=tile*NT;
  const int first=TF_G0+(int(sg)*TF_GN)/TF_SK,last=TF_G0+((int(sg)+1)*TF_GN)/TF_SK;
  constexpr auto descriptor=matmul2d_descriptor(16,NT,GS,false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<2>> op;
  tensor<device bfloat,dextents<int32_t,2>,tensor_inline> a((device bfloat*)X,dextents<int32_t,2>(TF_K,M));
  tensor<device uint4b_format,dextents<int32_t,2>,tensor_inline> b0((device uchar*)W,dextents<int32_t,2>(GS,NT));
  auto a0=a.slice(0,0);
  auto p=op.template get_destination_cooperative_tensor<decltype(a0),decltype(b0),float>();
  short column[8],row[8];float value[8];
  for(int i=0;i<8;i++){auto index=p.get_multidimensional_index(i);column[i]=index[0];row[i]=index[1];value[i]=0.0f;}
  for(int group=first;group<last;group++) {
    tensor<device uint4b_format,dextents<int32_t,2>,tensor_inline> b((device uchar*)W+((size_t)tile*KG+group)*(NT*GS/2),dextents<int32_t,2>(GS,NT));
    auto x=a.slice(group*GS,0);op.run(x,b,p);
    for(int i=0;i<8;i++) {
      const int n=n0+column[i];
      const float scale=float(SB[(size_t(group)*TF_N+n)*2]),bias=float(SB[(size_t(group)*TF_N+n)*2+1]);
#if TF_PRECOMPUTE_SUMS
      const float sum=XS[group*16+row[i]];
#else
      const float sum=tf_lane_xsum(X,row[i],group,M);
#endif
      value[i]=fma(scale,p[i],fma(bias,sum,value[i]));
    }
  }
  threadgroup float partial[(TF_SK>1?TF_SK-1:1)*8*64];
  if(TF_SK>1) {
    if(sg>0) for(int i=0;i<8;i++) partial[((sg-1)*8+i)*64+tip]=value[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) for(int part=1;part<TF_SK;part++)for(int i=0;i<8;i++)value[i]+=partial[((part-1)*8+i)*64+tip];
  }
  if(sg==0)for(int i=0;i<8;i++)if(row[i]<M)Y[size_t(row[i])*TF_YS+n0+column[i]]=TF_OUT(value[i]);
}
#endif

#if TF_BITS == 4 && defined(TF_REG) && !defined(TF_PROMPT_TN)
#ifndef TF_TB
#define TF_TB 1
#endif
struct tf_reg_load { uint4 w0, w1, q0, q1; float x0, x1; }; // a lane's codes, its two metadata quads and its two row sums for one group

// TF_SBI (decoded rows' lean loop only): metadata indexed as lane_projection.indexed_layout, TF_SBI distinct
// (scale | bias << 16) words, then a u16 index per group and column. A lane's quad of four columns is one uint2 of indices
// and four table words, the words the unindexed quad holds, so every product and FMA is unchanged. A group's indices load
// one group ahead with its codes; its table words at the top of its own iteration.
#if defined(TF_SBI)
inline uint4 tf_look(const device uint* tab, uint2 ix) {
  return uint4(tab[ix.x & 0xFFFFu], tab[ix.x >> 16], tab[ix.y & 0xFFFFu], tab[ix.y >> 16]);
}
#endif
// TF_SBI12: the same, with 12-bit indices: a lane's two quads are one packed_uint3 of eight indices, low bits first, as
// lane_projection.indexed12_layout. The table words read are those of the 16-bit indices.
#if defined(TF_SBI12)
inline uint4 tf_look12a(const device uint* tab, uint3 ix) {
  return uint4(tab[ix.x & 0xFFFu], tab[(ix.x >> 12) & 0xFFFu], tab[(ix.x >> 24) | ((ix.y & 0xFu) << 8)], tab[(ix.y >> 4) & 0xFFFu]);
}
inline uint4 tf_look12b(const device uint* tab, uint3 ix) {
  return uint4(tab[(ix.y >> 16) & 0xFFFu], tab[(ix.y >> 28) | ((ix.z & 0xFFu) << 4)], tab[(ix.z >> 8) & 0xFFFu], tab[ix.z >> 20]);
}
#endif

// TF_ROWS8: launches of at most 8 rows (decoded rows). A lane holds rows mb and mb + 8; the second is then always padding,
// so its accumulators, row sums, hand-over words and stores are left out. Every live row's words are unchanged.
#if defined(TF_ROWS8)
#define TF_LIVE(i) ((((i) >> 2) & 1) == 0)
#define TF_HALF(i) (((i) & 3) | (((i) >> 3) << 2))
#define TF_PART 8
#else
#define TF_LIVE(i) true
#define TF_HALF(i) (i)
#define TF_PART 16
#endif

inline tf_reg_load tf_reg_fetch(const device uint4* wl, const device uint4* sbv, const device float* XS, int g, int n, int m, int stride) {
#if defined(TF_ROWS8)
  return {wl[(size_t)g * TF_TB * 64], wl[(size_t)g * TF_TB * 64 + 32], sbv[((size_t)g * TF_N + n) / 4], sbv[((size_t)g * TF_N + n + 16) / 4], XS[g * stride + m], 0.0f};
#else
  return {wl[(size_t)g * TF_TB * 64], wl[(size_t)g * TF_TB * 64 + 32], sbv[((size_t)g * TF_N + n) / 4], sbv[((size_t)g * TF_N + n + 16) / 4], XS[g * stride + m], XS[g * stride + m + 8]};
#endif
}

// tf_lane's arithmetic with codes unpacked straight into tensor-op registers: every word equals tf_lane_pair's.
#if defined(TF_SWIGLU) && defined(TF_ROWS8)
#define TF_GU_TURNS 1
#endif
// TF_CHAIN_WALK (decoded rows on one SIMD group): a tile's K chains run in turn as one stream of groups, the code and index
// prefetches running over each cut, and each chain's words reduce in registers in the K split's order. TF_SK is 2 or 4.
#if defined(TF_SERIAL_SK2) && defined(TF_ROWS8) && defined(TF_REG_LEAN) && !defined(TF_PROMPT_SR)
#define TF_CHAIN_WALK 1
#endif
#if defined(TF_GU_TURNS)
[[max_total_threads_per_threadgroup(64)]]
#elif defined(TF_SWIGLU)
[[max_total_threads_per_threadgroup(128)]]
#elif defined(TF_SERIAL_SK2)
[[max_total_threads_per_threadgroup(32)]]
#else
[[max_total_threads_per_threadgroup(32 * TF_SK)]]
#endif
[[kernel]] void tf_lane_reg(const device bfloat* X [[buffer(0)]], const device uint4* W [[buffer(1)]], const device bfloat* SB [[buffer(2)]],
    const device int* mdims [[buffer(3)]], device TF_OUT* Y [[buffer(4)]], const device float* XS [[buffer(5)]],
#if defined(TF_SWIGLU)
    device float* HS [[buffer(6)]],
#elif defined(TF_COUNT)
    device atomic_uint* TC [[buffer(6)]],
#endif
    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int r0 = int(tg.y) * 16, rows = mdims[0] - r0;
  if (rows <= 0) return;
  X += (size_t)r0 * TF_K;
#if defined(TF_SWIGLU)
  Y += (size_t)r0 * (TF_N / 2); // SwiGLU rows are half the fused width
  HS += r0;
#else
  Y += (size_t)r0 * TF_YS;
#endif
  XS += r0;
#if defined(TF_GU_TURNS)
  const int lane = int(lanei), ks = int(sgi), M = min(rows, 16), stride = mdims[1];
#elif defined(TF_SWIGLU)
  const int lane = int(lanei), ks = int(sgi), M = min(rows, 16), stride = mdims[1], tile = 2 * int(tg.x) + (ks & 1) + (ks >> 1) * (TF_N / 64), n0 = tile * NT;
#else
  const int lane = int(lanei), ks = int(sgi), M = min(rows, 16), stride = mdims[1], tile = tf_tile(int(tg.x)), n0 = tile * NT;
#endif
  const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)X, dextents<int32_t, 2>(TF_K, M));
  auto a0 = a.slice(0, 0);
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto P = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a0)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  thread half2* b2 = (thread half2*)&cb[0];
  const device uint4* sbv = (const device uint4*)SB;
  const half2 bias = half2(1024.0h);
  float C[16];
#if defined(TF_GU_TURNS)
  float G[16];
  for (int turn = 0; turn < 2; turn++) {
  const int tile = 2 * int(tg.x) + ks + turn * (TF_N / 64), n0 = tile * NT;
#endif
  const device uint4* wl = W + ((size_t)(tile / TF_TB) * KG * TF_TB + tile % TF_TB) * 64 + lane;
#if defined(TF_CHAIN_WALK)
  // one stream over all K: a chain's first group is already prefetched by the chain before it
  const int end = TF_G0 + TF_GN;
  uint4 nw0 = wl[(size_t)TF_G0 * TF_TB * 64], nw1 = wl[(size_t)TF_G0 * TF_TB * 64 + 32];
#if defined(TF_SBI12)
  const device uint* tab = (const device uint*)SB;
  const device packed_uint3* ip = (const device packed_uint3*)(tab + TF_SBI) + n0 / 8 + nb / 4 + (size_t)TF_G0 * (TF_N / 8);
  uint3 ni = ip[0];
#elif defined(TF_SBI)
  const device uint* tab = (const device uint*)SB;
  const device uint2* ip = (const device uint2*)(tab + TF_SBI) + (n0 + nb) / 4 + (size_t)TF_G0 * (TF_N / 4);
  uint2 ni0 = ip[0], ni1 = ip[4];
#endif
  const device uint4* wp = wl + (size_t)TF_G0 * TF_TB * 64;
#endif
#if defined(TF_SERIAL_SK2)
  float C0[16];
  for (int split = 0; split < TF_SK; split++) {
  const int first = TF_G0 + (split * TF_GN) / TF_SK, last = TF_G0 + ((split + 1) * TF_GN) / TF_SK;
#else
#if defined(TF_SWIGLU)
  const int first = TF_G0, last = TF_G0 + TF_GN; // one K split: a threadgroup's SIMD groups are its chunk's four tiles
#else
  const int first = TF_G0 + (ks * TF_GN) / TF_SK, last = TF_G0 + ((ks + 1) * TF_GN) / TF_SK;
#endif
#endif
  for (int i = 0; i < 16; i++) C[i] = 0.0f;
#if defined(TF_REG_LEAN)
  // decoded rows prefetch only the next group's codes; a group's metadata quads and row sums load at the top of its own
  // iteration, under the unpack and the tensor op, so ten fewer registers stay live across the loop
#if defined(TF_PROMPT_SR)
  // prompt rows address the group's codes, scale/bias quads and row sums by pointers that advance a fixed step a group:
  // the same words as wl[g * TF_TB * 64], sbv[(g * TF_N + n0 + nb) / 4] and XS[g * stride + mb] (TF_N, n0 + nb are
  // multiples of 4), without the per-group 64-bit index arithmetic
  const device uint4* cp = wl + (size_t)first * TF_TB * 64;
  const device uint4* sp = sbv + ((size_t)first * TF_N + n0 + nb) / 4;
  const device float* xp = XS + first * stride + mb;
  const device bfloat* xv = X + (size_t)first * GS;
  uint4 nw0 = 0, nw1 = 0;
  if (first < last) { nw0 = cp[0]; nw1 = cp[32]; }
  for (int g = first; g < last; g++) {
    const tf_reg_load cur = {nw0, nw1, sp[0], sp[4], xp[0], xp[8]};
    sp += TF_N / 4;
    xp += stride;
#else
#if !defined(TF_CHAIN_WALK)
  const int end = last;
  uint4 nw0 = 0, nw1 = 0;
  if (first < last) { nw0 = wl[(size_t)first * TF_TB * 64]; nw1 = wl[(size_t)first * TF_TB * 64 + 32]; }
#if defined(TF_SBI12)
  const device uint* tab = (const device uint*)SB;
  const device packed_uint3* ip = (const device packed_uint3*)(tab + TF_SBI) + n0 / 8 + nb / 4;
  uint3 ni = 0;
  if (first < last) { ip += (size_t)first * (TF_N / 8); ni = ip[0]; }
#elif defined(TF_SBI)
  const device uint* tab = (const device uint*)SB;
  const device uint2* ip = (const device uint2*)(tab + TF_SBI) + (n0 + nb) / 4;
  uint2 ni0 = 0, ni1 = 0;
  if (first < last) { ip += (size_t)first * (TF_N / 4); ni0 = ip[0]; ni1 = ip[4]; }
#endif
  const device uint4* wp = wl + (size_t)first * TF_TB * 64;
#endif
  for (int g = first; g < last; g++) {
#if defined(TF_SBI12)
    const uint4 q0 = tf_look12a(tab, ni), q1 = tf_look12b(tab, ni);
#elif defined(TF_SBI)
    const uint4 q0 = tf_look(tab, ni0), q1 = tf_look(tab, ni1);
#else
    const uint4 q0 = sbv[((size_t)g * TF_N + n0 + nb) / 4], q1 = sbv[((size_t)g * TF_N + n0 + nb + 16) / 4];
#endif
#if defined(TF_ROWS8)
    const tf_reg_load cur = {nw0, nw1, q0, q1, XS[g * stride + mb], 0.0f};
#else
    const tf_reg_load cur = {nw0, nw1, q0, q1, XS[g * stride + mb], XS[g * stride + mb + 8]};
#endif
#endif
#else
  tf_reg_load next = {};
  if (first < last) next = tf_reg_fetch(wl, sbv, XS, first, n0 + nb, mb, stride);
  for (int g = first; g < last; g++) {
    const tf_reg_load cur = next;
#endif
    for (int w = 0; w < 8; w++) {
      // codes as fp16 subnormals (bits = code, value code * 2^-24): products are the usual ones scaled by 2^-24 exactly, and
      // the scales below carry the 2^24 back
      const uint word = w < 4 ? cur.w0[w] : cur.w1[w - 4];
      const int e = 16 * (w & 1) + 2 * (w >> 1);
      b2[e] = as_type<half2>(word & 0x000F000Fu);
      b2[e + 1] = as_type<half2>((word >> 4) & 0x000F000Fu);
      b2[e + 8] = as_type<half2>((word >> 8) & 0x000F000Fu);
      b2[e + 9] = as_type<half2>((word >> 12) & 0x000F000Fu);
    }
#if defined(TF_REG_LEAN)
    // no branch around the prefetch: a chain's last group reloads its own codes, which nothing reads
#if defined(TF_PROMPT_SR)
    cp += g + 1 < last ? TF_TB * 64 : 0;
    nw0 = cp[0];
    nw1 = cp[32];
#else
    { const int stay = int(g + 1 >= end); wp += (1 - stay) * TF_TB * 64; nw0 = wp[0]; nw1 = wp[32]; }
#if defined(TF_SBI12)
    // `end` spans all splits for chain-walk, but only this split otherwise.
    ip += size_t(g + 1 < end) * (TF_N / 8);
    ni = ip[0];
#elif defined(TF_SBI)
    ip += size_t(g + 1 < end) * (TF_N / 4);
    ni0 = ip[0]; ni1 = ip[4];
#endif
#endif
#else
    if (g + 1 < last) next = tf_reg_fetch(wl, sbv, XS, g + 1, n0 + nb, mb, stride);
#endif
#if defined(TF_REG_LEAN)
    const array<int32_t, 2> xstr = {1, TF_K};
#if defined(TF_PROMPT_SR)
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> x((device bfloat*)xv, dextents<int32_t, 2>(GS, M), xstr);
    xv += GS;
#else
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> x((device bfloat*)X + g * GS, dextents<int32_t, 2>(GS, M), xstr);
#endif
#else
    auto x = a.slice(g * GS, 0);
#endif
    op.run(x, cb, P);
    for (int j = 0; j < 4; j++) {
      const vec<bfloat, 2> v0 = as_type<vec<bfloat, 2>>(cur.q0[j]), v1 = as_type<vec<bfloat, 2>>(cur.q1[j]);
      const float s0 = float(v0.x), b0 = float(v0.y), s1 = float(v1.x), b1 = float(v1.y); // scales stored times 2^24
      C[j] = fma(s0, P[j], fma(b0, cur.x0, C[j]));
#if !defined(TF_ROWS8)
      C[4 + j] = fma(s0, P[4 + j], fma(b0, cur.x1, C[4 + j]));
#endif
      C[8 + j] = fma(s1, P[8 + j], fma(b1, cur.x0, C[8 + j]));
#if !defined(TF_ROWS8)
      C[12 + j] = fma(s1, P[12 + j], fma(b1, cur.x1, C[12 + j]));
#endif
    }
  }
#if defined(TF_SERIAL_SK2) && defined(TF_ROWS8)
    // the K splits' reduction order, in registers: ((c0 + c1) + c2) + c3
    for (int i = 0; i < 16; i++) if (TF_LIVE(i)) C0[i] = split == 0 ? C[i] : C0[i] + C[i];
  }
  for (int i = 0; i < 16; i++) if (TF_LIVE(i)) C[i] = C0[i];
#if defined(TF_GU_TURNS)
    if (turn == 0) for (int i = 0; i < 16; i++) if (TF_LIVE(i)) G[i] = C[i];
  }
#endif
#elif defined(TF_SERIAL_SK2)
    if (split == 0) for (int i = 0; i < 16; i++) C0[i] = C[i];
  }
  for (int i = 0; i < 16; i++) C[i] = C0[i] + C[i];
#elif defined(TF_ROWS8)
  threadgroup float part[(TF_SK > 1 ? TF_SK - 1 : 1) * TF_PART * 32];
  if (TF_SK > 1) {
    if (ks > 0) for (int i = 0; i < 16; i++) if (TF_LIVE(i)) part[((ks - 1) * TF_PART + TF_HALF(i)) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) for (int s2 = 1; s2 < TF_SK; s2++) for (int i = 0; i < 16; i++) if (TF_LIVE(i)) C[i] += part[((s2 - 1) * TF_PART + TF_HALF(i)) * 32 + lane];
  }
#else
  threadgroup float part[(TF_SK > 1 ? TF_SK - 1 : 1) * 16 * 32];
  if (TF_SK > 1) {
    if (ks > 0) for (int i = 0; i < 16; i++) part[((ks - 1) * 16 + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) for (int s2 = 1; s2 < TF_SK; s2++) for (int i = 0; i < 16; i++) C[i] += part[((s2 - 1) * 16 + i) * 32 + lane];
  }
#endif
#if defined(TF_SWIGLU) && !defined(TF_ROWS8)
  // prompt rows: 16 a block. Each tile's words round to bf16 as the unfused store does; SwiGLU reads them as the
  // activation kernel reads the gate/up rows, and each 64-column group's sum adds its values in order from zero.
  threadgroup bfloat gu16[4 * 16 * 32];
  threadgroup float hb16[16 * 64];
  for (int i = 0; i < 16; i++) {
    const int m = mb + 8 * ((i >> 2) & 1);
    gu16[(ks * 16 + m) * 32 + nb + 16 * (i >> 3) + (i & 3)] = static_cast<bfloat>(C[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t16 = ks * 32 + lane;
  for (int at = t16; at < 16 * 64; at += 128) {
    const int m = at / 64, c = at % 64;
    if (m >= M) continue;
    const float gf = float(gu16[((c >> 5) * 16 + m) * 32 + (c & 31)]);
    const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * float(gu16[((2 + (c >> 5)) * 16 + m) * 32 + (c & 31)]));
    Y[(size_t)m * (TF_N / 2) + int(tg.x) * 64 + c] = h;
    hb16[at] = float(h);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t16 < 16) {
    float acc = 0.0f;
    if (t16 < M)
      for (int i = 0; i < 64; i++) acc += hb16[t16 * 64 + i];
    HS[int(tg.x) * stride + t16] = acc;
  }
#elif defined(TF_SWIGLU)
  threadgroup bfloat hb[8 * 64];
  if (mb < M) {
    const int c0 = ks * 32 + nb;
    device TF_OUT* yr = Y + (size_t)mb * (TF_N / 2) + int(tg.x) * 64 + c0;
    threadgroup bfloat* hr = hb + mb * 64 + c0;
    for (int i = 0; i < 16; i++) {
      if (!TF_LIVE(i)) continue;
      const int c = 16 * (i >> 3) + (i & 3);
      const float gf = float(static_cast<bfloat>(G[i]));
      const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * float(static_cast<bfloat>(C[i])));
      yr[c] = h;
      hr[c] = h;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t = ks * 32 + lane;
  for (int m = t; m < stride; m += 64) {
    float acc = 0.0f;
    if (m < M)
      for (int i = 0; i < 64; i++) acc += float(hb[m * 64 + i]);
    HS[int(tg.x) * stride + m] = acc;
  }
#elif defined(TF_ROWS8)
  // a live word's row is mb, and N is whole tiles, so one row test and one pointer cover the lane's eight stores
  if (ks == 0 && mb < M) {
    device TF_OUT* yr = Y + (size_t)mb * TF_YS + n0 + nb;
    for (int i = 0; i < 16; i++) if (TF_LIVE(i)) yr[16 * (i >> 3) + (i & 3)] = static_cast<TF_OUT>(C[i]);
  }
#else
  if (ks == 0)
    for (int i = 0; i < 16; i++) {
      const int m = mb + 8 * ((i >> 2) & 1), n = n0 + nb + 16 * (i >> 3) + (i & 3);
      if (m < M && n < TF_N) Y[(size_t)m * TF_YS + n] = static_cast<TF_OUT>(C[i]);
    }
#endif
#if defined(TF_COUNT)
  // the norm that reads Y shares this launch's barrier interval: each threadgroup counts itself on TC[0] once its
  // stores are past a device-scope fence (q27_norm_xs' wait_for)
  threadgroup_barrier(mem_flags::mem_device);
  if (ks == 0 && lane == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_fetch_add_explicit(TC, 1u, memory_order_relaxed);
  }
#endif
}
#endif

#if TF_BITS == 4 && defined(TF_REG) && defined(TF_PROMPT_TN)
#ifndef TF_TB
#define TF_TB 1
#endif
// Prompt rows, TF_PROMPT_TN (2 or 4) adjacent 32-column tiles a SIMD group: a group's 16x64 activation slice loads once into
// cooperative registers and feeds both tiles' products, instead of each tile's tensor op reading it from device memory.
// Each tile unpacks its own codes into the same register words as tf_lane_reg, and applies the same scale/bias FMA chain,
// K split reduction and store as tf_lane_reg's prompt rows. Both operands in registers need K = 32 tensor ops, so each
// 64-deep group runs as a K = 32 multiply then a K = 32 multiply-accumulate into the same destination: a K = 64
// operand's registers are its two K = 32 halves' registers in order (left: elements 0-15, 16-31; right: 0-31, 32-63),
// and the pair's fp32 sums equal the single K = 64 product's bit for bit (checked on 2M random outputs on an M5 Pro).
#ifndef TF_UNROLL
#define TF_UNROLL _Pragma("clang loop unroll(full)")
#endif
// One tile of a group: unpack the codes held in nw0, nw1 (1024 + code as fp16 bits, codes (0,4) (1,5) (2,6) (3,7) of a word;
// subtracting 1024 is exact), load the next tile's (the next group's first tile after the last; a chain's last tile reloads
// its own, which nothing reads), run, then FMAs. A SIMD group's tiles lie in one code batch, so one code and one metadata
// pointer step through them and only one tile's codes stay live across a tensor op, whatever TN is.
#define TF_TN_TILE(t) { \
      const uint4 q0 = sp[0], q1 = sp[4]; \
      sp += (t) + 1 < TN ? NT / 4 : (TF_N - (TN - 1) * NT) / 4; \
      thread half2* b2l = (thread half2*)&cbl[0]; \
      thread half2* b2h = (thread half2*)&cbh[0]; \
      TF_UNROLL \
      for (int w = 0; w < 8; w++) { \
        const uint word = w < 4 ? nw0[w] : nw1[w - 4]; \
        thread half2* b2 = (w & 1) ? b2h : b2l; \
        const int e = 2 * (w >> 1); \
        b2[e] = as_type<half2>(word & 0x000F000Fu); \
        b2[e + 1] = as_type<half2>((word >> 4) & 0x000F000Fu); \
        b2[e + 8] = as_type<half2>((word >> 8) & 0x000F000Fu); \
        b2[e + 9] = as_type<half2>((word >> 12) & 0x000F000Fu); \
      } \
      cp += (t) + 1 < TN ? 64 : (g + 1 < last ? (TF_TB - TN + 1) * 64 : 0); \
      nw0 = cp[0]; \
      nw1 = cp[32]; \
      opm.run(xl0, cbl, P); \
      opa.run(xl1, cbh, P); \
      TF_UNROLL \
      for (int j = 0; j < 4; j++) { \
        const vec<bfloat, 2> v0 = as_type<vec<bfloat, 2>>(q0[j]), v1 = as_type<vec<bfloat, 2>>(q1[j]); \
        const float s0 = float(v0.x), b0 = float(v0.y), s1 = float(v1.x), b1 = float(v1.y); \
        C[t][j] = fma(s0, P[j], fma(b0, x0, C[t][j])); \
        C[t][4 + j] = fma(s0, P[4 + j], fma(b0, x1, C[t][4 + j])); \
        C[t][8 + j] = fma(s1, P[8 + j], fma(b1, x0, C[t][8 + j])); \
        C[t][12 + j] = fma(s1, P[12 + j], fma(b1, x1, C[t][12 + j])); \
      } \
}
// TF_RESID (prompt rows of the 5120-wide output projections): instead of the output rows, the kernel stores the residual
// stream's next rows, H + the output, as the norm after it would have formed them.
// TF_SWIGLU (prompt rows of the fused gate/up, TN 2, one K split): SIMD group 0 computes gate tiles 2x and 2x + 1, SIMD
// group 1 their up tiles TF_N / 64 tiles on, and the threadgroup applies tf_lane_reg's fused prompt epilogue to them.
#if defined(TF_SWIGLU)
#if TF_PROMPT_TN != 2 || TF_SK != 1
#error "the two-tile SwiGLU launch takes two tiles and one K split"
#endif
[[max_total_threads_per_threadgroup(64)]]
#else
[[max_total_threads_per_threadgroup(32 * TF_SK)]]
#endif
[[kernel]] void tf_lane_reg(const device bfloat* X [[buffer(0)]], const device uint4* W [[buffer(1)]], const device bfloat* SB [[buffer(2)]],
    const device int* mdims [[buffer(3)]], device TF_OUT* Y [[buffer(4)]], const device float* XS [[buffer(5)]],
#if defined(TF_SWIGLU)
    device float* HS [[buffer(6)]],
#elif defined(TF_RESID)
    const device bfloat* H [[buffer(6)]], device bfloat* RO [[buffer(7)]],
#endif
    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr int TN = TF_PROMPT_TN;
  const int r0 = int(tg.y) * 16, rows = mdims[0] - r0;
  if (rows <= 0) return;
  X += (size_t)r0 * TF_K;
#if defined(TF_SWIGLU)
  Y += (size_t)r0 * (TF_N / 2); // SwiGLU rows are half the fused width
  HS += r0;
  const int lane = int(lanei), ks = int(sgi), M = min(rows, 16), stride = mdims[1], tile0 = 2 * int(tg.x) + ks * (TF_N / 64);
#else
  Y += (size_t)r0 * TF_YS;
#if defined(TF_RESID)
  H += (size_t)r0 * TF_YS;
  RO += (size_t)r0 * TF_YS;
#endif
  const int lane = int(lanei), ks = int(sgi), M = min(rows, 16), stride = mdims[1], tile0 = tf_tile(TN * int(tg.x)); // a range's TN-aligned tiles
#endif
  XS += r0;
  const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  // each 64-deep group as two 32-deep products into the same sums: a K=64 operand's registers are its two K=32
  // halves' registers in order (left: elements 0-15 then 16-31; right: 0-31 then 32-63)
  constexpr auto dm = matmul2d_descriptor(16, NT, 32, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto da = matmul2d_descriptor(16, NT, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto d64 = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<dm, execution_simdgroup> opm;
  matmul2d<da, execution_simdgroup> opa;
  matmul2d<d64, execution_simdgroup> op64;
  auto x64 = op64.template get_left_input_cooperative_tensor<bfloat, half, float>();
  auto xl0 = opm.template get_left_input_cooperative_tensor<bfloat, half, float>();
  auto xl1 = opa.template get_left_input_cooperative_tensor<bfloat, half, float>();
#define TF_TN_RIGHT opm.template get_right_input_cooperative_tensor<bfloat, half, float>()
#define TF_TN_DEST opm.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(xl0)>, metal::remove_addrspace_t<decltype(cbl)>, float>()
  auto cbl = TF_TN_RIGHT; auto cbh = TF_TN_RIGHT;
  auto P = TF_TN_DEST;
  const half2 bias = half2(1024.0h);
#if defined(TF_SWIGLU)
  const int first = TF_G0, last = TF_G0 + TF_GN; // one K split: the threadgroup's SIMD groups are the gate and up pairs
#else
  const int first = TF_G0 + (ks * TF_GN) / TF_SK, last = TF_G0 + ((ks + 1) * TF_GN) / TF_SK;
#endif
  const device uint4* cp = W + ((size_t)(tile0 / TF_TB) * KG * TF_TB + tile0 % TF_TB) * 64 + lane + (size_t)first * TF_TB * 64;
  const device uint4* sp = (const device uint4*)SB + ((size_t)first * TF_N + tile0 * NT + nb) / 4;
  uint4 nw0 = 0, nw1 = 0;
  if (first < last) { nw0 = cp[0]; nw1 = cp[32]; }
  float C[TN][16];
  TF_UNROLL
  for (int t = 0; t < TN; t++) {
    TF_UNROLL
    for (int i = 0; i < 16; i++) C[t][i] = 0.0f;
  }
  const device float* xp = XS + first * stride + mb;
  const device bfloat* xv = X + (size_t)first * GS;
  const array<int32_t, 2> xstr = {1, TF_K};
  for (int g = first; g < last; g++) {
    const float x0 = xp[0], x1 = xp[8];
    xp += stride;
    x64.load(tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>((device bfloat*)xv, dextents<int32_t, 2>(GS, M), xstr));
    xv += GS;
    TF_UNROLL
    for (int i = 0; i < 16; i++) { xl0[i] = x64[i]; xl1[i] = x64[16 + i]; }
    TF_TN_TILE(0)
    TF_TN_TILE(1)
#if TF_PROMPT_TN == 4
    TF_TN_TILE(2)
    TF_TN_TILE(3)
#endif
  }
#if defined(TF_SWIGLU)
  // tf_lane_reg's fused prompt epilogue, with tile t of SIMD group ks as its tile 2 * ks + t: each tile's words round to
  // bf16 as the unfused store does; SwiGLU reads them as the activation kernel reads the gate/up rows, and each 64-column
  // group's sum adds its values in order from zero.
  threadgroup bfloat gu16[4 * 16 * 32];
  threadgroup float hb16[16 * 64];
  TF_UNROLL
  for (int t = 0; t < TN; t++) {
    TF_UNROLL
    for (int i = 0; i < 16; i++) {
      const int m = mb + 8 * ((i >> 2) & 1);
      gu16[((2 * ks + t) * 16 + m) * 32 + nb + 16 * (i >> 3) + (i & 3)] = static_cast<bfloat>(C[t][i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t16 = ks * 32 + lane;
  for (int at = t16; at < 16 * 64; at += 64) {
    const int m = at / 64, c = at % 64;
    if (m >= M) continue;
    const float gf = float(gu16[((c >> 5) * 16 + m) * 32 + (c & 31)]);
    const bfloat h = bfloat(gf / (1.0f + metal::exp(-gf)) * float(gu16[((2 + (c >> 5)) * 16 + m) * 32 + (c & 31)]));
    Y[(size_t)m * (TF_N / 2) + int(tg.x) * 64 + c] = h;
    hb16[at] = float(h);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t16 < 16) {
    float acc = 0.0f;
    if (t16 < M)
      for (int i = 0; i < 64; i++) acc += hb16[t16 * 64 + i];
    HS[int(tg.x) * stride + t16] = acc;
  }
#else
  threadgroup float part[(TF_SK > 1 ? TF_SK - 1 : 1) * TN * 16 * 32];
  if (TF_SK > 1) {
    if (ks > 0) {
      TF_UNROLL
      for (int t = 0; t < TN; t++) {
        TF_UNROLL
        for (int i = 0; i < 16; i++) part[(((ks - 1) * TN + t) * 16 + i) * 32 + lane] = C[t][i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) {
      for (int s2 = 1; s2 < TF_SK; s2++) {
        TF_UNROLL
        for (int t = 0; t < TN; t++) {
          TF_UNROLL
          for (int i = 0; i < 16; i++) C[t][i] += part[(((s2 - 1) * TN + t) * 16 + i) * 32 + lane];
        }
      }
    }
  }
  if (ks == 0) {
    TF_UNROLL
    for (int t = 0; t < TN; t++) {
      TF_UNROLL
      for (int i = 0; i < 16; i++) {
        const int m = mb + 8 * ((i >> 2) & 1), n = (tile0 + t) * NT + nb + 16 * (i >> 3) + (i & 3);
#if defined(TF_RESID)
        // the residual stream's next rows: the output word as the store rounds it, added to the hidden word as q27_norm_xs
        // adds them (h + r, rounded to bf16); the norm then reads these rows alone
        if (m < M && n < TF_N) RO[(size_t)m * TF_YS + n] = bfloat(float(H[(size_t)m * TF_YS + n]) + float(static_cast<bfloat>(C[t][i])));
#else
        if (m < M && n < TF_N) Y[(size_t)m * TF_YS + n] = static_cast<TF_OUT>(C[t][i]);
#endif
      }
    }
  }
#endif
}
#endif
