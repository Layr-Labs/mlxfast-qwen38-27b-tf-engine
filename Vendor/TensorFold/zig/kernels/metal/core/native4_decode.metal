#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Qwen27's target decoded DOWN projection: one 32-column tile, all G64 groups,
// native BF16 x device-U4 products. Host ABI admission validates every P slot
// before changing cfg[0] from query (1) to run (0).
constant uint TF_N4_N = 5120;
constant uint TF_N4_K = 17408;
constant uint TF_N4_GS = 64;
constant uint TF_N4_NG = TF_N4_K / TF_N4_GS;
constant uint TF_N4_TILES = TF_N4_N / 32;
constant uint TF_N4_RECORD_BASE = 4;
constant uint TF_N4_RECORD_WORDS = 36;
constant uint TF_N4_MAGIC = 0x4E345041u;
constant uint TF_N4_VERSION = 1u;
constant uint TF_N4_MAX_SLOTS = 16;

#ifndef TF_SK
#error TF_SK must be 2 or 4
#endif
#if TF_SK != 2 && TF_SK != 4
#error native4 decoded DOWN supports physical SK2 or SK4 only
#endif

inline uint tf_n4_pair(const device bfloat* SB, uint g, uint n) {
#if defined(TF_SBI12)
  const device uint* table = (const device uint*)SB;
  const uint col = n & 31u;
  const uint bit = 12u * ((col / 16u) * 4u + col % 4u);
  const uint tile_group = g * (TF_N4_N / 32u) + n / 32u;
  const uint at = TF_SBI + tile_group * 12u + ((col % 16u) / 4u) * 3u + bit / 32u;
  const uint shift = bit & 31u;
  // A 12-bit field at bit offset <=20 fits entirely in this word. Read the
  // following word only for the crossing offsets24/28. The final packed
  // word contains columns30/31; their offsets8/20 fit without another read.
  const uint packed = shift <= 20u ? (table[at] >> shift) :
      (table[at] >> shift) | (table[at + 1u] << (32u - shift));
  return table[packed & 0xFFFu];
#elif defined(TF_SBI)
  const device uint* table = (const device uint*)SB;
  const device ushort* index = (const device ushort*)(table + TF_SBI);
  return table[index[g * TF_N4_N + n]];
#else
  const device ushort* pair = (const device ushort*)SB + 2u * (g * TF_N4_N + n);
  return uint(pair[0]) | (uint(pair[1]) << 16);
#endif
}

#define TF_N4_ACC_SLOT(I) do { \
  const uint bit = 1u << (I); \
  if ((live_mask & bit) != 0u && uint(I) < cap) { \
    const uint packed = cfg[rec + 4u + 2u * (I)]; \
    const uint row = packed >> 5; \
    const uint col = packed & 31u; \
    if (row < rows) { \
      const uint pair = tf_n4_pair(SB, uint(g), tile * 32u + col); \
      const float folded_scale = float(as_type<bfloat>(ushort(pair & 0xFFFFu))); \
      const float scale = folded_scale * 0x1p-24f; \
      const float bias = float(as_type<bfloat>(ushort(pair >> 16))); \
      const float xsum = XS[uint(g) * stride + row]; \
      C[I] = fma(scale, P[ushort(I)], fma(bias, xsum, C[I])); \
    } \
  } \
} while (0)

[[max_total_threads_per_threadgroup(32 * TF_SK)]]
[[kernel]] void tf_lane_native4_decode(
    const device bfloat* X [[buffer(0)]],
    const device uchar* W [[buffer(1)]],
    const device bfloat* SB [[buffer(2)]],
    const device int* mdims [[buffer(3)]],
    device bfloat* Y [[buffer(4)]],
    const device float* XS [[buffer(5)]],
    device atomic_uint* TC [[buffer(6)]],
    device atomic_uint* WC [[buffer(9)]],
    constant uint& wait_for [[buffer(10)]],
    device uint* cfg [[buffer(11)]],
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tgs [[threadgroups_per_grid]]) {
  const uint rows = uint(mdims[0]);
  const uint stride = uint(mdims[1]);
  const uint tile = tg.x;
  const uint rec = TF_N4_RECORD_BASE + (sgi * 32u + lane) * TF_N4_RECORD_WORDS;

  // The rank runner zeroes/guards the ABI buffer and initializes header[0..3]
  // before this same specialization is invoked in QUERY mode. Build only MPP
  // descriptors here; QUERY returns before any input/weight/metadata read,
  // wait, operation, output, or counter update.
  const array<int32_t, 2> xstr = {1, int(TF_N4_K)};
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a(
      (device bfloat*)X, dextents<int32_t, 2>(int(TF_N4_K), int(rows)), xstr);
  auto a0 = a.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0(
      (device uchar*)W, dextents<int32_t, 2>(int(TF_N4_GS), 32));
  constexpr auto desc = matmul2d_descriptor(8, 32, 64, false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();

  if (cfg[0] == 1u) {
    const uint cap = P.get_capacity();
    uint valid_mask = 0u;
    cfg[rec + 0u] = sgi;
    cfg[rec + 1u] = lane;
    cfg[rec + 2u] = cap;
    for (uint i = 0u; i < TF_N4_MAX_SLOTS; ++i) {
      const bool valid = i < cap && P.is_valid_element(ushort(i));
      const uint at = rec + 4u + 2u * i;
      if (valid) {
        const auto xy = P.get_multidimensional_index(ushort(i));
        cfg[at] = uint(xy[0]);
        cfg[at + 1u] = uint(xy[1]);
        valid_mask |= 1u << i;
      } else {
        cfg[at] = 0xFFFFFFFFu;
        cfg[at + 1u] = 0xFFFFFFFFu;
      }
    }
    cfg[rec + 3u] = valid_mask;
    return;
  }

  // RUN mode is admitted by the host only after every query record and guard
  // has been checked and activated. Normal execution reads only that frozen
  // per-lane slot mask/packed coordinate; P accesses below are literal slots.
  const uint cap = cfg[rec + 2u];
  const uint live_mask = cfg[rec + 3u];
  const uint first = (sgi * TF_N4_NG) / TF_SK;
  const uint last = ((sgi + 1u) * TF_N4_NG) / TF_SK;
  float C[TF_N4_MAX_SLOTS];
  for (uint i = 0u; i < TF_N4_MAX_SLOTS; ++i) C[i] = 0.0f;

  if (sgi == 0u && lane == 0u) {
    for (uint spin = 0u; spin < (1u << 24) && atomic_load_explicit(&WC[0], memory_order_relaxed) < wait_for; ++spin) {}
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (atomic_fetch_add_explicit(&WC[1], 1u, memory_order_relaxed) + 1u == tgs.x * tgs.y) {
      atomic_store_explicit(&WC[0], 0u, memory_order_relaxed);
      atomic_store_explicit(&WC[1], 0u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint g = first; g < last; ++g) {
    const size_t group_byte = (size_t(tile) * TF_N4_NG + g) * (32u * TF_N4_GS / 2u);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)W + group_byte, dextents<int32_t, 2>(int(TF_N4_GS), 32));
    auto x = a.slice(int(g * TF_N4_GS), 0);
    op.run(x, b, P);
    TF_N4_ACC_SLOT(0);  TF_N4_ACC_SLOT(1);  TF_N4_ACC_SLOT(2);  TF_N4_ACC_SLOT(3);
    TF_N4_ACC_SLOT(4);  TF_N4_ACC_SLOT(5);  TF_N4_ACC_SLOT(6);  TF_N4_ACC_SLOT(7);
    TF_N4_ACC_SLOT(8);  TF_N4_ACC_SLOT(9);  TF_N4_ACC_SLOT(10); TF_N4_ACC_SLOT(11);
    TF_N4_ACC_SLOT(12); TF_N4_ACC_SLOT(13); TF_N4_ACC_SLOT(14); TF_N4_ACC_SLOT(15);
  }

  // P slot orders may differ between physical simdgroups. Stage by activated
  // logical output index, never by slot/lane position, then gather through
  // SG0's own public slot map. Coverage/uniqueness is proven by host admission.
  threadgroup float part[(TF_SK - 1) * 256];
  #define TF_N4_STAGE_SLOT(I) do { \
    const uint bit = 1u << (I); \
    if ((live_mask & bit) != 0u && uint(I) < cap) { \
      const uint packed = cfg[rec + 4u + 2u * (I)]; \
      if ((packed >> 5) < rows) part[(sgi - 1u) * 256u + packed] = C[I]; \
    } \
  } while (0)
  if (sgi > 0u) {
    TF_N4_STAGE_SLOT(0);  TF_N4_STAGE_SLOT(1);  TF_N4_STAGE_SLOT(2);  TF_N4_STAGE_SLOT(3);
    TF_N4_STAGE_SLOT(4);  TF_N4_STAGE_SLOT(5);  TF_N4_STAGE_SLOT(6);  TF_N4_STAGE_SLOT(7);
    TF_N4_STAGE_SLOT(8);  TF_N4_STAGE_SLOT(9);  TF_N4_STAGE_SLOT(10); TF_N4_STAGE_SLOT(11);
    TF_N4_STAGE_SLOT(12); TF_N4_STAGE_SLOT(13); TF_N4_STAGE_SLOT(14); TF_N4_STAGE_SLOT(15);
  }
  #undef TF_N4_STAGE_SLOT
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #define TF_N4_REDUCE_SLOT(I) do { \
    const uint bit = 1u << (I); \
    if ((live_mask & bit) != 0u && uint(I) < cap) { \
      const uint packed = cfg[rec + 4u + 2u * (I)]; \
      if ((packed >> 5) < rows) for (uint s = 1u; s < TF_SK; ++s) C[I] += part[(s - 1u) * 256u + packed]; \
    } \
  } while (0)
  if (sgi == 0u) {
    TF_N4_REDUCE_SLOT(0);  TF_N4_REDUCE_SLOT(1);  TF_N4_REDUCE_SLOT(2);  TF_N4_REDUCE_SLOT(3);
    TF_N4_REDUCE_SLOT(4);  TF_N4_REDUCE_SLOT(5);  TF_N4_REDUCE_SLOT(6);  TF_N4_REDUCE_SLOT(7);
    TF_N4_REDUCE_SLOT(8);  TF_N4_REDUCE_SLOT(9);  TF_N4_REDUCE_SLOT(10); TF_N4_REDUCE_SLOT(11);
    TF_N4_REDUCE_SLOT(12); TF_N4_REDUCE_SLOT(13); TF_N4_REDUCE_SLOT(14); TF_N4_REDUCE_SLOT(15);
  }
  #undef TF_N4_REDUCE_SLOT
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgi == 0u) {
    // Keep each slot's store index public and constant; row and tile column
    // are activated ABI data. Inactive/padded slots perform no store.
    #define TF_N4_STORE_SLOT(I) do { \
      const uint bit = 1u << (I); \
      if ((live_mask & bit) != 0u && uint(I) < cap) { \
        const uint packed = cfg[rec + 4u + 2u * (I)]; \
        const uint row = packed >> 5; \
        const uint col = packed & 31u; \
        if (row < rows) Y[size_t(row) * TF_N4_N + tile * 32u + col] = static_cast<bfloat>(C[I]); \
      } \
    } while (0)
    TF_N4_STORE_SLOT(0);  TF_N4_STORE_SLOT(1);  TF_N4_STORE_SLOT(2);  TF_N4_STORE_SLOT(3);
    TF_N4_STORE_SLOT(4);  TF_N4_STORE_SLOT(5);  TF_N4_STORE_SLOT(6);  TF_N4_STORE_SLOT(7);
    TF_N4_STORE_SLOT(8);  TF_N4_STORE_SLOT(9);  TF_N4_STORE_SLOT(10); TF_N4_STORE_SLOT(11);
    TF_N4_STORE_SLOT(12); TF_N4_STORE_SLOT(13); TF_N4_STORE_SLOT(14); TF_N4_STORE_SLOT(15);
    #undef TF_N4_STORE_SLOT
  }

  threadgroup_barrier(mem_flags::mem_device);
  if (sgi == 0u && lane == 0u) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_fetch_add_explicit(TC, 1u, memory_order_relaxed);
  }
}

#undef TF_N4_ACC_SLOT
