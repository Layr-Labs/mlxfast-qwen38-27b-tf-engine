#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Device-local startup qualification using synthetic inputs only, with no model or prompt bindings.
// Input: eight finite patterns, each 16 rows x 256 BF16 values.
// Output:16 uints per (pattern,lane), i.e.8*32*16 uints total.
// Record: flags,BsmallCap,BwideCap,PsmallCap,PwideCap,firstKind,half,index,
//         wideIndex,smallX,smallY,wideX,wideY,liveRows,gotBits,expectedBits.
// Flags:1 Bcapacity;2 Pcapacity;4 Bvalid;8 Bcoords;16 Pvalid;
//       32 Pcoords;64 expectedPcoords;128 product;256 groupFMA.
// FirstKind:1 Bcoordinates/validity;2 Pcoordinates/validity;3 product;4 groupFMA.
inline void tf_n16_first(thread uint* d, uint kind, uint half_tile, uint index,
                         uint wide_index, uint sx, uint sy, uint wx, uint wy,
                         uint live_rows, uint got, uint expected) {
  if (d[5] != 0) return;
  d[5] = kind; d[6] = half_tile; d[7] = index; d[8] = wide_index;
  d[9] = sx; d[10] = sy; d[11] = wx; d[12] = wy;
  d[13] = live_rows; d[14] = got; d[15] = expected;
}

inline half tf_n16_code(uint k, uint n, uint group, uint pattern) {
  uint hash = k * 1664525u ^ n * 1013904223u ^ group * 747796405u ^ pattern * 2891336453u;
  hash ^= hash >> 16; hash *= 2246822519u; hash ^= hash >> 13;
  return half(hash & 15u);
}

kernel void tf_lane_n16_qualify(const device bfloat* X [[buffer(0)]], device uint* status [[buffer(1)]],
                              uint lane [[thread_index_in_simdgroup]], uint pattern [[threadgroup_position_in_grid]]) {
  constexpr auto dsmall = matmul2d_descriptor(16, 16, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dwide = matmul2d_descriptor(16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<dsmall, execution_simdgroup> osmall;
  matmul2d<dwide, execution_simdgroup> owide;
  auto bsmall = osmall.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto bwide = owide.template get_right_input_cooperative_tensor<bfloat, half, float>();
  uint diagnostic[16] = {};
  uint bad = 0;
  const uint mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  const uint nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1);
  diagnostic[1] = uint(bsmall.get_capacity());
  diagnostic[2] = uint(bwide.get_capacity());
  if (diagnostic[1] != 32 || diagnostic[2] != 64) bad |= 1u;
  // No lane exits early: every collective operation remains SIMD-uniform even on a failed layout.
  for (uint live_rows = 1; live_rows <= 8; ++live_rows) {
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> asmall(
        (device bfloat*)X + pattern * 16 * 256, dextents<int32_t, 2>(256, int(live_rows)));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> awide(
        (device bfloat*)X + pattern * 16 * 256, dextents<int32_t, 2>(256, int(live_rows)));
    auto xsmall = asmall.slice(0, 0);
    auto xwide = awide.slice(0, 0);
    auto psmall = osmall.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(xsmall)>, metal::remove_addrspace_t<decltype(bsmall)>, float>();
    auto pwide = owide.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(xwide)>, metal::remove_addrspace_t<decltype(bwide)>, float>();
    diagnostic[3] = uint(psmall.get_capacity());
    diagnostic[4] = uint(pwide.get_capacity());
    if (diagnostic[3] != 8 || diagnostic[4] != 16) bad |= 2u;
    for (uint half_tile = 0; half_tile < 2; ++half_tile) {
      for (uint i = 0; i < min(uint(bsmall.get_capacity()), 32u); ++i) {
        const uint old_i = (i & 7u) | ((i & ~7u) << 1) | (half_tile << 3);
        auto qs = bsmall.get_multidimensional_index(i);
        if (!bsmall.is_valid_element(i) || old_i >= uint(bwide.get_capacity())) {
          bad |= 4u;
          tf_n16_first(diagnostic, 1, half_tile, i, old_i, uint(qs[0]), uint(qs[1]), 0, 0, live_rows, 0, 0);
          continue;
        }
        auto qw = bwide.get_multidimensional_index(old_i);
        if (!bwide.is_valid_element(old_i)) bad |= 4u;
        if (qs[0] != qw[0] || qs[1] + int(16u * half_tile) != qw[1]) {
          bad |= 8u;
          tf_n16_first(diagnostic, 1, half_tile, i, old_i, uint(qs[0]), uint(qs[1]), uint(qw[0]), uint(qw[1]), live_rows, 0, 0);
        }
      }
      for (uint i = 0; i < min(uint(psmall.get_capacity()), 8u); ++i) {
        const uint old_i = i + 8u * half_tile;
        auto qs = psmall.get_multidimensional_index(i);
        if (!psmall.is_valid_element(i) || old_i >= uint(pwide.get_capacity())) {
          bad |= 16u;
          tf_n16_first(diagnostic, 2, half_tile, i, old_i, uint(qs[0]), uint(qs[1]), 0, 0, live_rows, 0, 0);
          continue;
        }
        auto qw = pwide.get_multidimensional_index(old_i);
        if (!pwide.is_valid_element(old_i) || qs[0] + int(16u * half_tile) != qw[0] || qs[1] != qw[1]) {
          bad |= 32u;
          tf_n16_first(diagnostic, 2, half_tile, i, old_i, uint(qs[0]), uint(qs[1]), uint(qw[0]), uint(qw[1]), live_rows, 0, 0);
        }
        if (uint(qs[0]) != nb + (i & 3u) || uint(qs[1]) != mb + 8u * (i >> 2)) {
          bad |= 64u;
          tf_n16_first(diagnostic, 2, half_tile, i, old_i, uint(qs[0]), uint(qs[1]), uint(qw[0]), uint(qw[1]), live_rows, 0, 0);
        }
      }
      float csmall[8] = {}, cwide[8] = {};
      for (uint group = 0; group < 4; ++group) {
        for (uint i = 0; i < bwide.get_capacity(); ++i) {
          if (!bwide.is_valid_element(i)) continue;
          auto q = bwide.get_multidimensional_index(i);
          bwide[i] = tf_n16_code(uint(q[0]), uint(q[1]), group, pattern);
        }
        for (uint i = 0; i < bsmall.get_capacity(); ++i) {
          if (!bsmall.is_valid_element(i)) continue;
          const uint old_i = (i & 7u) | ((i & ~7u) << 1) | (half_tile << 3);
          // CPU proof establishes that this exact subset is what the vector half-bank loader produces.
          bsmall[i] = old_i < uint(bwide.get_capacity()) && bwide.is_valid_element(old_i) ? bwide[old_i] : half(0);
        }
        auto ssmall = asmall.slice(group * 64, 0);
        auto swide = awide.slice(group * 64, 0);
        osmall.run(ssmall, bsmall, psmall);
        owide.run(swide, bwide, pwide);
        for (uint i = 0; i < min(uint(psmall.get_capacity()), 8u); ++i) {
          const uint old_i = i + 8u * half_tile;
          const uint row = mb + 8u * (i >> 2);
          if (row >= live_rows || !psmall.is_valid_element(i) || old_i >= uint(pwide.get_capacity()) || !pwide.is_valid_element(old_i)) continue;
          const uint got = as_type<uint>(psmall[i]), expected = as_type<uint>(pwide[old_i]);
          if (got != expected) {
            bad |= 128u;
            auto qs = psmall.get_multidimensional_index(i);
            auto qw = pwide.get_multidimensional_index(old_i);
            tf_n16_first(diagnostic, 3, half_tile, i, old_i, uint(qs[0]), uint(qs[1]), uint(qw[0]), uint(qw[1]), live_rows, got, expected);
          }
          float sum = 0.0f;
          for (uint k = 0; k < 64; ++k) sum += float(X[pattern * 16 * 256 + row * 256 + group * 64 + k]);
          const uint column = nb + (i & 3u) + half_tile * 16;
          const float scale = float(as_type<bfloat>(ushort(0x3b00u + ((column + group) & 127u))));
          const float bias = float(as_type<bfloat>(ushort(0xbb00u + ((column + 3u * group) & 127u))));
          csmall[i] = fma(scale, psmall[i], fma(bias, sum, csmall[i]));
          cwide[i] = fma(scale, pwide[old_i], fma(bias, sum, cwide[i]));
          if (as_type<uint>(csmall[i]) != as_type<uint>(cwide[i])) {
            bad |= 256u;
            tf_n16_first(diagnostic, 4, half_tile, i, old_i, column, row, column, row, live_rows, as_type<uint>(csmall[i]), as_type<uint>(cwide[i]));
          }
        }
      }
    }
  }
  diagnostic[0] = bad;
  for (uint i = 0; i < 16; ++i) status[(pattern * 32 + lane) * 16 + i] = diagnostic[i];
}
