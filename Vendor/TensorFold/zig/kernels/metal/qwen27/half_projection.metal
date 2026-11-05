#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

Q27_SPECIALIZE

[[max_total_threads_per_threadgroup(32)]]
kernel void tf_q27_half_down(const device bfloat* X [[buffer(0)]],
                             const device half* W [[buffer(1)]],
                             const device bfloat* H [[buffer(2)]],
                             const device uint* OFF [[buffer(3)]],
                             const device ushort* EXK [[buffer(4)]],
                             const device float* DELTA [[buffer(5)]],
                             device bfloat* RO [[buffer(6)]],
                             constant uint& ROWS [[buffer(7)]],
                             uint2 tg [[threadgroup_position_in_grid]]) {
  const int n0 = int(tg.x) * 32;
  const int m0 = int(tg.y) * 16;
  const int rows = min(int(ROWS) - m0, 16);
  if (rows <= 0) return;

  const array<int32_t, 2> xs = {1, TF_K};
  const array<int32_t, 2> ws = {1, TF_K};
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(
      (device bfloat*)(X + (size_t)m0 * TF_K), dextents<int32_t, 2>(TF_K, 16), xs);
  tensor<device half, dextents<int32_t, 2>, tensor_inline> B(
      (device half*)(W + (size_t)n0 * TF_K), dextents<int32_t, 2>(TF_K, 32), ws);
  constexpr auto desc = matmul2d_descriptor(16, 32, static_cast<int>(metal::dynamic_extent), false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto C = op.template get_destination_cooperative_tensor<decltype(A), decltype(B), float>();
  op.run(A, B, C);
  #pragma unroll
  for (uint i = 0; i < C.get_capacity(); ++i) {
    if (!C.is_valid_element(i)) continue;
    const auto id = C.get_multidimensional_index(i);
    const int n = n0 + int(id[0]);
    const int m = m0 + int(id[1]);
    float value = C[i];
    for (uint j = OFF[n]; j < OFF[n + 1]; ++j) {
      value = fma(float(X[(size_t)m * TF_K + EXK[j]]), DELTA[j], value);
    }
    const bfloat projection = static_cast<bfloat>(value);
    const bfloat hidden = H[(size_t)m * TF_N + n];
    RO[(size_t)m * TF_N + n] = bfloat(float(hidden) + float(projection));
  }
}
