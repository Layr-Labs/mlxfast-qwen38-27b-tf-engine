#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Every dispatch row block contains exactly 128 readable rows.
[[max_total_threads_per_threadgroup(256)]]
kernel void q27_dense_prompt_gu(
    const device bfloat* X [[buffer(0)]],
    const device half* W [[buffer(1)]],
    const device uint* offsets [[buffer(2)]],
    const device ushort* indices [[buffer(3)]],
    const device float* residual [[buffer(4)]],
    device bfloat* GU [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr int M=128, N=64, K=5120;
  constexpr auto d=matmul2d_descriptor(M,N,K,false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<d,execution_simdgroups<8>> op;
  tensor<device bfloat,dextents<int32_t,2>,tensor_inline> a((device bfloat*)(X+size_t(tg.y)*M*K),dextents<int32_t,2>(K,M));
  tensor<device half,dextents<int32_t,2>,tensor_inline> b((device half*)(W+size_t(tg.x)*N*K),dextents<int32_t,2>(K,N));
  auto P=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
  op.run(a,b,P);
  for(ushort i=0;i<P.get_capacity();++i) if(P.is_valid_element(i)) {
    const auto xy=P.get_multidimensional_index(i);
    const uint row=tg.y*M+uint(xy[1]), column=tg.x*N+uint(xy[0]);
    if(row<rows) {
      float c=P[i]*(1.0f/256.0f);
      for(uint j=offsets[column];j<offsets[column+1];++j)
        c=fma(residual[j],float(X[size_t(row)*K+indices[j]]),c);
      GU[size_t(row)*34816+column]=bfloat(c);
    }
  }
}
