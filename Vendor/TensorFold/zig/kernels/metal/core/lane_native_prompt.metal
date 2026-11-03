// Native-u4 prompt-only affine GEMM. See Nemotron coop_qkv_1_0 for the packed tensor API.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct NativePromptArgs { uint n, k, slices, rows, sum_stride; };
template<uint SK> inline void native_prompt(const device bfloat* X,
    const device uchar* W, const device uint* SB,
    const device float* XS, device bfloat* Y, constant NativePromptArgs& p,
    uint3 tg, uint sg, uint lane, threadgroup float* partial) {
  constexpr auto dm = matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply);
  constexpr auto da = matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dm,execution_simdgroup> opm;
  matmul2d<da,execution_simdgroup> opa;
  auto lo = opm.template get_left_input_cooperative_tensor<bfloat,uint4b_format,float>();
  auto hi = opa.template get_left_input_cooperative_tensor<bfloat,uint4b_format,float>();
  const uint rb=tg.y*16, m=min(16u,p.rows-rb), ng=p.k/64;
  const uint chunk=(ng+SK-1)/SK, first=min(sg*chunk,ng), last=min(first+chunk,ng);
  const array<int32_t,2> xstr={1,int(p.k)}, wstr={1,64};
  using Right = tensor<device uint4b_format,dextents<int32_t,2>,tensor_inline>;
  auto P = opm.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(lo)>,Right,float>();
  ushort col[16], row[16];
  for (ushort i=0;i<16;i++) { auto c=P.get_multidimensional_index(i); col[i]=c[0]; row[i]=c[1]; }
  float C[2][16];
  for (uint t=0;t<2;t++) for (uint i=0;i<16;i++) C[t][i]=0.0f;
  for (uint g=first;g<last;g++) {
    const device bfloat* x=X+(size_t)rb*p.k+g*64;
    lo.load(tensor<device bfloat,dextents<int32_t,2>,tensor_inline>((device bfloat*)x,dextents<int32_t,2>(32,m),xstr));
    hi.load(tensor<device bfloat,dextents<int32_t,2>,tensor_inline>((device bfloat*)(x+32),dextents<int32_t,2>(32,m),xstr));
    #pragma clang loop unroll(full)
    for (uint t=0;t<2;t++) {
      const device uchar* w=W+((size_t(tg.x)*ng+g)*64+t*32)*32;
      Right wl((device uchar*)w,dextents<int32_t,2>(32,32),wstr);
      Right wh((device uchar*)w+16,dextents<int32_t,2>(32,32),wstr);
      opm.run(lo,wl,P);
      opa.run(hi,wh,P);
      #pragma clang loop unroll(full)
      for (uint i=0;i<16;i++) {
        const uint n=tg.x*64+t*32+col[i];
        const vec<bfloat,2> sb=as_type<vec<bfloat,2>>(SB[(size_t)g*p.n+n]);
        const float xs=row[i]<m?XS[(size_t)g*p.sum_stride+rb+row[i]]:0.0f;
        // Decode retains folded scale and its 2^-24 half operands; native codes restore original scale.
        const float s=float(sb[0])*0x1p-24f;
        C[t][i]=fma(s,P[i],fma(float(sb[1]),xs,C[t][i]));
      }
    }
  }
  if (SK>1) {
    if (sg>0) for (uint t=0;t<2;t++) for (uint i=0;i<16;i++)
      partial[(((sg-1)*2+t)*16+row[i])*32+col[i]]=C[t][i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0) for (uint s=1;s<SK;s++) for (uint t=0;t<2;t++) for (uint i=0;i<16;i++)
      C[t][i]+=partial[(((s-1)*2+t)*16+row[i])*32+col[i]];
  }
  if (sg==0) for (uint t=0;t<2;t++) for (uint i=0;i<16;i++) if (row[i]<m)
    Y[(size_t)(rb+row[i])*p.n+tg.x*64+t*32+col[i]]=bfloat(C[t][i]);
}

#define NATIVE_PROMPT_ENTRY(SK) \
kernel void tf_lane_native_prompt_sk##SK(const device bfloat* X [[buffer(0)]], \
    const device uchar* W [[buffer(1)]], const device uint* SB [[buffer(2)]], \
    const device float* XS [[buffer(3)]], device bfloat* Y [[buffer(4)]], \
    constant NativePromptArgs& p [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float partial[(SK>1?SK-1:1)*2*16*32]; \
  native_prompt<SK>(X,W,SB,XS,Y,p,tg,sg,lane,partial); \
}
NATIVE_PROMPT_ENTRY(1)
NATIVE_PROMPT_ENTRY(2)
NATIVE_PROMPT_ENTRY(4)
NATIVE_PROMPT_ENTRY(8)
