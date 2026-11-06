// Direct device-U4 K64 SG1 GU source/AOT premise. Default OFF and unregistered.
// Reuses legal A64coop/deviceU4 API from prior b7f1c082, EB71 Gate/Up SG ownership.
// Canonical [tile64][group80][col64][K64 packednibbles], originalS/B without rescaling.
// K64 versus twoK32 may change FP32 reduction; no numeric/M5/performance claim.
// Standalone SG1 M16N32 both-cooperative HALF carrier premise. Default OFF.
// Source contract only: no host registration, no M5 performance or numeric claim.
// Each of two physical SGs owns Gate or Up, each with two horizontal N32 tiles.
// SAME actual PSO runtime QUERY/RUN admission through buffer11 is mandatory.
// Query all64threads, cap16/all16valid, local512unique32x16 coords PER SG.
#ifndef TF_NATIVE_GU_M16_DIRECT_U4_K64_PROTOTYPE
#define TF_NATIVE_GU_M16_DIRECT_U4_K64_PROTOTYPE 0
#endif
#ifndef TF_NATIVE_GU_M16_DIRECT_U4_K64_TRACE_C
#define TF_NATIVE_GU_M16_DIRECT_U4_K64_TRACE_C 0
#endif
#if TF_NATIVE_GU_M16_DIRECT_U4_K64_PROTOTYPE
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct tf_native_gu_m16_direct_u4_k64_params {
  uint rows, x_stride, sb_stride, y_stride, sum_stride, tiles;
};
[[max_total_threads_per_threadgroup(64)]]
[[kernel]] void tf_native_gu_m16_direct_u4_k64_prototype(
    const device bfloat* X [[buffer(0)]],
    const device uchar* WG [[buffer(1)]],
    const device uchar* WU [[buffer(2)]],
    const device uint* SG [[buffer(3)]],
    const device uint* SU [[buffer(4)]],
    const device float* XS [[buffer(5)]],
    device bfloat* Y [[buffer(6)]], device float* HS [[buffer(7)]],
    constant tf_native_gu_m16_direct_u4_k64_params& p [[buffer(8)]],
#if TF_NATIVE_GU_M16_DIRECT_U4_K64_TRACE_C
    device float* trace_gate [[buffer(9)]], device float* trace_up [[buffer(10)]],
#endif
    device uint* abi [[buffer(11)]],
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto dm = matmul2d_descriptor(16,32,64,false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<dm,execution_simdgroup> op;
  using XT = tensor<device bfloat,dextents<int32_t,2>,tensor_inline>;
  using WT = tensor<device uint4b_format,dextents<int32_t,2>,tensor_inline>;
  auto A = op.template get_left_input_cooperative_tensor<bfloat,uint4b_format,float>();
  auto P = op.template get_destination_cooperative_tensor<decltype(A),WT,float>();
  const uint mode = abi[0];
  if (mode == 1u) {
    constexpr uint RECORD_WORDS=388u;
    const uint tid=sgi*32u+lane, base=4u+tid*RECORD_WORDS;
    const uint capacity=uint(P.get_capacity());
    if(tid==0u){abi[1]=1u;abi[2]=64u;abi[3]=RECORD_WORDS;}
    abi[base]=capacity;abi[base+1]=sgi;abi[base+2]=lane;abi[base+3]=1u;
    for(uint i=0;i<128u;++i){
      const uint at=base+4u+i*3u;
      const bool valid=i<capacity && P.is_valid_element(ushort(i));
      if(valid){const auto xy=P.get_multidimensional_index(ushort(i));abi[at]=1u;abi[at+1]=uint(xy[0]);abi[at+2]=uint(xy[1]);}
      else{abi[at]=0u;abi[at+1]=0xffffffffu;abi[at+2]=0xffffffffu;}
    }
    return; // TG-uniform QUERY before any parameter/input access or cooperative load.
  }
  if(mode!=0u)return;
  const size_t r0=size_t(tg.y)*16u;
  if(tg.x>=p.tiles || r0>=p.rows)return; // TG-uniform before all barriers.
  const uint M=min(p.rows-uint(r0),16u), n0=tg.x*64u;
  const array<int32_t,2> xstr={1,int32_t(p.x_stride)}, wstr={1,64};
  const device uchar* W=sgi==0u ? WG:WU; // SG-uniform Gate/Up selection.
  const device uint* SB=sgi==0u ? SG:SU;
  float C0[16],C1[16];
#define TF_BC16_APPLY(F) \
  F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) \
  F(8) F(9) F(10) F(11) F(12) F(13) F(14) F(15)
#define TF_BC16_ZERO(S) C0[S]=0.0f;C1[S]=0.0f;
  TF_BC16_APPLY(TF_BC16_ZERO)
#undef TF_BC16_ZERO
  for(uint g=0;g<80u;++g){
    const device bfloat* xg=X+r0*p.x_stride+g*64u;
    XT x((device bfloat*)xg,dextents<int32_t,2>(64,int32_t(M)),xstr);
    A.load(x); // One public BF16 K64 load reused by both horizontal N32 products.
    constexpr size_t GROUP_BYTES=2048u,TILE_BYTES=80u*GROUP_BYTES;
    const size_t wb=size_t(tg.x)*TILE_BYTES+g*GROUP_BYTES;
    WT w0((device uchar*)(W+wb),dextents<int32_t,2>(64,32),wstr);
    op.run(A,w0,P);
#define TF_BC16_FMA0(S) do { \
    constexpr ushort slot=ushort(S); \
    if(P.is_valid_element(slot)){const auto xy=P.get_multidimensional_index(slot);const uint col=uint(xy[0]),m=uint(xy[1]); \
      if(m<M && col<32u){const vec<bfloat,2> sb=as_type<vec<bfloat,2>>(SB[size_t(g)*p.sb_stride+n0+col]); \
        const float xs=XS[size_t(g)*p.sum_stride+r0+m];C0[S]=fma(float(sb.x),P[slot],fma(float(sb.y),xs,C0[S]));}} \
  } while(false);
    TF_BC16_APPLY(TF_BC16_FMA0)
#undef TF_BC16_FMA0
    WT w1((device uchar*)(W+wb+1024u),dextents<int32_t,2>(64,32),wstr);
    op.run(A,w1,P);
#define TF_BC16_FMA1(S) do { \
    constexpr ushort slot=ushort(S); \
    if(P.is_valid_element(slot)){const auto xy=P.get_multidimensional_index(slot);const uint col=uint(xy[0]),m=uint(xy[1]); \
      if(m<M && col<32u){const vec<bfloat,2> sb=as_type<vec<bfloat,2>>(SB[size_t(g)*p.sb_stride+n0+32u+col]); \
        const float xs=XS[size_t(g)*p.sum_stride+r0+m];C1[S]=fma(float(sb.x),P[slot],fma(float(sb.y),xs,C1[S]));}} \
  } while(false);
    TF_BC16_APPLY(TF_BC16_FMA1)
#undef TF_BC16_FMA1
  }
  threadgroup bfloat GU[2u*16u*64u]; // Separate Gate and Up planes: 4KiB.
  threadgroup bfloat H[16u*64u]; // Separate activation plane: 2KiB. No inplace rider.
#if TF_NATIVE_GU_M16_DIRECT_U4_K64_TRACE_C
#define TF_BC16_TRACE(S,COL,VALUE) do { \
  if(sgi==0u)trace_gate[(r0+m)*p.y_stride+n0+COL]=VALUE; \
  else trace_up[(r0+m)*p.y_stride+n0+COL]=VALUE; \
  } while(false);
#else
#define TF_BC16_TRACE(S,COL,VALUE)
#endif
#define TF_BC16_PUBLISH(S) do { \
    constexpr ushort slot=ushort(S); \
    if(P.is_valid_element(slot)){const auto xy=P.get_multidimensional_index(slot);const uint col=uint(xy[0]),m=uint(xy[1]); \
      if(m<M && col<32u){GU[sgi*16u*64u+m*64u+col]=bfloat(C0[S]);GU[sgi*16u*64u+m*64u+32u+col]=bfloat(C1[S]); \
        TF_BC16_TRACE(S,col,C0[S]) TF_BC16_TRACE(S,32u+col,C1[S])}} \
  } while(false);
  TF_BC16_APPLY(TF_BC16_PUBLISH)
#undef TF_BC16_PUBLISH
#undef TF_BC16_TRACE
  threadgroup_barrier(mem_flags::mem_threadgroup);
#define TF_BC16_ACTIVATE(S) do { \
    constexpr ushort slot=ushort(S); \
    if(P.is_valid_element(slot)){const auto xy=P.get_multidimensional_index(slot);const uint col=sgi*32u+uint(xy[0]),m=uint(xy[1]); \
      if(m<M && uint(xy[0])<32u){const bfloat gb=GU[m*64u+col],ub=GU[16u*64u+m*64u+col];const float gf=float(gb); \
        const bfloat h=bfloat(gf/(1.0f+metal::exp(-gf))*float(ub)); \
        Y[(r0+m)*p.y_stride+n0+col]=h;H[m*64u+col]=h;}} \
  } while(false);
  TF_BC16_APPLY(TF_BC16_ACTIVATE)
#undef TF_BC16_ACTIVATE
#undef TF_BC16_APPLY
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint t=sgi*32u+lane;
  if(t<M){float acc=0.0f;for(uint col=0;col<64u;++col)acc+=float(H[t*64u+col]);HS[size_t(tg.x)*p.sum_stride+r0+t]=acc;}
}
#endif
