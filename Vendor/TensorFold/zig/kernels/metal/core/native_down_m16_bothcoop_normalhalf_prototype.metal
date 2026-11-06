// Standalone PROMPT Down NORMAL-HALF carrier prototype. Default OFF, unregistered.
// Same M16/TN2/SK2 and per-output group/split/residual order as protected 989e.
// W is numeric HALF(q0..15), SB is ORIGINAL (unfolded) BF16 scale/bias.
// No stock-M5 bit identity, all-input equivalence, native allocation or speed claim.
#ifndef TF_NATIVE_DOWN_M16_BOTHCOOP_NORMALHALF_PROTOTYPE
#define TF_NATIVE_DOWN_M16_BOTHCOOP_NORMALHALF_PROTOTYPE 0
#endif
#ifndef TF_NATIVE_DOWN_M16_BOTHCOOP_NORMALHALF_TRACE_C
#define TF_NATIVE_DOWN_M16_BOTHCOOP_NORMALHALF_TRACE_C 0
#endif
#if TF_NATIVE_DOWN_M16_BOTHCOOP_NORMALHALF_PROTOTYPE
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct tf_down_m16_normalhalf_params {
  uint rows,x_stride,h_stride,y_stride,sb_stride,sum_stride,tiles;
};
[[max_total_threads_per_threadgroup(64)]]
kernel void tf_native_down_m16_bothcoop_normalhalf_prototype(
 const device bfloat* X[[buffer(0)]], const device uchar* W[[buffer(1)]],
 const device uint* SB[[buffer(2)]], const device float* XS[[buffer(3)]],
 const device bfloat* Hidden[[buffer(4)]], device bfloat* RO[[buffer(5)]],
 constant tf_down_m16_normalhalf_params& p[[buffer(6)]],
#if TF_NATIVE_DOWN_M16_BOTHCOOP_NORMALHALF_TRACE_C
 device float* trace_c0[[buffer(7)]],device float* trace_c1[[buffer(8)]],
 device float* trace_sum[[buffer(9)]],device bfloat* trace_projected[[buffer(10)]],
#endif
 device uint* abi[[buffer(11)]],uint sgi[[simdgroup_index_in_threadgroup]],
 uint lane[[thread_index_in_simdgroup]],uint2 tg[[threadgroup_position_in_grid]]) {
 constexpr auto dm=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply);
 constexpr auto da=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<dm,execution_simdgroup> opm;matmul2d<da,execution_simdgroup> opa;
 using XT=tensor<device bfloat,dextents<int32_t,2>,tensor_inline>;
 using WT=tensor<device half,dextents<int32_t,2>,tensor_inline>;
 auto A0=opm.template get_left_input_cooperative_tensor<bfloat,half,float>();
 auto A1=opa.template get_left_input_cooperative_tensor<bfloat,half,float>();
 auto B0=opm.template get_right_input_cooperative_tensor<bfloat,half,float>();
 auto B1=opa.template get_right_input_cooperative_tensor<bfloat,half,float>();
 auto P=opm.template get_destination_cooperative_tensor<decltype(A0),decltype(B0),float>();
 const uint mode=abi[0];
 if(mode==1u){
  const uint tid=sgi*32u+lane,rw=388u,base=4u+tid*rw,cap=uint(P.get_capacity());
  if(tid==0u){abi[0]=1u;abi[1]=1u;abi[2]=64u;abi[3]=rw;}
  abi[base]=cap;abi[base+1u]=sgi;abi[base+2u]=lane;abi[base+3u]=1u;
  for(uint i=0;i<128u;++i){const uint at=base+4u+i*3u;const bool valid=i<cap && P.is_valid_element(ushort(i));
   if(valid){const auto xy=P.get_multidimensional_index(ushort(i));abi[at]=1u;abi[at+1u]=uint(xy[0]);abi[at+2u]=uint(xy[1]);}
   else{abi[at]=0u;abi[at+1u]=0xffffffffu;abi[at+2u]=0xffffffffu;}
  }
  return; // Whole-TG QUERY returns before params, inputs, cooperative loads or output.
 }
 if(mode!=0u)return;
 const size_t r0=size_t(tg.y)*16u;
 if(tg.x>=p.tiles || r0>=p.rows)return; // Whole-TG uniform before reduction barrier.
 const uint M=min(p.rows-uint(r0),16u),n0=tg.x*64u;
 const array<int32_t,2> xstr={1,int32_t(p.x_stride)},wstr={1,64};
 float C0[16],C1[16];
#define TF_D16NH_APPLY(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8) F(9) F(10) F(11) F(12) F(13) F(14) F(15)
#define TF_D16NH_ZERO(S) C0[S]=0.0f;C1[S]=0.0f;
 TF_D16NH_APPLY(TF_D16NH_ZERO)
#undef TF_D16NH_ZERO
 const uint first=sgi*136u,last=first+136u;
 for(uint g=first;g<last;++g){
  const device bfloat* xg=X+r0*p.x_stride+g*64u;
  XT x0((device bfloat*)xg,dextents<int32_t,2>(32,int32_t(M)),xstr);
  XT x1((device bfloat*)(xg+32u),dextents<int32_t,2>(32,int32_t(M)),xstr);
  A0.load(x0);A1.load(x1); // One activation pair shared by both horizontal N32 tiles.
  constexpr size_t GROUP_BYTES=8192u,TILE_BYTES=272u*GROUP_BYTES;
  const size_t wb=size_t(tg.x)*TILE_BYTES+g*GROUP_BYTES;
  WT w00((device half*)(W+wb),dextents<int32_t,2>(32,32),wstr);
  WT w01((device half*)(W+wb+64u),dextents<int32_t,2>(32,32),wstr);
  B0.load(w00);B1.load(w01);opm.run(A0,B0,P);opa.run(A1,B1,P);
#define TF_D16NH_FMA0(S) do {constexpr ushort slot=ushort(S);if(P.is_valid_element(slot)){const auto xy=P.get_multidimensional_index(slot);const uint col=uint(xy[0]),m=uint(xy[1]);if(m<M && col<32u){const vec<bfloat,2> sb=as_type<vec<bfloat,2>>(SB[size_t(g)*p.sb_stride+n0+col]);const float xs=XS[size_t(g)*p.sum_stride+r0+m];C0[S]=fma(float(sb.x),P[slot],fma(float(sb.y),xs,C0[S]));}}}while(false);
  TF_D16NH_APPLY(TF_D16NH_FMA0)
#undef TF_D16NH_FMA0
  WT w10((device half*)(W+wb+4096u),dextents<int32_t,2>(32,32),wstr);
  WT w11((device half*)(W+wb+4096u+64u),dextents<int32_t,2>(32,32),wstr);
  B0.load(w10);B1.load(w11);opm.run(A0,B0,P);opa.run(A1,B1,P);
#define TF_D16NH_FMA1(S) do {constexpr ushort slot=ushort(S);if(P.is_valid_element(slot)){const auto xy=P.get_multidimensional_index(slot);const uint col=uint(xy[0]),m=uint(xy[1]);if(m<M && col<32u){const vec<bfloat,2> sb=as_type<vec<bfloat,2>>(SB[size_t(g)*p.sb_stride+n0+32u+col]);const float xs=XS[size_t(g)*p.sum_stride+r0+m];C1[S]=fma(float(sb.x),P[slot],fma(float(sb.y),xs,C1[S]));}}}while(false);
  TF_D16NH_APPLY(TF_D16NH_FMA1)
#undef TF_D16NH_FMA1
 }
 threadgroup float part[16u*64u];
 // Preserve the baseline's 8192-byte static resource control. Its diagnostic never runs for valid positive rows.
 threadgroup float tgpad[8192u/4u];
 if(int(p.rows)==-12345){tgpad[lane]=C0[0];threadgroup_barrier(mem_flags::mem_threadgroup);C0[0]=tgpad[lane^1u];}
#if TF_NATIVE_DOWN_M16_BOTHCOOP_NORMALHALF_TRACE_C
#define TF_D16NH_TRACE_PART(S,COL,VALUE) do {if(sgi==0u)trace_c0[(r0+m)*p.y_stride+n0+COL]=VALUE;else trace_c1[(r0+m)*p.y_stride+n0+COL]=VALUE;}while(false);
#else
#define TF_D16NH_TRACE_PART(S,COL,VALUE)
#endif
#define TF_D16NH_PART(S) do {constexpr ushort slot=ushort(S);if(P.is_valid_element(slot)){const auto xy=P.get_multidimensional_index(slot);const uint col=uint(xy[0]),m=uint(xy[1]);if(m<M && col<32u){TF_D16NH_TRACE_PART(S,col,C0[S]) TF_D16NH_TRACE_PART(S,32u+col,C1[S]) if(sgi==1u){part[m*64u+col]=C0[S];part[m*64u+32u+col]=C1[S];}}}}while(false);
 TF_D16NH_APPLY(TF_D16NH_PART)
#undef TF_D16NH_PART
#undef TF_D16NH_TRACE_PART
 threadgroup_barrier(mem_flags::mem_threadgroup);
#if TF_NATIVE_DOWN_M16_BOTHCOOP_NORMALHALF_TRACE_C
#define TF_D16NH_TRACE_OUT(COL,SUM,PROJ) trace_sum[(r0+m)*p.y_stride+n0+COL]=SUM;trace_projected[(r0+m)*p.y_stride+n0+COL]=PROJ;
#else
#define TF_D16NH_TRACE_OUT(COL,SUM,PROJ)
#endif
#define TF_D16NH_OUT(S) do {constexpr ushort slot=ushort(S);if(P.is_valid_element(slot)){const auto xy=P.get_multidimensional_index(slot);const uint col=uint(xy[0]),m=uint(xy[1]);if(m<M && col<32u){const float sum0=C0[S]+part[m*64u+col],sum1=C1[S]+part[m*64u+32u+col];const bfloat projected0=bfloat(sum0),projected1=bfloat(sum1);const bfloat h0=Hidden[(r0+m)*p.h_stride+n0+col],h1=Hidden[(r0+m)*p.h_stride+n0+32u+col];TF_D16NH_TRACE_OUT(col,sum0,projected0) TF_D16NH_TRACE_OUT(32u+col,sum1,projected1) RO[(r0+m)*p.y_stride+n0+col]=bfloat(float(h0)+float(projected0));RO[(r0+m)*p.y_stride+n0+32u+col]=bfloat(float(h1)+float(projected1));}}}while(false);
 if(sgi==0u){TF_D16NH_APPLY(TF_D16NH_OUT)}
#undef TF_D16NH_OUT
#undef TF_D16NH_TRACE_OUT
#undef TF_D16NH_APPLY
}
#endif
