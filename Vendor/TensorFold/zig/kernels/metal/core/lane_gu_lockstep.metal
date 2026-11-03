// K64 lockstep G/U scheduler for the guarded decoded indexed GU path. Same two SGs, original split chains and register activation.
// team-m5-08n: each of the two K split chains runs in its own SIMD group, four a threadgroup (two tiles x two splits), so
// twice the SIMD groups stream the weights at once; split 1's sums reach split 0's SIMD group through threadgroup memory
// and add as before (G0 + G), and split 0's SIMD groups run the epilogue. Every chain, word and sum is unchanged.
[[max_total_threads_per_threadgroup(128)]]
kernel void tf_lane_lockstep(const device bfloat* X [[buffer(0)]],const device uint4* W [[buffer(1)]],const device bfloat* SB [[buffer(2)]],const device int* mdims [[buffer(3)]],device bfloat* Y [[buffer(4)]],const device float* XS [[buffer(5)]],device float* HS [[buffer(6)]],
#if defined(TF_GU_COUNT)
 device atomic_uint* GC [[buffer(7)]],
#endif
 uint sgi [[simdgroup_index_in_threadgroup]],uint lanei [[thread_index_in_simdgroup]],uint2 tg [[threadgroup_position_in_grid]]){
 const int lane=int(lanei),ks=int(sgi)&1,split=int(sgi)>>1,M=min(mdims[0],16),stride=mdims[1];
 const int gt=2*int(tg.x)+ks,ut=gt+TF_N/64,gn=gt*32,un=ut*32;
 const int nb=4*(lane&1)+8*((lane>>3)&1),mb=((lane>>1)&3)+4*((lane>>4)&1);
 // team-m5-08j: each turn streams both tensor-op operands through K = 32 halves, so a turn's tensor ops hold half the
 // operand registers the K = 64 op held. The input's K 0-31 half loads into a K = 32 left operand (xl0) and the codes'
 // even words unpack into one K = 32 right operand (cbl) for a K = 32 multiply into P; then the K 32-63 half (xl1) and
 // the odd words, unpacked into the same registers, for a K = 32 multiply-accumulate. A K = 64 right operand's registers
 // are its two K = 32 halves' registers in order (half2 words 0-15, then 16-31), so each code unpacks into the word it
 // filled before, less 16 for the odd words, as the TF_PROMPT_TN two-tile kernel streams them; the pair's fp32 sums equal
 // the K = 64 product's (e-man07's bab46765: no differing bit over 2,097,152 outputs).
 constexpr auto dm=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply);
 constexpr auto da=matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<dm,execution_simdgroup> opm;
 matmul2d<da,execution_simdgroup> opa;
 auto xl0=opm.template get_left_input_cooperative_tensor<bfloat,half,float>();
 auto xl1=opa.template get_left_input_cooperative_tensor<bfloat,half,float>();
 auto cbl=opm.template get_right_input_cooperative_tensor<bfloat,half,float>();
 auto P=opm.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(xl0)>,metal::remove_addrspace_t<decltype(cbl)>,float>();
 thread half2* b2=(thread half2*)&cbl[0];
 const device uint4* gw=W+((size_t)(gt/TF_TB)*80*TF_TB+gt%TF_TB)*64+lane;
 const device uint4* uw=W+((size_t)(ut/TF_TB)*80*TF_TB+ut%TF_TB)*64+lane;
 const device uint* tab=(const device uint*)SB;
#if defined(TF_SBI12)
 const device packed_uint3* gi=(const device packed_uint3*)(tab+TF_SBI)+gn/8+nb/4;
 const device packed_uint3* ui=(const device packed_uint3*)(tab+TF_SBI)+un/8+nb/4;
#else
 const device uint2* gi=(const device uint2*)(tab+TF_SBI)+(gn+nb)/4;
 const device uint2* ui=(const device uint2*)(tab+TF_SBI)+(un+nb)/4;
#endif
 float G[16],U[16];
 {
  const int first=split*40,last=first+40;
  for(int i=0;i<16;i++){G[i]=0.0f;U[i]=0.0f;}
  const device uint4* gp=gw+(size_t)first*TF_TB*64;
  const device uint4* up=uw+(size_t)first*TF_TB*64;
  uint4 nw0=gp[0],nw1=gp[32];
#if defined(TF_SBI12)
  uint3 ni=gi[(size_t)first*(TF_N/8)];
#else
  uint2 ni0=gi[(size_t)first*(TF_N/4)],ni1=gi[(size_t)first*(TF_N/4)+4];
#endif
  for(int g=first;g<last;g++){
   const float x0=XS[g*stride+mb];
   const array<int32_t,2> xstr={1,TF_K};
   #pragma clang loop unroll(full)
   for(int turn=0;turn<2;turn++){
#if defined(TF_SBI12)
    const uint4 q0=tf_look12a(tab,ni),q1=tf_look12b(tab,ni);
#else
    const uint4 q0=tf_look(tab,ni0),q1=tf_look(tab,ni1);
#endif
    // the turn's codes: the prefetch below reloads nw0, nw1 before the odd words unpack
    const uint4 cw0=nw0,cw1=nw1;
    #pragma clang loop unroll(full)
    for(int w=0;w<8;w+=2){
     const uint word=w<4?cw0[w]:cw1[w-4];const int e=2*(w>>1);
     b2[e]=as_type<half2>(word&0x000F000Fu);b2[e+1]=as_type<half2>((word>>4)&0x000F000Fu);
     b2[e+8]=as_type<half2>((word>>8)&0x000F000Fu);b2[e+9]=as_type<half2>((word>>12)&0x000F000Fu);
    }
    if(turn==0){
     nw0=up[0];nw1=up[32];
#if defined(TF_SBI12)
     ni=ui[(size_t)g*(TF_N/8)];
#else
     ni0=ui[(size_t)g*(TF_N/4)];ni1=ui[(size_t)g*(TF_N/4)+4];
#endif
    }else{
     const int advance=g+1<last?TF_TB*64:0;gp+=advance;up+=advance;nw0=gp[0];nw1=gp[32];
     const int next=min(g+1,last-1);
#if defined(TF_SBI12)
     ni=gi[(size_t)next*(TF_N/8)];
#else
     ni0=gi[(size_t)next*(TF_N/4)];ni1=gi[(size_t)next*(TF_N/4)+4];
#endif
    }
    xl0.load(tensor<device bfloat,dextents<int32_t,2>,tensor_inline>((device bfloat*)X+(size_t)g*64,dextents<int32_t,2>(32,M),xstr));
    opm.run(xl0,cbl,P);
    #pragma clang loop unroll(full)
    for(int w=1;w<8;w+=2){
     const uint word=w<4?cw0[w]:cw1[w-4];const int e=2*(w>>1);
     b2[e]=as_type<half2>(word&0x000F000Fu);b2[e+1]=as_type<half2>((word>>4)&0x000F000Fu);
     b2[e+8]=as_type<half2>((word>>8)&0x000F000Fu);b2[e+9]=as_type<half2>((word>>12)&0x000F000Fu);
    }
    xl1.load(tensor<device bfloat,dextents<int32_t,2>,tensor_inline>((device bfloat*)X+(size_t)g*64+32,dextents<int32_t,2>(32,M),xstr));
    opa.run(xl1,cbl,P);
    for(int j=0;j<4;j++){
     const vec<bfloat,2> v0=as_type<vec<bfloat,2>>(q0[j]),v1=as_type<vec<bfloat,2>>(q1[j]);
     const float s0=float(v0.x),b0=float(v0.y),s1=float(v1.x),b1=float(v1.y);
     if(turn==0){G[j]=fma(s0,P[j],fma(b0,x0,G[j]));G[8+j]=fma(s1,P[8+j],fma(b1,x0,G[8+j]));}
     else{U[j]=fma(s0,P[j],fma(b0,x0,U[j]));U[8+j]=fma(s1,P[8+j],fma(b1,x0,U[8+j]));}
    }
   }
  }
 }
 threadgroup float gu1[2*2*16*32];
 if(split==1)for(int i=0;i<16;i++)if(TF_LIVE(i)){gu1[((ks*2)*16+i)*32+lane]=G[i];gu1[((ks*2+1)*16+i)*32+lane]=U[i];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(split==0)for(int i=0;i<16;i++)if(TF_LIVE(i)){G[i]=G[i]+gu1[((ks*2)*16+i)*32+lane];U[i]=U[i]+gu1[((ks*2+1)*16+i)*32+lane];}

 // Literal original two-SG register-local SwiGLU/HS epilogue; only C is named U.
 threadgroup bfloat hb[8*64];
 if(split==0&&mb<M){const int c0=ks*32+nb;device bfloat* yr=Y+(size_t)mb*(TF_N/2)+int(tg.x)*64+c0;threadgroup bfloat* hr=hb+mb*64+c0;
  for(int i=0;i<16;i++){if(!TF_LIVE(i))continue;const int c=16*(i>>3)+(i&3);const float gf=float(bfloat(G[i]));const bfloat h=bfloat(gf/(1.0f+metal::exp(-gf))*float(bfloat(U[i])));yr[c]=h;hr[c]=h;}
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 const int t=ks*32+lane;
 if(split==0)for(int m=t;m<stride;m+=64){float acc=0.0f;if(m<M)for(int i=0;i<64;i++)acc+=float(hb[m*64+i]);HS[int(tg.x)*stride+m]=acc;}
#if defined(TF_GU_COUNT)
 // the MLP down that reads Y and HS shares this launch's barrier interval: each threadgroup counts itself on GC[0] once its
 // stores are past a device-scope fence (tf_lane_reg's TF_WAIT)
 threadgroup_barrier(mem_flags::mem_device);
 if(split==0&&t==0){atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);atomic_fetch_add_explicit(GC,1u,memory_order_relaxed);}
#endif
}
