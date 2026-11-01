#include <metal_stdlib>
using namespace metal;
struct Linear { uint rows,n,k,x_stride,w_stride,y_stride; };
kernel void tf_bf16_linear(device const bfloat* X [[buffer(0)]],device const bfloat* W [[buffer(1)]],constant Linear& a [[buffer(2)]],device bfloat* Y [[buffer(3)]],uint2 block [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint col=block.x*8+sg,row=block.y;
  if(col>=a.n)return;
  float value=0.0f;
  for(uint k=lane;k<a.k;k+=32)value=fma(float(X[size_t(row)*a.x_stride+k]),float(W[size_t(col)*a.w_stride+k]),value);
  value=simd_sum(value);
  if(lane==0)Y[size_t(row)*a.y_stride+col]=bfloat(value);
}
struct Norm { uint rows,heads,dim,x_stride,y_stride,round_normalized;float eps; };
kernel void tf_draft_rms(device const bfloat* X [[buffer(0)]],device const bfloat* W [[buffer(1)]],constant Norm& a [[buffer(2)]],device bfloat* Y [[buffer(3)]],uint2 block [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float sums[32],inverse[1];
  float value=0.0f;
  const size_t base=size_t(block.y)*a.x_stride+block.x*a.dim;
  for(uint first=4*t;first<a.dim;first+=4*threads.x)for(uint j=0;j<4&&first+j<a.dim;j++){float x=float(X[base+first+j]);value+=x*x;}
  value=simd_sum(value);
  if(sg==0)sums[lane]=0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0)sums[sg]=value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){float total=simd_sum(sums[lane]);if(lane==0)inverse[0]=precise::rsqrt(total/float(a.dim)+a.eps);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint first=4*t;first<a.dim;first+=4*threads.x)for(uint j=0;j<4&&first+j<a.dim;j++){
    const uint col=first+j;
    float normalized=float(X[base+col])*inverse[0];
    normalized=float(bfloat(normalized));
    Y[size_t(block.y)*a.y_stride+block.x*a.dim+col]=bfloat(normalized*float(W[col]));
  }
}
struct Conv { uint rows,width,group,block,branch,residual,x_stride,dynamic_stride,y_stride,rounding; };
kernel void tf_dynamic_conv2(device const bfloat* X [[buffer(0)]],device const bfloat* D [[buffer(1)]],device const bfloat* B [[buffer(2)]],device const bfloat* R [[buffer(3)]],constant Conv& a [[buffer(4)]],device bfloat* Y [[buffer(5)]],uint2 index [[thread_position_in_grid]]) {
  if(index.x>=a.width||index.y>=a.rows)return;
  const uint groups=a.width/a.group,g=index.x/a.group;
  float total=0.0f;
  for(uint lag=0;lag<2;lag++) {
    const float x=lag&&index.y%a.block==0?0.0f:float(X[size_t(index.y-lag)*a.x_stride+index.x]);
    const float base=float(B[(a.branch*2+lag)*a.width+index.x]);
    const float dynamic=float(D[size_t(index.y)*a.dynamic_stride+(a.branch*2+lag)*groups+g]);
    total=float(bfloat(total+float(bfloat(base*x))));total=float(bfloat(total+float(bfloat(dynamic*x))));
  }
  const float conv=float(bfloat(total));
  Y[size_t(index.y)*a.y_stride+index.x]=bfloat(a.residual?float(R[size_t(index.y)*a.x_stride+index.x])+conv:conv);
}
struct Rope { uint rows,heads,dim,x_stride,y_stride;float theta; };
kernel void tf_full_rope(device const bfloat* X [[buffer(0)]],device const ulong* positions [[buffer(1)]],constant Rope& a [[buffer(2)]],device bfloat* Y [[buffer(3)]],uint3 index [[thread_position_in_grid]]) {
  if(index.x>=a.dim/2||index.y>=a.heads||index.z>=a.rows)return;
  const uint col=index.x,pairs=a.dim/2;
  const float angle=float(positions[index.z])*pow(a.theta,-float(col)/float(pairs));
  const float cs=cos(angle),sn=sin(angle);
  const size_t input=size_t(index.z)*a.x_stride+index.y*a.dim+col,output=size_t(index.z)*a.y_stride+index.y*a.dim+col;
  const float left=float(X[input]),right=float(X[input+pairs]);
  Y[output]=bfloat(left*cs-right*sn);Y[output+pairs]=bfloat(right*cs+left*sn);
}
struct Activation { uint rows,width,gate_stride,up_stride,y_stride,rounding; };
kernel void tf_draft_swiglu(device const bfloat* G [[buffer(0)]],device const bfloat* U [[buffer(1)]],constant Activation& a [[buffer(2)]],device bfloat* Y [[buffer(3)]],uint2 index [[thread_position_in_grid]]) {
  if(index.x>=a.width||index.y>=a.rows)return;
  const float gate=float(G[size_t(index.y)*a.gate_stride+index.x]),up=float(U[size_t(index.y)*a.up_stride+index.x]);
  {
    const bfloat g=bfloat(gate),u=bfloat(up);
    const bfloat exponential=bfloat(exp(abs(gate)));
    const bfloat denominator=bfloat(1.0f+float(exponential));
    const bfloat inverse=bfloat(1.0f/float(denominator));
    const bfloat sigmoid=gate<0.0f?inverse:bfloat(1.0f-float(inverse));
    const bfloat activation=bfloat(float(g)*float(sigmoid));
    Y[size_t(index.y)*a.y_stride+index.x]=bfloat(float(activation)*float(u));
  }
}

#define TF_XS_GROUP 64
#define TF_XS_MAX_WIDTH 5120
kernel void tf_draft_rms_xs(device const bfloat* X [[buffer(0)]],device const bfloat* W [[buffer(1)]],constant Norm& a [[buffer(2)]],device bfloat* Y [[buffer(3)]],device float* XS [[buffer(4)]],constant uint& xs_stride [[buffer(5)]],uint2 block [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float sums[32],inverse[1];
  threadgroup bfloat out[TF_XS_MAX_WIDTH];
  float value=0.0f;
  const size_t base=size_t(block.y)*a.x_stride+block.x*a.dim;
  for(uint first=4*t;first<a.dim;first+=4*threads.x)for(uint j=0;j<4&&first+j<a.dim;j++){float x=float(X[base+first+j]);value+=x*x;}
  value=simd_sum(value);
  if(sg==0)sums[lane]=0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0)sums[sg]=value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){float total=simd_sum(sums[lane]);if(lane==0)inverse[0]=precise::rsqrt(total/float(a.dim)+a.eps);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint first=4*t;first<a.dim;first+=4*threads.x)for(uint j=0;j<4&&first+j<a.dim;j++){
    const uint col=first+j;
    float normalized=float(X[base+col])*inverse[0];
    normalized=float(bfloat(normalized));
    const bfloat y=bfloat(normalized*float(W[col]));
    Y[size_t(block.y)*a.y_stride+block.x*a.dim+col]=y;
    out[col]=y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint groups=a.dim/TF_XS_GROUP;
  for(uint g=t;g<groups;g+=threads.x){
    float acc=0.0f;
    for(uint i=0;i<TF_XS_GROUP;i++)acc+=float(out[g*TF_XS_GROUP+i]);
    XS[size_t(g)*xs_stride+block.y]=acc;
    if(block.y==0)for(uint r=a.rows;r<xs_stride;r++)XS[size_t(g)*xs_stride+r]=0.0f;
  }
}
kernel void tf_dynamic_conv2_xs(device const bfloat* X [[buffer(0)]],device const bfloat* D [[buffer(1)]],device const bfloat* B [[buffer(2)]],device const bfloat* R [[buffer(3)]],constant Conv& a [[buffer(4)]],device bfloat* Y [[buffer(5)]],device float* XS [[buffer(6)]],constant uint& xs_stride [[buffer(7)]],uint2 index [[thread_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat out[TF_XS_GROUP];
  const uint groups=a.width/a.group,g=index.x/a.group;
  float total=0.0f;
  for(uint lag=0;lag<2;lag++) {
    const float x=lag&&index.y%a.block==0?0.0f:float(X[size_t(index.y-lag)*a.x_stride+index.x]);
    const float base=float(B[(a.branch*2+lag)*a.width+index.x]);
    const float dynamic=float(D[size_t(index.y)*a.dynamic_stride+(a.branch*2+lag)*groups+g]);
    total=float(bfloat(total+float(bfloat(base*x))));total=float(bfloat(total+float(bfloat(dynamic*x))));
  }
  const float conv=float(bfloat(total));
  const bfloat y=bfloat(a.residual?float(R[size_t(index.y)*a.x_stride+index.x])+conv:conv);
  Y[size_t(index.y)*a.y_stride+index.x]=y;
  out[t]=y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint group=index.x/TF_XS_GROUP;
  if(t==0){
    float acc=0.0f;
    for(uint i=0;i<TF_XS_GROUP;i++)acc+=float(out[i]);
    XS[size_t(group)*xs_stride+index.y]=acc;
  }
  if(index.y==0&&t>=1&&t<=xs_stride-a.rows)XS[size_t(group)*xs_stride+a.rows+t-1]=0.0f;
}
kernel void tf_draft_swiglu_xs(device const bfloat* G [[buffer(0)]],device const bfloat* U [[buffer(1)]],constant Activation& a [[buffer(2)]],device bfloat* Y [[buffer(3)]],device float* XS [[buffer(4)]],constant uint& xs_stride [[buffer(5)]],uint2 index [[thread_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat out[TF_XS_GROUP];
  const float gate=float(G[size_t(index.y)*a.gate_stride+index.x]),up=float(U[size_t(index.y)*a.up_stride+index.x]);
  bfloat y;
  {
    const bfloat g=bfloat(gate),u=bfloat(up);
    const bfloat exponential=bfloat(exp(abs(gate)));
    const bfloat denominator=bfloat(1.0f+float(exponential));
    const bfloat inverse=bfloat(1.0f/float(denominator));
    const bfloat sigmoid=gate<0.0f?inverse:bfloat(1.0f-float(inverse));
    const bfloat activation=bfloat(float(g)*float(sigmoid));
    y=bfloat(float(activation)*float(u));
  }
  Y[size_t(index.y)*a.y_stride+index.x]=y;
  out[t]=y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint group=index.x/TF_XS_GROUP;
  if(t==0){
    float acc=0.0f;
    for(uint i=0;i<TF_XS_GROUP;i++)acc+=float(out[i]);
    XS[size_t(group)*xs_stride+index.y]=acc;
  }
  if(index.y==0&&t>=1&&t<=xs_stride-a.rows)XS[size_t(group)*xs_stride+a.rows+t-1]=0.0f;
}
kernel void tf_dynamic_conv2_rms_xs(device const bfloat* X [[buffer(0)]],device const bfloat* D [[buffer(1)]],device const bfloat* B [[buffer(2)]],device const bfloat* R [[buffer(3)]],constant Conv& c [[buffer(4)]],device bfloat* C [[buffer(5)]],device const bfloat* W [[buffer(6)]],constant Norm& a [[buffer(7)]],device bfloat* Y [[buffer(8)]],device float* XS [[buffer(9)]],constant uint& xs_stride [[buffer(10)]],uint2 block [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float sums[32],inverse[1];
  threadgroup bfloat out[TF_XS_MAX_WIDTH];
  const uint row=block.y,groups=c.width/c.group;
  float value=0.0f;
  for(uint first=4*t;first<a.dim;first+=4*threads.x)for(uint j=0;j<4&&first+j<a.dim;j++){
    const uint col=first+j,g=col/c.group;
    float total=0.0f;
    for(uint lag=0;lag<2;lag++) {
      const float x=lag&&row%c.block==0?0.0f:float(X[size_t(row-lag)*c.x_stride+col]);
      const float base=float(B[(c.branch*2+lag)*c.width+col]);
      const float dynamic=float(D[size_t(row)*c.dynamic_stride+(c.branch*2+lag)*groups+g]);
      total=float(bfloat(total+float(bfloat(base*x))));total=float(bfloat(total+float(bfloat(dynamic*x))));
    }
    const float conv=float(bfloat(total));
    const bfloat y=bfloat(float(R[size_t(row)*c.x_stride+col])+conv);
    C[size_t(row)*c.y_stride+col]=y;
    out[col]=y;
    const float v=float(y);value+=v*v;
  }
  value=simd_sum(value);
  if(sg==0)sums[lane]=0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0)sums[sg]=value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){float total=simd_sum(sums[lane]);if(lane==0)inverse[0]=precise::rsqrt(total/float(a.dim)+a.eps);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint first=4*t;first<a.dim;first+=4*threads.x)for(uint j=0;j<4&&first+j<a.dim;j++){
    const uint col=first+j;
    float normalized=float(out[col])*inverse[0];
    normalized=float(bfloat(normalized));
    const bfloat y=bfloat(normalized*float(W[col]));
    Y[size_t(row)*a.y_stride+col]=y;
    out[col]=y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint xs_groups=a.dim/TF_XS_GROUP;
  for(uint g=t;g<xs_groups;g+=threads.x){
    float acc=0.0f;
    for(uint i=0;i<TF_XS_GROUP;i++)acc+=float(out[g*TF_XS_GROUP+i]);
    XS[size_t(g)*xs_stride+row]=acc;
    if(row==0)for(uint r=a.rows;r<xs_stride;r++)XS[size_t(g)*xs_stride+r]=0.0f;
  }
}
#define TF_ROPE_MAX_DIM 256
kernel void tf_draft_rms_rope(device const bfloat* X [[buffer(0)]],device const bfloat* W [[buffer(1)]],constant Norm& a [[buffer(2)]],device bfloat* N [[buffer(3)]],device const ulong* positions [[buffer(4)]],constant Rope& r [[buffer(5)]],device bfloat* Y [[buffer(6)]],uint2 block [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float sums[32],inverse[1];
  threadgroup bfloat normed[TF_ROPE_MAX_DIM];
  float value=0.0f;
  const size_t base=size_t(block.y)*a.x_stride+block.x*a.dim;
  for(uint first=4*t;first<a.dim;first+=4*threads.x)for(uint j=0;j<4&&first+j<a.dim;j++){float x=float(X[base+first+j]);value+=x*x;}
  value=simd_sum(value);
  if(sg==0)sums[lane]=0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0)sums[sg]=value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){float total=simd_sum(sums[lane]);if(lane==0)inverse[0]=precise::rsqrt(total/float(a.dim)+a.eps);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint first=4*t;first<a.dim;first+=4*threads.x)for(uint j=0;j<4&&first+j<a.dim;j++){
    const uint col=first+j;
    float normalized=float(X[base+col])*inverse[0];
    normalized=float(bfloat(normalized));
    const bfloat y=bfloat(normalized*float(W[col]));
    N[size_t(block.y)*a.y_stride+block.x*a.dim+col]=y;
    normed[col]=y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint pairs=r.dim/2;
  for(uint col=t;col<pairs;col+=threads.x){
    const float angle=float(positions[block.y])*pow(r.theta,-float(col)/float(pairs));
    const float cs=cos(angle),sn=sin(angle);
    const size_t output=size_t(block.y)*r.y_stride+block.x*r.dim+col;
    const float left=float(normed[col]),right=float(normed[col+pairs]);
    Y[output]=bfloat(left*cs-right*sn);Y[output+pairs]=bfloat(right*cs+left*sn);
  }
}
