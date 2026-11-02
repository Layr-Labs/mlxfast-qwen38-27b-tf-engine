#include <metal_stdlib>
using namespace metal;

constant uint groups=80, keep=8, slots=16, splits=16;
constant uint outputWidth=98816;
kernel void selectChannels(
    device const bfloat *x [[buffer(0)]], device const float *rms [[buffer(1)]],
    device uchar *indices [[buffer(2)]], device float *values [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]], uint channel [[thread_index_in_threadgroup]]) {
    uint group=tile.x, col=64*group+channel;
    float lower=float(x[col]),upper=float(x[col+32]);
    uint left=(as_type<uint>(abs(lower*rms[col])) & ~63u) | (63-channel);
    uint right=(as_type<uint>(abs(upper*rms[col+32])) & ~63u) | (31-channel);
    // Sort the two halves in opposite directions to form a bitonic sequence.
    for (uint width = 2; width <= 32; width *= 2) {
        for (uint step = width / 2; step; step /= 2) {
            bool descending = ((channel & width) == 0) == ((channel & step) == 0);
            uint other = simd_shuffle_xor(left, ushort(step));
            left = descending ? max(left, other) : min(left, other);
            other = simd_shuffle_xor(right, ushort(step));
            bool rightDescending = width == 32 ? !descending : descending;
            right = rightDescending ? max(right, other) : min(right, other);
        }
    }
    // Only the upper half can contain one of the twelve retained channels.
    left = max(left, right);
    for (uint step = 16; step; step /= 2) {
        uint other = simd_shuffle_xor(left, ushort(step));
        left = (channel & step) == 0 ? max(left, other) : min(left, other);
    }
    uint selected = 63 - (left & 63);
    ushort sourceLane = ushort(selected & 31);
    float a = simd_shuffle(lower, sourceLane);
    float b = simd_shuffle(upper, sourceLane);
    float value = selected < 32 ? a : b;
    uint at = group * slots;
    if (channel < keep) {
        indices[at + channel] = uchar(selected);
        values[at + channel] = value;
    }
    float sum = 0;
    for (ushort i = 0; i < keep; ++i) sum += simd_broadcast(value, i);
    if (channel == 0) values[at + keep] = sum;
}

kernel void sectorDown(device const uint *w [[buffer(0)]], device const uint4 *pairs [[buffer(1)]],
                       device const uchar *ids [[buffer(2)]], device const float *vals [[buffer(3)]],
                       device float4 *out [[buffer(4)]], uint3 pos [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint tile=pos.x, row=pos.y, split=pos.z*4+sg;
    float4 a=0,b=0;
    for(uint g=split;g<groups;g+=splits){
        uint tileGroup=tile*groups+g;
        uint4 lo=pairs[(tileGroup*32+lane)*2], hi=pairs[(tileGroup*32+lane)*2+1];
        float4 dl=0,dh=0;
        uint at=(row*groups+g)*slots;
        for(uint i=0;i<keep;i++){
            uint packed=w[(tileGroup*64+ids[at+i])*32+lane];
            float v=vals[at+i];
            dl=fma(float4((uint4(packed)>>uint4(0,4,8,12))&15u),v,dl);
            dh=fma(float4((uint4(packed)>>uint4(16,20,24,28))&15u),v,dh);
        }
        float sum=vals[at+keep];
        a+=fma(as_type<float4>((lo&65535u)<<16),dl,as_type<float4>(lo&0xffff0000u)*sum);
        b+=fma(as_type<float4>((hi&65535u)<<16),dh,as_type<float4>(hi&0xffff0000u)*sum);
    }
    uint first=(row*splits+split)*(outputWidth/4)+(tile*32+lane)*2;
    out[first]=a; out[first+1]=b;
}

// Four strongest tokens per permuted vocabulary bucket. Ties go to smaller original token IDs.
kernel void pickBucket(device const float4* parts [[buffer(0)]],device const uint* map [[buffer(1)]],
                       device uint* ids [[buffer(2)]],
                       uint3 tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    uint t=tid.x/32;if(t>=outputWidth/256)return;
    uint start=t*256+8*lane;
    float4 lo=0,hi=0;
    for(uint s=0;s<splits;s++){
        uint first=s*(outputWidth/4)+start/4;
        lo+=parts[first];hi+=parts[first+1];
    }
    float v[8];uint token[8];
    for(uint j=0;j<8;j++){
        v[j]=start+j<98592?(j<4?lo[j]:hi[j-4]):-INFINITY;
        token[j]=start+j<98592?map[start+j]:~0u;
    }
    for(uint pick=0;pick<4;pick++){
        float best=-INFINITY;uint id=~0u;
        for(uint j=0;j<8;j++)if(v[j]>best || (v[j]==best && token[j]<id)){best=v[j];id=token[j];}
        float local=best;best=simd_max(best);id=simd_min(local==best?id:~0u);
        if(!lane){ids[t*4+pick]=id;}
        for(uint j=0;j<8;j++)if(token[j]==id){v[j]=-INFINITY;token[j]=~0u;}
    }
}
#include <metal_stdlib>
using namespace metal;

// One SIMD group scores one arbitrary proposal using original checkpoint rows.
kernel void proposal_gather(device const uint *codes [[buffer(0)]],
                            device const bfloat *scales [[buffer(1)]],
                            device const bfloat *biases [[buffer(2)]],
                            device const bfloat *input [[buffer(3)]],
                            device const uint *ids [[buffer(4)]],
                            device bfloat *scores [[buffer(5)]],
                            constant uint &count [[buffer(6)]],
                            uint3 tid [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint i = tid.x >> 5;
    if (i >= count) return;
    uint id = ids[i];
    if (id >= 248320 || (id >= 98304 && id < 248032)) {
        if (!lane) scores[i] = bfloat(-INFINITY);
        return;
    }
    uint token = id < 98304 ? id : 98304 + (id - 248032);
    float partial = 0;
    for (uint w = lane; w < 640; w += 32) {
        uint pack = codes[token * 640 + w];
        uint meta = token * 80 + w / 8;
        float scale = float(scales[meta]), bias = float(biases[meta]);
        for (uint j = 0; j < 8; ++j) {
            float weight = fma(scale, float((pack >> (4 * j)) & 15u), bias);
            partial = fma(weight, float(input[8 * w + j]), partial);
        }
    }
    float score = simd_sum(partial);
    if (!lane) scores[i] = bfloat(score);
}
