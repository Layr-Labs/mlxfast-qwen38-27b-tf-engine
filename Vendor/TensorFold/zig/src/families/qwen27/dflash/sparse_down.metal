#include <metal_stdlib>
using namespace metal;

constant uint inputWidth = 17408;
constant uint outputWidth = 5120;
constant uint groups = inputWidth / 64;
constant uint keep = 12;
constant uint slots = 16;
constant uint splits = 32;

inline float activation(bfloat g, bfloat u) {
    float gate = float(g);
    bfloat exponential = bfloat(exp(abs(gate)));
    bfloat inverse = bfloat(1.0f / float(bfloat(1.0f + float(exponential))));
    bfloat sigmoid = gate < 0.0f ? inverse : bfloat(1.0f - float(inverse));
    return float(bfloat(float(bfloat(gate * float(sigmoid))) * float(u)));
}

kernel void selectChannels(
    device const bfloat *gate [[buffer(0)]],
    device const bfloat *up [[buffer(1)]],
    device uchar *indices [[buffer(2)]],
    device float *values [[buffer(3)]],
    constant uint2 &strides [[buffer(4)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint channel [[thread_index_in_threadgroup]]
) {
    uint group = tile.y * groups + tile.x;
    uint col = 64 * tile.x + channel;
    uint gi = tile.y * strides.x + col, ui = tile.y * strides.y + col;
    float lower = activation(gate[gi], up[ui]);
    float upper = activation(gate[gi+32], up[ui+32]);
    uint left = (uint(as_type<ushort>(bfloat(lower)) & 0x7fff) << 6) | (63 - channel);
    uint right = (uint(as_type<ushort>(bfloat(upper)) & 0x7fff) << 6) | (31 - channel);
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
kernel void finishSector(device const float4 *parts [[buffer(0)]], device bfloat4 *out [[buffer(1)]],
                         uint2 at [[thread_position_in_grid]]){
    float4 value=0;
    for(uint s=0;s<splits;s++)value+=parts[(at.y*splits+s)*(outputWidth/4)+at.x];
    out[at.y*(outputWidth/4)+at.x]=bfloat4(value);
}
