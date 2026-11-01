#include <metal_stdlib>
using namespace metal;

constant uint inputWidth = 17408;
constant uint outputWidth = 5120;
constant uint groups = inputWidth / 64;
constant uint keep = 12;
constant uint slots = 16;

kernel void selectChannels(
    device const bfloat *input [[buffer(0)]],
    device uchar *indices [[buffer(1)]],
    device float *values [[buffer(2)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint channel [[thread_index_in_threadgroup]]
) {
    threadgroup uint keys[64];
    uint group = tile.y * groups + tile.x;
    ushort bits = as_type<ushort>(input[64 * group + channel]);
    // Higher BF16 magnitude wins. Equal magnitudes retain the earlier channel.
    keys[channel] = (uint(bits & 0x7fff) << 6) | (63 - channel);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= 64; width *= 2) {
        for (uint step = width / 2; step; step /= 2) {
            uint a = keys[channel];
            uint b = keys[channel ^ step];
            bool high = ((channel & width) == 0) == ((channel & step) == 0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            keys[channel] = high ? max(a, b) : min(a, b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (channel < keep) {
        uint selected = 63 - (keys[channel] & 63);
        indices[group * slots + channel] = uchar(selected);
        values[group * slots + channel] = float(input[group * 64 + selected]);
    }
}

kernel void selectedDown(
    device const uchar *codes [[buffer(0)]],
    device const uint2 *metadata [[buffer(1)]],
    device const uchar *indices [[buffer(2)]],
    device const float *values [[buffer(3)]],
    device bfloat *out [[buffer(4)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint subgroup [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float2 partial[8][32];
    float2 acc = 0;
    for (uint group = subgroup; group < groups; group += 8) {
        uint tileGroup = tile.x * groups + group;
        uint2 pair = metadata[tileGroup * 32 + lane];
        float2 scale = as_type<float2>((pair & 0xffffu) << 16);
        float2 bias = as_type<float2>(pair & 0xffff0000u);
        uint base = (tile.y * groups + group) * slots;
        float2 codeSum = 0;
        float inputSum = 0;
        for (uint i = 0; i < keep; ++i) {
            float value = values[base + i];
            uint channel = indices[base + i];
            uint code = codes[(tileGroup * 64 + channel) * 32 + lane];
            codeSum = fma(value, float2(code & 15, code >> 4), codeSum);
            inputSum += value;
        }
        acc += fma(scale, codeSum, bias * inputSum);
    }
    partial[subgroup][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (subgroup == 0) {
        for (uint i = 1; i < 8; ++i) acc += partial[i][lane];
        uint first = tile.y * outputWidth + tile.x * 64 + lane * 2;
        out[first] = bfloat(acc.x);
        out[first + 1] = bfloat(acc.y);
    }
}
