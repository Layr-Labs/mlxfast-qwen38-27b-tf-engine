#include <metal_stdlib>
using namespace metal;

constant uint inputWidth = 17408;
constant uint outputWidth = 5120;
constant uint groups = inputWidth / 64;
constant uint keep = 12;
constant uint slots = 16;
constant uint splits = 4;

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

kernel void selectedDown(
    device const uchar *codes [[buffer(0)]],
    device const uint2 *metadata [[buffer(1)]],
    device const uchar *indices [[buffer(2)]],
    device const float *values [[buffer(3)]],
    device float *out [[buffer(4)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint subgroup [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float2 partial[8][32];
    float2 acc = 0;
    for (uint group = tile.z * 8 + subgroup; group < groups; group += 8 * splits) {
        uint tileGroup = tile.x * groups + group;
        uint2 pair = metadata[tileGroup * 32 + lane];
        float2 scale = as_type<float2>((pair & 0xffffu) << 16);
        float2 bias = as_type<float2>(pair & 0xffff0000u);
        uint base = (tile.y * groups + group) * slots;
        float2 codeSum = 0;
        float inputSum = values[base + keep];
        for (uint i = 0; i < keep; ++i) {
            float value = values[base + i];
            uint channel = indices[base + i];
            uint code = codes[(tileGroup * 64 + channel) * 32 + lane];
            codeSum = fma(value, float2(code & 15, code >> 4), codeSum);
        }
        acc += fma(scale, codeSum, bias * inputSum);
    }
    partial[subgroup][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (subgroup == 0) {
        for (uint i = 1; i < 8; ++i) acc += partial[i][lane];
        uint first = (tile.y * splits + tile.z) * outputWidth + tile.x * 64 + lane * 2;
        out[first] = acc.x;
        out[first + 1] = acc.y;
    }
}

kernel void finishDown(device const float *parts [[buffer(0)]], device bfloat *output [[buffer(1)]],
                       uint2 pos [[thread_position_in_grid]]) {
    float sum = 0;
    for (uint i = 0; i < splits; ++i) sum += parts[(pos.y * splits + i) * outputWidth + pos.x];
    output[pos.y * outputWidth + pos.x] = bfloat(sum);
}
