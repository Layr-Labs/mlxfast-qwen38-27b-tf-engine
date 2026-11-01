// DeltaNet: bf16 conv and gates, fp32 state, fixed reductions; steps and accepted chains share one delta update.
#include <metal_stdlib>
using namespace metal;

struct TfDeltaParams {
  uint rows, nk, nv, dk, dv, taps, slots, weight_flags;
  float eps;
  uint zba_stride;  // 0: z, a and b in their own rows; else one fused row of this many values (z, b, a)
  uint advance;     // 1: a shared round's one-row windows (always kept) step their state in place; the replay skips them
};
struct TfDeltaSegment { uint first, rows, state_slot, next_slot; };
struct TfDeltaKeep { uint first, rows, state_slot, next_slot; };

inline float tf_delta_weight(device const void* data, uint at, uint flags, uint bit) {
  if ((flags & (1u << bit)) != 0) return ((device const float*)data)[at];
  return float(((device const bfloat*)data)[at]);
}

// Every position uses the same per-lane decay, key-dot and delta update before a SIMD reduction.
inline void tf_delta_delta_update(thread float* state, device const bfloat* key,
                              float forget, float incoming, float mix, uint lane, uint count) {
  float remembered = 0.0f;
  for (uint part = 0; part < count; ++part) {
    const uint channel = lane * count + part;
    state[part] = state[part] * forget;
    remembered += state[part] * float(key[channel]);
  }
  remembered = simd_sum(remembered);
  const float correction = (incoming - remembered) * mix;
  for (uint part = 0; part < count; ++part) state[part] = state[part] + float(key[lane * count + part]) * correction;
}

inline float tf_delta_delta_read(thread const float* state, device const bfloat* query, uint lane, uint count) {
  float result = 0.0f;
  for (uint part = 0; part < count; ++part) result += state[part] * float(query[lane * count + part]);
  return simd_sum(result);
}
// Convolution, activation and head normalization retain every declared BF16 rounding point.

// C = 4: constant trip counts (arrays in registers); STEP (one row a slot, in place): next conv window into history.
template <uint C, bool STEP>
inline void tf_delta_gdn_pre_body(device const bfloat* projection,
    device const bfloat* a,
    device const bfloat* b,
    device const bfloat* history,
    device const void* weights,
    device const void* decay_weight,
    device const void* time_bias,
    device const uint* windows,
    device const uint* row_slot,
    constant TfDeltaParams& p,
    device bfloat* query,
    device bfloat* key,
    device bfloat* value,
    device float* decay,
    device bfloat* mixing,
    device bfloat* tails,
    uint3 group,
    uint lane) {
  const uint row = group.z, head = group.y;
  device bfloat* next_history = (device bfloat*)history; // STEP: this thread's own channels, read before written
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv;
  const uint keep = p.taps - 1, elements = C ? C : p.dk / 32;
  const uint channel0 = head < 2 * p.nk ? head * p.dk : 2 * p.nk * p.dk + (head - 2 * p.nk) * p.dv;
  float activated[C ? C : 8];
  for (uint part = 0; part < elements; ++part) {
    const uint channel = channel0 + lane * elements + part;
    float result = 0.0f;
    float inputs[8];
    for (uint tap = 0; tap < p.taps; ++tap) {
      const uint source = windows[row * p.taps + tap];
      const float input = source < keep ? float(history[(row_slot[row] * keep + source) * channels + channel]) : float(projection[(source - keep) * channels + channel]);
      if (STEP) inputs[tap] = input;
      result += tf_delta_weight(weights, channel * p.taps + tap, p.weight_flags, 0) * input;
    }
    const float rounded = float(bfloat(result));
    const float sigmoid = float(bfloat(1.0f / (1.0f + metal::exp(-rounded))));
    activated[part] = float(bfloat(rounded * sigmoid));
    if (STEP) {
      for (uint tap = 0; tap < keep; ++tap) next_history[(row_slot[row] * keep + tap) * channels + channel] = bfloat(inputs[tap + 1]);
    }
  }
  if (head < 2 * p.nk) {
    float square = 0.0f;
    for (uint part = 0; part < elements; ++part) square += activated[part] * activated[part];
    square = simd_sum(square);
    const float inverse = metal::rsqrt(square / float(p.dk) + 1e-6f);
    const float scale = head < p.nk ? float(bfloat(1.0f / float(p.dk))) : float(bfloat(metal::rsqrt(float(p.dk))));
    const uint output_head = head % p.nk;
    for (uint part = 0; part < elements; ++part) {
      const uint position = (row * p.nk + output_head) * p.dk + lane * elements + part;
      const bfloat normalized = bfloat(scale * float(bfloat(activated[part] * inverse)));
      if (head < p.nk) query[position] = normalized;
      else key[position] = normalized;
    }
    return;
  }
  const uint output_head = head - 2 * p.nk;
  for (uint part = 0; part < elements; ++part) value[(row * p.nv + output_head) * p.dv + lane * elements + part] = bfloat(activated[part]);
  if (lane == 0) {
    const uint at = row * p.nv + output_head, gate_at = p.zba_stride ? row * p.zba_stride + output_head : at;
    const float shifted = float(bfloat(float(a[gate_at]) + tf_delta_weight(time_bias, output_head, p.weight_flags, 2)));
    const float softplus = float(bfloat(metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted)))));
    decay[at] = metal::exp(-metal::exp(tf_delta_weight(decay_weight, output_head, p.weight_flags, 1)) * softplus);
    mixing[at] = bfloat(1.0f / (1.0f + metal::exp(-float(b[gate_at]))));
  }
}

kernel void tf_delta_gdn_pre(device const bfloat* projection [[buffer(0)]],
                        device const bfloat* a [[buffer(1)]],
                        device const bfloat* b [[buffer(2)]],
                        device const bfloat* history [[buffer(3)]],
                        device const void* weights [[buffer(4)]],
                        device const void* decay_weight [[buffer(5)]],
                        device const void* time_bias [[buffer(6)]],
                        device const uint* windows [[buffer(7)]],
                        device const uint* row_slot [[buffer(8)]],
                        constant TfDeltaParams& p [[buffer(9)]],
                        device bfloat* query [[buffer(10)]],
                        device bfloat* key [[buffer(11)]],
                        device bfloat* value [[buffer(12)]],
                        device float* decay [[buffer(13)]],
                        device bfloat* mixing [[buffer(14)]],
                        device bfloat* tails [[buffer(15)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  if (p.dk == 128 && p.dv == 128) tf_delta_gdn_pre_body<4, false>(projection, a, b, history, weights, decay_weight, time_bias, windows, row_slot, p, query, key, value, decay, mixing, tails, group, lane);
  else tf_delta_gdn_pre_body<0, false>(projection, a, b, history, weights, decay_weight, time_bias, windows, row_slot, p, query, key, value, decay, mixing, tails, group, lane);
}

// tf_delta_gdn_pre_body<4, false> for prompt rows (4 taps): a simdgroup steps up to 4 rows of a head on conv weights read
// once; bfloat4 rows, and the gates a lane a row. A group's rows split evenly over its four simdgroups (2 each for a
// verify's 8 rows), so none walks more rows than it must; each row's arithmetic is the same whichever walks it.
inline void tf_delta_gdn_pre_rows4(device const bfloat* projection, device const bfloat* a, device const bfloat* b,
    device const bfloat* history, device const void* weights, device const void* decay_weight, device const void* time_bias,
    device const uint* windows, device const uint* row_slot, constant TfDeltaParams& p, device bfloat* query,
    device bfloat* key, device bfloat* value, device float* decay, device bfloat* mixing, uint3 group, uint sg, uint lane) {
  const uint head = group.y, base = group.z * 16, in_group = metal::min(16u, p.rows - base), per = (in_group + 3) / 4;
  if (sg * per >= in_group) return;
  const uint first = base + sg * per, count = metal::min(per, in_group - sg * per);
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv;
  const uint keep = p.taps - 1;
  const uint channel0 = head < 2 * p.nk ? head * p.dk : 2 * p.nk * p.dk + (head - 2 * p.nk) * p.dv;
  const uint channel = channel0 + lane * 4;
  float weight[4][4];
  for (uint part = 0; part < 4; ++part)
    for (uint tap = 0; tap < 4; ++tap) weight[part][tap] = tf_delta_weight(weights, (channel + part) * p.taps + tap, p.weight_flags, 0);
  for (uint i = 0; i < count; ++i) {
    const uint row = first + i;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // the row's slot read once (row_slot holds p.rows words), not in each history tap's arm: no branch around a tap's load
    const uint slot = row_slot[row];
    for (uint tap = 0; tap < 4; ++tap) {
      const uint source = windows[row * p.taps + tap];
      const bfloat4 input = source < keep ? *(device const bfloat4*)(history + (slot * keep + source) * channels + channel)
                                          : *(device const bfloat4*)(projection + (source - keep) * channels + channel);
      for (uint part = 0; part < 4; ++part) result[part] += weight[part][tap] * float(input[part]);
    }
    float activated[4];
    for (uint part = 0; part < 4; ++part) {
      const float rounded = float(bfloat(result[part]));
      const float sigmoid = float(bfloat(1.0f / (1.0f + metal::exp(-rounded))));
      activated[part] = float(bfloat(rounded * sigmoid));
    }
    if (head < 2 * p.nk) {
      float square = 0.0f;
      for (uint part = 0; part < 4; ++part) square += activated[part] * activated[part];
      square = simd_sum(square);
      const float inverse = metal::rsqrt(square / float(p.dk) + 1e-6f);
      const float scale = head < p.nk ? float(bfloat(1.0f / float(p.dk))) : float(bfloat(metal::rsqrt(float(p.dk))));
      bfloat4 normalized;
      for (uint part = 0; part < 4; ++part) normalized[part] = bfloat(scale * float(bfloat(activated[part] * inverse)));
      *(device bfloat4*)((head < p.nk ? query : key) + (row * p.nk + head % p.nk) * p.dk + lane * 4) = normalized;
    } else {
      bfloat4 activation;
      for (uint part = 0; part < 4; ++part) activation[part] = bfloat(activated[part]);
      *(device bfloat4*)(value + (row * p.nv + head - 2 * p.nk) * p.dv + lane * 4) = activation;
    }
  }
  if (head >= 2 * p.nk && lane < count) {
    const uint output_head = head - 2 * p.nk, row = first + lane;
    const uint at = row * p.nv + output_head, gate_at = p.zba_stride ? row * p.zba_stride + output_head : at;
    const float shifted = float(bfloat(float(a[gate_at]) + tf_delta_weight(time_bias, output_head, p.weight_flags, 2)));
    const float softplus = float(bfloat(metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted)))));
    decay[at] = metal::exp(-metal::exp(tf_delta_weight(decay_weight, output_head, p.weight_flags, 1)) * softplus);
    mixing[at] = bfloat(1.0f / (1.0f + metal::exp(-float(b[gate_at]))));
  }
}

// tf_delta_gdn_pre past 16 rows of 128-wide heads: 4 simdgroups a group, 16 rows.
kernel void tf_delta_gdn_pre_rows(device const bfloat* projection [[buffer(0)]],
                        device const bfloat* a [[buffer(1)]],
                        device const bfloat* b [[buffer(2)]],
                        device const bfloat* history [[buffer(3)]],
                        device const void* weights [[buffer(4)]],
                        device const void* decay_weight [[buffer(5)]],
                        device const void* time_bias [[buffer(6)]],
                        device const uint* windows [[buffer(7)]],
                        device const uint* row_slot [[buffer(8)]],
                        constant TfDeltaParams& p [[buffer(9)]],
                        device bfloat* query [[buffer(10)]],
                        device bfloat* key [[buffer(11)]],
                        device bfloat* value [[buffer(12)]],
                        device float* decay [[buffer(13)]],
                        device bfloat* mixing [[buffer(14)]],
                        device atomic_uint* NC [[buffer(15)]],
                        constant uint& wait_for [[buffer(16)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint3 groups [[threadgroups_per_grid]],
                        uint t [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  // wait_for above 0: this prepare shares its barrier interval with the decoded qkv and zba projections it reads, whose
  // threadgroups count themselves on NC[0] past a device-scope fence (lane_projection's TF_COUNT); a group reads them once
  // the count reaches wait_for (bounded), past its own fence, and the last group leaves both words zero for the next pair
  if (wait_for != 0u) {
    if (t == 0) {
      for (uint spin = 0; spin < (1u << 24) && atomic_load_explicit(&NC[0], memory_order_relaxed) < wait_for; spin++) {}
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      const uint total = groups.x * groups.y * groups.z;
      if (atomic_fetch_add_explicit(&NC[1], 1u, memory_order_relaxed) + 1u == total) {
        atomic_store_explicit(&NC[0], 0u, memory_order_relaxed);
        atomic_store_explicit(&NC[1], 0u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (p.dk == 128 && p.dv == 128 && p.taps == 4) tf_delta_gdn_pre_rows4(projection, a, b, history, weights, decay_weight, time_bias, windows, row_slot, p, query, key, value, decay, mixing, group, sg, lane);
}

// tf_delta_gdn_pre_body<4, true> over 4 simdgroups a head; each lane's 4 squares summed in order before simd_sum.
inline void tf_delta_gdn_pre_step4(device const bfloat* projection, device const bfloat* a, device const bfloat* b,
    device bfloat* history, device const void* weights, device const void* decay_weight, device const void* time_bias,
    device const uint* windows, device const uint* row_slot, constant TfDeltaParams& p, device bfloat* query,
    device bfloat* key, device bfloat* value, device float* decay, device bfloat* mixing, threadgroup float* shared,
    uint3 group, uint sg, uint lane) {
  const uint row = group.z, head = group.y;
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv;
  const uint keep = p.taps - 1;
  const uint channel0 = head < 2 * p.nk ? head * p.dk : 2 * p.nk * p.dk + (head - 2 * p.nk) * p.dv;
  const uint channel = channel0 + lane * 4 + sg;
  float result = 0.0f;
  float inputs[8];
  for (uint tap = 0; tap < p.taps; ++tap) {
    const uint source = windows[row * p.taps + tap];
    const float input = source < keep ? float(history[(row_slot[row] * keep + source) * channels + channel]) : float(projection[(source - keep) * channels + channel]);
    inputs[tap] = input;
    result += tf_delta_weight(weights, channel * p.taps + tap, p.weight_flags, 0) * input;
  }
  const float rounded = float(bfloat(result));
  const float sigmoid = float(bfloat(1.0f / (1.0f + metal::exp(-rounded))));
  const float activated = float(bfloat(rounded * sigmoid));
  for (uint tap = 0; tap < keep; ++tap) history[(row_slot[row] * keep + tap) * channels + channel] = bfloat(inputs[tap + 1]);
  if (head < 2 * p.nk) {
    shared[sg * 32 + lane] = activated;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float square = 0.0f;
      for (uint part = 0; part < 4; ++part) square += shared[part * 32 + lane] * shared[part * 32 + lane];
      square = simd_sum(square);
      if (lane == 0) shared[128] = metal::rsqrt(square / float(p.dk) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = shared[128];
    const float scale = head < p.nk ? float(bfloat(1.0f / float(p.dk))) : float(bfloat(metal::rsqrt(float(p.dk))));
    const uint position = (row * p.nk + head % p.nk) * p.dk + lane * 4 + sg;
    const bfloat normalized = bfloat(scale * float(bfloat(activated * inverse)));
    if (head < p.nk) query[position] = normalized;
    else key[position] = normalized;
    return;
  }
  const uint output_head = head - 2 * p.nk;
  value[(row * p.nv + output_head) * p.dv + lane * 4 + sg] = bfloat(activated);
  if (sg == 0 && lane == 0) {
    const uint at = row * p.nv + output_head, gate_at = p.zba_stride ? row * p.zba_stride + output_head : at;
    const float shifted = float(bfloat(float(a[gate_at]) + tf_delta_weight(time_bias, output_head, p.weight_flags, 2)));
    const float softplus = float(bfloat(metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted)))));
    decay[at] = metal::exp(-metal::exp(tf_delta_weight(decay_weight, output_head, p.weight_flags, 1)) * softplus);
    mixing[at] = bfloat(1.0f / (1.0f + metal::exp(-float(b[gate_at]))));
  }
}

// tf_delta_gdn_pre for one-row segments kept in place (conv_commit's window into history); 128-wide heads 4 simdgroups.
kernel void tf_delta_gdn_pre_step(device const bfloat* projection [[buffer(0)]],
                        device const bfloat* a [[buffer(1)]],
                        device const bfloat* b [[buffer(2)]],
                        device bfloat* history [[buffer(3)]],
                        device const void* weights [[buffer(4)]],
                        device const void* decay_weight [[buffer(5)]],
                        device const void* time_bias [[buffer(6)]],
                        device const uint* windows [[buffer(7)]],
                        device const uint* row_slot [[buffer(8)]],
                        constant TfDeltaParams& p [[buffer(9)]],
                        device bfloat* query [[buffer(10)]],
                        device bfloat* key [[buffer(11)]],
                        device bfloat* value [[buffer(12)]],
                        device float* decay [[buffer(13)]],
                        device bfloat* mixing [[buffer(14)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float shared[129];
  if (p.dk == 128 && p.dv == 128) tf_delta_gdn_pre_step4(projection, a, b, history, weights, decay_weight, time_bias, windows, row_slot, p, query, key, value, decay, mixing, shared, group, sg, lane);
  else if (sg == 0) tf_delta_gdn_pre_body<0, true>(projection, a, b, history, weights, decay_weight, time_bias, windows, row_slot, p, query, key, value, decay, mixing, nullptr, group, lane);
}
// One tree node from its parent's state: tf_delta_delta_update and tf_delta_delta_read as the snapshot path runs them.
inline float4 tf_delta_tree_node(float4 parent_state, device const bfloat* query, device const bfloat* key, device const bfloat* value,
                                 device const float* decay, device const bfloat* mixing, constant TfDeltaParams& p, device bfloat* output,
                                 uint row, uint head, uint key_head, uint dimension, uint lane) {
  float state[4] = {parent_state[0], parent_state[1], parent_state[2], parent_state[3]};
  const uint at = row * p.nv + head;
  tf_delta_delta_update(state, key + (row * p.nk + key_head) * p.dk, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, 4);
  const float result = tf_delta_delta_read(state, query + (row * p.nk + key_head) * p.dk, lane, 4);
  if (lane == 0) output[at * p.dv + dimension] = bfloat(result);
  return float4(state[0], state[1], state[2], state[3]);
}

// Parent states (registers to 16 nodes, else snapshots) and the kept-path replay share one fp32 state update.
kernel void tf_delta_gdn_tree(device const bfloat* query [[buffer(0)]],
                         device const bfloat* key [[buffer(1)]],
                         device const bfloat* value [[buffer(2)]],
                         device const float* decay [[buffer(3)]],
                         device const bfloat* mixing [[buffer(4)]],
                         device const float* committed [[buffer(5)]],
                         device const int* parents [[buffer(6)]],
                         device const TfDeltaSegment* segments [[buffer(7)]],
                         constant TfDeltaParams& p [[buffer(8)]],
                         device bfloat* output [[buffer(9)]],
                         device float* snapshots [[buffer(10)]],
                         uint3 position [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk), count = p.dk / 32;
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * count;
  float state[8];
  if (count == 4 && span.rows <= 16) {
    // Up to 16 nodes hold states in named registers: a child takes its parent's through a uniform switch.
    float4 root;
    for (uint part = 0; part < 4; ++part) root[part] = committed[span.state_slot * state_stride + column + part];
    float4 kept0 = 0.0f, kept1 = 0.0f, kept2 = 0.0f, kept3 = 0.0f, kept4 = 0.0f, kept5 = 0.0f, kept6 = 0.0f, kept7 = 0.0f;
    float4 kept8 = 0.0f, kept9 = 0.0f, kept10 = 0.0f, kept11 = 0.0f, kept12 = 0.0f, kept13 = 0.0f, kept14 = 0.0f, kept15 = 0.0f;
#define TF_PICK(j) case j: s = kept##j; break;
#define TF_TREE_NODE(i) \
    if (i < span.rows) { \
      float4 s = root; \
      switch (parents[span.first + i]) { TF_PICK(0) TF_PICK(1) TF_PICK(2) TF_PICK(3) TF_PICK(4) TF_PICK(5) TF_PICK(6) TF_PICK(7) TF_PICK(8) TF_PICK(9) TF_PICK(10) TF_PICK(11) TF_PICK(12) TF_PICK(13) TF_PICK(14) default: break; } \
      kept##i = tf_delta_tree_node(s, query, key, value, decay, mixing, p, output, span.first + i, head, key_head, dimension, lane); \
    }
    TF_TREE_NODE(0) TF_TREE_NODE(1) TF_TREE_NODE(2) TF_TREE_NODE(3) TF_TREE_NODE(4) TF_TREE_NODE(5) TF_TREE_NODE(6) TF_TREE_NODE(7)
    TF_TREE_NODE(8) TF_TREE_NODE(9) TF_TREE_NODE(10) TF_TREE_NODE(11) TF_TREE_NODE(12) TF_TREE_NODE(13) TF_TREE_NODE(14) TF_TREE_NODE(15)
#undef TF_TREE_NODE
#undef TF_PICK
    return;
  }
  for (uint node = 0; node < span.rows; ++node) {
    const uint row = span.first + node;
    const int parent = parents[row];
    const uint source = parent < 0 ? span.state_slot * state_stride : (span.first + uint(parent)) * state_stride;
    for (uint part = 0; part < count; ++part) state[part] = parent < 0 ? committed[source + column + part] : snapshots[source + column + part];
    const uint at = row * p.nv + head;
    device const bfloat* k = key + (row * p.nk + key_head) * p.dk;
    device const bfloat* q = query + (row * p.nk + key_head) * p.dk;
    tf_delta_delta_update(state, k, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, count);
    const float result = tf_delta_delta_read(state, q, lane, count);
    if (lane == 0) output[at * p.dv + dimension] = bfloat(result);
    for (uint part = 0; part < count; ++part) snapshots[row * state_stride + column + part] = state[part];
  }
}

// C = 4 (128-wide heads) gives every per-lane loop a constant trip count (arrays in registers); C = 0 reads p.
template <uint C>
inline void tf_delta_gdn_replay_body(device const bfloat* key,
    device const bfloat* value,
    device const float* decay,
    device const bfloat* mixing,
    device const float* committed,
    device const uint* kept_rows,
    device const TfDeltaKeep* keeps,
    constant TfDeltaParams& p,
    device float* next,
    uint3 position,
    uint lane) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk), count = C ? C : p.dk / 32;
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * count;
  const TfDeltaKeep keep = keeps[segment_index];
  float state[C ? C : 8];
  for (uint part = 0; part < count; ++part) state[part] = committed[keep.state_slot * state_stride + column + part];
  for (uint step = 0; step < keep.rows; ++step) {
    const uint row = kept_rows[keep.first + step], at = row * p.nv + head;
    device const bfloat* k = key + (row * p.nk + key_head) * p.dk;
    tf_delta_delta_update(state, k, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, count);
  }
  for (uint part = 0; part < count; ++part) next[keep.next_slot * state_stride + column + part] = state[part];
}

kernel void tf_delta_gdn_replay(device const bfloat* key [[buffer(0)]],
                           device const bfloat* value [[buffer(1)]],
                           device const float* decay [[buffer(2)]],
                           device const bfloat* mixing [[buffer(3)]],
                           device const float* committed [[buffer(4)]],
                           device const uint* kept_rows [[buffer(5)]],
                           device const TfDeltaKeep* keeps [[buffer(6)]],
                           constant TfDeltaParams& p [[buffer(7)]],
                           device float* next [[buffer(8)]],
                           device const TfDeltaSegment* segments [[buffer(9)]],
                           device const float* before [[buffer(10)]],
                           uint3 position [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint segment = position.z / p.nv;
  if (p.advance != 0 && segments[segment].rows == 1) {
    const TfDeltaKeep keep = keeps[segment];
    if (keep.rows == 0) {
      const uint stride = p.nv * p.dv * p.dk;
      const uint column = ((position.z % p.nv) * p.dv + position.y) * p.dk + lane * 4;
      for (uint part = 0; part < 4; ++part)
        next[keep.next_slot * stride + column + part] = before[keep.state_slot * stride + column + part];
    }
    return;
  }
  if (p.dk == 128) tf_delta_gdn_replay_body<4>(key, value, decay, mixing, committed, kept_rows, keeps, p, next, position, lane);
  else tf_delta_gdn_replay_body<0>(key, value, decay, mixing, committed, kept_rows, keeps, p, next, position, lane);
}

// A deferred keep's rows: the pending row its first kept row lands at, the rows left after it, and the log's and pending
// rows a layer (their strides).
struct TfDeltaStash { uint base, room, log_rows, pending_rows; };
// A depth-one star keep that defers its replay: the kept rows tf_delta_gdn_replay would replay (keeps[0], kept_rows, in
// its order), their key, value, decay and mixing words copied unchanged into every layer's pending rows from s.base, at
// most s.room. A thread a 16-byte word of the key and value rows and a word of the gates; y is the layer. dk = dv = 128.
kernel void tf_delta_gdn_stash(device const uint4* key [[buffer(0)]],
                               device const uint4* value [[buffer(1)]],
                               device const uint* decay [[buffer(2)]],
                               device const ushort* mixing [[buffer(3)]],
                               device const uint* kept_rows [[buffer(4)]],
                               device const TfDeltaKeep* keeps [[buffer(5)]],
                               constant TfDeltaParams& p [[buffer(6)]],
                               constant TfDeltaStash& s [[buffer(7)]],
                               device uint4* pending_key [[buffer(8)]],
                               device uint4* pending_value [[buffer(9)]],
                               device uint* pending_decay [[buffer(10)]],
                               device ushort* pending_mixing [[buffer(11)]],
                               uint2 position [[thread_position_in_grid]]) {
  const uint word = position.x, layer = position.y;
  const uint keys = p.nk * p.dk / 8, values = p.nv * p.dv / 8;
  const TfDeltaKeep keep = keeps[0];
  for (uint step = 0; step < keep.rows && step < s.room; ++step) {
    const ulong from = ulong(layer) * s.log_rows + kept_rows[keep.first + step];
    const ulong to = ulong(layer) * s.pending_rows + s.base + step;
    if (word < keys) pending_key[to * keys + word] = key[from * keys + word];
    if (word < values) pending_value[to * values + word] = value[from * values + word];
    if (word < p.nv) {
      pending_decay[to * p.nv + word] = decay[from * p.nv + word];
      pending_mixing[to * p.nv + word] = mixing[from * p.nv + word];
    }
  }
}

// The pending rows, their slot, the pending rows a layer, and a layer's committed words (every slot's).
struct TfDeltaUnstash { uint rows, slot, pending_rows, layer_words; };
// The pending rows (tf_delta_gdn_stash's) replayed into slot u.slot's committed state of every layer, in place: a value
// column a simdgroup as in tf_delta_gdn_replay_body<4>, its lane words taking each row in keep order by
// tf_delta_delta_update with a kept row's arguments, so the state is the one the deferred keeps' replays would have left.
// z is layer * nv + head. dk = 128.
kernel void tf_delta_gdn_unstash(device const bfloat* key [[buffer(0)]],
                                 device const bfloat* value [[buffer(1)]],
                                 device const float* decay [[buffer(2)]],
                                 device const bfloat* mixing [[buffer(3)]],
                                 device float* state [[buffer(4)]],
                                 constant TfDeltaParams& p [[buffer(5)]],
                                 constant TfDeltaUnstash& u [[buffer(6)]],
                                 uint3 position [[thread_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint layer = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk);
  device float* words = state + ulong(layer) * u.layer_words + ulong(u.slot) * (p.nv * p.dv * p.dk) + (head * p.dv + dimension) * p.dk + lane * 4;
  const ulong first = ulong(layer) * u.pending_rows;
  device const bfloat* layer_key = key + first * (p.nk * p.dk);
  device const bfloat* layer_value = value + first * (p.nv * p.dv);
  device const float* layer_decay = decay + first * p.nv;
  device const bfloat* layer_mixing = mixing + first * p.nv;
  float state_words[4];
  for (uint part = 0; part < 4; ++part) state_words[part] = words[part];
  for (uint row = 0; row < u.rows; ++row) {
    const uint at = row * p.nv + head;
    tf_delta_delta_update(state_words, layer_key + (row * p.nk + key_head) * p.dk, layer_decay[at], float(layer_value[at * p.dv + dimension]), float(layer_mixing[at]), lane, 4);
  }
  for (uint part = 0; part < 4; ++part) words[part] = state_words[part];
}

// Accepted chains keep the tree path's delta update and sum order; a prompt chunk differs only in trip count.
template <uint C>
inline void tf_delta_gdn_chain_body(device const bfloat* query,
    device const bfloat* key,
    device const bfloat* value,
    device const float* decay,
    device const bfloat* mixing,
    device const float* committed,
    device const TfDeltaSegment* segments,
    constant TfDeltaParams& p,
    device bfloat* output,
    device float* next_state,
    uint3 position,
    uint lane) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk), count = C ? C : p.dk / 32;
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * count;
  float state[C ? C : 8];
  for (uint part = 0; part < count; ++part) state[part] = committed[span.state_slot * state_stride + column + part];
  for (uint node = 0; node < span.rows; ++node) {
    const uint row = span.first + node, at = row * p.nv + head;
    device const bfloat* k = key + (row * p.nk + key_head) * p.dk;
    device const bfloat* q = query + (row * p.nk + key_head) * p.dk;
    tf_delta_delta_update(state, k, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, count);
    const float result = tf_delta_delta_read(state, q, lane, count);
    if (lane == 0) output[at * p.dv + dimension] = bfloat(result);
  }
  for (uint part = 0; part < count; ++part) next_state[span.next_slot * state_stride + column + part] = state[part];
}

kernel void tf_delta_gdn_chain(device const bfloat* query [[buffer(0)]],
                          device const bfloat* key [[buffer(1)]],
                          device const bfloat* value [[buffer(2)]],
                          device const float* decay [[buffer(3)]],
                          device const bfloat* mixing [[buffer(4)]],
                          device const float* committed [[buffer(5)]],
                          device const TfDeltaSegment* segments [[buffer(6)]],
                          constant TfDeltaParams& p [[buffer(7)]],
                          device bfloat* output [[buffer(8)]],
                          device float* next_state [[buffer(9)]],
                          uint3 position [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  if (p.dk == 128) tf_delta_gdn_chain_body<4>(query, key, value, decay, mixing, committed, segments, p, output, next_state, position, lane);
  else tf_delta_gdn_chain_body<0>(query, key, value, decay, mixing, committed, segments, p, output, next_state, position, lane);
}
// Prompt chains (may differ in bits from decode): 8 lanes a set of CC value columns (4 unmarked, 2 marked), 16 key
// channels of each; dk = 128. Every column keeps its own arithmetic and order whatever CC is.
// Each column keeps its in-order 16-channel sums and xor 1, 2, 4 tree; the pair shares k and q, converted once a row.
#define TF_UNROLL _Pragma("clang loop unroll(full)")
template <bool M, uint CC>
inline void tf_delta_gdn_chain_wide_body(device const bfloat* query,
    device const bfloat* key,
    device const bfloat* value,
    device const float* decay,
    device const bfloat* mixing,
    device const float* committed,
    device const TfDeltaSegment* segments,
    constant TfDeltaParams& p,
    device bfloat* output,
    device float* next_state,
    const uint4 marks,
    device float* mark_state,
    uint3 position,
    uint lane) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint sub = lane & 7, dimension = position.y * (4 * CC) + (lane >> 3) * CC, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + sub * 16;
  float state[CC][16]; // state[c]: value column dimension + c, its words at column + c * p.dk
  TF_UNROLL for (uint c = 0; c < CC; ++c)
    TF_UNROLL for (uint i = 0; i < 16; ++i) state[c][i] = committed[span.state_slot * state_stride + column + c * p.dk + i];
  for (uint node = 0; node < span.rows; ++node) {
    const uint row = span.first + node, at = row * p.nv + head;
    device const bfloat* k = key + (row * p.nk + key_head) * p.dk + sub * 16;
    device const bfloat* q = query + (row * p.nk + key_head) * p.dk + sub * 16;
    float kf[16], qf[16], remembered[CC], correction[CC], result[CC];
    TF_UNROLL for (uint c = 0; c < CC; ++c) { remembered[c] = 0.0f; result[c] = 0.0f; }
    TF_UNROLL for (uint i = 0; i < 16; ++i) kf[i] = float(k[i]);
    const float forget = decay[at];
    TF_UNROLL for (uint i = 0; i < 16; ++i)
      TF_UNROLL for (uint c = 0; c < CC; ++c) {
        state[c][i] = state[c][i] * forget;
        remembered[c] += state[c][i] * kf[i];
      }
    TF_UNROLL for (uint step = 1; step < 8; step <<= 1)
      TF_UNROLL for (uint c = 0; c < CC; ++c) remembered[c] += simd_shuffle_xor(remembered[c], step);
    const float mix = float(mixing[at]);
    TF_UNROLL for (uint c = 0; c < CC; ++c) correction[c] = (float(value[at * p.dv + dimension + c]) - remembered[c]) * mix;
    TF_UNROLL for (uint i = 0; i < 16; ++i) qf[i] = float(q[i]);
    TF_UNROLL for (uint i = 0; i < 16; ++i)
      TF_UNROLL for (uint c = 0; c < CC; ++c) {
        state[c][i] = state[c][i] + kf[i] * correction[c];
        result[c] += state[c][i] * qf[i];
      }
    TF_UNROLL for (uint step = 1; step < 8; step <<= 1)
      TF_UNROLL for (uint c = 0; c < CC; ++c) result[c] += simd_shuffle_xor(result[c], step);
    if (sub == 0)
      TF_UNROLL for (uint c = 0; c < CC; ++c) output[at * p.dv + dimension + c] = bfloat(result[c]);
    if (M)
      for (uint m = 0; m < 4; ++m)
        if (marks[m] == node + 1)
          TF_UNROLL for (uint c = 0; c < CC; ++c)
            TF_UNROLL for (uint i = 0; i < 16; ++i) mark_state[(segment_index * 4 + m) * state_stride + column + c * p.dk + i] = state[c][i];
  }
  TF_UNROLL for (uint c = 0; c < CC; ++c)
    TF_UNROLL for (uint i = 0; i < 16; ++i) next_state[span.next_slot * state_stride + column + c * p.dk + i] = state[c][i];
}
#undef TF_UNROLL

kernel void tf_delta_gdn_chain_wide(device const bfloat* query [[buffer(0)]],
                          device const bfloat* key [[buffer(1)]],
                          device const bfloat* value [[buffer(2)]],
                          device const float* decay [[buffer(3)]],
                          device const bfloat* mixing [[buffer(4)]],
                          device const float* committed [[buffer(5)]],
                          device const TfDeltaSegment* segments [[buffer(6)]],
                          constant TfDeltaParams& p [[buffer(7)]],
                          device bfloat* output [[buffer(8)]],
                          device float* next_state [[buffer(9)]],
                          uint3 position [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  tf_delta_gdn_chain_wide_body<false, 4>(query, key, value, decay, mixing, committed, segments, p, output, next_state,
                                      uint4(0), next_state, position, lane);
}

// marks[segment] (its rows, 0 unused): mark_state[segment * 4 + m] is the state a segment ending there would leave.
kernel void tf_delta_gdn_chain_wide_marks(device const bfloat* query [[buffer(0)]],
                          device const bfloat* key [[buffer(1)]],
                          device const bfloat* value [[buffer(2)]],
                          device const float* decay [[buffer(3)]],
                          device const bfloat* mixing [[buffer(4)]],
                          device const float* committed [[buffer(5)]],
                          device const TfDeltaSegment* segments [[buffer(6)]],
                          constant TfDeltaParams& p [[buffer(7)]],
                          device bfloat* output [[buffer(8)]],
                          device float* next_state [[buffer(9)]],
                          device const uint4* marks [[buffer(10)]],
                          device float* mark_state [[buffer(11)]],
                          uint3 position [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  tf_delta_gdn_chain_wide_body<true, 2>(query, key, value, decay, mixing, committed, segments, p, output, next_state,
                                     marks[position.z / p.nv], mark_state, position, lane);
}

// Each mark's conv window into mark_conv[segment * 4 + m], read as conv_commit reads it for a segment ending there.
kernel void tf_delta_gdn_mark_conv(device const bfloat* committed [[buffer(0)]],
                              device const bfloat* projection [[buffer(1)]],
                              device const TfDeltaSegment* segments [[buffer(2)]],
                              device const uint* windows [[buffer(3)]],
                              constant TfDeltaParams& p [[buffer(4)]],
                              device const uint4* marks [[buffer(5)]],
                              device bfloat* mark_conv [[buffer(6)]],
                              uint3 position [[thread_position_in_grid]]) {
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv;
  const uint width = (p.taps - 1) * channels, item = position.x, mark = position.y, segment_index = position.z;
  if (item >= width || mark >= 4) return;
  const TfDeltaSegment span = segments[segment_index];
  const uint at = marks[segment_index][mark];
  if (at == 0 || at > span.rows) return;
  const uint row = span.first + at - 1;
  const uint source = windows[row * p.taps + item / channels + 1];
  mark_conv[(segment_index * 4 + mark) * width + item] = source < p.taps - 1
      ? committed[(span.state_slot * (p.taps - 1) + source) * channels + item % channels]
      : projection[(source - (p.taps - 1)) * channels + item % channels];
}
// The normalized value rounds to BF16 before the SiLU gate multiplies it.

// C = 4 (128-wide heads) gives every per-lane loop a constant trip count (arrays in registers); C = 0 reads p.
template <uint C>
inline void tf_delta_gdn_post_body(device const bfloat* recurrence,
    device const bfloat* gate,
    device const void* norm,
    constant TfDeltaParams& p,
    device bfloat* output,
    uint3 group,
    uint lane) {
  const uint row = group.z, head = group.y, elements = C ? C : p.dv / 32;
  float values[C ? C : 8];
  float square = 0.0f;
  const uint at = (row * p.nv + head) * p.dv + lane * elements, z_at = p.zba_stride ? row * p.zba_stride + head * p.dv + lane * elements : at;
  for (uint part = 0; part < elements; ++part) {
    values[part] = float(recurrence[at + part]);
    square += values[part] * values[part];
  }
  square = simd_sum(square);
  const float inverse = metal::rsqrt(square / float(p.dv) + p.eps);
  for (uint part = 0; part < elements; ++part) {
    const uint channel = lane * elements + part;
    const float normalized = float(bfloat(tf_delta_weight(norm, channel, p.weight_flags, 3) * (values[part] * inverse)));
    const float z = float(gate[z_at + part]);
    output[at + part] = bfloat(z / (1.0f + metal::exp(-z)) * normalized);
  }
}

kernel void tf_delta_gdn_post(device const bfloat* recurrence [[buffer(0)]],
                         device const bfloat* gate [[buffer(1)]],
                         device const void* norm [[buffer(2)]],
                         constant TfDeltaParams& p [[buffer(3)]],
                         device bfloat* output [[buffer(4)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  if (p.dv == 128) tf_delta_gdn_post_body<4>(recurrence, gate, norm, p, output, group, lane);
  else tf_delta_gdn_post_body<0>(recurrence, gate, norm, p, output, group, lane);
}

// tf_delta_gdn_post_body<4> on bfloat4 rows: a simdgroup a row, 4 rows a group of 128.
kernel void tf_delta_gdn_post_rows(device const bfloat* recurrence [[buffer(0)]],
                         device const bfloat* gate [[buffer(1)]],
                         device const void* norm [[buffer(2)]],
                         constant TfDeltaParams& p [[buffer(3)]],
                         device bfloat* output [[buffer(4)]],
                         device float* sums [[buffer(5)]],
                         constant uint& sums_stride [[buffer(6)]],
                         device atomic_uint* NC [[buffer(7)]],
                         constant uint& wait_for [[buffer(8)]],
                         device atomic_uint* OC [[buffer(9)]],
                         constant uint& count_out [[buffer(10)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 groups [[threadgroups_per_grid]],
                         uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  // wait_for above 0: this gate shares its barrier interval with the star scan that writes `recurrence`, whose threadgroups
  // count themselves on NC[0] past a device-scope fence; a group reads it once the count reaches wait_for (bounded), past
  // its own fence, and the last group leaves both words zero for the next pair
  if (wait_for != 0u) {
    if (t == 0) {
      for (uint spin = 0; spin < (1u << 24) && atomic_load_explicit(&NC[0], memory_order_relaxed) < wait_for; spin++) {}
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      const uint total = groups.x * groups.y * groups.z;
      if (atomic_fetch_add_explicit(&NC[1], 1u, memory_order_relaxed) + 1u == total) {
        atomic_store_explicit(&NC[0], 0u, memory_order_relaxed);
        atomic_store_explicit(&NC[1], 0u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint row = group.z * 4 + sg, head = group.y;
  if (row < p.rows && p.dv == 128) {
    const uint at = (row * p.nv + head) * p.dv + lane * 4, z_at = p.zba_stride ? row * p.zba_stride + head * p.dv + lane * 4 : at;
    const bfloat4 x = *(device const bfloat4*)(recurrence + at), z4 = *(device const bfloat4*)(gate + z_at);
    float values[4];
    float square = 0.0f;
    for (uint part = 0; part < 4; ++part) {
      values[part] = float(x[part]);
      square += values[part] * values[part];
    }
    square = simd_sum(square);
    const float inverse = metal::rsqrt(square / float(p.dv) + p.eps);
    bfloat4 y;
    for (uint part = 0; part < 4; ++part) {
      const float normalized = float(bfloat(tf_delta_weight(norm, lane * 4 + part, p.weight_flags, 3) * (values[part] * inverse)));
      const float z = float(z4[part]);
      y[part] = bfloat(z / (1.0f + metal::exp(-z)) * normalized);
    }
    *(device bfloat4*)(output + at) = y;
    if (sums_stride != 0) {
      simdgroup_barrier(mem_flags::mem_device);
      if ((lane & 15) == 0) {
        const int width = int(p.nv * p.dv), g = int((head * p.dv + lane * 4) / 64);
        float acc = 0.0f;
        for (int i = 0; i < 64; i++) acc += float(output[(size_t)row * width + g * 64 + i]);
        sums[g * int(sums_stride) + int(row)] = acc;
      }
    }
  }
  if (count_out != 0u) {
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(&OC[0], 1u, memory_order_relaxed);
    }
  }
}
// Projected rows and ancestor indices reconstruct the accepted tail without persisting every node's conv window.

kernel void tf_delta_gdn_conv_commit(device const bfloat* committed [[buffer(0)]],
                                device const bfloat* projection [[buffer(1)]],
                                device const uint* kept_rows [[buffer(2)]],
                                device const TfDeltaKeep* keeps [[buffer(3)]],
                                device const uint* windows [[buffer(4)]],
                                constant TfDeltaParams& p [[buffer(5)]],
                                device bfloat* next [[buffer(6)]],
                                uint3 position [[thread_position_in_grid]]) {
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv;
  const uint width = (p.taps - 1) * channels, item = position.x;
  const TfDeltaKeep keep = keeps[position.y];
  if (item >= width) return;
  if (keep.rows == 0) {
    next[keep.next_slot * width + item] = committed[keep.state_slot * width + item];
    return;
  }
  const uint row = kept_rows[keep.first + keep.rows - 1];
  const uint source = windows[row * p.taps + item / channels + 1];
  next[keep.next_slot * width + item] = source < p.taps - 1 ? committed[(keep.state_slot * (p.taps - 1) + source) * channels + item % channels] : projection[(source - (p.taps - 1)) * channels + item % channels];
}
kernel void tf_delta_gdn_conv_keep(device ushort* committed [[buffer(0)]],
                              device const ushort* projection [[buffer(1)]],
                              device const uint* kept_rows [[buffer(2)]],
                              device const TfDeltaKeep* keeps [[buffer(3)]],
                              device const uint* windows [[buffer(4)]],
                              constant TfDeltaParams& p [[buffer(5)]],
                              uint2 position [[thread_position_in_grid]]) {
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv, item = position.x, held = p.taps - 1;
  const TfDeltaKeep keep = keeps[position.y];
  if (item >= channels || keep.rows == 0) return;
  const uint row = kept_rows[keep.first + keep.rows - 1];
  device ushort* window = committed + (size_t)keep.state_slot * held * channels + item;
  ushort taken[7];
  for (uint t = 0; t < held; ++t) {
    const uint source = windows[row * p.taps + t + 1];
    taken[t] = source < held ? window[source * channels] : projection[(source - held) * channels + item];
  }
  for (uint t = 0; t < held; ++t) window[t * channels] = taken[t];
}
// Replay finishes before publication, which copies storage words without another floating-point operation.

kernel void tf_delta_gdn_publish(device const uint* next_state [[buffer(0)]],
                            device const ushort* next_conv [[buffer(1)]],
                            device const TfDeltaKeep* keeps [[buffer(2)]],
                            constant TfDeltaParams& p [[buffer(3)]],
                            device uint* state [[buffer(4)]],
                            device ushort* conv [[buffer(5)]],
                            uint2 at [[thread_position_in_grid]]) {
  const uint state_width = p.nv * p.dv * p.dk;
  const uint conv_width = (p.taps - 1) * (2 * p.nk * p.dk + p.nv * p.dv);
  const uint slot = keeps[at.y].next_slot;
  if (at.x < state_width) state[slot * state_width + at.x] = next_state[slot * state_width + at.x];
  if (at.x < conv_width) conv[slot * conv_width + at.x] = next_conv[slot * conv_width + at.x];
}

// The conv window alone, for commits whose replay wrote the state in place.
kernel void tf_delta_gdn_publish_conv(device const ushort* next_conv [[buffer(0)]],
                                 device const TfDeltaKeep* keeps [[buffer(1)]],
                                 constant TfDeltaParams& p [[buffer(2)]],
                                 device ushort* conv [[buffer(3)]],
                                 uint2 at [[thread_position_in_grid]]) {
  const uint conv_width = (p.taps - 1) * (2 * p.nk * p.dk + p.nv * p.dv);
  const uint slot = keeps[at.y].next_slot;
  if (at.x < conv_width) conv[slot * conv_width + at.x] = next_conv[slot * conv_width + at.x];
}

kernel void tf_delta_gdn_clear(device uint* state [[buffer(0)]],
                          device ushort* conv [[buffer(1)]],
                          constant TfDeltaParams& p [[buffer(2)]],
                          constant uint& slot [[buffer(3)]],
                          uint2 at [[thread_position_in_grid]]) {
  const uint state_width = p.nv * p.dv * p.dk;
  const uint conv_width = (p.taps - 1) * (2 * p.nk * p.dk + p.nv * p.dv);
  const ulong row = ulong(at.y) * p.slots + slot; // the whole pool's words pass 32 bits past 113 slots of a 27B
  if (at.x < state_width) state[row * state_width + at.x] = 0u;
  if (at.x < conv_width) conv[row * conv_width + at.x] = 0u;
}