// Slot steps preserve parent-state arithmetic while visiting the largest child last and recycling dead ancestors.
struct TfDeltaSlotStep { uint row; int read_slot; int write_slot; };
// SLOTS register slots: a chain needs one, a tree its peak of live branch states; fewer slots, fewer selects a node.
template <uint SLOTS>
kernel void tf_delta_gdn_tree_slots(device const bfloat* query [[buffer(0)]],
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
                         device const TfDeltaSlotStep* steps [[buffer(11)]],
                         device float* advanced [[buffer(12)]],
                         device float* before [[buffer(13)]],
                         uint3 position [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * 4;
  float4 root;
  for (uint part = 0; part < 4; ++part) root[part] = committed[span.state_slot * state_stride + column + part];
  float4 kept[SLOTS];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0; j < SLOTS; ++j) kept[j] = 0.0f;
  float4 last = root;
  for (uint at = 0; at < span.rows; ++at) {
    const TfDeltaSlotStep step = steps[span.first + at];
    float4 s = root;
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < SLOTS; ++j) if (int(j) == step.read_slot) s = kept[j];
    s = tf_delta_tree_node(s, query, key, value, decay, mixing, p, output, step.row, head, key_head, dimension, lane);
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < SLOTS; ++j) if (int(j) == step.write_slot) kept[j] = s;
    last = s;
  }
  // `advanced` aliases `committed`: each thread rewrites only the words it read
  if (p.advance != 0 && span.rows == 1)
    for (uint part = 0; part < 4; ++part) {
      const uint at = span.state_slot * state_stride + column + part;
      before[at] = root[part];
      advanced[at] = last[part];
    }
}
// One live state (a chain) for two adjacent value columns a simdgroup: each column's update and read are tree_slots<1>'s,
// in the same lanes and order; the row's key, query, decay and mixing serve both columns.
inline void tf_delta_tree_node2(thread float4& s0, thread float4& s1, device const bfloat* query, device const bfloat* key,
                                device const bfloat* value, device const float* decay, device const bfloat* mixing,
                                constant TfDeltaParams& p, device bfloat* output, uint row, uint head, uint key_head,
                                uint dimension, uint lane) {
  float a[4] = {s0[0], s0[1], s0[2], s0[3]};
  float b[4] = {s1[0], s1[1], s1[2], s1[3]};
  const uint at = row * p.nv + head;
  device const bfloat* k = key + (row * p.nk + key_head) * p.dk;
  device const bfloat* q = query + (row * p.nk + key_head) * p.dk;
  tf_delta_delta_update(a, k, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, 4);
  tf_delta_delta_update(b, k, decay[at], float(value[at * p.dv + dimension + 1]), float(mixing[at]), lane, 4);
  const float ra = tf_delta_delta_read(a, q, lane, 4);
  const float rb = tf_delta_delta_read(b, q, lane, 4);
  if (lane == 0) {
    output[at * p.dv + dimension] = bfloat(ra);
    output[at * p.dv + dimension + 1] = bfloat(rb);
  }
  s0 = float4(a[0], a[1], a[2], a[3]);
  s1 = float4(b[0], b[1], b[2], b[3]);
}
kernel void tf_delta_gdn_tree_pair(device const bfloat* query [[buffer(0)]],
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
                         device const TfDeltaSlotStep* steps [[buffer(11)]],
                         device float* advanced [[buffer(12)]],
                         device float* before [[buffer(13)]],
                         uint3 position [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y * 2, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * 4; // the second column's words are p.dk on
  float4 root0, root1;
  for (uint part = 0; part < 4; ++part) {
    root0[part] = committed[span.state_slot * state_stride + column + part];
    root1[part] = committed[span.state_slot * state_stride + column + p.dk + part];
  }
  float4 kept0 = 0.0f, kept1 = 0.0f;
  float4 last0 = root0, last1 = root1;
  for (uint at = 0; at < span.rows; ++at) {
    const TfDeltaSlotStep step = steps[span.first + at];
    float4 s0 = root0, s1 = root1;
    if (step.read_slot == 0) {
      s0 = kept0;
      s1 = kept1;
    }
    tf_delta_tree_node2(s0, s1, query, key, value, decay, mixing, p, output, step.row, head, key_head, dimension, lane);
    if (step.write_slot == 0) {
      kept0 = s0;
      kept1 = s1;
    }
    last0 = s0;
    last1 = s1;
  }
  // `advanced` aliases `committed`: each thread rewrites only the words it read
  if (p.advance != 0 && span.rows == 1)
    for (uint part = 0; part < 4; ++part) {
      const uint at = span.state_slot * state_stride + column + part;
      before[at] = root0[part];
      advanced[at] = last0[part];
      before[at + p.dk] = root1[part];
      advanced[at + p.dk] = last1[part];
    }
}

// tf_delta_gdn_pre_rows4 folded into tf_delta_gdn_tree_pair for a verify's rows: every lane computes its own four key
// channels' query and key (conv over the row's taps, SiLU, the 32-lane l2 norm), its SIMD group's two value columns and
// the row's decay and mixing, with tf_delta_gdn_pre_rows4's expressions and rounding points verbatim, then scans the row
// exactly as tf_delta_gdn_tree_pair does. The keep's replay still reads key, value, decay and mixing: one SIMD group a
// key head writes the keys, every SIMD group its own two values, one SIMD group a value head the decay and mixing.
inline void tf_fused_update(thread float* state, thread const float* kf, float forget, float incoming, float mix) {
  float remembered = 0.0f;
  for (uint part = 0; part < 4; ++part) {
    state[part] = state[part] * forget;
    remembered += state[part] * kf[part];
  }
  remembered = simd_sum(remembered);
  const float correction = (incoming - remembered) * mix;
  for (uint part = 0; part < 4; ++part) state[part] = state[part] + kf[part] * correction;
}
inline float tf_fused_read(thread const float* state, thread const float* qf) {
  float result = 0.0f;
  for (uint part = 0; part < 4; ++part) result += state[part] * qf[part];
  return simd_sum(result);
}
kernel void tf_delta_gdn_tree_pair_fused(device const bfloat* query [[buffer(0)]],
                         device bfloat* key [[buffer(1)]],
                         device bfloat* value [[buffer(2)]],
                         device float* decay [[buffer(3)]],
                         device bfloat* mixing [[buffer(4)]],
                         device const float* committed [[buffer(5)]],
                         device const int* parents [[buffer(6)]],
                         device const TfDeltaSegment* segments [[buffer(7)]],
                         constant TfDeltaParams& p [[buffer(8)]],
                         device bfloat* output [[buffer(9)]],
                         device float* snapshots [[buffer(10)]],
                         device const TfDeltaSlotStep* steps [[buffer(11)]],
                         device float* advanced [[buffer(12)]],
                         device float* before [[buffer(13)]],
                         device const bfloat* projection [[buffer(14)]],
                         device const bfloat* a_in [[buffer(15)]],
                         device const bfloat* b_in [[buffer(16)]],
                         device const bfloat* history [[buffer(17)]],
                         device const void* weights [[buffer(18)]],
                         device const void* decay_weight [[buffer(19)]],
                         device const void* time_bias [[buffer(20)]],
                         device const uint* windows [[buffer(21)]],
                         device const uint* row_slot [[buffer(22)]],
                         uint3 position [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y * 2, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * 4; // the second column's words are p.dk on
  float4 root0, root1;
  for (uint part = 0; part < 4; ++part) {
    root0[part] = committed[span.state_slot * state_stride + column + part];
    root1[part] = committed[span.state_slot * state_stride + column + p.dk + part];
  }
  const uint channels = 2 * p.nk * p.dk + p.nv * p.dv, keep = p.taps - 1;
  const uint qc = key_head * p.dk + lane * 4, kc = (p.nk + key_head) * p.dk + lane * 4;
  const uint vc = 2 * p.nk * p.dk + head * p.dv + dimension;
  float wq[4][4], wk[4][4], wv[2][4];
  for (uint part = 0; part < 4; ++part)
    for (uint tap = 0; tap < 4; ++tap) {
      wq[part][tap] = tf_delta_weight(weights, (qc + part) * p.taps + tap, p.weight_flags, 0);
      wk[part][tap] = tf_delta_weight(weights, (kc + part) * p.taps + tap, p.weight_flags, 0);
    }
  for (uint c = 0; c < 2; ++c)
    for (uint tap = 0; tap < 4; ++tap) wv[c][tap] = tf_delta_weight(weights, (vc + c) * p.taps + tap, p.weight_flags, 0);
  const bool key_writer = head % (p.nv / p.nk) == 0 && position.y == 0, head_writer = position.y == 0;
  const float scale_q = float(bfloat(1.0f / float(p.dk))), scale_k = float(bfloat(metal::rsqrt(float(p.dk))));
  float4 kept0 = 0.0f, kept1 = 0.0f;
  float4 last0 = root0, last1 = root1;
  for (uint at = 0; at < span.rows; ++at) {
    const TfDeltaSlotStep step = steps[span.first + at];
    const uint row = step.row;
    // tf_delta_gdn_pre_rows4 for this lane's channels of the row
    float rq[4] = {0.0f, 0.0f, 0.0f, 0.0f}, rk[4] = {0.0f, 0.0f, 0.0f, 0.0f}, rv[2] = {0.0f, 0.0f};
    for (uint tap = 0; tap < 4; ++tap) {
      const uint source = windows[row * p.taps + tap];
      device const bfloat* base = source < keep ? history + (row_slot[row] * keep + source) * channels : projection + (source - keep) * channels;
      const bfloat4 iq = *(device const bfloat4*)(base + qc), ik = *(device const bfloat4*)(base + kc);
      for (uint part = 0; part < 4; ++part) {
        rq[part] += wq[part][tap] * float(iq[part]);
        rk[part] += wk[part][tap] * float(ik[part]);
      }
      for (uint c = 0; c < 2; ++c) rv[c] += wv[c][tap] * float(base[vc + c]);
    }
    float aq[4], ak[4], av[2];
    for (uint part = 0; part < 4; ++part) {
      const float roundedq = float(bfloat(rq[part]));
      const float sigmoidq = float(bfloat(1.0f / (1.0f + metal::exp(-roundedq))));
      aq[part] = float(bfloat(roundedq * sigmoidq));
      const float roundedk = float(bfloat(rk[part]));
      const float sigmoidk = float(bfloat(1.0f / (1.0f + metal::exp(-roundedk))));
      ak[part] = float(bfloat(roundedk * sigmoidk));
    }
    for (uint c = 0; c < 2; ++c) {
      const float roundedv = float(bfloat(rv[c]));
      const float sigmoidv = float(bfloat(1.0f / (1.0f + metal::exp(-roundedv))));
      av[c] = float(bfloat(roundedv * sigmoidv));
    }
    float squareq = 0.0f, squarek = 0.0f;
    for (uint part = 0; part < 4; ++part) squareq += aq[part] * aq[part];
    for (uint part = 0; part < 4; ++part) squarek += ak[part] * ak[part];
    squareq = simd_sum(squareq);
    squarek = simd_sum(squarek);
    const float inverseq = metal::rsqrt(squareq / float(p.dk) + 1e-6f), inversek = metal::rsqrt(squarek / float(p.dk) + 1e-6f);
    float qf[4], kf[4];
    bfloat4 kn;
    for (uint part = 0; part < 4; ++part) {
      qf[part] = float(bfloat(scale_q * float(bfloat(aq[part] * inverseq))));
      kn[part] = bfloat(scale_k * float(bfloat(ak[part] * inversek)));
      kf[part] = float(kn[part]);
    }
    const bfloat v0 = bfloat(av[0]), v1 = bfloat(av[1]);
    const uint row_at = row * p.nv + head, gate_at = p.zba_stride ? row * p.zba_stride + head : row_at;
    const float shifted = float(bfloat(float(a_in[gate_at]) + tf_delta_weight(time_bias, head, p.weight_flags, 2)));
    const float softplus = float(bfloat(metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted)))));
    const float forget = metal::exp(-metal::exp(tf_delta_weight(decay_weight, head, p.weight_flags, 1)) * softplus);
    const bfloat mixb = bfloat(1.0f / (1.0f + metal::exp(-float(b_in[gate_at]))));
    // what the keep's replay reads, written once
    if (key_writer) *(device bfloat4*)(key + (row * p.nk + key_head) * p.dk + lane * 4) = kn;
    if (lane == 0) {
      value[row_at * p.dv + dimension] = v0;
      value[row_at * p.dv + dimension + 1] = v1;
      if (head_writer) {
        decay[row_at] = forget;
        mixing[row_at] = mixb;
      }
    }
    // tf_delta_gdn_tree_pair's step
    float4 s0 = root0, s1 = root1;
    if (step.read_slot == 0) {
      s0 = kept0;
      s1 = kept1;
    }
    float sa[4] = {s0[0], s0[1], s0[2], s0[3]};
    float sb[4] = {s1[0], s1[1], s1[2], s1[3]};
    tf_fused_update(sa, kf, forget, float(v0), float(mixb));
    tf_fused_update(sb, kf, forget, float(v1), float(mixb));
    const float ra = tf_fused_read(sa, qf);
    const float rb = tf_fused_read(sb, qf);
    if (lane == 0) {
      output[row_at * p.dv + dimension] = bfloat(ra);
      output[row_at * p.dv + dimension + 1] = bfloat(rb);
    }
    s0 = float4(sa[0], sa[1], sa[2], sa[3]);
    s1 = float4(sb[0], sb[1], sb[2], sb[3]);
    if (step.write_slot == 0) {
      kept0 = s0;
      kept1 = s1;
    }
    last0 = s0;
    last1 = s1;
  }
  if (p.advance != 0 && span.rows == 1)
    for (uint part = 0; part < 4; ++part) {
      const uint at = span.state_slot * state_stride + column + part;
      before[at] = root0[part];
      advanced[at] = last0[part];
      before[at + p.dk] = root1[part];
      advanced[at + p.dk] = last1[part];
    }
}
#define TF_TREE_SLOTS(N) template [[host_name("tf_delta_gdn_tree_slots_" #N)]] kernel void tf_delta_gdn_tree_slots<N>(device const bfloat*, device const bfloat*, device const bfloat*, device const float*, device const bfloat*, device const float*, device const int*, device const TfDeltaSegment*, constant TfDeltaParams&, device bfloat*, device float*, device const TfDeltaSlotStep*, device float*, device float*, uint3, uint);
TF_TREE_SLOTS(1)
TF_TREE_SLOTS(2)
TF_TREE_SLOTS(4)
TF_TREE_SLOTS(8)
TF_TREE_SLOTS(16)
