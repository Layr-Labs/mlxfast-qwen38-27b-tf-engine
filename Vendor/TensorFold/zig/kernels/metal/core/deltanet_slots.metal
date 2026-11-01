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
#define TF_TREE_SLOTS(N) template [[host_name("tf_delta_gdn_tree_slots_" #N)]] kernel void tf_delta_gdn_tree_slots<N>(device const bfloat*, device const bfloat*, device const bfloat*, device const float*, device const bfloat*, device const float*, device const int*, device const TfDeltaSegment*, constant TfDeltaParams&, device bfloat*, device float*, device const TfDeltaSlotStep*, device float*, device float*, uint3, uint);
TF_TREE_SLOTS(1)
TF_TREE_SLOTS(2)
TF_TREE_SLOTS(4)
TF_TREE_SLOTS(8)
TF_TREE_SLOTS(16)
// tf_delta_delta_update and tf_delta_delta_read on a lane's key and query words already in registers: the same statements
// (so the same products, fused multiply-adds, in-order lane sums and simd_sum), with each float(key[...]) a register.
inline void tf_delta_delta_update_words(thread float* state, thread const float* key, float forget, float incoming,
                                        float mix) {
  float remembered = 0.0f;
  for (uint part = 0; part < 4; ++part) {
    state[part] = state[part] * forget;
    remembered += state[part] * key[part];
  }
  remembered = simd_sum(remembered);
  const float correction = (incoming - remembered) * mix;
  for (uint part = 0; part < 4; ++part) state[part] = state[part] + key[part] * correction;
}
inline float tf_delta_delta_read_words(thread const float* state, thread const float* query) {
  float result = 0.0f;
  for (uint part = 0; part < 4; ++part) result += state[part] * query[part];
  return simd_sum(result);
}
// tf_delta_tree_node for two adjacent value columns (dk = 128): the row's key and query words, decay, mixing and both
// value words are loaded and converted once, ahead of either column's output store (a device store the compiler must
// assume may alias them). Each column then runs tf_delta_tree_node's update and read in the same lanes and order.
inline void tf_delta_tree_node_pair(thread float4& s0, thread float4& s1, device const bfloat* query,
                                    device const bfloat* key, device const bfloat* value, device const float* decay,
                                    device const bfloat* mixing, constant TfDeltaParams& p, device bfloat* output,
                                    uint row, uint head, uint key_head, uint dimension, uint lane) {
  const uint at = row * p.nv + head;
  device const bfloat* k = key + (row * p.nk + key_head) * p.dk + lane * 4;
  device const bfloat* q = query + (row * p.nk + key_head) * p.dk + lane * 4;
  float kf[4], qf[4];
  for (uint part = 0; part < 4; ++part) {
    kf[part] = float(k[part]);
    qf[part] = float(q[part]);
  }
  const float forget = decay[at], mix = float(mixing[at]);
  const float incoming0 = float(value[at * p.dv + dimension]), incoming1 = float(value[at * p.dv + dimension + 1]);
  float a[4] = {s0[0], s0[1], s0[2], s0[3]};
  float b[4] = {s1[0], s1[1], s1[2], s1[3]};
  tf_delta_delta_update_words(a, kf, forget, incoming0, mix);
  tf_delta_delta_update_words(b, kf, forget, incoming1, mix);
  const float ra = tf_delta_delta_read_words(a, qf);
  const float rb = tf_delta_delta_read_words(b, qf);
  if (lane == 0) {
    output[at * p.dv + dimension] = bfloat(ra);
    output[at * p.dv + dimension + 1] = bfloat(rb);
  }
  s0 = float4(a[0], a[1], a[2], a[3]);
  s1 = float4(b[0], b[1], b[2], b[3]);
}
// A depth-one star (each segment's node 0, then its other rows as children of node 0) for two adjacent value columns a
// simdgroup: every node of each column is tf_delta_tree_node's arithmetic from the same parent state as in
// tf_delta_gdn_tree_slots (tf_delta_tree_node_pair, a row's loads shared by the two columns), so each column's arithmetic
// is that kernel's; the star needs no slot steps, no parent selects and no kept states.
// Verify rounds only (no advance): no state is written.
kernel void tf_delta_gdn_star_pair(device const bfloat* query [[buffer(0)]],
                                   device const bfloat* key [[buffer(1)]],
                                   device const bfloat* value [[buffer(2)]],
                                   device const float* decay [[buffer(3)]],
                                   device const bfloat* mixing [[buffer(4)]],
                                   device const float* committed [[buffer(5)]],
                                   device const TfDeltaSegment* segments [[buffer(7)]],
                                   constant TfDeltaParams& p [[buffer(8)]],
                                   device bfloat* output [[buffer(9)]],
                                   device atomic_uint* SC [[buffer(10)]],
                                   constant uint& count_out [[buffer(11)]],
                                   uint3 position [[thread_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint d0 = 2 * position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint column = (head * p.dv + d0) * p.dk + lane * 4, base = span.state_slot * p.nv * p.dv * p.dk;
  float4 r0, r1;
  for (uint part = 0; part < 4; ++part) {
    r0[part] = committed[base + column + part];
    r1[part] = committed[base + column + p.dk + part];
  }
  tf_delta_tree_node_pair(r0, r1, query, key, value, decay, mixing, p, output, span.first, head, key_head, d0, lane);
  for (uint c = 1; c < span.rows; ++c) {
    float4 c0 = r0, c1 = r1; // node 0's states
    tf_delta_tree_node_pair(c0, c1, query, key, value, decay, mixing, p, output, span.first + c, head, key_head, d0, lane);
  }
  if (count_out != 0u) {
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(&SC[0], 1u, memory_order_relaxed);
    }
  }
}
struct TfDeltaPending { uint rows, persist; };
kernel void tf_delta_gdn_star_pending(device const bfloat* query [[buffer(0)]],
                                      device const bfloat* key [[buffer(1)]],
                                      device const bfloat* value [[buffer(2)]],
                                      device const float* decay [[buffer(3)]],
                                      device const bfloat* mixing [[buffer(4)]],
                                      device float* committed [[buffer(5)]],
                                      device const TfDeltaSegment* segments [[buffer(7)]],
                                      constant TfDeltaParams& p [[buffer(8)]],
                                      device bfloat* output [[buffer(9)]],
                                      device atomic_uint* SC [[buffer(10)]],
                                      constant uint& count_out [[buffer(11)]],
                                      device const bfloat* pending_key [[buffer(12)]],
                                      device const bfloat* pending_value [[buffer(13)]],
                                      device const float* pending_decay [[buffer(14)]],
                                      device const bfloat* pending_mixing [[buffer(15)]],
                                      constant TfDeltaPending& pending [[buffer(16)]],
                                      uint3 position [[thread_position_in_grid]],
                                      uint t [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint d0 = 2 * position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint column = (head * p.dv + d0) * p.dk + lane * 4, base = span.state_slot * p.nv * p.dv * p.dk;
  float a[4], b[4];
  for (uint part = 0; part < 4; ++part) {
    a[part] = committed[base + column + part];
    b[part] = committed[base + column + p.dk + part];
  }
  for (uint row = 0; row < pending.rows; ++row) {
    const uint at = row * p.nv + head;
    device const bfloat* k = pending_key + (row * p.nk + key_head) * p.dk;
    tf_delta_delta_update(a, k, pending_decay[at], float(pending_value[at * p.dv + d0]), float(pending_mixing[at]), lane, 4);
    tf_delta_delta_update(b, k, pending_decay[at], float(pending_value[at * p.dv + d0 + 1]), float(pending_mixing[at]), lane, 4);
  }
  if (pending.persist != 0u)
    for (uint part = 0; part < 4; ++part) {
      committed[base + column + part] = a[part];
      committed[base + column + p.dk + part] = b[part];
    }
  float4 r0 = float4(a[0], a[1], a[2], a[3]), r1 = float4(b[0], b[1], b[2], b[3]);
  tf_delta_tree_node_pair(r0, r1, query, key, value, decay, mixing, p, output, span.first, head, key_head, d0, lane);
  for (uint c = 1; c < span.rows; ++c) {
    float4 c0 = r0, c1 = r1; // node 0's states
    tf_delta_tree_node_pair(c0, c1, query, key, value, decay, mixing, p, output, span.first + c, head, key_head, d0, lane);
  }
  if (count_out != 0u) {
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(&SC[0], 1u, memory_order_relaxed);
    }
  }
}
