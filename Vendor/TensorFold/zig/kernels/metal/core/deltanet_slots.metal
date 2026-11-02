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
inline void tf_delta_delta_update_words2(thread float* a, thread float* b, thread const float* key, float forget,
                                         float incoming0, float incoming1, float mix) {
  float remembered0 = 0.0f, remembered1 = 0.0f;
  for (uint part = 0; part < 4; ++part) {
    a[part] = a[part] * forget;
    remembered0 += a[part] * key[part];
  }
  for (uint part = 0; part < 4; ++part) {
    b[part] = b[part] * forget;
    remembered1 += b[part] * key[part];
  }
  const float2 remembered = simd_sum(float2(remembered0, remembered1));
  const float correction0 = (incoming0 - remembered.x) * mix;
  const float correction1 = (incoming1 - remembered.y) * mix;
  for (uint part = 0; part < 4; ++part) a[part] = a[part] + key[part] * correction0;
  for (uint part = 0; part < 4; ++part) b[part] = b[part] + key[part] * correction1;
}
inline float2 tf_delta_delta_read_words2(thread const float* a, thread const float* b, thread const float* query) {
  float result0 = 0.0f, result1 = 0.0f;
  for (uint part = 0; part < 4; ++part) result0 += a[part] * query[part];
  for (uint part = 0; part < 4; ++part) result1 += b[part] * query[part];
  return simd_sum(float2(result0, result1));
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
  tf_delta_delta_update_words2(a, b, kf, forget, incoming0, incoming1, mix);
  const float2 rab = tf_delta_delta_read_words2(a, b, qf);
  const float ra = rab.x, rb = rab.y;
  if (lane == 0) {
    output[at * p.dv + dimension] = bfloat(ra);
    output[at * p.dv + dimension + 1] = bfloat(rb);
  }
  s0 = float4(a[0], a[1], a[2], a[3]);
  s1 = float4(b[0], b[1], b[2], b[3]);
}
template <uint W>
inline void tf_delta_star_children(float4 r0, float4 r1, device const bfloat* query, device const bfloat* key,
                                   device const bfloat* value, device const float* decay, device const bfloat* mixing,
                                   constant TfDeltaParams& p, device bfloat* output, uint row, uint head, uint key_head,
                                   uint dimension, uint lane) {
  float kf[W][4], qf[W][4], forget[W], mix[W], incoming0[W], incoming1[W];
  uint at[W];
  _Pragma("clang loop unroll(full)") for (uint w = 0; w < W; ++w) {
    at[w] = (row + w) * p.nv + head;
    device const bfloat* k = key + ((row + w) * p.nk + key_head) * p.dk + lane * 4;
    device const bfloat* q = query + ((row + w) * p.nk + key_head) * p.dk + lane * 4;
    for (uint part = 0; part < 4; ++part) {
      kf[w][part] = float(k[part]);
      qf[w][part] = float(q[part]);
    }
    forget[w] = decay[at[w]];
    mix[w] = float(mixing[at[w]]);
    incoming0[w] = float(value[at[w] * p.dv + dimension]);
    incoming1[w] = float(value[at[w] * p.dv + dimension + 1]);
  }
  float ra[W], rb[W];
  _Pragma("clang loop unroll(full)") for (uint w = 0; w < W; ++w) {
    float a[4] = {r0[0], r0[1], r0[2], r0[3]};
    float b[4] = {r1[0], r1[1], r1[2], r1[3]};
    tf_delta_delta_update_words2(a, b, kf[w], forget[w], incoming0[w], incoming1[w], mix[w]);
    const float2 rab = tf_delta_delta_read_words2(a, b, qf[w]);
    ra[w] = rab.x;
    rb[w] = rab.y;
  }
  if (lane == 0)
    _Pragma("clang loop unroll(full)") for (uint w = 0; w < W; ++w) {
      output[at[w] * p.dv + dimension] = bfloat(ra[w]);
      output[at[w] * p.dv + dimension + 1] = bfloat(rb[w]);
    }
}
inline void tf_delta_star_leaves(float4 r0, float4 r1, device const bfloat* query, device const bfloat* key,
                                 device const bfloat* value, device const float* decay, device const bfloat* mixing,
                                 constant TfDeltaParams& p, device bfloat* output, uint first, uint rows, uint head,
                                 uint key_head, uint dimension, uint lane) {
  uint c = 1;
  for (; c + 3 <= rows; c += 3) tf_delta_star_children<3>(r0, r1, query, key, value, decay, mixing, p, output, first + c, head, key_head, dimension, lane);
  if (c + 2 == rows) tf_delta_star_children<2>(r0, r1, query, key, value, decay, mixing, p, output, first + c, head, key_head, dimension, lane);
  else if (c + 1 == rows) tf_delta_star_children<1>(r0, r1, query, key, value, decay, mixing, p, output, first + c, head, key_head, dimension, lane);
}
inline void tf_delta_gdn_star_wait(device atomic_uint* PC, uint wait_prep, uint3 groups, uint t) {
  if (wait_prep == 0u) return;
  if (t == 0) {
    for (uint spin = 0; spin < (1u << 24) && atomic_load_explicit(&PC[0], memory_order_relaxed) < wait_prep; spin++) {}
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const uint total = groups.x * groups.y * groups.z;
    if (atomic_fetch_add_explicit(&PC[1], 1u, memory_order_relaxed) + 1u == total) {
      atomic_store_explicit(&PC[0], 0u, memory_order_relaxed);
      atomic_store_explicit(&PC[1], 0u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// A depth-one star (each segment's node 0, then its other rows as children of node 0) for two adjacent value columns a
// simdgroup: every node of each column is tf_delta_tree_node's arithmetic from the same parent state as in
// tf_delta_gdn_tree_slots (tf_delta_tree_node_pair, a row's loads shared by the two columns), so each column's arithmetic
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
                                   device atomic_uint* PC [[buffer(17)]],
                                   constant uint& wait_prep [[buffer(18)]],
                                   uint3 position [[thread_position_in_grid]],
                                   uint3 groups [[threadgroups_per_grid]],
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
  tf_delta_gdn_star_wait(PC, wait_prep, groups, t);
  tf_delta_tree_node_pair(r0, r1, query, key, value, decay, mixing, p, output, span.first, head, key_head, d0, lane);
  tf_delta_star_leaves(r0, r1, query, key, value, decay, mixing, p, output, span.first, span.rows, head, key_head, d0, lane);
  // count_out above 0: the gate that reads `output` shares this launch's barrier interval and waits on SC[0]
  if (count_out != 0u) {
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(&SC[0], 1u, memory_order_relaxed);
    }
  }
}
// A deferred replay's rows for the star (tf_delta_gdn_stash's, in keep order) and whether it writes the state back.
struct TfDeltaPending { uint rows, persist; };
template <uint N>
inline void tf_delta_star_pending_rows(thread float* a, thread float* b, device const bfloat* key, device const bfloat* value,
                                       device const float* decay, device const bfloat* mixing, constant TfDeltaParams& p,
                                       uint first, uint head, uint key_head, uint dimension, uint lane) {
  float kf[N][4], forget[N], mix[N], incoming0[N], incoming1[N];
  _Pragma("clang loop unroll(full)") for (uint i = 0; i < N; ++i) {
    const uint at = (first + i) * p.nv + head;
    device const bfloat* k = key + ((first + i) * p.nk + key_head) * p.dk + lane * 4;
    for (uint part = 0; part < 4; ++part) kf[i][part] = float(k[part]);
    forget[i] = decay[at];
    mix[i] = float(mixing[at]);
    incoming0[i] = float(value[at * p.dv + dimension]);
    incoming1[i] = float(value[at * p.dv + dimension + 1]);
  }
  _Pragma("clang loop unroll(full)") for (uint i = 0; i < N; ++i) {
    tf_delta_delta_update_words2(a, b, kf[i], forget[i], incoming0[i], incoming1[i], mix[i]);
  }
}
// tf_delta_gdn_star_pair from the state the deferred keeps' replays would have left: before node 0 each column's lane
// pending.persist writes them back over the committed words each lane read (its simdgroup's own two columns), before
// node 0 and so before the threadgroup counts itself.
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
                                      device atomic_uint* PC [[buffer(17)]],
                                      constant uint& wait_prep [[buffer(18)]],
                                      uint3 position [[thread_position_in_grid]],
                                      uint3 groups [[threadgroups_per_grid]],
                                      uint t [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint d0 = 2 * position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint column = (head * p.dv + d0) * p.dk + lane * 4, base = span.state_slot * p.nv * p.dv * p.dk;
  float a[4], b[4]; // columns d0 and d0 + 1, as tf_delta_gdn_replay_body<4> holds a column's state
  for (uint part = 0; part < 4; ++part) {
    a[part] = committed[base + column + part];
    b[part] = committed[base + column + p.dk + part];
  }
  uint row = 0;
  for (; row + 4 <= pending.rows; row += 4) tf_delta_star_pending_rows<4>(a, b, pending_key, pending_value, pending_decay, pending_mixing, p, row, head, key_head, d0, lane);
  if (row + 3 == pending.rows) tf_delta_star_pending_rows<3>(a, b, pending_key, pending_value, pending_decay, pending_mixing, p, row, head, key_head, d0, lane);
  else if (row + 2 == pending.rows) tf_delta_star_pending_rows<2>(a, b, pending_key, pending_value, pending_decay, pending_mixing, p, row, head, key_head, d0, lane);
  else if (row + 1 == pending.rows) tf_delta_star_pending_rows<1>(a, b, pending_key, pending_value, pending_decay, pending_mixing, p, row, head, key_head, d0, lane);
  if (pending.persist != 0u)
    for (uint part = 0; part < 4; ++part) {
      committed[base + column + part] = a[part];
      committed[base + column + p.dk + part] = b[part];
    }
  float4 r0 = float4(a[0], a[1], a[2], a[3]), r1 = float4(b[0], b[1], b[2], b[3]);
  tf_delta_gdn_star_wait(PC, wait_prep, groups, t);
  tf_delta_tree_node_pair(r0, r1, query, key, value, decay, mixing, p, output, span.first, head, key_head, d0, lane);
  tf_delta_star_leaves(r0, r1, query, key, value, decay, mixing, p, output, span.first, span.rows, head, key_head, d0, lane);
  // count_out above 0: the gate that reads `output` shares this launch's barrier interval and waits on SC[0]
  if (count_out != 0u) {
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(&SC[0], 1u, memory_order_relaxed);
    }
  }
}
