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
// tf_delta_tree_node2 with row's output written only when `write` (a star's other simdgroups repeat row 0's step).
inline void tf_delta_star_node2(thread float4& s0, thread float4& s1, device const bfloat* query, device const bfloat* key,
                                device const bfloat* value, device const float* decay, device const bfloat* mixing,
                                constant TfDeltaParams& p, device bfloat* output, uint row, uint head, uint key_head,
                                uint dimension, uint lane, bool write) {
  float a[4] = {s0[0], s0[1], s0[2], s0[3]};
  float b[4] = {s1[0], s1[1], s1[2], s1[3]};
  const uint at = row * p.nv + head;
  device const bfloat* k = key + (row * p.nk + key_head) * p.dk;
  device const bfloat* q = query + (row * p.nk + key_head) * p.dk;
  tf_delta_delta_update(a, k, decay[at], float(value[at * p.dv + dimension]), float(mixing[at]), lane, 4);
  tf_delta_delta_update(b, k, decay[at], float(value[at * p.dv + dimension + 1]), float(mixing[at]), lane, 4);
  const float ra = tf_delta_delta_read(a, q, lane, 4);
  const float rb = tf_delta_delta_read(b, q, lane, 4);
  if (write && lane == 0) {
    output[at * p.dv + dimension] = bfloat(ra);
    output[at * p.dv + dimension + 1] = bfloat(rb);
  }
  s0 = float4(a[0], a[1], a[2], a[3]);
  s1 = float4(b[0], b[1], b[2], b[3]);
}
// A depth-one star (row 0 the root, every later row its child) for two adjacent value columns, two simdgroups a pair:
// each takes row 0's step from the committed state, then its own children (rows 1 + sg, 3 + sg, ...) from row 0's
// state. Every step is tree_pair's update and read in the same lanes and order; simdgroup 0 alone writes row 0's output.
kernel void tf_delta_gdn_tree_star(device const bfloat* query [[buffer(0)]],
                         device const bfloat* key [[buffer(1)]],
                         device const bfloat* value [[buffer(2)]],
                         device const float* decay [[buffer(3)]],
                         device const bfloat* mixing [[buffer(4)]],
                         device const float* committed [[buffer(5)]],
                         device const int* parents [[buffer(6)]],
                         device const TfDeltaSegment* segments [[buffer(7)]],
                         constant TfDeltaParams& p [[buffer(8)]],
                         device bfloat* output [[buffer(9)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = group.z / p.nv, head = group.z % p.nv;
  const uint dimension = group.y * 2, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * 4; // the second column's words are p.dk on
  float4 root0, root1;
  for (uint part = 0; part < 4; ++part) {
    root0[part] = committed[span.state_slot * state_stride + column + part];
    root1[part] = committed[span.state_slot * state_stride + column + p.dk + part];
  }
  tf_delta_star_node2(root0, root1, query, key, value, decay, mixing, p, output, span.first, head, key_head, dimension, lane, sg == 0);
  for (uint child = 1 + sg; child < span.rows; child += 2) {
    float4 s0 = root0, s1 = root1;
    tf_delta_star_node2(s0, s1, query, key, value, decay, mixing, p, output, span.first + child, head, key_head, dimension, lane, true);
  }
}
#define TF_TREE_SLOTS(N) template [[host_name("tf_delta_gdn_tree_slots_" #N)]] kernel void tf_delta_gdn_tree_slots<N>(device const bfloat*, device const bfloat*, device const bfloat*, device const float*, device const bfloat*, device const float*, device const int*, device const TfDeltaSegment*, constant TfDeltaParams&, device bfloat*, device float*, device const TfDeltaSlotStep*, device float*, device float*, uint3, uint);
TF_TREE_SLOTS(1)
TF_TREE_SLOTS(2)
TF_TREE_SLOTS(4)
TF_TREE_SLOTS(8)
TF_TREE_SLOTS(16)
