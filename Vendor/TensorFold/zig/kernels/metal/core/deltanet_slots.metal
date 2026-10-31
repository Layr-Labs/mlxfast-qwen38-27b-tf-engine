// Slot steps preserve parent-state arithmetic while visiting the largest child last and recycling dead ancestors.
struct TfDeltaSlotStep { uint row; int read_slot; int write_slot; };

// Depth-one rounds' kept rows wait in a layer's stash (their key, value, decay and mixing words, as each round's prepare
// wrote them) until a later verify's state load applies them, instead of a replay behind every round. Round r of the
// TF_FOLD_ROUNDS since the state was last written holds rows 2r and 2r + 1 and its count of kept rows (at most two).
// Layout: key [2F][nk * dk] bf16, value [2F][nv * dv] bf16, decay [2F][nv] f32, mixing [2F][nv] bf16, counts [F] u32.
constant constexpr uint TF_FOLD_ROUNDS = 4;
struct TfDeltaStash {
  device const bfloat* key;
  device const bfloat* value;
  device const float* decay;
  device const bfloat* mixing;
  device const uint* count;
};
inline TfDeltaStash tf_delta_stash(device const uchar* base, constant TfDeltaParams& p) {
  const uint rows = 2 * TF_FOLD_ROUNDS;
  TfDeltaStash s;
  s.key = (device const bfloat*)base;
  s.value = (device const bfloat*)(base + rows * p.nk * p.dk * 2);
  s.decay = (device const float*)(base + rows * (p.nk * p.dk + p.nv * p.dv) * 2);
  s.mixing = (device const bfloat*)(base + rows * (p.nk * p.dk + p.nv * p.dv) * 2 + rows * p.nv * 4);
  s.count = (device const uint*)(base + rows * (p.nk * p.dk + p.nv * p.dv) * 2 + rows * p.nv * 6);
  return s;
}
// The first `rounds` stashed rounds' rows applied to one column's state in commit order, by the replay's update and
// operands (tf_delta_gdn_replay_body): the state the replays behind those rounds would have left.
inline void tf_delta_unstash(thread float* state, TfDeltaStash s, constant TfDeltaParams& p, uint rounds, uint head, uint key_head, uint dimension, uint lane) {
  for (uint r = 0; r < rounds; ++r) {
    const uint kept = min(s.count[r], 2u);
    for (uint i = 0; i < kept; ++i) {
      const uint row = 2 * r + i, at = row * p.nv + head;
      tf_delta_delta_update(state, s.key + (row * p.nk + key_head) * p.dk, s.decay[at], float(s.value[at * p.dv + dimension]), float(s.mixing[at]), lane, 4);
    }
  }
}

// The keep's half of a depth-one round's commit: its kept rows' words into stash round `round` (one window).
kernel void tf_delta_gdn_stash(device const bfloat* key [[buffer(0)]],
                               device const bfloat* value [[buffer(1)]],
                               device const float* decay [[buffer(2)]],
                               device const bfloat* mixing [[buffer(3)]],
                               device const uint* kept_rows [[buffer(4)]],
                               device const TfDeltaKeep* keeps [[buffer(5)]],
                               constant TfDeltaParams& p [[buffer(6)]],
                               device uchar* stash [[buffer(7)]],
                               constant uint& round [[buffer(8)]],
                               uint2 position [[thread_position_in_grid]],
                               uint2 grid [[threads_per_grid]]) {
  const TfDeltaKeep keep = keeps[0];
  const uint i = position.y, kept = min(keep.rows, 2u), rows = 2 * TF_FOLD_ROUNDS, at = 2 * round + i;
  device bfloat* sk = (device bfloat*)stash;
  device bfloat* sv = (device bfloat*)(stash + rows * p.nk * p.dk * 2);
  device float* sd = (device float*)(stash + rows * (p.nk * p.dk + p.nv * p.dv) * 2);
  device bfloat* sm = (device bfloat*)(stash + rows * (p.nk * p.dk + p.nv * p.dv) * 2 + rows * p.nv * 4);
  device uint* sc = (device uint*)(stash + rows * (p.nk * p.dk + p.nv * p.dv) * 2 + rows * p.nv * 6);
  if (i == 0 && position.x == 0) sc[round] = kept;
  if (i >= kept) return;
  const uint row = kept_rows[keep.first + i];
  for (uint t = position.x; t < p.nk * p.dk; t += grid.x) sk[at * p.nk * p.dk + t] = key[row * p.nk * p.dk + t];
  for (uint t = position.x; t < p.nv * p.dv; t += grid.x) sv[at * p.nv * p.dv + t] = value[row * p.nv * p.dv + t];
  for (uint t = position.x; t < p.nv; t += grid.x) {
    sd[at * p.nv + t] = decay[row * p.nv + t];
    sm[at * p.nv + t] = mixing[row * p.nv + t];
  }
}

// A stash applied on its own (no verify takes it): the state of `slot` advanced in place, a column a simdgroup.
kernel void tf_delta_gdn_unstash(device float* state [[buffer(0)]],
                                 device const uchar* stash [[buffer(1)]],
                                 constant TfDeltaParams& p [[buffer(2)]],
                                 constant uint& slot [[buffer(3)]],
                                 constant uint& rounds [[buffer(4)]],
                                 uint3 position [[thread_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint head = position.z, dimension = position.y, key_head = head / (p.nv / p.nk);
  const uint column = slot * (p.nv * p.dv * p.dk) + (head * p.dv + dimension) * p.dk + lane * 4;
  float s[4] = {state[column], state[column + 1], state[column + 2], state[column + 3]};
  tf_delta_unstash(s, tf_delta_stash(stash, p), p, rounds, head, key_head, dimension, lane);
  for (uint part = 0; part < 4; ++part) state[column + part] = s[part];
}
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
                         device const uchar* stash [[buffer(14)]],
                         constant uint& fold [[buffer(15)]],
                         uint3 position [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * 4;
  float4 root;
  for (uint part = 0; part < 4; ++part) root[part] = committed[span.state_slot * state_stride + column + part];
  // earlier rounds' kept rows, stashed by their keeps: applied to the loaded state; fold & 256 writes it back at the end
  if ((fold & 255) != 0) {
    float s[4] = {root[0], root[1], root[2], root[3]};
    tf_delta_unstash(s, tf_delta_stash(stash, p), p, fold & 255, head, key_head, dimension, lane);
    root = float4(s[0], s[1], s[2], s[3]);
  }
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
  else if ((fold & 256) != 0)
    for (uint part = 0; part < 4; ++part) advanced[span.state_slot * state_stride + column + part] = root[part];
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
                         device const uchar* stash [[buffer(14)]],
                         constant uint& fold [[buffer(15)]],
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
  // earlier rounds' kept rows, stashed by their keeps: applied to the loaded state; fold & 256 writes it back at the end
  if ((fold & 255) != 0) {
    const TfDeltaStash st = tf_delta_stash(stash, p);
    float a[4] = {root0[0], root0[1], root0[2], root0[3]};
    float b[4] = {root1[0], root1[1], root1[2], root1[3]};
    tf_delta_unstash(a, st, p, fold & 255, head, key_head, dimension, lane);
    tf_delta_unstash(b, st, p, fold & 255, head, key_head, dimension + 1, lane);
    root0 = float4(a[0], a[1], a[2], a[3]);
    root1 = float4(b[0], b[1], b[2], b[3]);
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
  else if ((fold & 256) != 0)
    for (uint part = 0; part < 4; ++part) {
      advanced[span.state_slot * state_stride + column + part] = root0[part];
      advanced[span.state_slot * state_stride + column + p.dk + part] = root1[part];
    }
}
#define TF_TREE_SLOTS(N) template [[host_name("tf_delta_gdn_tree_slots_" #N)]] kernel void tf_delta_gdn_tree_slots<N>(device const bfloat*, device const bfloat*, device const bfloat*, device const float*, device const bfloat*, device const float*, device const int*, device const TfDeltaSegment*, constant TfDeltaParams&, device bfloat*, device float*, device const TfDeltaSlotStep*, device float*, device float*, device const uchar*, constant uint&, uint3, uint);
TF_TREE_SLOTS(1)
TF_TREE_SLOTS(2)
TF_TREE_SLOTS(4)
TF_TREE_SLOTS(8)
TF_TREE_SLOTS(16)
