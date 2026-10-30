// Slot steps preserve parent-state arithmetic while visiting the largest child last and recycling dead ancestors.
struct TfDeltaSlotStep { uint row; int read_slot; int write_slot; };
// A lazy keep's stash for one layer: byte offsets of its keeps, rows, key, value, decay and mixing (on = 0: none).
struct TfDeltaLazy { uint on, keeps, rows, key, value, decay, mixing; };
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
                         constant TfDeltaLazy& lazy [[buffer(15)]],
                         uint3 position [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint dimension = position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint column = (head * p.dv + dimension) * p.dk + lane * 4;
  float4 root;
  for (uint part = 0; part < 4; ++part) root[part] = committed[span.state_slot * state_stride + column + part];
  // lazy: the last round's kept rows, stashed by its keep, step the state here as tf_delta_gdn_replay would, stored in place
  if (lazy.on != 0) {
    device const TfDeltaKeep* lazy_keeps = (device const TfDeltaKeep*)(stash + lazy.keeps);
    device const uint* lazy_rows = (device const uint*)(stash + lazy.rows);
    device const bfloat* lazy_key = (device const bfloat*)(stash + lazy.key);
    device const bfloat* lazy_value = (device const bfloat*)(stash + lazy.value);
    device const float* lazy_decay = (device const float*)(stash + lazy.decay);
    device const bfloat* lazy_mixing = (device const bfloat*)(stash + lazy.mixing);
    const TfDeltaKeep keep = lazy_keeps[segment_index];
    float state[4];
    for (uint part = 0; part < 4; ++part) state[part] = committed[keep.state_slot * state_stride + column + part];
    for (uint step = 0; step < keep.rows; ++step) {
      const uint row = lazy_rows[keep.first + step], at = row * p.nv + head;
      tf_delta_delta_update(state, lazy_key + (row * p.nk + key_head) * p.dk, lazy_decay[at], float(lazy_value[at * p.dv + dimension]), float(lazy_mixing[at]), lane, 4);
    }
    for (uint part = 0; part < 4; ++part) advanced[keep.next_slot * state_stride + column + part] = state[part];
    root = float4(state[0], state[1], state[2], state[3]);
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
}
#define TF_TREE_SLOTS(N) template [[host_name("tf_delta_gdn_tree_slots_" #N)]] kernel void tf_delta_gdn_tree_slots<N>(device const bfloat*, device const bfloat*, device const bfloat*, device const float*, device const bfloat*, device const float*, device const int*, device const TfDeltaSegment*, constant TfDeltaParams&, device bfloat*, device float*, device const TfDeltaSlotStep*, device float*, device float*, device const uchar*, constant TfDeltaLazy&, uint3, uint);
TF_TREE_SLOTS(1)
TF_TREE_SLOTS(2)
TF_TREE_SLOTS(4)
TF_TREE_SLOTS(8)
TF_TREE_SLOTS(16)
