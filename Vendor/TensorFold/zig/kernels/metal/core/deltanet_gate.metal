// team-m5-08x1: a decoded star verify's GDN gate in its scan's epilogue. Built as a library of its own, this file after
// core_deltanet (deltanet.metal and deltanet_slots.metal), so the record's library keeps the record's text.
// tf_delta_gdn_star_pair_gate and tf_delta_gdn_star_pending_gate are tf_delta_gdn_star_pair and tf_delta_gdn_star_pending
// statement for statement up to their rows' stores. Where those count every threadgroup on SC[0] for a gate launch that
// waits on the count in the same barrier interval, these count each threadgroup on its head's word past the same
// device-scope fence, and the threadgroup that completes its head's count runs tf_delta_gdn_post_rows' statements on the
// head's rows: no threadgroup waits on another.
//
// team-m5-08x2: the waiters' arrivals off their critical path. A counted waiter's thread 0 passes the count and its fence,
// then adds its group's arrival to the waiter's second word, whose last arriver leaves both words zero for the next pair.
// The record takes that arrival (a fetch-and-add whose result it compares) before the group's barrier, so every group of
// the launch queues on one word before it reads a row. Here the arrival goes out after the barrier and its result is
// compared at the group's end: in these scans' wait on the prepare (tf_delta_gdn_star_wait_late, its reset in
// tf_delta_gdn_star_gate) and in tf_delta_gdn_pre_rows_held_late, the record's prepare with only that moved. The spins,
// their fences and every data statement are the record's; the words' last reset only comes later in the same launch.

// tf_delta_gdn_post_rows' statements for one (row, head), in that kernel's order: the gains, the row's recurrence and z
// words, the in-lane sum of squares and its simd_sum, the rounded normalized values, the gated outputs and, with
// sums_stride above 0, the out projection's group sums of the stored outputs.
inline void tf_delta_gdn_gate_row(device const bfloat* recurrence, device const bfloat* gate, device const void* norm,
                                  constant TfDeltaParams& p, device bfloat* output, device float* sums, uint sums_stride,
                                  uint row, uint head, uint lane) {
  float gain[4];
  for (uint part = 0; part < 4; ++part) gain[part] = tf_delta_weight(norm, lane * 4 + part, p.weight_flags, 3);
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
    const float normalized = float(bfloat(gain[part] * (values[part] * inverse)));
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

// tf_delta_gdn_star_wait without its arrival in front of the barrier: thread 0 waits on the prepare's count (bounded) and
// passes its fence, as there; after the barrier it adds the group's arrival on PC[1] and keeps the result, which
// tf_delta_gdn_star_gate compares once the group's rows are stored (0 when there is no wait, or for other threads).
inline uint tf_delta_gdn_star_wait_late(device atomic_uint* PC, uint wait_prep, uint t) {
  if (wait_prep == 0u) return 0u;
  if (t == 0) {
    for (uint spin = 0; spin < (1u << 24) && atomic_load_explicit(&PC[0], memory_order_relaxed) < wait_prep; spin++) {}
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t == 0 ? atomic_fetch_add_explicit(&PC[1], 1u, memory_order_relaxed) + 1u : 0u;
}

// A star threadgroup's epilogue. Once its rows are stored, thread 0 passes the device-scope fence that the record's
// counted scan passes and counts the group on its head's word, HC[segment * nv + head]. The group that completes the
// head's count (per_head groups, dv / 32) leaves the word zero for the next launch, passes the fence that the record's
// waiting gate passes, and runs the gate on its segment's rows of the head, a simdgroup a row. The other groups return.
// team-m5-08x2: before that, the group whose arrival completed the prepare wait's count (prep_arrived of prep_total, the
// launch's groups) leaves both of its words zero, as tf_delta_gdn_star_wait's last arriver does.
inline void tf_delta_gdn_star_gate(device const bfloat* recurrence, device const bfloat* gate, device const void* norm,
                                   constant TfDeltaParams& p, device bfloat* output, device float* sums, uint sums_stride,
                                   device atomic_uint* HC, threadgroup uint& last, uint word, uint per_head,
                                   device atomic_uint* PC, uint prep_arrived, uint prep_total,
                                   TfDeltaSegment span, uint head, uint t, uint sg, uint simdgroups, uint lane) {
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) {
    if (prep_arrived == prep_total) {
      atomic_store_explicit(&PC[0], 0u, memory_order_relaxed);
      atomic_store_explicit(&PC[1], 0u, memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const uint arrived = atomic_fetch_add_explicit(&HC[word], 1u, memory_order_relaxed) + 1u;
    last = arrived == per_head ? 1u : 0u;
    if (arrived == per_head) {
      atomic_store_explicit(&HC[word], 0u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last == 0u || p.dv != 128) return;
  for (uint local = sg; local < span.rows; local += simdgroups) {
    const uint row = span.first + local;
    if (row >= p.rows) break;
    tf_delta_gdn_gate_row(recurrence, gate, norm, p, output, sums, sums_stride, row, head, lane);
  }
}

// tf_delta_gdn_star_pair, then the gate in the head's last threadgroup (tf_delta_gdn_star_gate). HC: a word for each
// (segment, head), zero before the launch and zero after it. Buffers 19 to 23 are tf_delta_gdn_post_rows' 1, 2, 4, 5 and 6.
kernel void tf_delta_gdn_star_pair_gate(device const bfloat* query [[buffer(0)]],
                                   device const bfloat* key [[buffer(1)]],
                                   device const bfloat* value [[buffer(2)]],
                                   device const float* decay [[buffer(3)]],
                                   device const bfloat* mixing [[buffer(4)]],
                                   device const float* committed [[buffer(5)]],
                                   device const TfDeltaSegment* segments [[buffer(7)]],
                                   constant TfDeltaParams& p [[buffer(8)]],
                                   device bfloat* output [[buffer(9)]],
                                   device atomic_uint* HC [[buffer(10)]],
                                   device atomic_uint* PC [[buffer(17)]],
                                   constant uint& wait_prep [[buffer(18)]],
                                   device const bfloat* gate [[buffer(19)]],
                                   device const void* norm [[buffer(20)]],
                                   device bfloat* gated [[buffer(21)]],
                                   device float* sums [[buffer(22)]],
                                   constant uint& sums_stride [[buffer(23)]],
                                   uint3 position [[thread_position_in_grid]],
                                   uint3 groups [[threadgroups_per_grid]],
                                   uint t [[thread_index_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint simdgroups [[simdgroups_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint last;
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint d0 = 2 * position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint column = (head * p.dv + d0) * p.dk + lane * 4, base = span.state_slot * p.nv * p.dv * p.dk;
  float4 r0, r1;
  for (uint part = 0; part < 4; ++part) {
    r0[part] = committed[base + column + part];
    r1[part] = committed[base + column + p.dk + part];
  }
  const uint kq_step = p.nk * p.dk, at_step = p.nv, vo_step = p.nv * p.dv; // node 0's row: span.first
  const uint kq = (span.first * p.nk + key_head) * p.dk + lane * 4, at = span.first * p.nv + head, vo = at * p.dv + d0;
  const uint prep_arrived = tf_delta_gdn_star_wait_late(PC, wait_prep, t);
  tf_delta_star_rows(r0, r1, query, key, value, decay, mixing, output, span.rows, kq, at, vo, kq_step, at_step, vo_step, lane);
  tf_delta_gdn_star_gate(output, gate, norm, p, gated, sums, sums_stride, HC, last, position.z, groups.y,
                         PC, prep_arrived, groups.x * groups.y * groups.z, span, head, t, sg, simdgroups, lane);
}

// tf_delta_gdn_star_pending, then the gate in the head's last threadgroup, as tf_delta_gdn_star_pair_gate.
kernel void tf_delta_gdn_star_pending_gate(device const bfloat* query [[buffer(0)]],
                                   device const bfloat* key [[buffer(1)]],
                                   device const bfloat* value [[buffer(2)]],
                                   device const float* decay [[buffer(3)]],
                                   device const bfloat* mixing [[buffer(4)]],
                                   device float* committed [[buffer(5)]],
                                   device const TfDeltaSegment* segments [[buffer(7)]],
                                   constant TfDeltaParams& p [[buffer(8)]],
                                   device bfloat* output [[buffer(9)]],
                                   device atomic_uint* HC [[buffer(10)]],
                                   device const bfloat* pending_key [[buffer(12)]],
                                   device const bfloat* pending_value [[buffer(13)]],
                                   device const float* pending_decay [[buffer(14)]],
                                   device const bfloat* pending_mixing [[buffer(15)]],
                                   constant TfDeltaPending& pending [[buffer(16)]],
                                   device atomic_uint* PC [[buffer(17)]],
                                   constant uint& wait_prep [[buffer(18)]],
                                   device const bfloat* gate [[buffer(19)]],
                                   device const void* norm [[buffer(20)]],
                                   device bfloat* gated [[buffer(21)]],
                                   device float* sums [[buffer(22)]],
                                   constant uint& sums_stride [[buffer(23)]],
                                   uint3 position [[thread_position_in_grid]],
                                   uint3 groups [[threadgroups_per_grid]],
                                   uint t [[thread_index_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint simdgroups [[simdgroups_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint last;
  const uint segment_index = position.z / p.nv, head = position.z % p.nv;
  const uint d0 = 2 * position.y, key_head = head / (p.nv / p.nk);
  const TfDeltaSegment span = segments[segment_index];
  const uint column = (head * p.dv + d0) * p.dk + lane * 4, base = span.state_slot * p.nv * p.dv * p.dk;
  float a[4], b[4]; // columns d0 and d0 + 1, as tf_delta_gdn_replay_body<4> holds a column's state
  for (uint part = 0; part < 4; ++part) {
    a[part] = committed[base + column + part];
    b[part] = committed[base + column + p.dk + part];
  }
  const uint kq_step = p.nk * p.dk, at_step = p.nv, vo_step = p.nv * p.dv; // the pending rows' strides are the rows'
  uint pkq = key_head * p.dk + lane * 4, pat = head, pvo = head * p.dv + d0, row = 0; // pending row 0
  for (; row + 4 <= pending.rows; row += 4) tf_delta_star_pending_rows<4>(a, b, pending_key, pending_value, pending_decay, pending_mixing, pkq, pat, pvo, kq_step, at_step, vo_step);
  if (row + 3 == pending.rows) tf_delta_star_pending_rows<3>(a, b, pending_key, pending_value, pending_decay, pending_mixing, pkq, pat, pvo, kq_step, at_step, vo_step);
  else if (row + 2 == pending.rows) tf_delta_star_pending_rows<2>(a, b, pending_key, pending_value, pending_decay, pending_mixing, pkq, pat, pvo, kq_step, at_step, vo_step);
  else if (row + 1 == pending.rows) tf_delta_star_pending_rows<1>(a, b, pending_key, pending_value, pending_decay, pending_mixing, pkq, pat, pvo, kq_step, at_step, vo_step);
  if (pending.persist != 0u) {
    *(device float4*)(committed + base + column) = float4(a[0], a[1], a[2], a[3]);
    *(device float4*)(committed + base + column + p.dk) = float4(b[0], b[1], b[2], b[3]);
  }
  float4 r0 = float4(a[0], a[1], a[2], a[3]), r1 = float4(b[0], b[1], b[2], b[3]);
  const uint kq = (span.first * p.nk + key_head) * p.dk + lane * 4, at = span.first * p.nv + head, vo = at * p.dv + d0;
  const uint prep_arrived = tf_delta_gdn_star_wait_late(PC, wait_prep, t);
  tf_delta_star_rows(r0, r1, query, key, value, decay, mixing, output, span.rows, kq, at, vo, kq_step, at_step, vo_step, lane);
  tf_delta_gdn_star_gate(output, gate, norm, p, gated, sums, sums_stride, HC, last, position.z, groups.y,
                         PC, prep_arrived, groups.x * groups.y * groups.z, span, head, t, sg, simdgroups, lane);
}

// team-m5-08x2: tf_delta_gdn_pre_rows_held (deltanet.metal) with its arrival on NC[1] after the barrier and its result
// compared at the group's end; every other statement is that kernel's. Same buffers, same dispatch (Round.prepareRowsAfter).
kernel void tf_delta_gdn_pre_rows_held_late(device const bfloat* projection [[buffer(0)]],
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
                        device atomic_uint* PC [[buffer(17)]],
                        constant uint& count_out [[buffer(18)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint3 groups [[threadgroups_per_grid]],
                        uint t [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  // 128-wide heads, 4 taps: the reads from outside this barrier interval go out before the wait below
  const bool rows4 = p.dk == 128 && p.dv == 128 && p.taps == 4;
  TfDeltaPreHeld held;
  if (rows4) tf_delta_gdn_pre_rows4_held(history, weights, decay_weight, time_bias, windows, row_slot, p, group, sg, lane, held);
  // wait_for above 0: this prepare shares its barrier interval with the decoded qkv and zba projections it reads, whose
  // threadgroups count themselves on NC[0] past a device-scope fence (lane_projection's TF_COUNT); a group reads them once
  // the count reaches wait_for (bounded), past its own fence, and the last group leaves both words zero for the next pair
  uint arrived = 0u;
  if (wait_for != 0u) {
    if (t == 0) {
      for (uint spin = 0; spin < (1u << 24) && atomic_load_explicit(&NC[0], memory_order_relaxed) < wait_for; spin++) {}
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) arrived = atomic_fetch_add_explicit(&NC[1], 1u, memory_order_relaxed) + 1u;
  }
  if (rows4) tf_delta_gdn_pre_rows4(projection, a, b, p, query, key, value, decay, mixing, group, sg, lane, held);
  if (count_out != 0u) {
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(&PC[0], 1u, memory_order_relaxed);
    }
  }
  // the group whose arrival completed the count leaves both words zero for the next pair
  if (t == 0 && arrived == groups.x * groups.y * groups.z) {
    atomic_store_explicit(&NC[0], 0u, memory_order_relaxed);
    atomic_store_explicit(&NC[1], 0u, memory_order_relaxed);
  }
}
