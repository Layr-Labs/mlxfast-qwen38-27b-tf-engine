// team-m5-08x1: a decoded star verify's GDN gate in its scan's epilogue. Built as a library of its own, this file after
// core_deltanet (deltanet.metal and deltanet_slots.metal), so the record's library keeps the record's text.
// tf_delta_gdn_star_pair_gate and tf_delta_gdn_star_pending_gate are tf_delta_gdn_star_pair and tf_delta_gdn_star_pending
// statement for statement up to their rows' stores. Where those count every threadgroup on SC[0] for a gate launch that
// waits on the count in the same barrier interval, these count each threadgroup on its head's word past the same
// device-scope fence, and the threadgroup that completes its head's count runs tf_delta_gdn_post_rows' statements on the
// head's rows: no threadgroup waits on another.

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

// A star threadgroup's epilogue. Once its rows are stored, thread 0 passes the device-scope fence that the record's
// counted scan passes and counts the group on its head's word, HC[segment * nv + head]. The group that completes the
// head's count (per_head groups, dv / 32) leaves the word zero for the next launch, passes the fence that the record's
// waiting gate passes, and runs the gate on its segment's rows of the head, a simdgroup a row. The other groups return.
inline void tf_delta_gdn_star_gate(device const bfloat* recurrence, device const bfloat* gate, device const void* norm,
                                   constant TfDeltaParams& p, device bfloat* output, device float* sums, uint sums_stride,
                                   device atomic_uint* HC, threadgroup uint& last, uint word, uint per_head,
                                   TfDeltaSegment span, uint head, uint t, uint sg, uint simdgroups, uint lane) {
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) {
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
  tf_delta_gdn_star_wait(PC, wait_prep, groups, t);
  tf_delta_star_rows(r0, r1, query, key, value, decay, mixing, output, span.rows, kq, at, vo, kq_step, at_step, vo_step, lane);
  tf_delta_gdn_star_gate(output, gate, norm, p, gated, sums, sums_stride, HC, last, position.z, groups.y, span, head, t, sg, simdgroups, lane);
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
  tf_delta_gdn_star_wait(PC, wait_prep, groups, t);
  tf_delta_star_rows(r0, r1, query, key, value, decay, mixing, output, span.rows, kq, at, vo, kq_step, at_step, vo_step, lane);
  tf_delta_gdn_star_gate(output, gate, norm, p, gated, sums, sums_stride, HC, last, position.z, groups.y, span, head, t, sg, simdgroups, lane);
}
