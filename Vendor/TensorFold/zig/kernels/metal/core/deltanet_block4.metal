// One 32-lane SIMD group handles one value column of one segment/head.
// Dispatch threads (32, dv, nv * segments), threads per group (32, 1, 1).
// Requires dk == 128 and nv divisible by nk. The arithmetic association differs
// from the leader scan; token-golden and performance checks are mandatory.
kernel void tf_delta_gdn_chain_block4(
    device const bfloat* query [[buffer(0)]],
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
  const uint segment = position.z / p.nv;
  const uint head = position.z % p.nv;
  const uint key_head = head / (p.nv / p.nk);
  const uint column = position.y;
  const TfDeltaSegment span = segments[segment];
  const uint state_stride = p.nv * p.dv * p.dk;
  const uint state_offset = (head * p.dv + column) * p.dk + lane * 4;
  float s[4];
  for (uint d = 0; d < 4; ++d)
    s[d] = committed[span.state_slot * state_stride + state_offset + d];

  for (uint first = 0; first < span.rows; first += 4) {
    const uint n = min(4u, span.rows - first);
    float keys[4][4], queries[4][4], products[4][5];
    float kk[4][4], qk[4][4], ks[4], qs[4], beta[4], val[4];
    for (uint i = 0; i < n; ++i) {
      const uint row = span.first + first + i;
      const uint kq = (row * p.nk + key_head) * p.dk + lane * 4;
      for (uint d = 0; d < 4; ++d) {
        keys[i][d] = float(key[kq + d]);
        queries[i][d] = float(query[kq + d]);
      }
      products[i][i + 1] = 1.0f;
      for (int j = int(i); j >= 0; --j) {
        const uint at = (span.first + first + uint(j)) * p.nv + head;
        products[i][j] = products[i][j + 1] * decay[at];
      }
      const uint at = row * p.nv + head;
      beta[i] = float(mixing[at]);
      val[i] = float(value[at * p.dv + column]);
      float sk = 0.0f, sq = 0.0f;
      for (uint d = 0; d < 4; ++d) {
        sk += keys[i][d] * s[d];
        sq += queries[i][d] * s[d];
      }
      ks[i] = simd_sum(sk);
      qs[i] = simd_sum(sq);
      for (uint j = 0; j <= i; ++j) {
        float ksum = 0.0f, qsum = 0.0f;
        for (uint d = 0; d < 4; ++d) {
          ksum += keys[i][d] * keys[j][d];
          qsum += queries[i][d] * keys[j][d];
        }
        kk[i][j] = simd_sum(ksum);
        qk[i][j] = simd_sum(qsum);
      }
    }
    float delta[4];
    for (uint i = 0; i < n; ++i) {
      float memory = products[i][0] * ks[i];
      for (uint j = 0; j < i; ++j)
        memory += products[i][j + 1] * kk[i][j] * delta[j];
      delta[i] = beta[i] * (val[i] - memory);
      float result = products[i][0] * qs[i];
      for (uint j = 0; j <= i; ++j)
        result += products[i][j + 1] * qk[i][j] * delta[j];
      if (lane == 0) {
        const uint at = (span.first + first + i) * p.nv + head;
        output[at * p.dv + column] = bfloat(result);
      }
    }
    for (uint d = 0; d < 4; ++d) {
      float next = products[n - 1][0] * s[d];
      for (uint j = 0; j < n; ++j)
        next += products[n - 1][j + 1] * keys[j][d] * delta[j];
      s[d] = next;
    }
  }
  for (uint d = 0; d < 4; ++d)
    next_state[span.next_slot * state_stride + state_offset + d] = s[d];
}
