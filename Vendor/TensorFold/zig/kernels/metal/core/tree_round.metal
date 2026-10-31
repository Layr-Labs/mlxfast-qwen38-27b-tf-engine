// Deterministic BF16 picks and a bounded verified-tree round.
#include <metal_stdlib>
using namespace metal;
struct Pick { uint token, nonfinite; };
struct Argmax { uint rows, vocab, stride; };
struct Match { uint rows, vocab, budget, eos_count; };
struct Result {
  uint status, stop, nonfinite, consumed_count, emitted_count, matched_count;
  uint bonus_emitted, pending_valid, pending_token, reserved;
  uint path[64], tokens[64];
};
kernel void tf_round_bf16_argmax(const device ushort* logits [[buffer(0)]],
                                 device Pick* picks [[buffer(1)]],
                                 constant Argmax& p [[buffer(2)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float values[256];
  threadgroup uint ids[256], masks[256];
  float best = -INFINITY;
  uint id = 0xffffffffu, mask = 0;
  for (ulong token = lane; token < p.vocab; token += 256) {
    const float value = as_type<float>(uint(logits[ulong(row) * p.stride + token]) << 16);
    if (!isfinite(value)) mask |= isnan(value) ? 1u : value > 0 ? 2u : 4u;
    else if (id == 0xffffffffu || value > best || (value == best && token < id)) {
      best = value;
      id = uint(token);
    }
  }
  values[lane] = best;
  ids[lane] = id;
  masks[lane] = mask;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 128; step > 0; step >>= 1) {
    if (lane < step) {
      const float other = values[lane + step];
      const uint other_id = ids[lane + step];
      if (other > values[lane] || (other == values[lane] && other_id < ids[lane])) {
        values[lane] = other;
        ids[lane] = other_id;
      }
      masks[lane] |= masks[lane + step];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) picks[row] = { masks[0] != 0 ? 0xffffffffu : ids[0], masks[0] };
}
struct RoundArgmaxPart { float value; uint token, nonfinite; };
static inline void round_argmax_reduce(threadgroup float* values, threadgroup uint* ids, threadgroup uint* masks, uint lane, uint width) {
  for (uint step = width / 2; step > 0; step >>= 1) {
    if (lane < step) {
      const float other = values[lane + step];
      const uint other_id = ids[lane + step];
      if (other > values[lane] || (other == values[lane] && other_id < ids[lane])) {
        values[lane] = other;
        ids[lane] = other_id;
      }
      masks[lane] |= masks[lane + step];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
kernel void tf_round_bf16_argmax_span(const device ushort* logits [[buffer(0)]],
                                      device RoundArgmaxPart* parts [[buffer(1)]],
                                      constant Argmax& p [[buffer(2)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint2 groups [[threadgroups_per_grid]],
                                      uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float values[256];
  threadgroup uint ids[256], masks[256];
  const uint span = group.x, row = group.y, spans = groups.x;
  const uint width = (p.vocab + spans - 1) / spans;
  const uint begin = min(p.vocab, span * width), end = min(p.vocab, begin + width);
  const device ushort* line = logits + ulong(row) * p.stride;
  float best = -INFINITY;
  uint id = 0xffffffffu, mask = 0;
  for (uint token = begin + lane; token < end; token += 256) {
    const float value = as_type<float>(uint(line[token]) << 16);
    if (!isfinite(value)) mask |= isnan(value) ? 1u : value > 0 ? 2u : 4u;
    else if (id == 0xffffffffu || value > best || (value == best && token < id)) {
      best = value;
      id = token;
    }
  }
  values[lane] = best;
  ids[lane] = id;
  masks[lane] = mask;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  round_argmax_reduce(values, ids, masks, lane, 256);
  if (lane == 0) parts[ulong(row) * spans + span] = { values[0], ids[0], masks[0] };
}
kernel void tf_round_bf16_argmax_merge(const device RoundArgmaxPart* parts [[buffer(0)]],
                                       device Pick* picks [[buffer(1)]],
                                       constant uint& spans [[buffer(2)]],
                                       uint row [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float values[64];
  threadgroup uint ids[64], masks[64];
  float best = -INFINITY;
  uint id = 0xffffffffu, mask = 0;
  for (uint span = lane; span < spans; span += 64) {
    const RoundArgmaxPart part = parts[ulong(row) * spans + span];
    mask |= part.nonfinite;
    if (part.token != 0xffffffffu && (id == 0xffffffffu || part.value > best || (part.value == best && part.token < id))) {
      best = part.value;
      id = part.token;
    }
  }
  values[lane] = best;
  ids[lane] = id;
  masks[lane] = mask;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  round_argmax_reduce(values, ids, masks, lane, 64);
  if (lane == 0) picks[row] = { masks[0] != 0 ? 0xffffffffu : ids[0], masks[0] };
}
kernel void tf_round_bf16_argmax_merge_match(const device RoundArgmaxPart* parts [[buffer(0)]],
                                             const device uint* tokens [[buffer(1)]],
                                             const device int* parents [[buffer(2)]],
                                             const device uint* eos [[buffer(3)]],
                                             device Result& out [[buffer(4)]],
                                             constant Match& p [[buffer(5)]],
                                             constant uint& spans [[buffer(6)]],
                                             uint lane [[thread_index_in_threadgroup]]) {
  threadgroup Pick row_picks[64];
  threadgroup float values[64];
  threadgroup uint ids[64], masks[64];
  for (uint row = 0; row < p.rows; row++) {
    float best = -INFINITY;
    uint id = 0xffffffffu, mask = 0;
    for (uint span = lane; span < spans; span += 64) {
      const RoundArgmaxPart part = parts[ulong(row) * spans + span];
      mask |= part.nonfinite;
      if (part.token != 0xffffffffu && (id == 0xffffffffu || part.value > best || (part.value == best && part.token < id))) {
        best = part.value;
        id = part.token;
      }
    }
    values[lane] = best;
    ids[lane] = id;
    masks[lane] = mask;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    round_argmax_reduce(values, ids, masks, lane, 64);
    if (lane == 0) row_picks[row] = { masks[0] != 0 ? 0xffffffffu : ids[0], masks[0] };
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane != 0) return;
  out.status = 0;
  out.stop = 0;
  out.nonfinite = 0;
  out.consumed_count = 0;
  out.emitted_count = 0;
  out.matched_count = 0;
  out.bonus_emitted = 0;
  out.pending_valid = 0;
  out.pending_token = 0;
  out.reserved = 0;
  for (uint i = 0; i < 64; i++) { out.path[i] = 0; out.tokens[i] = 0; }
  for (uint r = 0; r < p.rows; r++) {
    if (tokens[r] >= p.vocab || (r == 0 ? parents[r] != -1 : parents[r] < 0 || parents[r] >= int(r))) {
      out.status = 2; out.stop = 3; return;
    }
  }
  for (uint i = 0; i < p.eos_count; i++) if (eos[i] >= p.vocab) { out.status = 2; out.stop = 3; return; }
  bool bad_pick = false;
  for (uint r = 0; r < p.rows; r++) {
    out.nonfinite |= row_picks[r].nonfinite & 7u;
    bad_pick = bad_pick || (row_picks[r].nonfinite & ~7u) != 0 || (row_picks[r].nonfinite == 0 && row_picks[r].token >= p.vocab);
  }
  if (bad_pick || out.nonfinite != 0) { out.status = bad_pick ? 3u : 1u; out.stop = 3; return; }
  if (p.budget == 0) { out.stop = 1; return; }
  out.path[out.consumed_count++] = 0;
  uint node = 0;
  for (;;) {
    const uint token = row_picks[node].token;
    uint child = 0xffffffffu;
    for (uint r = node + 1; r < p.rows; r++) {
      if (parents[r] == int(node) && tokens[r] == token) { child = r; break; }
    }
    out.tokens[out.emitted_count++] = token;
    if (child == 0xffffffffu) out.bonus_emitted = 1;
    else out.matched_count++;
    bool hit_eos = false;
    for (uint i = 0; i < p.eos_count; i++) hit_eos = hit_eos || eos[i] == token;
    if (hit_eos) { out.stop = 2; return; }
    if (out.emitted_count == p.budget) { out.stop = 1; return; }
    if (child == 0xffffffffu) {
      out.pending_valid = 1;
      out.pending_token = token;
      return;
    }
    out.path[out.consumed_count++] = child;
    node = child;
  }
}
kernel void tf_tree_round_match(const device uint* tokens [[buffer(0)]],
                                const device int* parents [[buffer(1)]],
                                const device Pick* picks [[buffer(2)]],
                                const device uint* eos [[buffer(3)]],
                                device Result& out [[buffer(4)]],
                                constant Match& p [[buffer(5)]]) {
  out.status = 0;
  out.stop = 0;
  out.nonfinite = 0;
  out.consumed_count = 0;
  out.emitted_count = 0;
  out.matched_count = 0;
  out.bonus_emitted = 0;
  out.pending_valid = 0;
  out.pending_token = 0;
  out.reserved = 0;
  for (uint i = 0; i < 64; i++) { out.path[i] = 0; out.tokens[i] = 0; }
  for (uint row = 0; row < p.rows; row++) {
    if (tokens[row] >= p.vocab || (row == 0 ? parents[row] != -1 : parents[row] < 0 || parents[row] >= int(row))) {
      out.status = 2; out.stop = 3; return;
    }
  }
  for (uint i = 0; i < p.eos_count; i++) if (eos[i] >= p.vocab) { out.status = 2; out.stop = 3; return; }
  bool bad_pick = false;
  for (uint row = 0; row < p.rows; row++) {
    out.nonfinite |= picks[row].nonfinite & 7u;
    bad_pick = bad_pick || (picks[row].nonfinite & ~7u) != 0 || (picks[row].nonfinite == 0 && picks[row].token >= p.vocab);
  }
  if (bad_pick || out.nonfinite != 0) { out.status = bad_pick ? 3u : 1u; out.stop = 3; return; }
  if (p.budget == 0) { out.stop = 1; return; }
  out.path[out.consumed_count++] = 0;
  uint node = 0;
  for (;;) {
    const uint token = picks[node].token;
    uint child = 0xffffffffu;
    for (uint row = node + 1; row < p.rows; row++) {
      if (parents[row] == int(node) && tokens[row] == token) { child = row; break; }
    }
    out.tokens[out.emitted_count++] = token;
    if (child == 0xffffffffu) out.bonus_emitted = 1;
    else out.matched_count++;
    bool hit_eos = false;
    for (uint i = 0; i < p.eos_count; i++) hit_eos = hit_eos || eos[i] == token;
    if (hit_eos) { out.stop = 2; return; }
    if (out.emitted_count == p.budget) { out.stop = 1; return; }
    if (child == 0xffffffffu) {
      out.pending_valid = 1;
      out.pending_token = token;
      return;
    }
    out.path[out.consumed_count++] = child;
    node = child;
  }
}
