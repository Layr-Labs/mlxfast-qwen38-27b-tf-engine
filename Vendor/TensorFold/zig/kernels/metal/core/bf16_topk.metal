// One input scan builds local min-heaps, sorted once before exact score/lowest-token-ID merges.
#include <metal_stdlib>
using namespace metal;
// Each is also a constant in bf16_topk_gpu.zig, which does not compile when its value differs.
#define TF_TOPK_LANES 256
#define TF_TOPK_MAX_K 16
#define TF_TOPK_SPANS 16
struct Params { uint rows, vocab, k, stride; };
struct Entry { uint token, score_bits; };
struct Row { uint nonfinite, count; Entry entries[TF_TOPK_MAX_K]; };
inline uint tf_bf16_key(ushort raw) {
  const uint bits = (raw & 0x7fffu) == 0 ? 0u : uint(raw);
  return (bits & 0x8000u) != 0 ? (~bits & 0xffffu) : (bits | 0x8000u);
}
inline bool tf_topk_before(uint key, uint id, uint other_key, uint other_id) {
  return key > other_key || (key == other_key && id < other_id);
}
inline void tf_topk_swap(thread uint* key, thread uint* id, thread uint* bits, uint a, uint b) {
  const uint k = key[a], i = id[a], s = bits[a]; key[a] = key[b]; id[a] = id[b]; bits[a] = bits[b];
  key[b] = k; id[b] = i; bits[b] = s;
}
inline void tf_topk_sift(thread uint* key, thread uint* id, thread uint* bits, uint count) {
  uint at = 0;
  while (2 * at + 1 < count) {
    uint child = 2 * at + 1;
    if (child + 1 < count && tf_topk_before(key[child], id[child], key[child + 1], id[child + 1])) child++;
    if (!tf_topk_before(key[at], id[at], key[child], id[child])) break;
    tf_topk_swap(key, id, bits, at, child); at = child;
  }
}
// A rank's tournament: each SIMD group's best candidate by xor shuffles (the same "before" order), then the group bests
// through threadgroup memory, one barrier a rank (rank parity picks the half of `keys` it writes).
struct TfCand { uint key, id, bits; };
inline TfCand tf_simd_best(TfCand c) {
  for (ushort step = 16; step > 0; step >>= 1) {
    const TfCand o = { simd_shuffle_xor(c.key, step), simd_shuffle_xor(c.id, step), simd_shuffle_xor(c.bits, step) };
    if (tf_topk_before(o.key, o.id, c.key, c.id)) c = o;
  }
  return c;
}
inline TfCand tf_tg_best(TfCand c, uint lane, uint rank, threadgroup uint* keys, threadgroup uint* ids, threadgroup uint* score_bits) {
  const TfCand b = tf_simd_best(c);
  const uint hb = (rank & 1) * (TF_TOPK_LANES / 32);
  if ((lane & 31) == 0) { keys[hb + lane / 32] = b.key; ids[hb + lane / 32] = b.id; score_bits[hb + lane / 32] = b.bits; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  TfCand w = { keys[hb], ids[hb], score_bits[hb] };
  for (uint g = 1; g < TF_TOPK_LANES / 32; g++)
    if (tf_topk_before(keys[hb + g], ids[hb + g], w.key, w.id)) w = { keys[hb + g], ids[hb + g], score_bits[hb + g] };
  return w;
}
inline uint tf_tg_or(uint m, uint lane, threadgroup uint* masks) {
  const uint s = simd_or(m);
  if ((lane & 31) == 0) masks[lane / 32] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint all = 0;
  for (uint g = 0; g < TF_TOPK_LANES / 32; g++) all |= masks[g];
  return all;
}
// The scan, local heaps and tournament over tokens [begin, end) of one row, into one Row.
inline void tf_topk_span(const device ushort* logits, device Row& out, ulong begin, ulong end, constant Params& p,
                         uint lane, threadgroup uint* keys, threadgroup uint* ids, threadgroup uint* score_bits,
                         threadgroup uint* masks) {
  uint local_key[TF_TOPK_MAX_K], local_id[TF_TOPK_MAX_K], local_bits[TF_TOPK_MAX_K];
  for (uint i = 0; i < TF_TOPK_MAX_K; i++) { local_key[i] = 0; local_id[i] = 0xffffffffu; local_bits[i] = 0; }
  if (lane == 0) {
    out.nonfinite = 0; out.count = 0;
    for (uint i = 0; i < TF_TOPK_MAX_K; i++) out.entries[i] = { 0xffffffffu, 0u };
  }
  uint mask = 0, count = 0;
  for (ulong token = begin + lane; token < end; token += TF_TOPK_LANES) {
    const ushort raw = logits[token];
    if ((raw & 0x7f80u) == 0x7f80u) {
      mask |= (raw & 0x7fu) != 0 ? 1u : (raw & 0x8000u) == 0 ? 2u : 4u;
      continue;
    }
    const uint key = tf_bf16_key(raw), id = uint(token), bits = uint(raw) << 16;
    if (count < p.k) {
      uint at = count++;
      local_key[at] = key; local_id[at] = id; local_bits[at] = bits;
      while (at > 0) {
        const uint parent = (at - 1) / 2;
        if (!tf_topk_before(local_key[parent], local_id[parent], local_key[at], local_id[at])) break;
        tf_topk_swap(local_key, local_id, local_bits, at, parent); at = parent;
      }
    } else if (tf_topk_before(key, id, local_key[0], local_id[0])) {
      local_key[0] = key; local_id[0] = id; local_bits[0] = bits;
      tf_topk_sift(local_key, local_id, local_bits, count);
    }
  }
  for (uint last = count; last > 1; last--) {
    tf_topk_swap(local_key, local_id, local_bits, 0, last - 1);
    tf_topk_sift(local_key, local_id, local_bits, last - 1);
  }
  const uint all = tf_tg_or(mask, lane, masks);
  if (all != 0) { if (lane == 0) out.nonfinite = all; return; }
  uint cursor = 0;
  for (uint rank = 0; rank < p.k; rank++) {
    const TfCand c = cursor < p.k ? TfCand{ local_key[cursor], local_id[cursor], local_bits[cursor] } : TfCand{ 0u, 0xffffffffu, 0u };
    const TfCand w = tf_tg_best(c, lane, rank, keys, ids, score_bits);
    if (lane == 0) { out.entries[rank] = { w.id, w.bits }; out.count = rank + 1; }
    if (cursor < p.k && local_id[cursor] == w.id) cursor++;
  }
}

kernel void tf_bf16_topk(const device ushort* logits [[buffer(0)]],
                        device Row* output [[buffer(1)]], constant Params& p [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint keys[TF_TOPK_LANES], ids[TF_TOPK_LANES], score_bits[TF_TOPK_LANES], masks[TF_TOPK_LANES];
  tf_topk_span(logits + ulong(row) * p.stride, output[row], 0, p.vocab, p, lane, keys, ids, score_bits, masks);
}

// A wide row's spans each leave their top k and non-finite mask in part (the span's Row); the merge picks the row's.
kernel void tf_bf16_topk_span(const device ushort* logits [[buffer(0)]],
                             device Row* part [[buffer(1)]], constant Params& p [[buffer(2)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint keys[TF_TOPK_LANES], ids[TF_TOPK_LANES], score_bits[TF_TOPK_LANES], masks[TF_TOPK_LANES];
  const ulong width = (ulong(p.vocab) + TF_TOPK_SPANS - 1) / TF_TOPK_SPANS;
  const ulong begin = min(ulong(tg.x) * width, ulong(p.vocab)), end = min(begin + width, ulong(p.vocab));
  tf_topk_span(logits + ulong(tg.y) * p.stride, part[tg.y * TF_TOPK_SPANS + tg.x], begin, end, p, lane, keys, ids, score_bits, masks);
}

kernel void tf_bf16_topk_merge(const device Row* part [[buffer(0)]],
                              device Row* output [[buffer(1)]], constant Params& p [[buffer(2)]],
                              uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint keys[TF_TOPK_LANES], ids[TF_TOPK_LANES], score_bits[TF_TOPK_LANES], masks[TF_TOPK_LANES];
  const device Row* spans = part + row * TF_TOPK_SPANS;
  uint key = 0, id = 0xffffffffu, bits = 0, mask = 0;
  if (lane < TF_TOPK_SPANS) mask = spans[lane].nonfinite;
  const uint span = lane / TF_TOPK_MAX_K, at = lane % TF_TOPK_MAX_K;
  if (span < TF_TOPK_SPANS && at < p.k && spans[span].entries[at].token != 0xffffffffu) {
    bits = spans[span].entries[at].score_bits;
    key = tf_bf16_key(ushort(bits >> 16)); id = spans[span].entries[at].token;
  }
  if (lane == 0) {
    output[row].nonfinite = 0; output[row].count = 0;
    for (uint i = 0; i < TF_TOPK_MAX_K; i++) output[row].entries[i] = { 0xffffffffu, 0u };
  }
  const uint all = tf_tg_or(mask, lane, masks);
  if (all != 0) { if (lane == 0) output[row].nonfinite = all; return; }
  for (uint rank = 0; rank < p.k; rank++) {
    const TfCand w = tf_tg_best(TfCand{ key, id, bits }, lane, rank, keys, ids, score_bits);
    if (lane == 0) { output[row].entries[rank] = { w.id, w.bits }; output[row].count = rank + 1; }
    if (id == w.id) { key = 0; id = 0xffffffffu; bits = 0; }
  }
}

kernel void tf_bf16_topk_max(const device ushort* logits [[buffer(0)]],
                         device Row* output [[buffer(1)]],
                         constant Params& p [[buffer(2)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint keys[TF_TOPK_LANES], ids[TF_TOPK_LANES], score_bits[TF_TOPK_LANES], masks[TF_TOPK_LANES];
  threadgroup uint previous_key, previous_id;
  if (lane == 0) {
    output[row].nonfinite = 0;
    output[row].count = 0;
    for (uint i = 0; i < TF_TOPK_MAX_K; i++) output[row].entries[i] = { 0xffffffffu, 0u };
  }
  for (uint rank = 0; rank < p.k; rank++) {
    uint best_key = 0, best_id = 0xffffffffu, best_bits = 0, mask = 0;
    for (ulong token = lane; token < p.vocab; token += TF_TOPK_LANES) {
      const ushort raw = logits[ulong(row) * p.stride + token];
      if ((raw & 0x7f80u) == 0x7f80u) {
        mask |= (raw & 0x7fu) != 0 ? 1u : (raw & 0x8000u) == 0 ? 2u : 4u;
        continue;
      }
      const uint key = tf_bf16_key(raw);
      if (rank > 0 && (key > previous_key || (key == previous_key && token <= previous_id))) continue;
      if (key > best_key || (key == best_key && token < best_id)) {
        best_key = key;
        best_id = uint(token);
        best_bits = uint(raw) << 16;
      }
    }
    keys[lane] = best_key;
    ids[lane] = best_id;
    score_bits[lane] = best_bits;
    masks[lane] = mask;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = TF_TOPK_LANES / 2; step > 0; step >>= 1) {
      if (lane < step) {
        const uint other_key = keys[lane + step], other_id = ids[lane + step];
        if (other_key > keys[lane] || (other_key == keys[lane] && other_id < ids[lane])) {
          keys[lane] = other_key;
          ids[lane] = other_id;
          score_bits[lane] = score_bits[lane + step];
        }
        masks[lane] |= masks[lane + step];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (masks[0] != 0) {
      if (lane == 0) output[row].nonfinite = masks[0];
      return;
    }
    if (lane == 0) {
      output[row].entries[rank] = { ids[0], score_bits[0] };
      output[row].count = rank + 1;
      previous_key = keys[0];
      previous_id = ids[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The distinct 32-column tiles of rows' top-k tokens, tile 0 always among them: list[0] their count, list[1 + j] the
// tiles in increasing order; map[32 j + c] the id of tile list[1 + j]'s column c (ids[32 tile + c]) for each of the
// launch's `slots` (0 past the count). One threadgroup.
#define TF_TOPK_TILE_WORDS 256
#define TF_TOPK_TILE_SLOTS 272
struct TileParams { uint rows, tiles, slots; };
kernel void tf_bf16_topk_tiles(const device Row* picked [[buffer(0)]], const device uint* ids [[buffer(1)]],
                               device uint* list [[buffer(2)]], device uint* map [[buffer(3)]],
                               constant TileParams& p [[buffer(4)]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint marks[TF_TOPK_TILE_WORDS];
  threadgroup uint order[TF_TOPK_TILE_SLOTS];
  threadgroup uint found[1];
  const uint words = (p.tiles + 31) / 32;
  for (uint i = lane; i < words; i += TF_TOPK_LANES) atomic_store_explicit(&marks[i], i == 0 ? 1u : 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lane; i < p.rows * TF_TOPK_MAX_K; i += TF_TOPK_LANES) {
    const device Row& r = picked[i / TF_TOPK_MAX_K];
    const uint at = i % TF_TOPK_MAX_K, tile = r.entries[at].token / 32;
    if (at < r.count && tile < p.tiles) atomic_fetch_or_explicit(&marks[tile / 32], 1u << (tile % 32), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    uint n = 0;
    for (uint i = 0; i < words; i++)
      for (uint m = atomic_load_explicit(&marks[i], memory_order_relaxed); m != 0 && n < p.slots; m &= m - 1) order[n++] = 32 * i + ctz(m);
    found[0] = n;
    list[0] = n;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n = found[0];
  for (uint j = lane; j < n; j += TF_TOPK_LANES) list[1 + j] = order[j];
  for (uint at = lane; at < p.slots * 32; at += TF_TOPK_LANES) map[at] = at / 32 < n ? ids[32 * order[at / 32] + at % 32] : 0u;
}
#define TF_TOPK_WIDE_SPANS 64
#define TF_TOPK_HOLD 8
#define TF_TOPK_GROUPS (TF_TOPK_LANES / 32)
inline TfCand tf_simd_pick(TfCand mine, thread bool& taken) {
  const TfCand w = tf_simd_best(mine);
  taken = mine.id == w.id && mine.key == w.key && mine.key != 0u;
  return w;
}
kernel void tf_bf16_topk_wide(const device ushort* logits [[buffer(0)]],
                             device Row* part [[buffer(1)]], constant Params& p [[buffer(2)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]) {
  threadgroup uint masks[TF_TOPK_LANES];
  threadgroup uint list_key[TF_TOPK_GROUPS][TF_TOPK_MAX_K], list_id[TF_TOPK_GROUPS][TF_TOPK_MAX_K], list_bits[TF_TOPK_GROUPS][TF_TOPK_MAX_K];
  const ulong width = (ulong(p.vocab) + TF_TOPK_WIDE_SPANS - 1) / TF_TOPK_WIDE_SPANS;
  const ulong begin = min(ulong(tg.x) * width, ulong(p.vocab)), end = min(begin + width, ulong(p.vocab));
  const device ushort* row = logits + ulong(tg.y) * p.stride;
  device Row& out = part[tg.y * TF_TOPK_WIDE_SPANS + tg.x];
  uint key[TF_TOPK_HOLD], bits[TF_TOPK_HOLD], live = 0, mask = 0;
  for (uint i = 0; i < TF_TOPK_HOLD; i++) {
    const ulong token = begin + lane + ulong(i) * TF_TOPK_LANES;
    key[i] = 0; bits[i] = 0;
    if (token < end) {
      const ushort raw = row[token];
      if ((raw & 0x7f80u) == 0x7f80u) mask |= (raw & 0x7fu) != 0 ? 1u : (raw & 0x8000u) == 0 ? 2u : 4u;
      else { key[i] = tf_bf16_key(raw); bits[i] = uint(raw) << 16; live |= 1u << i; }
    }
  }
  if (lane == 0) {
    out.nonfinite = 0; out.count = 0;
    for (uint i = 0; i < TF_TOPK_MAX_K; i++) out.entries[i] = { 0xffffffffu, 0u };
  }
  const uint all = tf_tg_or(mask, lane, masks);
  if (all != 0) { if (lane == 0) out.nonfinite = all; return; }
  for (uint rank = 0; rank < TF_TOPK_MAX_K; rank++) {
    TfCand c = { 0u, 0xffffffffu, 0u };
    uint slot = TF_TOPK_HOLD;
    if (rank < p.k) for (uint i = 0; i < TF_TOPK_HOLD; i++) {
      const uint id = uint(begin + lane + ulong(i) * TF_TOPK_LANES);
      if (((live >> i) & 1u) != 0 && tf_topk_before(key[i], id, c.key, c.id)) { c = { key[i], id, bits[i] }; slot = i; }
    }
    bool taken;
    const TfCand w = tf_simd_pick(c, taken);
    if (taken) live &= ~(1u << slot);
    if (sl == 0) { list_key[sg][rank] = w.key; list_id[sg][rank] = w.id; list_bits[sg][rank] = w.bits; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const uint g = sl / 4, first = 4 * (sl % 4);
  uint cursor = 0;
  for (uint rank = 0; rank < p.k; rank++) {
    const uint at = first + cursor;
    const TfCand c = cursor < 4 ? TfCand{ list_key[g][at], list_id[g][at], list_bits[g][at] } : TfCand{ 0u, 0xffffffffu, 0u };
    bool taken;
    const TfCand w = tf_simd_pick(c, taken);
    if (taken) cursor++;
    if (sl == 0) { out.entries[rank] = { w.id, w.bits }; out.count = rank + 1; }
  }
}

kernel void tf_bf16_topk_wide_merge(const device Row* part [[buffer(0)]],
                                   device Row* output [[buffer(1)]], constant Params& p [[buffer(2)]],
                                   uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]) {
  threadgroup uint masks[TF_TOPK_LANES];
  threadgroup uint list_key[TF_TOPK_WIDE_SPANS][TF_TOPK_MAX_K], list_id[TF_TOPK_WIDE_SPANS][TF_TOPK_MAX_K], list_bits[TF_TOPK_WIDE_SPANS][TF_TOPK_MAX_K];
  const device Row* spans = part + row * TF_TOPK_WIDE_SPANS;
  for (uint e = lane; e < TF_TOPK_WIDE_SPANS * TF_TOPK_MAX_K; e += TF_TOPK_LANES) {
    const uint span = e / TF_TOPK_MAX_K, at = e % TF_TOPK_MAX_K;
    uint key = 0, id = 0xffffffffu, bits = 0;
    if (at < p.k && spans[span].entries[at].token != 0xffffffffu) {
      bits = spans[span].entries[at].score_bits;
      key = tf_bf16_key(ushort(bits >> 16)); id = spans[span].entries[at].token;
    }
    list_key[span][at] = key; list_id[span][at] = id; list_bits[span][at] = bits;
  }
  const uint mask = lane < TF_TOPK_WIDE_SPANS ? spans[lane].nonfinite : 0u;
  if (lane == 0) {
    output[row].nonfinite = 0; output[row].count = 0;
    for (uint i = 0; i < TF_TOPK_MAX_K; i++) output[row].entries[i] = { 0xffffffffu, 0u };
  }
  const uint all = tf_tg_or(mask, lane, masks);
  if (all != 0) { if (lane == 0) output[row].nonfinite = all; return; }
  if (sg != 0) return;
  uint ca = 0, cb = 0;
  const uint la = 2 * sl, lb = 2 * sl + 1;
  for (uint rank = 0; rank < p.k; rank++) {
    const TfCand a = ca < TF_TOPK_MAX_K ? TfCand{ list_key[la][ca], list_id[la][ca], list_bits[la][ca] } : TfCand{ 0u, 0xffffffffu, 0u };
    const TfCand b = cb < TF_TOPK_MAX_K ? TfCand{ list_key[lb][cb], list_id[lb][cb], list_bits[lb][cb] } : TfCand{ 0u, 0xffffffffu, 0u };
    const bool first_a = tf_topk_before(a.key, a.id, b.key, b.id) || b.key == 0u;
    const TfCand c = first_a ? a : b;
    bool taken;
    const TfCand w = tf_simd_pick(c, taken);
    if (taken) { if (first_a) ca++; else cb++; }
    if (sl == 0) { output[row].entries[rank] = { w.id, w.bits }; output[row].count = rank + 1; }
  }
}
