// Qwen's row arithmetic: fp32 accumulation, native checkpoint metadata and explicit activation rounding.
#include <metal_stdlib>
using namespace metal;

inline float q27_read(device const uchar* p, uint i, uint type) {
  if (type == 0) return float(((device const bfloat*)p)[i]);
  if (type == 1) return float(((device const half*)p)[i]);
  return ((device const float*)p)[i];
}

inline float q27_round(float v, uint type) {
  return type == 0 ? float(bfloat(v)) : type == 1 ? float(half(v)) : v;
}

inline void q27_write(device uchar* p, uint i, float v, uint type) {
  if (type == 0) ((device bfloat*)p)[i] = bfloat(v);
  else if (type == 1) ((device half*)p)[i] = half(v);
  else ((device float*)p)[i] = v;
}

struct Norm { uint width, rows, activation, gain, residual; float eps; };

// Each thread owns 16 contiguous values, then simd_sum and simdgroups in increasing order as row_glue.add_norm.
[[max_total_threads_per_threadgroup(1024)]]
kernel void q27_norm(device const uchar* X [[buffer(0)]], device const uchar* R [[buffer(1)]],
                     device const uchar* W [[buffer(2)]], constant Norm& a [[buffer(3)]],
                     device uchar* H [[buffer(4)]], device uchar* Y [[buffer(5)]],
                     uint2 group [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[32];
  const uint row = group.y, base = row * a.width + 16 * t;
  float values[16], square = 0.0f;
  for (uint j = 0; j < 16; ++j) {
    float h = q27_read(X, base + j, a.activation);
    if (a.residual) h = q27_round(h + q27_read(R, base + j, a.activation), a.activation);
    values[j] = h;
    q27_write(H, base + j, h, a.activation);
    square = fma(h, h, square);
  }
  square = simd_sum(square);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < a.width / 512; ++s) total += sums[s];
  const float inv = rsqrt(total / float(a.width) + a.eps);
  for (uint j = 0; j < 16; ++j) q27_write(Y, base + j, q27_read(W, 16 * t + j, a.gain) * (values[j] * inv), a.activation);
}

struct Head { uint width, rows, stride, activation, gain; float eps; };

// Head RMSNorm retains its native-input rounding before gain multiplication, as the shared Metal RMS implementation.
kernel void q27_head_norm(device const uchar* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
                          constant Head& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[32], inverse[1];
  float square = 0.0f;
  uint first = 4 * t;
  while (first < a.width) {
    for (uint j = 0; j != 4; ++j) {
      const uint column = first + j;
      const float value = column < a.width ? q27_read(X, row * a.stride + column, a.activation) : 0.0f;
      square += value * value;
    }
    first += 4 * threads;
  }
  square = simd_sum(square);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(sums[lane]);
    if (lane == 0) inverse[0] = precise::rsqrt(total / float(a.width) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint column = 4 * t; column < a.width; column += 4 * threads) {
    for (uint j = 0; j != 4 && column + j < a.width; ++j) {
      const uint at = column + j;
      const float value = q27_read(X, row * a.stride + at, a.activation);
      const float normal = q27_round(value * inverse[0], a.activation);
      q27_write(Y, row * a.width + at, q27_read(W, at, a.gain) * normal, a.activation);
    }
  }
}

struct Embed { uint width, rows, vocab, bits, group, metadata; };

// Native metadata promotes to FP32 for affine decode, then the activation rounds once.
kernel void q27_embed(device const uint* ids [[buffer(0)]], device const uint* words [[buffer(1)]],
                      device const uchar* scales [[buffer(2)]], device const uchar* biases [[buffer(3)]],
                      constant Embed& a [[buffer(4)]], device uchar* out [[buffer(5)]],
                      uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.rows || ids[at.y] >= a.vocab) return;
  const uint bit = at.x * a.bits, shift = bit % 32, index = ids[at.y] * (a.width * a.bits / 32) + bit / 32;
  uint code = words[index] >> shift;
  if (shift + a.bits > 32) code |= words[index + 1] << (32 - shift);
  code &= (1u << a.bits) - 1u;
  const uint meta = ids[at.y] * (a.width / a.group) + at.x / a.group;
  const float value = q27_read(scales, meta, a.metadata) * float(code) + q27_read(biases, meta, a.metadata);
  q27_write(out, at.y * a.width + at.x, value, a.metadata);
}

struct Rope { uint width, heads, rows, rotary, activation; float theta; uint h_end, w_end, plane; };

// Rotate paired halves at each row's position: pair d's axis plane under M-RoPE, plane 0 (t) otherwise.
kernel void q27_rope(device const uchar* X [[buffer(0)]], device const int* positions [[buffer(1)]],
                     constant Rope& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
                     uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.heads || at.z >= a.rows) return;
  const uint base = (at.z * a.heads + at.y) * a.width, pairs = a.rotary / 2;
  float value = q27_read(X, base + at.x, a.activation);
  if (at.x < a.rotary) {
    const bool second = at.x >= pairs;
    const uint d = second ? at.x - pairs : at.x;
    const uint axis = d % 3 == 1 && d < a.h_end ? 1u : (d % 3 == 2 && d < a.w_end ? 2u : 0u);
    const float angle = float(positions[axis * a.plane + at.z]) * pow(a.theta, -float(d) / float(pairs));
    const float partner = q27_read(X, base + (second ? d : d + pairs), a.activation);
    value = second ? value * cos(angle) + partner * sin(angle) : value * cos(angle) - partner * sin(angle);
  }
  q27_write(Y, base + at.x, value, a.activation);
}

struct HeadRope { uint width, heads, kv_heads, stride, rotary; float eps, theta; uint h_end, w_end, plane, key_stride, rows, packed_rows; };

// q27_head_norm then q27_rope for bf16 prompt rows 256 wide: a row's q heads (`stride` apart) then k heads (rows
// `key_stride` apart), four a group.
// Lane l holds the norm's threads l and 32 + l (columns 4l.., 128 + 4l..) and their sums; the rope turns the rounded
// values norm_q and norm_k held, pair d's angle from lane d. PACK: q heads land as q27_attn_pack_a packs one stream of
// packed_rows (kv head h / g's rows, row r's head h at r * g + h % g, rows * g on repeating row 0's first head).
// A stream's key and value caches, as the attention binding's caches table holds them.
struct Q27HeadCaches { device bfloat* keys; device bfloat* values; };
// DEC (decoded rows of one stream): beside QY and KY, a q head's rounded values also land where q27_attn_pack_a packs
// them (QA, packed_rows rows a kv head, rows * g on repeating row 0's first head), and a k head's rotated key and its
// value row land where q27_attn_append writes them (the row's cache slot from the plan's meta and nodes).
template <bool PACK, bool DEC>
inline void q27_head_norm_rope_rows(device const bfloat* Q, device const bfloat* K, device const bfloat* QW, device const bfloat* KW,
                                    device const int* positions, constant HeadRope& a, device bfloat* QY, device bfloat* KY,
                                    uint2 group, uint sg, uint lane, device bfloat* QA = nullptr,
                                    device const Q27HeadCaches* caches = nullptr, device const int* meta = nullptr,
                                    device const int* nodes = nullptr, device const bfloat* V = nullptr, uint v_stride = 0) {
  const uint head = group.x * 4 + sg, row = group.y;
  if (head >= a.heads + a.kv_heads) return;
  const bool query = head < a.heads;
  device const bfloat* x = query ? Q + (row * a.heads + head) * a.stride : K + row * a.key_stride + (head - a.heads) * a.width;
  device const bfloat* w = query ? QW : KW;
  const uint g = PACK ? a.heads / a.kv_heads : 0, kv_head = PACK ? head / g : 0;
  device bfloat* y = !query ? KY + (row * a.kv_heads + head - a.heads) * a.width
                            : QY + (PACK ? kv_head * a.packed_rows + row * g + head - kv_head * g : row * a.heads + head) * a.width;
  const bfloat4 xa = *(device const bfloat4*)(x + 4 * lane), xb = *(device const bfloat4*)(x + 128 + 4 * lane);
  float sa = 0.0f, sb = 0.0f;
  for (uint j = 0; j != 4; ++j) {
    const float value = float(xa[j]);
    sa += value * value;
  }
  for (uint j = 0; j != 4; ++j) {
    const float value = float(xb[j]);
    sb += value * value;
  }
  const float s0 = simd_sum(sa), s1 = simd_sum(sb);
  const float total = simd_sum(lane == 0 ? s0 : (lane == 1 ? s1 : 0.0f));
  const float inverse = precise::rsqrt(total / float(a.width) + a.eps);
  float na[4], nb[4];
  for (uint j = 0; j != 4; ++j) {
    const float normal = float(bfloat(float(xa[j]) * inverse));
    na[j] = float(bfloat(float(w[4 * lane + j]) * normal));
  }
  for (uint j = 0; j != 4; ++j) {
    const float normal = float(bfloat(float(xb[j]) * inverse));
    nb[j] = float(bfloat(float(w[128 + 4 * lane + j]) * normal));
  }
  const uint pairs = a.rotary / 2;
  float c = 0.0f, s = 0.0f;
  if (lane < pairs) {
    const uint d = lane;
    const uint axis = d % 3 == 1 && d < a.h_end ? 1u : (d % 3 == 2 && d < a.w_end ? 2u : 0u);
    const float angle = float(positions[axis * a.plane + row]) * pow(a.theta, -float(d) / float(pairs));
    c = cos(angle);
    s = sin(angle);
  }
  bfloat4 ya, yb;
  for (uint j = 0; j != 4; ++j) {
    const uint at = 4 * lane + j;
    const bool second = at >= pairs;
    const uint d = second ? at - pairs : at;
    const float partner = simd_shuffle(na[j], ushort(((second ? d : d + pairs) / 4) % 32));
    const float cosine = simd_shuffle(c, ushort(d % 32)), sine = simd_shuffle(s, ushort(d % 32));
    float value = na[j];
    if (at < a.rotary) value = second ? value * cosine + partner * sine : value * cosine - partner * sine;
    ya[j] = bfloat(value);
    yb[j] = bfloat(nb[j]);
  }
  *(device bfloat4*)(y + 4 * lane) = ya;
  *(device bfloat4*)(y + 128 + 4 * lane) = yb;
  if (PACK && query && row == 0 && head == kv_head * g)
    for (uint r = a.rows * g; r < a.packed_rows; ++r) {
      device bfloat* pad = QY + (kv_head * a.packed_rows + r) * a.width;
      *(device bfloat4*)(pad + 4 * lane) = ya;
      *(device bfloat4*)(pad + 128 + 4 * lane) = yb;
    }
  if (DEC && query) {
    const uint gq = a.heads / a.kv_heads, hq = head / gq;
    device bfloat* packed = QA + (hq * a.packed_rows + row * gq + head - hq * gq) * a.width;
    *(device bfloat4*)(packed + 4 * lane) = ya;
    *(device bfloat4*)(packed + 128 + 4 * lane) = yb;
    if (row == 0 && head == hq * gq)
      for (uint r = a.rows * gq; r < a.packed_rows; ++r) {
        device bfloat* pad = QA + (hq * a.packed_rows + r) * a.width;
        *(device bfloat4*)(pad + 4 * lane) = ya;
        *(device bfloat4*)(pad + 128 + 4 * lane) = yb;
      }
  }
  if (DEC && !query) {
    const uint hk = head - a.heads;
    const int st = nodes[4 * row + 1], base = 8 + st * 12;
    const uint slot = uint(meta[base + 4]) + row - uint(meta[base + 6]);
    device bfloat* key = caches[st].keys + ulong(hk) * uint(meta[base + 7]) + slot * a.width;
    *(device bfloat4*)(key + 4 * lane) = ya;
    *(device bfloat4*)(key + 128 + 4 * lane) = yb;
    device bfloat* value = caches[st].values + ulong(hk) * uint(meta[base + 8]) + slot * a.width;
    device const bfloat* source = V + row * v_stride + hk * a.width;
    *(device bfloat4*)(value + 4 * lane) = *(device const bfloat4*)(source + 4 * lane);
    *(device bfloat4*)(value + 128 + 4 * lane) = *(device const bfloat4*)(source + 128 + 4 * lane);
  }
}

// wait_for above 0 (decoded verify rows): the counted q+gate and k|v projections share this launch's barrier interval and
// count their threadgroups on NC[0] past a device fence; each group reads them once the count reaches wait_for (bounded),
// past its own fence, and the last group leaves both words zero for the next pair (as tf_delta_gdn_pre_rows does)
#define Q27_HEAD_NORM_ROPE(NAME, PACK) \
kernel void NAME(device const bfloat* Q [[buffer(0)]], device const bfloat* K [[buffer(1)]], device const bfloat* QW [[buffer(2)]], \
                 device const bfloat* KW [[buffer(3)]], device const int* positions [[buffer(4)]], constant HeadRope& a [[buffer(5)]], \
                 device bfloat* QY [[buffer(6)]], device bfloat* KY [[buffer(7)]], device atomic_uint* NC [[buffer(8)]], \
                 constant uint& wait_for [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]], \
                 uint2 groups [[threadgroups_per_grid]], uint t [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  if (wait_for != 0u) { \
    threadgroup uint dependency_ready; \
    if (t == 0) { \
      for (uint spin = 0; spin < (1u << 24) && atomic_load_explicit(&NC[0], memory_order_relaxed) < wait_for; spin++) {} \
      dependency_ready = uint(atomic_load_explicit(&NC[0], memory_order_relaxed) >= wait_for); \
      if (dependency_ready == 0u) { \
        atomic_store_explicit(&NC[2], 1u, memory_order_relaxed); \
      } else { \
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); \
        if (atomic_fetch_add_explicit(&NC[1], 1u, memory_order_relaxed) + 1u == groups.x * groups.y) { \
          atomic_store_explicit(&NC[0], 0u, memory_order_relaxed); \
          atomic_store_explicit(&NC[1], 0u, memory_order_relaxed); \
        } \
      } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (dependency_ready == 0u) return; \
  } \
  q27_head_norm_rope_rows<PACK, false>(Q, K, QW, KW, positions, a, QY, KY, group, sg, lane); \
}
Q27_HEAD_NORM_ROPE(q27_head_norm_rope, false)
Q27_HEAD_NORM_ROPE(q27_head_norm_rope_packed, true)
#undef Q27_HEAD_NORM_ROPE

// The decoded rows' one-pass kernel (DEC): q27_head_norm_rope's outputs, the packed query the prefix reads and the round's
// keys and values in their cache slots, so no q27_attn_pack_a or q27_attn_append follows. It waits on its counted
// projections as q27_head_norm_rope does; count_out above 0: every group counts itself on PC[0] once stored, for the
// prefix that shares its barrier interval.
kernel void q27_head_norm_rope_decoded(device const bfloat* Q [[buffer(0)]], device const bfloat* K [[buffer(1)]],
                 device const bfloat* QW [[buffer(2)]], device const bfloat* KW [[buffer(3)]], device const int* positions [[buffer(4)]],
                 constant HeadRope& a [[buffer(5)]], device bfloat* QY [[buffer(6)]], device bfloat* KY [[buffer(7)]],
                 device atomic_uint* NC [[buffer(8)]], constant uint& wait_for [[buffer(9)]], device bfloat* QA [[buffer(10)]],
                 device const Q27HeadCaches* caches [[buffer(11)]], device const int* meta [[buffer(12)]],
                 device const int* nodes [[buffer(13)]], device const bfloat* V [[buffer(14)]], constant uint& v_stride [[buffer(15)]],
                 device atomic_uint* PC [[buffer(16)]], constant uint& count_out [[buffer(17)]],
                 uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
                 uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
  if (wait_for != 0u) {
    threadgroup uint dependency_ready;
    if (t == 0) {
      for (uint spin = 0; spin < (1u << 24) && atomic_load_explicit(&NC[0], memory_order_relaxed) < wait_for; spin++) {}
      dependency_ready = uint(atomic_load_explicit(&NC[0], memory_order_relaxed) >= wait_for);
      if (dependency_ready == 0u) {
        atomic_store_explicit(&NC[2], 1u, memory_order_relaxed);
      } else {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (atomic_fetch_add_explicit(&NC[1], 1u, memory_order_relaxed) + 1u == groups.x * groups.y) {
          atomic_store_explicit(&NC[0], 0u, memory_order_relaxed);
          atomic_store_explicit(&NC[1], 0u, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dependency_ready == 0u) return;
  }
  q27_head_norm_rope_rows<false, true>(Q, K, QW, KW, positions, a, QY, KY, group, sg, lane, QA, caches, meta, nodes, V, v_stride);
  if (count_out != 0u) {
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(&PC[0], 1u, memory_order_relaxed);
    }
  }
}

struct Element { uint width, heads, rows, activation; };

// SiLU(gate) times up reads stacked [gate|up] rows with one final native activation rounding.
kernel void q27_mlp(device const uchar* GU [[buffer(0)]], constant Element& a [[buffer(1)]],
                    device uchar* OUT [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.rows) return;
  const float gate = q27_read(GU, at.y * 2 * a.width + at.x, a.activation);
  const float up = q27_read(GU, at.y * 2 * a.width + a.width + at.x, a.activation);
  q27_write(OUT, at.y * a.width + at.x, gate / (1.0f + exp(-gate)) * up, a.activation);
}

// Attention uses the native-rounded sigmoid before multiplying its output, matching the Metal row family.
kernel void q27_gate(device const uchar* O [[buffer(0)]], device const uchar* QG [[buffer(1)]],
                     constant Element& a [[buffer(2)]], device uchar* OUT [[buffer(3)]],
                     uint3 at [[thread_position_in_grid]]) {
  if (at.x >= a.width || at.y >= a.heads || at.z >= a.rows) return;
  const uint index = (at.z * a.heads + at.y) * a.width + at.x;
  const float gate = q27_read(QG, (at.z * a.heads + at.y) * 2 * a.width + a.width + at.x, a.activation);
  const float sig = q27_round(1.0f / (1.0f + exp(-gate)), a.activation);
  q27_write(OUT, index, q27_read(O, index, a.activation) * sig, a.activation);
}

struct Post { uint width, heads, rows, stride, offset, activation, gain; float eps; };

// GDN post-norm retains the rounded normalized result before the fp32 SiLU product, as row_glue.gdn_post.
kernel void q27_gdn_post(device const uchar* REC [[buffer(0)]], device const uchar* Z [[buffer(1)]],
                         device const uchar* W [[buffer(2)]], constant Post& a [[buffer(3)]],
                         device uchar* OUT [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint head = group.y, row = group.z, per = a.width / 32;
  float values[16], square = 0.0f;
  for (uint j = 0; j < per; ++j) {
    values[j] = q27_read(REC, (row * a.heads + head) * a.width + lane * per + j, a.activation);
    square += values[j] * values[j];
  }
  const float inv = rsqrt(simd_sum(square) / float(a.width) + a.eps);
  for (uint j = 0; j < per; ++j) {
    const uint d = lane * per + j;
    const float norm = q27_round(q27_read(W, d, a.gain) * (values[j] * inv), a.activation);
    const float z = q27_read(Z, row * a.stride + a.offset + head * a.width + d, a.activation);
    q27_write(OUT, (row * a.heads + head) * a.width + d, z / (1.0f + exp(-z)) * norm, a.activation);
  }
}
