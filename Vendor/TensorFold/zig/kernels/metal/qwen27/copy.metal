// Copies preserve BF16 bits while projection stacks become independent row-major views.
#include <metal_stdlib>
using namespace metal;

struct Slice { uint rows, width, stride, offset; };

kernel void q27_slice(device const ushort* X [[buffer(0)]], constant Slice& p [[buffer(1)]],
                      device ushort* Y [[buffer(2)]], uint2 t [[thread_position_in_grid]]) {
  if (t.x < p.width && t.y < p.rows) Y[t.y * p.width + t.x] = X[t.y * p.stride + p.offset + t.x];
}

struct Add { uint rows, width; };

kernel void q27_add(device const bfloat* X [[buffer(0)]], device const bfloat* R [[buffer(1)]],
                    constant Add& p [[buffer(2)]], device bfloat* Y [[buffer(3)]],
                    uint2 t [[thread_position_in_grid]]) {
  if (t.x < p.width && t.y < p.rows) {
    const uint i = t.y * p.width + t.x;
    Y[i] = bfloat(float(X[i]) + float(R[i]));
  }
}

struct Gather { uint rows, width, capacity, planes; };

kernel void q27_tap_gather(device const ushort* X [[buffer(0)]], device const uint* kept [[buffer(1)]],
                           constant Gather& p [[buffer(2)]], device ushort* Y [[buffer(3)]],
                           uint2 t [[thread_position_in_grid]]) {
  if (t.x >= p.width * p.planes || t.y >= p.rows || kept[t.y] >= p.capacity) return;
  const uint plane = t.x / p.width, column = t.x % p.width;
  Y[t.y * (p.width * p.planes) + t.x] = X[(plane * p.capacity + kept[t.y]) * p.width + column];
}

struct Take { uint rows, width; };

// An image row's bits over its placeholder's embedding (take[r]: its row in R, or -1 to keep the embedding).
[[max_total_threads_per_threadgroup(256)]] kernel void q27_take(device ushort* X [[buffer(0)]], device const ushort* R [[buffer(1)]],
                     device const int* take [[buffer(2)]], constant Take& p [[buffer(3)]],
                     uint2 t [[thread_position_in_grid]]) {
  if (t.x >= p.width || t.y >= p.rows || take[t.y] < 0) return;
  X[t.y * p.width + t.x] = R[ulong(take[t.y]) * p.width + t.x];
}

// Pulls a weight span's 128-byte lines into the GPU's caches ahead of its projection, while a latency-bound kernel
// leaves DRAM idle (the decoded GDN scan, decoded attention): one 16-byte load a line fetches the whole line. The guard
// (host values: key, never) is never true, as thread i < lines < never, but the compiler cannot prove it, so the loads
// stay and nothing is ever written.
kernel void q27_touch(device const uint4* W [[buffer(0)]], constant uint4& p [[buffer(1)]], device uint* sink [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= p.x) return; // p.x: lines, p.y: key, p.z: never
  const uint4 v = W[(size_t)i * 8];
  if (i == p.z && (v.x ^ v.y ^ v.z ^ v.w) == p.y) sink[0] = v.x;
}
