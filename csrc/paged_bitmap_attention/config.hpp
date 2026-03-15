#pragma once
/**
 * Paged Bitmap Attention — compile-time configuration and constants.
 */

#include <cuda_fp16.h>
#include <cfloat>

namespace paged_bitmap {

// KV block size — matches bitmap granularity (BLOCK_N in Triton version)
constexpr int BLOCK_N = 32;

// exp2f is a single HW instruction vs expf which compiles to multiple.
// Convert: exp(x) = exp2(x * log2(e)) where log2(e) = 1.44269504f
constexpr float LOG2E = 1.44269504f;

// Vectorization width for float4 loads (8 halves per float4)
constexpr int VEC_SIZE = 8;

// WMMA tile dimensions (m16n16k16)
constexpr int WMMA_TILE = 16;

// Thread model: always 4 warps = 128 threads regardless of BLOCK_M
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * 32;  // 128

}  // namespace paged_bitmap
