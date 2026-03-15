/**
 * Bitmap-driven sparse attention CUDA kernel for tree speculative decoding.
 *
 * Two-stage architecture with distinct block selection strategies:
 *
 *   Stage 1 (block-index): Attention to past context [0, past_len).
 *     Uses a precomputed int32 array of block IDs to visit. For dense past
 *     context, this is simply [0, 1, ..., num_past_blocks-1]. The index-based
 *     design enables future sparse past-context strategies (top-k KV selection,
 *     sliding window) without kernel changes.
 *
 *   Stage 2 (bit-level bitmap): Attention over tree region [past_len, past_len + N_tree).
 *     Uses uint64 bitmaps to skip KV blocks with no ancestors. Fine-grained
 *     traversal via __ffsll(mask) for ctz + mask &= (mask-1) to clear lowest
 *     set bit. One bitmap per q_block (all Q rows share the same visibility).
 *
 * Build: cd csrc && python setup_bitmap_attention.py build_ext --inplace
 * Usage: import bitmap_attention; bitmap_attention.bitmap_attention_fwd(...)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// Include CUTLASS for types and utilities
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

// ============================================================================
// KERNEL CONFIGURATION
// ============================================================================

// KV block size — matches bitmap granularity
constexpr int BLOCK_N = 32;

// exp2f is a single HW instruction vs expf which compiles to multiple.
// Convert: exp(x) = exp2(x * log2(e)) where log2(e) = 1.44269504f
constexpr float LOG2E = 1.44269504f;

// ============================================================================
// DEVICE HELPERS
// ============================================================================

/// Collaborative tile load: all threads in block load a [rows, cols] half tile
/// from global memory into shared memory. Uses float4 vectorized loads (8 halves
/// per transaction). Requires cols and gmem_stride to be multiples of 8.
__device__ __forceinline__ void load_tile_to_smem(
    half* __restrict__ smem,
    const half* __restrict__ gmem,
    int rows, int cols, int gmem_stride,
    int num_threads, int tid)
{
    const int vec_cols = cols / 8;

    int total_vecs = rows * vec_cols;
    float4* smem4 = reinterpret_cast<float4*>(smem);
    for (int idx = tid; idx < total_vecs; idx += num_threads) {
        int r = idx / vec_cols;
        int vc = idx % vec_cols;
        smem4[r * vec_cols + vc] =
            reinterpret_cast<const float4*>(gmem + r * gmem_stride)[vc];
    }
}

/// Collaborative tile load with bounds checking. Uses float4 vectorized loads
/// (8 halves per transaction). Requires cols and gmem_stride to be multiples of 8.
__device__ __forceinline__ void load_tile_to_smem_bounded(
    half* __restrict__ smem,
    const half* __restrict__ gmem,
    int valid_rows, int cols, int gmem_stride,
    int smem_rows, int num_threads, int tid)
{
    const int vec_cols = cols / 8;

    int total_vecs = smem_rows * vec_cols;
    float4* smem4 = reinterpret_cast<float4*>(smem);
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int idx = tid; idx < total_vecs; idx += num_threads) {
        int r = idx / vec_cols;
        int vc = idx % vec_cols;
        if (r < valid_rows) {
            smem4[r * vec_cols + vc] =
                reinterpret_cast<const float4*>(gmem + r * gmem_stride)[vc];
        } else {
            smem4[r * vec_cols + vc] = zero4;
        }
    }
}


/// Vectorized tree mask tile load: loads [BM][BLOCK_N] from padded global
/// tree_mask into SMEM using float4 (8 halves per transaction).
/// Templatized on BM for compile-time loop bound → full unrolling.
/// Requires: BLOCK_N multiple of 8, mask_stride multiple of 8.
/// Tree mask must be pre-padded with -inf beyond valid [N_Q, N_Q] region.
template <int BM>
__device__ __forceinline__ void load_mask_tile_vectorized(
    half* __restrict__ smem_tmask,
    const half* __restrict__ tree_mask_base,
    int q_start, int kv_col_start,
    int mask_stride, int num_threads, int tid)
{
    constexpr int VEC_COLS = BLOCK_N / 8;  // 32/8 = 4
    constexpr int total_vecs = BM * VEC_COLS;
    float4* smem4 = reinterpret_cast<float4*>(smem_tmask);

    #pragma unroll
    for (int idx = tid; idx < total_vecs; idx += num_threads) {
        int row = idx / VEC_COLS;
        int vc = idx % VEC_COLS;
        smem4[row * VEC_COLS + vc] = reinterpret_cast<const float4*>(
            tree_mask_base + (q_start + row) * mask_stride + kv_col_start)[vc];
    }
}


// ============================================================================
// MAIN KERNEL: Bitmap-driven sparse attention
// ============================================================================
//
// Template parameters:
//   HEAD_DIM:  head dimension (e.g. 128)
//   BLOCK_M:   query block size = number of threads per block
//
// Each thread handles one query row. Thread t computes output for query
// position (q_block * BLOCK_M + t).
//
// Shared memory layout:
//   smem_q [BLOCK_M][HEAD_DIM]       — Q tile, loaded once per kernel
//   smem_kv[BLOCK_N][HEAD_DIM]       — K or V tile, reloaded per KV block
//   smem_tmask[BLOCK_M][BLOCK_N]     — tree mask tile (Stage 2 only), vectorized load
//
// Stage 1: Iterates over block IDs from stage1_block_indices[] (coarse index).
// Stage 2: Iterates over set bits in uint64 bitmaps[] (fine-grained bitmap).
//

template <int HEAD_DIM, int BLOCK_M>
__global__ void bitmap_attention_fwd_kernel(
    const half* __restrict__ Q,           // [B, H, N_Q, D]
    const half* __restrict__ K,           // [B, H, N_KV, D]
    const half* __restrict__ V,           // [B, H, N_KV, D]
    half* __restrict__ O,                 // [B, H, N_Q, D]
    const half* __restrict__ tree_mask,   // [N_Q, N_Q] semantic mask (0 or -inf)
    const int32_t* __restrict__ stage1_block_indices,  // [num_stage1_blocks] block IDs for past context
    const int64_t* __restrict__ bitmaps,  // [num_q_blocks, W] bitmap for tree region
    const float sm_scale,
    const int past_len,
    const int N_Q,                        // number of tree query positions
    const int N_KV,                       // total KV length = past_len + N_Q
    const int num_stage1_blocks,          // number of past context blocks to visit
    const int W,                          // number of bitmap words per q_block
    const int mask_stride,                // padded tree mask row stride (in halves)
    const int H                           // number of heads
) {
    // Grid: (num_q_blocks, H, B)
    const int q_block = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;

    const int q_start = q_block * BLOCK_M;
    const int my_q = q_start + tid;  // this thread's query position
    const bool valid_q = (my_q < N_Q);

    // Pointers for this (batch, head)
    const half* Q_bh = Q + ((int64_t)batch * H * N_Q + (int64_t)head * N_Q) * HEAD_DIM;
    const half* K_bh = K + ((int64_t)batch * H * N_KV + (int64_t)head * N_KV) * HEAD_DIM;
    const half* V_bh = V + ((int64_t)batch * H * N_KV + (int64_t)head * N_KV) * HEAD_DIM;
    half* O_bh = O + ((int64_t)batch * H * N_Q + (int64_t)head * N_Q) * HEAD_DIM;

    // Dynamic shared memory
    extern __shared__ char smem_raw[];
    half* smem_q  = reinterpret_cast<half*>(smem_raw);
    half* smem_kv = smem_q + BLOCK_M * HEAD_DIM;
    // Tree mask tile in SMEM (Stage 2): [BLOCK_M][BLOCK_N] half
    half* smem_tmask = smem_kv + BLOCK_N * HEAD_DIM;

    // ---- Load Q tile into shared memory (once per kernel) ----
    load_tile_to_smem_bounded(
        smem_q, Q_bh + q_start * HEAD_DIM,
        min(BLOCK_M, N_Q - q_start), HEAD_DIM, HEAD_DIM,
        BLOCK_M, BLOCK_M, tid);
    __syncthreads();

    // Per-thread accumulators (register-resident)
    float o_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) o_acc[d] = 0.0f;
    float m_i = -FLT_MAX;   // running max (for online softmax)
    float l_i = 0.0f;       // running sum of exp

    // Pointer to this thread's Q row in shared memory
    const half* my_q_row = smem_q + tid * HEAD_DIM;

    // Pre-multiply scale by LOG2E so scores are in log2 space.
    // This lets us use exp2f(s - m) directly without per-exp LOG2E multiply.
    const float qk_scale = sm_scale * LOG2E;

    // ==================================================================
    // Stage 1: Block-index-driven attention to past context
    // Iterates over a precomputed list of block IDs rather than all
    // blocks in [0, past_len). For dense past context, the index array
    // is simply [0, 1, 2, ..., num_past_blocks-1].
    // ==================================================================
    for (int bi = 0; bi < num_stage1_blocks; bi++) {
        int block_id = stage1_block_indices[bi];
        int kv_start = block_id * BLOCK_N;
        int kv_len = min(BLOCK_N, past_len - kv_start);
        if (kv_len <= 0) continue;

        // Load K tile
        load_tile_to_smem_bounded(
            smem_kv, K_bh + kv_start * HEAD_DIM,
            kv_len, HEAD_DIM, HEAD_DIM,
            BLOCK_N, BLOCK_M, tid);
        __syncthreads();

        // Compute QK^T for this tile (half2 dot products)
        float s_vals[BLOCK_N];
        if (valid_q) {
            const half2* q2 = reinterpret_cast<const half2*>(my_q_row);
            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; j++) {
                float dot = 0.0f;
                const half2* k2 = reinterpret_cast<const half2*>(smem_kv + j * HEAD_DIM);
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM / 2; d++) {
                    float2 prod = __half22float2(__hmul2(q2[d], k2[d]));
                    dot += prod.x + prod.y;
                }
                s_vals[j] = (j < kv_len) ? (dot * qk_scale) : -FLT_MAX;
            }
        }
        __syncthreads();

        // Load V tile (reuse smem_kv)
        load_tile_to_smem_bounded(
            smem_kv, V_bh + kv_start * HEAD_DIM,
            kv_len, HEAD_DIM, HEAD_DIM,
            BLOCK_N, BLOCK_M, tid);
        __syncthreads();

        // Online softmax update + PV accumulation (half2 loads)
        if (valid_q) {
            float m_block = -FLT_MAX;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j++) {
                m_block = fmaxf(m_block, s_vals[j]);
            }

            float m_new = fmaxf(m_i, m_block);
            float scale_old = exp2f(m_i - m_new);

            l_i *= scale_old;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                o_acc[d] *= scale_old;
            }

            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; j++) {
                float p_j = exp2f(s_vals[j] - m_new);
                l_i += p_j;
                const half2* v2 = reinterpret_cast<const half2*>(smem_kv + j * HEAD_DIM);
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM / 2; d++) {
                    float2 v_f2 = __half22float2(v2[d]);
                    o_acc[2*d]   += p_j * v_f2.x;
                    o_acc[2*d+1] += p_j * v_f2.y;
                }
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    // ==================================================================
    // Stage 2: Bitmap-driven sparse tree attention
    // ==================================================================
    const int64_t* my_bitmap_row = bitmaps + q_block * W;

    for (int w = 0; w < W; w++) {
        // All threads read same bitmap word — L1 cache serves all warps
        uint64_t mask = static_cast<uint64_t>(my_bitmap_row[w]);

        // Traverse set bits: each set bit = one KV block to process
        while (mask != 0ULL) {
            int bit = __ffsll(static_cast<long long>(mask)) - 1;
            mask &= (mask - 1);  // clear lowest set bit

            int kv_block_id = w * 64 + bit;
            int kv_global_start = past_len + kv_block_id * BLOCK_N;
            int kv_len = min(BLOCK_N, N_KV - kv_global_start);
            if (kv_len <= 0) break;

            // Load K tile from tree region
            load_tile_to_smem_bounded(
                smem_kv, K_bh + kv_global_start * HEAD_DIM,
                kv_len, HEAD_DIM, HEAD_DIM,
                BLOCK_N, BLOCK_M, tid);

            // Load tree mask tile [BLOCK_M, BLOCK_N] into SMEM (vectorized float4)
            load_mask_tile_vectorized<BLOCK_M>(
                smem_tmask, tree_mask,
                q_start, kv_block_id * BLOCK_N,
                mask_stride, BLOCK_M, tid);
            __syncthreads();

            // Compute QK^T + apply tree mask (half2 dot products)
            float s_vals[BLOCK_N];
            if (valid_q) {
                const half2* q2 = reinterpret_cast<const half2*>(my_q_row);
                #pragma unroll 4
                for (int j = 0; j < BLOCK_N; j++) {
                    float dot = 0.0f;
                    const half2* k2 = reinterpret_cast<const half2*>(smem_kv + j * HEAD_DIM);
                    #pragma unroll 8
                    for (int d = 0; d < HEAD_DIM / 2; d++) {
                        float2 prod = __half22float2(__hmul2(q2[d], k2[d]));
                        dot += prod.x + prod.y;
                    }
                    dot *= qk_scale;

                    // Apply tree mask from SMEM (0.0 = attend, -inf = mask out)
                    // Mask values are {0, -inf} — safe to add in log2 space
                    if (j < kv_len) {
                        dot += __half2float(smem_tmask[tid * BLOCK_N + j]);
                    } else {
                        dot = -FLT_MAX;
                    }
                    s_vals[j] = dot;
                }
            }
            __syncthreads();

            // Load V tile (reuse smem_kv)
            load_tile_to_smem_bounded(
                smem_kv, V_bh + kv_global_start * HEAD_DIM,
                kv_len, HEAD_DIM, HEAD_DIM,
                BLOCK_N, BLOCK_M, tid);
            __syncthreads();

            // Online softmax update + PV accumulation (half2 loads)
            if (valid_q) {
                float m_block = -FLT_MAX;
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j++) {
                    m_block = fmaxf(m_block, s_vals[j]);
                }

                float m_new = fmaxf(m_i, m_block);
                float scale_old = exp2f(m_i - m_new);

                l_i *= scale_old;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; d++) {
                    o_acc[d] *= scale_old;
                }

                #pragma unroll 4
                for (int j = 0; j < BLOCK_N; j++) {
                    float p_j = exp2f(s_vals[j] - m_new);
                    l_i += p_j;
                    const half2* v2 = reinterpret_cast<const half2*>(smem_kv + j * HEAD_DIM);
                    #pragma unroll 8
                    for (int d = 0; d < HEAD_DIM / 2; d++) {
                        float2 v_f2 = __half22float2(v2[d]);
                        o_acc[2*d]   += p_j * v_f2.x;
                        o_acc[2*d+1] += p_j * v_f2.y;
                    }
                }
                m_i = m_new;
            }
            __syncthreads();
        }
    }

    // ==================================================================
    // Finalize: normalize output and write to global memory (half2 stores)
    // ==================================================================
    if (valid_q) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        half2* O_row = reinterpret_cast<half2*>(O_bh + my_q * HEAD_DIM);
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM / 2; d++) {
            O_row[d] = __floats2half2_rn(o_acc[2*d] * inv_l, o_acc[2*d+1] * inv_l);
        }
    }
}


// ============================================================================
// DENSE ATTENTION KERNEL (for comparison benchmarking)
// ============================================================================
// Same architecture but Stage 2 iterates ALL tree KV blocks (no bitmap).

template <int HEAD_DIM, int BLOCK_M>
__global__ void dense_attention_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const half* __restrict__ tree_mask,
    const float sm_scale,
    const int past_len,
    const int N_Q,
    const int N_KV,
    const int mask_stride,                // padded tree mask row stride (in halves)
    const int H
) {
    const int q_block = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;

    const int q_start = q_block * BLOCK_M;
    const int my_q = q_start + tid;
    const bool valid_q = (my_q < N_Q);

    const half* Q_bh = Q + ((int64_t)batch * H * N_Q + (int64_t)head * N_Q) * HEAD_DIM;
    const half* K_bh = K + ((int64_t)batch * H * N_KV + (int64_t)head * N_KV) * HEAD_DIM;
    const half* V_bh = V + ((int64_t)batch * H * N_KV + (int64_t)head * N_KV) * HEAD_DIM;
    half* O_bh = O + ((int64_t)batch * H * N_Q + (int64_t)head * N_Q) * HEAD_DIM;

    extern __shared__ char smem_raw[];
    half* smem_q  = reinterpret_cast<half*>(smem_raw);
    half* smem_kv = smem_q + BLOCK_M * HEAD_DIM;
    half* smem_tmask = smem_kv + BLOCK_N * HEAD_DIM;

    // Load Q tile
    load_tile_to_smem_bounded(
        smem_q, Q_bh + q_start * HEAD_DIM,
        min(BLOCK_M, N_Q - q_start), HEAD_DIM, HEAD_DIM,
        BLOCK_M, BLOCK_M, tid);
    __syncthreads();

    float o_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) o_acc[d] = 0.0f;
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const half* my_q_row = smem_q + tid * HEAD_DIM;

    // Pre-multiply scale by LOG2E (scores in log2 space, saves one FMUL per exp2f)
    const float qk_scale = sm_scale * LOG2E;

    // Stage 1: Dense past context (half2 dot products + vectorized PV)
    for (int kv_start = 0; kv_start < past_len; kv_start += BLOCK_N) {
        int kv_len = min(BLOCK_N, past_len - kv_start);

        load_tile_to_smem_bounded(
            smem_kv, K_bh + kv_start * HEAD_DIM,
            kv_len, HEAD_DIM, HEAD_DIM,
            BLOCK_N, BLOCK_M, tid);
        __syncthreads();

        float s_vals[BLOCK_N];
        if (valid_q) {
            const half2* q2 = reinterpret_cast<const half2*>(my_q_row);
            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; j++) {
                float dot = 0.0f;
                const half2* k2 = reinterpret_cast<const half2*>(smem_kv + j * HEAD_DIM);
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM / 2; d++) {
                    float2 prod = __half22float2(__hmul2(q2[d], k2[d]));
                    dot += prod.x + prod.y;
                }
                s_vals[j] = (j < kv_len) ? (dot * qk_scale) : -FLT_MAX;
            }
        }
        __syncthreads();

        load_tile_to_smem_bounded(
            smem_kv, V_bh + kv_start * HEAD_DIM,
            kv_len, HEAD_DIM, HEAD_DIM,
            BLOCK_N, BLOCK_M, tid);
        __syncthreads();

        if (valid_q) {
            float m_block = -FLT_MAX;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j++) {
                m_block = fmaxf(m_block, s_vals[j]);
            }
            float m_new = fmaxf(m_i, m_block);
            float scale_old = exp2f(m_i - m_new);
            l_i *= scale_old;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) o_acc[d] *= scale_old;
            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; j++) {
                float p_j = exp2f(s_vals[j] - m_new);
                l_i += p_j;
                const half2* v2 = reinterpret_cast<const half2*>(smem_kv + j * HEAD_DIM);
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM / 2; d++) {
                    float2 v_f2 = __half22float2(v2[d]);
                    o_acc[2*d]   += p_j * v_f2.x;
                    o_acc[2*d+1] += p_j * v_f2.y;
                }
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    // Stage 2: Dense tree attention (iterate ALL KV blocks)
    int tree_kv_blocks = (N_Q + BLOCK_N - 1) / BLOCK_N;
    for (int kv_block_id = 0; kv_block_id < tree_kv_blocks; kv_block_id++) {
        int kv_global_start = past_len + kv_block_id * BLOCK_N;
        int kv_len = min(BLOCK_N, N_KV - kv_global_start);
        if (kv_len <= 0) break;

        load_tile_to_smem_bounded(
            smem_kv, K_bh + kv_global_start * HEAD_DIM,
            kv_len, HEAD_DIM, HEAD_DIM,
            BLOCK_N, BLOCK_M, tid);

        // Load tree mask tile into SMEM (vectorized float4)
        load_mask_tile_vectorized<BLOCK_M>(
            smem_tmask, tree_mask,
            q_start, kv_block_id * BLOCK_N,
            mask_stride, BLOCK_M, tid);
        __syncthreads();

        float s_vals[BLOCK_N];
        if (valid_q) {
            const half2* q2 = reinterpret_cast<const half2*>(my_q_row);
            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; j++) {
                float dot = 0.0f;
                const half2* k2 = reinterpret_cast<const half2*>(smem_kv + j * HEAD_DIM);
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM / 2; d++) {
                    float2 prod = __half22float2(__hmul2(q2[d], k2[d]));
                    dot += prod.x + prod.y;
                }
                dot *= qk_scale;
                if (j < kv_len) {
                    dot += __half2float(smem_tmask[tid * BLOCK_N + j]);
                } else {
                    dot = -FLT_MAX;
                }
                s_vals[j] = dot;
            }
        }
        __syncthreads();

        load_tile_to_smem_bounded(
            smem_kv, V_bh + kv_global_start * HEAD_DIM,
            kv_len, HEAD_DIM, HEAD_DIM,
            BLOCK_N, BLOCK_M, tid);
        __syncthreads();

        if (valid_q) {
            float m_block = -FLT_MAX;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j++) {
                m_block = fmaxf(m_block, s_vals[j]);
            }
            float m_new = fmaxf(m_i, m_block);
            float scale_old = exp2f(m_i - m_new);
            l_i *= scale_old;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) o_acc[d] *= scale_old;
            #pragma unroll 4
            for (int j = 0; j < BLOCK_N; j++) {
                float p_j = exp2f(s_vals[j] - m_new);
                l_i += p_j;
                const half2* v2 = reinterpret_cast<const half2*>(smem_kv + j * HEAD_DIM);
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM / 2; d++) {
                    float2 v_f2 = __half22float2(v2[d]);
                    o_acc[2*d]   += p_j * v_f2.x;
                    o_acc[2*d+1] += p_j * v_f2.y;
                }
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    // Finalize (vectorized half2 output writes)
    if (valid_q) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        half2* O_row = reinterpret_cast<half2*>(O_bh + my_q * HEAD_DIM);
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM / 2; d++) {
            O_row[d] = __floats2half2_rn(o_acc[2*d] * inv_l, o_acc[2*d+1] * inv_l);
        }
    }
}


// ============================================================================
// HOST FUNCTIONS
// ============================================================================

/// Compute shared memory size for the bitmap attention kernel
static int get_smem_size(int block_m, int head_dim) {
    // smem_q[BLOCK_M][HEAD_DIM] + smem_kv[BLOCK_N][HEAD_DIM]
    // + smem_tmask[BLOCK_M][BLOCK_N]
    return (block_m + BLOCK_N) * head_dim * sizeof(half)
         + block_m * BLOCK_N * sizeof(half);
}

/// Pad tree_mask from [N_Q, N_Q] to [pad_rows, pad_cols] with -inf,
/// ensuring all vectorized tile loads are in-bounds and aligned.
static torch::Tensor pad_tree_mask(
    torch::Tensor tree_mask, int N_Q, int block_m)
{
    int num_q_blocks = (N_Q + block_m - 1) / block_m;
    int num_kv_blocks = (N_Q + BLOCK_N - 1) / BLOCK_N;
    int pad_rows = num_q_blocks * block_m;
    int pad_cols = num_kv_blocks * BLOCK_N;

    if (pad_rows == N_Q && pad_cols == N_Q) {
        return tree_mask;  // no padding needed
    }

    auto padded = torch::full(
        {pad_rows, pad_cols},
        -std::numeric_limits<float>::infinity(),
        tree_mask.options());
    padded.narrow(0, 0, N_Q).narrow(1, 0, N_Q).copy_(tree_mask);
    return padded.contiguous();
}

/// Launch bitmap attention kernel with template dispatch
torch::Tensor bitmap_attention_fwd(
    torch::Tensor Q,                      // [B, H, N_Q, D]
    torch::Tensor K,                      // [B, H, N_KV, D]
    torch::Tensor V,                      // [B, H, N_KV, D]
    torch::Tensor tree_mask,              // [N_Q, N_Q]
    torch::Tensor stage1_block_indices,   // [num_stage1_blocks] int32
    torch::Tensor bitmaps,                // [num_q_blocks, W]
    double sm_scale,
    int64_t past_len,
    int64_t W_val
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H, N_Q, D]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N_Q = Q.size(2);
    int D = Q.size(3);
    int N_KV = K.size(2);

    TORCH_CHECK(D == 64 || D == 128, "HEAD_DIM must be 64 or 128");
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(3) == D);
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == N_KV && V.size(3) == D);
    TORCH_CHECK(stage1_block_indices.scalar_type() == torch::kInt32,
                "stage1_block_indices must be int32");

    // Contiguous check
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    tree_mask = tree_mask.contiguous();
    stage1_block_indices = stage1_block_indices.contiguous();
    bitmaps = bitmaps.contiguous();

    auto O = torch::zeros({B, H, N_Q, D}, Q.options());

    // Choose BLOCK_M based on N_Q for efficiency
    int block_m;
    if (N_Q <= 32) block_m = 32;
    else if (N_Q <= 64) block_m = 64;
    else block_m = 128;

    // Pad tree_mask for vectorized float4 loading
    tree_mask = pad_tree_mask(tree_mask, N_Q, block_m);
    int mask_stride_int = static_cast<int>(tree_mask.stride(0));

    int num_q_blocks = (N_Q + block_m - 1) / block_m;
    int num_stage1 = static_cast<int>(stage1_block_indices.size(0));
    dim3 grid(num_q_blocks, H, B);
    int smem_size = get_smem_size(block_m, D);

    float scale_f = static_cast<float>(sm_scale);
    int W_int = static_cast<int>(W_val);

    // Dispatch on HEAD_DIM and BLOCK_M
    #define LAUNCH_KERNEL(HD, BM) do { \
        auto kern = bitmap_attention_fwd_kernel<HD, BM>; \
        cudaFuncSetAttribute(kern, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
        kern<<<grid, BM, smem_size>>>( \
            reinterpret_cast<const half*>(Q.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(K.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(V.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(O.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(tree_mask.data_ptr<at::Half>()), \
            stage1_block_indices.data_ptr<int32_t>(), \
            bitmaps.data_ptr<int64_t>(), \
            scale_f, static_cast<int>(past_len), N_Q, N_KV, \
            num_stage1, W_int, mask_stride_int, H); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } while (0)

    if (D == 128) {
        if (block_m == 32) LAUNCH_KERNEL(128, 32);
        else if (block_m == 64) LAUNCH_KERNEL(128, 64);
        else LAUNCH_KERNEL(128, 128);
    } else {  // D == 64
        if (block_m == 32) LAUNCH_KERNEL(64, 32);
        else if (block_m == 64) LAUNCH_KERNEL(64, 64);
        else LAUNCH_KERNEL(64, 128);
    }

    #undef LAUNCH_KERNEL

    return O;
}

/// Launch dense attention kernel (for benchmarking comparison)
torch::Tensor dense_attention_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor tree_mask,
    double sm_scale,
    int64_t past_len
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H, N_Q, D]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N_Q = Q.size(2);
    int D = Q.size(3);
    int N_KV = K.size(2);

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    tree_mask = tree_mask.contiguous();

    auto O = torch::zeros({B, H, N_Q, D}, Q.options());

    int block_m;
    if (N_Q <= 32) block_m = 32;
    else if (N_Q <= 64) block_m = 64;
    else block_m = 128;

    // Pad tree_mask for vectorized float4 loading
    tree_mask = pad_tree_mask(tree_mask, N_Q, block_m);
    int mask_stride_int = static_cast<int>(tree_mask.stride(0));

    int num_q_blocks = (N_Q + block_m - 1) / block_m;
    dim3 grid(num_q_blocks, H, B);
    // Dense kernel: Q + KV + tree mask tile
    int smem_size = (block_m + BLOCK_N) * D * sizeof(half)
                  + block_m * BLOCK_N * sizeof(half);

    float scale_f = static_cast<float>(sm_scale);

    #define LAUNCH_DENSE(HD, BM) do { \
        auto kern = dense_attention_fwd_kernel<HD, BM>; \
        cudaFuncSetAttribute(kern, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
        kern<<<grid, BM, smem_size>>>( \
            reinterpret_cast<const half*>(Q.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(K.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(V.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(O.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(tree_mask.data_ptr<at::Half>()), \
            scale_f, static_cast<int>(past_len), N_Q, N_KV, \
            mask_stride_int, H); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } while (0)

    if (D == 128) {
        if (block_m == 32) LAUNCH_DENSE(128, 32);
        else if (block_m == 64) LAUNCH_DENSE(128, 64);
        else LAUNCH_DENSE(128, 128);
    } else {
        if (block_m == 32) LAUNCH_DENSE(64, 32);
        else if (block_m == 64) LAUNCH_DENSE(64, 64);
        else LAUNCH_DENSE(64, 128);
    }

    #undef LAUNCH_DENSE

    return O;
}


// ============================================================================
// PYTHON BINDINGS
// ============================================================================

/// Return the BLOCK_M the kernel will use for a given N_Q.
/// Callers must precompute bitmaps with this BLOCK_M to match.
int64_t get_block_m(int64_t N_Q) {
    if (N_Q <= 32) return 32;
    else if (N_Q <= 64) return 64;
    else return 128;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Bitmap-driven sparse attention CUDA kernel";
    m.def("bitmap_attention_fwd", &bitmap_attention_fwd,
          "Bitmap sparse attention forward (two-stage: block-index past + bitmap tree)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("tree_mask"), py::arg("stage1_block_indices"), py::arg("bitmaps"),
          py::arg("sm_scale"), py::arg("past_len"), py::arg("W"));
    m.def("dense_attention_fwd", &dense_attention_fwd,
          "Dense attention forward (two-stage: dense past + dense tree with mask)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("tree_mask"), py::arg("sm_scale"), py::arg("past_len"));
    m.def("get_block_m", &get_block_m,
          "Return the BLOCK_M the kernel uses for a given N_Q",
          py::arg("N_Q"));
}
