/**
 * Paged Bitmap Attention CUDA kernel for tree speculative decoding.
 *
 * Combines bitmap block-skipping with zero-gather paged KV reads:
 * - K/V are stored in a flat paged buffer [max_tokens, H_kv, D]
 * - Token positions are mapped to buffer slots via kv_indices[pos] -> slot
 * - Bitmap-driven traversal skips KV blocks with no ancestors
 *
 * Two-stage architecture:
 *   Stage 1 (past context): Sequential iteration over [0, past_len).
 *     Indirect KV reads via token_slots from paged buffer. No mask.
 *
 *   Stage 2 (tree region): Bitmap-driven sparse traversal over tree tokens.
 *     uint64 bitmap traversal via __ffsll + mask &= (mask-1).
 *     Each set bit triggers: load token_slots -> indirect paged KV load ->
 *     apply tree mask -> online softmax + PV accumulate.
 *
 * GQA: multiple Q heads share one KV head (kv_head = q_head / GQA_GROUP).
 *
 * Build: cd csrc && python setup_paged_bitmap_attention.py build_ext --inplace
 * Usage: import paged_bitmap_attention_ext; paged_bitmap_attention_ext.forward(...)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

// CUTLASS headers for types
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

// ============================================================================
// KERNEL CONFIGURATION
// ============================================================================

constexpr int BLOCK_N = 32;          // KV block size (matches bitmap granularity)
constexpr float LOG2E = 1.44269504f; // log2(e) for exp2f conversion

// ============================================================================
// DEVICE HELPERS
// ============================================================================

/// Collaborative paged tile load: all threads in block cooperatively load
/// BLOCK_N rows of KV from the paged buffer into shared memory.
/// Each row may come from a different token slot.
/// kv_indices_ptr: pointer to token slot array for this region.
/// kv_start: first position index within the region.
/// valid_rows: number of valid rows (may be < BLOCK_N at boundaries).
template <int HEAD_DIM>
__device__ __forceinline__ void load_paged_tile_to_smem(
    half* __restrict__ smem,           // [BLOCK_N][HEAD_DIM]
    const half* __restrict__ kv_buf,
    const int32_t* __restrict__ kv_indices_ptr,
    int kv_start, int valid_rows, int h_kv,
    int stride_n, int stride_h,
    int num_threads, int tid)
{
    // Each thread loads multiple elements across the tile.
    // Strategy: thread tid loads elements (tid, tid+num_threads, ...) from
    // the linearized [BLOCK_N][HEAD_DIM] tile.
    constexpr int VEC_SIZE = 8;
    constexpr int VEC_COLS = HEAD_DIM / VEC_SIZE;
    const int total_vecs = BLOCK_N * VEC_COLS;

    float4* smem4 = reinterpret_cast<float4*>(smem);
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int idx = tid; idx < total_vecs; idx += num_threads) {
        int row = idx / VEC_COLS;
        int vc = idx % VEC_COLS;

        if (row < valid_rows) {
            int token_slot = kv_indices_ptr[kv_start + row];
            const float4* src4 = reinterpret_cast<const float4*>(
                kv_buf + (int64_t)token_slot * stride_n + (int64_t)h_kv * stride_h);
            smem4[row * VEC_COLS + vc] = src4[vc];
        } else {
            smem4[row * VEC_COLS + vc] = zero4;
        }
    }
}

/// Vectorized tree mask tile load: loads [BM][BLOCK_N] from padded global
/// tree_mask into SMEM using float4.
template <int BM>
__device__ __forceinline__ void load_mask_tile_vectorized(
    half* __restrict__ smem_tmask,
    const half* __restrict__ tree_mask_base,
    int q_start, int kv_col_start,
    int mask_stride, int num_threads, int tid)
{
    constexpr int VEC_COLS = BLOCK_N / 8;
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
// MAIN KERNEL: Paged Bitmap Attention
// ============================================================================
//
// Grid: (num_q_blocks, H_q)
// Threads: BLOCK_M per block (one thread per Q row)
//
// Shared memory layout:
//   smem_q     [BLOCK_M][HEAD_DIM]   — Q tile (loaded once)
//   smem_kv    [BLOCK_N][HEAD_DIM]   — K or V tile (reloaded per KV block)
//   smem_tmask [BLOCK_M][BLOCK_N]    — tree mask tile (Stage 2 only)
//

template <int HEAD_DIM, int BLOCK_M>
__global__ void paged_bitmap_attention_kernel(
    const half* __restrict__ Q,            // [H_q, N_tree, D]
    const half* __restrict__ K_buf,        // [max_tokens, H_kv, D] paged
    const half* __restrict__ V_buf,        // [max_tokens, H_kv, D] paged
    half* __restrict__ O,                  // [H_q, N_tree, D]
    const int32_t* __restrict__ kv_indices,// [total_kv] slot mapping
    const half* __restrict__ tree_mask,    // [N_tree_padded, N_tree_padded] 0/-inf
    const int64_t* __restrict__ bitmaps,   // [num_q_blocks, W] uint64
    const float sm_scale,
    const int N_tree,                      // number of tree query positions
    const int past_len,
    const int total_kv,                    // = past_len + N_tree
    const int W,                           // bitmap words per q_block
    const int mask_stride,                 // tree_mask row stride (halves)
    const int H_q,
    const int H_kv,
    const int GQA_GROUP,
    const int stride_buf_n,                // K_buf stride[0] (in halves)
    const int stride_buf_h                 // K_buf stride[1] (in halves)
) {
    const int q_block = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;

    // GQA head mapping
    const int h_kv = h_q / GQA_GROUP;

    const int q_start = q_block * BLOCK_M;
    const int my_q = q_start + tid;
    const bool valid_q = (my_q < N_tree);

    // Q and O pointers: [H_q, N_tree, D]
    const half* Q_h = Q + (int64_t)h_q * N_tree * HEAD_DIM;
    half* O_h = O + (int64_t)h_q * N_tree * HEAD_DIM;

    // Dynamic shared memory
    extern __shared__ char smem_raw[];
    half* smem_q     = reinterpret_cast<half*>(smem_raw);
    half* smem_kv    = smem_q + BLOCK_M * HEAD_DIM;
    half* smem_tmask = smem_kv + BLOCK_N * HEAD_DIM;

    // ---- Load Q tile into shared memory ----
    {
        constexpr int VEC_COLS = HEAD_DIM / 8;
        const int total_vecs = BLOCK_M * VEC_COLS;
        float4* smem4 = reinterpret_cast<float4*>(smem_q);
        const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int valid_rows = min(BLOCK_M, N_tree - q_start);
        if (valid_rows < 0) valid_rows = 0;

        for (int idx = tid; idx < total_vecs; idx += BLOCK_M) {
            int row = idx / VEC_COLS;
            int vc = idx % VEC_COLS;
            if (row < valid_rows) {
                smem4[row * VEC_COLS + vc] =
                    reinterpret_cast<const float4*>(Q_h + (q_start + row) * HEAD_DIM)[vc];
            } else {
                smem4[row * VEC_COLS + vc] = zero4;
            }
        }
    }
    __syncthreads();

    // Per-thread accumulators
    float o_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) o_acc[d] = 0.0f;
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const half* my_q_row = smem_q + tid * HEAD_DIM;
    const float qk_scale = sm_scale * LOG2E;

    // ==================================================================
    // Stage 1: Dense past context with paged KV reads
    // ==================================================================
    for (int kv_start = 0; kv_start < past_len; kv_start += BLOCK_N) {
        int kv_len = min(BLOCK_N, past_len - kv_start);

        // Collaborative paged K tile load
        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, K_buf, kv_indices,
            kv_start, kv_len, h_kv,
            stride_buf_n, stride_buf_h,
            BLOCK_M, tid);
        __syncthreads();

        // QK^T via half2 dot products
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

        // Collaborative paged V tile load (reuse smem_kv)
        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, V_buf, kv_indices,
            kv_start, kv_len, h_kv,
            stride_buf_n, stride_buf_h,
            BLOCK_M, tid);
        __syncthreads();

        // Online softmax + PV accumulation
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

    // ==================================================================
    // Stage 2: Bitmap-driven sparse tree attention with paged KV reads
    // ==================================================================
    const int64_t* my_bitmap = bitmaps + q_block * W;

    for (int w = 0; w < W; w++) {
        // All threads read same bitmap word
        uint64_t mask = static_cast<uint64_t>(my_bitmap[w]);

        while (mask != 0ULL) {
            int bit = __ffsll(static_cast<long long>(mask)) - 1;
            mask &= (mask - 1);

            int kv_block_id = w * 64 + bit;
            int kv_tree_start = kv_block_id * BLOCK_N; // offset within tree
            int kv_len = min(BLOCK_N, N_tree - kv_tree_start);
            if (kv_len <= 0) break;

            // kv_indices offset: past_len + tree position
            int kv_indices_offset = past_len + kv_tree_start;

            // Collaborative paged K tile load
            load_paged_tile_to_smem<HEAD_DIM>(
                smem_kv, K_buf, kv_indices,
                kv_indices_offset, kv_len, h_kv,
                stride_buf_n, stride_buf_h,
                BLOCK_M, tid);

            // Load tree mask tile [BLOCK_M, BLOCK_N]
            load_mask_tile_vectorized<BLOCK_M>(
                smem_tmask, tree_mask,
                q_start, kv_tree_start,
                mask_stride, BLOCK_M, tid);
            __syncthreads();

            // QK^T + tree mask
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

            // Collaborative paged V tile load
            load_paged_tile_to_smem<HEAD_DIM>(
                smem_kv, V_buf, kv_indices,
                kv_indices_offset, kv_len, h_kv,
                stride_buf_n, stride_buf_h,
                BLOCK_M, tid);
            __syncthreads();

            // Online softmax + PV accumulation
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
    }

    // ==================================================================
    // Finalize: normalize and write output
    // ==================================================================
    if (valid_q) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        half2* O_row = reinterpret_cast<half2*>(O_h + my_q * HEAD_DIM);
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM / 2; d++) {
            O_row[d] = __floats2half2_rn(o_acc[2*d] * inv_l, o_acc[2*d+1] * inv_l);
        }
    }
}


// ============================================================================
// DENSE REFERENCE KERNEL (paged, for benchmarking)
// ============================================================================
// Same two-stage arch but Stage 2 iterates ALL tree KV blocks (no bitmap).

template <int HEAD_DIM, int BLOCK_M>
__global__ void paged_dense_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K_buf,
    const half* __restrict__ V_buf,
    half* __restrict__ O,
    const int32_t* __restrict__ kv_indices,
    const half* __restrict__ tree_mask,
    const float sm_scale,
    const int N_tree,
    const int past_len,
    const int total_kv,
    const int mask_stride,
    const int H_q,
    const int H_kv,
    const int GQA_GROUP,
    const int stride_buf_n,
    const int stride_buf_h
) {
    const int q_block = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;

    const int h_kv = h_q / GQA_GROUP;

    const int q_start = q_block * BLOCK_M;
    const int my_q = q_start + tid;
    const bool valid_q = (my_q < N_tree);

    const half* Q_h = Q + (int64_t)h_q * N_tree * HEAD_DIM;
    half* O_h = O + (int64_t)h_q * N_tree * HEAD_DIM;

    extern __shared__ char smem_raw[];
    half* smem_q     = reinterpret_cast<half*>(smem_raw);
    half* smem_kv    = smem_q + BLOCK_M * HEAD_DIM;
    half* smem_tmask = smem_kv + BLOCK_N * HEAD_DIM;

    // Load Q tile
    {
        constexpr int VEC_COLS = HEAD_DIM / 8;
        const int total_vecs = BLOCK_M * VEC_COLS;
        float4* smem4 = reinterpret_cast<float4*>(smem_q);
        const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int valid_rows = min(BLOCK_M, N_tree - q_start);
        if (valid_rows < 0) valid_rows = 0;

        for (int idx = tid; idx < total_vecs; idx += BLOCK_M) {
            int row = idx / VEC_COLS;
            int vc = idx % VEC_COLS;
            if (row < valid_rows) {
                smem4[row * VEC_COLS + vc] =
                    reinterpret_cast<const float4*>(Q_h + (q_start + row) * HEAD_DIM)[vc];
            } else {
                smem4[row * VEC_COLS + vc] = zero4;
            }
        }
    }
    __syncthreads();

    float o_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) o_acc[d] = 0.0f;
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const half* my_q_row = smem_q + tid * HEAD_DIM;
    const float qk_scale = sm_scale * LOG2E;

    // Stage 1: Dense past context (paged)
    for (int kv_start = 0; kv_start < past_len; kv_start += BLOCK_N) {
        int kv_len = min(BLOCK_N, past_len - kv_start);

        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, K_buf, kv_indices,
            kv_start, kv_len, h_kv,
            stride_buf_n, stride_buf_h,
            BLOCK_M, tid);
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

        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, V_buf, kv_indices,
            kv_start, kv_len, h_kv,
            stride_buf_n, stride_buf_h,
            BLOCK_M, tid);
        __syncthreads();

        if (valid_q) {
            float m_block = -FLT_MAX;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j++) m_block = fmaxf(m_block, s_vals[j]);
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

    // Stage 2: Dense tree attention (all blocks, no bitmap)
    int tree_kv_blocks = (N_tree + BLOCK_N - 1) / BLOCK_N;
    for (int kv_block_id = 0; kv_block_id < tree_kv_blocks; kv_block_id++) {
        int kv_tree_start = kv_block_id * BLOCK_N;
        int kv_len = min(BLOCK_N, N_tree - kv_tree_start);
        if (kv_len <= 0) break;

        int kv_indices_offset = past_len + kv_tree_start;

        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, K_buf, kv_indices,
            kv_indices_offset, kv_len, h_kv,
            stride_buf_n, stride_buf_h,
            BLOCK_M, tid);

        load_mask_tile_vectorized<BLOCK_M>(
            smem_tmask, tree_mask,
            q_start, kv_tree_start,
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

        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, V_buf, kv_indices,
            kv_indices_offset, kv_len, h_kv,
            stride_buf_n, stride_buf_h,
            BLOCK_M, tid);
        __syncthreads();

        if (valid_q) {
            float m_block = -FLT_MAX;
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j++) m_block = fmaxf(m_block, s_vals[j]);
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

    // Finalize
    if (valid_q) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        half2* O_row = reinterpret_cast<half2*>(O_h + my_q * HEAD_DIM);
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM / 2; d++) {
            O_row[d] = __floats2half2_rn(o_acc[2*d] * inv_l, o_acc[2*d+1] * inv_l);
        }
    }
}


// ============================================================================
// HOST FUNCTIONS
// ============================================================================

static int get_smem_size(int block_m, int head_dim) {
    // smem_q[BLOCK_M][HEAD_DIM] + smem_kv[BLOCK_N][HEAD_DIM] + smem_tmask[BLOCK_M][BLOCK_N]
    return (block_m + BLOCK_N) * head_dim * sizeof(half)
         + block_m * BLOCK_N * sizeof(half);
}

/// Pad tree_mask from [N, N] to block boundaries with -inf.
static torch::Tensor pad_tree_mask(torch::Tensor tree_mask, int N_Q, int block_m) {
    int num_q_blocks = (N_Q + block_m - 1) / block_m;
    int num_kv_blocks = (N_Q + BLOCK_N - 1) / BLOCK_N;
    int pad_rows = num_q_blocks * block_m;
    int pad_cols = num_kv_blocks * BLOCK_N;

    if (pad_rows == N_Q && pad_cols == N_Q) return tree_mask;

    auto padded = torch::full({pad_rows, pad_cols},
        -std::numeric_limits<float>::infinity(), tree_mask.options());
    padded.narrow(0, 0, N_Q).narrow(1, 0, N_Q).copy_(tree_mask);
    return padded.contiguous();
}

/// Paged bitmap attention forward.
torch::Tensor paged_bitmap_attention_fwd(
    torch::Tensor Q,             // [H_q, N_tree, D]
    torch::Tensor K_buf,         // [max_tokens, H_kv, D]
    torch::Tensor V_buf,         // [max_tokens, H_kv, D]
    torch::Tensor kv_indices,    // [total_kv] int32
    torch::Tensor tree_mask,     // [N_tree, N_tree] fp16 (0/-inf)
    torch::Tensor bitmaps,       // [num_q_blocks, W] int64
    double sm_scale,
    int64_t past_len,
    int64_t W_val,
    int64_t H_kv
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kHalf, "Q must be float16");
    TORCH_CHECK(Q.dim() == 3, "Q must be [H_q, N_tree, D]");
    TORCH_CHECK(K_buf.dim() == 3, "K_buf must be [max_tokens, H_kv, D]");
    TORCH_CHECK(kv_indices.scalar_type() == torch::kInt32, "kv_indices must be int32");

    int H_q_val = Q.size(0);
    int N_tree = Q.size(1);
    int D = Q.size(2);
    int total_kv = static_cast<int>(past_len) + N_tree;

    TORCH_CHECK(D == 64 || D == 128, "HEAD_DIM must be 64 or 128");
    TORCH_CHECK(H_kv > 0 && H_q_val % H_kv == 0, "H_q must be divisible by H_kv");

    Q = Q.contiguous();
    K_buf = K_buf.contiguous();
    V_buf = V_buf.contiguous();
    kv_indices = kv_indices.contiguous();
    tree_mask = tree_mask.contiguous();
    bitmaps = bitmaps.contiguous();

    auto O = torch::zeros({H_q_val, N_tree, D}, Q.options());

    int block_m;
    if (N_tree <= 32) block_m = 32;
    else if (N_tree <= 64) block_m = 64;
    else block_m = 128;

    tree_mask = pad_tree_mask(tree_mask, N_tree, block_m);
    int mask_stride_int = static_cast<int>(tree_mask.stride(0));

    int GQA_GROUP = H_q_val / static_cast<int>(H_kv);
    int num_q_blocks = (N_tree + block_m - 1) / block_m;
    dim3 grid(num_q_blocks, H_q_val);
    int smem_size = get_smem_size(block_m, D);

    float scale_f = static_cast<float>(sm_scale);
    int W_int = static_cast<int>(W_val);

    // Strides of K_buf in elements (halves)
    int stride_buf_n = static_cast<int>(K_buf.stride(0));
    int stride_buf_h = static_cast<int>(K_buf.stride(1));

    #define LAUNCH_PAGED_BITMAP(HD, BM) do { \
        auto kern = paged_bitmap_attention_kernel<HD, BM>; \
        cudaFuncSetAttribute(kern, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
        kern<<<grid, BM, smem_size>>>( \
            reinterpret_cast<const half*>(Q.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(K_buf.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(V_buf.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(O.data_ptr<at::Half>()), \
            kv_indices.data_ptr<int32_t>(), \
            reinterpret_cast<const half*>(tree_mask.data_ptr<at::Half>()), \
            bitmaps.data_ptr<int64_t>(), \
            scale_f, N_tree, static_cast<int>(past_len), total_kv, \
            W_int, mask_stride_int, H_q_val, static_cast<int>(H_kv), \
            GQA_GROUP, stride_buf_n, stride_buf_h); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } while (0)

    if (D == 128) {
        if (block_m == 32) LAUNCH_PAGED_BITMAP(128, 32);
        else if (block_m == 64) LAUNCH_PAGED_BITMAP(128, 64);
        else LAUNCH_PAGED_BITMAP(128, 128);
    } else {
        if (block_m == 32) LAUNCH_PAGED_BITMAP(64, 32);
        else if (block_m == 64) LAUNCH_PAGED_BITMAP(64, 64);
        else LAUNCH_PAGED_BITMAP(64, 128);
    }
    #undef LAUNCH_PAGED_BITMAP

    return O;
}

/// Paged dense attention forward (for benchmarking comparison).
torch::Tensor paged_dense_attention_fwd(
    torch::Tensor Q,
    torch::Tensor K_buf,
    torch::Tensor V_buf,
    torch::Tensor kv_indices,
    torch::Tensor tree_mask,
    double sm_scale,
    int64_t past_len,
    int64_t H_kv
) {
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kHalf && Q.dim() == 3);

    int H_q_val = Q.size(0);
    int N_tree = Q.size(1);
    int D = Q.size(2);
    int total_kv = static_cast<int>(past_len) + N_tree;

    Q = Q.contiguous();
    K_buf = K_buf.contiguous();
    V_buf = V_buf.contiguous();
    kv_indices = kv_indices.contiguous();
    tree_mask = tree_mask.contiguous();

    auto O = torch::zeros({H_q_val, N_tree, D}, Q.options());

    int block_m;
    if (N_tree <= 32) block_m = 32;
    else if (N_tree <= 64) block_m = 64;
    else block_m = 128;

    tree_mask = pad_tree_mask(tree_mask, N_tree, block_m);
    int mask_stride_int = static_cast<int>(tree_mask.stride(0));

    int GQA_GROUP = H_q_val / static_cast<int>(H_kv);
    int num_q_blocks = (N_tree + block_m - 1) / block_m;
    dim3 grid(num_q_blocks, H_q_val);
    int smem_size = get_smem_size(block_m, D);

    float scale_f = static_cast<float>(sm_scale);
    int stride_buf_n = static_cast<int>(K_buf.stride(0));
    int stride_buf_h = static_cast<int>(K_buf.stride(1));

    #define LAUNCH_PAGED_DENSE(HD, BM) do { \
        auto kern = paged_dense_attention_kernel<HD, BM>; \
        cudaFuncSetAttribute(kern, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
        kern<<<grid, BM, smem_size>>>( \
            reinterpret_cast<const half*>(Q.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(K_buf.data_ptr<at::Half>()), \
            reinterpret_cast<const half*>(V_buf.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(O.data_ptr<at::Half>()), \
            kv_indices.data_ptr<int32_t>(), \
            reinterpret_cast<const half*>(tree_mask.data_ptr<at::Half>()), \
            scale_f, N_tree, static_cast<int>(past_len), total_kv, \
            mask_stride_int, H_q_val, static_cast<int>(H_kv), \
            GQA_GROUP, stride_buf_n, stride_buf_h); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } while (0)

    if (D == 128) {
        if (block_m == 32) LAUNCH_PAGED_DENSE(128, 32);
        else if (block_m == 64) LAUNCH_PAGED_DENSE(128, 64);
        else LAUNCH_PAGED_DENSE(128, 128);
    } else {
        if (block_m == 32) LAUNCH_PAGED_DENSE(64, 32);
        else if (block_m == 64) LAUNCH_PAGED_DENSE(64, 64);
        else LAUNCH_PAGED_DENSE(64, 128);
    }
    #undef LAUNCH_PAGED_DENSE

    return O;
}

/// Return the BLOCK_M used for a given N_tree.
int64_t get_block_m(int64_t N_tree) {
    if (N_tree <= 32) return 32;
    else if (N_tree <= 64) return 64;
    else return 128;
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Paged bitmap-driven sparse attention CUDA kernel";
    m.def("forward", &paged_bitmap_attention_fwd,
          "Paged bitmap attention forward (two-stage: paged past + bitmap tree)",
          py::arg("Q"), py::arg("K_buf"), py::arg("V_buf"),
          py::arg("kv_indices"), py::arg("tree_mask"), py::arg("bitmaps"),
          py::arg("sm_scale"), py::arg("past_len"), py::arg("W"), py::arg("H_kv"));
    m.def("dense_forward", &paged_dense_attention_fwd,
          "Paged dense attention forward (for benchmarking comparison)",
          py::arg("Q"), py::arg("K_buf"), py::arg("V_buf"),
          py::arg("kv_indices"), py::arg("tree_mask"),
          py::arg("sm_scale"), py::arg("past_len"), py::arg("H_kv"));
    m.def("get_block_m", &get_block_m,
          "Return BLOCK_M for a given N_tree", py::arg("N_tree"));
}
