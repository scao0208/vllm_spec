#pragma once
/**
 * Paged Bitmap Attention — kernel templates with WMMA tensor core MMA.
 *
 * Two kernels:
 *   1. paged_bitmap_attention_kernel — bitmap-driven sparse Stage 2
 *   2. paged_dense_attention_kernel  — dense Stage 2 (benchmark reference)
 *
 * Both share two-stage architecture:
 *   Stage 1: Dense past context with paged KV reads (no mask)
 *   Stage 2: Tree-region attention with paged KV reads + tree mask
 *
 * QK and PV computations use nvcuda::wmma m16n16k16 MMA instructions
 * (fp16 input, fp32 accumulator) for tensor core utilization.
 *
 * Grid: (num_q_blocks, H_q), Threads: NUM_THREADS (128) per block.
 *
 * Shared memory layout:
 *   smem_q      [BLOCK_M][HEAD_DIM]    half
 *   smem_kv     [BLOCK_N][HEAD_DIM]    half  (reused as V^T[HEAD_DIM][BLOCK_N])
 *   smem_tmask  [BLOCK_M][BLOCK_N]     half
 *   smem_scores [BLOCK_M][BLOCK_N]     float (reused for output tile store)
 *   smem_scale  [BLOCK_M]              float
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <mma.h>
#include "config.hpp"
#include "device_helpers.cuh"

using namespace nvcuda;

namespace paged_bitmap {

// ============================================================================
// WMMA QK: S[BLOCK_M, BLOCK_N] = Q[BLOCK_M, D] x K^T[D, BLOCK_N]
// K stored as [BLOCK_N, HEAD_DIM] row-major; loaded col_major gives K^T.
// Each warp handles a 32-row stripe of the output.
// ============================================================================

template <int HEAD_DIM, int BLOCK_M>
__device__ __forceinline__ void wmma_qk(
    const half* __restrict__ smem_q,
    const half* __restrict__ smem_kv,
    float* __restrict__ smem_scores,
    float qk_scale)
{
    const int warp_id = threadIdx.x / 32;
    const int warp_m_start = warp_id * 32;

    constexpr int N_TILES = BLOCK_N / WMMA_TILE;   // 2
    constexpr int K_TILES = HEAD_DIM / WMMA_TILE;   // 8

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][N_TILES];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++)
            wmma::fill_fragment(acc[mt][nt], 0.0f);

    #pragma unroll
    for (int kt = 0; kt < K_TILES; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag[N_TILES];

        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            int m_off = warp_m_start + mt * 16;
            if (m_off < BLOCK_M)
                wmma::load_matrix_sync(q_frag[mt],
                    smem_q + m_off * HEAD_DIM + kt * 16, HEAD_DIM);
            else
                wmma::fill_fragment(q_frag[mt], __float2half(0.0f));
        }

        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++)
            wmma::load_matrix_sync(k_frag[nt],
                smem_kv + nt * 16 * HEAD_DIM + kt * 16, HEAD_DIM);

        #pragma unroll
        for (int mt = 0; mt < 2; mt++)
            #pragma unroll
            for (int nt = 0; nt < N_TILES; nt++)
                wmma::mma_sync(acc[mt][nt], q_frag[mt], k_frag[nt], acc[mt][nt]);
    }

    // Store scaled results
    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        int m_off = warp_m_start + mt * 16;
        if (m_off < BLOCK_M) {
            #pragma unroll
            for (int nt = 0; nt < N_TILES; nt++) {
                #pragma unroll
                for (int i = 0; i < acc[mt][nt].num_elements; i++)
                    acc[mt][nt].x[i] *= qk_scale;
                wmma::store_matrix_sync(
                    smem_scores + m_off * BLOCK_N + nt * 16,
                    acc[mt][nt], BLOCK_N, wmma::mem_row_major);
            }
        }
    }
}

// ============================================================================
// WMMA PV: O[BLOCK_M, HEAD_DIM] += P[BLOCK_M, BLOCK_N] x V[BLOCK_N, HEAD_DIM]
// V^T stored as [HEAD_DIM][BLOCK_N]; loaded col_major gives V.
// O accumulator fragments persist across KV blocks for online softmax.
// ============================================================================

template <int HEAD_DIM, int BLOCK_M>
__device__ __forceinline__ void wmma_pv(
    const half* __restrict__ smem_P,
    const half* __restrict__ smem_vt,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>* o_frags)
{
    const int warp_id = threadIdx.x / 32;
    const int warp_m_start = warp_id * 32;

    constexpr int K_TILES = BLOCK_N / WMMA_TILE;    // 2
    constexpr int D_TILES = HEAD_DIM / WMMA_TILE;    // 8

    #pragma unroll
    for (int kt = 0; kt < K_TILES; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag[2];

        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            int m_off = warp_m_start + mt * 16;
            if (m_off < BLOCK_M)
                wmma::load_matrix_sync(p_frag[mt],
                    smem_P + m_off * BLOCK_N + kt * 16, BLOCK_N);
            else
                wmma::fill_fragment(p_frag[mt], __float2half(0.0f));
        }

        #pragma unroll
        for (int dt = 0; dt < D_TILES; dt++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> v_frag;
            wmma::load_matrix_sync(v_frag,
                smem_vt + dt * 16 * BLOCK_N + kt * 16, BLOCK_N);

            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                wmma::mma_sync(
                    o_frags[mt * D_TILES + dt],
                    p_frag[mt], v_frag,
                    o_frags[mt * D_TILES + dt]);
        }
    }
}

// ============================================================================
// Online softmax: read smem_scores, rescale O fragments, write P (fp16) to
// smem_tmask (reused as P buffer — safe because mask is no longer needed).
//
// Phase 1: threads read scores to registers (avoid in-place race condition)
// Phase 2: compute softmax, write smem_scale and P
// Phase 3: rescale O fragments using smem_scale
// ============================================================================

template <int BLOCK_M, int HEAD_DIM>
__device__ __forceinline__ void online_softmax_and_rescale(
    float* __restrict__ smem_scores,
    float* __restrict__ smem_scale,
    half* __restrict__ smem_P,         // [BLOCK_M][BLOCK_N] output P buffer
    float& m_i,
    float& l_i,
    int valid_rows,
    int kv_len,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>* o_frags)
{
    const int tid = threadIdx.x;
    constexpr int D_TILES = HEAD_DIM / WMMA_TILE;

    // Phase 1: Each thread reads its row of scores to registers
    float s_local[BLOCK_N];
    if (tid < BLOCK_M && tid < valid_rows) {
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++)
            s_local[j] = smem_scores[tid * BLOCK_N + j];
    }
    // No sync needed here — each thread only reads its own row,
    // and writes go to a separate buffer (smem_P, smem_scale).

    // Phase 2: Compute softmax, write scale and P
    if (tid < BLOCK_M && tid < valid_rows) {
        float m_old = m_i;
        float m_block = -FLT_MAX;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++)
            m_block = fmaxf(m_block, s_local[j]);

        float m_new = fmaxf(m_old, m_block);
        float scale_old = exp2f(m_old - m_new);
        smem_scale[tid] = scale_old;
        l_i *= scale_old;

        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            float p_j = exp2f(s_local[j] - m_new);
            l_i += p_j;
            smem_P[tid * BLOCK_N + j] = __float2half(p_j);
        }
        m_i = m_new;
    } else if (tid < BLOCK_M) {
        // Invalid rows: scale=1, P=0
        smem_scale[tid] = 1.0f;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++)
            smem_P[tid * BLOCK_N + j] = __float2half(0.0f);
    }
    __syncthreads();

    // Phase 3: Rescale O fragments using smem_scale
    // WMMA m16n16k16 accumulator element-to-row mapping (SM80+):
    //   x[0,1,4,5] -> row group_id
    //   x[2,3,6,7] -> row group_id + 8
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int group_id = lane_id / 4;
    const int warp_m_start = warp_id * 32;

    #pragma unroll
    for (int mt = 0; mt < 2; mt++) {
        int row0 = warp_m_start + mt * 16 + group_id;
        int row1 = row0 + 8;
        float s0 = (row0 < BLOCK_M) ? smem_scale[row0] : 1.0f;
        float s1 = (row1 < BLOCK_M) ? smem_scale[row1] : 1.0f;

        #pragma unroll
        for (int dt = 0; dt < D_TILES; dt++) {
            auto& frag = o_frags[mt * D_TILES + dt];
            frag.x[0] *= s0; frag.x[1] *= s0;
            frag.x[2] *= s1; frag.x[3] *= s1;
            frag.x[4] *= s0; frag.x[5] *= s0;
            frag.x[6] *= s1; frag.x[7] *= s1;
        }
    }
    __syncthreads();
}

// ============================================================================
// Apply tree mask to smem_scores and invalidate out-of-bounds
// ============================================================================

template <int BLOCK_M>
__device__ __forceinline__ void apply_mask_and_bounds(
    float* __restrict__ smem_scores,
    const half* __restrict__ smem_tmask,
    int valid_rows, int kv_len)
{
    const int tid = threadIdx.x;
    constexpr int total = BLOCK_M * BLOCK_N;
    for (int idx = tid; idx < total; idx += NUM_THREADS) {
        int r = idx / BLOCK_N;
        int c = idx % BLOCK_N;
        if (r < valid_rows && c < kv_len)
            smem_scores[idx] += __half2float(smem_tmask[idx]);
        else
            smem_scores[idx] = -FLT_MAX;
    }
}

// ============================================================================
// Invalidate out-of-bounds only (Stage 1, no tree mask)
// ============================================================================

template <int BLOCK_M>
__device__ __forceinline__ void apply_bounds_only(
    float* __restrict__ smem_scores,
    int valid_rows, int kv_len)
{
    const int tid = threadIdx.x;
    constexpr int total = BLOCK_M * BLOCK_N;
    for (int idx = tid; idx < total; idx += NUM_THREADS) {
        int r = idx / BLOCK_N;
        int c = idx % BLOCK_N;
        if (r >= valid_rows || c >= kv_len)
            smem_scores[idx] = -FLT_MAX;
    }
}

// ============================================================================
// Store O fragments to global memory (tile-by-tile to avoid large smem_O)
//
// Iterates over HEAD_DIM in 16-column stripes, using smem_scores as temp
// buffer for each stripe [BLOCK_M][16] floats.
// ============================================================================

template <int HEAD_DIM, int BLOCK_M>
__device__ __forceinline__ void store_output(
    half* __restrict__ O_h,
    float* __restrict__ smem_tmp,   // reuse smem_scores, >= BLOCK_M*16 floats
    float* __restrict__ smem_scale, // reuse for l_i storage
    const wmma::fragment<wmma::accumulator, 16, 16, 16, float>* o_frags,
    float l_i_local,
    int q_start, int valid_rows)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m_start = warp_id * 32;
    constexpr int D_TILES = HEAD_DIM / WMMA_TILE;

    // Write l_i to smem_scale (reused)
    if (tid < BLOCK_M)
        smem_scale[tid] = l_i_local;
    __syncthreads();

    // Process D_TILES stripes of 16 columns each
    #pragma unroll
    for (int dt = 0; dt < D_TILES; dt++) {
        // Store WMMA fragments for this d-tile to smem_tmp[BLOCK_M][16]
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            int m_off = warp_m_start + mt * 16;
            if (m_off < BLOCK_M) {
                wmma::store_matrix_sync(
                    smem_tmp + m_off * 16,
                    o_frags[mt * D_TILES + dt],
                    16, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Normalize and write to global (all threads cooperate)
        const int total_elems = valid_rows * 16;
        for (int idx = tid; idx < total_elems; idx += NUM_THREADS) {
            int row = idx / 16;
            int d = idx % 16;
            float inv_l = (smem_scale[row] > 0.0f) ? (1.0f / smem_scale[row]) : 0.0f;
            float val = smem_tmp[row * 16 + d] * inv_l;
            O_h[(q_start + row) * HEAD_DIM + dt * 16 + d] = __float2half(val);
        }
        __syncthreads();
    }
}


// ============================================================================
// Paged Bitmap Attention Kernel (Stage 2: bitmap-driven sparse)
// ============================================================================

template <int HEAD_DIM, int BLOCK_M>
__global__ void paged_bitmap_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K_buf,
    const half* __restrict__ V_buf,
    half* __restrict__ O,
    const int32_t* __restrict__ kv_indices,
    const half* __restrict__ tree_mask,
    const int64_t* __restrict__ bitmaps,
    const float sm_scale,
    const int N_tree,
    const int past_len,
    const int total_kv,
    const int W,
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
    const int valid_rows = min(BLOCK_M, N_tree - q_start);

    const half* Q_h = Q + (int64_t)h_q * N_tree * HEAD_DIM;
    half* O_h = O + (int64_t)h_q * N_tree * HEAD_DIM;

    // Shared memory
    extern __shared__ char smem_raw[];
    half*  smem_q      = reinterpret_cast<half*>(smem_raw);
    half*  smem_kv     = smem_q + BLOCK_M * HEAD_DIM;
    half*  smem_tmask  = smem_kv + BLOCK_N * HEAD_DIM;
    float* smem_scores = reinterpret_cast<float*>(smem_tmask + BLOCK_M * BLOCK_N);
    float* smem_scale  = smem_scores + BLOCK_M * BLOCK_N;

    const float qk_scale = sm_scale * LOG2E;

    // Load Q tile
    load_q_tile_to_smem<HEAD_DIM, BLOCK_M>(
        smem_q, Q_h + q_start * HEAD_DIM,
        max(0, valid_rows), NUM_THREADS, tid);
    __syncthreads();

    // Per-thread softmax state
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    // O accumulator fragments
    constexpr int D_TILES = HEAD_DIM / WMMA_TILE;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frags[2 * D_TILES];
    #pragma unroll
    for (int i = 0; i < 2 * D_TILES; i++)
        wmma::fill_fragment(o_frags[i], 0.0f);

    // ==================================================================
    // Stage 1: Dense past context
    // ==================================================================
    for (int kv_start = 0; kv_start < past_len; kv_start += BLOCK_N) {
        int kv_len = min(BLOCK_N, past_len - kv_start);

        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, K_buf, kv_indices, kv_start, kv_len, h_kv,
            stride_buf_n, stride_buf_h, NUM_THREADS, tid);
        __syncthreads();

        wmma_qk<HEAD_DIM, BLOCK_M>(smem_q, smem_kv, smem_scores, qk_scale);
        __syncthreads();

        apply_bounds_only<BLOCK_M>(smem_scores, valid_rows, kv_len);
        __syncthreads();

        // Use smem_tmask as P buffer (unused in Stage 1)
        online_softmax_and_rescale<BLOCK_M, HEAD_DIM>(
            smem_scores, smem_scale, smem_tmask,
            m_i, l_i, valid_rows, kv_len, o_frags);

        load_paged_tile_to_smem_transposed<HEAD_DIM>(
            reinterpret_cast<half*>(smem_kv), V_buf, kv_indices,
            kv_start, kv_len, h_kv,
            stride_buf_n, stride_buf_h, NUM_THREADS, tid);
        __syncthreads();

        wmma_pv<HEAD_DIM, BLOCK_M>(smem_tmask, reinterpret_cast<half*>(smem_kv), o_frags);
        __syncthreads();
    }

    // ==================================================================
    // Stage 2: Bitmap-driven sparse tree attention
    // ==================================================================
    const int64_t* my_bitmap = bitmaps + q_block * W;

    for (int w = 0; w < W; w++) {
        uint64_t bm = static_cast<uint64_t>(my_bitmap[w]);

        while (bm != 0ULL) {
            int bit = __ffsll(static_cast<long long>(bm)) - 1;
            bm &= (bm - 1);

            int kv_block_id = w * 64 + bit;
            int kv_tree_start = kv_block_id * BLOCK_N;
            int kv_len = min(BLOCK_N, N_tree - kv_tree_start);
            if (kv_len <= 0) break;

            int kv_off = past_len + kv_tree_start;

            // Load K + mask
            load_paged_tile_to_smem<HEAD_DIM>(
                smem_kv, K_buf, kv_indices, kv_off, kv_len, h_kv,
                stride_buf_n, stride_buf_h, NUM_THREADS, tid);
            load_mask_tile_vectorized<BLOCK_M>(
                smem_tmask, tree_mask, q_start, kv_tree_start,
                mask_stride, NUM_THREADS, tid);
            __syncthreads();

            wmma_qk<HEAD_DIM, BLOCK_M>(smem_q, smem_kv, smem_scores, qk_scale);
            __syncthreads();

            apply_mask_and_bounds<BLOCK_M>(smem_scores, smem_tmask, valid_rows, kv_len);
            __syncthreads();

            // Reuse smem_tmask as P buffer (mask already applied to scores)
            online_softmax_and_rescale<BLOCK_M, HEAD_DIM>(
                smem_scores, smem_scale, smem_tmask,
                m_i, l_i, valid_rows, kv_len, o_frags);

            load_paged_tile_to_smem_transposed<HEAD_DIM>(
                reinterpret_cast<half*>(smem_kv), V_buf, kv_indices,
                kv_off, kv_len, h_kv,
                stride_buf_n, stride_buf_h, NUM_THREADS, tid);
            __syncthreads();

            wmma_pv<HEAD_DIM, BLOCK_M>(smem_tmask, reinterpret_cast<half*>(smem_kv), o_frags);
            __syncthreads();
        }
    }

    // ==================================================================
    // Finalize
    // ==================================================================
    if (valid_rows > 0) {
        store_output<HEAD_DIM, BLOCK_M>(
            O_h, smem_scores, smem_scale, o_frags, l_i,
            q_start, valid_rows);
    }
}


// ============================================================================
// Paged Dense Attention Kernel (Stage 2: all blocks, no bitmap)
// ============================================================================

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
    const int valid_rows = min(BLOCK_M, N_tree - q_start);

    const half* Q_h = Q + (int64_t)h_q * N_tree * HEAD_DIM;
    half* O_h = O + (int64_t)h_q * N_tree * HEAD_DIM;

    extern __shared__ char smem_raw[];
    half*  smem_q      = reinterpret_cast<half*>(smem_raw);
    half*  smem_kv     = smem_q + BLOCK_M * HEAD_DIM;
    half*  smem_tmask  = smem_kv + BLOCK_N * HEAD_DIM;
    float* smem_scores = reinterpret_cast<float*>(smem_tmask + BLOCK_M * BLOCK_N);
    float* smem_scale  = smem_scores + BLOCK_M * BLOCK_N;

    const float qk_scale = sm_scale * LOG2E;

    load_q_tile_to_smem<HEAD_DIM, BLOCK_M>(
        smem_q, Q_h + q_start * HEAD_DIM,
        max(0, valid_rows), NUM_THREADS, tid);
    __syncthreads();

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    constexpr int D_TILES = HEAD_DIM / WMMA_TILE;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frags[2 * D_TILES];
    #pragma unroll
    for (int i = 0; i < 2 * D_TILES; i++)
        wmma::fill_fragment(o_frags[i], 0.0f);

    // Stage 1: Dense past context
    for (int kv_start = 0; kv_start < past_len; kv_start += BLOCK_N) {
        int kv_len = min(BLOCK_N, past_len - kv_start);

        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, K_buf, kv_indices, kv_start, kv_len, h_kv,
            stride_buf_n, stride_buf_h, NUM_THREADS, tid);
        __syncthreads();

        wmma_qk<HEAD_DIM, BLOCK_M>(smem_q, smem_kv, smem_scores, qk_scale);
        __syncthreads();

        apply_bounds_only<BLOCK_M>(smem_scores, valid_rows, kv_len);
        __syncthreads();

        online_softmax_and_rescale<BLOCK_M, HEAD_DIM>(
            smem_scores, smem_scale, smem_tmask,
            m_i, l_i, valid_rows, kv_len, o_frags);

        load_paged_tile_to_smem_transposed<HEAD_DIM>(
            reinterpret_cast<half*>(smem_kv), V_buf, kv_indices,
            kv_start, kv_len, h_kv,
            stride_buf_n, stride_buf_h, NUM_THREADS, tid);
        __syncthreads();

        wmma_pv<HEAD_DIM, BLOCK_M>(smem_tmask, reinterpret_cast<half*>(smem_kv), o_frags);
        __syncthreads();
    }

    // Stage 2: Dense tree attention
    int tree_kv_blocks = (N_tree + BLOCK_N - 1) / BLOCK_N;
    for (int kv_block_id = 0; kv_block_id < tree_kv_blocks; kv_block_id++) {
        int kv_tree_start = kv_block_id * BLOCK_N;
        int kv_len = min(BLOCK_N, N_tree - kv_tree_start);
        if (kv_len <= 0) break;

        int kv_off = past_len + kv_tree_start;

        load_paged_tile_to_smem<HEAD_DIM>(
            smem_kv, K_buf, kv_indices, kv_off, kv_len, h_kv,
            stride_buf_n, stride_buf_h, NUM_THREADS, tid);
        load_mask_tile_vectorized<BLOCK_M>(
            smem_tmask, tree_mask, q_start, kv_tree_start,
            mask_stride, NUM_THREADS, tid);
        __syncthreads();

        wmma_qk<HEAD_DIM, BLOCK_M>(smem_q, smem_kv, smem_scores, qk_scale);
        __syncthreads();

        apply_mask_and_bounds<BLOCK_M>(smem_scores, smem_tmask, valid_rows, kv_len);
        __syncthreads();

        online_softmax_and_rescale<BLOCK_M, HEAD_DIM>(
            smem_scores, smem_scale, smem_tmask,
            m_i, l_i, valid_rows, kv_len, o_frags);

        load_paged_tile_to_smem_transposed<HEAD_DIM>(
            reinterpret_cast<half*>(smem_kv), V_buf, kv_indices,
            kv_off, kv_len, h_kv,
            stride_buf_n, stride_buf_h, NUM_THREADS, tid);
        __syncthreads();

        wmma_pv<HEAD_DIM, BLOCK_M>(smem_tmask, reinterpret_cast<half*>(smem_kv), o_frags);
        __syncthreads();
    }

    // Finalize
    if (valid_rows > 0) {
        store_output<HEAD_DIM, BLOCK_M>(
            O_h, smem_scores, smem_scale, o_frags, l_i,
            q_start, valid_rows);
    }
}

}  // namespace paged_bitmap
