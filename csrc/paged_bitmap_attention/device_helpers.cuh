#pragma once
/**
 * Paged Bitmap Attention — device helper functions.
 *
 * Tile-loading primitives for paged KV buffers and tree masks.
 * All functions are __device__ __forceinline__ and templatized where needed.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "config.hpp"

namespace paged_bitmap {

// ============================================================================
// Paged tile loaders
// ============================================================================

/// Collaborative paged tile load: all threads in block cooperatively load
/// BLOCK_N rows of KV from the paged buffer into shared memory.
/// Each row may come from a different token slot (indirect addressing).
///
/// @param smem         destination [BLOCK_N][HEAD_DIM] in shared memory
/// @param kv_buf       flat paged buffer [max_tokens, H_kv, D]
/// @param kv_indices_ptr  token slot array for this region
/// @param kv_start     first position index within the region
/// @param valid_rows   number of valid rows (may be < BLOCK_N)
/// @param h_kv         KV head index
/// @param stride_n     kv_buf stride[0] (in halves)
/// @param stride_h     kv_buf stride[1] (in halves)
/// @param num_threads  blockDim.x
/// @param tid          threadIdx.x
template <int HEAD_DIM>
__device__ __forceinline__ void load_paged_tile_to_smem(
    half* __restrict__ smem,
    const half* __restrict__ kv_buf,
    const int32_t* __restrict__ kv_indices_ptr,
    int kv_start, int valid_rows, int h_kv,
    int stride_n, int stride_h,
    int num_threads, int tid)
{
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
                kv_buf + (int64_t)token_slot * stride_n
                       + (int64_t)h_kv * stride_h);
            smem4[row * VEC_COLS + vc] = src4[vc];
        } else {
            smem4[row * VEC_COLS + vc] = zero4;
        }
    }
}

// ============================================================================
// Tree mask tile loader
// ============================================================================

/// Vectorized tree mask tile load: loads [BM][BLOCK_N] from padded global
/// tree_mask into SMEM using float4 (8 halves per transaction).
///
/// @tparam BM          query block size (BLOCK_M)
/// @param smem_tmask   destination [BM][BLOCK_N] in shared memory
/// @param tree_mask_base  padded tree mask base pointer
/// @param q_start      first query row index
/// @param kv_col_start first KV column index
/// @param mask_stride  row stride of tree_mask (in halves)
/// @param num_threads  blockDim.x
/// @param tid          threadIdx.x
template <int BM>
__device__ __forceinline__ void load_mask_tile_vectorized(
    half* __restrict__ smem_tmask,
    const half* __restrict__ tree_mask_base,
    int q_start, int kv_col_start,
    int mask_stride, int num_threads, int tid)
{
    constexpr int VEC_COLS = BLOCK_N / VEC_SIZE;  // 32/8 = 4
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
// Q tile loader (bounded, with zero-fill)
// ============================================================================

/// Load Q tile [BLOCK_M][HEAD_DIM] into shared memory with bounds checking.
///
/// @tparam HEAD_DIM    head dimension
/// @tparam BLOCK_M     query block size
/// @param smem_q       destination [BLOCK_M][HEAD_DIM]
/// @param Q_ptr        source pointer for this (head, q_start) block
/// @param valid_rows   number of valid Q rows
/// @param num_threads  blockDim.x
/// @param tid          threadIdx.x
template <int HEAD_DIM, int BLOCK_M>
__device__ __forceinline__ void load_q_tile_to_smem(
    half* __restrict__ smem_q,
    const half* __restrict__ Q_ptr,
    int valid_rows,
    int num_threads, int tid)
{
    constexpr int VEC_COLS = HEAD_DIM / VEC_SIZE;
    const int total_vecs = BLOCK_M * VEC_COLS;
    float4* smem4 = reinterpret_cast<float4*>(smem_q);
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int idx = tid; idx < total_vecs; idx += num_threads) {
        int row = idx / VEC_COLS;
        int vc = idx % VEC_COLS;
        if (row < valid_rows) {
            smem4[row * VEC_COLS + vc] =
                reinterpret_cast<const float4*>(Q_ptr + row * HEAD_DIM)[vc];
        } else {
            smem4[row * VEC_COLS + vc] = zero4;
        }
    }
}

// ============================================================================
// Transposed paged V tile loader (for WMMA PV matmul)
// ============================================================================

/// Load V[BLOCK_N][HEAD_DIM] from paged buffer and write as V^T[HEAD_DIM][BLOCK_N]
/// in shared memory for WMMA col_major consumption.
///
/// @param smem_vt       destination [HEAD_DIM][BLOCK_N] in shared memory
/// @param v_buf         flat paged buffer [max_tokens, H_kv, D]
/// @param kv_indices_ptr  token slot array
/// @param kv_start      first position index
/// @param valid_rows    number of valid V rows (may be < BLOCK_N)
/// @param h_kv          KV head index
/// @param stride_n      v_buf stride[0] (in halves)
/// @param stride_h      v_buf stride[1] (in halves)
/// @param num_threads   blockDim.x
/// @param tid           threadIdx.x
template <int HEAD_DIM>
__device__ __forceinline__ void load_paged_tile_to_smem_transposed(
    half* __restrict__ smem_vt,
    const half* __restrict__ v_buf,
    const int32_t* __restrict__ kv_indices_ptr,
    int kv_start, int valid_rows, int h_kv,
    int stride_n, int stride_h,
    int num_threads, int tid)
{
    // Total elements: HEAD_DIM * BLOCK_N
    // Each thread handles multiple elements
    constexpr int total_elems = HEAD_DIM * BLOCK_N;

    for (int idx = tid; idx < total_elems; idx += num_threads) {
        int d = idx / BLOCK_N;   // destination row in V^T (head dim)
        int n = idx % BLOCK_N;   // destination col in V^T (kv position)

        half val;
        if (n < valid_rows) {
            int token_slot = kv_indices_ptr[kv_start + n];
            val = v_buf[(int64_t)token_slot * stride_n
                      + (int64_t)h_kv * stride_h + d];
        } else {
            val = __float2half(0.0f);
        }
        smem_vt[d * BLOCK_N + n] = val;
    }
}

}  // namespace paged_bitmap
