/**
 * Paged Bitmap Attention — host launch functions and Python bindings.
 *
 * Build: cd csrc && python setup_paged_bitmap_attention.py build_ext --inplace
 * Usage: import paged_bitmap_attention_ext; paged_bitmap_attention_ext.forward(...)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// CUTLASS headers for types
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "paged_bitmap_attention/config.hpp"
#include "paged_bitmap_attention/kernel.cuh"

using namespace paged_bitmap;

// ============================================================================
// Host helpers
// ============================================================================

static int get_smem_size(int block_m, int head_dim) {
    // smem_q[BLOCK_M][HEAD_DIM]         half
    // smem_kv[BLOCK_N][HEAD_DIM]        half  (reused as V^T[HEAD_DIM][BLOCK_N])
    // smem_tmask[BLOCK_M][BLOCK_N]      half  (reused as P buffer after mask applied)
    // smem_scores[BLOCK_M][BLOCK_N]     float (reused as output tile temp)
    // smem_scale[BLOCK_M]               float (reused for l_i in output)
    int smem_q      = block_m * head_dim * sizeof(half);
    int smem_kv     = BLOCK_N * head_dim * sizeof(half);
    int smem_tmask  = block_m * BLOCK_N * sizeof(half);
    int smem_scores = block_m * BLOCK_N * sizeof(float);
    int smem_scale  = block_m * sizeof(float);
    return smem_q + smem_kv + smem_tmask + smem_scores + smem_scale;
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

// ============================================================================
// Forward functions
// ============================================================================

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

    int stride_buf_n = static_cast<int>(K_buf.stride(0));
    int stride_buf_h = static_cast<int>(K_buf.stride(1));

    #define LAUNCH_PAGED_BITMAP(HD, BM) do { \
        auto kern = paged_bitmap_attention_kernel<HD, BM>; \
        cudaFuncSetAttribute(kern, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
        kern<<<grid, NUM_THREADS, smem_size>>>( \
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
        kern<<<grid, NUM_THREADS, smem_size>>>( \
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
// Python bindings
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
