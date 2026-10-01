/**
 * 2:4 Sparse GEMM using CUTLASS — SM80+ (forward compatible to Blackwell).
 *
 * Exposes:
 *   compress_24(W)              → (compressed, metadata_reordered)
 *   sparse_gemm_24(X, W_c, E, N) → Y = X @ W^T
 *
 * Build: python setup_sparse_gemm.py build_ext --inplace
 * Usage: import sparse_gemm_cutlass; sparse_gemm_cutlass.sparse_gemm_24(...)
 */

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"

// ============================================================================
// CUTLASS SPARSE GEMM CONFIGURATION (SM80)
// ============================================================================
//
// Y = X @ W^T where W is [N, K] with 2:4 sparsity along K.
// CUTLASS SparseGemm applies sparsity to operand A, so we set:
//   A = W_compressed (sparse), B = X^T (dense), D = Y^T
//   D[N, M] = W[N, K] @ X^T[K, M]  →  transpose to Y[M, N].

using ElementA = cutlass::half_t;     // W: sparse weight (compressed)
using ElementB = cutlass::half_t;     // X: input activations (dense)
using ElementC = cutlass::half_t;     // Y: output
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;      // W: [N, K] row-major (sparse)
using LayoutB = cutlass::layout::ColumnMajor;    // X^T: [K, M] col-major = X[M,K] row-major
using LayoutC = cutlass::layout::RowMajor;       // D: [N, M] row-major

// SM80 sparse GEMM tile configuration
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

static int const kStages = 3;
static int const kAlignmentA = 8;  // 128-bit / 16-bit = 8 elements
static int const kAlignmentB = 8;

using SpGemm = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignmentA,
    kAlignmentB
>;

using Gemm = SpGemm;

// Metadata type from CUTLASS kernel (uint16_t for SM80 with kMaxID2==2)
using GemmElementE = typename Gemm::ElementE;


// ============================================================================
// 2:4 COMPRESSION KERNEL
// ============================================================================

/**
 * Compress [N, K] weight to 2:4 format: keep 2 largest per group of 4.
 * Output metadata is in "natural" RowMajor [N, K/16] layout.
 */
__global__ void compress_24_kernel(
    const half* __restrict__ W,       // [N, K]
    half* __restrict__ compressed,    // [N, K/2]
    uint16_t* __restrict__ metadata,  // [N, K/16] RowMajor (natural order)
    int N, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const half* w_row = W + row * K;
    half* c_row = compressed + row * (K / 2);
    uint16_t* m_row = metadata + row * (K / 16);

    // Process groups of 16 elements (4 groups of 4) → one uint16 metadata
    for (int k16 = 0; k16 < K / 16; k16++) {
        uint16_t meta = 0;
        int c_offset = k16 * 8;  // 16/2 = 8 compressed values per group of 16

        for (int g = 0; g < 4; g++) {
            int base = k16 * 16 + g * 4;
            float vals[4];
            for (int i = 0; i < 4; i++) {
                vals[i] = __half2float(w_row[base + i]);
            }

            // Find 2 largest by magnitude
            float abs_vals[4];
            int indices[4] = {0, 1, 2, 3};
            for (int i = 0; i < 4; i++) abs_vals[i] = fabsf(vals[i]);

            // Simple selection sort for 4 elements (top 2 by abs value)
            for (int i = 0; i < 2; i++) {
                for (int j = i + 1; j < 4; j++) {
                    if (abs_vals[j] > abs_vals[i]) {
                        float tmp = abs_vals[i]; abs_vals[i] = abs_vals[j]; abs_vals[j] = tmp;
                        int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
                    }
                }
            }

            // Keep top 2, sorted by original position
            int keep[2];
            if (indices[0] < indices[1]) {
                keep[0] = indices[0];
                keep[1] = indices[1];
            } else {
                keep[0] = indices[1];
                keep[1] = indices[0];
            }

            // Store compressed values
            c_row[c_offset + g * 2 + 0] = w_row[base + keep[0]];
            c_row[c_offset + g * 2 + 1] = w_row[base + keep[1]];

            // Encode metadata: bit[1:0] = first index, bit[3:2] = second index
            uint16_t idx = (keep[1] << 2) | keep[0];
            meta |= (idx << (g * 4));
        }

        m_row[k16] = meta;
    }
}


// ============================================================================
// METADATA REORDERING KERNEL
// ============================================================================
//
// CUTLASS SM80 sparse MMA uses LayoutE = ColumnMajorInterleaved<2>.
// Metadata must be reordered for ldmatrix (see host_reorder.h::reorder_meta).
//
// Combined operation: RowMajor source → reorder → ColumnMajorInterleaved<2> dest
//
// ColumnMajorInterleaved<2> offset for (row, col):
//   (col / 2) * stride + row * 2 + col % 2   where stride = M * 2

__global__ void reorder_meta_kernel(
    const uint16_t* __restrict__ src,   // [M, K16] RowMajor: src[m * K16 + k]
    uint16_t* __restrict__ dest,        // [M, K16] ColumnMajorInterleaved<2>
    int M, int K16
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || k >= K16) return;

    // Step 1: Row interleaving (group=32, interweave=4 for sizeof(uint16_t)==2)
    int group = 32;
    int interweave = 4;
    int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
    int dest_col = k;

    // Step 2: 2×2 Z-to-N swizzle
    if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
    } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
    }

    // Read from RowMajor, write to ColumnMajorInterleaved<2>
    int stride = M * 2;
    int dest_offset = (dest_col / 2) * stride + dest_row * 2 + dest_col % 2;
    dest[dest_offset] = src[m * K16 + k];
}


// ============================================================================
// COMPRESS + REORDER (HOST FUNCTION)
// ============================================================================

/**
 * Compress dense weight [N, K] to 2:4 format with CUTLASS-compatible metadata.
 *
 * Returns:
 *   compressed: [N, K/2] half — packed non-zero values
 *   metadata:   reordered for CUTLASS SM80 sparse MMA (ColumnMajor + interleave)
 */
std::tuple<torch::Tensor, torch::Tensor> compress_24(torch::Tensor W) {
    TORCH_CHECK(W.is_cuda(), "W must be on CUDA");
    TORCH_CHECK(W.scalar_type() == torch::kHalf, "W must be float16");
    TORCH_CHECK(W.dim() == 2, "W must be 2D");

    int N = W.size(0);
    int K = W.size(1);
    TORCH_CHECK(K % 16 == 0, "K must be divisible by 16 for 2:4 sparsity");

    auto compressed = torch::empty({N, K / 2}, W.options());

    // Step 1: Compress to natural RowMajor metadata [N, K/16]
    int K16 = K / 16;
    auto metadata_raw = torch::empty({N, K16},
        torch::TensorOptions().dtype(torch::kInt16).device(W.device()));

    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        compress_24_kernel<<<blocks, threads>>>(
            reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
            reinterpret_cast<half*>(compressed.data_ptr<at::Half>()),
            reinterpret_cast<uint16_t*>(metadata_raw.data_ptr<int16_t>()),
            N, K
        );
    }

    // Step 2: Reorder metadata for CUTLASS sparse MMA layout.
    // LayoutE = ColumnMajorInterleaved<2>: needs N * K16 elements,
    // with stride = N * 2. Allocate flat buffer.
    int meta_size = N * K16;
    auto metadata_reordered = torch::empty({meta_size},
        torch::TensorOptions().dtype(torch::kInt16).device(W.device()));

    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (K16 + 15) / 16);
        reorder_meta_kernel<<<grid, block>>>(
            reinterpret_cast<const uint16_t*>(metadata_raw.data_ptr<int16_t>()),
            reinterpret_cast<uint16_t*>(metadata_reordered.data_ptr<int16_t>()),
            N, K16
        );
    }

    return std::make_tuple(compressed, metadata_reordered);
}


// ============================================================================
// SPARSE GEMM EXECUTION
// ============================================================================

/**
 * Y = X @ W^T  where W is compressed 2:4.
 *
 * Args:
 *   X: [M, K] dense input (half)
 *   W_compressed: [N, K/2] compressed weight (half)
 *   metadata: reordered metadata from compress_24()
 *   N_out: output dimension
 *
 * Returns:
 *   Y: [M, N] output (half)
 */
torch::Tensor sparse_gemm_24(
    torch::Tensor X,
    torch::Tensor W_compressed,
    torch::Tensor metadata,
    int64_t N_out
) {
    TORCH_CHECK(X.is_cuda() && W_compressed.is_cuda() && metadata.is_cuda());
    TORCH_CHECK(X.scalar_type() == torch::kHalf);

    int M = X.size(0);
    int K = X.size(1);
    int N = N_out;

    // D[N, M] = W[N, K] @ X^T[K, M]  →  transpose to Y[M, N]
    // CUTLASS epilogue requires N_cutlass (= M) to be a multiple of 8.
    // Pad both problem size and X input.
    int M_pad = ((M + 7) / 8) * 8;
    torch::Tensor X_eff = X;
    if (M_pad != M) {
        X_eff = torch::zeros({M_pad, K}, X.options());
        X_eff.slice(0, 0, M).copy_(X);
    }
    auto D = torch::empty({N, M_pad}, X.options());
    cutlass::gemm::GemmCoord problem_size(N, M_pad, K);

    // Leading dimensions
    int lda = K / Gemm::kSparse;  // W_compressed: [N, K/2] row-major
    int ldb = K;                  // X^T: [K, M] col-major
    int ldc = M_pad;              // D: [N, M_pad] row-major
    int lde = N * 2;              // E: ColumnMajorInterleaved<2>, stride = N * 2

    // Construct arguments
    typename Gemm::Arguments args{
        problem_size,
        {reinterpret_cast<ElementA*>(W_compressed.data_ptr<at::Half>()), lda},
        {reinterpret_cast<ElementB*>(X_eff.data_ptr<at::Half>()), ldb},
        {reinterpret_cast<ElementC*>(D.data_ptr<at::Half>()), ldc},
        {reinterpret_cast<ElementC*>(D.data_ptr<at::Half>()), ldc},
        {reinterpret_cast<GemmElementE*>(metadata.data_ptr<int16_t>()), lde},
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
    };

    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS sparse GEMM cannot implement: ",
        cutlass::cutlassGetStatusString(status));

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)workspace_size},
        torch::TensorOptions().dtype(torch::kByte).device(X.device()));

    status = gemm_op.initialize(args, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS init failed");

    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS sparse GEMM failed");

    // D[N, M_pad] → slice to [N, M] → transpose to Y[M, N]
    auto D_trimmed = (M_pad == M) ? D : D.slice(1, 0, M);
    return D_trimmed.t().contiguous();
}


// ============================================================================
// PYTHON BINDINGS
// ============================================================================

std::string get_gemm_info() {
    std::ostringstream oss;
    oss << "Gemm::kSparse = " << Gemm::kSparse << "\n";
    oss << "Gemm::kElementsPerElementE = " << Gemm::kElementsPerElementE << "\n";
    oss << "sizeof(Gemm::ElementE) = " << sizeof(typename Gemm::ElementE) << "\n";
    oss << "Gemm::LayoutE is ColumnMajor = " <<
        std::is_same<typename Gemm::LayoutE, cutlass::layout::ColumnMajor>::value << "\n";
    oss << "Gemm::LayoutE is RowMajor = " <<
        std::is_same<typename Gemm::LayoutE, cutlass::layout::RowMajor>::value << "\n";

    // Check interleaved layouts
    oss << "Gemm::LayoutE is ColumnMajorInterleaved<2> = " <<
        std::is_same<typename Gemm::LayoutE, cutlass::layout::ColumnMajorInterleaved<2>>::value << "\n";
    oss << "Gemm::LayoutE is RowMajorInterleaved<2> = " <<
        std::is_same<typename Gemm::LayoutE, cutlass::layout::RowMajorInterleaved<2>>::value << "\n";
    return oss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS 2:4 Sparse GEMM (SM80+)";
    m.def("compress_24", &compress_24,
          "Compress dense [N, K] to 2:4 format → (compressed, metadata_reordered)");
    m.def("sparse_gemm_24", &sparse_gemm_24,
          "Y = X @ W^T with 2:4 sparse W");
    m.def("get_gemm_info", &get_gemm_info,
          "Print CUTLASS sparse GEMM configuration");
}
