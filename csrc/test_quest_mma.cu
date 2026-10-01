/**
 * Quest page scoring via mma.sync Tensor Core on sm_120
 *
 * Decomposes: score_j = Σ_d max(q[d]*kmax_j[d], q[d]*kmin_j[d])
 * Into:       score_j = dot(q_pos, kmax_j) + dot(q_neg, kmin_j)
 * Where:      q_pos = max(q, 0), q_neg = min(q, 0)
 *
 * This is two GEMMs: Scores = Q_pos @ K_max^T + Q_neg @ K_min^T
 * Using mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *
 * Build: nvcc -arch=sm_120 -std=c++17 -O2 test_quest_mma.cu -o test_quest_mma
 * Run:   CUDA_VISIBLE_DEVICES=4 ./test_quest_mma
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// Constants
// ============================================================================
// mma.sync.m16n8k16: A=[16,16] row-major, B=[8,16] col-major (i.e. [16,8] row-major transposed)
// Output D=[16,8] row-major
// A distributed: 4 registers (a0..a3), each uint32 = 2 x half
// B distributed: 2 registers (b0..b1), each uint32 = 2 x half
// D distributed: 4 registers (d0..d3), each float

// Thread mapping for m16n8k16:
//   32 threads in a warp. Thread t:
//   A fragment: thread t owns rows t/4 and t/4+8, columns (t%4)*2..(t%4)*2+1 for k in [0..7]
//               and same rows, same column offsets for k in [8..15]
//   B fragment: thread t owns rows t/4 and t/4+8 (in col-major = columns), cols (t%4)*2..(t%4)*2+1
//   D fragment: thread t owns (row=t/4, col=0..1) and (row=t/4+8, col=0..1)
//               mapped as d0=(t/4, 2*(t%4)), d1=(t/4, 2*(t%4)+1), d2=(t/4+8, 2*(t%4)), d3=(t/4+8, 2*(t%4)+1)

// Wait -- d mapping for m16n8 output:
// d0 = C[t/4][2*(t%2)]        where t%4 < 2 → group0
// Actually the canonical mapping for m16n8k16 D fragment:
//   thread t:
//     d0 = D[t/4,     (t%2)*2  ]
//     d1 = D[t/4,     (t%2)*2+1]
//     d2 = D[t/4 + 8, (t%2)*2  ]
//     d3 = D[t/4 + 8, (t%2)*2+1]
//   But wait, n=8, so col range is 0..7. Actually:
//   The grouping is by (t%4) for columns:
//     d0 = D[t/4,     (t%4)*2    % 8] -- no, let me just trust the PTX docs.

// For correctness validation, we won't rely on exact thread-to-element mapping.
// Instead, we'll use wmma API for the reference and mma.sync for the test,
// or simply compute the full result and compare against CPU reference.

// ============================================================================
// Reference: CPU Quest page scoring
// ============================================================================
void quest_score_cpu(const float* q,       // [M, K] (M=num_queries, K=head_dim)
                     const float* kmax,     // [N, K] (N=num_pages)
                     const float* kmin,     // [N, K]
                     float* scores,         // [M, N]
                     int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int d = 0; d < K; d++) {
                float qd = q[i * K + d];
                float kmaxd = kmax[j * K + d];
                float kmind = kmin[j * K + d];
                s += fmaxf(qd * kmaxd, qd * kmind);
            }
            scores[i * N + j] = s;
        }
    }
}

// Reference: CPU decomposed Quest (Q_pos @ Kmax^T + Q_neg @ Kmin^T)
void quest_score_decomposed_cpu(const float* q, const float* kmax, const float* kmin,
                                float* scores, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int d = 0; d < K; d++) {
                float qd = q[i * K + d];
                float q_pos = fmaxf(qd, 0.0f);
                float q_neg = fminf(qd, 0.0f);
                s += q_pos * kmax[j * K + d] + q_neg * kmin[j * K + d];
            }
            scores[i * N + j] = s;
        }
    }
}

// ============================================================================
// GPU kernel: Quest scoring via wmma (simpler API, validates the approach)
// ============================================================================
// Uses wmma 16x16x16 to compute:
//   Scores[M,N] = Q_pos[M,K] @ Kmax^T[K,N] + Q_neg[M,K] @ Kmin^T[K,N]
//
// Grid: (ceil(N/16), ceil(M/16))  — each warp computes a 16x16 output tile
// One warp per block for simplicity.

__global__ void quest_score_wmma_kernel(
    const half* __restrict__ q_pos,   // [M, K] row-major
    const half* __restrict__ q_neg,   // [M, K] row-major
    const half* __restrict__ kmax,    // [N, K] row-major (loaded as col-major B = Kmax^T)
    const half* __restrict__ kmin,    // [N, K] row-major
    float* __restrict__ scores,       // [M, N] row-major
    int M, int N, int K)
{
    int tile_n = blockIdx.x * 16;
    int tile_m = blockIdx.y * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Accumulate Q_pos @ Kmax^T
    for (int k = 0; k < K; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

        // A = Q_pos[tile_m:tile_m+16, k:k+16], row-major, ldA = K
        wmma::load_matrix_sync(a_frag, q_pos + tile_m * K + k, K);
        // B = Kmax[tile_n:tile_n+16, k:k+16], col-major (= Kmax^T), ldB = K
        wmma::load_matrix_sync(b_frag, kmax + tile_n * K + k, K);

        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Accumulate Q_neg @ Kmin^T
    for (int k = 0; k < K; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, q_neg + tile_m * K + k, K);
        wmma::load_matrix_sync(b_frag, kmin + tile_n * K + k, K);

        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Store result
    wmma::store_matrix_sync(scores + tile_m * N + tile_n, acc, N, wmma::mem_row_major);
}

// ============================================================================
// GPU kernel: Quest scoring via mma.sync PTX with shared memory tiling
// ============================================================================
// BM=16, BN=template, BK=16. WARPS=BN/8 warps sharing A tile.
// Each warp: 16×8 output (one mma.sync m16n8k16 per k-step).
// Two passes: Q_pos@Kmax^T then Q_neg@Kmin^T, accumulated.
// Grid: (ceil(N/BN), ceil(M/16))
//
// Smem layout: A[16][16+PAD] + B[BN][16+PAD]  (PAD=8 for bank-conflict-free)

template<int BN_TILE>
__global__ void quest_score_mma_smem_kernel(
    const half* __restrict__ q_pos,
    const half* __restrict__ q_neg,
    const half* __restrict__ kmax,
    const half* __restrict__ kmin,
    float* __restrict__ scores,
    int M, int N, int K)
{
    constexpr int BM = 16, BN = BN_TILE, BK = 16, PAD = 8;
    constexpr int A_STRIDE = BK + PAD;  // 24 halfs per row
    constexpr int B_STRIDE = BK + PAD;
    constexpr int WARPS = BN / 8;

    extern __shared__ half smem[];
    half* A_smem = smem;                          // [BM][A_STRIDE]
    half* B_smem = smem + BM * A_STRIDE;          // [BN][B_STRIDE]

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int groupID = lane_id / 4;
    const int tidInGroup = lane_id % 4;
    const int nthreads = WARPS * 32;

    // Output accumulator
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // Process two GEMMs: Q_pos@Kmax^T then Q_neg@Kmin^T
    const half* A_ptrs[2] = {q_pos, q_neg};
    const half* B_ptrs[2] = {kmax, kmin};

    for (int gemm = 0; gemm < 2; gemm++) {
        const half* A_global = A_ptrs[gemm];
        const half* B_global = B_ptrs[gemm];

        for (int k = 0; k < K; k += BK) {
            // --- Cooperative load A[BM][BK] into A_smem ---
            // BM*BK = 16*16 = 256 elements, nthreads threads
            for (int idx = tid; idx < BM * BK; idx += nthreads) {
                int r = idx / BK;
                int c = idx % BK;
                int gm = tile_m + r;
                int gk = k + c;
                A_smem[r * A_STRIDE + c] = (gm < M && gk < K) ?
                    A_global[gm * K + gk] : __float2half(0.0f);
            }

            // --- Cooperative load B[BN][BK] into B_smem ---
            // BN*BK elements, nthreads threads
            for (int idx = tid; idx < BN * BK; idx += nthreads) {
                int r = idx / BK;
                int c = idx % BK;
                int gn = tile_n + r;
                int gk = k + c;
                B_smem[r * B_STRIDE + c] = (gn < N && gk < K) ?
                    B_global[gn * K + gk] : __float2half(0.0f);
            }

            __syncthreads();

            // --- Each warp reads fragments from smem and issues mma ---
            // A fragments (shared across warps)
            // a0 = A[groupID, kLo], a1 = A[groupID+8, kLo]
            // a2 = A[groupID, kHi], a3 = A[groupID+8, kHi]
            uint32_t a0 = *(uint32_t*)&A_smem[groupID * A_STRIDE + tidInGroup * 2];
            uint32_t a1 = *(uint32_t*)&A_smem[(groupID + 8) * A_STRIDE + tidInGroup * 2];
            uint32_t a2 = *(uint32_t*)&A_smem[groupID * A_STRIDE + 8 + tidInGroup * 2];
            uint32_t a3 = *(uint32_t*)&A_smem[(groupID + 8) * A_STRIDE + 8 + tidInGroup * 2];

            // B fragment: this warp handles N columns [warp_id*8, warp_id*8+8)
            // b0 = B[groupID + warp_n_offset, kLo], b1 = B[..., kHi]
            int b_n = warp_id * 8 + groupID;
            uint32_t b0 = *(uint32_t*)&B_smem[b_n * B_STRIDE + tidInGroup * 2];
            uint32_t b1 = *(uint32_t*)&B_smem[b_n * B_STRIDE + 8 + tidInGroup * 2];

            float c0 = d0, c1 = d1, c2 = d2, c3 = d3;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3));

            __syncthreads();
        }
    }

    // Store results: D[groupID, tidInGroup*2], D[groupID, tidInGroup*2+1], +8 rows
    int gm0 = tile_m + groupID;
    int gm1 = tile_m + groupID + 8;
    int gn0 = tile_n + warp_id * 8 + tidInGroup * 2;
    int gn1 = gn0 + 1;

    if (gm0 < M && gn0 < N) scores[gm0 * N + gn0] = d0;
    if (gm0 < M && gn1 < N) scores[gm0 * N + gn1] = d1;
    if (gm1 < M && gn0 < N) scores[gm1 * N + gn0] = d2;
    if (gm1 < M && gn1 < N) scores[gm1 * N + gn1] = d3;
}

// ============================================================================
// GPU kernel: mma.sync + smem v2 — load full K=128 into smem at once
// ============================================================================
// Only 2 syncthreads per GEMM (load, compute), total 4 per block.
// Smem: A[16][K+PAD] + B[BN][K+PAD].  K=128, PAD=8 → A_STRIDE=136
// Vectorized uint4 loads for coalescing (16 bytes = 8 halfs per thread).

template<int BN_TILE, int K_DIM = 128>
__global__ void quest_score_mma_smem_v2_kernel(
    const half* __restrict__ q_pos,
    const half* __restrict__ q_neg,
    const half* __restrict__ kmax,
    const half* __restrict__ kmin,
    float* __restrict__ scores,
    int M, int N)
{
    constexpr int BM = 16, BN = BN_TILE, K = K_DIM, PAD = 8;
    constexpr int A_STRIDE = K + PAD;  // 136
    constexpr int B_STRIDE = K + PAD;
    constexpr int WARPS = BN / 8;
    constexpr int NTHREADS = WARPS * 32;

    extern __shared__ half smem[];
    half* A_smem = smem;
    half* B_smem = smem + BM * A_STRIDE;

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int groupID = lane_id / 4;
    const int tidInGroup = lane_id % 4;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    const half* A_ptrs[2] = {q_pos, q_neg};
    const half* B_ptrs[2] = {kmax, kmin};

    for (int gemm = 0; gemm < 2; gemm++) {
        const half* A_global = A_ptrs[gemm];
        const half* B_global = B_ptrs[gemm];

        // --- Load A[BM][K] into A_smem with vectorized uint4 (8 halfs) ---
        constexpr int A_ELEMS = BM * K;         // 2048
        constexpr int A_VEC_COUNT = A_ELEMS / 8; // 256 uint4 loads
        for (int i = tid; i < A_VEC_COUNT; i += NTHREADS) {
            int flat = i * 8;
            int row = flat / K;    // 0..15
            int col = flat % K;    // 0,8,16,...,120
            int gm = tile_m + row;
            if (gm < M) {
                uint4 val = *(const uint4*)(&A_global[gm * K + col]);
                *(uint4*)(&A_smem[row * A_STRIDE + col]) = val;
            } else {
                uint4 zero = {0, 0, 0, 0};
                *(uint4*)(&A_smem[row * A_STRIDE + col]) = zero;
            }
        }

        // --- Load B[BN][K] into B_smem with vectorized uint4 ---
        constexpr int B_ELEMS = BN * K;
        constexpr int B_VEC_COUNT = B_ELEMS / 8;
        for (int i = tid; i < B_VEC_COUNT; i += NTHREADS) {
            int flat = i * 8;
            int row = flat / K;
            int col = flat % K;
            int gn = tile_n + row;
            if (gn < N) {
                uint4 val = *(const uint4*)(&B_global[gn * K + col]);
                *(uint4*)(&B_smem[row * B_STRIDE + col]) = val;
            } else {
                uint4 zero = {0, 0, 0, 0};
                *(uint4*)(&B_smem[row * B_STRIDE + col]) = zero;
            }
        }

        __syncthreads();

        // --- K-loop: 8 mma iterations, NO syncthreads ---
        for (int kk = 0; kk < K; kk += 16) {
            // A fragments from A_smem
            uint32_t a0 = *(uint32_t*)&A_smem[groupID * A_STRIDE + kk + tidInGroup * 2];
            uint32_t a1 = *(uint32_t*)&A_smem[(groupID + 8) * A_STRIDE + kk + tidInGroup * 2];
            uint32_t a2 = *(uint32_t*)&A_smem[groupID * A_STRIDE + kk + 8 + tidInGroup * 2];
            uint32_t a3 = *(uint32_t*)&A_smem[(groupID + 8) * A_STRIDE + kk + 8 + tidInGroup * 2];

            // B fragments from B_smem
            int b_n = warp_id * 8 + groupID;
            uint32_t b0 = *(uint32_t*)&B_smem[b_n * B_STRIDE + kk + tidInGroup * 2];
            uint32_t b1 = *(uint32_t*)&B_smem[b_n * B_STRIDE + kk + 8 + tidInGroup * 2];

            float c0 = d0, c1 = d1, c2 = d2, c3 = d3;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3));
        }

        __syncthreads();  // before next GEMM overwrites smem
    }

    // Store
    int gm0 = tile_m + groupID;
    int gm1 = tile_m + groupID + 8;
    int gn0 = tile_n + warp_id * 8 + tidInGroup * 2;
    int gn1 = gn0 + 1;
    if (gm0 < M && gn0 < N) scores[gm0 * N + gn0] = d0;
    if (gm0 < M && gn1 < N) scores[gm0 * N + gn1] = d1;
    if (gm1 < M && gn0 < N) scores[gm1 * N + gn0] = d2;
    if (gm1 < M && gn1 < N) scores[gm1 * N + gn1] = d3;
}

// ============================================================================
// GPU kernel: Quest scoring via mma.sync PTX (no smem, for comparison)
// ============================================================================
// Each warp computes a 16x8 output tile using mma.sync.m16n8k16
// Grid: (ceil(N/8), ceil(M/16), 1)

// Helper: load A fragment for mma.sync m16n8k16
// A is [16, 16] in row-major. Thread t reads:
//   a0 = A[t/4, (t%4)*2 .. (t%4)*2+1]       for k in [0..7] → actually k cols 0..7
//   a1 = A[t/4, (t%4)*2 .. (t%4)*2+1]       for k in [8..15]
//   a2 = A[t/4+8, (t%4)*2 .. (t%4)*2+1]     for k in [0..7]
//   a3 = A[t/4+8, (t%4)*2 .. (t%4)*2+1]     for k in [8..15]
// Each register = 2 consecutive half values packed as uint32.

__device__ __forceinline__ uint32_t pack_half2(half a, half b) {
    uint32_t r;
    half2 h = make_half2(a, b);
    r = *reinterpret_cast<uint32_t*>(&h);
    return r;
}

__global__ void quest_score_mma_kernel(
    const half* __restrict__ q_pos,
    const half* __restrict__ q_neg,
    const half* __restrict__ kmax,
    const half* __restrict__ kmin,
    float* __restrict__ scores,
    int M, int N, int K)
{
    int tile_n = blockIdx.x * 8;
    int tile_m = blockIdx.y * 16;
    int tid = threadIdx.x;  // 0..31 within the warp

    // Thread-to-row mapping for mma m16n8k16
    int row0 = tid / 4;         // rows 0..7
    int row1 = tid / 4 + 8;    // rows 8..15
    int col_group = tid % 4;    // determines which k-columns

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    // Accumulate Q_pos @ Kmax^T
    for (int k_base = 0; k_base < K; k_base += 16) {
        // Load A fragment (Q_pos) — rows tile_m+row0, tile_m+row1
        // k_cols: col_group*2, col_group*2+1 for first 8 k; same for next 8 k
        int k0 = k_base + col_group * 2;
        int k8 = k_base + 8 + col_group * 2;

        // Fragment layout: {row0_kLo, row1_kLo, row0_kHi, row1_kHi}
        //   a0 = A[groupID,   k_lo]    a1 = A[groupID+8, k_lo]
        //   a2 = A[groupID,   k_hi]    a3 = A[groupID+8, k_hi]
        int r0 = tile_m + row0;
        int r1 = tile_m + row1;
        uint32_t a0 = (r0 < M) ? pack_half2(q_pos[r0 * K + k0], q_pos[r0 * K + k0 + 1]) : 0;
        uint32_t a1 = (r1 < M) ? pack_half2(q_pos[r1 * K + k0], q_pos[r1 * K + k0 + 1]) : 0;
        uint32_t a2 = (r0 < M) ? pack_half2(q_pos[r0 * K + k8], q_pos[r0 * K + k8 + 1]) : 0;
        uint32_t a3 = (r1 < M) ? pack_half2(q_pos[r1 * K + k8], q_pos[r1 * K + k8 + 1]) : 0;

        // Load B fragment (Kmax transposed) — B is col-major [K, N_tile]
        // For m16n8k16 B: b0 = B[col_group*2..col_group*2+1, n_col] packed
        // B is Kmax^T, so B[k, n] = Kmax[n, k]
        // Thread mapping for B: thread t reads
        //   b0 = B[col_group*2..col_group*2+1, row0_b]  (k in [0..7])
        //   b1 = B[col_group*2..col_group*2+1, row0_b]  (k in [8..15])
        // where row0_b maps to n dimension
        // Actually for B (col-major, [K,8]):
        //   b0: k = col_group*2, col_group*2+1 with n = tid/4
        //   b1: k = 8+col_group*2, 8+col_group*2+1 with n = tid/4
        int n_idx = tile_n + tid / 4;  // n column for this thread

        uint32_t b0, b1;
        if (n_idx < N) {
            b0 = pack_half2(kmax[n_idx * K + k0], kmax[n_idx * K + k0 + 1]);
            b1 = pack_half2(kmax[n_idx * K + k8], kmax[n_idx * K + k8 + 1]);
        } else {
            b0 = b1 = 0;
        }

        {
            float c0 = d0, c1 = d1, c2 = d2, c3 = d3;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " {%0, %1, %2, %3},"
                " {%4, %5, %6, %7},"
                " {%8, %9},"
                " {%10, %11, %12, %13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3)
            );
        }
    }

    // Accumulate Q_neg @ Kmin^T (same structure, different data)
    for (int k_base = 0; k_base < K; k_base += 16) {
        int k0 = k_base + col_group * 2;
        int k8 = k_base + 8 + col_group * 2;
        int r0 = tile_m + row0;
        int r1 = tile_m + row1;

        uint32_t a0 = (r0 < M) ? pack_half2(q_neg[r0 * K + k0], q_neg[r0 * K + k0 + 1]) : 0;
        uint32_t a1 = (r1 < M) ? pack_half2(q_neg[r1 * K + k0], q_neg[r1 * K + k0 + 1]) : 0;
        uint32_t a2 = (r0 < M) ? pack_half2(q_neg[r0 * K + k8], q_neg[r0 * K + k8 + 1]) : 0;
        uint32_t a3 = (r1 < M) ? pack_half2(q_neg[r1 * K + k8], q_neg[r1 * K + k8 + 1]) : 0;

        int n_idx = tile_n + tid / 4;
        uint32_t b0, b1;
        if (n_idx < N) {
            b0 = pack_half2(kmin[n_idx * K + k0], kmin[n_idx * K + k0 + 1]);
            b1 = pack_half2(kmin[n_idx * K + k8], kmin[n_idx * K + k8 + 1]);
        } else {
            b0 = b1 = 0;
        }

        {
            float c0 = d0, c1 = d1, c2 = d2, c3 = d3;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " {%0, %1, %2, %3},"
                " {%4, %5, %6, %7},"
                " {%8, %9},"
                " {%10, %11, %12, %13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3)
            );
        }
    }

    // Store results — D fragment mapping for m16n8:
    //   d0 = D[row0, (tid%2)*2  ]
    //   d1 = D[row0, (tid%2)*2+1]
    //   d2 = D[row1, (tid%2)*2  ]
    //   d3 = D[row1, (tid%2)*2+1]
    // But wait, the correct mapping for m16n8k16 D fragment:
    //   Thread t (0..31):
    //     row_pair = t / 4  (0..7)
    //     col_pair = t % 4  → but n=8 only needs t%2 for col indexing
    // Actually the PTX spec for m16n8k16:
    //   D is [16, 8], thread t holds:
    //     d0 = D[groupID*8 + (t%4)*2,     (t/4)%2 * ... ]
    // This is getting complicated. Let me use the known mapping:
    //
    // For mma m16n8k16 with .f32 accumulator:
    //   Thread t in warp (0..31):
    //   groupID = t / 4  (0..7)
    //   tidInGroup = t % 4  (0..3)
    //
    //   d0 = D[groupID,     tidInGroup*2  ]  -- NO, n=8 but tidInGroup*2 goes to 6
    //   Hmm, that's only 4*2=8 columns, matches n=8!
    //   d0 = D[groupID,     tidInGroup * 2    ]
    //   d1 = D[groupID,     tidInGroup * 2 + 1]
    //   d2 = D[groupID + 8, tidInGroup * 2    ]
    //   d3 = D[groupID + 8, tidInGroup * 2 + 1]
    int out_row0 = tile_m + tid / 4;
    int out_row1 = tile_m + tid / 4 + 8;
    int out_col0 = tile_n + (tid % 4) * 2;
    int out_col1 = out_col0 + 1;

    if (out_row0 < M && out_col0 < N) scores[out_row0 * N + out_col0] = d0;
    if (out_row0 < M && out_col1 < N) scores[out_row0 * N + out_col1] = d1;
    if (out_row1 < M && out_col0 < N) scores[out_row1 * N + out_col0] = d2;
    if (out_row1 < M && out_col1 < N) scores[out_row1 * N + out_col1] = d3;
}

// ============================================================================
// Test helpers
// ============================================================================
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void fill_random(float* data, int n, float lo, float hi) {
    for (int i = 0; i < n; i++)
        data[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

void float_to_half(const float* src, half* dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = __float2half(src[i]);
}

// ============================================================================
// Debug: mma.sync m16n8k16 fragment mapping test
// A = [16,16] row-major, B = [16,8] col-major, D = A @ B
// A[m,k] = (m==0 && k<16) ? 1.0 : 0.0 (first row all-1s)
// B[k,n] = k+1 for all n (B col-major, same column repeated)
// Expected: D[0,n] = sum(k=0..15, 1.0 * (k+1)) = 136.0 for all n
//           D[m>0,n] = 0.0
// ============================================================================
__global__ void debug_mma_kernel(float* out) {
    int tid = threadIdx.x;
    int row0 = tid / 4;
    int row1 = tid / 4 + 8;
    int col_group = tid % 4;

    // Build A fragment: row 0 = all 1.0h, rest = 0
    // Fragment layout: a0=A[groupID,kLo] a1=A[groupID+8,kLo] a2=A[groupID,kHi] a3=A[groupID+8,kHi]
    half one = __float2half(1.0f);
    half zero = __float2half(0.0f);
    uint32_t a0, a1, a2, a3;
    a0 = (row0 == 0) ? pack_half2(one, one) : pack_half2(zero, zero);  // A[groupID, kLo]
    a1 = pack_half2(zero, zero);  // A[groupID+8, kLo] — row 8+ always 0
    a2 = (row0 == 0) ? pack_half2(one, one) : pack_half2(zero, zero);  // A[groupID, kHi]
    a3 = pack_half2(zero, zero);  // A[groupID+8, kHi] — row 8+ always 0

    // Build B fragment: B[k,n] = k+1 for all n
    // Thread tid: b0 = B[col_group*2, tid/4], B[col_group*2+1, tid/4]
    //           = (col_group*2+1, col_group*2+2)
    // b1 = B[8+col_group*2, tid/4], B[8+col_group*2+1, tid/4]
    //    = (8+col_group*2+1, 8+col_group*2+2)
    half k0_val = __float2half((float)(col_group * 2 + 1));
    half k1_val = __float2half((float)(col_group * 2 + 2));
    half k8_val = __float2half((float)(8 + col_group * 2 + 1));
    half k9_val = __float2half((float)(8 + col_group * 2 + 2));
    uint32_t b0 = pack_half2(k0_val, k1_val);
    uint32_t b1 = pack_half2(k8_val, k9_val);

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0, %1, %2, %3},"
        " {%4, %5, %6, %7},"
        " {%8, %9},"
        " {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );

    // D[row0, (tid%4)*2] = d0, D[row0, (tid%4)*2+1] = d1
    // D[row1, (tid%4)*2] = d2, D[row1, (tid%4)*2+1] = d3
    out[row0 * 8 + (tid % 4) * 2]     = d0;
    out[row0 * 8 + (tid % 4) * 2 + 1] = d1;
    out[row1 * 8 + (tid % 4) * 2]     = d2;
    out[row1 * 8 + (tid % 4) * 2 + 1] = d3;
}

bool test_debug_mma() {
    printf("[Debug] mma.sync m16n8k16 fragment mapping\n");
    float *d_out, h_out[128];
    CHECK_CUDA(cudaMalloc(&d_out, 128 * 4));
    debug_mma_kernel<<<1, 32>>>(d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 128 * 4, cudaMemcpyDeviceToHost));

    printf("  D[0,0..7] (expect 136.0): ");
    for (int j = 0; j < 8; j++) printf("%.1f ", h_out[j]);
    printf("\n");
    printf("  D[1,0..7] (expect 0.0):   ");
    for (int j = 0; j < 8; j++) printf("%.1f ", h_out[8 + j]);
    printf("\n");

    bool ok = true;
    for (int j = 0; j < 8; j++) {
        if (h_out[j] != 136.0f) { ok = false; printf("  D[0,%d]=%.1f != 136.0\n", j, h_out[j]); }
    }
    for (int m = 1; m < 16; m++) {
        for (int j = 0; j < 8; j++) {
            if (h_out[m * 8 + j] != 0.0f) {
                ok = false;
                if (m < 3) printf("  D[%d,%d]=%.1f != 0.0\n", m, j, h_out[m * 8 + j]);
            }
        }
    }
    printf("  %s\n", ok ? "PASS" : "FAIL");
    cudaFree(d_out);
    return ok;
}

// ============================================================================
// Test 1: Validate decomposition (CPU)
// ============================================================================
bool test_decomposition() {
    printf("[Test 1] Validate decomposition: max(q*kmax,q*kmin) == q_pos*kmax + q_neg*kmin\n");
    int M = 4, N = 8, K = 16;
    float q[64], kmax[128], kmin[128], scores_orig[32], scores_decomp[32];

    srand(42);
    fill_random(q, M * K, -1.0f, 1.0f);
    fill_random(kmax, N * K, -0.5f, 2.0f);
    // kmin <= kmax element-wise
    for (int i = 0; i < N * K; i++)
        kmin[i] = kmax[i] - fabsf((float)rand() / RAND_MAX);

    quest_score_cpu(q, kmax, kmin, scores_orig, M, N, K);
    quest_score_decomposed_cpu(q, kmax, kmin, scores_decomp, M, N, K);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++)
        max_err = fmaxf(max_err, fabsf(scores_orig[i] - scores_decomp[i]));

    printf("  max abs error: %.6e\n", max_err);
    bool ok = max_err < 1e-5f;
    printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ============================================================================
// Test 2: wmma Quest scoring
// ============================================================================
bool test_wmma_quest() {
    printf("[Test 2] wmma Quest scoring (16x16x16)\n");

    // Padded to multiples of 16
    const int M = 16, N = 32, K = 128;  // 16 queries, 32 pages, head_dim=128
    float *h_q, *h_kmax, *h_kmin, *h_scores_cpu, *h_scores_gpu;
    half *h_q_pos_h, *h_q_neg_h, *h_kmax_h, *h_kmin_h;

    h_q = new float[M * K];
    h_kmax = new float[N * K];
    h_kmin = new float[N * K];
    h_scores_cpu = new float[M * N];
    h_scores_gpu = new float[M * N];
    h_q_pos_h = new half[M * K];
    h_q_neg_h = new half[M * K];
    h_kmax_h = new half[N * K];
    h_kmin_h = new half[N * K];

    srand(123);
    fill_random(h_q, M * K, -1.0f, 1.0f);
    fill_random(h_kmax, N * K, -0.5f, 2.0f);
    for (int i = 0; i < N * K; i++)
        h_kmin[i] = h_kmax[i] - fabsf(0.5f * (float)rand() / RAND_MAX);

    // CPU reference
    quest_score_cpu(h_q, h_kmax, h_kmin, h_scores_cpu, M, N, K);

    // Prepare FP16 q_pos, q_neg
    for (int i = 0; i < M * K; i++) {
        h_q_pos_h[i] = __float2half(fmaxf(h_q[i], 0.0f));
        h_q_neg_h[i] = __float2half(fminf(h_q[i], 0.0f));
    }
    float_to_half(h_kmax, h_kmax_h, N * K);
    float_to_half(h_kmin, h_kmin_h, N * K);

    // GPU alloc
    half *d_q_pos, *d_q_neg, *d_kmax, *d_kmin;
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_q_pos, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_q_neg, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmax, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmin, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_scores, M * N * 4));

    CHECK_CUDA(cudaMemcpy(d_q_pos, h_q_pos_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_neg, h_q_neg_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmax, h_kmax_h, N * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmin, h_kmin_h, N * K * 2, cudaMemcpyHostToDevice));

    // Launch wmma kernel
    dim3 grid(N / 16, M / 16);
    dim3 block(32);  // one warp
    quest_score_wmma_kernel<<<grid, block>>>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_scores_gpu, d_scores, M * N * 4, cudaMemcpyDeviceToHost));

    // Compare
    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float diff = fabsf(h_scores_cpu[i * N + j] - h_scores_gpu[i * N + j]);
            float rel = diff / (fabsf(h_scores_cpu[i * N + j]) + 1e-6f);
            if (rel > max_err) max_err = rel;
            if (rel > 0.05f) {  // 5% relative tolerance for fp16
                if (err_count < 5)
                    printf("  [%d,%d] cpu=%.4f gpu=%.4f diff=%.4f\n",
                           i, j, h_scores_cpu[i * N + j], h_scores_gpu[i * N + j], diff);
                err_count++;
            }
        }
    }
    printf("  max relative error: %.4f%% (%d errors > 5%%)\n", max_err * 100.0f, err_count);
    printf("  sample: cpu[0,0]=%.4f gpu[0,0]=%.4f  cpu[0,1]=%.4f gpu[0,1]=%.4f\n",
           h_scores_cpu[0], h_scores_gpu[0], h_scores_cpu[1], h_scores_gpu[1]);
    bool ok = err_count == 0;
    printf("  %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_q_pos); cudaFree(d_q_neg); cudaFree(d_kmax); cudaFree(d_kmin); cudaFree(d_scores);
    delete[] h_q; delete[] h_kmax; delete[] h_kmin; delete[] h_scores_cpu; delete[] h_scores_gpu;
    delete[] h_q_pos_h; delete[] h_q_neg_h; delete[] h_kmax_h; delete[] h_kmin_h;
    return ok;
}

// ============================================================================
// Test 3: mma.sync PTX Quest scoring
// ============================================================================
bool test_mma_quest() {
    printf("[Test 3] mma.sync PTX Quest scoring (m16n8k16)\n");

    const int M = 16, N = 32, K = 128;
    float *h_q, *h_kmax, *h_kmin, *h_scores_cpu, *h_scores_gpu;
    half *h_q_pos_h, *h_q_neg_h, *h_kmax_h, *h_kmin_h;

    h_q = new float[M * K];
    h_kmax = new float[N * K];
    h_kmin = new float[N * K];
    h_scores_cpu = new float[M * N];
    h_scores_gpu = new float[M * N];
    h_q_pos_h = new half[M * K];
    h_q_neg_h = new half[M * K];
    h_kmax_h = new half[N * K];
    h_kmin_h = new half[N * K];

    srand(456);
    fill_random(h_q, M * K, -1.0f, 1.0f);
    fill_random(h_kmax, N * K, -0.5f, 2.0f);
    for (int i = 0; i < N * K; i++)
        h_kmin[i] = h_kmax[i] - fabsf(0.5f * (float)rand() / RAND_MAX);

    quest_score_cpu(h_q, h_kmax, h_kmin, h_scores_cpu, M, N, K);

    for (int i = 0; i < M * K; i++) {
        h_q_pos_h[i] = __float2half(fmaxf(h_q[i], 0.0f));
        h_q_neg_h[i] = __float2half(fminf(h_q[i], 0.0f));
    }
    float_to_half(h_kmax, h_kmax_h, N * K);
    float_to_half(h_kmin, h_kmin_h, N * K);

    half *d_q_pos, *d_q_neg, *d_kmax, *d_kmin;
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_q_pos, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_q_neg, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmax, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmin, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_scores, M * N * 4));
    CHECK_CUDA(cudaMemset(d_scores, 0, M * N * 4));

    CHECK_CUDA(cudaMemcpy(d_q_pos, h_q_pos_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_neg, h_q_neg_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmax, h_kmax_h, N * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmin, h_kmin_h, N * K * 2, cudaMemcpyHostToDevice));

    // mma.sync kernel: each warp handles 16x8, grid covers M x N
    dim3 grid((N + 7) / 8, (M + 15) / 16);
    dim3 block(32);
    quest_score_mma_kernel<<<grid, block>>>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_scores_gpu, d_scores, M * N * 4, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float diff = fabsf(h_scores_cpu[i * N + j] - h_scores_gpu[i * N + j]);
            float rel = diff / (fabsf(h_scores_cpu[i * N + j]) + 1e-6f);
            if (rel > max_err) max_err = rel;
            if (rel > 0.05f) {
                if (err_count < 5)
                    printf("  [%d,%d] cpu=%.4f gpu=%.4f diff=%.4f\n",
                           i, j, h_scores_cpu[i * N + j], h_scores_gpu[i * N + j], diff);
                err_count++;
            }
        }
    }
    printf("  max relative error: %.4f%% (%d errors > 5%%)\n", max_err * 100.0f, err_count);
    printf("  sample: cpu[0,0]=%.4f gpu[0,0]=%.4f  cpu[15,31]=%.4f gpu[15,31]=%.4f\n",
           h_scores_cpu[0], h_scores_gpu[0], h_scores_cpu[15 * N + 31], h_scores_gpu[15 * N + 31]);
    bool ok = err_count == 0;
    printf("  %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_q_pos); cudaFree(d_q_neg); cudaFree(d_kmax); cudaFree(d_kmin); cudaFree(d_scores);
    delete[] h_q; delete[] h_kmax; delete[] h_kmin; delete[] h_scores_cpu; delete[] h_scores_gpu;
    delete[] h_q_pos_h; delete[] h_q_neg_h; delete[] h_kmax_h; delete[] h_kmin_h;
    return ok;
}

// ============================================================================
// Test 4: Realistic size (63 queries, 1800 pages, head_dim=128)
// ============================================================================
bool test_realistic_size() {
    printf("[Test 4] Realistic: M=64 queries, N=1800 pages, K=128 head_dim (wmma)\n");

    // Pad M to 64 (next multiple of 16 >= 63), pad N to 1808 (next multiple of 16)
    const int M_real = 63, N_real = 1800, K = 128;
    const int M = 64, N = 1808;  // padded

    float *h_q = new float[M * K]();
    float *h_kmax = new float[N * K]();
    float *h_kmin = new float[N * K]();
    float *h_scores_cpu = new float[M * N]();
    float *h_scores_gpu = new float[M * N]();

    srand(789);
    fill_random(h_q, M_real * K, -1.0f, 1.0f);  // only fill real rows
    fill_random(h_kmax, N_real * K, -0.5f, 2.0f);
    for (int i = 0; i < N_real * K; i++)
        h_kmin[i] = h_kmax[i] - fabsf(0.5f * (float)rand() / RAND_MAX);

    quest_score_cpu(h_q, h_kmax, h_kmin, h_scores_cpu, M_real, N, K);

    // Convert to half
    half *h_q_pos_h = new half[M * K];
    half *h_q_neg_h = new half[M * K];
    half *h_kmax_h = new half[N * K];
    half *h_kmin_h = new half[N * K];

    for (int i = 0; i < M * K; i++) {
        h_q_pos_h[i] = __float2half(fmaxf(h_q[i], 0.0f));
        h_q_neg_h[i] = __float2half(fminf(h_q[i], 0.0f));
    }
    for (int i = 0; i < N * K; i++) {
        h_kmax_h[i] = __float2half(h_kmax[i]);
        h_kmin_h[i] = __float2half(h_kmin[i]);
    }

    half *d_q_pos, *d_q_neg, *d_kmax, *d_kmin;
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_q_pos, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_q_neg, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmax, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmin, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_scores, M * N * 4));

    CHECK_CUDA(cudaMemcpy(d_q_pos, h_q_pos_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_neg, h_q_neg_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmax, h_kmax_h, N * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmin, h_kmin_h, N * K * 2, cudaMemcpyHostToDevice));

    dim3 grid(N / 16, M / 16);
    dim3 block(32);
    quest_score_wmma_kernel<<<grid, block>>>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_scores_gpu, d_scores, M * N * 4, cudaMemcpyDeviceToHost));

    // Compare only real (non-padded) entries
    // Use combined tolerance: abs > 0.5 OR rel > 10% (fp16 accumulation over K=128)
    float max_abs = 0.0f, max_rel = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M_real; i++) {
        for (int j = 0; j < N_real; j++) {
            float cpu_val = h_scores_cpu[i * N + j];
            float gpu_val = h_scores_gpu[i * N + j];
            float diff = fabsf(cpu_val - gpu_val);
            float rel = diff / (fabsf(cpu_val) + 1e-3f);
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            // Error if absolute diff > 0.5 AND relative > 10%
            if (diff > 0.5f && rel > 0.10f) {
                if (err_count < 3)
                    printf("  [%d,%d] cpu=%.4f gpu=%.4f abs=%.4f rel=%.2f%%\n",
                           i, j, cpu_val, gpu_val, diff, rel * 100);
                err_count++;
            }
        }
    }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    // warmup
    for (int i = 0; i < 10; i++)
        quest_score_wmma_kernel<<<grid, block>>>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    int ITERS = 100;
    for (int i = 0; i < ITERS; i++)
        quest_score_wmma_kernel<<<grid, block>>>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("  max abs=%.4f  max rel=%.4f%%  (%d significant errors)\n",
           max_abs, max_rel * 100.0f, err_count);
    printf("  wmma kernel time: %.3f us (avg over %d iters)\n", ms / ITERS * 1000.0f, ITERS);
    printf("  M=%d(real %d) N=%d(real %d) K=%d  → %.2f GFLOPS\n",
           M, M_real, N, N_real, K,
           2.0f * M * N * K * 2 / (ms / ITERS * 1e-3f) / 1e9f);  // 2 GEMMs
    bool ok = err_count == 0;
    printf("  %s\n", ok ? "PASS" : "FAIL");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_q_pos); cudaFree(d_q_neg); cudaFree(d_kmax); cudaFree(d_kmin); cudaFree(d_scores);
    delete[] h_q; delete[] h_kmax; delete[] h_kmin; delete[] h_scores_cpu; delete[] h_scores_gpu;
    delete[] h_q_pos_h; delete[] h_q_neg_h; delete[] h_kmax_h; delete[] h_kmin_h;
    return ok;
}

// ============================================================================
// Test 5: Realistic size with mma.sync PTX (compare timing vs wmma)
// ============================================================================
bool test_realistic_mma() {
    printf("[Test 5] Realistic: M=64, N=1808, K=128 (mma.sync PTX)\n");

    const int M_real = 63, N_real = 1800, K = 128;
    const int M = 64, N = 1808;

    float *h_q = new float[M * K]();
    float *h_kmax = new float[N * K]();
    float *h_kmin = new float[N * K]();
    float *h_scores_cpu = new float[M * N]();
    float *h_scores_gpu = new float[M * N]();

    srand(789);
    fill_random(h_q, M_real * K, -1.0f, 1.0f);
    fill_random(h_kmax, N_real * K, -0.5f, 2.0f);
    for (int i = 0; i < N_real * K; i++)
        h_kmin[i] = h_kmax[i] - fabsf(0.5f * (float)rand() / RAND_MAX);

    quest_score_cpu(h_q, h_kmax, h_kmin, h_scores_cpu, M_real, N, K);

    half *h_q_pos_h = new half[M * K];
    half *h_q_neg_h = new half[M * K];
    half *h_kmax_h = new half[N * K];
    half *h_kmin_h = new half[N * K];
    for (int i = 0; i < M * K; i++) {
        h_q_pos_h[i] = __float2half(fmaxf(h_q[i], 0.0f));
        h_q_neg_h[i] = __float2half(fminf(h_q[i], 0.0f));
    }
    for (int i = 0; i < N * K; i++) {
        h_kmax_h[i] = __float2half(h_kmax[i]);
        h_kmin_h[i] = __float2half(h_kmin[i]);
    }

    half *d_q_pos, *d_q_neg, *d_kmax, *d_kmin;
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_q_pos, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_q_neg, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmax, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmin, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_scores, M * N * 4));
    CHECK_CUDA(cudaMemcpy(d_q_pos, h_q_pos_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_neg, h_q_neg_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmax, h_kmax_h, N * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmin, h_kmin_h, N * K * 2, cudaMemcpyHostToDevice));

    // mma.sync: 16x8 output tile
    dim3 grid((N + 7) / 8, (M + 15) / 16);
    dim3 block(32);
    quest_score_mma_kernel<<<grid, block>>>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_scores_gpu, d_scores, M * N * 4, cudaMemcpyDeviceToHost));

    float max_abs = 0.0f, max_rel = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M_real; i++) {
        for (int j = 0; j < N_real; j++) {
            float diff = fabsf(h_scores_cpu[i*N+j] - h_scores_gpu[i*N+j]);
            float rel = diff / (fabsf(h_scores_cpu[i*N+j]) + 1e-3f);
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            if (diff > 0.5f && rel > 0.10f) err_count++;
        }
    }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < 10; i++)
        quest_score_mma_kernel<<<grid, block>>>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    int ITERS = 100;
    for (int i = 0; i < ITERS; i++)
        quest_score_mma_kernel<<<grid, block>>>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("  max abs=%.4f  max rel=%.4f%%  (%d significant errors)\n",
           max_abs, max_rel * 100.0f, err_count);
    printf("  mma.sync kernel time: %.3f us (avg over %d iters)\n", ms / ITERS * 1000.0f, ITERS);
    printf("  M=%d N=%d K=%d  → %.2f GFLOPS\n", M, N, K,
           2.0f * M * N * K * 2 / (ms / ITERS * 1e-3f) / 1e9f);
    bool ok = err_count == 0;
    printf("  %s\n", ok ? "PASS" : "FAIL");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_q_pos); cudaFree(d_q_neg); cudaFree(d_kmax); cudaFree(d_kmin); cudaFree(d_scores);
    delete[] h_q; delete[] h_kmax; delete[] h_kmin; delete[] h_scores_cpu; delete[] h_scores_gpu;
    delete[] h_q_pos_h; delete[] h_q_neg_h; delete[] h_kmax_h; delete[] h_kmin_h;
    return ok;
}

// ============================================================================
// Test 6: Realistic size with smem-tiled mma.sync (sweep BN tile sizes)
// ============================================================================
template<int BN>
float bench_smem_kernel(half* d_q_pos, half* d_q_neg, half* d_kmax, half* d_kmin,
                        float* d_scores, float* h_scores_gpu,
                        const float* h_scores_cpu,
                        int M, int M_real, int N, int N_real, int K)
{
    constexpr int BM = 16;
    constexpr int PAD = 8;
    constexpr int A_STRIDE = 16 + PAD;
    constexpr int B_STRIDE = 16 + PAD;
    constexpr int WARPS = BN / 8;
    int smem_bytes = (BM * A_STRIDE + BN * B_STRIDE) * sizeof(half);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(WARPS * 32);

    CHECK_CUDA(cudaMemset(d_scores, 0, M * N * 4));
    quest_score_mma_smem_kernel<BN><<<grid, block, smem_bytes>>>(
        d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_scores_gpu, d_scores, M * N * 4, cudaMemcpyDeviceToHost));

    // Validate
    int err_count = 0;
    for (int i = 0; i < M_real; i++)
        for (int j = 0; j < N_real; j++) {
            float diff = fabsf(h_scores_cpu[i*N+j] - h_scores_gpu[i*N+j]);
            float rel = diff / (fabsf(h_scores_cpu[i*N+j]) + 1e-3f);
            if (diff > 0.5f && rel > 0.10f) err_count++;
        }

    // Timing
    for (int i = 0; i < 10; i++)
        quest_score_mma_smem_kernel<BN><<<grid, block, smem_bytes>>>(
            d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    int ITERS = 100;
    for (int i = 0; i < ITERS; i++)
        quest_score_mma_smem_kernel<BN><<<grid, block, smem_bytes>>>(
            d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float us = ms / ITERS * 1000.0f;
    float gflops = 2.0f * M * N * K * 2 / (ms / ITERS * 1e-3f) / 1e9f;

    printf("  BN=%3d  warps=%2d  grid=(%3d,%d)  %s  %.3f us  %.0f GFLOPS\n",
           BN, WARPS, grid.x, grid.y,
           err_count == 0 ? "PASS" : "FAIL",
           us, gflops);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return (err_count == 0) ? us : -1.0f;
}

// ============================================================================
// Benchmark helper for smem v2 kernel (full-K load)
// ============================================================================
template<int BN>
float bench_smem_v2_kernel(half* d_q_pos, half* d_q_neg, half* d_kmax, half* d_kmin,
                           float* d_scores, float* h_scores_gpu,
                           const float* h_scores_cpu,
                           int M, int M_real, int N, int N_real)
{
    constexpr int BM = 16, K = 128, PAD = 8;
    constexpr int A_STRIDE = K + PAD;   // 136
    constexpr int B_STRIDE = K + PAD;
    constexpr int WARPS = BN / 8;
    int smem_bytes = (BM * A_STRIDE + BN * B_STRIDE) * sizeof(half);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(WARPS * 32);

    // Check smem limit
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (smem_bytes > max_smem) {
        printf("  BN=%3d  warps=%2d  SKIP (smem %d > %d)\n", BN, WARPS, smem_bytes, max_smem);
        return -1.0f;
    }

    CHECK_CUDA(cudaMemset(d_scores, 0, M * N * 4));
    quest_score_mma_smem_v2_kernel<BN><<<grid, block, smem_bytes>>>(
        d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_scores_gpu, d_scores, M * N * 4, cudaMemcpyDeviceToHost));

    // Validate
    int err_count = 0;
    for (int i = 0; i < M_real; i++)
        for (int j = 0; j < N_real; j++) {
            float diff = fabsf(h_scores_cpu[i*N+j] - h_scores_gpu[i*N+j]);
            float rel = diff / (fabsf(h_scores_cpu[i*N+j]) + 1e-3f);
            if (diff > 0.5f && rel > 0.10f) err_count++;
        }

    // Timing
    for (int i = 0; i < 10; i++)
        quest_score_mma_smem_v2_kernel<BN><<<grid, block, smem_bytes>>>(
            d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    int ITERS = 100;
    for (int i = 0; i < ITERS; i++)
        quest_score_mma_smem_v2_kernel<BN><<<grid, block, smem_bytes>>>(
            d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float us = ms / ITERS * 1000.0f;
    float gflops = 2.0f * M * N * K * 2 / (ms / ITERS * 1e-3f) / 1e9f;

    printf("  BN=%3d  warps=%2d  grid=(%3d,%d)  smem=%5dB  %s  %.3f us  %.0f GFLOPS\n",
           BN, WARPS, grid.x, grid.y, smem_bytes,
           err_count == 0 ? "PASS" : "FAIL",
           us, gflops);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return (err_count == 0) ? us : -1.0f;
}

bool test_smem_mma() {
    printf("[Test 6] Smem-tiled mma.sync v1 (BK=16): M=64, N=1808, K=128 (sweep BN)\n");

    const int M_real = 63, N_real = 1800, K = 128;
    const int M = 64, N = 1808;

    float *h_q = new float[M * K]();
    float *h_kmax = new float[N * K]();
    float *h_kmin = new float[N * K]();
    float *h_scores_cpu = new float[M * N]();
    float *h_scores_gpu = new float[M * N]();

    srand(789);
    fill_random(h_q, M_real * K, -1.0f, 1.0f);
    fill_random(h_kmax, N_real * K, -0.5f, 2.0f);
    for (int i = 0; i < N_real * K; i++)
        h_kmin[i] = h_kmax[i] - fabsf(0.5f * (float)rand() / RAND_MAX);
    quest_score_cpu(h_q, h_kmax, h_kmin, h_scores_cpu, M_real, N, K);

    half *h_q_pos_h = new half[M * K];
    half *h_q_neg_h = new half[M * K];
    half *h_kmax_h = new half[N * K];
    half *h_kmin_h = new half[N * K];
    for (int i = 0; i < M * K; i++) {
        h_q_pos_h[i] = __float2half(fmaxf(h_q[i], 0.0f));
        h_q_neg_h[i] = __float2half(fminf(h_q[i], 0.0f));
    }
    for (int i = 0; i < N * K; i++) {
        h_kmax_h[i] = __float2half(h_kmax[i]);
        h_kmin_h[i] = __float2half(h_kmin[i]);
    }

    half *d_q_pos, *d_q_neg, *d_kmax, *d_kmin;
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_q_pos, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_q_neg, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmax, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmin, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_scores, M * N * 4));
    CHECK_CUDA(cudaMemcpy(d_q_pos, h_q_pos_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_neg, h_q_neg_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmax, h_kmax_h, N * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmin, h_kmin_h, N * K * 2, cudaMemcpyHostToDevice));

    float t8  = bench_smem_kernel<8> (d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real, K);
    float t16 = bench_smem_kernel<16>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real, K);
    float t32 = bench_smem_kernel<32>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real, K);
    float t64 = bench_smem_kernel<64>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real, K);
    float t128= bench_smem_kernel<128>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real, K);

    bool ok = (t8 > 0 && t16 > 0 && t32 > 0 && t64 > 0 && t128 > 0);
    printf("  %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_q_pos); cudaFree(d_q_neg); cudaFree(d_kmax); cudaFree(d_kmin); cudaFree(d_scores);
    delete[] h_q; delete[] h_kmax; delete[] h_kmin; delete[] h_scores_cpu; delete[] h_scores_gpu;
    delete[] h_q_pos_h; delete[] h_q_neg_h; delete[] h_kmax_h; delete[] h_kmin_h;
    return ok;
}

// ============================================================================
// Test 7: Smem v2 (full-K load) mma.sync sweep
// ============================================================================
bool test_smem_mma_v2() {
    printf("[Test 7] Smem-tiled mma.sync v2 (full K=128 load): M=64, N=1808 (sweep BN)\n");

    const int M_real = 63, N_real = 1800, K = 128;
    const int M = 64, N = 1808;

    float *h_q = new float[M * K]();
    float *h_kmax = new float[N * K]();
    float *h_kmin = new float[N * K]();
    float *h_scores_cpu = new float[M * N]();
    float *h_scores_gpu = new float[M * N]();

    srand(789);
    fill_random(h_q, M_real * K, -1.0f, 1.0f);
    fill_random(h_kmax, N_real * K, -0.5f, 2.0f);
    for (int i = 0; i < N_real * K; i++)
        h_kmin[i] = h_kmax[i] - fabsf(0.5f * (float)rand() / RAND_MAX);
    quest_score_cpu(h_q, h_kmax, h_kmin, h_scores_cpu, M_real, N, K);

    half *h_q_pos_h = new half[M * K];
    half *h_q_neg_h = new half[M * K];
    half *h_kmax_h = new half[N * K];
    half *h_kmin_h = new half[N * K];
    for (int i = 0; i < M * K; i++) {
        h_q_pos_h[i] = __float2half(fmaxf(h_q[i], 0.0f));
        h_q_neg_h[i] = __float2half(fminf(h_q[i], 0.0f));
    }
    for (int i = 0; i < N * K; i++) {
        h_kmax_h[i] = __float2half(h_kmax[i]);
        h_kmin_h[i] = __float2half(h_kmin[i]);
    }

    half *d_q_pos, *d_q_neg, *d_kmax, *d_kmin;
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_q_pos, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_q_neg, M * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmax, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_kmin, N * K * 2));
    CHECK_CUDA(cudaMalloc(&d_scores, M * N * 4));
    CHECK_CUDA(cudaMemcpy(d_q_pos, h_q_pos_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_neg, h_q_neg_h, M * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmax, h_kmax_h, N * K * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kmin, h_kmin_h, N * K * 2, cudaMemcpyHostToDevice));

    float t8  = bench_smem_v2_kernel<8> (d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real);
    float t16 = bench_smem_v2_kernel<16>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real);
    float t32 = bench_smem_v2_kernel<32>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real);
    float t64 = bench_smem_v2_kernel<64>(d_q_pos, d_q_neg, d_kmax, d_kmin, d_scores, h_scores_gpu, h_scores_cpu, M, M_real, N, N_real);

    bool ok = (t8 > 0 && t16 > 0 && t32 > 0 && t64 > 0);
    printf("  %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_q_pos); cudaFree(d_q_neg); cudaFree(d_kmax); cudaFree(d_kmin); cudaFree(d_scores);
    delete[] h_q; delete[] h_kmax; delete[] h_kmin; delete[] h_scores_cpu; delete[] h_scores_gpu;
    delete[] h_q_pos_h; delete[] h_q_neg_h; delete[] h_kmax_h; delete[] h_kmin_h;
    return ok;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    int pass = 0, fail = 0;

    if (test_debug_mma()) pass++; else fail++;
    printf("\n");
    if (test_decomposition()) pass++; else fail++;
    printf("\n");
    if (test_wmma_quest()) pass++; else fail++;
    printf("\n");
    if (test_mma_quest()) pass++; else fail++;
    printf("\n");
    if (test_realistic_size()) pass++; else fail++;
    printf("\n");
    if (test_realistic_mma()) pass++; else fail++;
    printf("\n");
    if (test_smem_mma()) pass++; else fail++;
    printf("\n");
    if (test_smem_mma_v2()) pass++; else fail++;
    printf("\n");

    printf("========================================\n");
    printf("Results: %d PASS, %d FAIL\n", pass, fail);
    printf("========================================\n");
    return fail > 0 ? 1 : 0;
}
