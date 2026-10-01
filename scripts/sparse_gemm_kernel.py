"""
Block Sparse GEMM Triton kernel — prototype for sparse projection layers.

Replaces dense nn.Linear (X @ W) with block-sparse W, skipping zero blocks.
Covers all GEMM in a transformer layer:
  - Attention: QKV projection, O projection
  - FFN: up projection, gate projection, down projection

Weight W [K, N] is stored in BSC (Block Sparse Column) format:
  - values:  [nnz_blocks, BLOCK_K, BLOCK_N] — non-zero block data
  - col_ptr: [N // BLOCK_N + 1]             — column-block pointers
  - row_idx: [nnz_blocks]                   — k-block index per non-zero block

Usage:
    cd scripts
    python sparse_gemm_kernel.py validate          # correctness check
    python sparse_gemm_kernel.py bench              # benchmark vs cuBLAS
    python sparse_gemm_kernel.py bench --profile    # with roofline info
"""

import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# BLOCK SPARSE FORMAT UTILITIES
# ============================================================================

def dense_to_bsc(W, BLOCK_K, BLOCK_N, sparsity_threshold=1e-6):
    """Convert dense weight [K, N] to Block Sparse Column format.

    A block is considered zero if all elements have abs < sparsity_threshold.

    Returns:
        values:  [nnz, BLOCK_K, BLOCK_N] fp16
        col_ptr: [num_col_blocks + 1] int32
        row_idx: [nnz] int32
        nnz:     number of non-zero blocks
    """
    K, N = W.shape
    assert K % BLOCK_K == 0 and N % BLOCK_N == 0, \
        f"W shape ({K}, {N}) must be divisible by block size ({BLOCK_K}, {BLOCK_N})"

    num_k_blocks = K // BLOCK_K
    num_n_blocks = N // BLOCK_N

    blocks = []
    row_indices = []
    col_pointers = [0]

    for n in range(num_n_blocks):
        n_start = n * BLOCK_N
        n_end = n_start + BLOCK_N
        for k in range(num_k_blocks):
            k_start = k * BLOCK_K
            k_end = k_start + BLOCK_K
            block = W[k_start:k_end, n_start:n_end]
            if block.abs().max() > sparsity_threshold:
                blocks.append(block)
                row_indices.append(k)
        col_pointers.append(len(blocks))

    if len(blocks) == 0:
        values = torch.empty(0, BLOCK_K, BLOCK_N, dtype=W.dtype, device=W.device)
    else:
        values = torch.stack(blocks).contiguous()

    col_ptr = torch.tensor(col_pointers, dtype=torch.int32, device=W.device)
    row_idx = torch.tensor(row_indices, dtype=torch.int32, device=W.device)

    return values, col_ptr, row_idx, len(blocks)


def make_block_sparse_weight(K, N, BLOCK_K, BLOCK_N, sparsity, device="cuda",
                             dtype=torch.float16):
    """Create a random weight matrix with block-level sparsity.

    Args:
        sparsity: fraction of blocks that are ZERO (0.0 = dense, 0.5 = 50% zero).

    Returns:
        W_dense: [K, N] — dense version (for reference)
        values, col_ptr, row_idx, nnz — BSC format
    """
    W = torch.randn(K, N, dtype=dtype, device=device) * 0.02

    num_k_blocks = K // BLOCK_K
    num_n_blocks = N // BLOCK_N
    total_blocks = num_k_blocks * num_n_blocks
    num_zero = int(total_blocks * sparsity)

    # Random block mask
    flat_idx = torch.randperm(total_blocks)[:num_zero]
    for idx in flat_idx:
        kb = idx // num_n_blocks
        nb = idx % num_n_blocks
        W[kb * BLOCK_K:(kb + 1) * BLOCK_K,
          nb * BLOCK_N:(nb + 1) * BLOCK_N] = 0.0

    values, col_ptr, row_idx, nnz = dense_to_bsc(W, BLOCK_K, BLOCK_N)
    return W, values, col_ptr, row_idx, nnz


# ============================================================================
# TRITON BLOCK SPARSE GEMM KERNEL
# ============================================================================

@triton.jit
def _sparse_gemm_kernel(
    # Pointers
    X_ptr, values_ptr, col_ptr_ptr, row_idx_ptr, Y_ptr,
    # Matrix dims
    M, K, N,
    # Block dims (constexpr for compilation)
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    # Strides
    stride_xm, stride_xk,
    stride_ym, stride_yn,
):
    """Block sparse GEMM: Y = X @ W_sparse.

    Grid: (M // BLOCK_M, N // BLOCK_N)
    Each program computes one [BLOCK_M, BLOCK_N] output tile.
    Only iterates over non-zero K-blocks for this N-column (via col_ptr/row_idx).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Column pointer range for this N-block
    col_start = tl.load(col_ptr_ptr + pid_n)
    col_end = tl.load(col_ptr_ptr + pid_n + 1)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # M/N-offsets (fixed for this program)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Iterate over non-zero K-blocks
    for idx in range(col_start, col_end):
        # Which K-block
        k_block = tl.load(row_idx_ptr + idx)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                     other=0.0)

        # Load W block from values: [BLOCK_K, BLOCK_N]
        # values layout: [nnz, BLOCK_K, BLOCK_N] contiguous
        w_base = idx * BLOCK_K * BLOCK_N
        bk_offs = tl.arange(0, BLOCK_K)
        bn_offs = tl.arange(0, BLOCK_N)
        w_ptrs = values_ptr + w_base + bk_offs[:, None] * BLOCK_N + bn_offs[None, :]
        w = tl.load(w_ptrs)

        # Accumulate
        acc += tl.dot(x, w)

    # Store output tile
    y_ptrs = Y_ptr + m_offs[:, None] * stride_ym + n_offs[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


def sparse_gemm(X, values, col_ptr, row_idx, N,
                BLOCK_M=64, BLOCK_K=64, BLOCK_N=64):
    """Y = X @ W_sparse using Triton block sparse kernel.

    Args:
        X: [M, K] input activations
        values: [nnz, BLOCK_K, BLOCK_N] non-zero weight blocks
        col_ptr: [N // BLOCK_N + 1] column pointers
        row_idx: [nnz] row-block indices
        N: output dimension

    Returns:
        Y: [M, N]
    """
    M, K = X.shape
    Y = torch.empty(M, N, dtype=X.dtype, device=X.device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _sparse_gemm_kernel[grid](
        X, values, col_ptr, row_idx, Y,
        M, K, N,
        BLOCK_M, BLOCK_K, BLOCK_N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
    )
    return Y


# ============================================================================
# TRANSFORMER LAYER SIMULATION
# ============================================================================

def sparse_transformer_layer(x, weights_sparse, weights_dense=None):
    """Run one transformer layer with block sparse projections.

    Args:
        x: [B, seq_len, hidden] input
        weights_sparse: dict with BSC format for each projection:
            {name: (values, col_ptr, row_idx, N, BLOCK_K, BLOCK_N)}
            names: q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj
        weights_dense: optional dict {name: W_dense} for reference comparison

    Returns:
        output: [B, seq_len, hidden]
    """
    B, seq_len, hidden = x.shape
    x_2d = x.reshape(-1, hidden)  # [M, hidden]
    M = x_2d.shape[0]

    def proj(name, inp):
        vals, cp, ri, N, BK, BN = weights_sparse[name]
        return sparse_gemm(inp, vals, cp, ri, N, BLOCK_M=64, BLOCK_K=BK, BLOCK_N=BN)

    # === Attention projections ===
    q = proj("q_proj", x_2d)   # [M, hidden]
    k = proj("k_proj", x_2d)   # [M, kv_dim]
    v = proj("v_proj", x_2d)   # [M, kv_dim]

    # Simplified attention (skip actual attention computation — focus on GEMM)
    # In practice, attention QK^T and score*V happen here (not projection)
    attn_out = q  # placeholder

    o = proj("o_proj", attn_out)  # [M, hidden]

    # Residual
    h = x_2d + o

    # === FFN projections (SwiGLU) ===
    up = proj("up_proj", h)     # [M, intermediate]
    gate = proj("gate_proj", h)  # [M, intermediate]
    ffn_hidden = F.silu(gate) * up
    down = proj("down_proj", ffn_hidden)  # [M, hidden]

    # Residual
    output = h + down
    return output.reshape(B, seq_len, hidden)


# ============================================================================
# VALIDATION
# ============================================================================

def validate():
    """Validate sparse GEMM against dense torch.matmul."""
    print("=" * 60)
    print("Block Sparse GEMM Validation")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float16

    test_configs = [
        # (M, K, N, BLOCK_K, BLOCK_N, sparsity)
        (128, 4096, 4096, 64, 64, 0.5),    # O proj (8B)
        (128, 4096, 14336, 64, 64, 0.5),   # up/gate proj (8B)
        (128, 14336, 4096, 64, 64, 0.5),   # down proj (8B)
        (128, 8192, 28672, 64, 64, 0.5),   # up/gate proj (70B)
        (32, 4096, 4096, 64, 64, 0.75),    # high sparsity
        (32, 4096, 4096, 128, 128, 0.5),   # larger blocks
    ]

    all_pass = True
    for M, K, N, BK, BN, sparsity in test_configs:
        W_dense, values, col_ptr, row_idx, nnz = make_block_sparse_weight(
            K, N, BK, BN, sparsity, device=device, dtype=dtype
        )
        X = torch.randn(M, K, dtype=dtype, device=device) * 0.1

        # Reference
        Y_ref = X @ W_dense

        # Sparse
        Y_sparse = sparse_gemm(X, values, col_ptr, row_idx, N,
                                BLOCK_M=64, BLOCK_K=BK, BLOCK_N=BN)

        # Compare (fp16 tolerance)
        max_err = (Y_ref - Y_sparse).abs().max().item()
        rel_err = max_err / (Y_ref.abs().max().item() + 1e-8)

        total_blocks = (K // BK) * (N // BN)
        actual_sparsity = 1.0 - nnz / total_blocks

        status = "PASS" if rel_err < 0.02 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  [{status}] M={M:5d} K={K:5d} N={N:5d} "
              f"block={BK}x{BN} sparsity={actual_sparsity:.0%} "
              f"nnz={nnz}/{total_blocks} "
              f"max_err={max_err:.6f} rel_err={rel_err:.6f}")

    # Full layer validation
    print(f"\n{'=' * 60}")
    print("Full Layer Validation (Llama 8B dimensions)")
    print("=" * 60)

    hidden, intermediate, kv_dim = 4096, 14336, 1024
    BK, BN = 64, 64
    sparsity = 0.5
    M_seq = 64  # batch * seq_len

    proj_configs = {
        "q_proj":    (hidden, hidden),
        "k_proj":    (hidden, kv_dim),
        "v_proj":    (hidden, kv_dim),
        "o_proj":    (hidden, hidden),
        "up_proj":   (hidden, intermediate),
        "gate_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }

    x = torch.randn(1, M_seq, hidden, dtype=dtype, device=device) * 0.1
    x_2d = x.reshape(-1, hidden)

    for name, (K_dim, N_dim) in proj_configs.items():
        W_dense, values, col_ptr, row_idx, nnz = make_block_sparse_weight(
            K_dim, N_dim, BK, BN, sparsity, device=device, dtype=dtype
        )

        if name == "down_proj":
            inp = torch.randn(M_seq, K_dim, dtype=dtype, device=device) * 0.1
        else:
            inp = x_2d

        Y_ref = inp @ W_dense
        Y_sparse = sparse_gemm(inp, values, col_ptr, row_idx, N_dim,
                                BLOCK_M=64, BLOCK_K=BK, BLOCK_N=BN)

        max_err = (Y_ref - Y_sparse).abs().max().item()
        rel_err = max_err / (Y_ref.abs().max().item() + 1e-8)
        status = "PASS" if rel_err < 0.02 else "FAIL"
        if status == "FAIL":
            all_pass = False

        total_blocks = (K_dim // BK) * (N_dim // BN)
        print(f"  [{status}] {name:10s}: ({K_dim:5d}, {N_dim:5d}) "
              f"nnz={nnz}/{total_blocks} rel_err={rel_err:.6f}")

    print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark():
    """Benchmark sparse GEMM vs dense cuBLAS at various sparsity levels."""
    print("=" * 60)
    print("Block Sparse GEMM Benchmark (Triton vs cuBLAS)")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float16

    # Llama 8B projections
    proj_configs = [
        ("qkv_proj",  4096, 6144),    # Q+K+V fused (4096 + 1024 + 1024)
        ("o_proj",     4096, 4096),
        ("up_proj",    4096, 14336),
        ("gate_proj",  4096, 14336),
        ("down_proj",  14336, 4096),
    ]

    sparsity_levels = [0.0, 0.25, 0.50, 0.75, 0.875]
    M = 128  # batch * seq_len (typical for spec decode verification)
    BK, BN = 64, 64
    warmup, repeat = 50, 200

    for name, K, N in proj_configs:
        print(f"\n  {name} [{K} x {N}], M={M}:")
        print(f"  {'sparsity':>10s} {'cuBLAS_ms':>10s} {'Triton_ms':>10s} "
              f"{'speedup':>8s} {'nnz_blocks':>10s}")
        print(f"  {'-'*52}")

        for sparsity in sparsity_levels:
            W_dense, values, col_ptr, row_idx, nnz = make_block_sparse_weight(
                K, N, BK, BN, sparsity, device=device, dtype=dtype
            )
            X = torch.randn(M, K, dtype=dtype, device=device) * 0.1
            total_blocks = (K // BK) * (N // BN)

            # cuBLAS (dense)
            for _ in range(warmup):
                _ = X @ W_dense
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                _ = X @ W_dense
            torch.cuda.synchronize()
            cublas_ms = (time.perf_counter() - t0) / repeat * 1000

            # Triton sparse
            for _ in range(warmup):
                _ = sparse_gemm(X, values, col_ptr, row_idx, N,
                                BLOCK_M=64, BLOCK_K=BK, BLOCK_N=BN)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                _ = sparse_gemm(X, values, col_ptr, row_idx, N,
                                BLOCK_M=64, BLOCK_K=BK, BLOCK_N=BN)
            torch.cuda.synchronize()
            triton_ms = (time.perf_counter() - t0) / repeat * 1000

            speedup = cublas_ms / triton_ms if triton_ms > 0 else float('inf')
            print(f"  {sparsity:>9.0%} {cublas_ms:>10.3f} {triton_ms:>10.3f} "
                  f"{speedup:>7.2f}x {nnz:>5d}/{total_blocks}")

    # Full layer benchmark
    print(f"\n{'=' * 60}")
    print("Full Transformer Layer (all 7 projections, Llama 8B)")
    print("=" * 60)

    hidden, intermediate, kv_dim = 4096, 14336, 1024
    all_projs = {
        "q_proj":    (hidden, hidden),
        "k_proj":    (hidden, kv_dim),
        "v_proj":    (hidden, kv_dim),
        "o_proj":    (hidden, hidden),
        "up_proj":   (hidden, intermediate),
        "gate_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }

    for sparsity in [0.0, 0.50, 0.75]:
        # Build all projections
        cublas_total = 0.0
        triton_total = 0.0

        for pname, (K_dim, N_dim) in all_projs.items():
            W_dense, values, col_ptr, row_idx, nnz = make_block_sparse_weight(
                K_dim, N_dim, BK, BN, sparsity, device=device, dtype=dtype
            )

            if pname == "down_proj":
                inp = torch.randn(M, K_dim, dtype=dtype, device=device) * 0.1
            else:
                inp = torch.randn(M, hidden, dtype=dtype, device=device) * 0.1

            # cuBLAS
            for _ in range(warmup):
                _ = inp @ W_dense
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                _ = inp @ W_dense
            torch.cuda.synchronize()
            cublas_ms = (time.perf_counter() - t0) / repeat * 1000
            cublas_total += cublas_ms

            # Triton
            for _ in range(warmup):
                _ = sparse_gemm(inp, values, col_ptr, row_idx, N_dim,
                                BLOCK_M=64, BLOCK_K=BK, BLOCK_N=BN)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                _ = sparse_gemm(inp, values, col_ptr, row_idx, N_dim,
                                BLOCK_M=64, BLOCK_K=BK, BLOCK_N=BN)
            torch.cuda.synchronize()
            triton_ms = (time.perf_counter() - t0) / repeat * 1000
            triton_total += triton_ms

        speedup = cublas_total / triton_total if triton_total > 0 else float('inf')
        print(f"  sparsity={sparsity:.0%}: cuBLAS={cublas_total:.3f}ms "
              f"Triton={triton_total:.3f}ms speedup={speedup:.2f}x")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"

    if cmd == "validate":
        validate()
    elif cmd == "bench":
        benchmark()
    else:
        print(f"Usage: python {sys.argv[0]} [validate|bench]")
        sys.exit(1)
