"""
Benchmark 2:4 structured sparsity (cuSPARSELt/CUTLASS) vs dense cuBLAS.

Uses PyTorch's semi-structured sparsity API (torch >= 2.1).
Weight layout: [N, K] (nn.Linear convention), forward: Y = X @ W.T

Usage:
    cd scripts
    python bench_24_sparsity.py
"""

import time
import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# Use CUTLASS backend (faster on Hopper+)
SparseSemiStructuredTensor._FORCE_CUTLASS = True


def make_24_sparse_weight(N, K, device="cuda", dtype=torch.float16):
    """Create a [N, K] weight matrix with 2:4 structured sparsity.

    Every group of 4 consecutive elements along K (columns) has exactly 2 zeros.
    This matches the hardware 2:4 sparsity pattern for Sparse Tensor Core.
    """
    W = torch.randn(N, K, dtype=dtype, device=device) * 0.02
    # Apply 2:4 pattern along K dimension (columns)
    for col_start in range(0, K, 4):
        col_end = min(col_start + 4, K)
        width = col_end - col_start
        if width < 4:
            break
        mask = torch.ones(N, 4, dtype=torch.bool, device=device)
        # Vectorized: for each row, pick 2 random positions to zero
        indices = torch.stack([torch.randperm(4, device=device)[:2] for _ in range(N)])
        for i in range(N):
            mask[i, indices[i]] = False
        W[:, col_start:col_end] *= mask.to(dtype)
    return W


def time_fn(fn, warmup=100, repeat=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def main():
    device = "cuda"
    dtype = torch.float16

    print("=" * 80)
    print("2:4 Structured Sparsity (cuSPARSELt/CUTLASS) vs Dense cuBLAS")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 80)

    # LLM projection dims: (label, K=in_features, N=out_features)
    # Weight is [N, K], forward: Y[M,N] = X[M,K] @ W[N,K].T
    proj_configs = [
        ("qkv_proj (8B)",   4096, 6144),
        ("o_proj (8B)",     4096, 4096),
        ("up_proj (8B)",    4096, 14336),
        ("gate_proj (8B)",  4096, 14336),
        ("down_proj (8B)",  14336, 4096),
        ("qkv_proj (70B)",  8192, 10240),
        ("up_proj (70B)",   8192, 28672),
        ("down_proj (70B)", 28672, 8192),
    ]

    M_values = [1, 8, 32, 64, 128, 256, 512]

    for pname, K, N in proj_configs:
        print(f"\n  {pname} [K={K}, N={N}]  (W is [{N}×{K}])")
        print(f"  {'M':>5s} {'cuBLAS':>9s} {'2:4 sp':>9s} {'speedup':>8s}")
        print(f"  {'─' * 35}")

        W_dense = torch.randn(N, K, dtype=dtype, device=device) * 0.02
        W_24 = make_24_sparse_weight(N, K, device=device, dtype=dtype)

        try:
            W_sparse = to_sparse_semi_structured(W_24)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        for M in M_values:
            X = torch.randn(M, K, dtype=dtype, device=device)

            # Y = X @ W.T → [M, K] @ [K, N] = [M, N]
            t_dense = time_fn(lambda: torch.mm(X, W_dense.t()))
            t_sparse = time_fn(lambda: torch.mm(X, W_sparse.t()))

            speedup = t_dense / t_sparse if t_sparse > 0 else 0
            print(f"  {M:>5d} {t_dense:>8.3f}ms {t_sparse:>8.3f}ms {speedup:>7.2f}x")

    # Full layer comparison
    print(f"\n{'=' * 80}")
    print("Full Transformer Layer (7 projections, Llama 8B, M=128)")
    print("=" * 80)

    M = 128
    # (name, K=in_features, N=out_features)
    layer_projs = [
        ("q_proj",    4096, 4096),
        ("k_proj",    4096, 1024),
        ("v_proj",    4096, 1024),
        ("o_proj",    4096, 4096),
        ("up_proj",   4096, 14336),
        ("gate_proj", 4096, 14336),
        ("down_proj", 14336, 4096),
    ]

    total_dense = 0.0
    total_sparse = 0.0

    for pname, K, N in layer_projs:
        W_dense = torch.randn(N, K, dtype=dtype, device=device) * 0.02
        W_24 = make_24_sparse_weight(N, K, device=device, dtype=dtype)
        W_sparse = to_sparse_semi_structured(W_24)
        X = torch.randn(M, K, dtype=dtype, device=device)

        t_d = time_fn(lambda: torch.mm(X, W_dense.t()))
        t_s = time_fn(lambda: torch.mm(X, W_sparse.t()))

        total_dense += t_d
        total_sparse += t_s

        print(f"  {pname:>10s}: dense={t_d:.3f}ms  2:4={t_s:.3f}ms  speedup={t_d / t_s:.2f}x")

    print(f"  {'─' * 55}")
    print(f"  {'TOTAL':>10s}: dense={total_dense:.3f}ms  2:4={total_sparse:.3f}ms  "
          f"speedup={total_dense / total_sparse:.2f}x")


if __name__ == "__main__":
    main()
