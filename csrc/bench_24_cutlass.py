"""
Benchmark CUTLASS 2:4 sparse GEMM vs cuBLAS dense on Blackwell.

Compares up to three kernels:
  - cuBLAS dense (torch.mm)
  - CUTLASS SM80 (2.x mma.sp.sync, forward-compatible to Blackwell)
  - CUTLASS SM120 (3.x SM100 tcgen05.mma.sp via UMMA, native Blackwell)

Usage:
    cd csrc
    python bench_24_cutlass.py
"""

import time
import torch
import sparse_gemm_cutlass as sp80

try:
    import sparse_gemm_sm120 as sp3x
    HAS_3X = True
    LABEL_3X = "SM120"
except ImportError:
    HAS_3X = False
    print("WARNING: sparse_gemm_sm120 not available — build with "
          "setup_sparse_gemm_sm120.py first. Skipping 3.x benchmarks.\n")


def make_24_weight(N, K, device="cuda", dtype=torch.float16):
    """Create [N, K] weight with 2:4 sparsity — fully vectorized on GPU."""
    W = torch.randn(N, K, dtype=dtype, device=device) * 0.02
    # Reshape to [N, K//4, 4], find 2 smallest per group, zero them
    W4 = W.reshape(N, K // 4, 4)
    _, idx = W4.abs().topk(2, dim=-1, largest=False)
    mask = torch.ones_like(W4, dtype=torch.bool)
    mask.scatter_(-1, idx, False)
    W4.mul_(mask.to(dtype))
    return W.reshape(N, K)


def time_fn(fn, warmup=200, repeat=1000):
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

    print("=" * 90)
    print("CUTLASS 2:4 Sparse GEMM vs cuBLAS Dense")
    print(f"GPU: {torch.cuda.get_device_name()}")
    if HAS_3X:
        print(f"Kernels: cuBLAS | SM80 (CUTLASS 2.x) | {LABEL_3X} (CUTLASS 3.x TCGEN05)")
    else:
        print("Kernels: cuBLAS | SM80 (CUTLASS 2.x)")
    print("=" * 90)

    proj_configs = [
        ("qkv_proj (8B)",  4096, 6144),
        ("o_proj (8B)",    4096, 4096),
        ("up_proj (8B)",   4096, 14336),
        ("gate_proj (8B)", 4096, 14336),
        ("down_proj (8B)", 14336, 4096),
        ("qkv_proj (70B)", 8192, 10240),
        ("up_proj (70B)",  8192, 28672),
        ("down_proj (70B)", 28672, 8192),
    ]

    M_values = [1, 8, 32, 64, 128, 256, 512]

    for pname, K, N in proj_configs:
        print(f"\n  {pname} [K={K}, N={N}]")
        if HAS_3X:
            print(f"  {'M':>5s} {'cuBLAS':>9s} {'SM80':>9s} {'sp80':>7s}"
                  f" {LABEL_3X:>9s} {'sp3x':>7s} {'3x/80':>7s}")
            print(f"  {'─' * 62}")
        else:
            print(f"  {'M':>5s} {'cuBLAS':>9s} {'SM80':>9s} {'speedup':>8s}")
            print(f"  {'─' * 35}")

        W_dense = torch.randn(N, K, dtype=dtype, device=device) * 0.02
        W_24 = make_24_weight(N, K, device, dtype)
        c80, m80 = sp80.compress_24(W_24)
        if HAS_3X:
            c3x, m3x = sp3x.compress_24(W_24)

        for M in M_values:
            X = torch.randn(M, K, dtype=dtype, device=device)
            t_dense = time_fn(lambda: torch.mm(X, W_dense.t()))
            t_sm80 = time_fn(lambda: sp80.sparse_gemm_24(X, c80, m80, N))
            sp80_x = t_dense / t_sm80 if t_sm80 > 0 else 0

            if HAS_3X:
                t_3x = time_fn(lambda: sp3x.sparse_gemm_24(X, c3x, m3x, N))
                sp3x_x = t_dense / t_3x if t_3x > 0 else 0
                vs = t_sm80 / t_3x if t_3x > 0 else 0
                print(f"  {M:>5d} {t_dense:>8.3f}ms {t_sm80:>8.3f}ms {sp80_x:>6.2f}x"
                      f" {t_3x:>8.3f}ms {sp3x_x:>6.2f}x {vs:>6.2f}x")
            else:
                print(f"  {M:>5d} {t_dense:>8.3f}ms {t_sm80:>8.3f}ms {sp80_x:>7.2f}x")

    # Full layer benchmark helper
    def bench_full_layer(label, layer_projs, M_values):
        for M in M_values:
            print(f"\n{'=' * 90}")
            print(f"Full Transformer Layer ({label}, M={M})")
            print("=" * 90)

            total_dense = 0.0
            total_sm80 = 0.0
            total_3x = 0.0

            for pname, K, N in layer_projs:
                W_24 = make_24_weight(N, K, device, dtype)
                W_dense = torch.randn(N, K, dtype=dtype, device=device) * 0.02
                c80, m80 = sp80.compress_24(W_24)
                X = torch.randn(M, K, dtype=dtype, device=device)

                t_d = time_fn(lambda: torch.mm(X, W_dense.t()))
                t_80 = time_fn(lambda: sp80.sparse_gemm_24(X, c80, m80, N))

                total_dense += t_d
                total_sm80 += t_80

                if HAS_3X:
                    c3x, m3x = sp3x.compress_24(W_24)
                    t_3 = time_fn(lambda: sp3x.sparse_gemm_24(X, c3x, m3x, N))
                    total_3x += t_3
                    print(f"  {pname:>10s}: cuBLAS={t_d:.3f}ms  SM80={t_80:.3f}ms  "
                          f"{LABEL_3X}={t_3:.3f}ms  sp3x={t_d / t_3:.2f}x")
                else:
                    print(f"  {pname:>10s}: cuBLAS={t_d:.3f}ms  SM80={t_80:.3f}ms  "
                          f"speedup={t_d / t_80:.2f}x")

            print(f"  {'─' * 75}")
            if HAS_3X:
                print(f"  {'TOTAL':>10s}: cuBLAS={total_dense:.3f}ms  "
                      f"SM80={total_sm80:.3f}ms ({total_dense / total_sm80:.2f}x)  "
                      f"{LABEL_3X}={total_3x:.3f}ms ({total_dense / total_3x:.2f}x)")
            else:
                print(f"  {'TOTAL':>10s}: cuBLAS={total_dense:.3f}ms  "
                      f"SM80={total_sm80:.3f}ms  "
                      f"speedup={total_dense / total_sm80:.2f}x")

    # Full layer 8B
    layer_projs_8b = [
        ("q_proj",    4096, 4096),
        ("k_proj",    4096, 1024),
        ("v_proj",    4096, 1024),
        ("o_proj",    4096, 4096),
        ("up_proj",   4096, 14336),
        ("gate_proj", 4096, 14336),
        ("down_proj", 14336, 4096),
    ]
    bench_full_layer("Llama 8B", layer_projs_8b, [128, 512, 2048])

    # Full layer 70B
    layer_projs_70b = [
        ("q_proj",    8192, 8192),
        ("k_proj",    8192, 1024),
        ("v_proj",    8192, 1024),
        ("o_proj",    8192, 8192),
        ("up_proj",   8192, 28672),
        ("gate_proj", 8192, 28672),
        ("down_proj", 28672, 8192),
    ]
    bench_full_layer("Llama 70B", layer_projs_70b, [128, 512, 2048])


if __name__ == "__main__":
    main()
