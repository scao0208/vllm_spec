"""
Hierarchical Sparse GEMM — three Triton kernel variants inspired by SpInfer's
BitmapTile / TCTile / GroupTile hierarchy.

Variants:
  V0: BSC block-sparse, BLOCK=64 (baseline, coarse skip)
  V1: BSC block-sparse, BLOCK=16 (Idea 1: TCTile-level fine skip)
  V2: Hierarchical two-level: GroupTile 64×64 skip + K-sub-step bitmap
      + density-adaptive (DENSE GroupTiles skip bitmap overhead)
  V3: V2 with BLOCK_M=128 (Idea 3: larger output tile → Hopper WGMMA)

Usage:
    cd scripts
    python sparse_gemm_hierarchical.py validate   # correctness
    python sparse_gemm_hierarchical.py bench       # full profile
"""

import sys
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import triton
import triton.language as tl


# ============================================================================
# COMMON UTILITIES
# ============================================================================

def make_sparse_weight(K, N, sparsity, block_size, device="cuda", dtype=torch.float16):
    """Create weight with block-level sparsity at given block_size granularity."""
    W = torch.randn(K, N, dtype=dtype, device=device) * 0.02
    num_kb = K // block_size
    num_nb = N // block_size
    total = num_kb * num_nb
    num_zero = int(total * sparsity)
    flat_idx = torch.randperm(total, device="cpu")[:num_zero]
    for idx in flat_idx:
        kb = idx // num_nb
        nb = idx % num_nb
        W[kb * block_size:(kb + 1) * block_size,
          nb * block_size:(nb + 1) * block_size] = 0.0
    return W


# ============================================================================
# V0: BSC BLOCK=64 (coarse skip)
# ============================================================================

def compress_bsc(W, BK, BN, threshold=1e-6):
    """BSC format: col_ptr, row_idx, values (dense sub-blocks)."""
    K, N = W.shape
    num_kb, num_nb = K // BK, N // BN
    blocks, rows, col_ptrs = [], [], [0]
    for n in range(num_nb):
        for k in range(num_kb):
            blk = W[k * BK:(k + 1) * BK, n * BN:(n + 1) * BN]
            if blk.abs().max() > threshold:
                blocks.append(blk)
                rows.append(k)
        col_ptrs.append(len(blocks))
    if blocks:
        values = torch.stack(blocks).contiguous()
    else:
        values = torch.empty(0, BK, BN, dtype=W.dtype, device=W.device)
    col_ptr = torch.tensor(col_ptrs, dtype=torch.int32, device=W.device)
    row_idx = torch.tensor(rows, dtype=torch.int32, device=W.device)
    return values, col_ptr, row_idx


@triton.jit
def _bsc_gemm(
    X_ptr, vals_ptr, col_ptr_ptr, row_idx_ptr, Y_ptr,
    M, K, N,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    stride_xm, stride_xk, stride_ym, stride_yn,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    col_start = tl.load(col_ptr_ptr + pid_n)
    col_end = tl.load(col_ptr_ptr + pid_n + 1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    for idx in range(col_start, col_end):
        kb = tl.load(row_idx_ptr + idx)
        k_offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                     mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        w_base = idx * BLOCK_K * BLOCK_N
        w = tl.load(vals_ptr + w_base + tl.arange(0, BLOCK_K)[:, None] * BLOCK_N
                     + tl.arange(0, BLOCK_N)[None, :])
        acc += tl.dot(x, w)
    y_ptrs = Y_ptr + m_offs[:, None] * stride_ym + n_offs[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


def bsc_gemm(X, values, col_ptr, row_idx, N, BM, BK, BN):
    M, K = X.shape
    Y = torch.empty(M, N, dtype=X.dtype, device=X.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _bsc_gemm[grid](X, values, col_ptr, row_idx, Y,
                     M, K, N, BM, BK, BN,
                     X.stride(0), X.stride(1), Y.stride(0), Y.stride(1))
    return Y


# ============================================================================
# V2: HIERARCHICAL — GroupTile 64×64 + K-sub-step bitmap + density-adaptive
# ============================================================================

# GroupTile format:
#   gtile_col_ptr:  [N//64 + 1]    — CSC pointers for non-zero GroupTiles
#   gtile_row_idx:  [nnz_gtiles]   — K-group index
#   gtile_type:     [nnz_gtiles]   — 0=DENSE, 1=SPARSE
#   ksub_bitmap:    [nnz_gtiles]   — 4-bit bitmap: which K-sub-steps (of 16) are non-zero
#   ksub_offset:    [nnz_gtiles+1] — cumulative count of non-zero K-sub-steps
#   ksub_values:    [total_ksubs, 16, 64] — non-zero K-sub-step tiles (16×64)
#
# DENSE GroupTiles also store their K-sub-steps in ksub_values (all 4).
# The only difference: DENSE skips bitmap check (always processes 4 sub-steps).

GTILE_K = 64
GTILE_N = 64
KSUB_K = 16  # K-sub-step size

def compress_hierarchical(W, dense_threshold=0.75, zero_threshold=1e-6):
    """Compress W into hierarchical format.

    Args:
        dense_threshold: if >this fraction of K-sub-steps are non-zero, mark DENSE
    """
    K, N = W.shape
    assert K % GTILE_K == 0 and N % GTILE_N == 0
    num_kg = K // GTILE_K
    num_ng = N // GTILE_N
    num_ksubs_per_gtile = GTILE_K // KSUB_K  # 4

    col_ptrs = [0]
    row_indices = []
    types = []
    bitmaps = []
    ksub_tiles = []
    ksub_offsets = [0]

    for ng in range(num_ng):
        n0 = ng * GTILE_N
        for kg in range(num_kg):
            k0 = kg * GTILE_K
            gtile = W[k0:k0 + GTILE_K, n0:n0 + GTILE_N]

            if gtile.abs().max() <= zero_threshold:
                continue  # skip zero GroupTile

            row_indices.append(kg)

            # Build K-sub-step bitmap
            bitmap = 0
            ksub_list = []
            for s in range(num_ksubs_per_gtile):
                sk0 = s * KSUB_K
                sub = gtile[sk0:sk0 + KSUB_K, :]
                if sub.abs().max() > zero_threshold:
                    bitmap |= (1 << s)
                    ksub_list.append(sub)

            nnz_subs = len(ksub_list)
            density = nnz_subs / num_ksubs_per_gtile

            if density > dense_threshold:
                # DENSE: store all 4 sub-steps (fill zero ones too)
                types.append(0)
                bitmaps.append(0xF)  # all 4 bits set
                for s in range(num_ksubs_per_gtile):
                    sk0 = s * KSUB_K
                    ksub_tiles.append(gtile[sk0:sk0 + KSUB_K, :])
                ksub_offsets.append(ksub_offsets[-1] + num_ksubs_per_gtile)
            else:
                # SPARSE: store only non-zero sub-steps
                types.append(1)
                bitmaps.append(bitmap)
                ksub_tiles.extend(ksub_list)
                ksub_offsets.append(ksub_offsets[-1] + nnz_subs)

        col_ptrs.append(len(row_indices))

    device = W.device
    if ksub_tiles:
        ksub_values = torch.stack(ksub_tiles).contiguous()
    else:
        ksub_values = torch.empty(0, KSUB_K, GTILE_N, dtype=W.dtype, device=device)

    return (
        torch.tensor(col_ptrs, dtype=torch.int32, device=device),
        torch.tensor(row_indices, dtype=torch.int32, device=device),
        torch.tensor(types, dtype=torch.int32, device=device),
        torch.tensor(bitmaps, dtype=torch.int32, device=device),
        torch.tensor(ksub_offsets, dtype=torch.int32, device=device),
        ksub_values,
    )


@triton.jit
def _hier_gemm(
    X_ptr, ksub_vals_ptr, gtile_col_ptr_ptr, gtile_row_idx_ptr,
    gtile_type_ptr, ksub_bitmap_ptr, ksub_offset_ptr, Y_ptr,
    M, K, N,
    BLOCK_M: tl.constexpr,
    stride_xm, stride_xk, stride_ym, stride_yn,
):
    """Hierarchical sparse GEMM with density-adaptive dispatch.

    DENSE GroupTiles: iterate all 4 K-sub-steps (no bitmap check).
    SPARSE GroupTiles: check bitmap, skip zero K-sub-steps.
    """
    KSUB: tl.constexpr = 16
    GTILE_N_CST: tl.constexpr = 64
    NUM_KSUBS: tl.constexpr = 4

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    g_start = tl.load(gtile_col_ptr_ptr + pid_n)
    g_end = tl.load(gtile_col_ptr_ptr + pid_n + 1)

    acc = tl.zeros((BLOCK_M, GTILE_N_CST), dtype=tl.float32)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * GTILE_N_CST + tl.arange(0, GTILE_N_CST)

    for g_idx in range(g_start, g_end):
        kg = tl.load(gtile_row_idx_ptr + g_idx)
        gtype = tl.load(gtile_type_ptr + g_idx)
        ks_base = tl.load(ksub_offset_ptr + g_idx)

        if gtype == 0:
            # DENSE: process all 4 K-sub-steps, no bitmap check
            for s in range(NUM_KSUBS):
                k_start = kg * 64 + s * KSUB
                k_offs = k_start + tl.arange(0, KSUB)
                x = tl.load(X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                             mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0.0)
                v_idx = ks_base + s
                w = tl.load(ksub_vals_ptr + v_idx * KSUB * GTILE_N_CST
                             + tl.arange(0, KSUB)[:, None] * GTILE_N_CST
                             + tl.arange(0, GTILE_N_CST)[None, :])
                acc += tl.dot(x, w)
        else:
            # SPARSE: check bitmap per K-sub-step
            bitmap = tl.load(ksub_bitmap_ptr + g_idx)
            ks_local = 0  # track position within this GroupTile's sparse ksubs
            for s in range(NUM_KSUBS):
                bit = (bitmap >> s) & 1
                if bit == 1:
                    k_start = kg * 64 + s * KSUB
                    k_offs = k_start + tl.arange(0, KSUB)
                    x = tl.load(X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                                 mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0.0)
                    v_idx = ks_base + ks_local
                    w = tl.load(ksub_vals_ptr + v_idx * KSUB * GTILE_N_CST
                                 + tl.arange(0, KSUB)[:, None] * GTILE_N_CST
                                 + tl.arange(0, GTILE_N_CST)[None, :])
                    acc += tl.dot(x, w)
                    ks_local += 1

    y_ptrs = Y_ptr + m_offs[:, None] * stride_ym + n_offs[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


def hier_gemm(X, data, N, BLOCK_M=64):
    col_ptr, row_idx, types, bitmaps, ksub_offsets, ksub_values = data
    M, K = X.shape
    Y = torch.empty(M, N, dtype=X.dtype, device=X.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, GTILE_N))
    _hier_gemm[grid](
        X, ksub_values, col_ptr, row_idx, types, bitmaps, ksub_offsets, Y,
        M, K, N, BLOCK_M,
        X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
    )
    return Y


# ============================================================================
# V3: HIERARCHICAL with BLOCK_M=128 (Hopper WGMMA)
# ============================================================================

@triton.jit
def _hier_gemm_128(
    X_ptr, ksub_vals_ptr, gtile_col_ptr_ptr, gtile_row_idx_ptr,
    gtile_type_ptr, ksub_bitmap_ptr, ksub_offset_ptr, Y_ptr,
    M, K, N,
    stride_xm, stride_xk, stride_ym, stride_yn,
):
    """Same as _hier_gemm but BLOCK_M=128 for Hopper WGMMA."""
    BLOCK_M: tl.constexpr = 128
    KSUB: tl.constexpr = 16
    GTILE_N_CST: tl.constexpr = 64
    NUM_KSUBS: tl.constexpr = 4

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    g_start = tl.load(gtile_col_ptr_ptr + pid_n)
    g_end = tl.load(gtile_col_ptr_ptr + pid_n + 1)

    acc = tl.zeros((BLOCK_M, GTILE_N_CST), dtype=tl.float32)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * GTILE_N_CST + tl.arange(0, GTILE_N_CST)

    for g_idx in range(g_start, g_end):
        kg = tl.load(gtile_row_idx_ptr + g_idx)
        gtype = tl.load(gtile_type_ptr + g_idx)
        ks_base = tl.load(ksub_offset_ptr + g_idx)

        if gtype == 0:
            for s in range(NUM_KSUBS):
                k_start = kg * 64 + s * KSUB
                k_offs = k_start + tl.arange(0, KSUB)
                x = tl.load(X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                             mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0.0)
                v_idx = ks_base + s
                w = tl.load(ksub_vals_ptr + v_idx * KSUB * GTILE_N_CST
                             + tl.arange(0, KSUB)[:, None] * GTILE_N_CST
                             + tl.arange(0, GTILE_N_CST)[None, :])
                acc += tl.dot(x, w)
        else:
            bitmap = tl.load(ksub_bitmap_ptr + g_idx)
            ks_local = 0
            for s in range(NUM_KSUBS):
                bit = (bitmap >> s) & 1
                if bit == 1:
                    k_start = kg * 64 + s * KSUB
                    k_offs = k_start + tl.arange(0, KSUB)
                    x = tl.load(X_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                                 mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0.0)
                    v_idx = ks_base + ks_local
                    w = tl.load(ksub_vals_ptr + v_idx * KSUB * GTILE_N_CST
                                 + tl.arange(0, KSUB)[:, None] * GTILE_N_CST
                                 + tl.arange(0, GTILE_N_CST)[None, :])
                    acc += tl.dot(x, w)
                    ks_local += 1

    y_ptrs = Y_ptr + m_offs[:, None] * stride_ym + n_offs[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


def hier_gemm_128(X, data, N):
    col_ptr, row_idx, types, bitmaps, ksub_offsets, ksub_values = data
    M, K = X.shape
    Y = torch.empty(M, N, dtype=X.dtype, device=X.device)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, GTILE_N))
    _hier_gemm_128[grid](
        X, ksub_values, col_ptr, row_idx, types, bitmaps, ksub_offsets, Y,
        M, K, N,
        X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
    )
    return Y


# ============================================================================
# VALIDATION
# ============================================================================

def validate():
    print("=" * 70)
    print("Hierarchical Sparse GEMM — Validation")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float16
    all_pass = True

    configs = [
        # (M, K, N, sparsity, block_prune_size, label)
        (128, 4096, 4096,  0.50, 16, "o_proj 50% @16"),
        (128, 4096, 14336, 0.50, 16, "up_proj 50% @16"),
        (128, 14336, 4096, 0.50, 16, "down_proj 50% @16"),
        (128, 4096, 4096,  0.75, 16, "o_proj 75% @16"),
        (128, 4096, 14336, 0.75, 16, "up_proj 75% @16"),
        (64,  4096, 4096,  0.50, 64, "o_proj 50% @64"),
        (64,  4096, 14336, 0.50, 64, "up_proj 50% @64"),
    ]

    for M, K, N, sparsity, bps, label in configs:
        W = make_sparse_weight(K, N, sparsity, bps, device=device, dtype=dtype)
        X = torch.randn(M, K, dtype=dtype, device=device) * 0.1
        Y_ref = X @ W

        results = {}

        # V0: BSC-64
        v0, cp0, ri0 = compress_bsc(W, 64, 64)
        Y0 = bsc_gemm(X, v0, cp0, ri0, N, 64, 64, 64)
        results["V0 BSC-64"] = Y0

        # V1: BSC-16
        v1, cp1, ri1 = compress_bsc(W, 16, 16)
        Y1 = bsc_gemm(X, v1, cp1, ri1, N, 64, 16, 16)
        results["V1 BSC-16"] = Y1

        # V2: Hierarchical
        h_data = compress_hierarchical(W, dense_threshold=0.75)
        Y2 = hier_gemm(X, h_data, N, BLOCK_M=64)
        results["V2 Hier-64"] = Y2

        # V3: Hierarchical BLOCK_M=128
        if M >= 128:
            Y3 = hier_gemm_128(X, h_data, N)
            results["V3 Hier-128"] = Y3

        for name, Y in results.items():
            max_err = (Y_ref - Y).abs().max().item()
            rel_err = max_err / (Y_ref.abs().max().item() + 1e-8)
            ok = rel_err < 0.02
            if not ok:
                all_pass = False
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:24s} {name:14s} "
                  f"rel_err={rel_err:.6f}")

    print(f"\n{'All passed!' if all_pass else 'SOME FAILED'}")
    return all_pass


# ============================================================================
# BENCHMARK
# ============================================================================

def _time_fn(fn, warmup=50, repeat=200):
    """Time a function with warmup and repeat."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def benchmark():
    print("=" * 90)
    print("Hierarchical Sparse GEMM — Benchmark (all times in ms)")
    print("=" * 90)

    device = "cuda"
    dtype = torch.float16

    proj_configs = [
        ("up_proj",   4096, 14336),
        ("down_proj", 14336, 4096),
        ("o_proj",    4096, 4096),
    ]
    sparsity_levels = [0.0, 0.25, 0.50, 0.75, 0.875]
    M = 128
    warmup, repeat = 100, 500

    for pname, K, N in proj_configs:
        print(f"\n  {pname} [{K}×{N}], M={M}")
        print(f"  {'sprs':>5s} {'cuBLAS':>8s} {'V0-64':>8s} {'V1-16':>8s} "
              f"{'V2-hier':>8s} {'V3-h128':>8s} │ "
              f"{'V0/cB':>6s} {'V1/cB':>6s} {'V2/cB':>6s} {'V3/cB':>6s}")
        print(f"  {'─' * 84}")

        for sparsity in sparsity_levels:
            # Create weight with 16×16 block sparsity (finest meaningful granularity)
            W = make_sparse_weight(K, N, sparsity, 16, device=device, dtype=dtype)
            X = torch.randn(M, K, dtype=dtype, device=device) * 0.1

            # cuBLAS
            t_cublas = _time_fn(lambda: X @ W, warmup, repeat)

            # V0: BSC-64
            v0, cp0, ri0 = compress_bsc(W, 64, 64)
            t_v0 = _time_fn(lambda: bsc_gemm(X, v0, cp0, ri0, N, 64, 64, 64), warmup, repeat)

            # V1: BSC-16
            v1, cp1, ri1 = compress_bsc(W, 16, 16)
            t_v1 = _time_fn(lambda: bsc_gemm(X, v1, cp1, ri1, N, 64, 16, 16), warmup, repeat)

            # V2: Hierarchical BM=64
            h_data = compress_hierarchical(W, dense_threshold=0.75)
            t_v2 = _time_fn(lambda: hier_gemm(X, h_data, N, BLOCK_M=64), warmup, repeat)

            # V3: Hierarchical BM=128
            t_v3 = _time_fn(lambda: hier_gemm_128(X, h_data, N), warmup, repeat)

            s0 = t_cublas / t_v0 if t_v0 > 0 else 0
            s1 = t_cublas / t_v1 if t_v1 > 0 else 0
            s2 = t_cublas / t_v2 if t_v2 > 0 else 0
            s3 = t_cublas / t_v3 if t_v3 > 0 else 0

            print(f"  {sparsity:>4.0%} {t_cublas:>8.3f} {t_v0:>8.3f} {t_v1:>8.3f} "
                  f"{t_v2:>8.3f} {t_v3:>8.3f} │ "
                  f"{s0:>5.2f}x {s1:>5.2f}x {s2:>5.2f}x {s3:>5.2f}x")

    # ── Sparsity pattern comparison ──
    print(f"\n{'=' * 90}")
    print("Sparsity Pattern Comparison (up_proj 4096×14336, 75% sparsity)")
    print("=" * 90)

    K, N = 4096, 14336

    patterns = {
        "block-16":  lambda: make_sparse_weight(K, N, 0.75, 16, device, dtype),
        "block-64":  lambda: make_sparse_weight(K, N, 0.75, 64, device, dtype),
        "block-128": lambda: make_sparse_weight(K, N, 0.75, 128, device, dtype),
    }

    print(f"  {'pattern':>12s} {'cuBLAS':>8s} {'V0-64':>8s} {'V1-16':>8s} "
          f"{'V2-hier':>8s} {'V3-h128':>8s} │ "
          f"{'V2/cB':>6s} {'V3/cB':>6s}")
    print(f"  {'─' * 78}")

    for pname, make_W in patterns.items():
        W = make_W()
        X = torch.randn(M, K, dtype=dtype, device=device) * 0.1

        t_cublas = _time_fn(lambda: X @ W, warmup, repeat)

        v0, cp0, ri0 = compress_bsc(W, 64, 64)
        t_v0 = _time_fn(lambda: bsc_gemm(X, v0, cp0, ri0, N, 64, 64, 64), warmup, repeat)

        v1, cp1, ri1 = compress_bsc(W, 16, 16)
        t_v1 = _time_fn(lambda: bsc_gemm(X, v1, cp1, ri1, N, 64, 16, 16), warmup, repeat)

        h_data = compress_hierarchical(W, dense_threshold=0.75)
        t_v2 = _time_fn(lambda: hier_gemm(X, h_data, N, BLOCK_M=64), warmup, repeat)
        t_v3 = _time_fn(lambda: hier_gemm_128(X, h_data, N), warmup, repeat)

        s2 = t_cublas / t_v2 if t_v2 > 0 else 0
        s3 = t_cublas / t_v3 if t_v3 > 0 else 0

        print(f"  {pname:>12s} {t_cublas:>8.3f} {t_v0:>8.3f} {t_v1:>8.3f} "
              f"{t_v2:>8.3f} {t_v3:>8.3f} │ "
              f"{s2:>5.2f}x {s3:>5.2f}x")

    # ── Metadata overhead ──
    print(f"\n{'=' * 90}")
    print("Metadata Overhead (up_proj 4096×14336, 75% block-16 sparsity)")
    print("=" * 90)

    W = make_sparse_weight(4096, 14336, 0.75, 16, device, dtype)

    v0, cp0, ri0 = compress_bsc(W, 64, 64)
    v1, cp1, ri1 = compress_bsc(W, 16, 16)
    h_data = compress_hierarchical(W, dense_threshold=0.75)
    h_col_ptr, h_row_idx, h_types, h_bitmaps, h_ksub_off, h_ksub_vals = h_data

    def mem_kb(*tensors):
        return sum(t.nelement() * t.element_size() for t in tensors) / 1024

    print(f"  Original dense:  {W.nelement() * 2 / 1024:.1f} KB")
    print(f"  V0 BSC-64:       values={mem_kb(v0):.1f} KB  meta={mem_kb(cp0, ri0):.1f} KB  "
          f"total={mem_kb(v0, cp0, ri0):.1f} KB ({mem_kb(v0, cp0, ri0) / (W.nelement() * 2 / 1024):.0%})")
    print(f"  V1 BSC-16:       values={mem_kb(v1):.1f} KB  meta={mem_kb(cp1, ri1):.1f} KB  "
          f"total={mem_kb(v1, cp1, ri1):.1f} KB ({mem_kb(v1, cp1, ri1) / (W.nelement() * 2 / 1024):.0%})")
    print(f"  V2 Hier:         values={mem_kb(h_ksub_vals):.1f} KB  "
          f"meta={mem_kb(h_col_ptr, h_row_idx, h_types, h_bitmaps, h_ksub_off):.1f} KB  "
          f"total={mem_kb(h_ksub_vals, h_col_ptr, h_row_idx, h_types, h_bitmaps, h_ksub_off):.1f} KB "
          f"({mem_kb(h_ksub_vals, h_col_ptr, h_row_idx, h_types, h_bitmaps, h_ksub_off) / (W.nelement() * 2 / 1024):.0%})")

    # GroupTile stats
    n_dense = (h_types == 0).sum().item()
    n_sparse = (h_types == 1).sum().item()
    n_total = n_dense + n_sparse
    total_gtiles = (4096 // 64) * (14336 // 64)
    print(f"  GroupTiles: {n_total}/{total_gtiles} non-zero "
          f"({n_dense} DENSE, {n_sparse} SPARSE)")


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
