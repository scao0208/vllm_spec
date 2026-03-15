"""
Test and benchmark the CUDA bitmap attention kernel.

Validates against PyTorch reference attention and benchmarks bitmap vs dense.

Usage:
    cd scripts
    python test_bitmap_attention_cuda.py validate
    python test_bitmap_attention_cuda.py bench
    python test_bitmap_attention_cuda.py build   # build the extension first

Build the extension first:
    cd csrc && python setup_bitmap_attention.py build_ext --inplace && cd ../scripts
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

# Add csrc to path for the compiled extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csrc'))

from eagle_tree_choices import EAGLE_TREES, get_tree_stats
from bitmap_sparse_attention import (
    precompute_bitmap_metadata, _build_tree_mask, make_dense_stage1_indices
)


def _load_extension():
    """Load the compiled CUDA extension."""
    try:
        import bitmap_attention
        return bitmap_attention
    except ImportError:
        print("ERROR: bitmap_attention extension not found.")
        print("Build it first:")
        print("  cd csrc && python setup_bitmap_attention.py build_ext --inplace")
        sys.exit(1)


def pytorch_reference_attention(q, k, v, past_len, tree_mask_2d, scale):
    """Compute attention using PyTorch as reference.

    Args:
        q: [B, H, N_Q, D]
        k: [B, H, N_KV, D]
        v: [B, H, N_KV, D]
        past_len: number of past context tokens
        tree_mask_2d: [N_Q, N_Q] semantic mask (0 or -inf)
        scale: softmax scale

    Returns:
        output: [B, H, N_Q, D]
    """
    B, H, N_Q, D = q.shape
    N_KV = k.shape[2]

    # Build full attention mask [N_Q, N_KV]
    # Past context: all visible (0.0)
    # Tree region: use tree_mask
    full_mask = torch.zeros(N_Q, N_KV, device=q.device, dtype=q.dtype)
    full_mask[:, past_len:past_len + N_Q] = tree_mask_2d

    # Compute attention manually
    # QK^T: [B, H, N_Q, N_KV]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    scores += full_mask.float().unsqueeze(0).unsqueeze(0)  # broadcast mask

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # PV: [B, H, N_Q, D]
    output = torch.matmul(attn, v.float())
    return output.half()


def validate(device='cuda'):
    """Validate CUDA bitmap kernel against PyTorch reference."""
    ext = _load_extension()

    trees_to_test = ['mc_sim_7b_63', 'regular_512']
    # Filter to available trees
    trees_to_test = [t for t in trees_to_test if t in EAGLE_TREES]

    configs = [
        # (past_len, B, H, D)
        (64, 2, 4, 128),
        (128, 1, 8, 128),
        (256, 2, 4, 64),
    ]

    all_passed = True

    for tree_name in trees_to_test:
        tree_choices = EAGLE_TREES[tree_name]
        stats = get_tree_stats(tree_choices)

        for past_len, B, H, D in configs:
            N_tree = len(tree_choices) + 1
            N_KV = past_len + N_tree

            # Get BLOCK_M that the kernel will use for this N_Q
            block_m = ext.get_block_m(N_tree)

            # Precompute bitmap metadata with matching BLOCK_M
            bitmaps, W, num_kv_blocks = precompute_bitmap_metadata(
                tree_choices, BLOCK_M=block_m, BLOCK_N=32)

            torch.manual_seed(42)
            q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
            k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
            v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
            scale = 1.0 / (D ** 0.5)

            # Build tree mask [N_tree, N_tree]
            tree_mask_4d = _build_tree_mask(tree_choices, N_tree, device)
            tree_mask_2d = tree_mask_4d.squeeze(0).squeeze(0)  # [N_tree, N_tree]

            # PyTorch reference
            out_ref = pytorch_reference_attention(
                q, k, v, past_len, tree_mask_2d, scale)

            # CUDA bitmap kernel
            bitmaps_dev = bitmaps.to(device)
            stage1_indices = make_dense_stage1_indices(past_len).to(device)
            out_bitmap = ext.bitmap_attention_fwd(
                q, k, v, tree_mask_2d, stage1_indices, bitmaps_dev,
                scale, past_len, W)

            # CUDA dense kernel
            out_dense = ext.dense_attention_fwd(
                q, k, v, tree_mask_2d, scale, past_len)

            # Compare
            diff_bitmap = (out_bitmap.float() - out_ref.float()).abs()
            diff_dense = (out_dense.float() - out_ref.float()).abs()

            max_diff_bitmap = diff_bitmap.max().item()
            mean_diff_bitmap = diff_bitmap.mean().item()
            max_diff_dense = diff_dense.max().item()
            mean_diff_dense = diff_dense.mean().item()

            passed_bitmap = max_diff_bitmap < 5e-2
            passed_dense = max_diff_dense < 5e-2

            status_b = "PASS" if passed_bitmap else "FAIL"
            status_d = "PASS" if passed_dense else "FAIL"

            print(f"Tree={tree_name} ({stats['total_nodes']}n), "
                  f"past={past_len}, B={B}, H={H}, D={D}")
            print(f"  Bitmap vs ref: max={max_diff_bitmap:.2e}, "
                  f"mean={mean_diff_bitmap:.2e} [{status_b}]")
            print(f"  Dense  vs ref: max={max_diff_dense:.2e}, "
                  f"mean={mean_diff_dense:.2e} [{status_d}]")

            # Also check bitmap matches dense
            diff_bd = (out_bitmap.float() - out_dense.float()).abs()
            max_diff_bd = diff_bd.max().item()
            passed_bd = max_diff_bd < 1e-3
            status_bd = "PASS" if passed_bd else "FAIL"
            print(f"  Bitmap vs dense: max={max_diff_bd:.2e} [{status_bd}]")

            if not (passed_bitmap and passed_dense and passed_bd):
                all_passed = False

    if all_passed:
        print("\nAll validation tests PASSED")
    else:
        print("\nSome tests FAILED")
    return all_passed


def bench(device='cuda', warmup=20, rep=100):
    """Benchmark bitmap vs dense CUDA kernels."""
    ext = _load_extension()

    tree_name = 'regular_512'
    if tree_name not in EAGLE_TREES:
        print(f"Tree {tree_name} not found, available: {list(EAGLE_TREES.keys())}")
        return

    tree_choices = EAGLE_TREES[tree_name]
    stats = get_tree_stats(tree_choices)
    N_tree = len(tree_choices) + 1

    configs = [
        # (past_len, B, H, D)
        (64, 4, 32, 128),
        (256, 4, 32, 128),
        (1024, 4, 32, 128),
        (4096, 4, 32, 128),
    ]

    print(f"Tree: {tree_name} ({stats['total_nodes']} nodes, "
          f"depth {stats['max_depth']})")
    print(f"{'past':>6} {'B':>3} {'H':>3} {'D':>4}  "
          f"{'Dense(ms)':>10} {'Bitmap(ms)':>11} {'Speedup':>8}")
    print("-" * 60)

    for past_len, B, H, D in configs:
        N_KV = past_len + N_tree
        scale = 1.0 / (D ** 0.5)

        block_m = ext.get_block_m(N_tree)
        bitmaps, W, _ = precompute_bitmap_metadata(
            tree_choices, BLOCK_M=block_m, BLOCK_N=32)
        bitmaps_dev = bitmaps.to(device)
        stage1_indices = make_dense_stage1_indices(past_len).to(device)

        q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
        k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
        v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)

        tree_mask_4d = _build_tree_mask(tree_choices, N_tree, device)
        tree_mask_2d = tree_mask_4d.squeeze(0).squeeze(0)

        # Warmup
        for _ in range(warmup):
            ext.dense_attention_fwd(q, k, v, tree_mask_2d, scale, past_len)
            ext.bitmap_attention_fwd(
                q, k, v, tree_mask_2d, stage1_indices, bitmaps_dev,
                scale, past_len, W)
        torch.cuda.synchronize()

        # Benchmark dense
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            ext.dense_attention_fwd(q, k, v, tree_mask_2d, scale, past_len)
        torch.cuda.synchronize()
        t_dense = (time.perf_counter() - t0) / rep * 1000

        # Benchmark bitmap
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            ext.bitmap_attention_fwd(
                q, k, v, tree_mask_2d, stage1_indices, bitmaps_dev,
                scale, past_len, W)
        torch.cuda.synchronize()
        t_bitmap = (time.perf_counter() - t0) / rep * 1000

        speedup = t_dense / t_bitmap if t_bitmap > 0 else float('inf')
        print(f"{past_len:>6} {B:>3} {H:>3} {D:>4}  "
              f"{t_dense:>10.3f} {t_bitmap:>11.3f} {speedup:>7.2f}x")


def build():
    """Build the CUDA extension."""
    import subprocess
    csrc_dir = os.path.join(os.path.dirname(__file__), '..', 'csrc')
    print(f"Building in {csrc_dir}...")
    result = subprocess.run(
        [sys.executable, 'setup_bitmap_attention.py', 'build_ext', '--inplace'],
        cwd=csrc_dir,
        capture_output=False
    )
    if result.returncode == 0:
        print("Build succeeded!")
    else:
        print("Build failed!")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_bitmap_attention_cuda.py [build|validate|bench]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'build':
        build()
    elif cmd == 'validate':
        validate()
    elif cmd == 'bench':
        bench()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python test_bitmap_attention_cuda.py [build|validate|bench]")
        sys.exit(1)
