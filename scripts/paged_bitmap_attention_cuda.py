"""
Paged Bitmap Attention — compiled CUDA kernel wrapper.

Calls the C++ extension built from csrc/paged_bitmap_attention.cu.

Two-stage architecture:
  Stage 1: Dense past context with paged KV reads (indirect token slot indexing)
  Stage 2: Bitmap-driven sparse tree attention with paged KV reads + tree mask

Build the extension first:
    cd csrc && python setup_paged_bitmap_attention.py build_ext --inplace

Usage:
    cd scripts
    python paged_bitmap_attention_cuda.py validate
    python paged_bitmap_attention_cuda.py bench
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add csrc to path for the compiled extension
_csrc_dir = os.path.join(os.path.dirname(__file__), "..", "csrc")
sys.path.insert(0, _csrc_dir)

try:
    import paged_bitmap_attention_ext
except ImportError as e:
    print(f"Failed to import paged_bitmap_attention_ext: {e}")
    print("Build it first: cd csrc && python setup_paged_bitmap_attention.py build_ext --inplace")
    sys.exit(1)


# ============================================================================
# GPU BITMAP COMPUTATION (reused from paged_bitmap_attention.py)
# ============================================================================

def compute_bitmap_from_mask_gpu(tree_mask_bool, BLOCK_M=32, BLOCK_N=32):
    """Compute per-q_block uint64 bitmaps from boolean tree mask, entirely on GPU.

    Args:
        tree_mask_bool: [N_tree, N_tree] boolean tensor (GPU).
        BLOCK_M: Q block size.
        BLOCK_N: KV block size.

    Returns:
        bitmaps: [num_q_blocks, W] int64 tensor (GPU).
        W: number of uint64 words per q_block.
    """
    device = tree_mask_bool.device
    N_tree = tree_mask_bool.shape[0]
    num_q_blocks = (N_tree + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (N_tree + BLOCK_N - 1) // BLOCK_N

    pad_q = num_q_blocks * BLOCK_M - N_tree
    pad_kv = num_kv_blocks * BLOCK_N - N_tree
    if pad_q > 0 or pad_kv > 0:
        mask_padded = F.pad(tree_mask_bool, (0, pad_kv, 0, pad_q), value=False)
    else:
        mask_padded = tree_mask_bool

    mask_blocked = mask_padded.reshape(num_q_blocks, BLOCK_M,
                                       num_kv_blocks, BLOCK_N)
    block_active = mask_blocked.any(dim=3).any(dim=1)  # [num_q_blocks, num_kv_blocks]

    W = (num_kv_blocks + 63) // 64
    pad_to_w64 = W * 64 - num_kv_blocks
    if pad_to_w64 > 0:
        block_active = F.pad(block_active, (0, pad_to_w64), value=False)

    block_active = block_active.reshape(num_q_blocks, W, 64)
    powers = (1 << torch.arange(64, device=device, dtype=torch.int64))
    bitmaps = (block_active.to(torch.int64) * powers[None, None, :]).sum(dim=2)

    return bitmaps, W


# ============================================================================
# LAUNCH WRAPPER
# ============================================================================

def paged_bitmap_attention_cuda(q_heads, k_buf, v_buf, kv_indices,
                                 tree_mask_2d, bitmaps, W, past_len,
                                 sm_scale, H_kv, BLOCK_M=None, BLOCK_N=32):
    """Launch the compiled CUDA paged bitmap attention kernel.

    Args:
        q_heads: [H_q, N_tree, D] fp16
        k_buf: [max_tokens, H_kv, D] fp16 — paged K buffer
        v_buf: [max_tokens, H_kv, D] fp16 — paged V buffer
        kv_indices: [total_kv] int32 — token slot mapping
        tree_mask_2d: [N_tree, N_tree] fp16 — 0 or -inf tree mask
        bitmaps: [num_q_blocks, W] int64 — per-q_block bitmap
        W: int — number of uint64 words per q_block
        past_len: int
        sm_scale: float
        H_kv: int
        BLOCK_M: int (auto-selected if None)
        BLOCK_N: int (must be 32, matches kernel)

    Returns:
        output: [H_q, N_tree, D] fp16
    """
    return paged_bitmap_attention_ext.forward(
        q_heads, k_buf, v_buf, kv_indices,
        tree_mask_2d, bitmaps,
        sm_scale, past_len, W, H_kv)


def paged_dense_attention_cuda(q_heads, k_buf, v_buf, kv_indices,
                                tree_mask_2d, past_len, sm_scale, H_kv):
    """Launch the compiled CUDA paged dense attention kernel (no bitmap).

    Args:
        q_heads: [H_q, N_tree, D] fp16
        k_buf, v_buf: [max_tokens, H_kv, D] fp16
        kv_indices: [total_kv] int32
        tree_mask_2d: [N_tree, N_tree] fp16
        past_len: int
        sm_scale: float
        H_kv: int

    Returns:
        output: [H_q, N_tree, D] fp16
    """
    return paged_bitmap_attention_ext.dense_forward(
        q_heads, k_buf, v_buf, kv_indices,
        tree_mask_2d, sm_scale, past_len, H_kv)


# ============================================================================
# DENSE REFERENCE (PyTorch, fp32)
# ============================================================================

def _dense_reference(q_heads, k_full, v_full, tree_mask_bool, past_len,
                     sm_scale, H_kv):
    """Dense attention reference (PyTorch, fp32 accumulation).

    Args:
        q_heads: [H_q, N_tree, D]
        k_full: [total_kv, H_kv, D]
        v_full: [total_kv, H_kv, D]
        tree_mask_bool: [N_tree, N_tree] bool
        past_len: int
        sm_scale: float
        H_kv: int
    """
    H_q, N_tree, D = q_heads.shape
    total_kv = k_full.shape[0]

    full_mask = torch.ones(N_tree, total_kv, dtype=torch.bool,
                           device=q_heads.device)
    full_mask[:, past_len:past_len + N_tree] = tree_mask_bool

    attn_mask = torch.where(full_mask, 0.0, float('-inf')).to(torch.float32)

    gqa_group = H_q // H_kv
    outputs = []
    for h_q in range(H_q):
        h_kv = h_q // gqa_group
        q = q_heads[h_q].float()
        k = k_full[:, h_kv, :].float()
        v = v_full[:, h_kv, :].float()

        scores = torch.matmul(q, k.T) * sm_scale + attn_mask
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        outputs.append(out)

    return torch.stack(outputs).to(q_heads.dtype)


# ============================================================================
# VALIDATION
# ============================================================================

def validate(device='cuda'):
    """Validate CUDA paged bitmap kernel against dense PyTorch reference."""
    from eagle_tree_choices import EAGLE_TREES

    BLOCK_N = 32

    for tree_name in ["mc_sim_7b_63", "regular_512"]:
        tree_choices = EAGLE_TREES[tree_name]
        N_tree = len(tree_choices) + 1

        for past_len in [0, 64, 256]:
            total_kv = past_len + N_tree
            H_q = 8
            H_kv = 2
            D = 128
            sm_scale = 1.0 / (D ** 0.5)

            # Get BLOCK_M from kernel
            BLOCK_M = paged_bitmap_attention_ext.get_block_m(N_tree)

            print(f"\n{'='*60}")
            print(f"Validating CUDA paged bitmap: {tree_name} "
                  f"(N_tree={N_tree}, past={past_len}, "
                  f"H_q={H_q}, H_kv={H_kv}, BLOCK_M={BLOCK_M})")
            print(f"{'='*60}")

            torch.manual_seed(42)

            # Simulate paged KV buffer (larger, with gaps)
            max_tokens = total_kv + 200
            k_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                                device=device)
            v_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                                device=device)

            # Non-contiguous token slot mapping
            perm = torch.randperm(max_tokens, device=device)
            kv_indices = perm[:total_kv].to(torch.int32)

            q_heads = torch.randn(H_q, N_tree, D, dtype=torch.float16,
                                  device=device)

            # Build tree mask from tree_choices
            tree_mask_bool = torch.zeros(N_tree, N_tree, dtype=torch.bool,
                                         device=device)
            node_to_idx = {(): 0}
            for i, node in enumerate(tree_choices):
                node_to_idx[node] = i + 1
            tree_mask_bool[0, 0] = True
            for i, node in enumerate(tree_choices):
                idx = i + 1
                tree_mask_bool[idx, 0] = True
                for pl in range(1, len(node)):
                    anc_idx = node_to_idx[node[:pl]]
                    tree_mask_bool[idx, anc_idx] = True
                tree_mask_bool[idx, idx] = True

            # Dense PyTorch reference (gather contiguous)
            k_full = k_buf[kv_indices.long()]
            v_full = v_buf[kv_indices.long()]
            out_ref = _dense_reference(q_heads, k_full, v_full,
                                       tree_mask_bool, past_len, sm_scale, H_kv)

            # GPU bitmap computation
            bitmaps, W = compute_bitmap_from_mask_gpu(tree_mask_bool,
                                                       BLOCK_M=BLOCK_M,
                                                       BLOCK_N=BLOCK_N)

            # Float tree mask for kernel
            tree_mask_2d = torch.where(
                tree_mask_bool,
                torch.zeros(1, device=device, dtype=torch.float16),
                torch.full((1,), float('-inf'), device=device, dtype=torch.float16),
            )

            # ---- Test bitmap kernel ----
            out_bitmap = paged_bitmap_attention_cuda(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                bitmaps, W, past_len, sm_scale, H_kv)

            max_diff = (out_bitmap - out_ref).abs().max().item()
            mean_diff = (out_bitmap - out_ref).abs().mean().item()
            cos_sim = F.cosine_similarity(
                out_bitmap.float().flatten(), out_ref.float().flatten(), dim=0
            ).item()
            passed = max_diff < 5e-2
            print(f"  [bitmap]  max={max_diff:.2e}  mean={mean_diff:.2e}  "
                  f"cos={cos_sim:.6f}  {'PASS' if passed else 'FAIL'}")

            # ---- Test dense kernel ----
            out_dense = paged_dense_attention_cuda(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                past_len, sm_scale, H_kv)

            max_diff_d = (out_dense - out_ref).abs().max().item()
            mean_diff_d = (out_dense - out_ref).abs().mean().item()
            cos_sim_d = F.cosine_similarity(
                out_dense.float().flatten(), out_ref.float().flatten(), dim=0
            ).item()
            passed_d = max_diff_d < 5e-2
            print(f"  [dense]   max={max_diff_d:.2e}  mean={mean_diff_d:.2e}  "
                  f"cos={cos_sim_d:.6f}  {'PASS' if passed_d else 'FAIL'}")

            # ---- Cross-check bitmap vs dense ----
            cross_diff = (out_bitmap - out_dense).abs().max().item()
            print(f"  [bitmap vs dense] max diff={cross_diff:.2e}")

    print("\nAll validation complete.")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark(device='cuda', warmup=20, rep=100):
    """Microbenchmark paged bitmap vs paged dense CUDA kernels."""
    from eagle_tree_choices import EAGLE_TREES, regular_tree_512

    BLOCK_N = 32

    tree_choices = regular_tree_512
    N_tree = len(tree_choices) + 1
    H_q = 8
    H_kv = 2
    D = 128
    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = paged_bitmap_attention_ext.get_block_m(N_tree)

    print(f"Benchmark: regular_512 (N_tree={N_tree}), H_q={H_q}, H_kv={H_kv}, "
          f"D={D}, BLOCK_M={BLOCK_M}")
    print(f"{'past_len':>10}  {'bitmap_us':>12}  {'dense_us':>12}  {'speedup':>8}")
    print("-" * 50)

    for past_len in [0, 256, 1024, 4096]:
        total_kv = past_len + N_tree
        max_tokens = total_kv + 500

        torch.manual_seed(42)
        k_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                            device=device)
        v_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                            device=device)
        perm = torch.randperm(max_tokens, device=device)
        kv_indices = perm[:total_kv].to(torch.int32)
        q_heads = torch.randn(H_q, N_tree, D, dtype=torch.float16,
                              device=device)

        # Build tree mask
        tree_mask_bool = torch.zeros(N_tree, N_tree, dtype=torch.bool,
                                     device=device)
        node_to_idx = {(): 0}
        for i, node in enumerate(tree_choices):
            node_to_idx[node] = i + 1
        tree_mask_bool[0, 0] = True
        for i, node in enumerate(tree_choices):
            idx = i + 1
            tree_mask_bool[idx, 0] = True
            for pl in range(1, len(node)):
                anc_idx = node_to_idx[node[:pl]]
                tree_mask_bool[idx, anc_idx] = True
            tree_mask_bool[idx, idx] = True

        bitmaps, W = compute_bitmap_from_mask_gpu(tree_mask_bool,
                                                   BLOCK_M=BLOCK_M,
                                                   BLOCK_N=BLOCK_N)
        tree_mask_2d = torch.where(
            tree_mask_bool,
            torch.zeros(1, device=device, dtype=torch.float16),
            torch.full((1,), float('-inf'), device=device, dtype=torch.float16),
        )

        # Warmup
        for _ in range(warmup):
            paged_bitmap_attention_cuda(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                bitmaps, W, past_len, sm_scale, H_kv)
            paged_dense_attention_cuda(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                past_len, sm_scale, H_kv)
        torch.cuda.synchronize()

        # Benchmark bitmap
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            start_events[i].record()
            paged_bitmap_attention_cuda(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                bitmaps, W, past_len, sm_scale, H_kv)
            end_events[i].record()
        torch.cuda.synchronize()
        bitmap_us = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events)) / rep * 1000

        # Benchmark dense
        start_events2 = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end_events2 = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            start_events2[i].record()
            paged_dense_attention_cuda(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                past_len, sm_scale, H_kv)
            end_events2[i].record()
        torch.cuda.synchronize()
        dense_us = sum(s.elapsed_time(e) for s, e in zip(start_events2, end_events2)) / rep * 1000

        speedup = dense_us / bitmap_us if bitmap_us > 0 else float('inf')
        print(f"{past_len:>10}  {bitmap_us:>12.1f}  {dense_us:>12.1f}  {speedup:>8.2f}x")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "validate":
        validate()
    elif cmd == "bench":
        benchmark()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python paged_bitmap_attention_cuda.py [validate|bench]")
        sys.exit(1)
