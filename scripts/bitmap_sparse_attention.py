"""
Bitmap-driven sparse tree attention Triton kernel for Scenario A.

Promotes tree bitmasks from a "mask representation" to an **execution schedule**
that skips loading and computing on invisible KV blocks entirely, using
register-resident uint64 bitmap words with ctz/ffs + clear-lowest-bit traversal.

Two-stage architecture (same as eagle_tree_attention.py):
  - Stage 1 (off-diagonal): Dense attention to past context [0, past_len).
    No mask needed — all Q positions attend to all past KV. Uses TMA streaming.
  - Stage 2 (on-diagonal):  Bitmap-driven sparse attention over tree region
    [past_len, past_len + N_tree). Only visits KV blocks where at least one
    ancestor exists, as indicated by set bits in a uint64 bitmap.

Bitmap layout (from deep-research-report.md):
  - Each KV block maps to one bit: block_id → word[block_id // 64], bit[block_id % 64]
  - uint64 words stored per q_block: bitmap[q_block, w] where w = 0..W-1
  - Traversal: while mask != 0: b = ctz(mask); mask &= (mask - 1); process block b
  - Metadata overhead: 8 * ceil(N_blocks/64) bytes per q_block (negligible vs KV tiles)

Comparison with sparse_tree_kernel.py:
  - sparse_tree_kernel uses explicit index lists (block_indices/block_counts),
    requiring indirect loads from global memory per iteration
  - bitmap_sparse_attention uses register-resident uint64 masks with integer
    intrinsics (ctz + popcount), eliminating pointer chasing overhead
  - Both skip the same set of KV blocks — identical sparsity, different traversal

Usage:
    cd scripts
    python bitmap_sparse_attention.py validate
    python bitmap_sparse_attention.py bench
    python bitmap_sparse_attention.py stats
"""

import torch
import triton
import triton.language as tl

from sparse_tree_utils import precompute_ancestor_indices
from eagle_tree_choices import get_tree_stats


# ============================================================================
# TRITON HELPERS
# ============================================================================

@triton.jit
def _ctz64(mask):
    """Count trailing zeros of 64-bit integer (position of lowest set bit).

    Uses popcount(lowest_bit - 1) with explicit 64-bit handling:
    splits into two 32-bit halves so libdevice.popc (__popc) works correctly
    even when the set bit is at position >= 32.

    Safe across all Triton versions (avoids relying on popc/popcll dispatch).
    """
    lowest_bit = mask & (-mask)
    val = lowest_bit - 1
    # Split into two 32-bit halves for portable popcount
    lo = (val & 0xFFFFFFFF).to(tl.int32)
    hi = ((val >> 32) & 0xFFFFFFFF).to(tl.int32)
    return tl.extra.cuda.libdevice.popc(lo) + tl.extra.cuda.libdevice.popc(hi)


# ============================================================================
# HOST-SIDE: Bitmap precomputation
# ============================================================================

def precompute_bitmap_metadata(tree_choices, BLOCK_M=32, BLOCK_N=32):
    """Convert tree structure → per-q_block uint64 bitmap words.

    For each q_block, computes the union of all ancestor KV blocks across
    all Q positions in the block, then packs into uint64 bitmap words.

    Args:
        tree_choices: Sorted tree_choices list.
        BLOCK_M: Q block size (must match kernel).
        BLOCK_N: KV block size (must match kernel).

    Returns:
        bitmaps: [num_q_blocks, W] int64 tensor.
            Packed uint64 bitmap per q_block. Bit b in word w means
            KV block (w * 64 + b) should be visited.
        W: Number of uint64 words per q_block.
        num_kv_blocks: Total number of KV blocks in tree region.
    """
    ancestor_indices, _, ancestor_counts = precompute_ancestor_indices(tree_choices)

    N_tree = len(tree_choices) + 1  # +1 for root
    num_q_blocks = (N_tree + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (N_tree + BLOCK_N - 1) // BLOCK_N
    W = (num_kv_blocks + 63) // 64  # number of uint64 words

    bitmaps = torch.zeros(num_q_blocks, W, dtype=torch.int64)

    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min(q_start + BLOCK_M, N_tree)

        for q_pos in range(q_start, q_end):
            n_anc = int(ancestor_counts[q_pos].item())
            for a in range(n_anc):
                anc_idx = int(ancestor_indices[q_pos, a].item())
                kv_block_id = anc_idx // BLOCK_N
                word_idx = kv_block_id // 64
                bit_idx = kv_block_id % 64
                bitmaps[qb, word_idx] |= (1 << bit_idx)

    return bitmaps, W, num_kv_blocks


def make_dense_stage1_indices(past_len, BLOCK_N=32):
    """Generate a dense block index array for Stage 1 (all past context blocks).

    For fully dense past context, returns [0, 1, 2, ..., num_past_blocks-1].
    The CUDA bitmap kernel iterates over this array to select which KV blocks
    to visit in Stage 1.

    Args:
        past_len: Number of past context tokens.
        BLOCK_N: KV block size (must match kernel).

    Returns:
        stage1_indices: [num_past_blocks] int32 tensor.
    """
    num_past_blocks = (past_len + BLOCK_N - 1) // BLOCK_N
    return torch.arange(num_past_blocks, dtype=torch.int32)


def validate_bitmap_against_index_list(tree_choices, BLOCK_M=32, BLOCK_N=32):
    """Validate that bitmap and index-list representations select identical blocks.

    Returns True if they match exactly.
    """
    from sparse_tree_kernel import precompute_block_sparse_metadata

    bitmaps, W, num_kv_blocks = precompute_bitmap_metadata(tree_choices, BLOCK_M, BLOCK_N)
    block_indices, block_counts, _ = precompute_block_sparse_metadata(tree_choices, BLOCK_M, BLOCK_N)

    num_q_blocks = bitmaps.shape[0]
    all_match = True

    for qb in range(num_q_blocks):
        # Extract set bits from bitmap
        bitmap_blocks = set()
        for w in range(W):
            mask = int(bitmaps[qb, w].item())
            while mask:
                b = (mask & -mask).bit_length() - 1  # ctz equivalent
                bitmap_blocks.add(w * 64 + b)
                mask &= mask - 1

        # Extract from index list
        n_blocks = int(block_counts[qb].item())
        index_blocks = set()
        for i in range(n_blocks):
            index_blocks.add(int(block_indices[qb, i].item()))

        if bitmap_blocks != index_blocks:
            print(f"Q block {qb}: MISMATCH")
            print(f"  Bitmap: {sorted(bitmap_blocks)}")
            print(f"  Index:  {sorted(index_blocks)}")
            all_match = False

    return all_match


# ============================================================================
# TRITON KERNEL: Bitmap-driven Stage 2
# ============================================================================

@triton.jit
def _bitmap_stage2_inner(acc, l_i, m_i, q,
                          desc_k, desc_v, desc_tree_mask,
                          tree_mask_row_offset, dtype: tl.constexpr,
                          start_m, qk_scale,
                          BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                          BLOCK_N: tl.constexpr,
                          offs_m: tl.constexpr, offs_n: tl.constexpr,
                          N_CTX_Q: tl.constexpr, N_CTX_KV: tl.constexpr,
                          # Bitmap metadata
                          bitmap_ptr,
                          past_len,
                          W: tl.constexpr):
    """Stage 2: bitmap-driven sparse attention over tree region.

    Instead of iterating over an index list (indirect loads from global memory),
    loads uint64 bitmap words into registers and uses ctz + clear-lowest-bit
    to enumerate set bits. Each set bit → one KV block to visit.

    The bitmap is per-q_block: all Q rows in the block share the same bitmap,
    ensuring warp-coherent control flow (no divergence within the CTA).
    """

    for w in tl.range(0, W):
        # Load bitmap word into register — tiny metadata, one load per word
        mask = tl.load(bitmap_ptr + w)

        # Traverse set bits: ctz + clear-lowest-bit loop
        while mask != 0:
            # Count trailing zeros to get position of lowest set bit
            # Uses portable 64-bit popcount (split into two 32-bit halves)
            b = _ctz64(mask)
            # Clear lowest set bit
            mask = mask & (mask - 1)

            # Compute KV block ID and position
            kv_block_id = w * 64 + b
            start_n = past_len + kv_block_id * BLOCK_N

            # Load K and compute QK
            k = desc_k.load([start_n, 0]).T
            qk = tl.dot(q, k)

            # Boundary mask
            boundary_mask = (start_n + offs_n[None, :]) < N_CTX_KV

            # Tree mask: load block for fine-grained within-block masking
            tree_col = kv_block_id * BLOCK_N
            tree_mask_block = desc_tree_mask.load([tree_mask_row_offset, tree_col])
            qk = qk * qk_scale + tree_mask_block + tl.where(boundary_mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]

            # Load V and accumulate
            v = desc_v.load([start_n, 0])
            p = p.to(dtype)
            acc = tl.dot(p, v, acc)

            l_i = l_i * alpha + l_ij
            m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _stage1_inner(acc, l_i, m_i, q,
                  desc_k, desc_v,
                  dtype: tl.constexpr, qk_scale,
                  BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  offs_m: tl.constexpr, offs_n: tl.constexpr,
                  N_CTX_KV: tl.constexpr, past_len):
    """Stage 1: dense attention to past context [0, past_len). No mask needed.

    Identical to sparse_tree_kernel._stage1_inner — kept here to avoid
    cross-file JIT dependency issues.
    """

    for start_n in tl.range(0, past_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([start_n, 0]).T
        qk = tl.dot(q, k)

        # Boundary mask for past context
        past_ctx_mask = (start_n + offs_n[None, :]) < past_len
        boundary_mask = (start_n + offs_n[None, :]) < N_CTX_KV
        combined_mask = boundary_mask & past_ctx_mask
        qk = qk * qk_scale + tl.where(combined_mask, 0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = desc_v.load([start_n, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_bitmap(
    Q, K, V, O, M, Tree_mask,
    Bitmaps,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_tz, stride_th, stride_tm, stride_tn,
    stride_bm_q,  # bitmap stride for Q block dim (= W)
    Z, H, N_CTX_Q, N_CTX_KV,
    TREE_MASK_DIM,  # Padded tree mask dimension (>= N_CTX_Q, aligned to 8)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    W: tl.constexpr,
):
    """Bitmap-driven sparse tree attention kernel.

    Stage 1: Dense attention to past context (TMA streaming, no mask).
    Stage 2: Bitmap-driven sparse attention to tree region.
    """
    dtype = tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qk_scale = sm_scale * 1.44269504
    past_len = N_CTX_KV - N_CTX_Q

    # Base pointers for this (batch, head)
    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    o_base = O + off_z * stride_oz + off_h * stride_oh
    t_base = Tree_mask + off_z * stride_tz + off_h * stride_th

    # Per-head TensorDescriptors
    desc_q = tl.make_tensor_descriptor(
        q_base, shape=[N_CTX_Q, HEAD_DIM], strides=[stride_qm, stride_qk],
        block_shape=[BLOCK_M, HEAD_DIM]
    )
    desc_k = tl.make_tensor_descriptor(
        k_base, shape=[N_CTX_KV, HEAD_DIM], strides=[stride_kn, stride_kk],
        block_shape=[BLOCK_N, HEAD_DIM]
    )
    desc_v = tl.make_tensor_descriptor(
        v_base, shape=[N_CTX_KV, HEAD_DIM], strides=[stride_vn, stride_vk],
        block_shape=[BLOCK_N, HEAD_DIM]
    )
    desc_o = tl.make_tensor_descriptor(
        o_base, shape=[N_CTX_Q, HEAD_DIM], strides=[stride_om, stride_ok],
        block_shape=[BLOCK_M, HEAD_DIM]
    )
    desc_tree_mask = tl.make_tensor_descriptor(
        t_base, shape=[TREE_MASK_DIM, TREE_MASK_DIM], strides=[stride_tm, stride_tn],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    # Load Q block
    qo_offset = start_m * BLOCK_M
    q = desc_q.load([qo_offset, 0])

    # Initialize accumulators
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q_valid_mask = offs_m < N_CTX_Q

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Tree mask row offset
    tree_mask_row_offset = start_m * BLOCK_M

    # ---- Stage 1: Dense past context ----
    if past_len > 0:
        acc, l_i, m_i = _stage1_inner(
            acc, l_i, m_i, q, desc_k, desc_v,
            dtype, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N, offs_m, offs_n,
            N_CTX_KV, past_len
        )

    # ---- Stage 2: Bitmap-driven sparse tree region ----
    # Pointer to this q_block's bitmap row
    bitmap_ptr = Bitmaps + start_m * stride_bm_q

    acc, l_i, m_i = _bitmap_stage2_inner(
        acc, l_i, m_i, q, desc_k, desc_v, desc_tree_mask,
        tree_mask_row_offset, dtype, start_m, qk_scale,
        BLOCK_M, HEAD_DIM, BLOCK_N, offs_m, offs_n,
        N_CTX_Q, N_CTX_KV,
        bitmap_ptr, past_len, W
    )

    # Epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # Store M
    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i, mask=q_valid_mask)

    # Store output
    out_mask = q_valid_mask[:, None]
    acc_masked = tl.where(out_mask, acc.to(dtype), 0.0)
    desc_o.store([qo_offset, 0], acc_masked)


# ============================================================================
# AUTOGRAD WRAPPER
# ============================================================================

class _attention_bitmap(torch.autograd.Function):
    """Bitmap-driven sparse tree attention."""

    @staticmethod
    def forward(ctx, q, k, v, tree_mask, sm_scale,
                bitmaps, W):
        """
        Args:
            q: [B, H, N_CTX_Q, D]
            k: [B, H, N_CTX_KV, D]
            v: [B, H, N_CTX_KV, D]
            tree_mask: [B, H, N_CTX_Q, N_CTX_Q] or broadcastable.
            sm_scale: softmax scale
            bitmaps: [num_q_blocks, W] int64 (on device)
            W: number of uint64 words per q_block (compile-time)
        """
        B, H, N_CTX_Q, HEAD_DIM = q.shape
        N_CTX_KV = k.shape[2]

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Prepare tree_mask
        if tree_mask is not None:
            if tree_mask.shape[0] == 1 and B > 1:
                tree_mask = tree_mask.expand(B, -1, -1, -1)
            if tree_mask.shape[1] == 1 and H > 1:
                tree_mask = tree_mask.expand(-1, H, -1, -1)
            tree_mask = tree_mask.contiguous().to(q.dtype)
        else:
            tree_mask = torch.zeros((B, H, N_CTX_Q, N_CTX_Q),
                                    device=q.device, dtype=q.dtype)

        # Pad for TMA alignment if needed
        TMA_ALIGN = 8
        mask_dim = tree_mask.shape[-1]
        N_PAD = (N_CTX_Q + TMA_ALIGN - 1) // TMA_ALIGN * TMA_ALIGN
        if mask_dim < N_PAD:
            pad_n = N_PAD - mask_dim
            tree_mask = torch.nn.functional.pad(
                tree_mask, (0, pad_n, 0, pad_n), value=float('-inf')
            ).contiguous()
        elif mask_dim >= N_PAD:
            N_PAD = mask_dim  # already padded

        # Ensure bitmaps on device
        if bitmaps.device != q.device:
            bitmaps = bitmaps.to(device=q.device)

        # Output
        o = torch.empty_like(q)
        M = torch.empty((B, H, N_CTX_Q), device=q.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(N_CTX_Q, BLOCK_M), B * H)

        # Multi-GPU support
        device_idx = q.device.index if q.device.index is not None else 0
        prev_device = torch.cuda.current_device()
        torch.cuda.set_device(device_idx)

        try:
            _attn_fwd_bitmap[grid](
                q, k, v, o, M, tree_mask,
                bitmaps,
                sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                tree_mask.stride(0), tree_mask.stride(1),
                tree_mask.stride(2), tree_mask.stride(3),
                bitmaps.stride(0),  # stride_bm_q
                B, H, N_CTX_Q, N_CTX_KV,
                N_PAD,  # TREE_MASK_DIM
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                W=W,
            )
        finally:
            torch.cuda.set_device(prev_device)

        return o


# ============================================================================
# HELPERS
# ============================================================================

def _ensure_triton_allocator():
    """Set Triton memory allocator if not already set."""
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    triton.set_allocator(alloc_fn)


def _build_tree_mask(tree_choices, N_tree, device, dtype=torch.float16):
    """Build tree attention mask (shared helper)."""
    node_to_idx = {(): 0}
    for i, node in enumerate(tree_choices):
        node_to_idx[node] = i + 1

    tree_mask = torch.full((N_tree, N_tree), float('-inf'),
                           device=device, dtype=dtype)
    tree_mask[0, 0] = 0.0
    for i, node in enumerate(tree_choices):
        idx = i + 1
        tree_mask[idx, 0] = 0.0
        for prefix_len in range(1, len(node)):
            anc_idx = node_to_idx[node[:prefix_len]]
            tree_mask[idx, anc_idx] = 0.0
        tree_mask[idx, idx] = 0.0
    return tree_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]


def print_bitmap_stats(tree_choices, BLOCK_M=32, BLOCK_N=32):
    """Print bitmap statistics for a tree."""
    bitmaps, W, num_kv_blocks = precompute_bitmap_metadata(tree_choices, BLOCK_M, BLOCK_N)
    stats = get_tree_stats(tree_choices)
    N_tree = len(tree_choices) + 1
    num_q_blocks = bitmaps.shape[0]

    # Count total active blocks and dense baseline
    total_active = 0
    total_dense = 0
    for qb in range(num_q_blocks):
        # Active blocks from bitmap
        active = 0
        for w in range(W):
            mask = int(bitmaps[qb, w].item())
            active += bin(mask).count('1')
        total_active += active

        # Dense baseline: causal triangular
        q_end_block = min((qb + 1) * BLOCK_M, N_tree)
        dense_blocks = (q_end_block + BLOCK_N - 1) // BLOCK_N
        total_dense += dense_blocks

    metadata_bytes = num_q_blocks * W * 8  # 8 bytes per uint64
    kv_block_bytes = 2 * BLOCK_N * 128 * 2  # 2 * Bc * d * sizeof(fp16) = K + V

    print(f"\nBitmap Statistics for {stats['total_nodes']}-node tree "
          f"(depth {stats['max_depth']}):")
    print(f"  BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print(f"  Q blocks: {num_q_blocks}, KV blocks: {num_kv_blocks}")
    print(f"  Bitmap words per q_block (W): {W}")
    print(f"  Metadata per q_block: {W * 8} bytes ({W} uint64 words)")
    print(f"  Total metadata: {metadata_bytes} bytes")
    print(f"  Stage 2 KV block loads:")
    print(f"    Dense (causal):  {total_dense}")
    print(f"    Bitmap (sparse): {total_active}")
    print(f"    Reduction: {(1 - total_active / total_dense) * 100:.1f}%")
    print(f"  KV block size: {kv_block_bytes} bytes ({kv_block_bytes / 1024:.1f} KiB)")
    print(f"  Metadata / KV ratio: {metadata_bytes / (total_active * kv_block_bytes) * 100:.3f}%")

    # Per q_block breakdown
    print(f"\n  Per-q_block active blocks:")
    for qb in range(min(num_q_blocks, 8)):
        active = 0
        for w in range(W):
            mask = int(bitmaps[qb, w].item())
            active += bin(mask).count('1')
        # Dense blocks for this q_block
        q_end = min((qb + 1) * BLOCK_M, N_tree)
        dense = (q_end + BLOCK_N - 1) // BLOCK_N
        print(f"    q_block {qb}: {active}/{dense} blocks "
              f"({active/dense*100:.0f}% dense)")
    if num_q_blocks > 8:
        print(f"    ... ({num_q_blocks - 8} more q_blocks)")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_bitmap_kernel(tree_choices, past_len=64, B=2, H=4, D=128,
                            device='cuda'):
    """Validate CUDA bitmap kernel against CUDA dense kernel and PyTorch ref."""
    import sys as _sys, os as _os
    csrc_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'csrc')
    if csrc_dir not in _sys.path:
        _sys.path.insert(0, csrc_dir)
    import bitmap_attention as ext

    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree

    BLOCK_M = ext.get_block_m(N_tree)
    bitmaps, W, num_kv_blocks = precompute_bitmap_metadata(tree_choices, BLOCK_M)

    torch.manual_seed(42)
    q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    scale = 1.0 / (D ** 0.5)

    tree_mask_4d = _build_tree_mask(tree_choices, N_tree, device)
    tree_mask_2d = tree_mask_4d.squeeze(0).squeeze(0)

    stage1_indices = make_dense_stage1_indices(past_len).to(device)
    bitmaps_dev = bitmaps.to(device)

    # CUDA dense kernel (reference)
    out_dense = ext.dense_attention_fwd(q, k, v, tree_mask_2d, scale, past_len)

    # CUDA bitmap kernel
    out_bitmap = ext.bitmap_attention_fwd(
        q, k, v, tree_mask_2d, stage1_indices, bitmaps_dev, scale, past_len, W)

    max_diff = (out_bitmap - out_dense).abs().max().item()
    mean_diff = (out_bitmap - out_dense).abs().mean().item()

    stats = get_tree_stats(tree_choices)
    print(f"Tree: {stats['total_nodes']} nodes, depth {stats['max_depth']}")
    print(f"Past context: {past_len}, W={W} bitmap words, BLOCK_M={BLOCK_M}")
    print(f"Bitmap vs Dense CUDA — max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e}")

    if max_diff < 1e-2:
        print("PASSED: CUDA bitmap matches CUDA dense")
    else:
        print("FAILED: Outputs differ significantly")

    # PyTorch reference
    tree_mask_float = tree_mask_4d.to(torch.float32)
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    full_mask = torch.zeros(1, 1, N_tree, N_KV, device=device, dtype=torch.float32)
    full_mask[:, :, :, past_len:] = tree_mask_float
    scores = scores + full_mask
    out_ref = torch.matmul(torch.softmax(scores, dim=-1), v_f).half()

    ref_diff = (out_bitmap - out_ref).abs().max().item()
    print(f"Bitmap vs PyTorch ref — max diff: {ref_diff:.2e}")
    if ref_diff < 5e-2:
        print("PASSED: CUDA bitmap matches PyTorch reference")
    else:
        print("WARNING: Larger diff vs PyTorch (expected for fp16 accumulation)")

    return max_diff


def validate_triton_bitmap(tree_choices, past_len=64, B=2, H=4, D=128,
                            device='cuda'):
    """Validate Triton bitmap kernel against PyTorch reference."""
    _ensure_triton_allocator()

    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree

    BLOCK_M = 32
    bitmaps, W, num_kv_blocks = precompute_bitmap_metadata(tree_choices, BLOCK_M)

    torch.manual_seed(42)
    q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    scale = 1.0 / (D ** 0.5)

    tree_mask_4d = _build_tree_mask(tree_choices, N_tree, device)

    # PyTorch reference
    tree_mask_float = tree_mask_4d.to(torch.float32)
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    full_mask = torch.zeros(1, 1, N_tree, N_KV, device=device, dtype=torch.float32)
    full_mask[:, :, :, past_len:] = tree_mask_float
    scores = scores + full_mask
    out_ref = torch.matmul(torch.softmax(scores, dim=-1), v_f).half()

    # Triton bitmap kernel
    bitmaps_dev = bitmaps.to(device)
    out_triton = _attention_bitmap.apply(q, k, v, tree_mask_4d, scale, bitmaps_dev, W)

    max_diff = (out_triton - out_ref).abs().max().item()
    mean_diff = (out_triton - out_ref).abs().mean().item()

    stats = get_tree_stats(tree_choices)
    print(f"Tree: {stats['total_nodes']} nodes, depth {stats['max_depth']}")
    print(f"Past context: {past_len}, W={W} bitmap words")
    print(f"Triton bitmap vs PyTorch ref — max diff: {max_diff:.2e}, mean: {mean_diff:.2e}")

    if max_diff < 5e-2:
        print("PASSED: Triton bitmap matches PyTorch reference")
    else:
        print("FAILED: Triton bitmap differs significantly from reference")

    return max_diff


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_bitmap_vs_others(tree_choices, past_len=256, B=4, H=8, D=128,
                                device='cuda', warmup=10, rep=100):
    """Microbenchmark CUDA bitmap vs CUDA dense kernels."""
    import sys as _sys, os as _os, time
    csrc_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'csrc')
    if csrc_dir not in _sys.path:
        _sys.path.insert(0, csrc_dir)
    import bitmap_attention as ext

    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree
    scale = 1.0 / (D ** 0.5)

    BLOCK_M = ext.get_block_m(N_tree)
    bitmaps, W, _ = precompute_bitmap_metadata(tree_choices, BLOCK_M)

    q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)

    tree_mask_4d = _build_tree_mask(tree_choices, N_tree, device)
    tree_mask_2d = tree_mask_4d.squeeze(0).squeeze(0)
    stage1_indices = make_dense_stage1_indices(past_len).to(device)
    bitmaps_dev = bitmaps.to(device)

    # Warmup
    for _ in range(warmup):
        ext.dense_attention_fwd(q, k, v, tree_mask_2d, scale, past_len)
        ext.bitmap_attention_fwd(q, k, v, tree_mask_2d, stage1_indices,
                                  bitmaps_dev, scale, past_len, W)
    torch.cuda.synchronize()

    # Benchmark CUDA dense
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        ext.dense_attention_fwd(q, k, v, tree_mask_2d, scale, past_len)
    torch.cuda.synchronize()
    t_dense = (time.perf_counter() - t0) / rep * 1000

    # Benchmark CUDA bitmap
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        ext.bitmap_attention_fwd(q, k, v, tree_mask_2d, stage1_indices,
                                  bitmaps_dev, scale, past_len, W)
    torch.cuda.synchronize()
    t_bitmap = (time.perf_counter() - t0) / rep * 1000

    stats = get_tree_stats(tree_choices)
    total_bitmap_blocks = 0
    num_q_blocks = bitmaps.shape[0]
    for qb in range(num_q_blocks):
        for w in range(W):
            mask = int(bitmaps[qb, w].item())
            total_bitmap_blocks += bin(mask).count('1')
    total_dense_blocks = sum(
        (min((qb + 1) * BLOCK_M, N_tree) + 31) // 32
        for qb in range(num_q_blocks)
    )

    print(f"\nTree: {stats['total_nodes']} nodes, depth {stats['max_depth']}")
    print(f"Past context: {past_len}, B={B}, H={H}, D={D}, BLOCK_M={BLOCK_M}")
    print(f"Stage 2 blocks: bitmap={total_bitmap_blocks}, dense={total_dense_blocks}")
    print(f"CUDA Dense:   {t_dense:.3f} ms")
    print(f"CUDA Bitmap:  {t_bitmap:.3f} ms  ({t_dense / t_bitmap:.2f}x vs dense)")
    if t_bitmap < t_dense:
        print(f"Bitmap speedup: {(1 - t_bitmap / t_dense) * 100:.1f}% faster")
    else:
        print(f"Bitmap overhead: {(t_bitmap / t_dense - 1) * 100:.1f}% slower")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    from eagle_tree_choices import EAGLE_TREES

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # First validate bitmap metadata against index lists
        print("=" * 60)
        print("Validating bitmap metadata against index lists")
        print("=" * 60)
        for name in ["mc_sim_7b_63", "regular_512"]:
            tree = EAGLE_TREES[name]
            match = validate_bitmap_against_index_list(tree)
            print(f"  {name}: {'PASSED' if match else 'FAILED'}")

        # Then validate CUDA kernel output
        for name in ["mc_sim_7b_63", "regular_512"]:
            print(f"\n{'=' * 60}")
            print(f"Validating CUDA bitmap kernel: {name}")
            print(f"{'=' * 60}")
            validate_bitmap_kernel(EAGLE_TREES[name])

        # Then validate Triton kernel output
        for name in ["mc_sim_7b_63", "regular_512"]:
            print(f"\n{'=' * 60}")
            print(f"Validating Triton bitmap kernel: {name}")
            print(f"{'=' * 60}")
            validate_triton_bitmap(EAGLE_TREES[name])

    elif len(sys.argv) > 1 and sys.argv[1] == "bench":
        from eagle_tree_choices import regular_tree_512
        for past_len in [64, 256, 1024]:
            benchmark_bitmap_vs_others(regular_tree_512, past_len=past_len)

    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        for name in ["mc_sim_7b_63", "regular_512"]:
            print_bitmap_stats(EAGLE_TREES[name])

    else:
        print("Usage: python bitmap_sparse_attention.py [validate|bench|stats]")
