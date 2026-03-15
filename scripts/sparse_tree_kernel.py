"""
Sparse tree attention Triton kernel for Scenario A.

Based on ea_attn_exp's _attn_fwd_strided kernel, but Stage 2 uses
block-sparse iteration: only visits KV blocks that contain ancestors
of the current Q block, skipping all-zero blocks entirely.

For regular_512 (binary tree, depth 8, 511 nodes):
  - Dense Stage 2: 136 block loads total (causal triangular)
  - Sparse Stage 2: 71 block loads total
  - ~48% reduction in Stage 2 K/V loads and QK compute
"""

import torch
import triton
import triton.language as tl

from sparse_tree_utils import precompute_ancestor_indices, compute_dfs_permutation
from eagle_tree_choices import get_tree_stats


# ============================================================================
# HOST-SIDE: Precompute block-sparse metadata
# ============================================================================

def precompute_block_sparse_metadata(tree_choices, BLOCK_M=32, BLOCK_N=32):
    """Precompute which KV blocks each Q block needs to visit in Stage 2.

    For each Q block, finds the set of KV blocks containing at least one
    ancestor of any Q position in that block.

    Args:
        tree_choices: Sorted tree_choices list.
        BLOCK_M: Q block size (must match kernel).
        BLOCK_N: KV block size (must match kernel).

    Returns:
        block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32 tensor.
            KV block indices to visit for each Q block.
        block_counts: [num_q_blocks] int32 tensor.
            Number of valid entries per Q block.
        MAX_SPARSE_BLOCKS: Maximum number of blocks any Q block needs.
    """
    ancestor_indices, ancestor_mask, ancestor_counts = precompute_ancestor_indices(tree_choices)

    N_tree = len(tree_choices) + 1  # +1 for root
    num_q_blocks = (N_tree + BLOCK_M - 1) // BLOCK_M

    # For each Q block, find KV blocks containing ancestors
    block_lists = []
    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min(q_start + BLOCK_M, N_tree)

        kv_blocks_needed = set()
        for q_pos in range(q_start, q_end):
            n_anc = int(ancestor_counts[q_pos].item())
            for a in range(n_anc):
                anc_idx = int(ancestor_indices[q_pos, a].item())
                kv_blocks_needed.add(anc_idx // BLOCK_N)

        block_lists.append(sorted(kv_blocks_needed))

    MAX_SPARSE_BLOCKS = max(len(bl) for bl in block_lists)

    block_indices = torch.zeros(num_q_blocks, MAX_SPARSE_BLOCKS, dtype=torch.int32)
    block_counts = torch.zeros(num_q_blocks, dtype=torch.int32)

    for qb, bl in enumerate(block_lists):
        block_counts[qb] = len(bl)
        for i, kv_b in enumerate(bl):
            block_indices[qb, i] = kv_b

    return block_indices, block_counts, MAX_SPARSE_BLOCKS


def precompute_subtree_metadata(tree_choices, BLOCK_M=32, BLOCK_N=32):
    """Precompute block-sparse metadata using DFS permutation (subtree grouping).

    DFS reordering makes subtree nodes contiguous in both Q and KV dimensions.
    Nodes in the same Q block now share most ancestors, so their ancestor
    KV-block union is much tighter than with BFS ordering.

    The Triton kernel itself is unchanged — only the input ordering and
    block_indices/block_counts differ.

    Args:
        tree_choices: Sorted tree_choices list (BFS order).
        BLOCK_M: Q block size (must match kernel).
        BLOCK_N: KV block size (must match kernel).

    Returns:
        perm:           [N_tree] int64. perm[new_pos] = old_pos.
        inv_perm:       [N_tree] int64. inv_perm[old_pos] = new_pos.
        block_indices:  [num_q_blocks, MAX_SPARSE_BLOCKS] int32.
                        KV block indices (in DFS order) for each Q block.
        block_counts:   [num_q_blocks] int32.
        MAX_SPARSE_BLOCKS: int.
    """
    perm, inv_perm = compute_dfs_permutation(tree_choices)

    # Ancestor info in original (BFS) coordinates
    ancestor_indices, _, ancestor_counts = precompute_ancestor_indices(tree_choices)

    N_tree = len(tree_choices) + 1
    num_q_blocks = (N_tree + BLOCK_M - 1) // BLOCK_M

    block_lists = []
    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min(q_start + BLOCK_M, N_tree)

        kv_blocks_needed = set()
        for new_q_pos in range(q_start, q_end):
            # Map DFS Q position back to original position
            old_q_pos = int(perm[new_q_pos].item())
            n_anc = int(ancestor_counts[old_q_pos].item())
            for a in range(n_anc):
                # Ancestor in original coordinates → map to DFS KV position
                old_anc_pos = int(ancestor_indices[old_q_pos, a].item())
                new_anc_pos = int(inv_perm[old_anc_pos].item())
                kv_blocks_needed.add(new_anc_pos // BLOCK_N)

        block_lists.append(sorted(kv_blocks_needed))

    MAX_SPARSE_BLOCKS = max(len(bl) for bl in block_lists)

    block_indices = torch.zeros(num_q_blocks, MAX_SPARSE_BLOCKS, dtype=torch.int32)
    block_counts = torch.zeros(num_q_blocks, dtype=torch.int32)

    for qb, bl in enumerate(block_lists):
        block_counts[qb] = len(bl)
        for i, kv_b in enumerate(bl):
            block_indices[qb, i] = kv_b

    return perm, inv_perm, block_indices, block_counts, MAX_SPARSE_BLOCKS


# ============================================================================
# TRITON KERNEL: Sparse Stage 2 with block-level skipping
# ============================================================================

@triton.jit
def _sparse_stage2_inner(acc, l_i, m_i, q,
                         desc_k, desc_v, desc_tree_mask,
                         tree_mask_row_offset, dtype: tl.constexpr,
                         start_m, qk_scale,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                         BLOCK_N: tl.constexpr,
                         offs_m: tl.constexpr, offs_n: tl.constexpr,
                         N_CTX_Q: tl.constexpr, N_CTX_KV: tl.constexpr,
                         # Sparse block metadata
                         block_indices_ptr, block_count,
                         past_len,
                         MAX_SPARSE_BLOCKS: tl.constexpr):
    """Stage 2 inner loop: iterate only over ancestor KV blocks.

    Optimizations vs naive approach:
    1. Runtime loop bound (block_count) — zero empty iterations
    2. tl.load with runtime index — no constexpr guard needed
    """

    # Runtime-bounded loop: only iterate over actual ancestor blocks
    # No wasted iterations — if block_count=3, loop runs exactly 3 times
    for b_idx in tl.range(0, block_count):
        # Load KV block index from precomputed list
        kv_block_idx = tl.load(block_indices_ptr + b_idx)

        # Compute actual KV position
        start_n = past_len + kv_block_idx * BLOCK_N

        # Load K and compute QK
        k = desc_k.load([start_n, 0]).T
        qk = tl.dot(q, k)

        # Boundary mask
        boundary_mask = (start_n + offs_n[None, :]) < N_CTX_KV

        # Tree mask: load from the correct column position
        tree_col = kv_block_idx * BLOCK_N
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
    """Stage 1: dense attention to past context [0, past_len). No mask needed."""

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
def _attn_fwd_sparse(
    Q, K, V, O, M, Tree_mask,
    Block_indices, Block_counts,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_tz, stride_th, stride_tm, stride_tn,
    stride_bi_m,  # block_indices stride for Q block dim
    Z, H, N_CTX_Q, N_CTX_KV,
    TREE_MASK_DIM,  # Padded tree mask dimension (>= N_CTX_Q, aligned to 8)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_SPARSE_BLOCKS: tl.constexpr,
):
    """Sparse tree attention kernel.

    Stage 1: Dense attention to past context (same as ea_attn_exp).
    Stage 2: Block-sparse attention to tree region (only ancestor blocks).
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

    # ---- Stage 2: Sparse tree region ----
    # Load block count for this Q block
    block_count = tl.load(Block_counts + start_m)

    # Pointer to this Q block's list of KV block indices
    block_indices_ptr = Block_indices + start_m * stride_bi_m

    acc, l_i, m_i = _sparse_stage2_inner(
        acc, l_i, m_i, q, desc_k, desc_v, desc_tree_mask,
        tree_mask_row_offset, dtype, start_m, qk_scale,
        BLOCK_M, HEAD_DIM, BLOCK_N, offs_m, offs_n,
        N_CTX_Q, N_CTX_KV,
        block_indices_ptr, block_count, past_len,
        MAX_SPARSE_BLOCKS
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

def pad_tree_mask_for_tma(tree_mask, N_CTX_Q):
    """Pad tree_mask for TMA alignment on Hopper (one-time, cache the result).

    Args:
        tree_mask: [N_CTX_Q, N_CTX_Q] or [1, 1, N_CTX_Q, N_CTX_Q] fp16 tensor.
        N_CTX_Q: Actual (unpadded) tree size.

    Returns:
        (padded_mask, N_PAD): Padded tree_mask and padded dimension.
    """
    TMA_ALIGN = 8  # fp16: 8 elements = 16 bytes
    N_PAD = (N_CTX_Q + TMA_ALIGN - 1) // TMA_ALIGN * TMA_ALIGN
    if N_PAD != N_CTX_Q:
        pad_n = N_PAD - N_CTX_Q
        if tree_mask.dim() == 2:
            tree_mask = torch.nn.functional.pad(
                tree_mask, (0, pad_n, 0, pad_n), value=float('-inf')
            )
        else:
            tree_mask = torch.nn.functional.pad(
                tree_mask, (0, pad_n, 0, pad_n), value=float('-inf')
            )
    return tree_mask.contiguous(), N_PAD


class _attention_sparse(torch.autograd.Function):
    """Sparse tree attention with block-level KV skipping."""

    @staticmethod
    def forward(ctx, q, k, v, tree_mask, sm_scale,
                block_indices, block_counts, MAX_SPARSE_BLOCKS):
        """
        Args:
            q: [B, H, N_CTX_Q, D]
            k: [B, H, N_CTX_KV, D]
            v: [B, H, N_CTX_KV, D]
            tree_mask: [B, H, N_CTX_Q, N_CTX_Q] or broadcastable.
                       If already TMA-padded (last dim is multiple of 8),
                       skip padding for zero overhead.
            sm_scale: softmax scale
            block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32 (on device)
            block_counts: [num_q_blocks] int32 (on device)
            MAX_SPARSE_BLOCKS: compile-time constant
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

        # Pad for TMA alignment if needed (skip if already padded)
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

        # Ensure metadata on device
        if block_indices.device != q.device:
            block_indices = block_indices.to(device=q.device)
        if block_counts.device != q.device:
            block_counts = block_counts.to(device=q.device)

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
            _attn_fwd_sparse[grid](
                q, k, v, o, M, tree_mask,
                block_indices, block_counts,
                sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                tree_mask.stride(0), tree_mask.stride(1),
                tree_mask.stride(2), tree_mask.stride(3),
                block_indices.stride(0),  # stride_bi_m
                B, H, N_CTX_Q, N_CTX_KV,
                N_PAD,  # TREE_MASK_DIM
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                MAX_SPARSE_BLOCKS=MAX_SPARSE_BLOCKS,
            )
        finally:
            torch.cuda.set_device(prev_device)

        return o


class _attention_subtree(torch.autograd.Function):
    """Sparse tree attention with DFS subtree grouping.

    Permutes Q and KV tree region into DFS order so that subtree nodes are
    contiguous. This tightens per-Q-block ancestor unions, reducing the number
    of KV blocks visited in Stage 2.

    Reuses the same _attn_fwd_sparse kernel — only the data layout changes.
    """

    @staticmethod
    def forward(ctx, q, k, v, tree_mask, sm_scale,
                block_indices, block_counts, MAX_SPARSE_BLOCKS,
                perm, inv_perm):
        """
        Args:
            q: [B, H, N_CTX_Q, D]
            k: [B, H, N_CTX_KV, D]  (past_context ++ tree_region)
            v: [B, H, N_CTX_KV, D]
            tree_mask: [B, H, N_CTX_Q, N_CTX_Q] or broadcastable.
            sm_scale: softmax scale
            block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32 (DFS order)
            block_counts: [num_q_blocks] int32
            MAX_SPARSE_BLOCKS: compile-time constant
            perm:     [N_CTX_Q] int64. perm[new] = old.
            inv_perm: [N_CTX_Q] int64. inv_perm[old] = new.
        """
        B, H, N_CTX_Q, HEAD_DIM = q.shape
        N_CTX_KV = k.shape[2]
        past_len = N_CTX_KV - N_CTX_Q

        # Move permutation tensors to device
        perm = perm.to(q.device)
        inv_perm = inv_perm.to(q.device)

        # ---- Permute Q into DFS order ----
        q = q[:, :, perm, :].contiguous()

        # ---- Permute KV tree region into DFS order (past context unchanged) ----
        k_past = k[:, :, :past_len, :]
        k_tree = k[:, :, past_len:, :][:, :, perm, :]
        k = torch.cat([k_past, k_tree], dim=2).contiguous()

        v_past = v[:, :, :past_len, :]
        v_tree = v[:, :, past_len:, :][:, :, perm, :]
        v = torch.cat([v_past, v_tree], dim=2).contiguous()

        # ---- Permute tree_mask rows (Q dim) and cols (KV dim) ----
        if tree_mask is not None:
            if tree_mask.shape[0] == 1 and B > 1:
                tree_mask = tree_mask.expand(B, -1, -1, -1)
            if tree_mask.shape[1] == 1 and H > 1:
                tree_mask = tree_mask.expand(-1, H, -1, -1)
            tree_mask = tree_mask[:, :, perm, :][:, :, :, perm]
            tree_mask = tree_mask.contiguous().to(q.dtype)
        else:
            tree_mask = torch.zeros((B, H, N_CTX_Q, N_CTX_Q),
                                    device=q.device, dtype=q.dtype)

        # ---- TMA padding (same as _attention_sparse) ----
        TMA_ALIGN = 8
        mask_dim = tree_mask.shape[-1]
        N_PAD = (N_CTX_Q + TMA_ALIGN - 1) // TMA_ALIGN * TMA_ALIGN
        if mask_dim < N_PAD:
            pad_n = N_PAD - mask_dim
            tree_mask = torch.nn.functional.pad(
                tree_mask, (0, pad_n, 0, pad_n), value=float('-inf')
            ).contiguous()
        elif mask_dim >= N_PAD:
            N_PAD = mask_dim

        # Ensure metadata on device
        if block_indices.device != q.device:
            block_indices = block_indices.to(device=q.device)
        if block_counts.device != q.device:
            block_counts = block_counts.to(device=q.device)

        # Output (in DFS order)
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
            _attn_fwd_sparse[grid](
                q, k, v, o, M, tree_mask,
                block_indices, block_counts,
                sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                tree_mask.stride(0), tree_mask.stride(1),
                tree_mask.stride(2), tree_mask.stride(3),
                block_indices.stride(0),  # stride_bi_m
                B, H, N_CTX_Q, N_CTX_KV,
                N_PAD,  # TREE_MASK_DIM
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                MAX_SPARSE_BLOCKS=MAX_SPARSE_BLOCKS,
            )
        finally:
            torch.cuda.set_device(prev_device)

        # ---- Inverse-permute output back to original order ----
        o = o[:, :, inv_perm, :].contiguous()

        return o


# ============================================================================
# VALIDATION
# ============================================================================

def _ensure_triton_allocator():
    """Set Triton memory allocator if not already set."""
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    triton.set_allocator(alloc_fn)


def validate_sparse_kernel(tree_choices, past_len=64, B=2, H=4, D=128,
                           device='cuda'):
    """Validate sparse Triton kernel against dense ea_attn_exp kernel."""
    _ensure_triton_allocator()
    from eagle_tree_attention import _attention_strided

    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree

    # Precompute metadata
    block_indices, block_counts, MAX_SPARSE_BLOCKS = \
        precompute_block_sparse_metadata(tree_choices)

    torch.manual_seed(42)
    q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    scale = 1.0 / (D ** 0.5)

    # Build dense tree mask
    from sparse_tree_utils import precompute_ancestor_indices
    node_to_idx = {(): 0}
    for i, node in enumerate(tree_choices):
        node_to_idx[node] = i + 1

    tree_mask = torch.full((N_tree, N_tree), float('-inf'),
                           device=device, dtype=torch.float16)
    tree_mask[0, 0] = 0.0
    for i, node in enumerate(tree_choices):
        idx = i + 1
        tree_mask[idx, 0] = 0.0
        for prefix_len in range(1, len(node)):
            anc_idx = node_to_idx[node[:prefix_len]]
            tree_mask[idx, anc_idx] = 0.0
        tree_mask[idx, idx] = 0.0

    tree_mask_4d = tree_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

    # Dense (ea_attn_exp strided)
    out_dense = _attention_strided.apply(q, k, v, tree_mask_4d, scale)

    # Sparse (our new kernel)
    out_sparse = _attention_sparse.apply(q, k, v, tree_mask_4d, scale,
                                         block_indices, block_counts,
                                         MAX_SPARSE_BLOCKS)

    max_diff = (out_sparse - out_dense).abs().max().item()
    mean_diff = (out_sparse - out_dense).abs().mean().item()

    stats = get_tree_stats(tree_choices)
    print(f"Tree: {stats['total_nodes']} nodes, depth {stats['max_depth']}")
    print(f"Past context: {past_len}, MAX_SPARSE_BLOCKS: {MAX_SPARSE_BLOCKS}")
    print(f"Max absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    if max_diff < 1e-2:
        print("PASSED: Sparse kernel matches dense kernel")
    else:
        print("FAILED: Outputs differ significantly")

    return max_diff


def validate_subtree_kernel(tree_choices, past_len=64, B=2, H=4, D=128,
                             device='cuda'):
    """Validate subtree (DFS) kernel against dense ea_attn_exp kernel."""
    _ensure_triton_allocator()
    from eagle_tree_attention import _attention_strided

    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree

    # Precompute subtree metadata
    perm, inv_perm, block_indices, block_counts, MAX_SPARSE_BLOCKS = \
        precompute_subtree_metadata(tree_choices)

    torch.manual_seed(42)
    q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    scale = 1.0 / (D ** 0.5)

    # Build dense tree mask (original BFS order)
    node_to_idx = {(): 0}
    for i, node in enumerate(tree_choices):
        node_to_idx[node] = i + 1

    tree_mask = torch.full((N_tree, N_tree), float('-inf'),
                           device=device, dtype=torch.float16)
    tree_mask[0, 0] = 0.0
    for i, node in enumerate(tree_choices):
        idx = i + 1
        tree_mask[idx, 0] = 0.0
        for prefix_len in range(1, len(node)):
            anc_idx = node_to_idx[node[:prefix_len]]
            tree_mask[idx, anc_idx] = 0.0
        tree_mask[idx, idx] = 0.0

    tree_mask_4d = tree_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

    # Dense reference (original order)
    out_dense = _attention_strided.apply(q, k, v, tree_mask_4d, scale)

    # Subtree (DFS permuted, same kernel)
    out_subtree = _attention_subtree.apply(q, k, v, tree_mask_4d, scale,
                                           block_indices, block_counts,
                                           MAX_SPARSE_BLOCKS,
                                           perm, inv_perm)

    max_diff = (out_subtree - out_dense).abs().max().item()
    mean_diff = (out_subtree - out_dense).abs().mean().item()

    stats = get_tree_stats(tree_choices)
    print(f"Tree: {stats['total_nodes']} nodes, depth {stats['max_depth']}")
    print(f"Past context: {past_len}, MAX_SPARSE_BLOCKS: {MAX_SPARSE_BLOCKS}")
    print(f"Max absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    if max_diff < 1e-2:
        print("PASSED: Subtree kernel matches dense kernel")
    else:
        print("FAILED: Outputs differ significantly")

    return max_diff


def _build_tree_mask(tree_choices, N_tree, device, dtype=torch.float16):
    """Build tree attention mask (shared helper for benchmarks)."""
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


def benchmark_sparse_vs_dense(tree_choices, past_len=256, B=4, H=8, D=128,
                               device='cuda', warmup=10, rep=100):
    """Microbenchmark dense vs sparse vs subtree Stage 2."""
    _ensure_triton_allocator()
    from eagle_tree_attention import _attention_strided
    import time

    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree
    scale = 1.0 / (D ** 0.5)

    # Precompute metadata for both sparse variants
    block_indices_bfs, block_counts_bfs, MAX_BFS = \
        precompute_block_sparse_metadata(tree_choices)
    perm, inv_perm, block_indices_dfs, block_counts_dfs, MAX_DFS = \
        precompute_subtree_metadata(tree_choices)

    q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)

    tree_mask_4d = _build_tree_mask(tree_choices, N_tree, device)

    # Warmup all three
    for _ in range(warmup):
        _attention_strided.apply(q, k, v, tree_mask_4d, scale)
        _attention_sparse.apply(q, k, v, tree_mask_4d, scale,
                                block_indices_bfs, block_counts_bfs, MAX_BFS)
        _attention_subtree.apply(q, k, v, tree_mask_4d, scale,
                                 block_indices_dfs, block_counts_dfs, MAX_DFS,
                                 perm, inv_perm)
    torch.cuda.synchronize()

    # Benchmark dense
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        _attention_strided.apply(q, k, v, tree_mask_4d, scale)
    torch.cuda.synchronize()
    t_dense = (time.perf_counter() - t0) / rep * 1000

    # Benchmark sparse (BFS)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        _attention_sparse.apply(q, k, v, tree_mask_4d, scale,
                                block_indices_bfs, block_counts_bfs, MAX_BFS)
    torch.cuda.synchronize()
    t_sparse = (time.perf_counter() - t0) / rep * 1000

    # Benchmark subtree (DFS)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        _attention_subtree.apply(q, k, v, tree_mask_4d, scale,
                                 block_indices_dfs, block_counts_dfs, MAX_DFS,
                                 perm, inv_perm)
    torch.cuda.synchronize()
    t_subtree = (time.perf_counter() - t0) / rep * 1000

    stats = get_tree_stats(tree_choices)
    total_bfs = int(block_counts_bfs.sum().item())
    total_dfs = int(block_counts_dfs.sum().item())
    print(f"\nTree: {stats['total_nodes']} nodes, depth {stats['max_depth']}")
    print(f"Past context: {past_len}, B={B}, H={H}, D={D}")
    print(f"Stage 2 loads: BFS={total_bfs}, DFS={total_dfs} ({(1-total_dfs/total_bfs)*100:.0f}% fewer)")
    print(f"Dense (ea_attn_exp):   {t_dense:.3f} ms")
    print(f"Sparse (BFS):          {t_sparse:.3f} ms  ({t_dense/t_sparse:.2f}x vs dense)")
    print(f"Subtree (DFS):         {t_subtree:.3f} ms  ({t_dense/t_subtree:.2f}x vs dense)")
    if t_subtree < t_sparse:
        print(f"Subtree vs Sparse:     {(1-t_subtree/t_sparse)*100:.1f}% faster")
    else:
        print(f"Subtree vs Sparse:     {(t_subtree/t_sparse-1)*100:.1f}% slower (permutation overhead)")


if __name__ == "__main__":
    import sys
    from eagle_tree_choices import regular_tree_512, mc_sim_7b_63, EAGLE_TREES

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        for name in ["mc_sim_7b_63", "regular_512"]:
            print(f"\n{'='*60}")
            print(f"Validating sparse (BFS): {name}")
            print(f"{'='*60}")
            validate_sparse_kernel(EAGLE_TREES[name])

        for name in ["mc_sim_7b_63", "regular_512"]:
            print(f"\n{'='*60}")
            print(f"Validating subtree (DFS): {name}")
            print(f"{'='*60}")
            validate_subtree_kernel(EAGLE_TREES[name])

    elif len(sys.argv) > 1 and sys.argv[1] == "bench":
        for past_len in [64, 256, 1024]:
            benchmark_sparse_vs_dense(regular_tree_512, past_len=past_len)

    else:
        print("Usage: python sparse_tree_kernel.py [validate|bench]")
