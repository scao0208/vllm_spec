"""
Unified Sparse Tree Attention.

Treats past context + draft tree as one unified causal tree.
Selects top-k important KV positions, gathers them, and computes
attention using a block-sparse Triton kernel with a pruned tree mask.

Key difference from sparse_tree_kernel.py:
- No Stage 1/Stage 2 split — single-stage block-sparse attention
- KV is gathered from a sparse subset of the full cache
- Mask covers ALL positions (past + tree) uniformly
- Relaxed pruning: gaps in ancestor chain are allowed; queries attend
  to whichever selected KV positions are their ancestors in the
  original combined tree. Missing intermediaries are simply skipped.

Usage:
    python unified_sparse_attention.py validate   # correctness check
    python unified_sparse_attention.py bench      # microbenchmark
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sparse_tree_utils import precompute_ancestor_indices
from eagle_tree_choices import get_tree_stats


# ============================================================================
# HOST-SIDE UTILITIES
# ============================================================================

def build_combined_ancestor_mask(tree_choices, past_len):
    """Build full causal mask for the combined past-chain + draft-tree.

    The combined structure is a tree where:
    - Positions [0, past_len) form a chain (each sees all predecessors)
    - Positions [past_len, past_len + N_tree) follow the draft tree

    Returns:
        mask: [N_tree, past_len + N_tree] bool.
              mask[i, j] = True iff query i can attend to KV j.
    """
    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree

    mask = torch.zeros(N_tree, N_KV, dtype=torch.bool)

    # All tree queries can attend to all past positions
    mask[:, :past_len] = True

    # Tree region: ancestor structure
    node_to_idx = {(): 0}
    for i, node in enumerate(tree_choices):
        node_to_idx[node] = i + 1

    # Root attends to itself
    mask[0, past_len] = True

    for i, node in enumerate(tree_choices):
        idx = i + 1
        mask[idx, past_len] = True  # root
        for prefix_len in range(1, len(node)):
            anc_idx = node_to_idx[node[:prefix_len]]
            mask[idx, past_len + anc_idx] = True
        mask[idx, past_len + idx] = True  # self

    return mask


def oracle_importance_scores(q, k, sm_scale, combined_mask):
    """Compute oracle importance: mean attention weight per KV position.

    Args:
        q: [B, H, N_Q, D]
        k: [B, H, N_KV, D]
        sm_scale: softmax scale
        combined_mask: [N_Q, N_KV] bool

    Returns:
        importance: [N_KV] float32 on CPU
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    attn_mask = torch.where(combined_mask, 0.0, float('-inf')).to(q.dtype)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).to(q.device)
    scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)  # [B, H, N_Q, N_KV]
    importance = weights.mean(dim=(0, 1, 2)).cpu()  # [N_KV]
    return importance


def select_important_kv(importance, top_k, always_keep=None):
    """Select top-k KV positions by importance score, sorted.

    Args:
        importance: [N_KV] importance scores
        top_k: number of KV to select
        always_keep: optional 1-D tensor of indices to always include

    Returns:
        selected_idx: [K] sorted int64 tensor (K <= top_k)
    """
    if always_keep is not None:
        reserved = set(always_keep.tolist())
        mask = torch.ones(importance.shape[0], dtype=torch.bool)
        mask[always_keep] = False
        remaining = importance.clone()
        remaining[~mask] = -float('inf')
        k_extra = min(top_k - len(reserved), int(mask.sum().item()))
        if k_extra > 0:
            _, top_idx = remaining.topk(k_extra)
            selected = torch.cat([always_keep, top_idx])
        else:
            selected = always_keep
    else:
        k = min(top_k, importance.shape[0])
        _, selected = importance.topk(k)

    return selected.sort().values.to(torch.int64)


def build_pruned_mask(selected_idx, combined_mask):
    """Build pruned attention mask for selected KV positions.

    Args:
        selected_idx: [K] sorted global KV indices (CPU)
        combined_mask: [N_Q, N_KV] bool (CPU)

    Returns:
        pruned_mask: [N_Q, K] float16 (0 or -inf), CPU
    """
    bool_mask = combined_mask[:, selected_idx]
    return torch.where(bool_mask, 0.0, float('-inf')).to(torch.float16)


def precompute_unified_block_metadata(pruned_mask, BLOCK_M=32, BLOCK_N=32):
    """Precompute block-sparse metadata from pruned mask.

    For each Q block, identifies which KV blocks have at least one
    non-masked entry (value > -inf).

    Returns:
        block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32
        block_counts:  [num_q_blocks] int32
        MAX_SPARSE_BLOCKS: int
    """
    N_Q, N_KV = pruned_mask.shape
    num_q_blocks = (N_Q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (N_KV + BLOCK_N - 1) // BLOCK_N

    block_lists = []
    for qb in range(num_q_blocks):
        qs = qb * BLOCK_M
        qe = min(qs + BLOCK_M, N_Q)
        needed = []
        for kb in range(num_kv_blocks):
            ks = kb * BLOCK_N
            ke = min(ks + BLOCK_N, N_KV)
            block = pruned_mask[qs:qe, ks:ke]
            if (block > float('-inf')).any():
                needed.append(kb)
        block_lists.append(needed)

    MAX_SPARSE_BLOCKS = max((len(bl) for bl in block_lists), default=1)

    block_indices = torch.zeros(num_q_blocks, MAX_SPARSE_BLOCKS, dtype=torch.int32)
    block_counts = torch.zeros(num_q_blocks, dtype=torch.int32)
    for qb, bl in enumerate(block_lists):
        block_counts[qb] = len(bl)
        for i, kb in enumerate(bl):
            block_indices[qb, i] = kb

    return block_indices, block_counts, MAX_SPARSE_BLOCKS


def precompute_unified_block_metadata_gpu(pruned_mask, BLOCK_M=32, BLOCK_N=32):
    """GPU-based block metadata computation (no CPU round-trip).

    Same semantics as precompute_unified_block_metadata but operates entirely
    on GPU tensors, avoiding the .cpu() transfer that blocks the CUDA stream
    and can cause NCCL timeouts with large contexts.

    Args:
        pruned_mask: [N_Q, N_KV] float tensor on GPU (0 or -inf).

    Returns:
        block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32 on same device
        block_counts:  [num_q_blocks] int32 on same device
        MAX_SPARSE_BLOCKS: int
    """
    N_Q, N_KV = pruned_mask.shape
    num_q_blocks = (N_Q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (N_KV + BLOCK_N - 1) // BLOCK_N

    # Pad mask to exact block boundaries
    pad_q = num_q_blocks * BLOCK_M - N_Q
    pad_kv = num_kv_blocks * BLOCK_N - N_KV
    if pad_q > 0 or pad_kv > 0:
        mask_padded = F.pad(pruned_mask, (0, pad_kv, 0, pad_q), value=float('-inf'))
    else:
        mask_padded = pruned_mask

    # Reshape into blocks: [num_q_blocks, BLOCK_M, num_kv_blocks, BLOCK_N]
    mask_blocked = mask_padded.reshape(num_q_blocks, BLOCK_M, num_kv_blocks, BLOCK_N)

    # Per-block: has at least one non-masked entry? [num_q_blocks, num_kv_blocks]
    block_active = (mask_blocked > float('-inf')).any(dim=3).any(dim=1)

    # Count active KV blocks per Q block
    block_counts = block_active.sum(dim=1).to(torch.int32)  # [num_q_blocks]
    MAX_SPARSE_BLOCKS = max(int(block_counts.max().item()), 1)

    # Build block_indices: stable-sort so active blocks come first in index order
    sort_key = (~block_active).to(torch.int32)
    _, sorted_idx = sort_key.sort(dim=1, stable=True)
    block_indices = sorted_idx[:, :MAX_SPARSE_BLOCKS].to(torch.int32).contiguous()

    return block_indices, block_counts, MAX_SPARSE_BLOCKS


# ============================================================================
# TRITON KERNEL — Single-stage block-sparse masked attention
# ============================================================================

@triton.jit
def _unified_inner(acc, l_i, m_i, q,
                   desc_k, desc_v, desc_mask,
                   mask_row_offset, dtype: tl.constexpr,
                   qk_scale,
                   BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                   BLOCK_N: tl.constexpr,
                   offs_n: tl.constexpr,
                   N_CTX_KV: tl.constexpr,
                   block_indices_ptr, block_count,
                   MAX_SPARSE_BLOCKS: tl.constexpr):
    """Block-sparse attention inner loop (no Stage 1/2 split)."""

    for b_idx in tl.range(0, block_count):
        kv_block_idx = tl.load(block_indices_ptr + b_idx)
        start_n = kv_block_idx * BLOCK_N

        k = desc_k.load([start_n, 0]).T
        qk = tl.dot(q, k)

        boundary_mask = (start_n + offs_n[None, :]) < N_CTX_KV
        mask_col = kv_block_idx * BLOCK_N
        mask_block = desc_mask.load([mask_row_offset, mask_col])
        qk = qk * qk_scale + mask_block + tl.where(boundary_mask, 0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

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
def _attn_fwd_unified(
    Q, K, V, O, LSE, Attn_mask,
    Block_indices, Block_counts,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_mz, stride_mh, stride_mm, stride_mn,
    stride_bi_m,
    Z, H, N_CTX_Q, N_CTX_KV,
    MASK_DIM_Q,   # padded Q dim of mask
    MASK_DIM_KV,  # padded KV dim of mask
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_SPARSE_BLOCKS: tl.constexpr,
):
    """Unified sparse attention kernel — single stage, block-sparse."""
    dtype = tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qk_scale = sm_scale * 1.44269504

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    o_base = O + off_z * stride_oz + off_h * stride_oh
    m_base = Attn_mask + off_z * stride_mz + off_h * stride_mh

    desc_q = tl.make_tensor_descriptor(
        q_base, shape=[N_CTX_Q, HEAD_DIM], strides=[stride_qm, stride_qk],
        block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(
        k_base, shape=[N_CTX_KV, HEAD_DIM], strides=[stride_kn, stride_kk],
        block_shape=[BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(
        v_base, shape=[N_CTX_KV, HEAD_DIM], strides=[stride_vn, stride_vk],
        block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(
        o_base, shape=[N_CTX_Q, HEAD_DIM], strides=[stride_om, stride_ok],
        block_shape=[BLOCK_M, HEAD_DIM])
    desc_mask = tl.make_tensor_descriptor(
        m_base, shape=[MASK_DIM_Q, MASK_DIM_KV], strides=[stride_mm, stride_mn],
        block_shape=[BLOCK_M, BLOCK_N])

    qo_offset = start_m * BLOCK_M
    q = desc_q.load([qo_offset, 0])

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q_valid_mask = offs_m < N_CTX_Q

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    mask_row_offset = start_m * BLOCK_M
    block_count = tl.load(Block_counts + start_m)
    block_indices_ptr = Block_indices + start_m * stride_bi_m

    acc, l_i, m_i = _unified_inner(
        acc, l_i, m_i, q, desc_k, desc_v, desc_mask,
        mask_row_offset, dtype, qk_scale,
        BLOCK_M, HEAD_DIM, BLOCK_N, offs_n,
        N_CTX_KV,
        block_indices_ptr, block_count,
        MAX_SPARSE_BLOCKS)

    # Epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    lse_ptrs = LSE + off_hz * N_CTX_Q + offs_m
    tl.store(lse_ptrs, m_i, mask=q_valid_mask)

    out_mask = q_valid_mask[:, None]
    acc_masked = tl.where(out_mask, acc.to(dtype), 0.0)
    desc_o.store([qo_offset, 0], acc_masked)


# ============================================================================
# AUTOGRAD WRAPPER
# ============================================================================

class _attention_unified_sparse(torch.autograd.Function):
    """Unified sparse attention: gathered KV + block-sparse mask."""

    @staticmethod
    def forward(ctx, q, k_sel, v_sel, mask_4d, sm_scale,
                block_indices, block_counts, MAX_SPARSE_BLOCKS):
        """
        Args:
            q:       [B, H, N_Q, D]
            k_sel:   [B, H, N_sel, D]   (gathered selected KV)
            v_sel:   [B, H, N_sel, D]
            mask_4d: [B|1, H|1, N_Q, N_sel] fp16 (0 or -inf)
            sm_scale: softmax scale
            block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32
            block_counts:  [num_q_blocks] int32
            MAX_SPARSE_BLOCKS: int (compile-time constant)
        """
        B, H, N_Q, HEAD_DIM = q.shape
        N_KV = k_sel.shape[2]

        q = q.contiguous()
        k_sel = k_sel.contiguous()
        v_sel = v_sel.contiguous()

        # Expand mask
        if mask_4d.shape[0] == 1 and B > 1:
            mask_4d = mask_4d.expand(B, -1, -1, -1)
        if mask_4d.shape[1] == 1 and H > 1:
            mask_4d = mask_4d.expand(-1, H, -1, -1)
        mask_4d = mask_4d.contiguous().to(q.dtype)

        # TMA pad mask so last two dims are multiples of 8
        TMA_ALIGN = 8
        N_Q_PAD = (N_Q + TMA_ALIGN - 1) // TMA_ALIGN * TMA_ALIGN
        N_KV_PAD = (N_KV + TMA_ALIGN - 1) // TMA_ALIGN * TMA_ALIGN
        cur_q = mask_4d.shape[2]
        cur_kv = mask_4d.shape[3]
        pad_kv = max(N_KV_PAD - cur_kv, 0)
        pad_q = max(N_Q_PAD - cur_q, 0)
        if pad_kv > 0 or pad_q > 0:
            mask_4d = F.pad(mask_4d, (0, pad_kv, 0, pad_q),
                            value=float('-inf')).contiguous()
        MASK_DIM_Q = mask_4d.shape[2]
        MASK_DIM_KV = mask_4d.shape[3]

        # Device
        if block_indices.device != q.device:
            block_indices = block_indices.to(device=q.device)
        if block_counts.device != q.device:
            block_counts = block_counts.to(device=q.device)

        o = torch.empty_like(q)
        LSE = torch.empty((B, H, N_Q), device=q.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(N_Q, BLOCK_M), B * H)

        device_idx = q.device.index if q.device.index is not None else 0
        prev_device = torch.cuda.current_device()
        torch.cuda.set_device(device_idx)

        try:
            _attn_fwd_unified[grid](
                q, k_sel, v_sel, o, LSE, mask_4d,
                block_indices, block_counts,
                sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k_sel.stride(0), k_sel.stride(1), k_sel.stride(2), k_sel.stride(3),
                v_sel.stride(0), v_sel.stride(1), v_sel.stride(2), v_sel.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                mask_4d.stride(0), mask_4d.stride(1),
                mask_4d.stride(2), mask_4d.stride(3),
                block_indices.stride(0),
                B, H, N_Q, N_KV,
                MASK_DIM_Q, MASK_DIM_KV,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                MAX_SPARSE_BLOCKS=MAX_SPARSE_BLOCKS,
            )
        finally:
            torch.cuda.set_device(prev_device)

        return o


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def _ensure_triton_allocator(device='cuda'):
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device=device)
    triton.set_allocator(alloc_fn)


def unified_sparse_attention(q, k_full, v_full, tree_choices, past_len,
                              sm_scale, top_k_ratio=0.3, importance=None):
    """Unified sparse tree attention.

    Selects top-k important KV from the full cache (past + tree),
    gathers them, builds a pruned ancestor mask, and computes attention
    via a single-stage block-sparse Triton kernel.

    Args:
        q:            [B, H, N_tree, D]
        k_full:       [B, H, N_KV, D]   (N_KV = past_len + N_tree)
        v_full:       [B, H, N_KV, D]
        tree_choices: tree structure (list of tuples)
        past_len:     number of past context tokens
        sm_scale:     softmax scale
        top_k_ratio:  fraction of KV to keep (0-1)
        importance:   [N_KV] pre-computed scores, or None for oracle

    Returns:
        output:       [B, H, N_tree, D]
        selected_idx: [K] selected KV indices
    """
    B, H, N_Q, D = q.shape
    N_KV = k_full.shape[2]
    N_tree = len(tree_choices) + 1
    assert N_Q == N_tree

    top_k = max(1, int(N_KV * top_k_ratio))

    # Combined ancestor mask (CPU)
    combined_mask = build_combined_ancestor_mask(tree_choices, past_len)

    # Importance scoring
    if importance is None:
        importance = oracle_importance_scores(q, k_full, sm_scale, combined_mask)

    # Select
    selected_idx = select_important_kv(importance, top_k)
    N_sel = selected_idx.shape[0]

    # Gather KV
    idx_dev = selected_idx.to(q.device)
    idx_exp = idx_dev.view(1, 1, N_sel, 1).expand(B, H, -1, D)
    k_sel = torch.gather(k_full, 2, idx_exp).contiguous()
    v_sel = torch.gather(v_full, 2, idx_exp).contiguous()

    # Pruned mask [N_Q, N_sel] → [1, 1, N_Q, N_sel]
    pruned_mask = build_pruned_mask(selected_idx, combined_mask)
    mask_4d = pruned_mask.unsqueeze(0).unsqueeze(0).to(q.device)

    # Block metadata
    block_indices, block_counts, MAX_SPARSE_BLOCKS = \
        precompute_unified_block_metadata(pruned_mask)

    output = _attention_unified_sparse.apply(
        q, k_sel, v_sel, mask_4d, sm_scale,
        block_indices, block_counts, MAX_SPARSE_BLOCKS)

    return output, selected_idx


# ============================================================================
# VALIDATION
# ============================================================================

def validate_unified_sparse(tree_choices, past_len=256, B=2, H=4, D=128,
                             device='cuda'):
    """Validate unified sparse attention against dense reference."""
    _ensure_triton_allocator(device)
    from sparse_tree_utils import dense_tree_attention_reference

    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree
    scale = 1.0 / (D ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)

    # Dense reference
    out_dense = dense_tree_attention_reference(q, k, v, past_len, tree_choices,
                                               scale)

    # top_k_ratio=1.0 → keep all KV (should match dense)
    print("=== Correctness: top_k_ratio=1.0 (keep all KV) ===")
    out_full, sel_full = unified_sparse_attention(
        q, k, v, tree_choices, past_len, scale, top_k_ratio=1.0)
    max_diff = (out_full - out_dense).abs().max().item()
    mean_diff = (out_full - out_dense).abs().mean().item()
    print(f"  Selected: {len(sel_full)}/{N_KV}")
    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  {'PASSED' if max_diff < 1e-2 else 'FAILED'}")

    # Various pruning ratios
    for ratio in [0.5, 0.3, 0.1]:
        print(f"\n=== Quality: top_k_ratio={ratio} ===")
        out_pruned, sel = unified_sparse_attention(
            q, k, v, tree_choices, past_len, scale, top_k_ratio=ratio)
        max_diff = (out_pruned - out_dense).abs().max().item()
        mean_diff = (out_pruned - out_dense).abs().mean().item()
        cos_sim = F.cosine_similarity(
            out_pruned.float().flatten(), out_dense.float().flatten(), dim=0
        ).item()
        print(f"  Selected: {len(sel)}/{N_KV}")
        print(f"  Max diff:  {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")
        print(f"  Cos sim:   {cos_sim:.6f}")

    return max_diff


def benchmark_unified_sparse(tree_choices, past_len=1024, B=4, H=8, D=128,
                              device='cuda', warmup=20, rep=100):
    """Microbenchmark unified sparse vs dense attention."""
    _ensure_triton_allocator(device)
    from eagle_tree_attention import _attention_strided
    from sparse_tree_kernel import _build_tree_mask, pad_tree_mask_for_tma
    import time

    N_tree = len(tree_choices) + 1
    N_KV = past_len + N_tree
    scale = 1.0 / (D ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(B, H, N_tree, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=torch.float16, device=device)

    tree_mask_4d = _build_tree_mask(tree_choices, N_tree, device)
    # Pad tree mask for TMA alignment (row stride must be 16-byte multiple)
    padded_2d, _ = pad_tree_mask_for_tma(tree_mask_4d.squeeze(0).squeeze(0),
                                          N_tree)
    tree_mask_4d = padded_2d.unsqueeze(0).unsqueeze(0)

    stats = get_tree_stats(tree_choices)
    print(f"\nTree: {stats['total_nodes']} nodes, depth {stats['max_depth']}")
    print(f"Past context: {past_len}, B={B}, H={H}, D={D}")
    print(f"Total KV: {N_KV}")

    # ----- Dense baseline -----
    for _ in range(warmup):
        _attention_strided.apply(q, k, v, tree_mask_4d, scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        _attention_strided.apply(q, k, v, tree_mask_4d, scale)
    torch.cuda.synchronize()
    t_dense = (time.perf_counter() - t0) / rep * 1000
    print(f"\nDense (ea_attn_exp):  {t_dense:.3f} ms")

    # ----- Unified sparse at various ratios -----
    combined_mask = build_combined_ancestor_mask(tree_choices, past_len)
    importance = oracle_importance_scores(q, k, scale, combined_mask)

    for ratio in [0.5, 0.3, 0.1]:
        top_k = max(1, int(N_KV * ratio))
        selected_idx = select_important_kv(importance, top_k)
        N_sel = selected_idx.shape[0]

        # Pre-gather
        idx_dev = selected_idx.to(device)
        idx_exp = idx_dev.view(1, 1, N_sel, 1).expand(B, H, -1, D)
        k_sel = torch.gather(k, 2, idx_exp).contiguous()
        v_sel = torch.gather(v, 2, idx_exp).contiguous()

        pruned_mask = build_pruned_mask(selected_idx, combined_mask)
        mask_4d = pruned_mask.unsqueeze(0).unsqueeze(0).to(device)
        block_indices, block_counts, MAX_SPARSE_BLOCKS = \
            precompute_unified_block_metadata(pruned_mask)

        # Block stats
        total_blocks = int(block_counts.sum().item())
        num_q_blocks = block_counts.shape[0]
        dense_blocks = num_q_blocks * ((N_sel + 31) // 32)

        # Warmup
        for _ in range(warmup):
            _attention_unified_sparse.apply(
                q, k_sel, v_sel, mask_4d, scale,
                block_indices, block_counts, MAX_SPARSE_BLOCKS)
        torch.cuda.synchronize()

        # Kernel only
        t0 = time.perf_counter()
        for _ in range(rep):
            _attention_unified_sparse.apply(
                q, k_sel, v_sel, mask_4d, scale,
                block_indices, block_counts, MAX_SPARSE_BLOCKS)
        torch.cuda.synchronize()
        t_kernel = (time.perf_counter() - t0) / rep * 1000

        # Kernel + gather
        t0 = time.perf_counter()
        for _ in range(rep):
            k_g = torch.gather(k, 2, idx_exp).contiguous()
            v_g = torch.gather(v, 2, idx_exp).contiguous()
            _attention_unified_sparse.apply(
                q, k_g, v_g, mask_4d, scale,
                block_indices, block_counts, MAX_SPARSE_BLOCKS)
        torch.cuda.synchronize()
        t_with_gather = (time.perf_counter() - t0) / rep * 1000

        print(f"\n  ratio={ratio} (KV: {N_sel}/{N_KV}, "
              f"blocks: {total_blocks}/{dense_blocks}):")
        print(f"    Kernel only:  {t_kernel:.3f} ms "
              f"({t_dense/t_kernel:.2f}x vs dense)")
        print(f"    With gather:  {t_with_gather:.3f} ms "
              f"({t_dense/t_with_gather:.2f}x vs dense)")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    from eagle_tree_choices import EAGLE_TREES, regular_tree_512, mc_sim_7b_63

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        for name in ["mc_sim_7b_63", "regular_512"]:
            print(f"\n{'='*60}")
            print(f"Validating unified sparse: {name}")
            print(f"{'='*60}")
            validate_unified_sparse(EAGLE_TREES[name])

    elif len(sys.argv) > 1 and sys.argv[1] == "bench":
        for past_len in [256, 1024, 4096]:
            benchmark_unified_sparse(regular_tree_512, past_len=past_len)

    else:
        print("Usage: python unified_sparse_attention.py [validate|bench]")
