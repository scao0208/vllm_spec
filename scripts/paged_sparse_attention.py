"""
Paged KV Cache Direct-Read Sparse Attention.

Triton kernel that reads K/V directly from SGLang's flat paged KV buffer
via indirect token-slot indexing. No gather, no CPU metadata computation.

Key differences from unified_sparse_attention.py:
- Zero gather: reads K/V in-place from paged buffer via token slot indices
- Zero CPU: importance scoring + mask building + block metadata all on GPU
- No GQA expand: handles GQA by mapping Q heads to KV heads in the grid
- No TMA: uses tl.load with indirect indexing (paged memory is non-contiguous)

Usage:
    python paged_sparse_attention.py validate   # correctness check
    python paged_sparse_attention.py bench      # microbenchmark
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# GPU-ONLY IMPORTANCE SCORING + SELECTION
# ============================================================================

def select_important_kv_gpu(q_first, k_buf, kv_indices, past_len, N_tree,
                            top_k_ratio):
    """All-GPU importance scoring and KV selection.

    Uses the first query token as importance proxy. Scores all past K
    positions, selects top-k, appends all tree positions.

    Args:
        q_first: [H_kv, D] — first query token (importance proxy)
        k_buf: [max_tokens, H_kv, D] — paged KV buffer
        kv_indices: [total_kv] int32 — token slot indices
        past_len: int
        N_tree: int
        top_k_ratio: float

    Returns:
        selected: [N_sel] int64 — sorted indices into [0, total_kv)
    """
    if past_len == 0 or top_k_ratio >= 1.0:
        return torch.arange(past_len + N_tree, device=q_first.device,
                            dtype=torch.int64)

    # Gather past K via index (one-time, only past portion)
    past_slots = kv_indices[:past_len]  # [past_len]
    past_k = k_buf[past_slots]  # [past_len, H_kv, D]

    # Score: dot product averaged over KV heads
    # q_first: [H_kv, D], past_k: [past_len, H_kv, D]
    scores = torch.einsum('hd,nhd->n', q_first, past_k)  # [past_len]

    n_keep = max(1, int(past_len * top_k_ratio))
    if n_keep >= past_len:
        return torch.arange(past_len + N_tree, device=q_first.device,
                            dtype=torch.int64)

    _, topk_idx = scores.topk(n_keep)  # [n_keep] — indices in [0, past_len)

    tree_idx = torch.arange(past_len, past_len + N_tree, device=q_first.device)
    selected = torch.cat([topk_idx, tree_idx]).sort().values
    return selected.to(torch.int64)


# ============================================================================
# GPU-ONLY MASK + BLOCK METADATA
# ============================================================================

def build_mask_and_metadata_gpu(tree_mask_bool, selected, past_len, N_tree,
                                BLOCK_M=32, BLOCK_N=64):
    """Build pruned attention mask and block-sparse metadata entirely on GPU.

    Args:
        tree_mask_bool: [N_tree, N_tree] bool — original tree mask
        selected: [N_sel] int64 — sorted selected indices (0..total_kv-1)
        past_len: int
        N_tree: int
        BLOCK_M: int
        BLOCK_N: int

    Returns:
        mask_2d: [N_tree, N_sel_padded] fp16 — 0 or -inf, padded to BLOCK_N
        block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32
        block_counts: [num_q_blocks] int32
        MAX_SPARSE_BLOCKS: int
        N_sel: int — actual (unpadded) number of selected positions
    """
    device = selected.device
    N_sel = selected.shape[0]

    # Build mask [N_tree, N_sel]
    mask_2d = torch.zeros(N_tree, N_sel, device=device, dtype=torch.float16)

    # Tree region columns: apply tree mask
    is_tree = selected >= past_len
    tree_cols = torch.where(is_tree)[0]
    if tree_cols.numel() > 0:
        tree_local = selected[tree_cols] - past_len  # local tree indices
        tree_mask_cols = tree_mask_bool[:, tree_local]  # [N_tree, n_tree_sel]
        mask_2d[:, tree_cols] = torch.where(
            tree_mask_cols,
            torch.zeros(1, device=device, dtype=torch.float16),
            torch.full((1,), float('-inf'), device=device, dtype=torch.float16),
        )
    # Past columns stay 0 (all query positions attend to all selected past)

    # Pad to block boundaries
    num_q_blocks = (N_tree + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (N_sel + BLOCK_N - 1) // BLOCK_N
    pad_q = num_q_blocks * BLOCK_M - N_tree
    pad_kv = num_kv_blocks * BLOCK_N - N_sel
    if pad_q > 0 or pad_kv > 0:
        mask_padded = F.pad(mask_2d, (0, pad_kv, 0, pad_q), value=float('-inf'))
    else:
        mask_padded = mask_2d

    # Block activity: [num_q_blocks, BLOCK_M, num_kv_blocks, BLOCK_N]
    mask_blocked = mask_padded.reshape(num_q_blocks, BLOCK_M,
                                       num_kv_blocks, BLOCK_N)
    block_active = (mask_blocked > float('-inf')).any(dim=3).any(dim=1)

    # Block counts and indices
    block_counts = block_active.sum(dim=1).to(torch.int32)
    MAX_SPARSE_BLOCKS = max(int(block_counts.max().item()), 1)

    # Stable sort: active blocks first, preserving index order
    sort_key = (~block_active).to(torch.int32)
    _, sorted_idx = sort_key.sort(dim=1, stable=True)
    block_indices = sorted_idx[:, :MAX_SPARSE_BLOCKS].to(torch.int32).contiguous()

    return mask_padded, block_indices, block_counts, MAX_SPARSE_BLOCKS, N_sel


# ============================================================================
# TRITON KERNEL — Paged indirect-index block-sparse attention
# ============================================================================

@triton.jit
def _paged_sparse_inner(acc, l_i, m_i, q,
                        k_buf_ptr, v_buf_ptr,
                        selected_slots_ptr,
                        mask_ptr, mask_stride_m, mask_stride_n,
                        mask_row_offset,
                        stride_buf_n, stride_buf_h, stride_buf_d,
                        h_kv_offset,
                        dtype: tl.constexpr,
                        qk_scale,
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                        BLOCK_N: tl.constexpr,
                        N_SEL: tl.constexpr,
                        block_indices_ptr, block_count,
                        MAX_SPARSE_BLOCKS: tl.constexpr):
    """Block-sparse inner loop reading K/V from paged buffer via indirect index."""
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)

    for b_idx in tl.range(0, block_count):
        kv_block_idx = tl.load(block_indices_ptr + b_idx)
        start_n = kv_block_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Boundary mask for this KV block
        sel_mask = offs_n < N_SEL

        # Load token slot indices for selected positions in this block
        token_slots = tl.load(selected_slots_ptr + offs_n,
                              mask=sel_mask, other=0)

        # Indirect load K: k_buf[token_slots[j], h_kv, :]
        # k_buf layout: [max_tokens, H_kv, D]
        k_offsets = (token_slots[:, None] * stride_buf_n
                     + h_kv_offset
                     + offs_d[None, :] * stride_buf_d)
        k = tl.load(k_buf_ptr + k_offsets,
                     mask=sel_mask[:, None], other=0.0)
        # k: [BLOCK_N, HEAD_DIM]

        # QK dot product
        qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]

        # Load mask block
        mask_offsets = ((mask_row_offset + offs_m)[:, None] * mask_stride_m
                        + offs_n[None, :] * mask_stride_n)
        mask_block = tl.load(mask_ptr + mask_offsets,
                             mask=sel_mask[None, :], other=float('-inf'))

        # Apply mask + boundary
        boundary_mask = offs_n[None, :] < N_SEL
        qk = qk * qk_scale + mask_block + tl.where(boundary_mask, 0.0, -1.0e6)

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # Indirect load V: v_buf[token_slots[j], h_kv, :]
        v_offsets = (token_slots[:, None] * stride_buf_n
                     + h_kv_offset
                     + offs_d[None, :] * stride_buf_d)
        v = tl.load(v_buf_ptr + v_offsets,
                     mask=sel_mask[:, None], other=0.0)

        p = p.to(dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_paged_sparse(
    Q, K_buf, V_buf, O,
    Selected_slots,
    Attn_mask,
    Block_indices, Block_counts,
    sm_scale,
    stride_qm, stride_qk,
    stride_buf_n, stride_buf_h, stride_buf_d,
    stride_om, stride_ok,
    stride_mm, stride_mn,
    stride_bi_m,
    N_CTX_Q, N_SEL,
    MASK_DIM_M,
    MASK_DIM_N: tl.constexpr,
    H_KV: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_SPARSE_BLOCKS: tl.constexpr,
):
    """Paged sparse attention kernel.

    Grid: [num_q_blocks, H_q]
    Each program handles one Q-block x one Q-head.
    GQA: maps Q head -> KV head internally via integer division.
    """
    dtype = tl.float16
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)

    # GQA: map Q head to KV head (consecutive Q heads share same KV head)
    off_h_kv = off_h_q // GQA_GROUP
    h_kv_offset = off_h_kv * stride_buf_h

    qk_scale = sm_scale * 1.44269504

    # Q pointer for this head: Q is [N_tree, D] per head, laid out as
    # [H_q, N_tree, D] with strides
    q_base = Q + off_h_q * N_CTX_Q * stride_qm
    o_base = O + off_h_q * N_CTX_Q * stride_om

    # Load Q block
    qo_offset = start_m * BLOCK_M
    offs_m = qo_offset + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m[:, None] < N_CTX_Q
    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=q_mask, other=0.0)

    # Init accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Block-sparse iteration
    mask_row_offset = start_m * BLOCK_M
    block_count = tl.load(Block_counts + start_m)
    block_indices_ptr = Block_indices + start_m * stride_bi_m

    acc, l_i, m_i = _paged_sparse_inner(
        acc, l_i, m_i, q,
        K_buf, V_buf,
        Selected_slots,
        Attn_mask, stride_mm, stride_mn,
        mask_row_offset,
        stride_buf_n, stride_buf_h, stride_buf_d,
        h_kv_offset,
        dtype, qk_scale,
        BLOCK_M, HEAD_DIM, BLOCK_N,
        N_SEL,
        block_indices_ptr, block_count,
        MAX_SPARSE_BLOCKS)

    # Epilogue: normalize
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # Store output
    q_valid_mask = offs_m[:, None] < N_CTX_Q
    acc_out = tl.where(q_valid_mask, acc.to(dtype), 0.0)
    tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
             acc_out, mask=q_valid_mask)


# ============================================================================
# PYTHON LAUNCH WRAPPER
# ============================================================================

def _ensure_triton_allocator(device='cuda'):
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device=device)
    triton.set_allocator(alloc_fn)


def paged_sparse_attention(q_heads, k_buf, v_buf, selected_slots, mask_2d,
                           block_indices, block_counts, MAX_SPARSE_BLOCKS,
                           N_sel, sm_scale, H_kv,
                           BLOCK_M=32, BLOCK_N=64):
    """Launch paged sparse attention kernel.

    Args:
        q_heads: [H_q, N_tree, D] fp16 — query for one request
        k_buf: [max_tokens, H_kv, D] fp16 — paged K buffer
        v_buf: [max_tokens, H_kv, D] fp16 — paged V buffer
        selected_slots: [N_sel] int32 — token slot indices in k_buf/v_buf
        mask_2d: [MASK_M, MASK_N] fp16 — padded attention mask (0 or -inf)
        block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32
        block_counts: [num_q_blocks] int32
        MAX_SPARSE_BLOCKS: int
        N_sel: int — actual number of selected positions
        sm_scale: float
        H_kv: int — number of KV heads (for GQA mapping)
        BLOCK_M: int
        BLOCK_N: int

    Returns:
        output: [H_q, N_tree, D] fp16
    """
    H_q, N_tree, D = q_heads.shape
    output = torch.empty_like(q_heads)
    gqa_group = H_q // H_kv

    grid = (triton.cdiv(N_tree, BLOCK_M), H_q)

    _attn_fwd_paged_sparse[grid](
        q_heads, k_buf, v_buf, output,
        selected_slots,
        mask_2d,
        block_indices, block_counts,
        sm_scale,
        q_heads.stride(1), q_heads.stride(2),           # stride_qm, stride_qk
        k_buf.stride(0), k_buf.stride(1), k_buf.stride(2),  # stride_buf_n/h/d
        output.stride(1), output.stride(2),              # stride_om, stride_ok
        mask_2d.stride(0), mask_2d.stride(1),            # stride_mm, stride_mn
        block_indices.stride(0),                         # stride_bi_m
        N_tree, N_sel,
        mask_2d.shape[0],                                # MASK_DIM_M
        MASK_DIM_N=mask_2d.shape[1],
        H_KV=H_kv,
        GQA_GROUP=gqa_group,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MAX_SPARSE_BLOCKS=MAX_SPARSE_BLOCKS,
    )

    return output


# ============================================================================
# VALIDATION
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
    gqa_rep = H_q // H_kv

    # Build full mask [N_tree, total_kv]: past=True, tree=tree_mask_bool
    full_mask = torch.ones(N_tree, total_kv, dtype=torch.bool,
                           device=q_heads.device)
    full_mask[:, past_len:past_len + N_tree] = tree_mask_bool

    attn_mask = torch.where(full_mask, 0.0, float('-inf')).to(torch.float32)

    outputs = []
    for h_q in range(H_q):
        h_kv = h_q // (H_q // H_kv)
        q = q_heads[h_q].float()  # [N_tree, D]
        k = k_full[:, h_kv, :].float()  # [total_kv, D]
        v = v_full[:, h_kv, :].float()  # [total_kv, D]

        scores = torch.matmul(q, k.T) * sm_scale + attn_mask  # [N_tree, total_kv]
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # [N_tree, D]
        outputs.append(out)

    return torch.stack(outputs).to(q_heads.dtype)  # [H_q, N_tree, D]


def validate(device='cuda'):
    """Validate paged sparse kernel against dense reference."""
    from eagle_tree_choices import EAGLE_TREES

    _ensure_triton_allocator(device)

    for tree_name in ["mc_sim_7b_63", "regular_512"]:
        tree_choices = EAGLE_TREES[tree_name]
        N_tree = len(tree_choices) + 1
        past_len = 256
        total_kv = past_len + N_tree
        H_q = 8
        H_kv = 2
        D = 128
        sm_scale = 1.0 / (D ** 0.5)

        print(f"\n{'='*60}")
        print(f"Validating paged sparse: {tree_name} "
              f"(N_tree={N_tree}, past={past_len}, H_q={H_q}, H_kv={H_kv})")
        print(f"{'='*60}")

        torch.manual_seed(42)

        # Simulate paged KV buffer (larger than needed, with gaps)
        max_tokens = total_kv + 200
        k_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                            device=device)
        v_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                            device=device)

        # Simulate non-contiguous token slot mapping
        perm = torch.randperm(max_tokens, device=device)
        kv_indices = perm[:total_kv].to(torch.int32)

        q_heads = torch.randn(H_q, N_tree, D, dtype=torch.float16,
                              device=device)

        # Build tree mask from tree_choices
        from sparse_tree_utils import precompute_ancestor_indices
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

        # Dense reference: gather to contiguous for ref computation
        k_full = k_buf[kv_indices.long()]  # [total_kv, H_kv, D]
        v_full = v_buf[kv_indices.long()]

        out_dense = _dense_reference(q_heads, k_full, v_full, tree_mask_bool,
                                     past_len, sm_scale, H_kv)

        # Test 1: ratio=1.0 (keep all)
        print("\n--- ratio=1.0 (keep all KV) ---")
        q_first = q_heads[0, 0, :]  # [D] — use head 0
        # Expand to [H_kv, D] by repeating
        q_first_hkv = q_heads[:H_kv, 0, :]  # [H_kv, D]

        selected = select_important_kv_gpu(q_first_hkv, k_buf, kv_indices,
                                           past_len, N_tree, 1.0)
        N_sel_val = selected.shape[0]
        assert N_sel_val == total_kv, f"Expected {total_kv}, got {N_sel_val}"

        selected_slots = kv_indices[selected.long()]
        mask_2d, block_indices, block_counts, MAX_SPARSE, N_sel = \
            build_mask_and_metadata_gpu(tree_mask_bool, selected, past_len,
                                        N_tree, BLOCK_M=32, BLOCK_N=64)

        out_paged = paged_sparse_attention(
            q_heads, k_buf, v_buf, selected_slots, mask_2d,
            block_indices, block_counts, MAX_SPARSE,
            N_sel, sm_scale, H_kv, BLOCK_M=32, BLOCK_N=64)

        max_diff = (out_paged - out_dense).abs().max().item()
        mean_diff = (out_paged - out_dense).abs().mean().item()
        cos_sim = F.cosine_similarity(
            out_paged.float().flatten(), out_dense.float().flatten(), dim=0
        ).item()
        print(f"  Max diff:  {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")
        print(f"  Cos sim:   {cos_sim:.6f}")
        passed = max_diff < 1e-2
        print(f"  {'PASSED' if passed else 'FAILED'}")

        # Test 2: various pruning ratios
        for ratio in [0.5, 0.3, 0.1]:
            print(f"\n--- ratio={ratio} ---")
            selected = select_important_kv_gpu(q_first_hkv, k_buf, kv_indices,
                                               past_len, N_tree, ratio)
            N_sel_val = selected.shape[0]

            selected_slots = kv_indices[selected.long()]
            mask_2d, block_indices, block_counts, MAX_SPARSE, N_sel = \
                build_mask_and_metadata_gpu(tree_mask_bool, selected, past_len,
                                            N_tree, BLOCK_M=32, BLOCK_N=64)

            out_pruned = paged_sparse_attention(
                q_heads, k_buf, v_buf, selected_slots, mask_2d,
                block_indices, block_counts, MAX_SPARSE,
                N_sel, sm_scale, H_kv, BLOCK_M=32, BLOCK_N=64)

            max_diff = (out_pruned - out_dense).abs().max().item()
            mean_diff = (out_pruned - out_dense).abs().mean().item()
            cos_sim = F.cosine_similarity(
                out_pruned.float().flatten(), out_dense.float().flatten(), dim=0
            ).item()
            print(f"  Selected: {N_sel_val}/{total_kv}")
            print(f"  Max diff:  {max_diff:.2e}")
            print(f"  Mean diff: {mean_diff:.2e}")
            print(f"  Cos sim:   {cos_sim:.6f}")


def benchmark(device='cuda', warmup=20, rep=100):
    """Microbenchmark paged sparse vs gather + unified sparse."""
    import time
    from eagle_tree_choices import EAGLE_TREES, regular_tree_512

    _ensure_triton_allocator(device)

    tree_choices = regular_tree_512
    N_tree = len(tree_choices) + 1
    H_q = 8
    H_kv = 2
    D = 128
    sm_scale = 1.0 / (D ** 0.5)

    for past_len in [256, 1024, 4096]:
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

        q_first_hkv = q_heads[:H_kv, 0, :]

        print(f"\n{'='*60}")
        print(f"past_len={past_len}, N_tree={N_tree}, total_kv={total_kv}")
        print(f"H_q={H_q}, H_kv={H_kv}, D={D}")
        print(f"{'='*60}")

        for ratio in [1.0, 0.5, 0.3]:
            selected = select_important_kv_gpu(q_first_hkv, k_buf, kv_indices,
                                               past_len, N_tree, ratio)
            selected_slots = kv_indices[selected.long()]
            mask_2d, block_indices, block_counts, MAX_SPARSE, N_sel = \
                build_mask_and_metadata_gpu(tree_mask_bool, selected, past_len,
                                            N_tree, BLOCK_M=32, BLOCK_N=64)

            # Warmup
            for _ in range(warmup):
                paged_sparse_attention(
                    q_heads, k_buf, v_buf, selected_slots, mask_2d,
                    block_indices, block_counts, MAX_SPARSE,
                    N_sel, sm_scale, H_kv)
            torch.cuda.synchronize()

            # Kernel-only timing
            t0 = time.perf_counter()
            for _ in range(rep):
                paged_sparse_attention(
                    q_heads, k_buf, v_buf, selected_slots, mask_2d,
                    block_indices, block_counts, MAX_SPARSE,
                    N_sel, sm_scale, H_kv)
            torch.cuda.synchronize()
            t_kernel = (time.perf_counter() - t0) / rep * 1000

            # Full pipeline: selection + mask + kernel
            for _ in range(warmup):
                sel = select_important_kv_gpu(q_first_hkv, k_buf, kv_indices,
                                              past_len, N_tree, ratio)
                ss = kv_indices[sel.long()]
                m2d, bi, bc, ms, ns = build_mask_and_metadata_gpu(
                    tree_mask_bool, sel, past_len, N_tree)
                paged_sparse_attention(q_heads, k_buf, v_buf, ss, m2d,
                                       bi, bc, ms, ns, sm_scale, H_kv)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(rep):
                sel = select_important_kv_gpu(q_first_hkv, k_buf, kv_indices,
                                              past_len, N_tree, ratio)
                ss = kv_indices[sel.long()]
                m2d, bi, bc, ms, ns = build_mask_and_metadata_gpu(
                    tree_mask_bool, sel, past_len, N_tree)
                paged_sparse_attention(q_heads, k_buf, v_buf, ss, m2d,
                                       bi, bc, ms, ns, sm_scale, H_kv)
            torch.cuda.synchronize()
            t_full = (time.perf_counter() - t0) / rep * 1000

            print(f"\n  ratio={ratio}: N_sel={N_sel}/{total_kv}")
            print(f"    Kernel only:    {t_kernel:.3f} ms")
            print(f"    Full pipeline:  {t_full:.3f} ms")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
    elif len(sys.argv) > 1 and sys.argv[1] == "bench":
        benchmark()
    else:
        print("Usage: python paged_sparse_attention.py [validate|bench]")
