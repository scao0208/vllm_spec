"""
Paged Bitmap Attention: bitmap block-skipping + zero-gather paged KV reads.

Combines the best of two approaches:
- bitmap_sparse_attention.py: register-resident uint64 bitmap traversal
  (ctz + clear-lowest-bit) to skip KV blocks with no ancestors
- paged_sparse_attention.py: indirect token-slot indexing to read K/V
  directly from SGLang's flat paged buffer (no gather, no GQA expand)

Two-stage architecture:
  - Stage 1 (off-diagonal): Sequential iteration over past context blocks.
    Reads K/V via indirect token_slots from paged buffer. No mask.
  - Stage 2 (on-diagonal): Bitmap-driven sparse traversal over tree region.
    Each set bit in the uint64 bitmap triggers a paged KV load + tree mask.

Key differences from bitmap_sparse_attention.py:
  - No TMA (paged memory is non-contiguous) — works on Ampere+, not just Hopper
  - GQA handled in-kernel (Q head -> KV head mapping in grid)
  - Zero gather: reads K/V in-place from paged buffer

Key differences from paged_sparse_attention.py:
  - No importance scoring / top-k selection — uses ALL past context + bitmap
    pruning for tree region only
  - Two-stage: separate past (dense) and tree (bitmap-sparse) loops
  - No block_indices/block_counts indirection — bitmap registers drive iteration

Usage:
    cd scripts
    python paged_bitmap_attention.py validate
    python paged_bitmap_attention.py bench
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# GPU BITMAP COMPUTATION
# ============================================================================

def compute_bitmap_from_mask_gpu(tree_mask_bool, BLOCK_M=32, BLOCK_N=32):
    """Compute per-q_block uint64 bitmaps from boolean tree mask, entirely on GPU.

    Replaces the CPU Python-loop version (_compute_bitmap_from_mask) with
    vectorized GPU operations:
      1. Pad mask to block boundaries
      2. Reshape to [num_q_blocks, BLOCK_M, num_kv_blocks, BLOCK_N]
      3. .any(dim=3).any(dim=1) → block_active [num_q_blocks, num_kv_blocks]
      4. Pack bits into uint64 words

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

    # Pad to block boundaries
    pad_q = num_q_blocks * BLOCK_M - N_tree
    pad_kv = num_kv_blocks * BLOCK_N - N_tree
    if pad_q > 0 or pad_kv > 0:
        mask_padded = F.pad(tree_mask_bool, (0, pad_kv, 0, pad_q), value=False)
    else:
        mask_padded = tree_mask_bool

    # Reshape to block grid and reduce
    mask_blocked = mask_padded.reshape(num_q_blocks, BLOCK_M,
                                       num_kv_blocks, BLOCK_N)
    block_active = mask_blocked.any(dim=3).any(dim=1)  # [num_q_blocks, num_kv_blocks]

    # Pack bits into uint64 words
    W = (num_kv_blocks + 63) // 64
    pad_to_w64 = W * 64 - num_kv_blocks
    if pad_to_w64 > 0:
        block_active = F.pad(block_active, (0, pad_to_w64), value=False)

    # Reshape to [num_q_blocks, W, 64] and pack
    block_active = block_active.reshape(num_q_blocks, W, 64)
    powers = (1 << torch.arange(64, device=device, dtype=torch.int64))  # [64]
    bitmaps = (block_active.to(torch.int64) * powers[None, None, :]).sum(dim=2)

    return bitmaps, W


# ============================================================================
# TRITON HELPERS
# ============================================================================

@triton.jit
def _ctz64(mask):
    """Count trailing zeros of 64-bit integer (position of lowest set bit).

    Uses popcount(lowest_bit - 1) with explicit 64-bit handling:
    splits into two 32-bit halves so libdevice.popc works correctly.
    """
    lowest_bit = mask & (-mask)
    val = lowest_bit - 1
    lo = (val & 0xFFFFFFFF).to(tl.int32)
    hi = ((val >> 32) & 0xFFFFFFFF).to(tl.int32)
    return tl.extra.cuda.libdevice.popc(lo) + tl.extra.cuda.libdevice.popc(hi)


# ============================================================================
# TRITON KERNEL: Paged Bitmap Attention
# ============================================================================

@triton.jit
def _paged_bitmap_stage1(acc, l_i, m_i, q,
                          k_buf_ptr, v_buf_ptr,
                          kv_indices_ptr,
                          stride_buf_n, stride_buf_h, stride_buf_d,
                          h_kv_offset,
                          dtype: tl.constexpr,
                          qk_scale,
                          past_len,
                          BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                          BLOCK_N: tl.constexpr):
    """Stage 1: dense attention to past context via paged KV reads.

    Iterates sequentially over past context blocks [0, past_len).
    Each block loads token_slots from kv_indices, then indirect-reads K/V
    from the paged buffer. No mask needed.
    """
    offs_d = tl.arange(0, HEAD_DIM)

    for start_n in tl.range(0, past_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        sel_mask = offs_n < past_len

        # Load token slot indices
        token_slots = tl.load(kv_indices_ptr + offs_n,
                              mask=sel_mask, other=0)

        # Indirect load K: k_buf[token_slots[j], h_kv, :]
        k_offsets = (token_slots[:, None] * stride_buf_n
                     + h_kv_offset
                     + offs_d[None, :] * stride_buf_d)
        k = tl.load(k_buf_ptr + k_offsets,
                     mask=sel_mask[:, None], other=0.0)

        # QK dot product
        qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]

        # Boundary mask
        boundary_mask = offs_n[None, :] < past_len
        qk = qk * qk_scale + tl.where(boundary_mask, 0.0, -1.0e6)

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # Indirect load V
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
def _paged_bitmap_stage2(acc, l_i, m_i, q,
                          k_buf_ptr, v_buf_ptr,
                          kv_indices_ptr,
                          tree_mask_ptr, tree_mask_stride_m, tree_mask_stride_n,
                          tree_mask_row_offset,
                          stride_buf_n, stride_buf_h, stride_buf_d,
                          h_kv_offset,
                          dtype: tl.constexpr,
                          qk_scale,
                          past_len,
                          bitmap_ptr,
                          N_tree,
                          BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                          BLOCK_N: tl.constexpr,
                          W: tl.constexpr):
    """Stage 2: bitmap-driven sparse attention over tree region via paged KV reads.

    Loads uint64 bitmap words into registers and uses ctz + clear-lowest-bit
    to enumerate set bits. Each set bit triggers:
      1. Compute KV positions in kv_indices (past_len + block_id * BLOCK_N)
      2. Load token_slots via kv_indices
      3. Indirect load K/V from paged buffer
      4. Load and apply tree mask tile
    """
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)

    for w in tl.range(0, W):
        mask_word = tl.load(bitmap_ptr + w)

        while mask_word != 0:
            b = _ctz64(mask_word)
            mask_word = mask_word & (mask_word - 1)

            kv_block_id = w * 64 + b
            start_n = kv_block_id * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Boundary mask for tree region
            sel_mask = offs_n < N_tree

            # KV indices offset by past_len (tree tokens start after past)
            kv_idx_offs = past_len + offs_n
            token_slots = tl.load(kv_indices_ptr + kv_idx_offs,
                                  mask=sel_mask, other=0)

            # Indirect load K
            k_offsets = (token_slots[:, None] * stride_buf_n
                         + h_kv_offset
                         + offs_d[None, :] * stride_buf_d)
            k = tl.load(k_buf_ptr + k_offsets,
                         mask=sel_mask[:, None], other=0.0)

            # QK dot product
            qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]

            # Tree mask: load tile
            mask_offsets = ((tree_mask_row_offset + offs_m)[:, None] * tree_mask_stride_m
                            + offs_n[None, :] * tree_mask_stride_n)
            tree_mask_block = tl.load(tree_mask_ptr + mask_offsets,
                                       mask=sel_mask[None, :], other=float('-inf'))

            # Apply mask + boundary
            boundary_mask = offs_n[None, :] < N_tree
            qk = qk * qk_scale + tree_mask_block + tl.where(boundary_mask, 0.0, -1.0e6)

            # Online softmax update
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]

            # Indirect load V
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
def _attn_fwd_paged_bitmap(
    Q, K_buf, V_buf, O,
    KV_indices,
    Tree_mask,
    Bitmaps,
    sm_scale,
    stride_qm, stride_qk,
    stride_buf_n, stride_buf_h, stride_buf_d,
    stride_om, stride_ok,
    stride_tm, stride_tn,
    stride_bm_q,  # bitmap stride for Q block dim (= W)
    N_CTX_Q,
    past_len,
    H_KV: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    W: tl.constexpr,
):
    """Paged bitmap attention kernel.

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

    # Q pointer for this head: Q is [H_q, N_tree, D]
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

    # ---- Stage 1: Dense past context (paged reads) ----
    if past_len > 0:
        acc, l_i, m_i = _paged_bitmap_stage1(
            acc, l_i, m_i, q,
            K_buf, V_buf,
            KV_indices,
            stride_buf_n, stride_buf_h, stride_buf_d,
            h_kv_offset,
            dtype, qk_scale, past_len,
            BLOCK_M, HEAD_DIM, BLOCK_N)

    # ---- Stage 2: Bitmap-driven sparse tree region (paged reads) ----
    bitmap_ptr = Bitmaps + start_m * stride_bm_q
    tree_mask_row_offset = start_m * BLOCK_M

    acc, l_i, m_i = _paged_bitmap_stage2(
        acc, l_i, m_i, q,
        K_buf, V_buf,
        KV_indices,
        Tree_mask, stride_tm, stride_tn,
        tree_mask_row_offset,
        stride_buf_n, stride_buf_h, stride_buf_d,
        h_kv_offset,
        dtype, qk_scale, past_len,
        bitmap_ptr,
        N_CTX_Q,
        BLOCK_M, HEAD_DIM, BLOCK_N, W)

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


def paged_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                            tree_mask_2d, bitmaps, W, past_len,
                            sm_scale, H_kv,
                            BLOCK_M=32, BLOCK_N=32):
    """Launch paged bitmap attention kernel.

    Args:
        q_heads: [H_q, N_tree, D] fp16 — query for one request
        k_buf: [max_tokens, H_kv, D] fp16 — paged K buffer
        v_buf: [max_tokens, H_kv, D] fp16 — paged V buffer
        kv_indices: [total_kv] int32 — token slot indices (past + tree)
        tree_mask_2d: [N_tree, N_tree] fp16 — 0 or -inf tree mask
        bitmaps: [num_q_blocks, W] int64 — per-q_block bitmap (GPU)
        W: int — number of uint64 words per q_block
        past_len: int — number of past context tokens
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

    _attn_fwd_paged_bitmap[grid](
        q_heads, k_buf, v_buf, output,
        kv_indices,
        tree_mask_2d,
        bitmaps,
        sm_scale,
        q_heads.stride(1), q_heads.stride(2),
        k_buf.stride(0), k_buf.stride(1), k_buf.stride(2),
        output.stride(1), output.stride(2),
        tree_mask_2d.stride(0), tree_mask_2d.stride(1),
        bitmaps.stride(0),
        N_tree,
        past_len,
        H_KV=H_kv,
        GQA_GROUP=gqa_group,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        W=W,
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

    # Build full mask [N_tree, total_kv]: past=True, tree=tree_mask_bool
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


def validate(device='cuda'):
    """Validate paged bitmap kernel against dense reference."""
    from eagle_tree_choices import EAGLE_TREES

    _ensure_triton_allocator(device)

    for tree_name in ["mc_sim_7b_63", "regular_512"]:
        tree_choices = EAGLE_TREES[tree_name]
        N_tree = len(tree_choices) + 1

        for past_len in [0, 64, 256]:
            total_kv = past_len + N_tree
            H_q = 8
            H_kv = 2
            D = 128
            sm_scale = 1.0 / (D ** 0.5)

            print(f"\n{'='*60}")
            print(f"Validating paged bitmap: {tree_name} "
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

            # Dense reference: gather to contiguous
            k_full = k_buf[kv_indices.long()]
            v_full = v_buf[kv_indices.long()]
            out_dense = _dense_reference(q_heads, k_full, v_full,
                                         tree_mask_bool, past_len, sm_scale, H_kv)

            # GPU bitmap computation
            bitmaps, W = compute_bitmap_from_mask_gpu(tree_mask_bool)

            # Build float tree mask for kernel
            tree_mask_2d = torch.where(
                tree_mask_bool,
                torch.zeros(1, device=device, dtype=torch.float16),
                torch.full((1,), float('-inf'), device=device, dtype=torch.float16),
            )

            # Run paged bitmap kernel
            out_paged = paged_bitmap_attention(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                bitmaps, W, past_len, sm_scale, H_kv)

            max_diff = (out_paged - out_dense).abs().max().item()
            mean_diff = (out_paged - out_dense).abs().mean().item()
            cos_sim = F.cosine_similarity(
                out_paged.float().flatten(), out_dense.float().flatten(), dim=0
            ).item()
            print(f"  Max diff:  {max_diff:.2e}")
            print(f"  Mean diff: {mean_diff:.2e}")
            print(f"  Cos sim:   {cos_sim:.6f}")
            passed = max_diff < 5e-2
            print(f"  {'PASSED' if passed else 'FAILED'}")

            if not passed:
                print(f"  WARNING: max_diff {max_diff:.2e} exceeds threshold 5e-2")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark(device='cuda', warmup=20, rep=100):
    """Microbenchmark paged bitmap vs paged sparse and dense reference."""
    import time
    from eagle_tree_choices import EAGLE_TREES, regular_tree_512

    _ensure_triton_allocator(device)

    tree_choices = regular_tree_512
    N_tree = len(tree_choices) + 1
    H_q = 8
    H_kv = 2
    D = 128
    sm_scale = 1.0 / (D ** 0.5)

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

        # GPU bitmap
        bitmaps, W = compute_bitmap_from_mask_gpu(tree_mask_bool)
        tree_mask_2d = torch.where(
            tree_mask_bool,
            torch.zeros(1, device=device, dtype=torch.float16),
            torch.full((1,), float('-inf'), device=device, dtype=torch.float16),
        )

        print(f"\n{'='*60}")
        print(f"past_len={past_len}, N_tree={N_tree}, total_kv={total_kv}")
        print(f"H_q={H_q}, H_kv={H_kv}, D={D}, W={W}")
        print(f"{'='*60}")

        # Warmup
        for _ in range(warmup):
            paged_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                   tree_mask_2d, bitmaps, W, past_len,
                                   sm_scale, H_kv)
        torch.cuda.synchronize()

        # Benchmark kernel
        t0 = time.perf_counter()
        for _ in range(rep):
            paged_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                   tree_mask_2d, bitmaps, W, past_len,
                                   sm_scale, H_kv)
        torch.cuda.synchronize()
        t_kernel = (time.perf_counter() - t0) / rep * 1000

        # Benchmark full pipeline (bitmap computation + kernel)
        for _ in range(warmup):
            bm, w = compute_bitmap_from_mask_gpu(tree_mask_bool)
            paged_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                   tree_mask_2d, bm, w, past_len,
                                   sm_scale, H_kv)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(rep):
            bm, w = compute_bitmap_from_mask_gpu(tree_mask_bool)
            paged_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                   tree_mask_2d, bm, w, past_len,
                                   sm_scale, H_kv)
        torch.cuda.synchronize()
        t_full = (time.perf_counter() - t0) / rep * 1000

        print(f"  Kernel only:    {t_kernel:.3f} ms")
        print(f"  Full pipeline:  {t_full:.3f} ms")
        print(f"  Bitmap compute: {t_full - t_kernel:.3f} ms")


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
        print("Usage: python paged_bitmap_attention.py [validate|bench]")
