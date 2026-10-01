"""
Tile-Level Bitmap Attention: bitmap as tile-level execution schedule.

Key difference from paged_bitmap_attention.py:
  - Bitmap granularity: BLOCK_N=16 (configurable) vs fixed 32
  - Each bit = one TC-friendly KV tile. bit=1 → load + MMA, bit=0 → skip entirely
  - Within visited tiles: fine-grained tree_mask loaded ONLY for those tiles
  - The bitmap IS the execution schedule at tile level, not just block selection

Two-stage architecture (same concept as paged_bitmap_attention):
  - Stage 1 (off-diagonal): Dense sequential over past context, paged KV reads
  - Stage 2 (on-diagonal): Tile-level bitmap traversal over tree region

For regular_512 with BLOCK_N=16: 32 KV tiles → 1 uint64 word, ~58% tiles skipped.
With BLOCK_N=32: 16 KV tiles → 1 uint64 word, ~48% tiles skipped.

Usage:
    cd scripts
    python tile_bitmap_attention.py validate
    python tile_bitmap_attention.py bench
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# GPU BITMAP COMPUTATION (parametric BLOCK_N)
# ============================================================================

def compute_tile_bitmap(tree_mask_bool, BLOCK_M=32, BLOCK_N=16):
    """Compute tile-level bitmap from boolean tree mask.

    Each bit = one BLOCK_N-sized KV tile.
    bit=1 if ANY Q row in the q_block sees ANY KV position in that tile.

    Args:
        tree_mask_bool: [N_tree, N_tree] boolean tensor (GPU).
        BLOCK_M: Q tile size.
        BLOCK_N: KV tile size (default 16 for finer granularity).

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
    powers = (1 << torch.arange(64, device=device, dtype=torch.int64))
    bitmaps = (block_active.to(torch.int64) * powers[None, None, :]).sum(dim=2)

    return bitmaps, W


# ============================================================================
# QUEST-STYLE SPARSE STAGE 1 HELPERS
# ============================================================================

def compute_page_metadata(k_buf, kv_indices, past_len, H_kv, D, page_size=16):
    """Compute per-page min/max K statistics for Quest-style page selection.

    Args:
        k_buf: [max_tokens, H_kv, D] fp16 — paged K buffer
        kv_indices: [total_kv] int32 — token slot indices
        past_len: int — number of past context tokens
        H_kv: int — number of KV heads
        D: int — head dimension
        page_size: int — tokens per page for metadata aggregation

    Returns:
        page_min: [num_pages, H_kv, D] fp32
        page_max: [num_pages, H_kv, D] fp32
    """
    # Gather past K values
    k_past = k_buf[kv_indices[:past_len].long()]  # [past_len, H_kv, D]

    num_pages = (past_len + page_size - 1) // page_size
    padded_len = num_pages * page_size

    # Pad to full pages
    if padded_len > past_len:
        pad_tokens = padded_len - past_len
        k_padded = torch.nn.functional.pad(k_past, (0, 0, 0, 0, 0, pad_tokens))
    else:
        k_padded = k_past

    # Reshape into pages: [num_pages, page_size, H_kv, D]
    k_pages = k_padded.reshape(num_pages, page_size, H_kv, D).float()

    # For partial last page, set padding to neutral values
    if padded_len > past_len:
        pad_start = past_len % page_size
        # Set padding positions to +inf for min and -inf for max so they're ignored
        k_pages[-1, pad_start:] = 0  # will be overwritten below
        page_max = k_pages.clone()
        page_min = k_pages.clone()
        page_max[-1, pad_start:] = float('-inf')
        page_min[-1, pad_start:] = float('inf')
        page_max = page_max.amax(dim=1)  # [num_pages, H_kv, D]
        page_min = page_min.amin(dim=1)
    else:
        page_max = k_pages.amax(dim=1)
        page_min = k_pages.amin(dim=1)

    return page_min, page_max


def estimate_page_importance(q_rep, page_min, page_max):
    """Estimate page importance scores using Quest formula.

    For each dimension d: score_d = max(Q[d] * K_max[d], Q[d] * K_min[d])
    Page score = sum over D and H_kv dimensions.

    Args:
        q_rep: [H_kv, D] fp32 — representative query (mean over tree tokens, grouped by KV head)
        page_min: [num_pages, H_kv, D] fp32
        page_max: [num_pages, H_kv, D] fp32

    Returns:
        page_scores: [num_pages] fp32
    """
    # q_rep: [H_kv, D], page_max/min: [num_pages, H_kv, D]
    score_max = q_rep[None, :, :] * page_max  # [num_pages, H_kv, D]
    score_min = q_rep[None, :, :] * page_min
    score_d = torch.maximum(score_max, score_min)  # element-wise max
    page_scores = score_d.sum(dim=(1, 2))  # [num_pages]
    return page_scores


def select_important_pages(page_scores, page_budget, kv_indices, past_len, page_size=16):
    """Select top-k important pages and rebuild kv_indices.

    Args:
        page_scores: [num_pages] fp32
        page_budget: int — number of pages to keep
        kv_indices: [total_kv] int32 — original token slot indices
        past_len: int — original past context length
        page_size: int — tokens per page

    Returns:
        new_kv_indices: [new_past_len + tree_len] int32
        new_past_len: int
    """
    num_pages = len(page_scores)
    page_budget = min(page_budget, num_pages)

    # Select top-k pages, sort for spatial locality
    _, top_page_indices = torch.topk(page_scores, page_budget)
    top_page_indices, _ = top_page_indices.sort()

    # Expand pages to token positions
    offsets = torch.arange(page_size, device=kv_indices.device)
    positions = top_page_indices[:, None] * page_size + offsets[None, :]  # [budget, page_size]
    positions = positions.reshape(-1)

    # Mask out positions beyond past_len (partial last page)
    valid = positions < past_len
    positions = positions[valid]

    # Gather selected past slots
    selected_past = kv_indices[positions.long()]

    # Concatenate with tree region
    tree_indices = kv_indices[past_len:]
    new_kv_indices = torch.cat([selected_past, tree_indices])
    new_past_len = len(selected_past)

    return new_kv_indices, new_past_len


# ============================================================================
# QUEST APPROACH 1: Cross-round caching (amortized K gather)
# ============================================================================

def update_page_metadata_incremental(page_min, page_max, k_buf, kv_indices,
                                      old_past_len, new_past_len,
                                      H_kv, D, page_size=16):
    """Incrementally update page metadata for newly accepted tokens.

    Only gathers and processes the K values for positions [old_past_len, new_past_len).
    Updates the affected pages' min/max in-place.

    Args:
        page_min: [old_num_pages, H_kv, D] fp32 — existing per-page min (modified in-place)
        page_max: [old_num_pages, H_kv, D] fp32 — existing per-page max (modified in-place)
        k_buf: [max_tokens, H_kv, D] fp16 — paged K buffer
        kv_indices: [total_kv] int32 — token slot indices
        old_past_len: int — previous past length
        new_past_len: int — current past length
        H_kv, D: int — head dimensions
        page_size: int

    Returns:
        page_min: [new_num_pages, H_kv, D] fp32 (may be extended)
        page_max: [new_num_pages, H_kv, D] fp32 (may be extended)
    """
    if new_past_len <= old_past_len:
        return page_min, page_max

    new_num_pages = (new_past_len + page_size - 1) // page_size
    old_num_pages = page_min.shape[0]

    # Extend if new pages appeared
    if new_num_pages > old_num_pages:
        ext = new_num_pages - old_num_pages
        page_min = torch.cat([page_min,
            torch.full((ext, H_kv, D), float('inf'), device=page_min.device, dtype=page_min.dtype)])
        page_max = torch.cat([page_max,
            torch.full((ext, H_kv, D), float('-inf'), device=page_max.device, dtype=page_max.dtype)])

    # Gather only new tokens
    new_k = k_buf[kv_indices[old_past_len:new_past_len].long()].float()  # [delta, H_kv, D]

    # Update affected pages
    first_affected_page = old_past_len // page_size
    for p in range(first_affected_page, new_num_pages):
        start = max(p * page_size, old_past_len)
        end = min((p + 1) * page_size, new_past_len)
        local_start = start - old_past_len
        local_end = end - old_past_len
        k_slice = new_k[local_start:local_end]  # [slice_len, H_kv, D]
        page_min[p] = torch.minimum(page_min[p], k_slice.amin(dim=0))
        page_max[p] = torch.maximum(page_max[p], k_slice.amax(dim=0))

    return page_min, page_max


# ============================================================================
# QUEST APPROACH 2: Physical-page metadata via set_kv_buffer monkey-patch
# ============================================================================

def create_physical_page_metadata(max_total_tokens, H_kv, D, page_size, device='cuda'):
    """Allocate physical-page metadata buffers.

    Returns:
        phys_page_min: [max_phys_pages, H_kv, D] fp32
        phys_page_max: [max_phys_pages, H_kv, D] fp32
    """
    max_phys_pages = (max_total_tokens + page_size - 1) // page_size
    phys_page_min = torch.full((max_phys_pages, H_kv, D), float('inf'),
                                device=device, dtype=torch.float32)
    phys_page_max = torch.full((max_phys_pages, H_kv, D), float('-inf'),
                                device=device, dtype=torch.float32)
    return phys_page_min, phys_page_max


def update_physical_page_metadata(phys_page_min, phys_page_max,
                                   cache_loc, cache_k, page_size):
    """Update physical-page metadata from a set_kv_buffer call (PyTorch fallback).

    Called inside monkey-patched set_kv_buffer for layer 0.

    Args:
        phys_page_min/max: [max_phys_pages, H_kv, D] fp32
        cache_loc: [N] int32/int64 — slot indices being written
        cache_k: [N, H_kv, D] — K values being written
        page_size: int
    """
    phys_pages = (cache_loc // page_size).long()  # [N]
    k_float = cache_k.float()

    # scatter_reduce for min/max
    pp_expanded = phys_pages[:, None, None].expand_as(k_float)
    phys_page_min.scatter_reduce_(0, pp_expanded, k_float, reduce='amin',
                                   include_self=True)
    phys_page_max.scatter_reduce_(0, pp_expanded, k_float, reduce='amax',
                                   include_self=True)


# ============================================================================
# FUSED KV WRITE + PAGE METADATA (Triton)
# ============================================================================

@triton.jit
def _fused_kv_write_page_metadata_kernel(
    cache_k_ptr, cache_v_ptr,     # [N, ROW_DIM] input
    k_buf_ptr, v_buf_ptr,         # [max_tokens, ROW_DIM] cache
    loc_ptr,                       # [N] int32/int64
    page_min_ptr, page_max_ptr,   # [max_pages * ROW_DIM] fp32
    N,
    ROW_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused KV cache write + page metadata update.

    Each program handles one token:
    1. Read K, V from input
    2. Scatter-write K, V to cache at loc[token_idx]
    3. Atomic min/max update page metadata with K values (fp32)
    """
    token_idx = tl.program_id(0)
    if token_idx >= N:
        return

    loc_val = tl.load(loc_ptr + token_idx).to(tl.int64)
    page_id = loc_val // PAGE_SIZE

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < ROW_DIM

    in_base = token_idx.to(tl.int64) * ROW_DIM
    out_base = loc_val * ROW_DIM

    # Load K and V from input
    k_val = tl.load(cache_k_ptr + in_base + d_offs, mask=d_mask)
    v_val = tl.load(cache_v_ptr + in_base + d_offs, mask=d_mask)

    # Scatter-write to KV cache
    tl.store(k_buf_ptr + out_base + d_offs, k_val, mask=d_mask)
    tl.store(v_buf_ptr + out_base + d_offs, v_val, mask=d_mask)

    # Atomic update page metadata (fp32)
    k_f32 = k_val.to(tl.float32)
    meta_base = page_id * ROW_DIM
    tl.atomic_min(page_min_ptr + meta_base + d_offs, k_f32, mask=d_mask)
    tl.atomic_max(page_max_ptr + meta_base + d_offs, k_f32, mask=d_mask)


def fused_kv_write_page_metadata(cache_k, cache_v, k_buffer, v_buffer, loc,
                                  page_min, page_max, page_size):
    """Fused KV cache write + page metadata update via single Triton kernel.

    Replaces: index_put (KV write) + scatter_reduce (metadata)
    with one kernel launch. Zero intermediate allocations.
    """
    N = loc.shape[0]
    if N == 0:
        return

    k_flat = cache_k.reshape(N, -1)
    v_flat = cache_v.reshape(N, -1)
    ROW_DIM = k_flat.shape[1]

    if not k_flat.is_contiguous():
        k_flat = k_flat.contiguous()
    if not v_flat.is_contiguous():
        v_flat = v_flat.contiguous()

    k_buf_flat = k_buffer.reshape(-1, ROW_DIM)
    v_buf_flat = v_buffer.reshape(-1, ROW_DIM)
    BLOCK_D = triton.next_power_of_2(ROW_DIM)

    _fused_kv_write_page_metadata_kernel[(N,)](
        k_flat, v_flat,
        k_buf_flat, v_buf_flat,
        loc,
        page_min.reshape(-1), page_max.reshape(-1),
        N,
        ROW_DIM=ROW_DIM,
        PAGE_SIZE=page_size,
        BLOCK_D=BLOCK_D,
    )


@triton.jit
def _update_page_metadata_kernel(
    cache_k_ptr,           # [N, ROW_DIM] fp16/bf16 input
    loc_ptr,               # [N] int32/int64
    page_min_ptr,          # [max_pages * ROW_DIM] fp32
    page_max_ptr,          # [max_pages * ROW_DIM] fp32
    N,
    ROW_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Metadata-only update: read K from input, atomic min/max on page metadata.
    Does NOT write to KV cache — use alongside original set_kv_buffer."""
    token_idx = tl.program_id(0)
    if token_idx >= N:
        return

    loc_val = tl.load(loc_ptr + token_idx).to(tl.int64)
    page_id = loc_val // PAGE_SIZE

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < ROW_DIM

    in_base = token_idx.to(tl.int64) * ROW_DIM
    k_val = tl.load(cache_k_ptr + in_base + d_offs, mask=d_mask)
    k_f32 = k_val.to(tl.float32)

    meta_base = page_id * ROW_DIM
    tl.atomic_min(page_min_ptr + meta_base + d_offs, k_f32, mask=d_mask)
    tl.atomic_max(page_max_ptr + meta_base + d_offs, k_f32, mask=d_mask)


def update_physical_page_metadata_triton(phys_page_min, phys_page_max,
                                          cache_loc, cache_k, page_size):
    """Triton metadata update (replaces scatter_reduce). No intermediate allocs."""
    N = cache_loc.shape[0]
    if N == 0:
        return

    k_flat = cache_k.reshape(N, -1)
    ROW_DIM = k_flat.shape[1]
    if not k_flat.is_contiguous():
        k_flat = k_flat.contiguous()
    BLOCK_D = triton.next_power_of_2(ROW_DIM)

    _update_page_metadata_kernel[(N,)](
        k_flat, cache_loc,
        phys_page_min.reshape(-1), phys_page_max.reshape(-1),
        N,
        ROW_DIM=ROW_DIM,
        PAGE_SIZE=page_size,
        BLOCK_D=BLOCK_D,
    )


def quest_from_physical_pages(q_rep, phys_page_min, phys_page_max,
                               kv_indices, past_len, quest_ratio, page_size):
    """Compute Quest page selection using physical-page metadata (no K gather).

    Args:
        q_rep: [H_kv, D] fp32 — representative query
        phys_page_min/max: [max_phys_pages, H_kv, D] fp32 — global metadata
        kv_indices: [total_kv] int32 — original token slot indices
        past_len: int
        quest_ratio: float — fraction of pages to keep
        page_size: int

    Returns:
        new_kv_indices: [new_past_len + tree_len] int32
        new_past_len: int
    """
    # Map past token positions to physical pages
    past_slots = kv_indices[:past_len].long()
    past_phys_pages = past_slots // page_size  # [past_len]
    unique_phys_pages, inverse = torch.unique(past_phys_pages, return_inverse=True)

    # Look up precomputed metadata (no K gather!)
    page_min = phys_page_min[unique_phys_pages]  # [num_unique, H_kv, D]
    page_max = phys_page_max[unique_phys_pages]

    # Score pages
    scores = estimate_page_importance(q_rep, page_min, page_max)

    # Select top-k physical pages
    budget = max(1, int(len(unique_phys_pages) * quest_ratio))
    budget = min(budget, len(unique_phys_pages))
    _, top_idx = torch.topk(scores, budget)
    selected_phys = unique_phys_pages[top_idx]

    # Build per-position keep mask
    keep_mask = torch.isin(past_phys_pages, selected_phys)

    # Gather selected past slots
    selected_past = past_slots[keep_mask].to(kv_indices.dtype)

    # Concatenate with tree region
    tree_indices = kv_indices[past_len:]
    new_kv_indices = torch.cat([selected_past, tree_indices])
    new_past_len = len(selected_past)

    return new_kv_indices, new_past_len


# ============================================================================
# FUSED QUEST PAGE SCORING (Triton — eliminates K gather for per-round mode)
# ============================================================================

@triton.jit
def _quest_page_score_kernel(
    k_buf_ptr,          # [max_tokens, H_kv, D] fp16 — paged K buffer (flat)
    kv_indices_ptr,     # [total_kv] int32 — token slot indices
    q_rep_ptr,          # [ROW_DIM] fp32 — representative query (H_kv * D flattened)
    page_scores_ptr,    # [num_pages] fp32 — output scores
    past_len,
    stride_buf_tok,     # stride between tokens in k_buf (= H_kv * D)
    ROW_DIM: tl.constexpr,   # H_kv * D
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,    # next_power_of_2(ROW_DIM)
):
    """One program per page. Reads K from paged buffer, computes page score directly."""
    page_id = tl.program_id(0)
    start = page_id * PAGE_SIZE
    end = tl.minimum(start + PAGE_SIZE, past_len)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < ROW_DIM

    # Load q_rep once per page (shared across all tokens in page)
    q_val = tl.load(q_rep_ptr + d_offs, mask=d_mask, other=0.0)

    # Initialize running min/max
    running_min = tl.full([BLOCK_D], float('inf'), dtype=tl.float32)
    running_max = tl.full([BLOCK_D], float('-inf'), dtype=tl.float32)

    # Iterate over tokens in this page
    for t in range(PAGE_SIZE):
        pos = start + t
        if pos < end:
            # Indirect read: kv_indices[pos] -> slot in k_buf
            slot = tl.load(kv_indices_ptr + pos).to(tl.int64)
            k_base = slot * stride_buf_tok
            k_val = tl.load(k_buf_ptr + k_base + d_offs, mask=d_mask, other=0.0)
            k_f32 = k_val.to(tl.float32)
            running_min = tl.minimum(running_min, k_f32)
            running_max = tl.maximum(running_max, k_f32)

    # Compute page score: sum_d max(q * k_max, q * k_min)
    score_max = q_val * running_max
    score_min = q_val * running_min
    score_d = tl.maximum(score_max, score_min)
    # Mask out padding dimensions
    score_d = tl.where(d_mask, score_d, 0.0)
    page_score = tl.sum(score_d)

    tl.store(page_scores_ptr + page_id, page_score)


def quest_page_scores_triton(k_buf, kv_indices, past_len, q_rep, page_size=16):
    """Compute Quest page scores directly from paged K buffer using Triton.

    Replaces compute_page_metadata() + estimate_page_importance() with a single
    fused kernel. No intermediate K gather or page_min/page_max tensors.

    Args:
        k_buf: [max_tokens, H_kv, D] fp16 — paged K buffer
        kv_indices: [total_kv] int32 — token slot indices
        past_len: int — number of past context tokens
        q_rep: [H_kv, D] fp32 — representative query
        page_size: int

    Returns:
        page_scores: [num_pages] fp32
    """
    num_pages = (past_len + page_size - 1) // page_size
    if num_pages == 0:
        return torch.empty(0, device=k_buf.device, dtype=torch.float32)

    H_kv = k_buf.shape[1]
    D = k_buf.shape[2]
    ROW_DIM = H_kv * D
    BLOCK_D = triton.next_power_of_2(ROW_DIM)
    stride_buf_tok = k_buf.stride(0)  # H_kv * D for contiguous

    page_scores = torch.empty(num_pages, device=k_buf.device, dtype=torch.float32)

    _quest_page_score_kernel[(num_pages,)](
        k_buf.reshape(-1),           # flat view
        kv_indices,
        q_rep.reshape(-1).contiguous(),  # [ROW_DIM] fp32
        page_scores,
        past_len,
        stride_buf_tok,
        ROW_DIM=ROW_DIM,
        PAGE_SIZE=page_size,
        BLOCK_D=BLOCK_D,
    )

    return page_scores


_powers64_cache = {}

def build_quest_bitmap(page_scores, page_budget, past_len, page_size, BLOCK_N):
    """Build a Stage 1 bitmap from Quest page scores.

    Each bit in the bitmap = one BLOCK_N-sized KV block in the past context.
    bit=1 → important block (Quest selected), bit=0 → skip.

    Args:
        page_scores: [num_pages] fp32
        page_budget: int — number of pages to keep
        past_len: int
        page_size: int — Quest page size (may differ from BLOCK_N)
        BLOCK_N: int — kernel tile size

    Returns:
        quest_bitmap: [W_past] int64 — packed bitmap for Stage 1
        W_past: int — number of uint64 words
    """
    device = page_scores.device
    num_pages = len(page_scores)
    page_budget = min(page_budget, num_pages)
    num_past_blocks = (past_len + BLOCK_N - 1) // BLOCK_N
    W_past = (num_past_blocks + 63) // 64
    padded = W_past * 64

    # Select top-k pages and map to block indices in one fused step
    _, top_page_indices = torch.topk(page_scores, page_budget)

    # Direct block index computation (no intermediate bool array)
    if page_size == BLOCK_N:
        block_ids = top_page_indices.long()
        block_ids = block_ids[block_ids < num_past_blocks]
    else:
        blk_starts = (top_page_indices * page_size) // BLOCK_N
        tok_ends = torch.clamp(top_page_indices * page_size + page_size, max=past_len)
        blk_ends = torch.clamp((tok_ends + BLOCK_N - 1) // BLOCK_N, max=num_past_blocks)
        # Each page covers blk_starts to blk_ends-1; for ps<=BN that's 1-2 blocks
        block_ids = torch.cat([blk_starts, blk_ends - 1]).long()
        block_ids = block_ids.clamp(0, num_past_blocks - 1)

    # Deduplicate to avoid scatter_add_ double-counting
    block_ids = block_ids.unique()

    # Compute word index and bit position directly
    word_idx = block_ids // 64
    bit_pos = block_ids % 64
    bit_vals = (1 << bit_pos).to(torch.int64)

    # Scatter into bitmap words (safe: no duplicates after unique())
    quest_bitmap = torch.zeros(W_past, dtype=torch.int64, device=device)
    quest_bitmap.scatter_add_(0, word_idx, bit_vals)

    return quest_bitmap, W_past


def build_quest_block_ids(page_scores, page_budget, past_len, page_size, BLOCK_N):
    """Build a compact sorted block ID list from Quest page scores.

    Returns a sorted int32 tensor of active BLOCK_N-sized block IDs.
    Used by Stage 1 block-list traversal (regular tl.range loop, pipelinable).

    Args:
        page_scores: [num_pages] fp32
        page_budget: int — number of pages to keep
        past_len: int
        page_size: int — Quest page size
        BLOCK_N: int — kernel tile size

    Returns:
        block_ids: [num_active] int32 — sorted active block IDs
    """
    device = page_scores.device
    num_pages = len(page_scores)
    page_budget = min(page_budget, num_pages)
    num_past_blocks = (past_len + BLOCK_N - 1) // BLOCK_N

    _, top_page_indices = torch.topk(page_scores, page_budget)

    if page_size == BLOCK_N:
        block_ids = top_page_indices.long()
        block_ids = block_ids[block_ids < num_past_blocks]
    else:
        blk_starts = (top_page_indices * page_size) // BLOCK_N
        tok_ends = torch.clamp(top_page_indices * page_size + page_size, max=past_len)
        blk_ends = torch.clamp((tok_ends + BLOCK_N - 1) // BLOCK_N, max=num_past_blocks)
        block_ids = torch.cat([blk_starts, blk_ends - 1]).long()
        block_ids = block_ids.clamp(0, num_past_blocks - 1)

    # Deduplicate and sort for spatial locality
    block_ids = block_ids.unique(sorted=True)

    return block_ids.to(torch.int32)


# ============================================================================
# TRITON HELPERS
# ============================================================================

@triton.jit
def _ctz64(mask):
    """Count trailing zeros of 64-bit integer (position of lowest set bit)."""
    lowest_bit = mask & (-mask)
    val = lowest_bit - 1
    lo = (val & 0xFFFFFFFF).to(tl.int32)
    hi = ((val >> 32) & 0xFFFFFFFF).to(tl.int32)
    return tl.extra.cuda.libdevice.popc(lo) + tl.extra.cuda.libdevice.popc(hi)


# ============================================================================
# TRITON KERNEL: Tile-Level Bitmap Attention
# ============================================================================

@triton.jit
def _tile_bitmap_stage1(acc, l_i, m_i, q,
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

    Same as paged_bitmap_attention Stage 1 — sequential iteration over
    past context blocks, indirect KV reads, no mask.
    """
    offs_d = tl.arange(0, HEAD_DIM)

    for start_n in tl.range(0, past_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        sel_mask = offs_n < past_len

        token_slots = tl.load(kv_indices_ptr + offs_n,
                              mask=sel_mask, other=0)

        k_offsets = (token_slots[:, None] * stride_buf_n
                     + h_kv_offset
                     + offs_d[None, :] * stride_buf_d)
        k = tl.load(k_buf_ptr + k_offsets,
                     mask=sel_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k))

        boundary_mask = offs_n[None, :] < past_len
        qk = qk * qk_scale + tl.where(boundary_mask, 0.0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

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
def _tile_bitmap_stage1_sparse(acc, l_i, m_i, q,
                                k_buf_ptr, v_buf_ptr,
                                kv_indices_ptr,
                                stride_buf_n, stride_buf_h, stride_buf_d,
                                h_kv_offset,
                                dtype: tl.constexpr,
                                qk_scale,
                                past_len,
                                active_blocks_ptr,
                                num_active_blocks,
                                BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                                BLOCK_N: tl.constexpr):
    """Stage 1 with Quest block list: only visits important past blocks.

    Uses a compact block ID list (from Quest page selection) instead of
    bitmap ctz traversal. Regular tl.range loop enables Triton software
    pipelining for memory load/compute overlap.
    """
    offs_d = tl.arange(0, HEAD_DIM)

    for i in tl.range(0, num_active_blocks):
        block_id = tl.load(active_blocks_ptr + i)
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        sel_mask = offs_n < past_len

        token_slots = tl.load(kv_indices_ptr + offs_n,
                              mask=sel_mask, other=0)

        k_offsets = (token_slots[:, None] * stride_buf_n
                     + h_kv_offset
                     + offs_d[None, :] * stride_buf_d)
        k = tl.load(k_buf_ptr + k_offsets,
                     mask=sel_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k))

        boundary_mask = offs_n[None, :] < past_len
        qk = qk * qk_scale + tl.where(boundary_mask, 0.0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

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
def _tile_bitmap_stage1_bitmap_masked(acc, l_i, m_i, q,
                                       k_buf_ptr, v_buf_ptr,
                                       kv_indices_ptr,
                                       stride_buf_n, stride_buf_h, stride_buf_d,
                                       h_kv_offset,
                                       dtype: tl.constexpr,
                                       qk_scale,
                                       past_len,
                                       quest_bitmap_ptr,
                                       BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                                       BLOCK_N: tl.constexpr):
    """Stage 1 with bitmap-masked dense loop: no ctz, no block-list.

    Iterates over ALL past blocks with tl.range (pipelinable).
    Bitmap bit-test decides whether to load+compute or skip.
    Sequential block_id → sequential kv_indices → hardware prefetch works.
    Empty iterations: ~4 cycles each (register bit-test only).
    """
    offs_d = tl.arange(0, HEAD_DIM)
    num_past_blocks = (past_len + BLOCK_N - 1) // BLOCK_N

    for block_id in tl.range(0, num_past_blocks):
        # Bitmap bit-test: L1-cached word load + register shift
        word_idx = block_id // 64
        bit_pos = block_id % 64
        word = tl.load(quest_bitmap_ptr + word_idx)
        is_active = ((word >> bit_pos) & 1) != 0

        if is_active:
            start_n = block_id * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            sel_mask = offs_n < past_len

            token_slots = tl.load(kv_indices_ptr + offs_n,
                                  mask=sel_mask, other=0)

            k_offsets = (token_slots[:, None] * stride_buf_n
                         + h_kv_offset
                         + offs_d[None, :] * stride_buf_d)
            k = tl.load(k_buf_ptr + k_offsets,
                         mask=sel_mask[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k))

            boundary_mask = offs_n[None, :] < past_len
            qk = qk * qk_scale + tl.where(boundary_mask, 0.0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]

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
def _tile_bitmap_stage2(acc, l_i, m_i, q,
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
    """Tile-level bitmap-driven Stage 2.

    Each bit in bitmap = one BLOCK_N-sized KV tile.
    bit=1 → load KV tile from paged buffer, execute QK MMA + PV MMA
    bit=0 → skip entirely (no load, no compute)

    Within visited tiles: load fine-grained tree_mask tile for boundary masking.
    """
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)

    for w in tl.range(0, W):
        mask_word = tl.load(bitmap_ptr + w)

        while mask_word != 0:
            b = _ctz64(mask_word)
            mask_word = mask_word & (mask_word - 1)

            kv_tile_id = w * 64 + b
            start_n = kv_tile_id * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Boundary guard
            sel_mask = offs_n < N_tree

            # --- Load KV tile from paged buffer (indirect) ---
            kv_idx_offs = past_len + offs_n
            token_slots = tl.load(kv_indices_ptr + kv_idx_offs,
                                  mask=sel_mask, other=0)

            k_offsets = (token_slots[:, None] * stride_buf_n
                         + h_kv_offset
                         + offs_d[None, :] * stride_buf_d)
            k = tl.load(k_buf_ptr + k_offsets,
                         mask=sel_mask[:, None], other=0.0)

            # --- QK MMA (Tensor Core) ---
            qk = tl.dot(q, tl.trans(k))

            # --- Fine-grained within-tile mask (ONLY for visited tiles) ---
            mask_offsets = ((tree_mask_row_offset + offs_m)[:, None] * tree_mask_stride_m
                            + offs_n[None, :] * tree_mask_stride_n)
            tree_mask_tile = tl.load(tree_mask_ptr + mask_offsets,
                                      mask=sel_mask[None, :], other=float('-inf'))

            boundary = offs_n[None, :] < N_tree
            qk = qk * qk_scale + tree_mask_tile + tl.where(boundary, 0.0, -1.0e6)

            # --- Online softmax ---
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]

            # --- PV MMA (Tensor Core) ---
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
def _attn_fwd_tile_bitmap(
    Q, K_buf, V_buf, O,
    KV_indices,
    Tree_mask,
    Bitmaps,
    Active_blocks,      # [num_active] int32 block IDs or dummy (ignored when USE_QUEST!=1)
    Quest_bitmap,       # [W_past] int64 bitmap or dummy (ignored when USE_QUEST!=2)
    sm_scale,
    stride_qm, stride_qk,
    stride_buf_n, stride_buf_h, stride_buf_d,
    stride_om, stride_ok,
    stride_tm, stride_tn,
    stride_bm_q,
    N_CTX_Q,
    past_len,
    num_active_blocks,  # runtime: number of active blocks (0 = dense)
    H_KV: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    W: tl.constexpr,
    USE_QUEST: tl.constexpr,  # 0=dense, 1=block-list, 2=bitmap-masked dense
):
    """Tile-level bitmap attention kernel.

    Grid: [num_q_blocks, H_q]
    Each program handles one Q-block x one Q-head.
    GQA: maps Q head -> KV head via integer division.

    USE_QUEST=0: Stage 1 dense sequential loop.
    USE_QUEST=1: Stage 1 block-list (tl.range over compact IDs).
    USE_QUEST=2: Stage 1 bitmap-masked dense (tl.range over all blocks, bitmap skip).
    """
    dtype = tl.float16
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)

    off_h_kv = off_h_q // GQA_GROUP
    h_kv_offset = off_h_kv * stride_buf_h

    qk_scale = sm_scale * 1.44269504

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

    # ---- Stage 1: Past context (paged reads) ----
    if past_len > 0:
        if USE_QUEST == 1:
            # Quest block list: only visit important blocks via tl.range
            acc, l_i, m_i = _tile_bitmap_stage1_sparse(
                acc, l_i, m_i, q,
                K_buf, V_buf,
                KV_indices,
                stride_buf_n, stride_buf_h, stride_buf_d,
                h_kv_offset,
                dtype, qk_scale, past_len,
                Active_blocks,
                num_active_blocks,
                BLOCK_M, HEAD_DIM, BLOCK_N)
        elif USE_QUEST == 2:
            # Bitmap-masked dense: tl.range over all blocks, bitmap skip
            acc, l_i, m_i = _tile_bitmap_stage1_bitmap_masked(
                acc, l_i, m_i, q,
                K_buf, V_buf,
                KV_indices,
                stride_buf_n, stride_buf_h, stride_buf_d,
                h_kv_offset,
                dtype, qk_scale, past_len,
                Quest_bitmap,
                BLOCK_M, HEAD_DIM, BLOCK_N)
        else:
            # Dense: visit all blocks
            acc, l_i, m_i = _tile_bitmap_stage1(
                acc, l_i, m_i, q,
                K_buf, V_buf,
                KV_indices,
                stride_buf_n, stride_buf_h, stride_buf_d,
                h_kv_offset,
                dtype, qk_scale, past_len,
                BLOCK_M, HEAD_DIM, BLOCK_N)

    # ---- Stage 2: Tile-level bitmap-driven tree region ----
    bitmap_ptr = Bitmaps + start_m * stride_bm_q
    tree_mask_row_offset = start_m * BLOCK_M

    acc, l_i, m_i = _tile_bitmap_stage2(
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


def tile_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                           tree_mask_2d, bitmaps, W, past_len,
                           sm_scale, H_kv,
                           BLOCK_M=32, BLOCK_N=16,
                           quest_block_ids=None,
                           quest_bitmap=None):
    """Launch tile-level bitmap attention kernel.

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
        BLOCK_M: int — Q tile size
        BLOCK_N: int — KV tile size (default 16 for finer granularity)
        quest_block_ids: [num_active] int32 — Quest block-list (USE_QUEST=1)
        quest_bitmap: [W_past] int64 — Quest bitmap for masked dense (USE_QUEST=2)

    Returns:
        output: [H_q, N_tree, D] fp16
    """
    H_q, N_tree, D = q_heads.shape
    output = torch.empty_like(q_heads)
    gqa_group = H_q // H_kv
    device = q_heads.device

    dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
    dummy_i64 = torch.zeros(1, dtype=torch.int64, device=device)

    # Quest mode selection: bitmap-masked > block-list > dense
    if quest_bitmap is not None and len(quest_bitmap) > 0:
        use_quest = 2
        num_active = 0
        active_blocks = dummy_i32
        quest_bm = quest_bitmap
    elif quest_block_ids is not None and len(quest_block_ids) > 0:
        use_quest = 1
        num_active = len(quest_block_ids)
        active_blocks = quest_block_ids
        quest_bm = dummy_i64
    else:
        use_quest = 0
        num_active = 0
        active_blocks = dummy_i32
        quest_bm = dummy_i64

    grid = (triton.cdiv(N_tree, BLOCK_M), H_q)

    _attn_fwd_tile_bitmap[grid](
        q_heads, k_buf, v_buf, output,
        kv_indices,
        tree_mask_2d,
        bitmaps,
        active_blocks,
        quest_bm,
        sm_scale,
        q_heads.stride(1), q_heads.stride(2),
        k_buf.stride(0), k_buf.stride(1), k_buf.stride(2),
        output.stride(1), output.stride(2),
        tree_mask_2d.stride(0), tree_mask_2d.stride(1),
        bitmaps.stride(0),
        N_tree,
        past_len,
        num_active,
        H_KV=H_kv,
        GQA_GROUP=gqa_group,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        W=W,
        USE_QUEST=use_quest,
    )

    return output


# ============================================================================
# BATCHED KERNEL: Single launch for multiple requests
# ============================================================================

@triton.jit
def _attn_fwd_tile_bitmap_batched(
    Q, K_buf, V_buf, O,
    KV_indices,          # [bs, max_total_kv]
    Tree_mask,           # [N_tree, N_tree]
    Bitmaps,             # [num_q_blocks, W]
    Past_lens,           # [bs] int32
    sm_scale,
    stride_q_bs, stride_q_h, stride_q_m, stride_q_d,
    stride_buf_n, stride_buf_h, stride_buf_d,
    stride_o_bs, stride_o_h, stride_o_m, stride_o_d,
    stride_tm, stride_tn,
    stride_bm_q,
    stride_kv_bs,        # stride for batch dim of KV_indices
    N_CTX_Q,
    H_KV: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    W: tl.constexpr,
):
    """Batched tile-bitmap attention kernel.

    Grid: [bs, num_q_blocks, H_q]
    Each program handles one request x one Q-block x one Q-head.

    Uses dense Stage 1 (Quest selection baked into kv_indices).
    Shared tree_mask and bitmaps across all requests.
    """
    dtype = tl.float16
    batch_id = tl.program_id(0)
    start_m = tl.program_id(1)
    off_h_q = tl.program_id(2)

    off_h_kv = off_h_q // GQA_GROUP
    h_kv_offset = off_h_kv * stride_buf_h

    qk_scale = sm_scale * 1.44269504

    # Per-request past length
    past_len = tl.load(Past_lens + batch_id)

    # Per-request Q/O base pointers
    q_base = Q + batch_id * stride_q_bs + off_h_q * stride_q_h
    o_base = O + batch_id * stride_o_bs + off_h_q * stride_o_h

    # Per-request kv_indices pointer
    kv_indices_ptr = KV_indices + batch_id * stride_kv_bs

    # Load Q block
    qo_offset = start_m * BLOCK_M
    offs_m = qo_offset + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m[:, None] < N_CTX_Q
    q = tl.load(q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
                mask=q_mask, other=0.0)

    # Init accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ---- Stage 1: Dense past context (paged reads) ----
    if past_len > 0:
        acc, l_i, m_i = _tile_bitmap_stage1(
            acc, l_i, m_i, q,
            K_buf, V_buf,
            kv_indices_ptr,
            stride_buf_n, stride_buf_h, stride_buf_d,
            h_kv_offset,
            dtype, qk_scale, past_len,
            BLOCK_M, HEAD_DIM, BLOCK_N)

    # ---- Stage 2: Tile-level bitmap-driven tree region ----
    bitmap_ptr = Bitmaps + start_m * stride_bm_q
    tree_mask_row_offset = start_m * BLOCK_M

    acc, l_i, m_i = _tile_bitmap_stage2(
        acc, l_i, m_i, q,
        K_buf, V_buf,
        kv_indices_ptr,
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
    tl.store(o_base + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d,
             acc_out, mask=q_valid_mask)


def tile_bitmap_attention_batched(q_batched, k_buf, v_buf, kv_indices_batched,
                                   tree_mask_2d, bitmaps, W, past_lens,
                                   sm_scale, H_kv,
                                   BLOCK_M=32, BLOCK_N=32):
    """Launch batched tile-bitmap attention kernel.

    Single kernel launch for all requests. Quest selection should be
    pre-baked into kv_indices_batched (rebuilt kv_indices).

    Args:
        q_batched: [bs, H_q, N_tree, D] fp16
        k_buf, v_buf: [max_tokens, H_kv, D] fp16 shared paged buffer
        kv_indices_batched: [bs, max_total_kv] int32 padded
        tree_mask_2d: [N_tree, N_tree] fp16 shared (0 or -inf)
        bitmaps: [num_q_blocks, W] int64 shared
        W: int — number of uint64 words per q_block
        past_lens: [bs] int32 — per-request past context length
        sm_scale: float
        H_kv: int — number of KV heads
        BLOCK_M, BLOCK_N: int — tile sizes

    Returns:
        output: [bs, H_q, N_tree, D] fp16
    """
    bs, H_q, N_tree, D = q_batched.shape
    output = torch.empty_like(q_batched)
    gqa_group = H_q // H_kv

    grid = (bs, triton.cdiv(N_tree, BLOCK_M), H_q)

    _attn_fwd_tile_bitmap_batched[grid](
        q_batched, k_buf, v_buf, output,
        kv_indices_batched,
        tree_mask_2d, bitmaps, past_lens,
        sm_scale,
        q_batched.stride(0), q_batched.stride(1), q_batched.stride(2), q_batched.stride(3),
        k_buf.stride(0), k_buf.stride(1), k_buf.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        tree_mask_2d.stride(0), tree_mask_2d.stride(1),
        bitmaps.stride(0),
        kv_indices_batched.stride(0),
        N_tree,
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
    """Dense attention reference (PyTorch, fp32 accumulation)."""
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


def validate(device='cuda'):
    """Validate tile bitmap kernel against dense reference."""
    from eagle_tree_choices import EAGLE_TREES

    _ensure_triton_allocator(device)

    for tree_name in ["mc_sim_7b_63", "regular_512"]:
        tree_choices = EAGLE_TREES[tree_name]
        N_tree = len(tree_choices) + 1

        for past_len in [0, 64, 256, 1024]:
            for BLOCK_N in [16, 32]:
                total_kv = past_len + N_tree
                H_q = 8
                H_kv = 2
                D = 128
                sm_scale = 1.0 / (D ** 0.5)

                print(f"\n{'='*60}")
                print(f"Validating tile bitmap: {tree_name} "
                      f"(N_tree={N_tree}, past={past_len}, "
                      f"BLOCK_N={BLOCK_N}, H_q={H_q}, H_kv={H_kv})")
                print(f"{'='*60}")

                torch.manual_seed(42)

                max_tokens = total_kv + 200
                k_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                                    device=device)
                v_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                                    device=device)

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

                # Dense reference
                k_full = k_buf[kv_indices.long()]
                v_full = v_buf[kv_indices.long()]
                out_dense = _dense_reference(q_heads, k_full, v_full,
                                             tree_mask_bool, past_len,
                                             sm_scale, H_kv)

                # Tile bitmap
                bitmaps, W = compute_tile_bitmap(tree_mask_bool,
                                                  BLOCK_N=BLOCK_N)

                num_kv_tiles = (N_tree + BLOCK_N - 1) // BLOCK_N
                num_set_bits = 0
                for qb in range(bitmaps.shape[0]):
                    for ww in range(W):
                        val = bitmaps[qb, ww].item()
                        num_set_bits += bin(val & 0xFFFFFFFFFFFFFFFF).count('1')
                total_tiles = bitmaps.shape[0] * num_kv_tiles
                skip_pct = (1.0 - num_set_bits / total_tiles) * 100 if total_tiles > 0 else 0
                print(f"  Tiles: {num_kv_tiles}, W={W}, "
                      f"set bits={num_set_bits}/{total_tiles}, "
                      f"skip={skip_pct:.1f}%")

                tree_mask_2d = torch.where(
                    tree_mask_bool,
                    torch.zeros(1, device=device, dtype=torch.float16),
                    torch.full((1,), float('-inf'), device=device,
                               dtype=torch.float16),
                )

                out_tile = tile_bitmap_attention(
                    q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                    bitmaps, W, past_len, sm_scale, H_kv,
                    BLOCK_N=BLOCK_N)

                max_diff = (out_tile - out_dense).abs().max().item()
                mean_diff = (out_tile - out_dense).abs().mean().item()
                cos_sim = F.cosine_similarity(
                    out_tile.float().flatten(),
                    out_dense.float().flatten(), dim=0
                ).item()
                print(f"  Max diff:  {max_diff:.2e}")
                print(f"  Mean diff: {mean_diff:.2e}")
                print(f"  Cos sim:   {cos_sim:.6f}")
                passed = max_diff < 5e-2
                print(f"  {'PASSED' if passed else 'FAILED'}")

                if not passed:
                    print(f"  WARNING: max_diff {max_diff:.2e} exceeds threshold 5e-2")

                # Validate Quest bitmap-masked mode (100% budget = all blocks)
                if past_len > 0:
                    num_past_blocks = (past_len + BLOCK_N - 1) // BLOCK_N
                    W_past = (num_past_blocks + 63) // 64
                    # All-ones bitmap: every block is active
                    all_ones = torch.full((W_past,), -1, dtype=torch.int64,
                                         device=device)  # -1 = 0xFFFF...
                    out_bm_masked = tile_bitmap_attention(
                        q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                        bitmaps, W, past_len, sm_scale, H_kv,
                        BLOCK_N=BLOCK_N, quest_bitmap=all_ones)
                    diff_bm = (out_bm_masked - out_dense).abs().max().item()
                    status = "PASSED" if diff_bm < 5e-2 else "FAILED"
                    print(f"  Bitmap-masked (100% budget): max_diff={diff_bm:.2e} {status}")


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark(device='cuda', warmup=20, rep=100):
    """Microbenchmark tile bitmap vs paged bitmap at different BLOCK_N."""
    import time
    from eagle_tree_choices import EAGLE_TREES, regular_tree_512

    _ensure_triton_allocator(device)

    tree_choices = regular_tree_512
    N_tree = len(tree_choices) + 1
    H_q = 8
    H_kv = 2
    D = 128
    sm_scale = 1.0 / (D ** 0.5)

    # Build tree mask once
    tree_mask_bool = torch.zeros(N_tree, N_tree, dtype=torch.bool, device=device)
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

    tree_mask_2d = torch.where(
        tree_mask_bool,
        torch.zeros(1, device=device, dtype=torch.float16),
        torch.full((1,), float('-inf'), device=device, dtype=torch.float16),
    )

    # Also import paged_bitmap for comparison
    try:
        from paged_bitmap_attention import (
            paged_bitmap_attention,
            compute_bitmap_from_mask_gpu,
        )
        has_paged_bitmap = True
    except ImportError:
        has_paged_bitmap = False

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

        print(f"\n{'='*60}")
        print(f"past_len={past_len}, N_tree={N_tree}, total_kv={total_kv}")
        print(f"H_q={H_q}, H_kv={H_kv}, D={D}")
        print(f"{'='*60}")

        for BLOCK_N in [16, 32]:
            bitmaps, W = compute_tile_bitmap(tree_mask_bool, BLOCK_N=BLOCK_N)

            # Tile reduction stats
            num_kv_tiles = (N_tree + BLOCK_N - 1) // BLOCK_N
            num_set_bits = 0
            for qb in range(bitmaps.shape[0]):
                for ww in range(W):
                    val = bitmaps[qb, ww].item()
                    num_set_bits += bin(val & 0xFFFFFFFFFFFFFFFF).count('1')
            total_tiles = bitmaps.shape[0] * num_kv_tiles
            skip_pct = (1.0 - num_set_bits / total_tiles) * 100 if total_tiles > 0 else 0

            # Warmup
            for _ in range(warmup):
                tile_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                      tree_mask_2d, bitmaps, W, past_len,
                                      sm_scale, H_kv, BLOCK_N=BLOCK_N)
            torch.cuda.synchronize()

            # Kernel only
            t0 = time.perf_counter()
            for _ in range(rep):
                tile_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                      tree_mask_2d, bitmaps, W, past_len,
                                      sm_scale, H_kv, BLOCK_N=BLOCK_N)
            torch.cuda.synchronize()
            t_kernel = (time.perf_counter() - t0) / rep * 1000

            # Full pipeline (bitmap + kernel)
            for _ in range(warmup):
                bm, w = compute_tile_bitmap(tree_mask_bool, BLOCK_N=BLOCK_N)
                tile_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                      tree_mask_2d, bm, w, past_len,
                                      sm_scale, H_kv, BLOCK_N=BLOCK_N)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(rep):
                bm, w = compute_tile_bitmap(tree_mask_bool, BLOCK_N=BLOCK_N)
                tile_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                      tree_mask_2d, bm, w, past_len,
                                      sm_scale, H_kv, BLOCK_N=BLOCK_N)
            torch.cuda.synchronize()
            t_full = (time.perf_counter() - t0) / rep * 1000

            print(f"\n  BLOCK_N={BLOCK_N}: tiles={num_kv_tiles}, W={W}, "
                  f"skip={skip_pct:.1f}%")
            print(f"    Kernel only:    {t_kernel:.3f} ms")
            print(f"    Full pipeline:  {t_full:.3f} ms")
            print(f"    Bitmap compute: {t_full - t_kernel:.3f} ms")

        # Compare against paged_bitmap (BLOCK_N=32) if available
        if has_paged_bitmap:
            bitmaps_32, W_32 = compute_bitmap_from_mask_gpu(tree_mask_bool)

            for _ in range(warmup):
                paged_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                       tree_mask_2d, bitmaps_32, W_32,
                                       past_len, sm_scale, H_kv)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(rep):
                paged_bitmap_attention(q_heads, k_buf, v_buf, kv_indices,
                                       tree_mask_2d, bitmaps_32, W_32,
                                       past_len, sm_scale, H_kv)
            torch.cuda.synchronize()
            t_paged = (time.perf_counter() - t0) / rep * 1000

            print(f"\n  paged_bitmap (BLOCK_N=32 ref): {t_paged:.3f} ms")


# ============================================================================
# BATCHED VALIDATION
# ============================================================================

def validate_batched(device='cuda'):
    """Validate batched kernel against single-request kernel."""
    from eagle_tree_choices import EAGLE_TREES

    _ensure_triton_allocator(device)

    for tree_name in ["mc_sim_7b_63", "regular_512"]:
        tree_choices = EAGLE_TREES[tree_name]
        N_tree = len(tree_choices) + 1

        for bs in [1, 2, 4, 8]:
            for BLOCK_N in [16, 32]:
                H_q = 8
                H_kv = 2
                D = 128
                sm_scale = 1.0 / (D ** 0.5)

                # Random per-request past lengths
                torch.manual_seed(42)
                past_lens_list = [torch.randint(4096, 65536, (1,)).item() for _ in range(bs)]

                max_total_kv = max(p + N_tree for p in past_lens_list)
                max_tokens = max_total_kv + 500

                k_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
                                    device=device)
                v_buf = torch.randn(max_tokens, H_kv, D, dtype=torch.float16,
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

                bitmaps, W = compute_tile_bitmap(tree_mask_bool, BLOCK_N=BLOCK_N)
                tree_mask_2d = torch.where(
                    tree_mask_bool,
                    torch.zeros(1, device=device, dtype=torch.float16),
                    torch.full((1,), float('-inf'), device=device,
                               dtype=torch.float16),
                )

                # Per-request data
                kv_indices_list = []
                q_heads_list = []
                for i in range(bs):
                    past_len = past_lens_list[i]
                    total_kv = past_len + N_tree
                    perm = torch.randperm(max_tokens, device=device)
                    kv_indices = perm[:total_kv].to(torch.int32)
                    kv_indices_list.append(kv_indices)

                    q_h = torch.randn(H_q, N_tree, D, dtype=torch.float16,
                                      device=device)
                    q_heads_list.append(q_h)

                # --- Single-request reference ---
                ref_outputs = []
                for i in range(bs):
                    out_i = tile_bitmap_attention(
                        q_heads_list[i], k_buf, v_buf, kv_indices_list[i],
                        tree_mask_2d, bitmaps, W, past_lens_list[i],
                        sm_scale, H_kv, BLOCK_N=BLOCK_N)
                    ref_outputs.append(out_i)
                ref_stacked = torch.stack(ref_outputs)  # [bs, H_q, N_tree, D]

                # --- Batched kernel ---
                kv_indices_batched = torch.zeros(bs, max_total_kv,
                                                  dtype=torch.int32, device=device)
                past_lens_t = torch.zeros(bs, dtype=torch.int32, device=device)
                q_batched = torch.stack(q_heads_list)  # [bs, H_q, N_tree, D]

                for i in range(bs):
                    n = len(kv_indices_list[i])
                    kv_indices_batched[i, :n] = kv_indices_list[i]
                    past_lens_t[i] = past_lens_list[i]

                out_batched = tile_bitmap_attention_batched(
                    q_batched, k_buf, v_buf, kv_indices_batched,
                    tree_mask_2d, bitmaps, W, past_lens_t,
                    sm_scale, H_kv, BLOCK_N=BLOCK_N)

                max_diff = (out_batched - ref_stacked).abs().max().item()
                mean_diff = (out_batched - ref_stacked).abs().mean().item()
                passed = max_diff < 1e-3
                print(f"{tree_name} bs={bs} BN={BLOCK_N} "
                      f"past=[{min(past_lens_list)}-{max(past_lens_list)}]: "
                      f"max_diff={max_diff:.2e} mean={mean_diff:.2e} "
                      f"{'PASSED' if passed else 'FAILED'}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
    elif len(sys.argv) > 1 and sys.argv[1] == "validate_batched":
        validate_batched()
    elif len(sys.argv) > 1 and sys.argv[1] == "bench":
        benchmark()
    else:
        print("Usage: python tile_bitmap_attention.py [validate|validate_batched|bench]")
