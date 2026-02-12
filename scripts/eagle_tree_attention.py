import pytest
import torch
import torch.nn.functional as F
import os
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

TRITON_AVAILABLE = True

def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

# use fixed block sizeconfig
configs = [
    triton.Config(dict(BLOCK_M=32, BLOCK_N=32), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook)
]
# autotune configs
# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
#     for BM in [8, 16, 32, 64, 128]\
#     for BN in [32, 64, 128]\
#     for s in NUM_STAGES_OPTIONS \
#     for w in [4, 8]\
# ]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]

def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9

def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)

def prune_invalid_configs(configs, named_args, **kwargs):
    # Support both old (N_CTX) and new (N_CTX_Q, N_CTX_KV) parameter names for backward compatibility
    N_CTX_Q = kwargs.get("N_CTX_Q", kwargs.get("N_CTX", 0))
    N_CTX_KV = kwargs.get("N_CTX_KV", kwargs.get("N_CTX", 0))
    STAGE = kwargs["STAGE"]

    # Filter out configs where BLOCK_M > N_CTX_Q (Q length)
    # Filter out configs where BLOCK_M < BLOCK_N when causal is True
    return [
        conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX_Q and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]



@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    desc_tree_mask, #
                    tree_mask_row_offset, #  Row offset: batch*head*Q_row position
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX_Q: tl.constexpr, N_CTX_KV: tl.constexpr,  #
                    warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    # For cross-attention: Q[i] attends to KV[0..past_len+i] where past_len = N_CTX_KV - N_CTX_Q
    # This means each Q position can see all past context plus causally-masked current tokens
    past_len = N_CTX_KV - N_CTX_Q

    if STAGE == 1:
        # Off-diagonal: attend to PAST CONTEXT ONLY [0, past_len)
        # All Q positions can attend to full past context without mask
        lo, hi = 0, past_len
    elif STAGE == 2:
        # On-diagonal: DRAFT REGION with tree mask
        # Q block at start_m processes draft tokens [past_len, past_len + (start_m + 1) * BLOCK_M)
        # But each Q position can only see up to its own position (enforced by tree mask)
        lo = past_len
        hi = past_len + (start_m + 1) * BLOCK_M
    # causal = False
    else:
        lo, hi = 0, N_CTX_KV

    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo

    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)

        # Boundary mask: mask out KV positions >= N_CTX_KV (handles non-power-of-2 lengths)
        boundary_mask = (start_n + offs_n[None, :]) < N_CTX_KV

        if STAGE == 2:
            # Tree mask: load the correct block based on current KV position
            # Tree mask is [N_CTX_Q, N_CTX_Q], maps to KV positions [past_len, past_len + N_CTX_Q)
            tree_col = start_n - past_len
            tree_mask_block = desc_tree_mask.load([tree_mask_row_offset, tree_col])
            qk = qk * qk_scale + tree_mask_block + tl.where(boundary_mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        elif STAGE == 1:
            # STAGE 1: Off-diagonal in causal mode
            # For cross-attention with tree mask, this handles past context [0, past_len)
            # Must mask out any draft positions >= past_len that might be in this block
            # (This happens when the block spans the past_len boundary)
            past_ctx_mask = (start_n + offs_n[None, :]) < past_len
            combined_mask = boundary_mask & past_ctx_mask
            qk = qk * qk_scale + tl.where(combined_mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        else:
            # STAGE 3: Non-causal mode - all positions are visible
            qk = qk * qk_scale + tl.where(boundary_mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i

# @triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX_Q", "N_CTX_KV", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
#                  prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, desc_tree_mask, N_CTX_Q, N_CTX_KV,  #
              N_CTX_Q_PADDED, N_CTX_KV_PADDED,  # Padded dimensions for per-head offsets
              TREE_MASK_COLS,  # Padded tree_mask column size
              TREE_MASK_ROWS_PER_HEAD,  # Padded tree_mask rows per head
            #   stride_oz, stride_oh, stride_om, stride_od,  #
              HEAD_DIM: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              BLOCK_M: tl.constexpr = 32,
              BLOCK_N: tl.constexpr = 32,
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0) # Q block index = q_len / q_block_size
    off_hz = tl.program_id(1) # batch * Head block index
    off_z = off_hz // H # batch index
    off_h = off_hz % H # head index

    # Separate dimensions for Q and K/V (using PADDED dimensions for memory layout)
    y_dim_q = Z * H * N_CTX_Q_PADDED
    y_dim_kv = Z * H * N_CTX_KV_PADDED
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim_q, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim_kv], strides=[N_CTX_KV_PADDED, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim_kv, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim_kv, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim_q, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    # Tree mask shape: [B*H*TREE_MASK_ROWS_PER_HEAD, TREE_MASK_COLS] (padded per head)
    # Block shape: [BLOCK_M, BLOCK_N] to load mask blocks aligned with Q/KV blocks
    y_dim_tree_mask = Z * H * TREE_MASK_ROWS_PER_HEAD
    desc_tree_mask = _maybe_make_tensor_desc(desc_tree_mask, shape=[y_dim_tree_mask, TREE_MASK_COLS], strides=[TREE_MASK_COLS, 1], block_shape=[BLOCK_M, BLOCK_N])

    # Separate offsets for Q and K/V (using PADDED dimensions to avoid cross-head access)
    offset_q_y = off_z * (N_CTX_Q_PADDED * H) + off_h * N_CTX_Q_PADDED  # batch and head offset
    offset_kv_y = off_z * (N_CTX_KV_PADDED * H) + off_h * N_CTX_KV_PADDED
    qo_offset_y = offset_q_y + start_m * BLOCK_M

    # Tree mask row offset: includes batch, head (with padding), AND Q block position
    # Each head now has TREE_MASK_ROWS_PER_HEAD rows (padded from N_CTX_Q)
    tree_mask_row_offset = off_z * (H * TREE_MASK_ROWS_PER_HEAD) + off_h * TREE_MASK_ROWS_PER_HEAD + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # first index of the Q block and the continuous index in this block
    offs_n = tl.arange(0, BLOCK_N) # continuous index in the K/V block
    q_valid_mask = offs_m < N_CTX_Q

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        desc_tree_mask, #
                                        tree_mask_row_offset, #
                                        offset_kv_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n,  #
                                        N_CTX_Q, N_CTX_KV,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        desc_tree_mask, #
                                        tree_mask_row_offset, #
                                        offset_kv_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n,  #
                                        N_CTX_Q, N_CTX_KV,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue for block fusion
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    # M matrix size is based on Q length - use masked store for boundary handling
    m_ptrs = M + off_hz * N_CTX_Q + offs_m

    out_mask = q_valid_mask[:, None]
    # Store m_i with 1D mask (m_i shape: [BLOCK_M])
    tl.store(m_ptrs, m_i, mask=q_valid_mask)
    acc_masked = tl.where(out_mask, acc.to(dtype), 0.0) # mask out invalid positions
    desc_o.store([qo_offset_y, 0], acc_masked) # store acc_masked to o matrix

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, tree_mask, sm_scale, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        # Extract Q and KV sequence lengths (support cross-attention)
        N_CTX_Q = q.shape[2]  # Q sequence length
        N_CTX_KV = k.shape[2]  # K/V sequence length
        assert k.shape[2] == v.shape[2], "K and V must have same sequence length"

        B, H = q.shape[0], q.shape[1]

        # Determine stage based on tree_mask:
        # - tree_mask is not None → stage=3 (causal with tree mask)
        # - tree_mask is None → stage=1 (non-causal, full attention)
        # Use max BLOCK_N from autotune configs for padding to ensure all configs work
        # The kernel loads BLOCK_N consecutive rows, so padding must accommodate the largest block
        MAX_BLOCK_N = 128  # Must match max value in autotune configs

        if tree_mask is not None:
            # Expand tree_mask if it's broadcast (shape [1, 1, ...])
            if tree_mask.shape[0] == 1 and B > 1:
                tree_mask = tree_mask.expand(B, -1, -1, -1)
            if tree_mask.shape[1] == 1 and H > 1:
                tree_mask = tree_mask.expand(-1, H, -1, -1)

            # Make contiguous, convert to same dtype as q, and flatten to [B*H*N_CTX_Q, N_CTX_Q]
            tree_mask = tree_mask.contiguous()
            # Convert tree_mask to same dtype as q for consistent computation
            tree_mask = tree_mask.to(q.dtype)
            tree_mask_flat = tree_mask.view(B * H * N_CTX_Q, N_CTX_Q)

            # Pad tree_mask to avoid out-of-bounds access
            # Columns: pad to multiple of BLOCK_N, filled with -inf (masked)
            # Rows: need to pad for last Q block - the kernel loads BLOCK_M rows at a time
            # For each head, we have N_CTX_Q rows. The last Q block may extend beyond.
            pad_cols = (MAX_BLOCK_N - N_CTX_Q % MAX_BLOCK_N) % MAX_BLOCK_N
            pad_rows = (MAX_BLOCK_N - N_CTX_Q % MAX_BLOCK_N) % MAX_BLOCK_N  # Same padding for rows
            # Vectorized padding: reshape to 3D, pad, reshape back
            # tree_mask_flat is [B*H*N_CTX_Q, N_CTX_Q], reshape to [B*H, N_CTX_Q, N_CTX_Q]
            tree_mask_3d = tree_mask_flat.view(B * H, N_CTX_Q, N_CTX_Q)
            # F.pad format: (left, right, top, bottom) for last two dims
            # Pad columns (last dim) by pad_cols, rows (second-to-last) by pad_rows
            tree_mask_padded_3d = F.pad(tree_mask_3d, (0, pad_cols, 0, pad_rows), value=float('-inf'))
            # Reshape back to 2D: [B*H * (N_CTX_Q + pad_rows), N_CTX_Q + pad_cols]
            tree_mask_flat = tree_mask_padded_3d.view(B * H * (N_CTX_Q + pad_rows), N_CTX_Q + pad_cols)

            stage = 3  # Causal mode with tree mask
            # Debug: print padded tree_mask shape
            # print(f"[DEBUG] tree_mask_flat padded shape: {tree_mask_flat.shape}, rows_per_head={N_CTX_Q + pad_rows}")
        else:
            # No tree mask - create dummy and use non-causal mode
            # Also pad to be a multiple of BLOCK_N
            pad_cols = (MAX_BLOCK_N - N_CTX_Q % MAX_BLOCK_N) % MAX_BLOCK_N
            pad_rows = (MAX_BLOCK_N - N_CTX_Q % MAX_BLOCK_N) % MAX_BLOCK_N
            padded_cols = N_CTX_Q + pad_cols
            padded_rows = B * H * (N_CTX_Q + pad_rows)
            tree_mask_flat = torch.zeros((padded_rows, padded_cols), device=q.device, dtype=q.dtype)
            stage = 1  # Non-causal mode

        # Reshape tensors to 2D with per-head padding to avoid cross-head memory access
        pad_q = (MAX_BLOCK_N - N_CTX_Q % MAX_BLOCK_N) % MAX_BLOCK_N
        pad_kv = (MAX_BLOCK_N - N_CTX_KV % MAX_BLOCK_N) % MAX_BLOCK_N
        N_CTX_Q_PADDED = N_CTX_Q + pad_q
        N_CTX_KV_PADDED = N_CTX_KV + pad_kv

        # Vectorized padding: reshape to 3D, pad second dim, reshape to 2D
        q_contig = q.contiguous().view(B * H, N_CTX_Q, HEAD_DIM_K)
        k_contig = k.contiguous().view(B * H, N_CTX_KV, HEAD_DIM_K)
        v_contig = v.contiguous().view(B * H, N_CTX_KV, HEAD_DIM_K)

        # F.pad format for 3D: (left, right) for last dim, then (top, bottom) for second-to-last
        # We want to pad the sequence dimension (dim=1) by pad_q/pad_kv
        q_padded = F.pad(q_contig, (0, 0, 0, pad_q), value=0)  # [B*H, N_CTX_Q_PADDED, HEAD_DIM_K]
        k_padded = F.pad(k_contig, (0, 0, 0, pad_kv), value=0)  # [B*H, N_CTX_KV_PADDED, HEAD_DIM_K]
        v_padded = F.pad(v_contig, (0, 0, 0, pad_kv), value=0)  # [B*H, N_CTX_KV_PADDED, HEAD_DIM_K]

        # Reshape to 2D for kernel
        q_flat = q_padded.view(B * H * N_CTX_Q_PADDED, HEAD_DIM_K)
        k_flat = k_padded.view(B * H * N_CTX_KV_PADDED, HEAD_DIM_K)
        v_flat = v_padded.view(B * H * N_CTX_KV_PADDED, HEAD_DIM_K)
        o_flat = torch.empty((B * H * N_CTX_Q_PADDED, HEAD_DIM_K), device=q.device, dtype=q.dtype)

        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Use device_descriptor for Hopper + warpspec.
        y_dim_q = B * H * N_CTX_Q
        y_dim_kv = B * H * N_CTX_KV
        # Get actual tree_mask dimensions (may be padded)
        tree_mask_cols = tree_mask_flat.shape[1]
        # Padded rows per head = total rows / (B * H)
        tree_mask_rows_per_head = tree_mask_flat.shape[0] // (B * H)
        y_dim_tree_mask = tree_mask_flat.shape[0]

        # Ensure all device-dependent code runs on correct device for multi-GPU support
        # Get device index for explicit device handling
        device_idx = q.device.index if q.device.index is not None else 0
        prev_device = torch.cuda.current_device()
        torch.cuda.set_device(device_idx)

        try:
            # Use TensorDescriptor only on supported hardware
            use_tensor_desc = supports_host_descriptor() and not (is_hopper() and warp_specialize)
            if use_tensor_desc:
                # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
                dummy_block = [1, 1]
                desc_q = TensorDescriptor(q_flat, shape=[y_dim_q, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                if q.dtype == torch.float8_e5m2:
                    desc_v = TensorDescriptor(v_flat, shape=[HEAD_DIM_K, y_dim_kv], strides=[N_CTX_KV, 1],
                                              block_shape=dummy_block)
                else:
                    desc_v = TensorDescriptor(v_flat, shape=[y_dim_kv, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                              block_shape=dummy_block)
                desc_k = TensorDescriptor(k_flat, shape=[y_dim_kv, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_o = TensorDescriptor(o_flat, shape=[y_dim_q, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_tree_mask = TensorDescriptor(tree_mask_flat, shape=[y_dim_tree_mask, tree_mask_cols], strides=[tree_mask_cols, 1], block_shape=dummy_block)
            else:
                desc_q = q_flat
                desc_v = v_flat
                desc_k = k_flat
                desc_o = o_flat
                desc_tree_mask = tree_mask_flat

            def grid(META):
                return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

            ctx.grid = grid
            if is_blackwell() and warp_specialize:
                if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                    extra_kern_args["maxnreg"] = 168
                else:
                    extra_kern_args["maxnreg"] = 80

            def alloc_fn(size: int, align: int, _):
                return torch.empty(size, dtype=torch.int8, device=q.device)

            triton.set_allocator(alloc_fn)

            _attn_fwd[grid](
                sm_scale, M,
                q.shape[0], q.shape[1],  #
                desc_q, desc_k, desc_v, desc_o,  #
                desc_tree_mask, #
                N_CTX_Q=q.shape[2],  #
                N_CTX_KV=k.shape[2],  #
                N_CTX_Q_PADDED=N_CTX_Q_PADDED,  #
                N_CTX_KV_PADDED=N_CTX_KV_PADDED,  #
                TREE_MASK_COLS=tree_mask_cols,  # Padded tree_mask column size
                TREE_MASK_ROWS_PER_HEAD=tree_mask_rows_per_head,  # Padded tree_mask rows per head
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                warp_specialize=warp_specialize,  #
                IS_HOPPER=is_hopper(),  #
                **extra_kern_args)
        finally:
            # Restore previous device
            torch.cuda.set_device(prev_device)

        # Vectorized output extraction: reshape to 3D, slice off padding, reshape to 4D
        # o_flat is [B*H*N_CTX_Q_PADDED, HEAD_DIM_K]
        o_padded_3d = o_flat.view(B * H, N_CTX_Q_PADDED, HEAD_DIM_K)
        # Remove padding by slicing first N_CTX_Q rows per head
        o_3d = o_padded_3d[:, :N_CTX_Q, :].contiguous()  # [B*H, N_CTX_Q, HEAD_DIM_K]
        # Reshape to [B, H, N_CTX_Q, HEAD_DIM_K]
        o = o_3d.view(B, H, N_CTX_Q, HEAD_DIM_K)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.desc_tree_mask = desc_tree_mask
        ctx.N_CTX_Q = N_CTX_Q  # Save for backward pass
        ctx.N_CTX_KV = N_CTX_KV  # Save for backward pass
        return o


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False


# Alias for backward compatibility
_attention_fast = _attention


# ============================================================================
# STRIDED KERNEL WITH PER-HEAD TENSOR DESCRIPTORS
# No padding needed - each head gets its own TensorDescriptor with correct bounds
# ============================================================================

@triton.jit
def _attn_fwd_strided_inner(acc, l_i, m_i, q,
                            desc_k, desc_v, desc_tree_mask,
                            tree_mask_row_offset, offset_kv, dtype: tl.constexpr,
                            start_m, qk_scale,
                            BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                            STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                            N_CTX_Q: tl.constexpr, N_CTX_KV: tl.constexpr):
    """Inner attention loop for strided kernel."""
    past_len = N_CTX_KV - N_CTX_Q

    if STAGE == 1:
        lo, hi = 0, past_len
    elif STAGE == 2:
        lo = past_len
        hi = past_len + (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX_KV

    offsetk_y = offset_kv + lo

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)

        boundary_mask = (start_n + offs_n[None, :]) < N_CTX_KV

        if STAGE == 2:
            tree_col = start_n - past_len
            tree_mask_block = desc_tree_mask.load([tree_mask_row_offset, tree_col])
            qk = qk * qk_scale + tree_mask_block + tl.where(boundary_mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        elif STAGE == 1:
            past_ctx_mask = (start_n + offs_n[None, :]) < past_len
            combined_mask = boundary_mask & past_ctx_mask
            qk = qk * qk_scale + tl.where(combined_mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        else:
            qk = qk * qk_scale + tl.where(boundary_mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = desc_v.load([offsetk_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_strided(
    Q, K, V, O, M, Tree_mask,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_tz, stride_th, stride_tm, stride_tn,
    Z, H, N_CTX_Q, N_CTX_KV,
    STAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Strided attention kernel with per-head TensorDescriptors."""
    dtype = tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qk_scale = sm_scale * 1.44269504

    # Compute base pointers for this (batch, head) - these are ELEMENT offsets
    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    o_base = O + off_z * stride_oz + off_h * stride_oh
    t_base = Tree_mask + off_z * stride_tz + off_h * stride_th

    # Create per-head TensorDescriptors with shape [N, D] - boundaries match head boundaries!
    # Strides: [stride_m, stride_k] for row-major within this head
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
        t_base, shape=[N_CTX_Q, N_CTX_Q], strides=[stride_tm, stride_tn],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    # Load Q for this block
    qo_offset = start_m * BLOCK_M
    q = desc_q.load([qo_offset, 0])

    # Initialize accumulators
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q_valid_mask = offs_m < N_CTX_Q

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Tree mask row offset (within this head's tree_mask)
    tree_mask_row_offset = start_m * BLOCK_M

    # Stage 1: off-diagonal (past context)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_strided_inner(
            acc, l_i, m_i, q, desc_k, desc_v, desc_tree_mask,
            tree_mask_row_offset, 0, dtype, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N, 4 - STAGE, offs_m, offs_n,
            N_CTX_Q, N_CTX_KV
        )

    # Stage 2: on-diagonal (tree region)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_strided_inner(
            acc, l_i, m_i, q, desc_k, desc_v, desc_tree_mask,
            tree_mask_row_offset, 0, dtype, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n,
            N_CTX_Q, N_CTX_KV
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


class _attention_strided(torch.autograd.Function):
    """Strided attention with per-head TensorDescriptors - no padding, correct boundaries."""

    @staticmethod
    def forward(ctx, q, k, v, tree_mask, sm_scale):
        B, H, N_CTX_Q, HEAD_DIM = q.shape
        N_CTX_KV = k.shape[2]

        # Make contiguous - keep 4D shape
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Prepare tree_mask
        has_tree_mask = tree_mask is not None
        if has_tree_mask:
            if tree_mask.shape[0] == 1 and B > 1:
                tree_mask = tree_mask.expand(B, -1, -1, -1)
            if tree_mask.shape[1] == 1 and H > 1:
                tree_mask = tree_mask.expand(-1, H, -1, -1)
            tree_mask = tree_mask.contiguous().to(q.dtype)
            stage = 3  # Causal with tree mask
        else:
            tree_mask = torch.zeros((B, H, N_CTX_Q, N_CTX_Q), device=q.device, dtype=q.dtype)
            stage = 1  # Non-causal

        # Output tensor - same 4D shape
        o = torch.empty_like(q)
        M = torch.empty((B, H, N_CTX_Q), device=q.device, dtype=torch.float32)

        # Grid
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(N_CTX_Q, BLOCK_M), B * H)

        # Multi-GPU support
        device_idx = q.device.index if q.device.index is not None else 0
        prev_device = torch.cuda.current_device()
        torch.cuda.set_device(device_idx)

        try:
            _attn_fwd_strided[grid](
                q, k, v, o, M, tree_mask,
                sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                tree_mask.stride(0), tree_mask.stride(1), tree_mask.stride(2), tree_mask.stride(3),
                B, H, N_CTX_Q, N_CTX_KV,
                STAGE=stage,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        finally:
            torch.cuda.set_device(prev_device)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        return o


BATCH, N_HEADS = 4, 32
# vary seq length for fixed head and batch=4
configs = []
for HEAD_DIM in [64, 128]:
    for mode in ["fwd"]:
        for causal in [True, False]:
            # Enable warpspec for causal fwd on Hopper
            enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
            for warp_specialize in [False, True] if enable_ws else [False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=[2**i for i in range(6, 15)],
                        line_arg="provider",
                        line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
                        line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
                        styles=[("red", "-"), ("green", "-")],
                        ylabel="TFLOPS",
                        plot_name=
                        f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                        args={
                            "H": N_HEADS,
                            "BATCH": BATCH,
                            "HEAD_DIM": HEAD_DIM,
                            "mode": mode,
                            "causal": causal,
                            "warp_specialize": warp_specialize,
                        },
                    ))
                
attention = _attention.apply

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    ms = 0.0  # Initialize ms to avoid linter error
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, warp_specialize)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)

def plot_ratio_vs_ctx():
    """Plot Triton/Flash-2 ratio vs N_CTX from existing CSV files."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    csv_files = [
        ("d128", "fused-attention-batch4-head32-d128-fwd-causal=True-warp_specialize=False.csv"),
        ("d64", "fused-attention-batch4-head32-d64-fwd-causal=True-warp_specialize=False.csv"),
    ]

    plt.figure(figsize=(8, 5))
    for label, f in csv_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            ratios = df["Triton [FP16]"] / df["Flash-2"]
            plt.plot(df["N_CTX"], ratios, 'o-', linewidth=2, markersize=6, label=label)
            print(f"{label}: mean ratio = {ratios.mean():.3f}")

    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Ratio=1.0')
    plt.xscale('log', base=2)
    plt.xlabel('N_CTX')
    plt.ylabel('Triton [FP16] / Flash-2')
    plt.title('Performance Ratio vs Context Length')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('triton_flash_ratio.png', dpi=150)
    print("Saved: triton_flash_ratio.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_ratio_vs_ctx()
    else:
        bench_flash_attention.run(save_path=".", print_data=True)