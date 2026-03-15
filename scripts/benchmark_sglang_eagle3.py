"""
Benchmark EAGLE3 speculative decoding with SGLang (dynamic tree).

SGLang implements EAGLE3's dynamic tree building at runtime using
confidence-based reranking, unlike vLLM which only supports static trees.

Usage:
    cd scripts
    conda activate vllm-spec

    CUDA_VISIBLE_DEVICES=4,5 python benchmark_sglang_eagle3.py \
        --target-model /home/dataset_model/model/Llama-3.1-8B-Instruct \
        --draft-model /home/dataset_model/model/EAGLE3-LLaMA3.1-Instruct-8B \
        --tp-size 2 \
        --num-steps 5 \
        --eagle-topk 8 \
        --num-draft-tokens 32 \
        --dataset /home/dataset_model/dataset/gsm8k \
        --num-prompts 50

    # Profile attention time during EAGLE3 verification:
    CUDA_VISIBLE_DEVICES=4,5 python benchmark_sglang_eagle3.py \
        --target-model /home/dataset_model/model/Llama-3.1-8B-Instruct \
        --draft-model /home/dataset_model/model/EAGLE3-LLaMA3.1-Instruct-8B \
        --tp-size 2 --num-draft-tokens 63 --profile-attention \
        --dataset /home/dataset_model/dataset/gsm8k --num-prompts 50

    # Replace attention with subtree kernel during verification:
    CUDA_VISIBLE_DEVICES=4,5 python benchmark_sglang_eagle3.py \
        --target-model /home/dataset_model/model/Llama-3.1-8B-Instruct \
        --draft-model /home/dataset_model/model/EAGLE3-LLaMA3.1-Instruct-8B \
        --tp-size 2 --num-draft-tokens 63 --use-subtree-kernel \
        --dataset /home/dataset_model/dataset/gsm8k --num-prompts 50

    # Replace attention with CUDA bitmap kernel during verification:
    CUDA_VISIBLE_DEVICES=4,5 python benchmark_sglang_eagle3.py \
        --target-model /home/dataset_model/model/Llama-3.1-8B-Instruct \
        --draft-model /home/dataset_model/model/EAGLE3-LLaMA3.1-Instruct-8B \
        --tp-size 2 --num-draft-tokens 63 --use-bitmap-kernel \
        --dataset /home/dataset_model/dataset/gsm8k --num-prompts 50

    # Paged bitmap attention (GPU bitmaps + zero-gather paged KV reads):
    CUDA_VISIBLE_DEVICES=4,5 python benchmark_sglang_eagle3.py \
        --target-model /home/dataset_model/model/Llama-3.1-8B-Instruct \
        --draft-model /home/dataset_model/model/EAGLE3-LLaMA3.1-Instruct-8B \
        --tp-size 2 --num-draft-tokens 63 --use-paged-bitmap \
        --disable-cuda-graph \
        --dataset /home/dataset_model/dataset/gsm8k --num-prompts 50
"""

import argparse
import json
import os
import sys
import tempfile
import time


import sglang as sgl
from datasets import detect_and_load


# Scripts directory (for subprocess path setup)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_PROMPTS = [
    "Write a Python function to implement binary search.",
    "Explain the concept of quantum entanglement in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Write a short story about a robot learning to paint.",
    "Explain how neural networks learn through backpropagation.",
    "What is the time complexity of quicksort and why?",
    "Describe the process of photosynthesis step by step.",
    "Write a bash script to find all files larger than 100MB.",
    "Explain the CAP theorem in distributed systems.",
    "What are the SOLID principles in software engineering?",
]


# ============================================================================
# Attention profiling (runs in scheduler subprocess)
# ============================================================================

def _apply_profiling_patch(profile_file):
    """Monkey-patch RadixAttention.forward to collect CUDA event timing.

    Called inside the scheduler subprocess. Instruments layer 0 only to
    measure per-call attention time without double-counting across layers.
    Uses deferred CUDA event timing (no sync during benchmark).

    Writes JSON to profile_file on process exit:
        verify_extend_times_ms: list of per-call ms during verification
        other_extend_times_ms: list of per-call ms during other modes
        verify_extend_q_tokens: list of Q token counts per verify call
    """
    import atexit
    import torch

    from sglang.srt.layers.radix_attention import RadixAttention

    _events = []  # (start_event, end_event, is_verify, q_tokens)
    _orig_forward = RadixAttention.forward

    def _profiled_forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        # Only instrument layer 0 to avoid N_layers x counting
        if self.layer_id != 0:
            return _orig_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _orig_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)
        end.record()

        is_verify = (hasattr(forward_batch, 'forward_mode') and
                     forward_batch.forward_mode.is_target_verify())
        _events.append((start, end, is_verify, q.shape[0]))

        return result

    RadixAttention.forward = _profiled_forward

    def _write_results():
        try:
            import torch as _torch
            _torch.cuda.synchronize()
        except Exception:
            pass
        verify_times = []
        other_times = []
        verify_q_tokens = []
        for start, end, is_verify, qt in _events:
            try:
                ms = start.elapsed_time(end)
            except Exception:
                continue
            if is_verify:
                verify_times.append(ms)
                verify_q_tokens.append(qt)
            else:
                other_times.append(ms)
        data = {
            "verify_extend_times_ms": verify_times,
            "other_extend_times_ms": other_times,
            "verify_extend_q_tokens": verify_q_tokens,
        }
        try:
            with open(profile_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[PROFILE] Error writing results: {e}", flush=True)

    atexit.register(_write_results)


# ============================================================================
# NVTX instrumentation for baseline FlashInfer path (runs in scheduler subprocess)
# ============================================================================

def _apply_nvtx_baseline_patch():
    """Add NVTX markers around FlashInfer's forward_extend for baseline profiling.

    Wraps FlashInferAttnBackend.forward_extend with NVTX ranges so that
    nsys can attribute GPU kernels (GEMM, attention, etc.) to the correct
    layer and forward mode (verify vs other).
    """
    import torch
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

    _orig_forward_extend = FlashInferAttnBackend.forward_extend

    def _nvtx_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        is_verify = (hasattr(forward_batch, 'forward_mode') and
                     forward_batch.forward_mode.is_target_verify())
        mode = "verify" if is_verify else "other"
        torch.cuda.nvtx.range_push(f"flashinfer_fwd_extend_L{layer.layer_id}_{mode}")
        result = _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)
        torch.cuda.nvtx.range_pop()
        return result

    FlashInferAttnBackend.forward_extend = _nvtx_forward_extend

    # Also wrap forward_decode for decode-phase visibility
    _orig_forward_decode = FlashInferAttnBackend.forward_decode

    def _nvtx_forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        torch.cuda.nvtx.range_push(f"flashinfer_fwd_decode_L{layer.layer_id}")
        result = _orig_forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache)
        torch.cuda.nvtx.range_pop()
        return result

    FlashInferAttnBackend.forward_decode = _nvtx_forward_decode
    print("[NVTX] Patched FlashInferAttnBackend with NVTX markers (baseline)", flush=True)


# ============================================================================
# Subtree kernel replacement (runs in scheduler subprocess)
# ============================================================================

def _apply_subtree_kernel_patch(profile_file=None, nvtx_enabled=False):
    """Replace FlashInfer attention with our subtree kernel during verification.

    Monkey-patches FlashInferAttnBackend.forward_extend to:
    1. Save KV to cache (same as original)
    2. Gather K/V from paged cache into contiguous tensors
    3. Convert boolean custom_mask -> [-inf, 0] tree mask
    4. Compute DFS permutation + block-sparse metadata at runtime
    5. Call _attention_subtree Triton kernel

    Optimizations over naive per-layer-per-request computation:
    - Cross-layer DFS metadata cache: tree mask is identical across 32 layers,
      so metadata (perm, inv_perm, block_indices, block_counts, tree_mask_4d)
      is computed once on layer 0 and reused for layers 1-31.
    - KV buffer lookup hoisted outside per-request loop (once per layer).

    WARNING: The KV gather adds O(B * seq_len * H * D) overhead per layer.
    This is a proof-of-concept for correctness, not a performance optimization.

    Optionally profiles both the original and subtree kernel times if
    profile_file is set.
    """
    import torch
    import triton

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sparse_tree_kernel import _attention_subtree

    # Set Triton allocator (required for TMA on Hopper)
    def _alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    try:
        triton.set_allocator(_alloc_fn)
    except Exception:
        pass

    _orig_forward_extend = FlashInferAttnBackend.forward_extend
    _timing_data = []  # (original_ms, subtree_ms, n_tree, past_len_avg)

    # Cross-layer DFS metadata cache.
    # Key: id(custom_mask) — same tensor object across all 32 layers within
    # one verification step. Value: list of per-request tuples
    # (perm, inv_perm, block_indices, block_counts, MAX_SPARSE, tree_mask_4d).
    _metadata_cache = {}

    # NVTX helper: no-ops when disabled to avoid overhead
    _nvtx = nvtx_enabled
    def _nvtx_push(name):
        if _nvtx:
            torch.cuda.nvtx.range_push(name)
    def _nvtx_pop():
        if _nvtx:
            torch.cuda.nvtx.range_pop()

    def _subtree_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        # Non-verification: always use original
        if not (hasattr(forward_batch, 'forward_mode') and
                forward_batch.forward_mode.is_target_verify()):
            return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        _nvtx_push(f"subtree_fwd_extend_L{layer.layer_id}")

        # --- Save KV cache (same as original) ---
        _nvtx_push("save_kv_cache")
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )
        _nvtx_pop()  # save_kv_cache

        # --- Kernel replacement: gather K/V and run subtree kernel ---
        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        D = layer.head_dim
        sm_scale = layer.scaling

        spec_info = forward_batch.spec_info
        N_tree = spec_info.draft_token_num
        bs = len(forward_batch.req_pool_indices)

        # --- Cross-layer metadata cache ---
        _nvtx_push("metadata_cache_check")
        mask_id = id(spec_info.custom_mask)
        cached_meta = _metadata_cache.get(mask_id)
        if cached_meta is None:
            # Cache miss (layer 0): compute metadata for all requests
            per_req_meta = []
            for i in range(bs):
                tree_mask_bool = _extract_tree_mask_for_request(
                    spec_info.custom_mask, forward_batch.seq_lens, i, N_tree
                )
                # Convert bool -> [-inf, 0] float mask
                tree_mask = torch.where(
                    tree_mask_bool,
                    torch.tensor(0.0, device=q.device, dtype=q.dtype),
                    torch.tensor(float('-inf'), device=q.device, dtype=q.dtype),
                )
                tree_mask_4d = tree_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_tree, N_tree]

                perm, inv_perm, block_indices, block_counts, MAX_SPARSE = \
                    _compute_subtree_metadata_from_mask(tree_mask_bool, N_tree)

                per_req_meta.append((
                    perm.to(q.device),
                    inv_perm.to(q.device),
                    block_indices.to(q.device),
                    block_counts.to(q.device),
                    MAX_SPARSE,
                    tree_mask_4d,
                ))

            # Evict stale entries (keep only current mask)
            _metadata_cache.clear()
            _metadata_cache[mask_id] = per_req_meta
            cached_meta = per_req_meta
        _nvtx_pop()  # metadata_cache_check

        # Reshape Q: [bs*N_tree, H_q*D] -> per-request processing
        q_flat = q.view(-1, H_q, D)  # [bs*N_tree, H_q, D]
        gqa_rep = H_q // H_kv if H_kv < H_q else 1

        # Hoist KV buffer lookup outside per-request loop (once per layer)
        _nvtx_push("kv_buffer_lookup")
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        _nvtx_pop()  # kv_buffer_lookup

        _nvtx_push("per_request_loop")
        outputs = []
        for i in range(bs):
            _nvtx_push(f"req_{i}")
            req_pool_idx = forward_batch.req_pool_indices[i]
            past_len = forward_batch.seq_lens[i].item()
            total_kv = past_len + N_tree

            # Extract Q for this request: [N_tree, H_q, D] -> [1, H_q, N_tree, D]
            _nvtx_push("q_reshape")
            q_i = q_flat[i * N_tree : (i + 1) * N_tree]  # [N_tree, H_q, D]
            q_4d = q_i.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H_q, N_tree, D]
            _nvtx_pop()  # q_reshape

            # Gather K/V from paged cache
            _nvtx_push("kv_gather")
            kv_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :total_kv]
            k_full = k_buf[kv_indices]  # [total_kv, H_kv, D]
            v_full = v_buf[kv_indices]  # [total_kv, H_kv, D]

            # Reshape to [1, H_kv, total_kv, D]
            k_4d = k_full.permute(1, 0, 2).unsqueeze(0).contiguous()
            v_4d = v_full.permute(1, 0, 2).unsqueeze(0).contiguous()
            _nvtx_pop()  # kv_gather

            # GQA: expand KV heads to match Q heads
            _nvtx_push("gqa_expand")
            if gqa_rep > 1:
                k_4d = k_4d.repeat_interleave(gqa_rep, dim=1)
                v_4d = v_4d.repeat_interleave(gqa_rep, dim=1)
            _nvtx_pop()  # gqa_expand

            # Read cached metadata for this request
            perm, inv_perm, block_indices, block_counts, MAX_SPARSE, tree_mask_4d = \
                cached_meta[i]

            # Run subtree kernel
            _nvtx_push("kernel_call")
            out_i = _attention_subtree.apply(
                q_4d, k_4d, v_4d, tree_mask_4d, sm_scale,
                block_indices, block_counts,
                MAX_SPARSE,
                perm, inv_perm,
            )
            _nvtx_pop()  # kernel_call

            # out_i: [1, H_q, N_tree, D] -> [N_tree, H_q, D]
            outputs.append(out_i.squeeze(0).permute(1, 0, 2))
            _nvtx_pop()  # req_{i}
        _nvtx_pop()  # per_request_loop

        # Concatenate: [bs*N_tree, H_q, D] -> [bs*N_tree, H_q*D]
        _nvtx_push("output_concat")
        out = torch.cat(outputs, dim=0)  # [bs*N_tree, H_q, D]
        result = out.reshape(-1, H_q * D)
        _nvtx_pop()  # output_concat

        _nvtx_pop()  # subtree_fwd_extend_L{layer_id}
        return result

    FlashInferAttnBackend.forward_extend = _subtree_forward_extend
    print("[SUBTREE] Patched FlashInferAttnBackend.forward_extend with subtree kernel",
          flush=True)
    if nvtx_enabled:
        print("[SUBTREE] NVTX markers enabled", flush=True)


# ============================================================================
# Unified sparse attention with KV importance selection (runs in scheduler subprocess)
# ============================================================================

def _apply_unified_sparse_patch(nvtx_enabled=False, top_k_ratio=0.3, max_past=4096):
    """Replace FlashInfer attention with unified sparse attention during verification.

    Instead of attending to ALL KV, selects the most important past-context KV
    positions using the first query token as importance proxy, keeps all tree
    positions, then runs the unified sparse Triton kernel.

    Cross-layer caching: importance scoring + KV selection happens only on
    layer 0; layers 1-31 reuse the same selected indices (same Q/KV structure).

    Args:
        nvtx_enabled: Add NVTX markers for nsys profiling.
        top_k_ratio: Fraction of past context KV to keep (0-1).
        max_past: Max past context length before falling back to original
            FlashInfer. Prevents CPU-bound metadata computation from blocking
            NCCL allgather in TP>1 setups. Set to 0 to disable fallback.
    """
    import torch
    import triton

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from unified_sparse_attention import (
        _attention_unified_sparse,
        precompute_unified_block_metadata_gpu,
    )

    def _alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    try:
        triton.set_allocator(_alloc_fn)
    except Exception:
        pass

    _orig_forward_extend = FlashInferAttnBackend.forward_extend

    # Cross-layer cache: keyed by id(custom_mask).
    # Stores per-request: (selected_indices, mask_4d, block_indices, block_counts, MAX_SPARSE)
    _selection_cache = {}

    _nvtx = nvtx_enabled
    def _nvtx_push(name):
        if _nvtx:
            torch.cuda.nvtx.range_push(name)
    def _nvtx_pop():
        if _nvtx:
            torch.cuda.nvtx.range_pop()

    _fallback_count = [0]  # mutable counter for logging

    def _unified_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if not (hasattr(forward_batch, 'forward_mode') and
                forward_batch.forward_mode.is_target_verify()):
            return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        # Fallback for large past contexts: the per-request importance scoring
        # and metadata computation is too slow for large contexts, causing one
        # TP rank to stall and trigger NCCL allgather timeout.
        if max_past > 0:
            bs = len(forward_batch.req_pool_indices)
            batch_max_past = max(forward_batch.seq_lens[i].item() for i in range(bs))
            if batch_max_past > max_past:
                _fallback_count[0] += 1
                if _fallback_count[0] <= 3:
                    print(f"[UNIFIED] Fallback to FlashInfer: past_len={batch_max_past} > {max_past}",
                          flush=True)
                return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        _nvtx_push(f"unified_sparse_L{layer.layer_id}")

        # Save KV cache (same as original)
        _nvtx_push("save_kv_cache")
        cache_loc = (forward_batch.out_cache_loc if not layer.is_cross_attention
                     else forward_batch.encoder_out_cache_loc)
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale)
        _nvtx_pop()

        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        D = layer.head_dim
        sm_scale = layer.scaling
        spec_info = forward_batch.spec_info
        N_tree = spec_info.draft_token_num
        bs = len(forward_batch.req_pool_indices)
        gqa_rep = H_q // H_kv if H_kv < H_q else 1

        _nvtx_push("kv_buffer_lookup")
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        _nvtx_pop()

        q_flat = q.view(-1, H_q, D)

        # Check cross-layer cache
        mask_id = id(spec_info.custom_mask)
        cached = _selection_cache.get(mask_id)
        is_layer0 = (cached is None)

        if is_layer0:
            # Layer 0: compute importance, select KV, build masks, cache everything
            _nvtx_push("importance_selection_L0")
            per_req_cache = []
            for i in range(bs):
                req_pool_idx = forward_batch.req_pool_indices[i]
                past_len = forward_batch.seq_lens[i].item()
                total_kv = past_len + N_tree

                # Gather all K for importance scoring
                kv_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :total_kv]
                k_full = k_buf[kv_indices]  # [total_kv, H_kv, D]

                # Importance: first query token's QK with all K positions
                q_i0 = q_flat[i * N_tree]  # [H_q, D]
                # Use first H_kv heads for scoring
                q_score = q_i0[:H_kv, :]  # [H_kv, D]
                k_score = k_full.permute(1, 0, 2)  # [H_kv, total_kv, D]
                scores = torch.bmm(q_score.unsqueeze(1), k_score.transpose(1, 2))  # [H_kv, 1, total_kv]
                importance = scores.squeeze(1).mean(dim=0)  # [total_kv]

                # Select: top-k past + all tree positions
                n_past_keep = max(1, int(past_len * top_k_ratio))
                if past_len > 0 and n_past_keep < past_len:
                    _, past_topk = importance[:past_len].topk(n_past_keep)
                    tree_idx = torch.arange(past_len, total_kv, device=q.device)
                    selected = torch.cat([past_topk, tree_idx]).sort().values
                else:
                    selected = torch.arange(total_kv, device=q.device)
                N_sel = selected.shape[0]

                # Build pruned mask [N_tree, N_sel]
                tree_mask_bool = _extract_tree_mask_for_request(
                    spec_info.custom_mask, forward_batch.seq_lens, i, N_tree)

                mask = torch.zeros(N_tree, N_sel, device=q.device, dtype=q.dtype)
                # Vectorized: identify which selected positions are tree positions
                is_tree = selected >= past_len
                tree_cols = torch.where(is_tree)[0]
                if tree_cols.numel() > 0:
                    tree_local = selected[tree_cols] - past_len  # local tree indices
                    tree_mask_cols = tree_mask_bool[:, tree_local].to(q.dtype)
                    # 0 where True, -inf where False
                    mask[:, tree_cols] = torch.where(
                        tree_mask_cols > 0.5,
                        torch.zeros_like(tree_mask_cols),
                        torch.full_like(tree_mask_cols, float('-inf')))
                # Past columns stay 0 (all visible)

                mask_4d = mask.unsqueeze(0).unsqueeze(0)

                block_indices, block_counts, MAX_SPARSE = \
                    precompute_unified_block_metadata_gpu(mask)

                per_req_cache.append((
                    selected,
                    kv_indices,
                    mask_4d,
                    block_indices,
                    block_counts,
                    MAX_SPARSE,
                ))

            _selection_cache.clear()
            _selection_cache[mask_id] = per_req_cache
            cached = per_req_cache
            _nvtx_pop()  # importance_selection_L0

        # Process each request using cached selection
        _nvtx_push("per_request_loop")
        outputs = []
        for i in range(bs):
            _nvtx_push(f"req_{i}")
            selected, kv_indices, mask_4d, block_indices, block_counts, MAX_SPARSE = cached[i]
            N_sel = selected.shape[0]

            # Q
            _nvtx_push("q_reshape")
            q_i = q_flat[i * N_tree : (i + 1) * N_tree]
            q_4d = q_i.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H_q, N_tree, D]
            _nvtx_pop()

            # Gather SELECTED KV only (layers 1-31 skip full gather)
            _nvtx_push("kv_gather_selected")
            sel_kv_indices = kv_indices[selected]
            k_sel = k_buf[sel_kv_indices].permute(1, 0, 2).unsqueeze(0).contiguous()
            v_sel = v_buf[sel_kv_indices].permute(1, 0, 2).unsqueeze(0).contiguous()
            _nvtx_pop()

            _nvtx_push("gqa_expand")
            if gqa_rep > 1:
                k_sel = k_sel.repeat_interleave(gqa_rep, dim=1)
                v_sel = v_sel.repeat_interleave(gqa_rep, dim=1)
            _nvtx_pop()

            _nvtx_push("kernel_call")
            out_i = _attention_unified_sparse.apply(
                q_4d, k_sel, v_sel, mask_4d, sm_scale,
                block_indices, block_counts, MAX_SPARSE)
            _nvtx_pop()

            outputs.append(out_i.squeeze(0).permute(1, 0, 2))
            _nvtx_pop()  # req_i
        _nvtx_pop()  # per_request_loop

        _nvtx_push("output_concat")
        out = torch.cat(outputs, dim=0)
        result = out.reshape(-1, H_q * D)
        _nvtx_pop()

        _nvtx_pop()  # unified_sparse_L
        return result

    FlashInferAttnBackend.forward_extend = _unified_forward_extend
    print(f"[UNIFIED] Patched with unified sparse attention (ratio={top_k_ratio}, "
          f"max_past={max_past})", flush=True)
    if nvtx_enabled:
        print("[UNIFIED] NVTX markers enabled", flush=True)


# ============================================================================
# Paged sparse attention — zero gather, zero CPU (runs in scheduler subprocess)
# ============================================================================

def _apply_paged_sparse_patch(nvtx_enabled=False, top_k_ratio=0.3, max_past=0):
    """Replace FlashInfer attention with paged sparse Triton kernel during verification.

    Reads K/V directly from SGLang's flat paged KV buffer via indirect token-slot
    indexing. No gather to contiguous memory, no CPU metadata computation, no GQA
    expand. All operations (importance scoring, mask building, block metadata,
    attention) run on GPU.

    Cross-layer caching: layer 0 computes importance + selection + mask + metadata;
    layers 1-31 reuse the same cached data (tree structure is identical across layers).

    Args:
        nvtx_enabled: Add NVTX markers for nsys profiling.
        top_k_ratio: Fraction of past context KV to keep (0-1).
        max_past: Max past context for sparse kernel. Falls back to FlashInfer
            when past_len exceeds this. Set 0 to disable fallback.
    """
    import torch
    import triton

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from paged_sparse_attention import (
        select_important_kv_gpu,
        build_mask_and_metadata_gpu,
        paged_sparse_attention,
    )

    def _alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    try:
        triton.set_allocator(_alloc_fn)
    except Exception:
        pass

    _orig_forward_extend = FlashInferAttnBackend.forward_extend

    # Cross-layer cache: keyed by id(custom_mask).
    # Per-request: (selected_slots, mask_2d, block_indices, block_counts, MAX_SPARSE, N_sel)
    _selection_cache = {}

    _nvtx = nvtx_enabled
    def _nvtx_push(name):
        if _nvtx:
            torch.cuda.nvtx.range_push(name)
    def _nvtx_pop():
        if _nvtx:
            torch.cuda.nvtx.range_pop()

    _fallback_count = [0]

    def _paged_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if not (hasattr(forward_batch, 'forward_mode') and
                forward_batch.forward_mode.is_target_verify()):
            return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        # Fallback for large past contexts
        if max_past > 0:
            bs = len(forward_batch.req_pool_indices)
            batch_max_past = max(forward_batch.seq_lens[i].item() for i in range(bs))
            if batch_max_past > max_past:
                _fallback_count[0] += 1
                if _fallback_count[0] <= 3:
                    print(f"[PAGED] Fallback to FlashInfer: past_len={batch_max_past} > {max_past}",
                          flush=True)
                return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        _nvtx_push(f"paged_sparse_L{layer.layer_id}")

        # Save KV cache (same as original)
        _nvtx_push("save_kv_cache")
        cache_loc = (forward_batch.out_cache_loc if not layer.is_cross_attention
                     else forward_batch.encoder_out_cache_loc)
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale)
        _nvtx_pop()

        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        D = layer.head_dim
        sm_scale = layer.scaling
        spec_info = forward_batch.spec_info
        N_tree = spec_info.draft_token_num
        bs = len(forward_batch.req_pool_indices)

        _nvtx_push("kv_buffer_lookup")
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        _nvtx_pop()

        q_flat = q.view(-1, H_q, D)  # [bs*N_tree, H_q, D]

        # Check cross-layer cache
        mask_id = id(spec_info.custom_mask)
        cached = _selection_cache.get(mask_id)
        is_layer0 = (cached is None)

        if is_layer0:
            _nvtx_push("importance_selection_L0")
            per_req_cache = []
            for i in range(bs):
                req_pool_idx = forward_batch.req_pool_indices[i]
                past_len = forward_batch.seq_lens[i].item()
                total_kv = past_len + N_tree

                kv_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :total_kv]

                # Importance scoring: use first query token, KV heads only
                q_first_hkv = q_flat[i * N_tree, :H_kv, :]  # [H_kv, D]

                _nvtx_push("select_kv")
                selected = select_important_kv_gpu(
                    q_first_hkv, k_buf, kv_indices, past_len, N_tree, top_k_ratio)
                _nvtx_pop()

                # Map selected logical indices -> physical token slots
                selected_slots = kv_indices[selected.long()].to(torch.int32)

                # Build tree mask and metadata
                _nvtx_push("build_mask_metadata")
                tree_mask_bool = _extract_tree_mask_for_request(
                    spec_info.custom_mask, forward_batch.seq_lens, i, N_tree)

                mask_2d, block_indices, block_counts, MAX_SPARSE, N_sel = \
                    build_mask_and_metadata_gpu(tree_mask_bool, selected,
                                                past_len, N_tree)
                _nvtx_pop()

                per_req_cache.append((
                    selected_slots,
                    mask_2d,
                    block_indices,
                    block_counts,
                    MAX_SPARSE,
                    N_sel,
                ))

            _selection_cache.clear()
            _selection_cache[mask_id] = per_req_cache
            cached = per_req_cache
            _nvtx_pop()  # importance_selection_L0

        # Process each request: launch paged sparse kernel
        _nvtx_push("per_request_loop")
        outputs = []
        for i in range(bs):
            _nvtx_push(f"req_{i}")
            selected_slots, mask_2d, block_indices, block_counts, MAX_SPARSE, N_sel = cached[i]

            # Q for this request: [N_tree, H_q, D] -> [H_q, N_tree, D]
            _nvtx_push("q_reshape")
            q_i = q_flat[i * N_tree : (i + 1) * N_tree]  # [N_tree, H_q, D]
            q_heads = q_i.permute(1, 0, 2).contiguous()  # [H_q, N_tree, D]
            _nvtx_pop()

            # Launch kernel: reads K/V directly from k_buf/v_buf
            _nvtx_push("kernel_call")
            out_i = paged_sparse_attention(
                q_heads, k_buf, v_buf, selected_slots, mask_2d,
                block_indices, block_counts, MAX_SPARSE,
                N_sel, sm_scale, H_kv)
            _nvtx_pop()

            # out_i: [H_q, N_tree, D] -> [N_tree, H_q, D]
            outputs.append(out_i.permute(1, 0, 2))
            _nvtx_pop()  # req_i
        _nvtx_pop()  # per_request_loop

        _nvtx_push("output_concat")
        out = torch.cat(outputs, dim=0)  # [bs*N_tree, H_q, D]
        result = out.reshape(-1, H_q * D)
        _nvtx_pop()

        _nvtx_pop()  # paged_sparse_L
        return result

    FlashInferAttnBackend.forward_extend = _paged_forward_extend
    print(f"[PAGED] Patched with paged sparse attention (ratio={top_k_ratio}, "
          f"max_past={max_past})", flush=True)
    if nvtx_enabled:
        print("[PAGED] NVTX markers enabled", flush=True)


# ============================================================================
# CUDA bitmap attention kernel (runs in scheduler subprocess)
# ============================================================================

def _compute_bitmap_from_mask(tree_mask_bool, BLOCK_M, BLOCK_N=32):
    """Compute per-q_block uint64 bitmaps from boolean tree mask.

    For each q_block, sets bit b in word w if ANY Q position in the block
    attends to ANY KV position in tree KV block (w*64 + b).

    Args:
        tree_mask_bool: [N_tree, N_tree] boolean tensor (GPU or CPU).
        BLOCK_M: Q block size (must match CUDA kernel's BLOCK_M).
        BLOCK_N: KV block size (32, matches kernel).

    Returns:
        bitmaps: [num_q_blocks, W] int64 tensor (CPU).
        W: number of uint64 words per q_block.
    """
    import torch
    mask_cpu = tree_mask_bool.cpu()
    N_tree = mask_cpu.shape[0]
    num_q_blocks = (N_tree + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (N_tree + BLOCK_N - 1) // BLOCK_N
    W = (num_kv_blocks + 63) // 64

    bitmaps = torch.zeros(num_q_blocks, W, dtype=torch.int64)
    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min(q_start + BLOCK_M, N_tree)
        for kb in range(num_kv_blocks):
            kv_start = kb * BLOCK_N
            kv_end = min(kv_start + BLOCK_N, N_tree)
            if mask_cpu[q_start:q_end, kv_start:kv_end].any():
                bitmaps[qb, kb // 64] |= (1 << (kb % 64))

    return bitmaps, W


def _apply_bitmap_kernel_patch(nvtx_enabled=False):
    """Replace FlashInfer attention with CUDA bitmap attention during verification.

    Monkey-patches FlashInferAttnBackend.forward_extend to:
    1. Save KV to cache (same as original)
    2. Gather K/V from paged cache into contiguous tensors
    3. Convert boolean custom_mask -> [-inf, 0] tree mask
    4. Compute bitmap metadata from tree mask at runtime
    5. Call CUDA bitmap_attention kernel (two-stage: block-index past + bitmap tree)

    Cross-layer caching: tree mask, bitmaps, and stage1 indices are computed once
    on layer 0 and reused for layers 1-31 (tree structure identical across layers).

    WARNING: The KV gather adds O(B * seq_len * H * D) overhead per layer.
    This is a proof-of-concept for correctness comparison, not a performance optimization.
    """
    import torch

    # Add csrc to path for the compiled CUDA extension
    csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csrc')
    if csrc_dir not in sys.path:
        sys.path.insert(0, csrc_dir)

    try:
        import bitmap_attention as ext
    except ImportError:
        print("[BITMAP] ERROR: bitmap_attention CUDA extension not found.", flush=True)
        print("[BITMAP] Build: cd csrc && python setup_bitmap_attention.py build_ext --inplace",
              flush=True)
        return

    from bitmap_sparse_attention import make_dense_stage1_indices
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

    _orig_forward_extend = FlashInferAttnBackend.forward_extend

    # Cross-layer cache: layer 0 recomputes, layers 1+ reuse.
    # Per-request: (tree_mask_2d, bitmaps_dev, stage1_indices, W)
    _metadata_cache = {'data': None}

    _nvtx = nvtx_enabled
    def _nvtx_push(name):
        if _nvtx:
            torch.cuda.nvtx.range_push(name)
    def _nvtx_pop():
        if _nvtx:
            torch.cuda.nvtx.range_pop()

    def _bitmap_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        # Non-verification: use original FlashInfer path
        if not (hasattr(forward_batch, 'forward_mode') and
                forward_batch.forward_mode.is_target_verify()):
            return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        _nvtx_push(f"bitmap_fwd_extend_L{layer.layer_id}")

        # --- Save KV cache (same as original) ---
        _nvtx_push("save_kv_cache")
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )
        _nvtx_pop()  # save_kv_cache

        # --- Kernel replacement: gather K/V and run bitmap kernel ---
        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        D = layer.head_dim
        sm_scale = layer.scaling

        spec_info = forward_batch.spec_info
        N_tree = spec_info.draft_token_num
        bs = len(forward_batch.req_pool_indices)
        gqa_rep = H_q // H_kv if H_kv < H_q else 1

        # --- Cross-layer metadata cache ---
        # Layer 0 always recomputes; layers 1+ reuse layer 0's result.
        # Avoids id() reuse bugs from Python object recycling.
        _nvtx_push("metadata_cache_check")
        if layer.layer_id == 0:
            BLOCK_M = ext.get_block_m(N_tree)
            per_req_meta = []
            for i in range(bs):
                past_len_i = forward_batch.seq_lens[i].item()

                tree_mask_bool = _extract_tree_mask_for_request(
                    spec_info.custom_mask, forward_batch.seq_lens, i, N_tree
                )

                # Convert bool -> [-inf, 0] fp16 mask for CUDA kernel
                tree_mask_2d = torch.where(
                    tree_mask_bool,
                    torch.tensor(0.0, device=q.device, dtype=torch.float16),
                    torch.tensor(float('-inf'), device=q.device, dtype=torch.float16),
                )  # [N_tree, N_tree]

                # Compute bitmaps from boolean mask
                bitmaps, W = _compute_bitmap_from_mask(tree_mask_bool, BLOCK_M)
                bitmaps_dev = bitmaps.to(q.device)

                # Stage 1: dense block indices for past context
                stage1_indices = make_dense_stage1_indices(past_len_i).to(q.device)

                per_req_meta.append((tree_mask_2d, bitmaps_dev, stage1_indices, W))

            _metadata_cache['data'] = per_req_meta
        cached_meta = _metadata_cache['data']
        _nvtx_pop()  # metadata_cache_check

        # Reshape Q: [bs*N_tree, H_q*D] -> per-request processing
        q_flat = q.view(-1, H_q, D)  # [bs*N_tree, H_q, D]

        # Hoist KV buffer lookup outside per-request loop (once per layer)
        _nvtx_push("kv_buffer_lookup")
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        _nvtx_pop()  # kv_buffer_lookup

        _nvtx_push("per_request_loop")
        outputs = []
        for i in range(bs):
            _nvtx_push(f"req_{i}")
            req_pool_idx = forward_batch.req_pool_indices[i]
            past_len_i = forward_batch.seq_lens[i].item()
            total_kv = past_len_i + N_tree

            # Extract Q: [N_tree, H_q, D] -> [1, H_q, N_tree, D]
            _nvtx_push("q_reshape")
            q_i = q_flat[i * N_tree : (i + 1) * N_tree]  # [N_tree, H_q, D]
            q_4d = q_i.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H_q, N_tree, D]
            _nvtx_pop()  # q_reshape

            # Gather K/V from paged cache
            _nvtx_push("kv_gather")
            kv_indices = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :total_kv]
            k_full = k_buf[kv_indices]  # [total_kv, H_kv, D]
            v_full = v_buf[kv_indices]  # [total_kv, H_kv, D]

            # Reshape to [1, H_kv, total_kv, D]
            k_4d = k_full.permute(1, 0, 2).unsqueeze(0).contiguous()
            v_4d = v_full.permute(1, 0, 2).unsqueeze(0).contiguous()
            _nvtx_pop()  # kv_gather

            # GQA: expand KV heads to match Q heads
            _nvtx_push("gqa_expand")
            if gqa_rep > 1:
                k_4d = k_4d.repeat_interleave(gqa_rep, dim=1)
                v_4d = v_4d.repeat_interleave(gqa_rep, dim=1)
            _nvtx_pop()  # gqa_expand

            # Read cached metadata for this request
            tree_mask_2d, bitmaps_dev, stage1_indices, W = cached_meta[i]

            # Run CUDA bitmap attention kernel
            _nvtx_push("kernel_call")
            out_i = ext.bitmap_attention_fwd(
                q_4d, k_4d, v_4d, tree_mask_2d,
                stage1_indices, bitmaps_dev,
                sm_scale, past_len_i, W)
            _nvtx_pop()  # kernel_call

            # out_i: [1, H_q, N_tree, D] -> [N_tree, H_q, D]
            outputs.append(out_i.squeeze(0).permute(1, 0, 2))
            _nvtx_pop()  # req_{i}
        _nvtx_pop()  # per_request_loop

        # Concatenate: [bs*N_tree, H_q, D] -> [bs*N_tree, H_q*D]
        _nvtx_push("output_concat")
        out = torch.cat(outputs, dim=0)  # [bs*N_tree, H_q, D]
        result = out.reshape(-1, H_q * D)
        _nvtx_pop()  # output_concat

        _nvtx_pop()  # bitmap_fwd_extend_L{layer_id}
        return result

    FlashInferAttnBackend.forward_extend = _bitmap_forward_extend
    print("[BITMAP] Patched FlashInferAttnBackend.forward_extend with CUDA bitmap kernel",
          flush=True)
    if nvtx_enabled:
        print("[BITMAP] NVTX markers enabled", flush=True)


# ============================================================================
# Paged bitmap attention — bitmap block-skipping + zero-gather paged KV reads
# ============================================================================

def _apply_paged_bitmap_patch(nvtx_enabled=False):
    """Replace FlashInfer attention with paged bitmap Triton kernel during verification.

    Combines bitmap block-skipping (from bitmap_sparse_attention) with zero-gather
    paged KV reads (from paged_sparse_attention). No KV gather, no GQA expand,
    GPU-only bitmap computation.

    Two-stage architecture:
      - Stage 1: Dense sequential iteration over past context, paged KV reads
      - Stage 2: Bitmap-driven sparse traversal over tree region, paged KV reads

    Cross-layer caching: layer 0 computes GPU bitmaps + tree mask; layers 1-31
    reuse (tree structure is identical across layers).
    """
    import torch
    import triton

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from paged_bitmap_attention import (
        compute_bitmap_from_mask_gpu,
        paged_bitmap_attention,
    )

    def _alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    try:
        triton.set_allocator(_alloc_fn)
    except Exception:
        pass

    _orig_forward_extend = FlashInferAttnBackend.forward_extend

    # Cross-layer cache: keyed by id(custom_mask).
    # Per-request: (tree_mask_2d, bitmaps, kv_indices, W)
    _metadata_cache = {}

    _nvtx = nvtx_enabled
    def _nvtx_push(name):
        if _nvtx:
            torch.cuda.nvtx.range_push(name)
    def _nvtx_pop():
        if _nvtx:
            torch.cuda.nvtx.range_pop()

    _debug = os.environ.get('PAGED_BITMAP_DEBUG', '0') == '1'
    _debug_count = [0]
    _debug_max = 3

    def _compute_our_kernel(q, forward_batch, layer, k_buf, v_buf):
        """Compute paged bitmap attention result (no KV save)."""
        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        D = layer.head_dim
        sm_scale = layer.scaling
        spec_info = forward_batch.spec_info
        N_tree = spec_info.draft_token_num
        bs = len(forward_batch.req_pool_indices)
        q_flat = q.view(-1, H_q, D)

        outputs = []
        for i in range(bs):
            req_pool_idx = forward_batch.req_pool_indices[i]
            past_len = forward_batch.seq_lens[i].item()
            total_kv = past_len + N_tree
            kv_indices = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :total_kv].to(torch.int32)
            tree_mask_bool = _extract_tree_mask_for_request(
                spec_info.custom_mask, forward_batch.seq_lens, i, N_tree)
            bitmaps, W = compute_bitmap_from_mask_gpu(tree_mask_bool)
            tree_mask_2d = torch.where(
                tree_mask_bool,
                torch.zeros(1, device=q.device, dtype=torch.float16),
                torch.full((1,), float('-inf'), device=q.device, dtype=torch.float16),
            )
            q_i = q_flat[i * N_tree : (i + 1) * N_tree]
            q_heads = q_i.permute(1, 0, 2).contiguous()
            out_i = paged_bitmap_attention(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                bitmaps, W, past_len, sm_scale, H_kv)
            outputs.append(out_i.permute(1, 0, 2))
        out = torch.cat(outputs, dim=0)
        return out.reshape(-1, H_q * D)

    def _paged_bitmap_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if not (hasattr(forward_batch, 'forward_mode') and
                forward_batch.forward_mode.is_target_verify()):
            return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        # DEBUG MODE: compare against FlashInfer on first few calls (layer 0 only)
        if _debug and layer.layer_id == 0 and _debug_count[0] < _debug_max:
            _debug_count[0] += 1
            step = _debug_count[0]

            # Get reference from FlashInfer (saves KV + computes attention)
            ref_result = _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

            # Get KV buffers (KV already saved by FlashInfer)
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            # Compute our kernel result (no KV save needed)
            our_result = _compute_our_kernel(q, forward_batch, layer, k_buf, v_buf)

            spec_info = forward_batch.spec_info
            N_tree = spec_info.draft_token_num
            bs = len(forward_batch.req_pool_indices)
            H_q = layer.tp_q_head_num
            H_kv = layer.tp_k_head_num
            D = layer.head_dim

            diff = (our_result.float() - ref_result.float()).abs()
            print(f"\n[DEBUG] Step {step}, Layer 0 comparison:", flush=True)
            print(f"  bs={bs}, N_tree={N_tree}, H_q={H_q}, H_kv={H_kv}, D={D}", flush=True)
            print(f"  sm_scale={layer.scaling}", flush=True)
            print(f"  q shape: {q.shape}, dtype: {q.dtype}", flush=True)
            print(f"  k_buf shape: {k_buf.shape}, dtype: {k_buf.dtype}", flush=True)
            for i in range(bs):
                past_len = forward_batch.seq_lens[i].item()
                print(f"  req[{i}]: past_len={past_len}", flush=True)

            # Per-token error analysis
            ref_2d = ref_result.view(N_tree, H_q * D).float()
            our_2d = our_result.view(N_tree, H_q * D).float()
            per_token_cos = torch.nn.functional.cosine_similarity(ref_2d, our_2d, dim=1)
            per_token_l2 = (ref_2d - our_2d).norm(dim=1)
            ref_l2 = ref_2d.norm(dim=1)
            print(f"  Per-token cosine sim (first 10): {per_token_cos[:10].tolist()}", flush=True)
            print(f"  Per-token cosine sim (last 5):  {per_token_cos[-5:].tolist()}", flush=True)
            print(f"  Per-token L2 diff  (first 10): {per_token_l2[:10].tolist()}", flush=True)
            print(f"  Per-token ref norm (first 10): {ref_l2[:10].tolist()}", flush=True)

            # Per-head error analysis
            ref_3d = ref_result.view(N_tree, H_q, D).float()
            our_3d = our_result.view(N_tree, H_q, D).float()
            for h in range(min(H_q, 4)):
                h_cos = torch.nn.functional.cosine_similarity(
                    ref_3d[:, h, :].flatten(), our_3d[:, h, :].flatten(), dim=0)
                h_diff = (ref_3d[:, h, :] - our_3d[:, h, :]).abs().max()
                print(f"  Head {h}: cosine={h_cos.item():.6f}, max_diff={h_diff.item():.6e}", flush=True)

            # Check if Stage 1 is the problem: token 0 only attends to past+self
            # If token 0 is correct but token 1+ wrong, bug is in Stage 2
            t0_cos = torch.nn.functional.cosine_similarity(
                ref_2d[0], our_2d[0], dim=0)
            print(f"  Token 0 cosine: {t0_cos.item():.6f}", flush=True)

            cos = torch.nn.functional.cosine_similarity(
                our_result.float().flatten(), ref_result.float().flatten(), dim=0)
            print(f"  Overall cosine sim: {cos.item():.6f}", flush=True)

            # Return FlashInfer result so model stays correct
            return ref_result

        # Non-debug path for layers > 0 or after debug limit
        if _debug and layer.layer_id > 0:
            return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        _nvtx_push(f"paged_bitmap_L{layer.layer_id}")

        # Save KV cache (same as original)
        _nvtx_push("save_kv_cache")
        cache_loc = (forward_batch.out_cache_loc if not layer.is_cross_attention
                     else forward_batch.encoder_out_cache_loc)
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale)
        _nvtx_pop()

        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        D = layer.head_dim
        sm_scale = layer.scaling
        spec_info = forward_batch.spec_info
        N_tree = spec_info.draft_token_num
        bs = len(forward_batch.req_pool_indices)

        _nvtx_push("kv_buffer_lookup")
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        _nvtx_pop()

        q_flat = q.view(-1, H_q, D)  # [bs*N_tree, H_q, D]

        # Cross-layer cache: layer 0 always recomputes; layers 1+ reuse.
        # Use layer_id == 0 instead of id()-based cache miss to avoid
        # stale hits from Python object id() reuse across verify steps.
        if layer.layer_id == 0:
            _nvtx_push("bitmap_computation_L0")
            per_req_cache = []
            for i in range(bs):
                req_pool_idx = forward_batch.req_pool_indices[i]
                past_len = forward_batch.seq_lens[i].item()
                total_kv = past_len + N_tree

                kv_indices = forward_batch.req_to_token_pool.req_to_token[
                    req_pool_idx, :total_kv].to(torch.int32)

                # Extract tree mask and compute GPU bitmaps
                tree_mask_bool = _extract_tree_mask_for_request(
                    spec_info.custom_mask, forward_batch.seq_lens, i, N_tree)

                _nvtx_push("gpu_bitmap")
                bitmaps, W = compute_bitmap_from_mask_gpu(tree_mask_bool)
                _nvtx_pop()

                # Convert bool -> [-inf, 0] fp16 mask
                tree_mask_2d = torch.where(
                    tree_mask_bool,
                    torch.zeros(1, device=q.device, dtype=torch.float16),
                    torch.full((1,), float('-inf'), device=q.device, dtype=torch.float16),
                )

                per_req_cache.append((tree_mask_2d, bitmaps, kv_indices, W, past_len))

            _metadata_cache['data'] = per_req_cache
            _nvtx_pop()  # bitmap_computation_L0
        cached = _metadata_cache['data']

        # Process each request: launch paged bitmap kernel
        _nvtx_push("per_request_loop")
        outputs = []
        for i in range(bs):
            _nvtx_push(f"req_{i}")
            tree_mask_2d, bitmaps, kv_indices, W, past_len_i = cached[i]

            # Q for this request: [N_tree, H_q, D] -> [H_q, N_tree, D]
            _nvtx_push("q_reshape")
            q_i = q_flat[i * N_tree : (i + 1) * N_tree]
            q_heads = q_i.permute(1, 0, 2).contiguous()  # [H_q, N_tree, D]
            _nvtx_pop()

            # Launch kernel: reads K/V directly from k_buf/v_buf
            _nvtx_push("kernel_call")
            out_i = paged_bitmap_attention(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                bitmaps, W, past_len_i, sm_scale, H_kv)
            _nvtx_pop()

            # out_i: [H_q, N_tree, D] -> [N_tree, H_q, D]
            outputs.append(out_i.permute(1, 0, 2))
            _nvtx_pop()  # req_i
        _nvtx_pop()  # per_request_loop

        _nvtx_push("output_concat")
        out = torch.cat(outputs, dim=0)  # [bs*N_tree, H_q, D]
        result = out.reshape(-1, H_q * D)
        _nvtx_pop()

        _nvtx_pop()  # paged_bitmap_L
        return result

    FlashInferAttnBackend.forward_extend = _paged_bitmap_forward_extend
    print("[PAGED_BITMAP] Patched with paged bitmap attention (GPU bitmaps, zero gather)",
          flush=True)
    if nvtx_enabled:
        print("[PAGED_BITMAP] NVTX markers enabled", flush=True)


def _apply_tile_bitmap_patch(nvtx_enabled=False, block_n=16,
                             quest_page_budget_ratio=None,
                             quest_token_budget=None,
                             quest_page_size=16,
                             quest_skip_layers=0,
                             quest_mode="per-round",
                             quest_stage1_impl="block_list"):
    """Replace FlashInfer attention with tile-level bitmap Triton kernel.

    Same two-stage architecture as _apply_paged_bitmap_patch, but with
    configurable BLOCK_N (default 16 for finer tile granularity).
    Each bitmap bit = one BLOCK_N-sized KV tile. bit=1 → load+MMA, bit=0 → skip.

    Cross-layer caching: layer 0 computes GPU bitmaps + tree mask; layers 1-31
    reuse (tree structure is identical across layers).

    Quest sparse Stage 1 (optional): when quest_page_budget_ratio is set,
    uses per-page min/max K metadata to select only the most important pages,
    reducing Stage 1 KV reads proportionally.

    Quest modes:
      - per-round: full K gather every verify round (original)
      - cached: cache page metadata across verify rounds (amortized)
      - physical-page: monkey-patch set_kv_buffer for zero K gather
    """
    import torch
    import triton

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from tile_bitmap_attention import (
        compute_tile_bitmap,
        tile_bitmap_attention,
        tile_bitmap_attention_batched,
    )
    _quest_enabled = quest_page_budget_ratio is not None or quest_token_budget is not None
    _quest_stage1_impl = quest_stage1_impl  # "block_list" or "bitmap_masked"
    if _quest_enabled:
        from tile_bitmap_attention import (
            compute_page_metadata,
            estimate_page_importance,
            select_important_pages,
            quest_page_scores_triton,
            build_quest_block_ids,
            build_quest_bitmap,
        )
        if quest_mode == "cached":
            from tile_bitmap_attention import update_page_metadata_incremental
        elif quest_mode == "physical-page":
            from tile_bitmap_attention import (
                create_physical_page_metadata,
                update_physical_page_metadata_triton,
                fused_kv_write_page_metadata,
                quest_from_physical_pages,
            )

    def _alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    try:
        triton.set_allocator(_alloc_fn)
    except Exception:
        pass

    _orig_forward_extend = FlashInferAttnBackend.forward_extend
    _metadata_cache = {}
    _block_n = block_n
    _quest_ratio = quest_page_budget_ratio
    _quest_token_budget = quest_token_budget
    _quest_page_size = quest_page_size
    _quest_skip_layers = quest_skip_layers
    _quest_mode = quest_mode

    # Approach 1 (cached): per-request page metadata cache
    _quest_page_cache = {}  # req_pool_idx → (page_min, page_max, cached_past_len)

    # Approach 2 (physical-page): global physical-page metadata buffers
    _phys_meta = {}  # 'min'/'max' → tensor, initialized lazily on first call

    _nvtx = nvtx_enabled
    def _nvtx_push(name):
        if _nvtx:
            torch.cuda.nvtx.range_push(name)
    def _nvtx_pop():
        if _nvtx:
            torch.cuda.nvtx.range_pop()

    def _tile_bitmap_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if not (hasattr(forward_batch, 'forward_mode') and
                forward_batch.forward_mode.is_target_verify()):
            return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        _nvtx_push(f"tile_bitmap_L{layer.layer_id}")

        # Save KV cache
        _nvtx_push("save_kv_cache")
        cache_loc = (forward_batch.out_cache_loc if not layer.is_cross_attention
                     else forward_batch.encoder_out_cache_loc)
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale)
        _nvtx_pop()

        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        D = layer.head_dim
        sm_scale = layer.scaling
        spec_info = forward_batch.spec_info
        N_tree = spec_info.draft_token_num
        bs = len(forward_batch.req_pool_indices)

        _nvtx_push("kv_buffer_lookup")
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        _nvtx_pop()

        q_flat = q.view(-1, H_q, D)

        # Cross-layer cache: layer 0 builds batched metadata; layers 1+ reuse.
        if layer.layer_id == 0:
            _nvtx_push("metadata_L0")

            # Shared tree mask and bitmaps (same tree for all requests in EAGLE3)
            tree_mask_bool = _extract_tree_mask_for_request(
                spec_info.custom_mask, forward_batch.seq_lens, 0, N_tree)
            bitmaps, W_val = compute_tile_bitmap(tree_mask_bool, BLOCK_N=_block_n)
            tree_mask_2d = torch.where(
                tree_mask_bool,
                torch.zeros(1, device=q.device, dtype=torch.float16),
                torch.full((1,), float('-inf'), device=q.device,
                           dtype=torch.float16),
            )

            # Pre-allocate batched tensors
            max_total_kv = int(forward_batch.seq_lens.max().item()) + N_tree
            kv_indices_batched = torch.zeros(bs, max_total_kv,
                                              dtype=torch.int32, device=q.device)
            past_lens_t = torch.zeros(bs, dtype=torch.int32, device=q.device)

            # For Quest skip layers: also store original (dense) kv_indices
            need_orig = _quest_enabled and _quest_skip_layers > 0
            if need_orig:
                kv_indices_orig = torch.zeros(bs, max_total_kv,
                                               dtype=torch.int32, device=q.device)
                past_lens_orig = torch.zeros(bs, dtype=torch.int32, device=q.device)

            for i in range(bs):
                req_pool_idx = forward_batch.req_pool_indices[i]
                past_len_i = forward_batch.seq_lens[i].item()
                total_kv = past_len_i + N_tree

                kv_indices = forward_batch.req_to_token_pool.req_to_token[
                    req_pool_idx, :total_kv].to(torch.int32)

                # Store original for skip layers
                if need_orig:
                    kv_indices_orig[i, :total_kv] = kv_indices
                    past_lens_orig[i] = past_len_i

                # Quest: rebuild kv_indices with page selection
                if _quest_enabled and past_len_i > _quest_page_size:
                    _nvtx_push(f"quest_rebuild_{i}")
                    q_i = q_flat[i * N_tree : (i + 1) * N_tree]
                    q_mean = q_i.float().mean(dim=0)
                    gqa_group = H_q // H_kv
                    q_rep = q_mean.view(H_kv, gqa_group, D).mean(dim=1)

                    if _quest_mode == "per-round":
                        page_scores = quest_page_scores_triton(
                            k_buf, kv_indices, past_len_i, q_rep,
                            page_size=_quest_page_size)
                        num_pages = len(page_scores)
                        if _quest_token_budget is not None:
                            page_budget = max(1, (_quest_token_budget + _quest_page_size - 1) // _quest_page_size)
                        else:
                            page_budget = max(1, int(num_pages * _quest_ratio))
                        kv_indices, past_len_i = select_important_pages(
                            page_scores, page_budget, kv_indices, past_len_i,
                            _quest_page_size)

                    elif _quest_mode == "cached":
                        req_key = req_pool_idx.item()
                        orig_past = past_len_i
                        orig_kv = kv_indices
                        if req_key not in _quest_page_cache:
                            page_min, page_max = compute_page_metadata(
                                k_buf, orig_kv, orig_past, H_kv, D,
                                page_size=_quest_page_size)
                            _quest_page_cache[req_key] = (page_min, page_max, orig_past)
                        else:
                            cached_min, cached_max, cached_past = _quest_page_cache[req_key]
                            page_min, page_max = update_page_metadata_incremental(
                                cached_min, cached_max, k_buf, orig_kv,
                                cached_past, orig_past, H_kv, D,
                                page_size=_quest_page_size)
                            _quest_page_cache[req_key] = (page_min, page_max, orig_past)

                        page_scores = estimate_page_importance(q_rep, page_min, page_max)
                        num_pages = len(page_scores)
                        if _quest_token_budget is not None:
                            page_budget = max(1, (_quest_token_budget + _quest_page_size - 1) // _quest_page_size)
                        else:
                            page_budget = max(1, int(num_pages * _quest_ratio))
                        kv_indices, past_len_i = select_important_pages(
                            page_scores, page_budget, orig_kv, orig_past,
                            _quest_page_size)

                    elif _quest_mode == "physical-page":
                        kv_indices, past_len_i = quest_from_physical_pages(
                            q_rep, _phys_meta['min'], _phys_meta['max'],
                            kv_indices, past_len_i,
                            _quest_ratio if _quest_ratio is not None else 0.5,
                            _quest_page_size)

                    _nvtx_pop()

                total_kv_i = len(kv_indices)
                kv_indices_batched[i, :total_kv_i] = kv_indices
                past_lens_t[i] = past_len_i

            cache_data = {
                'tree_mask_2d': tree_mask_2d,
                'bitmaps': bitmaps,
                'W': W_val,
                'kv_indices': kv_indices_batched,
                'past_lens': past_lens_t,
            }
            if need_orig:
                cache_data['kv_indices_orig'] = kv_indices_orig
                cache_data['past_lens_orig'] = past_lens_orig

            _metadata_cache['data'] = cache_data
            _nvtx_pop()

        cached = _metadata_cache['data']

        # Select kv_indices: original (skip layers) or Quest-rebuilt
        skip_quest = (_quest_enabled and _quest_skip_layers > 0
                      and layer.layer_id < _quest_skip_layers)
        if skip_quest:
            use_kv = cached['kv_indices_orig']
            use_past = cached['past_lens_orig']
        else:
            use_kv = cached['kv_indices']
            use_past = cached['past_lens']

        # Reshape Q for batched kernel: [bs*N_tree, H_q, D] -> [bs, H_q, N_tree, D]
        _nvtx_push("q_reshape")
        q_batched = q_flat.view(bs, N_tree, H_q, D).permute(0, 2, 1, 3).contiguous()
        _nvtx_pop()

        # Single batched kernel launch (replaces per-request loop)
        _nvtx_push("kernel_call")
        out = tile_bitmap_attention_batched(
            q_batched, k_buf, v_buf, use_kv,
            cached['tree_mask_2d'], cached['bitmaps'], cached['W'], use_past,
            sm_scale, H_kv, BLOCK_N=_block_n)
        _nvtx_pop()

        # Reshape output: [bs, H_q, N_tree, D] -> [bs*N_tree, H_q*D]
        result = out.permute(0, 2, 1, 3).reshape(-1, H_q * D)

        _nvtx_pop()  # tile_bitmap_L
        return result

    FlashInferAttnBackend.forward_extend = _tile_bitmap_forward_extend
    print(f"[TILE_BITMAP] Patched with BATCHED tile bitmap attention "
          f"(BLOCK_N={block_n}, GPU bitmaps, zero gather, single kernel launch)", flush=True)
    if nvtx_enabled:
        print("[TILE_BITMAP] NVTX markers enabled", flush=True)

    # Physical-page mode: monkey-patch set_kv_buffer to maintain page metadata
    # Uses fused Triton kernel: KV write + page min/max update in one launch
    if _quest_enabled and quest_mode == "physical-page":
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        _orig_set_kv = MHATokenToKVPool.set_kv_buffer

        def _patched_set_kv_buffer(self, layer, loc, cache_k, cache_v,
                                    k_scale=None, v_scale=None,
                                    layer_id_override=None):
            # Always use original for KV write (all layers)
            _orig_set_kv(self, layer, loc, cache_k, cache_v,
                         k_scale, v_scale, layer_id_override)
            lid = layer_id_override if layer_id_override is not None else layer.layer_id
            if lid == 0 and cache_k is not None:
                # Lazy init: allocate metadata buffers on first call
                if 'min' not in _phys_meta:
                    H_kv_local = cache_k.shape[1] if cache_k.dim() == 3 else cache_k.shape[-2]
                    D_local = cache_k.shape[-1]
                    max_tokens = self.get_key_buffer(0).shape[0]
                    pmin, pmax = create_physical_page_metadata(
                        max_tokens, H_kv_local, D_local,
                        quest_page_size, device=cache_k.device)
                    _phys_meta['min'] = pmin
                    _phys_meta['max'] = pmax
                    print(f"[QUEST-PHYS] Allocated physical page metadata: "
                          f"{pmin.shape[0]} pages, page_size={quest_page_size} "
                          f"(Triton metadata kernel)", flush=True)
                # Triton metadata update (replaces scatter_reduce)
                k_for_meta = cache_k
                if k_for_meta.dim() == 2:
                    H_kv_local = _phys_meta['min'].shape[1]
                    D_local = _phys_meta['min'].shape[2]
                    k_for_meta = k_for_meta.view(-1, H_kv_local, D_local)
                update_physical_page_metadata_triton(
                    _phys_meta['min'], _phys_meta['max'],
                    loc, k_for_meta, quest_page_size)

        MHATokenToKVPool.set_kv_buffer = _patched_set_kv_buffer
        print(f"[QUEST-PHYS] Patched set_kv_buffer with Triton metadata kernel "
              f"(page_size={quest_page_size})", flush=True)

        # Hook allocator.free() to reset page metadata when slots are returned
        # Without this, page min/max accumulates stale values across requests
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        _orig_alloc_free = TokenToKVPoolAllocator.free

        def _patched_alloc_free(self_alloc, free_index):
            if free_index.numel() > 0 and 'min' in _phys_meta:
                freed_pages = (free_index // quest_page_size).unique()
                freed_pages = freed_pages.clamp(0, _phys_meta['min'].shape[0] - 1)
                _phys_meta['min'][freed_pages] = float('inf')
                _phys_meta['max'][freed_pages] = float('-inf')
            _orig_alloc_free(self_alloc, free_index)

        TokenToKVPoolAllocator.free = _patched_alloc_free
        print(f"[QUEST-PHYS] Patched allocator.free() for page metadata reset",
              flush=True)


def _apply_paged_bitmap_cuda_patch(nvtx_enabled=False):
    """Replace FlashInfer attention with compiled CUDA paged bitmap kernel.

    Same algorithm as _apply_paged_bitmap_patch but uses the compiled C++
    extension from csrc/paged_bitmap_attention.cu instead of Triton.
    """
    import torch
    import sys as _sys

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

    # Import CUDA extension
    _csrc_dir = os.path.join(os.path.dirname(__file__), "..", "csrc")
    if _csrc_dir not in _sys.path:
        _sys.path.insert(0, _csrc_dir)
    import paged_bitmap_attention_ext

    # Reuse GPU bitmap computation from Triton version
    from paged_bitmap_attention import compute_bitmap_from_mask_gpu

    _orig_forward_extend = FlashInferAttnBackend.forward_extend
    _metadata_cache = {}

    _nvtx = nvtx_enabled
    def _nvtx_push(name):
        if _nvtx:
            torch.cuda.nvtx.range_push(name)
    def _nvtx_pop():
        if _nvtx:
            torch.cuda.nvtx.range_pop()

    def _paged_bitmap_cuda_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if not (hasattr(forward_batch, 'forward_mode') and
                forward_batch.forward_mode.is_target_verify()):
            return _orig_forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache)

        _nvtx_push(f"paged_bitmap_cuda_L{layer.layer_id}")

        # Save KV cache
        _nvtx_push("save_kv_cache")
        cache_loc = (forward_batch.out_cache_loc if not layer.is_cross_attention
                     else forward_batch.encoder_out_cache_loc)
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale)
        _nvtx_pop()

        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        D = layer.head_dim
        sm_scale = layer.scaling
        spec_info = forward_batch.spec_info
        N_tree = spec_info.draft_token_num
        bs = len(forward_batch.req_pool_indices)

        _nvtx_push("kv_buffer_lookup")
        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        _nvtx_pop()

        q_flat = q.view(-1, H_q, D)

        if layer.layer_id == 0:
            _nvtx_push("bitmap_computation_L0")
            per_req_cache = []
            BLOCK_M = paged_bitmap_attention_ext.get_block_m(N_tree)
            for i in range(bs):
                req_pool_idx = forward_batch.req_pool_indices[i]
                past_len = forward_batch.seq_lens[i].item()
                total_kv = past_len + N_tree

                kv_indices = forward_batch.req_to_token_pool.req_to_token[
                    req_pool_idx, :total_kv].to(torch.int32)

                tree_mask_bool = _extract_tree_mask_for_request(
                    spec_info.custom_mask, forward_batch.seq_lens, i, N_tree)

                _nvtx_push("gpu_bitmap")
                bitmaps, W = compute_bitmap_from_mask_gpu(
                    tree_mask_bool, BLOCK_M=BLOCK_M, BLOCK_N=32)
                _nvtx_pop()

                tree_mask_2d = torch.where(
                    tree_mask_bool,
                    torch.zeros(1, device=q.device, dtype=torch.float16),
                    torch.full((1,), float('-inf'), device=q.device, dtype=torch.float16),
                )

                per_req_cache.append((tree_mask_2d, bitmaps, kv_indices, W, past_len))

            _metadata_cache['data'] = per_req_cache
            _nvtx_pop()
        cached = _metadata_cache['data']

        _nvtx_push("per_request_loop")
        outputs = []
        for i in range(bs):
            _nvtx_push(f"req_{i}")
            tree_mask_2d, bitmaps, kv_indices, W, past_len_i = cached[i]

            _nvtx_push("q_reshape")
            q_i = q_flat[i * N_tree : (i + 1) * N_tree]
            q_heads = q_i.permute(1, 0, 2).contiguous()
            _nvtx_pop()

            _nvtx_push("kernel_call")
            out_i = paged_bitmap_attention_ext.forward(
                q_heads, k_buf, v_buf, kv_indices, tree_mask_2d,
                bitmaps, sm_scale, past_len_i, W, H_kv)
            _nvtx_pop()

            outputs.append(out_i.permute(1, 0, 2))
            _nvtx_pop()
        _nvtx_pop()

        _nvtx_push("output_concat")
        out = torch.cat(outputs, dim=0)
        result = out.reshape(-1, H_q * D)
        _nvtx_pop()

        _nvtx_pop()
        return result

    FlashInferAttnBackend.forward_extend = _paged_bitmap_cuda_forward_extend
    print("[PAGED_BITMAP_CUDA] Patched with compiled CUDA paged bitmap attention",
          flush=True)
    if nvtx_enabled:
        print("[PAGED_BITMAP_CUDA] NVTX markers enabled", flush=True)


def _extract_tree_mask_for_request(custom_mask, seq_lens, req_idx, N_tree):
    """Extract [N_tree, N_tree] tree mask for one request from flat custom_mask.

    FlashInfer custom_mask layout: for each request i, a flat block of
    (seq_lens[i] + N_tree) * N_tree booleans, representing the 2D mask
    [N_tree rows (Q), seq_lens[i]+N_tree cols (KV)]. Requests concatenated.
    """
    import torch

    # Compute offset into flat mask
    offset = 0
    for j in range(req_idx):
        past_j = seq_lens[j].item()
        offset += (past_j + N_tree) * N_tree

    past_len = seq_lens[req_idx].item()
    total_kv = past_len + N_tree

    # Extract 2D mask for this request
    mask_flat = custom_mask[offset : offset + total_kv * N_tree]
    mask_2d = mask_flat.view(N_tree, total_kv)

    # Extract tree region (last N_tree columns)
    return mask_2d[:, past_len:]  # [N_tree, N_tree] bool


def _compute_subtree_metadata_from_mask(tree_mask_bool, N_tree, BLOCK_M=32, BLOCK_N=32):
    """Compute DFS permutation + block-sparse metadata from boolean tree mask.

    Reconstructs tree parent-child structure from the boolean mask,
    computes DFS traversal order, then builds block-sparse indices
    in the DFS-permuted coordinate space.

    Args:
        tree_mask_bool: [N_tree, N_tree] boolean tensor on GPU.
        N_tree: number of tree tokens.

    Returns:
        perm, inv_perm: [N_tree] int64 DFS permutation tensors (CPU).
        block_indices: [num_q_blocks, MAX_SPARSE_BLOCKS] int32 (CPU).
        block_counts: [num_q_blocks] int32 (CPU).
        MAX_SPARSE_BLOCKS: int.
    """
    import torch

    # Move to CPU for tree reconstruction
    mask_cpu = tree_mask_bool.cpu()

    # Build parent-child adjacency from mask.
    # For each node i, parent = largest j < i where mask[i, j] = True.
    # This works because EAGLE3 generates tokens level-by-level (parents < children).
    children = {i: [] for i in range(N_tree)}
    for i in range(1, N_tree):
        row = mask_cpu[i]
        ancestors = row.nonzero(as_tuple=True)[0]
        ancestors_below = ancestors[ancestors < i]
        parent = ancestors_below.max().item() if len(ancestors_below) > 0 else 0
        children[parent].append(i)

    # DFS traversal (iterative, left-to-right)
    perm_list = []
    stack = [0]
    while stack:
        node = stack.pop()
        perm_list.append(node)
        for child in reversed(children[node]):
            stack.append(child)

    perm = torch.tensor(perm_list, dtype=torch.int64)
    inv_perm = torch.empty(N_tree, dtype=torch.int64)
    inv_perm[perm] = torch.arange(N_tree)

    # Compute block-sparse metadata in DFS order
    num_q_blocks = (N_tree + BLOCK_M - 1) // BLOCK_M
    block_lists = []
    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min(q_start + BLOCK_M, N_tree)
        kv_blocks = set()
        for new_q in range(q_start, q_end):
            old_q = perm[new_q].item()
            ancestors = mask_cpu[old_q].nonzero(as_tuple=True)[0]
            for anc in ancestors:
                new_anc = inv_perm[anc].item()
                kv_blocks.add(new_anc // BLOCK_N)
        block_lists.append(sorted(kv_blocks))

    MAX_SPARSE_BLOCKS = max(len(bl) for bl in block_lists) if block_lists else 1
    block_indices = torch.zeros(num_q_blocks, MAX_SPARSE_BLOCKS, dtype=torch.int32)
    block_counts = torch.zeros(num_q_blocks, dtype=torch.int32)
    for qb, bl in enumerate(block_lists):
        block_counts[qb] = len(bl)
        for idx, kv_b in enumerate(bl):
            block_indices[qb, idx] = kv_b

    return perm, inv_perm, block_indices, block_counts, MAX_SPARSE_BLOCKS


# ============================================================================
# Scheduler subprocess wrappers (must be module-level for pickling)
# ============================================================================

def _patched_run_scheduler_process(server_args, port_args, gpu_id, tp_rank,
                                   moe_ep_rank, pp_rank, dp_rank, pipe_writer):
    """Scheduler process wrapper that applies monkey-patches before starting.

    Reads configuration from environment variables:
        SGLANG_BENCH_SCRIPTS_DIR: path to scripts/ for imports
        SGLANG_BENCH_FLAGS: comma-separated flags (profile, subtree)
        SGLANG_ATTN_PROFILE_FILE: path for profiling JSON output
    """
    # Ensure scripts directory is in path (needed for sparse_tree_kernel imports)
    scripts_dir = os.environ.get("SGLANG_BENCH_SCRIPTS_DIR", "")
    if scripts_dir and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    flags = os.environ.get("SGLANG_BENCH_FLAGS", "")
    profile_file = os.environ.get("SGLANG_ATTN_PROFILE_FILE", "")

    if "profile" in flags and profile_file:
        _apply_profiling_patch(profile_file)
        print(f"[PROFILE] Attention profiling enabled in TP rank {tp_rank}", flush=True)

    if "tile_bitmap" in flags:
        nvtx = "nvtx" in flags
        block_n = int(os.environ.get("SGLANG_TILE_BITMAP_BLOCK_N", "16"))
        quest_ratio_str = os.environ.get("SGLANG_QUEST_PAGE_BUDGET_RATIO", "")
        quest_ratio = float(quest_ratio_str) if quest_ratio_str else None
        quest_token_budget_str = os.environ.get("SGLANG_QUEST_TOKEN_BUDGET", "")
        quest_token_budget = int(quest_token_budget_str) if quest_token_budget_str else None
        quest_page_size = int(os.environ.get("SGLANG_QUEST_PAGE_SIZE", "16"))
        quest_skip_layers = int(os.environ.get("SGLANG_QUEST_SKIP_LAYERS", "0"))
        quest_mode = os.environ.get("SGLANG_QUEST_MODE", "per-round")
        quest_stage1_impl = os.environ.get("SGLANG_QUEST_STAGE1_IMPL", "block_list")
        _apply_tile_bitmap_patch(nvtx_enabled=nvtx, block_n=block_n,
                                 quest_page_budget_ratio=quest_ratio,
                                 quest_token_budget=quest_token_budget,
                                 quest_page_size=quest_page_size,
                                 quest_skip_layers=quest_skip_layers,
                                 quest_mode=quest_mode,
                                 quest_stage1_impl=quest_stage1_impl)
        if quest_ratio is not None or quest_token_budget is not None:
            budget_desc = f"token_budget={quest_token_budget}" if quest_token_budget else f"ratio={quest_ratio}"
            print(f"[TILE_BITMAP] Tile bitmap kernel enabled in TP rank {tp_rank} "
                  f"(BLOCK_N={block_n}, Quest mode={quest_mode}, impl={quest_stage1_impl}, "
                  f"{budget_desc}, page_size={quest_page_size}, "
                  f"skip_layers={quest_skip_layers})", flush=True)
        else:
            print(f"[TILE_BITMAP] Tile bitmap kernel enabled in TP rank {tp_rank} "
                  f"(BLOCK_N={block_n})", flush=True)
    elif "paged_bitmap_cuda" in flags:
        nvtx = "nvtx" in flags
        _apply_paged_bitmap_cuda_patch(nvtx_enabled=nvtx)
        print(f"[PAGED_BITMAP_CUDA] CUDA paged bitmap kernel enabled in TP rank {tp_rank}", flush=True)
    elif "paged_bitmap" in flags:
        nvtx = "nvtx" in flags
        _apply_paged_bitmap_patch(nvtx_enabled=nvtx)
        print(f"[PAGED_BITMAP] Paged bitmap kernel enabled in TP rank {tp_rank}", flush=True)
    elif "bitmap" in flags:
        nvtx = "nvtx" in flags
        _apply_bitmap_kernel_patch(nvtx_enabled=nvtx)
        print(f"[BITMAP] CUDA bitmap kernel enabled in TP rank {tp_rank}", flush=True)
    elif "paged" in flags:
        ratio = float(os.environ.get("SGLANG_KV_TOP_K_RATIO", "0.3"))
        paged_max_past = int(os.environ.get("SGLANG_PAGED_MAX_PAST", "0"))
        nvtx = "nvtx" in flags
        _apply_paged_sparse_patch(nvtx_enabled=nvtx, top_k_ratio=ratio,
                                  max_past=paged_max_past)
        print(f"[PAGED] Paged sparse kernel enabled in TP rank {tp_rank} "
              f"(ratio={ratio}, max_past={paged_max_past})", flush=True)
    elif "unified" in flags:
        ratio = float(os.environ.get("SGLANG_KV_TOP_K_RATIO", "0.3"))
        unified_max_past = int(os.environ.get("SGLANG_UNIFIED_MAX_PAST", "4096"))
        nvtx = "nvtx" in flags
        _apply_unified_sparse_patch(nvtx_enabled=nvtx, top_k_ratio=ratio,
                                    max_past=unified_max_past)
        print(f"[UNIFIED] Unified sparse kernel enabled in TP rank {tp_rank} "
              f"(max_past={unified_max_past})", flush=True)
    elif "subtree" in flags:
        nvtx = "nvtx" in flags
        _apply_subtree_kernel_patch(
            profile_file if "profile" in flags else None,
            nvtx_enabled=nvtx,
        )
        print(f"[SUBTREE] Subtree kernel enabled in TP rank {tp_rank}", flush=True)
    elif "nvtx" in flags:
        # NVTX without subtree kernel: instrument baseline FlashInfer path
        _apply_nvtx_baseline_patch()

    from sglang.srt.managers.scheduler import run_scheduler_process
    return run_scheduler_process(server_args, port_args, gpu_id, tp_rank,
                                 moe_ep_rank, pp_rank, dp_rank, pipe_writer)


# ============================================================================
# CLI and benchmark logic
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE3 speculative decoding with SGLang"
    )
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=5,
                        help="Depth of autoregressive drafting (default: 5)")
    parser.add_argument("--eagle-topk", type=int, default=8,
                        help="Branching factor per step (default: 8)")
    parser.add_argument("--num-draft-tokens", type=int, default=63,
                        help="Max parallel verification tokens (default: 63)")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=48,
                        help="Max running requests / batch size (default: 48)")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model dtype (default: float16)")
    parser.add_argument("--disable-cuda-graph", action="store_true",
                        help="Disable CUDA graph capture")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=None,
                        help="Fraction of GPU memory for static allocation (model + KV cache)")
    parser.add_argument("--log-level", type=str, default="error",
                        help="SGLang log level (default: error). Use 'info' for scheduling details.")
    # Profiling and kernel replacement flags
    parser.add_argument("--profile-attention", action="store_true",
                        help="Profile attention kernel time during tree verification. "
                             "Instruments RadixAttention.forward (layer 0) with CUDA event timing.")
    parser.add_argument("--use-subtree-kernel", action="store_true",
                        help="Replace FlashInfer attention with subtree Triton kernel "
                             "during EAGLE3 verification. Gathers K/V from paged cache, "
                             "computes DFS metadata at runtime. Adds overhead from gather. "
                             "Requires Hopper GPU (sm_90+) for TMA.")
    parser.add_argument("--profile-nvtx", action="store_true",
                        help="Add NVTX markers around kernel stages. "
                             "Use with nsys profile --trace=cuda,nvtx.")
    parser.add_argument("--use-unified-sparse", action="store_true",
                        help="Replace FlashInfer attention with unified sparse kernel. "
                             "Selects important KV from past context via first-query scoring, "
                             "keeps all tree positions, uses block-sparse Triton kernel.")
    parser.add_argument("--kv-top-k-ratio", type=float, default=0.3,
                        help="Fraction of past-context KV to keep (default: 0.3). "
                             "Only used with --use-unified-sparse.")
    parser.add_argument("--unified-sparse-max-past", type=int, default=4096,
                        help="Max past context length for unified sparse kernel "
                             "(default: 4096). Falls back to FlashInfer when "
                             "past_len exceeds this. Set 0 to disable fallback.")
    parser.add_argument("--use-bitmap-kernel", action="store_true",
                        help="Replace FlashInfer attention with CUDA bitmap attention "
                             "kernel during EAGLE3 verification. Two-stage: block-index "
                             "past context + bit-level bitmap tree region. Gathers K/V "
                             "from paged cache. Requires building csrc/bitmap_attention.cu.")
    parser.add_argument("--use-paged-sparse", action="store_true",
                        help="Replace FlashInfer attention with paged sparse Triton "
                             "kernel. Reads K/V directly from paged buffer via "
                             "indirect indexing — zero gather, zero CPU metadata. "
                             "Uses --kv-top-k-ratio for importance selection.")
    parser.add_argument("--paged-sparse-max-past", type=int, default=0,
                        help="Max past context for paged sparse kernel (default: 0 = "
                             "no fallback). Falls back to FlashInfer when past_len "
                             "exceeds this.")
    parser.add_argument("--use-paged-bitmap", action="store_true",
                        help="Replace FlashInfer attention with paged bitmap Triton "
                             "kernel. Combines bitmap block-skipping with zero-gather "
                             "paged KV reads. GPU-only bitmap computation, no KV "
                             "gather, no GQA expand. Works on Ampere+.")
    parser.add_argument("--use-paged-bitmap-cuda", action="store_true",
                        help="Replace FlashInfer attention with compiled CUDA paged "
                             "bitmap kernel (csrc/paged_bitmap_attention.cu). Same "
                             "algorithm as --use-paged-bitmap but compiled C++ extension "
                             "instead of Triton.")
    parser.add_argument("--use-tile-bitmap", action="store_true",
                        help="Replace FlashInfer attention with tile-level bitmap "
                             "Triton kernel. Same two-stage paged architecture as "
                             "--use-paged-bitmap but with configurable BLOCK_N "
                             "(default 16) for finer tile granularity. Each bitmap "
                             "bit = one TC-friendly KV tile.")
    parser.add_argument("--tile-block-n", type=int, default=16,
                        choices=[16, 32],
                        help="KV tile size for tile bitmap kernel (default: 16). "
                             "Only used with --use-tile-bitmap.")
    parser.add_argument("--quest-page-budget-ratio", type=float, default=None,
                        help="Quest-style sparse Stage 1: fraction of KV pages to keep "
                             "(0.0-1.0). None = dense Stage 1. Only with --use-tile-bitmap.")
    parser.add_argument("--quest-token-budget", type=int, default=None,
                        help="Quest-style sparse Stage 1: absolute token budget "
                             "(fixed number of tokens to keep). Overrides --quest-page-budget-ratio. "
                             "Only with --use-tile-bitmap.")
    parser.add_argument("--quest-page-size", type=int, default=16,
                        help="Quest page size in tokens for per-page min/max metadata "
                             "(default: 16). Only with --quest-page-budget-ratio.")
    parser.add_argument("--quest-skip-layers", type=int, default=0,
                        help="Number of initial layers to skip Quest (use full KV). "
                             "Quest paper recommends 2. Default: 0 (apply to all layers).")
    parser.add_argument("--quest-mode", type=str, default="per-round",
                        choices=["per-round", "cached", "physical-page"],
                        help="Quest page selection mode: "
                             "'per-round' = full K gather every verify round (baseline), "
                             "'cached' = cache page metadata across rounds (amortized), "
                             "'physical-page' = monkey-patch set_kv_buffer for zero K gather. "
                             "Only with --quest-page-budget-ratio.")
    parser.add_argument("--quest-stage1-impl", type=str, default="block_list",
                        choices=["block_list", "bitmap_masked"],
                        help="Quest Stage 1 in-kernel implementation: "
                             "'block_list' = compact sorted block IDs + tl.range, "
                             "'bitmap_masked' = dense tl.range over all blocks with bitmap predication. "
                             "Only with --quest-token-budget or --quest-page-budget-ratio.")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Max prompt tokens for dataset filtering (longbench only)")
    parser.add_argument("--attention-backend", type=str, default=None,
                        choices=["triton", "flashinfer"],
                        help="Force attention backend (default: auto-detect)")
    parser.add_argument("--output-log", type=str, default=None,
                        help="Save benchmark results (metrics + sample outputs) to a JSON file "
                             "in the current directory. Auto-generates filename if 'auto'.")
    return parser.parse_args()


def read_profile_results(profile_file, total_time, total_verify_ct):
    """Read and display attention profiling results from subprocess JSON.

    Returns dict with profiling metrics, or None if profiling data unavailable.
    """
    # Wait for subprocess atexit handlers to flush
    for _ in range(30):
        if os.path.exists(profile_file):
            # Wait a bit more for write to complete
            time.sleep(0.5)
            break
        time.sleep(0.2)

    if not os.path.exists(profile_file):
        print(f"[PROFILE] Warning: profile file not found at {profile_file}")
        print("[PROFILE] The scheduler subprocess may not have flushed yet.")
        return None

    try:
        with open(profile_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[PROFILE] Warning: could not read profile file: {e}")
        return None

    verify_times = data.get("verify_extend_times_ms", [])
    other_times = data.get("other_extend_times_ms", [])
    verify_q_tokens = data.get("verify_extend_q_tokens", [])

    if not verify_times:
        print("[PROFILE] No verify attention calls recorded.")
        print("[PROFILE] Possible causes:")
        print("  - CUDA graph captured the attention calls (try --disable-cuda-graph)")
        print("  - The subprocess didn't apply the profiling patch")
        return None

    total_verify_attn_ms = sum(verify_times)
    avg_verify_attn_ms = total_verify_attn_ms / len(verify_times)
    total_other_attn_ms = sum(other_times) if other_times else 0
    avg_q_tokens = sum(verify_q_tokens) / len(verify_q_tokens) if verify_q_tokens else 0

    # Attention fraction: this measures layer 0 only.
    # Full-model attention = N_layers * layer_0_time (approximate).
    total_time_ms = total_time * 1000
    attn_pct_of_total = (total_verify_attn_ms / total_time_ms) * 100 if total_time_ms > 0 else 0

    # Per-iteration metrics
    attn_per_iteration_ms = total_verify_attn_ms / total_verify_ct if total_verify_ct > 0 else 0
    iter_time_ms = (total_time / total_verify_ct) * 1000 if total_verify_ct > 0 else 0
    attn_pct_of_iteration = (attn_per_iteration_ms / iter_time_ms) * 100 if iter_time_ms > 0 else 0

    print("\n" + "-" * 70)
    print("ATTENTION PROFILING (layer 0 only):")
    print(f"  Verify attention calls: {len(verify_times)}")
    print(f"  Other attention calls:  {len(other_times)}")
    print(f"  Avg verify attn time:   {avg_verify_attn_ms:.4f} ms  (layer 0)")
    print(f"  Total verify attn time: {total_verify_attn_ms:.2f} ms (layer 0)")
    print(f"  Avg Q tokens per call:  {avg_q_tokens:.0f}")
    print(f"  Attn per iteration:     {attn_per_iteration_ms:.4f} ms (layer 0)")
    print(f"  Iteration time:         {iter_time_ms:.4f} ms")
    print(f"  Attn % of iteration:    {attn_pct_of_iteration:.3f}%  (layer 0)")
    print(f"  Attn % of total time:   {attn_pct_of_total:.3f}%  (layer 0)")

    return {
        "verify_attn_calls": len(verify_times),
        "other_attn_calls": len(other_times),
        "avg_verify_attn_ms": avg_verify_attn_ms,
        "total_verify_attn_ms": total_verify_attn_ms,
        "avg_q_tokens": avg_q_tokens,
        "attn_per_iteration_ms": attn_per_iteration_ms,
        "attn_pct_of_iteration": attn_pct_of_iteration,
        "attn_pct_of_total": attn_pct_of_total,
    }


def main():
    args = parse_args()

    # Determine if we need to patch the scheduler subprocess
    need_patch = (args.profile_attention or args.use_subtree_kernel
                  or args.profile_nvtx or args.use_unified_sparse
                  or args.use_paged_sparse or args.use_bitmap_kernel
                  or args.use_paged_bitmap or args.use_paged_bitmap_cuda
                  or args.use_tile_bitmap)

    # Set up environment for subprocess communication
    profile_file = None
    if need_patch:
        os.environ["SGLANG_BENCH_SCRIPTS_DIR"] = _SCRIPTS_DIR

        flags = []
        if args.profile_attention:
            profile_file = tempfile.mktemp(suffix=".json", prefix="sglang_attn_profile_")
            os.environ["SGLANG_ATTN_PROFILE_FILE"] = profile_file
            flags.append("profile")
            print(f"[PROFILE] Attention profiling enabled, output: {profile_file}")
        if args.use_tile_bitmap:
            flags.append("tile_bitmap")
            os.environ["SGLANG_TILE_BITMAP_BLOCK_N"] = str(args.tile_block_n)
            quest_enabled = args.quest_page_budget_ratio is not None or args.quest_token_budget is not None
            if quest_enabled:
                if args.quest_token_budget is not None:
                    os.environ["SGLANG_QUEST_TOKEN_BUDGET"] = str(args.quest_token_budget)
                if args.quest_page_budget_ratio is not None:
                    os.environ["SGLANG_QUEST_PAGE_BUDGET_RATIO"] = str(args.quest_page_budget_ratio)
                os.environ["SGLANG_QUEST_PAGE_SIZE"] = str(args.quest_page_size)
                os.environ["SGLANG_QUEST_SKIP_LAYERS"] = str(args.quest_skip_layers)
                os.environ["SGLANG_QUEST_MODE"] = args.quest_mode
                os.environ["SGLANG_QUEST_STAGE1_IMPL"] = args.quest_stage1_impl
                budget_desc = (f"token_budget={args.quest_token_budget}"
                               if args.quest_token_budget else
                               f"ratio={args.quest_page_budget_ratio}")
                print(f"[TILE_BITMAP] Will use tile bitmap attention "
                      f"(BLOCK_N={args.tile_block_n}, Quest sparse Stage 1: "
                      f"mode={args.quest_mode}, impl={args.quest_stage1_impl}, "
                      f"{budget_desc}, page_size={args.quest_page_size}, "
                      f"skip_layers={args.quest_skip_layers})")
            else:
                print(f"[TILE_BITMAP] Will use tile bitmap attention "
                      f"(BLOCK_N={args.tile_block_n}, GPU bitmaps, zero gather)")
        elif args.use_paged_bitmap_cuda:
            flags.append("paged_bitmap_cuda")
            print("[PAGED_BITMAP_CUDA] Will use compiled CUDA paged bitmap attention")
        elif args.use_paged_bitmap:
            flags.append("paged_bitmap")
            print("[PAGED_BITMAP] Will use paged bitmap attention (GPU bitmaps, zero gather)")
        elif args.use_bitmap_kernel:
            flags.append("bitmap")
            print("[BITMAP] Will replace attention with CUDA bitmap kernel during verification")
            print("[BITMAP] WARNING: KV cache gather adds overhead per layer per request")
        elif args.use_paged_sparse:
            flags.append("paged")
            os.environ["SGLANG_KV_TOP_K_RATIO"] = str(args.kv_top_k_ratio)
            os.environ["SGLANG_PAGED_MAX_PAST"] = str(args.paged_sparse_max_past)
            print(f"[PAGED] Will use paged sparse attention (ratio={args.kv_top_k_ratio}, "
                  f"max_past={args.paged_sparse_max_past})")
        elif args.use_unified_sparse:
            flags.append("unified")
            os.environ["SGLANG_KV_TOP_K_RATIO"] = str(args.kv_top_k_ratio)
            os.environ["SGLANG_UNIFIED_MAX_PAST"] = str(args.unified_sparse_max_past)
            print(f"[UNIFIED] Will use unified sparse attention (ratio={args.kv_top_k_ratio}, "
                  f"max_past={args.unified_sparse_max_past})")
        elif args.use_subtree_kernel:
            flags.append("subtree")
            print("[SUBTREE] Will replace attention with subtree kernel during verification")
            print("[SUBTREE] WARNING: KV cache gather adds overhead per layer per request")
        if args.profile_nvtx:
            flags.append("nvtx")
            if args.use_tile_bitmap:
                print("[NVTX] NVTX markers enabled for tile bitmap kernel stages")
            elif args.use_paged_bitmap_cuda:
                print("[NVTX] NVTX markers enabled for CUDA paged bitmap kernel stages")
            elif args.use_paged_bitmap:
                print("[NVTX] NVTX markers enabled for paged bitmap kernel stages")
            elif args.use_bitmap_kernel:
                print("[NVTX] NVTX markers enabled for CUDA bitmap kernel stages")
            elif args.use_unified_sparse:
                print("[NVTX] NVTX markers enabled for unified sparse kernel stages")
            elif args.use_subtree_kernel:
                print("[NVTX] NVTX markers enabled for subtree kernel stages")
            else:
                print("[NVTX] NVTX markers enabled for baseline FlashInfer path")
        os.environ["SGLANG_BENCH_FLAGS"] = ",".join(flags)

    # Load prompts
    if args.dataset:
        prompts = detect_and_load(args.dataset, args.num_prompts,
                                  max_tokens=args.max_context_tokens)
    else:
        prompts = SAMPLE_PROMPTS
        if args.num_prompts is not None:
            prompts = prompts[:args.num_prompts]
    print(f"Running with {len(prompts)} prompts")

    print(f"\nTarget model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"TP size: {args.tp_size}")
    print(f"EAGLE3 params: num_steps={args.num_steps}, "
          f"eagle_topk={args.eagle_topk}, "
          f"num_draft_tokens={args.num_draft_tokens}")

    # Create SGLang engine with EAGLE3 dynamic tree
    engine_kwargs = dict(
        model_path=args.target_model,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=args.draft_model,
        speculative_num_steps=args.num_steps,
        speculative_eagle_topk=args.eagle_topk,
        speculative_num_draft_tokens=args.num_draft_tokens,
        tp_size=args.tp_size,
        max_running_requests=args.max_num_seqs,
        dtype=args.dtype,
        disable_cuda_graph=args.disable_cuda_graph,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        log_level=args.log_level,
        watchdog_timeout=1800,
    )
    if (args.quest_page_budget_ratio is not None or args.quest_token_budget is not None) and args.quest_mode == "physical-page":
        engine_kwargs["page_size"] = args.quest_page_size
        print(f"[CONFIG] Setting SGLang page_size={args.quest_page_size} "
              f"to match quest_page_size")
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
        print(f"[CONFIG] Forcing attention backend: {args.attention_backend}")

    if need_patch:
        # Use custom Engine subclass that injects our scheduler wrapper
        from sglang.srt.entrypoints.engine import Engine as SGLangEngine

        class PatchedEngine(SGLangEngine):
            run_scheduler_process_func = staticmethod(_patched_run_scheduler_process)

        engine = PatchedEngine(**engine_kwargs)
    else:
        engine = sgl.Engine(**engine_kwargs)

    sampling_params = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    # Run benchmark
    print(f"\nGenerating up to {args.max_tokens} tokens per prompt...")
    start_time = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    # Extract per-request metrics
    total_output_tokens = 0
    total_verify_ct = 0
    total_accepted_tokens = 0
    has_spec_metrics = False

    for i, output in enumerate(outputs):
        completion_tokens = output["meta_info"]["completion_tokens"]
        total_output_tokens += completion_tokens

        if "spec_verify_ct" in output["meta_info"]:
            has_spec_metrics = True
            verify_ct = output["meta_info"]["spec_verify_ct"]
            total_verify_ct += verify_ct

        if "spec_accepted_tokens" in output["meta_info"]:
            total_accepted_tokens += output["meta_info"]["spec_accepted_tokens"]

    throughput = total_output_tokens / total_time

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (SGLang EAGLE3 Dynamic Tree)")
    print("=" * 70)
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"EAGLE3: num_steps={args.num_steps}, topk={args.eagle_topk}, "
          f"num_draft_tokens={args.num_draft_tokens}")
    print(f"Num prompts: {len(prompts)}")
    print("-" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")

    if has_spec_metrics:
        avg_acceptance_length = total_output_tokens / total_verify_ct if total_verify_ct > 0 else 0
        time_per_iteration_ms = (total_time / total_verify_ct) * 1000 if total_verify_ct > 0 else 0
        time_per_token_ms = (total_time / total_output_tokens) * 1000 if total_output_tokens > 0 else 0

        print("-" * 70)
        print("SPECULATION METRICS:")
        print(f"  Total verify rounds: {total_verify_ct}")
        print(f"  Total accepted tokens: {total_accepted_tokens}")
        print(f"  Avg acceptance length: {avg_acceptance_length:.2f}")
        print("-" * 70)
        print("SPEEDUP METRICS:")
        print(f"  Num iterations: {total_verify_ct}")
        print(f"  Time per iteration: {time_per_iteration_ms:.4f} ms")
        print(f"  Tokens per iteration (avg acceptance): {avg_acceptance_length:.2f}")
        print(f"  Time per token: {time_per_token_ms:.4f} ms")

    # Read and display attention profiling results
    profile_metrics = None
    if args.profile_attention and profile_file:
        # Shutdown engine first so subprocess flushes atexit handlers
        engine.shutdown()
        profile_metrics = read_profile_results(profile_file, total_time, total_verify_ct)
        # Clean up profile file
        try:
            os.unlink(profile_file)
        except OSError:
            pass
        engine = None  # Mark as already shut down

    print("=" * 70)

    # Try to get server-level aggregate metrics
    if engine is not None:
        try:
            server_info = engine.get_server_info()
            internal_states = server_info.get("internal_states", [{}])
            if internal_states:
                avg_spec = internal_states[0].get("avg_spec_accept_length")
                if avg_spec is not None:
                    print(f"\nServer avg_spec_accept_length: {avg_spec:.2f}")
        except Exception:
            pass

    # Print sample outputs
    print("\n--- Sample outputs ---")
    for i in range(min(3, len(outputs))):
        text = outputs[i]["text"][:200]
        print(f"\nPrompt {i}: {prompts[i][:80]}...")
        print(f"Output: {text}...")

    # JSON output
    json_metrics = {
        "framework": "sglang",
        "speculative_algorithm": "EAGLE3",
        "num_steps": args.num_steps,
        "eagle_topk": args.eagle_topk,
        "num_draft_tokens": args.num_draft_tokens,
        "total_time_s": total_time,
        "total_tokens": total_output_tokens,
        "throughput_tokens_per_s": throughput,
        "num_prompts": len(prompts),
    }

    if has_spec_metrics:
        json_metrics["total_verify_rounds"] = total_verify_ct
        json_metrics["total_accepted_tokens"] = total_accepted_tokens
        json_metrics["avg_acceptance_length"] = avg_acceptance_length
        json_metrics["time_per_iteration_ms"] = time_per_iteration_ms
        json_metrics["time_per_token_ms"] = time_per_token_ms

    if profile_metrics:
        json_metrics["profile"] = profile_metrics

    if args.use_tile_bitmap:
        json_metrics["attention_kernel"] = "tile_bitmap"
        json_metrics["tile_block_n"] = args.tile_block_n
        if args.quest_page_budget_ratio is not None or args.quest_token_budget is not None:
            if args.quest_token_budget is not None:
                json_metrics["quest_token_budget"] = args.quest_token_budget
            if args.quest_page_budget_ratio is not None:
                json_metrics["quest_page_budget_ratio"] = args.quest_page_budget_ratio
            json_metrics["quest_page_size"] = args.quest_page_size
            json_metrics["quest_skip_layers"] = args.quest_skip_layers
            json_metrics["quest_mode"] = args.quest_mode
            json_metrics["quest_stage1_impl"] = args.quest_stage1_impl
    elif args.use_paged_bitmap_cuda:
        json_metrics["attention_kernel"] = "paged_bitmap_cuda"
    elif args.use_paged_bitmap:
        json_metrics["attention_kernel"] = "paged_bitmap"
    elif args.use_bitmap_kernel:
        json_metrics["attention_kernel"] = "bitmap_cuda"
    elif args.use_paged_sparse:
        json_metrics["attention_kernel"] = "paged_sparse"
        json_metrics["kv_top_k_ratio"] = args.kv_top_k_ratio
        json_metrics["paged_sparse_max_past"] = args.paged_sparse_max_past
    elif args.use_unified_sparse:
        json_metrics["attention_kernel"] = "unified_sparse"
        json_metrics["kv_top_k_ratio"] = args.kv_top_k_ratio
        json_metrics["unified_sparse_max_past"] = args.unified_sparse_max_past
    elif args.use_subtree_kernel:
        json_metrics["attention_kernel"] = "subtree"

    print(f"\nJSON metrics: {json.dumps(json_metrics)}")

    # Save output log (plain text, same format as stdout)
    if args.output_log is not None:
        log_path = args.output_log
        if log_path == "auto":
            parts = ["sglang"]
            if args.use_tile_bitmap:
                parts.append(f"tile_bn{args.tile_block_n}")
            if args.quest_token_budget is not None:
                parts.append(f"quest_tb{args.quest_token_budget}")
            elif args.quest_page_budget_ratio is not None:
                parts.append(f"quest_r{args.quest_page_budget_ratio}")
            if args.quest_skip_layers > 0:
                parts.append(f"skip{args.quest_skip_layers}")
            if args.quest_page_budget_ratio is not None or args.quest_token_budget is not None:
                parts.append(f"ps{args.quest_page_size}")
                parts.append(args.quest_mode.replace("-", ""))
                if args.quest_stage1_impl != "block_list":
                    parts.append(args.quest_stage1_impl)
            parts.append(f"ndt{args.num_draft_tokens}")
            parts.append(f"{len(prompts)}p")
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "_".join(parts) + ".log")

        with open(log_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("BENCHMARK RESULTS (SGLang EAGLE3 Dynamic Tree)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Target model: {args.target_model}\n")
            f.write(f"Draft model: {args.draft_model}\n")
            f.write(f"EAGLE3: num_steps={args.num_steps}, topk={args.eagle_topk}, "
                    f"num_draft_tokens={args.num_draft_tokens}\n")
            f.write(f"Num prompts: {len(prompts)}\n")
            if args.use_tile_bitmap:
                f.write(f"Attention: tile_bitmap (BLOCK_N={args.tile_block_n})\n")
                if args.quest_token_budget is not None:
                    f.write(f"Quest: token_budget={args.quest_token_budget}, "
                            f"page_size={args.quest_page_size}, skip_layers={args.quest_skip_layers}, "
                            f"mode={args.quest_mode}\n")
                elif args.quest_page_budget_ratio is not None:
                    f.write(f"Quest: ratio={args.quest_page_budget_ratio}, "
                            f"page_size={args.quest_page_size}, skip_layers={args.quest_skip_layers}, "
                            f"mode={args.quest_mode}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Total output tokens: {total_output_tokens}\n")
            f.write(f"Throughput: {throughput:.2f} tokens/s\n")
            if has_spec_metrics:
                f.write("-" * 70 + "\n")
                f.write(f"Avg acceptance length: {avg_acceptance_length:.2f}\n")
                f.write(f"Time per iteration: {time_per_iteration_ms:.4f} ms\n")
                f.write(f"Time per token: {time_per_token_ms:.4f} ms\n")
            f.write("=" * 70 + "\n")
            f.write("\n--- Sample outputs ---\n")
            for i in range(len(outputs)):
                f.write(f"\n[Prompt {i}]: {prompts[i][:200]}...\n")
                f.write(f"[Output {i}]: {outputs[i]['text']}\n")
            f.write(f"\nJSON metrics: {json.dumps(json_metrics)}\n")
        print(f"\nOutput log saved to: {log_path}")

    if engine is not None:
        engine.shutdown()


if __name__ == "__main__":
    main()
