"""
Benchmark EAGLE3 speculative decoding with SGLang (dynamic tree).

SGLang implements EAGLE3's dynamic tree building at runtime using
confidence-based reranking, unlike vLLM which only supports static trees.

Usage:
    cd scripts
    conda activate vllm-spec

    # Baseline EAGLE3 (FlashInfer attention):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_sglang_eagle3.py \
        --target-model meta-llama/Llama-3.3-70B-Instruct \
        --draft-model /path/to/EAGLE3-LLaMA3.1-Instruct-70B \
        --tp-size 4 \
        --num-steps 5 \
        --eagle-topk 8 \
        --num-draft-tokens 64 \
        --dataset /path/to/gsm8k \
        --num-prompts 50

    # Profile attention time during EAGLE3 verification:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_sglang_eagle3.py \
        --target-model meta-llama/Llama-3.3-70B-Instruct \
        --draft-model /path/to/EAGLE3-LLaMA3.1-Instruct-70B \
        --tp-size 4 --num-draft-tokens 64 --profile-attention \
        --dataset /path/to/gsm8k --num-prompts 50
"""

import argparse
import json
import os
import sys
import tempfile
import time

import sglang as sgl
from datasets import detect_and_load

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

def _apply_profiling_patch(profile_file, tp_rank=0):
    """Monkey-patch model components with CUDA event timing.

    Called inside the scheduler subprocess. Instruments layer 0 only.
    Uses CUDA events for near-zero overhead GPU timing.
    Periodically syncs and dumps data to file (survives SIGKILL).
    Only TP rank 0 writes the profile file.

    Patches:
      1. LlamaDecoderLayer.__init__ — propagate layer_id to MLP and decoder layer
      2. RadixAttention.forward — core attention kernel timing
      3. LlamaAttention.forward — full attention block (QKV proj + RoPE + attn + O proj)
      4. LlamaMLP.forward — full MLP block (gate_up + SiLU + down + allreduce)
      5. EAGLEWorker.draft — full Phase 1 (draft prep + draft_forward + build_tree)
      6. LlamaDecoderLayer.forward — full decoder layer (layer 0, verify only)
      7. LogitsProcessor.forward — lm_head + logits (verify only)
      8. EAGLEWorker.forward_draft_extend_after_decode — Phase 3 (draft extend)
      9. EagleVerifyInput.verify — tree verification logic (CPU-side)
     10. EAGLEWorker.verify — full Phase 2 (target forward + tree verify + overhead)
     11. EAGLEWorker.forward_batch_generation — full iteration (P1+P2+P3+inter-phase)
    """
    import atexit
    import json as _json

    import torch

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

    _orig_radix_forward = RadixAttention.forward
    _orig_llama_attn_forward = LlamaAttention.forward
    _orig_mlp_forward = LlamaMLP.forward
    _orig_decoder_init = LlamaDecoderLayer.__init__
    _orig_decoder_forward = LlamaDecoderLayer.forward

    # Optional imports for new patches
    try:
        from sglang.srt.speculative.eagle_worker import EAGLEWorker
        _orig_draft = EAGLEWorker.draft
        _orig_draft_extend = EAGLEWorker.forward_draft_extend_after_decode
        _orig_eagle_verify = EAGLEWorker.verify
        _orig_fbg = EAGLEWorker.forward_batch_generation
        _has_eagle_worker = True
    except ImportError:
        _has_eagle_worker = False

    try:
        from sglang.srt.layers.logits_processor import LogitsProcessor
        _orig_logits_forward = LogitsProcessor.forward
        _has_logits_processor = True
    except ImportError:
        _has_logits_processor = False

    try:
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        _orig_verify = EagleVerifyInput.verify
        _has_eagle_verify = True
    except ImportError:
        _has_eagle_verify = False

    _state = {
        # RadixAttention (core attention kernel)
        "verify_events": [], "other_events": [],
        "verify_times_ms": [], "other_times_ms": [],
        "verify_q_tokens": [], "verify_context_lens": [], "verify_extend_lens": [],
        "verify_count": 0,
        # LlamaAttention (full attention block)
        "attn_block_verify_events": [], "attn_block_other_events": [],
        "attn_block_verify_times_ms": [], "attn_block_other_times_ms": [],
        # LlamaMLP (full MLP block)
        "mlp_verify_events": [], "mlp_other_events": [],
        "mlp_verify_times_ms": [], "mlp_other_times_ms": [],
        # Shared flag: set by layer-0 attention, read by layer-0 MLP
        "in_verify": False,
        # EAGLEWorker.draft (full Phase 1: prep + draft_forward + build_tree)
        "draft_events": [], "draft_times_ms": [],
        # LlamaDecoderLayer (layer 0, verify only)
        "decoder_layer_verify_events": [], "decoder_layer_verify_times_ms": [],
        # LogitsProcessor (verify only)
        "logits_verify_events": [], "logits_verify_times_ms": [],
        # EAGLEWorker.forward_draft_extend_after_decode (Phase 3)
        "draft_extend_events": [], "draft_extend_times_ms": [],
        # EagleVerifyInput.verify (tree verification logic)
        "tree_verify_events": [], "tree_verify_times_ms": [],
        # EAGLEWorker.verify (full Phase 2)
        "eagle_verify_events": [], "eagle_verify_times_ms": [],
        # EAGLEWorker.forward_batch_generation (full iteration)
        "fbg_events": [], "fbg_times_ms": [],
    }

    _EVENT_KEYS = [
        ("verify_events", "verify_times_ms"),
        ("other_events", "other_times_ms"),
        ("attn_block_verify_events", "attn_block_verify_times_ms"),
        ("attn_block_other_events", "attn_block_other_times_ms"),
        ("mlp_verify_events", "mlp_verify_times_ms"),
        ("mlp_other_events", "mlp_other_times_ms"),
        ("draft_events", "draft_times_ms"),
        ("decoder_layer_verify_events", "decoder_layer_verify_times_ms"),
        ("logits_verify_events", "logits_verify_times_ms"),
        ("draft_extend_events", "draft_extend_times_ms"),
        ("tree_verify_events", "tree_verify_times_ms"),
        ("eagle_verify_events", "eagle_verify_times_ms"),
        ("fbg_events", "fbg_times_ms"),
    ]

    def _flush():
        """Sync GPU, compute elapsed times for all event types, dump to file."""
        has_events = any(_state[k] for k, _ in _EVENT_KEYS)
        if not has_events:
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            return
        # Process all event types
        for key_events, key_times in _EVENT_KEYS:
            for s, e in _state[key_events]:
                try:
                    _state[key_times].append(s.elapsed_time(e))
                except Exception:
                    pass
            _state[key_events].clear()
        if tp_rank == 0:
            try:
                with open(profile_file, "w") as f:
                    _json.dump({
                        "verify_extend_times_ms": _state["verify_times_ms"],
                        "other_extend_times_ms": _state["other_times_ms"],
                        "verify_extend_q_tokens": _state["verify_q_tokens"],
                        "verify_context_lens": _state["verify_context_lens"],
                        "verify_extend_lens": _state["verify_extend_lens"],
                        "attn_block_verify_times_ms": _state["attn_block_verify_times_ms"],
                        "attn_block_other_times_ms": _state["attn_block_other_times_ms"],
                        "mlp_verify_times_ms": _state["mlp_verify_times_ms"],
                        "mlp_other_times_ms": _state["mlp_other_times_ms"],
                        "draft_times_ms": _state["draft_times_ms"],
                        "decoder_layer_verify_times_ms": _state["decoder_layer_verify_times_ms"],
                        "logits_verify_times_ms": _state["logits_verify_times_ms"],
                        "draft_extend_times_ms": _state["draft_extend_times_ms"],
                        "tree_verify_times_ms": _state["tree_verify_times_ms"],
                        "eagle_verify_times_ms": _state["eagle_verify_times_ms"],
                        "fbg_times_ms": _state["fbg_times_ms"],
                    }, f)
            except Exception:
                pass

    # Patch 1: LlamaDecoderLayer.__init__ → propagate layer_id to MLP and decoder layer
    def _patched_decoder_init(self, *args, **kwargs):
        _orig_decoder_init(self, *args, **kwargs)
        try:
            lid = self.self_attn.attn.layer_id
            self.mlp._profile_layer_id = lid
            self._profile_layer_id = lid
        except Exception:
            pass
    LlamaDecoderLayer.__init__ = _patched_decoder_init

    # Patch 2: RadixAttention.forward (core attention kernel, layer 0)
    def _profiled_radix_forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        if self.layer_id != 0:
            return _orig_radix_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)

        is_verify = (hasattr(forward_batch, 'forward_mode') and
                     forward_batch.forward_mode.is_target_verify())
        _state["in_verify"] = is_verify

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _orig_radix_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)
        end.record()

        if is_verify:
            _state["verify_events"].append((start, end))
            _state["verify_q_tokens"].append(q.shape[0])
            ctx = 0
            ext = q.shape[0]
            try:
                seq_lens_cpu = getattr(forward_batch, 'seq_lens_cpu', None)
                if seq_lens_cpu is not None:
                    bs = forward_batch.batch_size
                    ctx = int(seq_lens_cpu[:bs].sum().item())
            except Exception:
                pass
            _state["verify_context_lens"].append(ctx)
            _state["verify_extend_lens"].append(ext)
        else:
            _state["other_events"].append((start, end))
        return result
    RadixAttention.forward = _profiled_radix_forward

    # Patch 3: LlamaAttention.forward (full attention block, layer 0)
    def _profiled_llama_attn_forward(self, positions, hidden_states, forward_batch):
        if self.attn.layer_id != 0:
            return _orig_llama_attn_forward(self, positions, hidden_states, forward_batch)

        is_verify = (hasattr(forward_batch, 'forward_mode') and
                     forward_batch.forward_mode.is_target_verify())

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _orig_llama_attn_forward(self, positions, hidden_states, forward_batch)
        end.record()

        if is_verify:
            _state["attn_block_verify_events"].append((start, end))
        else:
            _state["attn_block_other_events"].append((start, end))
        return result
    LlamaAttention.forward = _profiled_llama_attn_forward

    # Patch 4: LlamaMLP.forward (full MLP block, layer 0)
    def _profiled_mlp_forward(self, x, forward_batch=None, use_reduce_scatter=False):
        layer_id = getattr(self, '_profile_layer_id', -1)
        if layer_id != 0:
            return _orig_mlp_forward(self, x, forward_batch, use_reduce_scatter)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _orig_mlp_forward(self, x, forward_batch, use_reduce_scatter)
        end.record()

        if _state["in_verify"]:
            _state["mlp_verify_events"].append((start, end))
        else:
            _state["mlp_other_events"].append((start, end))
        return result
    LlamaMLP.forward = _profiled_mlp_forward

    # Patch 5: EAGLEWorker.draft — full Phase 1 (prep + draft_forward + build_tree)
    if _has_eagle_worker:
        def _profiled_draft(self, batch):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _orig_draft(self, batch)
            end.record()
            _state["draft_events"].append((start, end))
            return result
        EAGLEWorker.draft = _profiled_draft

    # Patch 6: LlamaDecoderLayer.forward (layer 0, verify mode only)
    def _profiled_decoder_forward(self, positions, hidden_states, forward_batch, residual):
        layer_id = getattr(self, '_profile_layer_id', -1)
        if layer_id != 0:
            return _orig_decoder_forward(self, positions, hidden_states, forward_batch, residual)
        is_verify = (hasattr(forward_batch, 'forward_mode') and
                     forward_batch.forward_mode.is_target_verify())
        if not is_verify:
            return _orig_decoder_forward(self, positions, hidden_states, forward_batch, residual)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _orig_decoder_forward(self, positions, hidden_states, forward_batch, residual)
        end.record()
        _state["decoder_layer_verify_events"].append((start, end))
        # Periodic flush AFTER end.record() so sync CPU time is NOT captured
        # by the decoder layer event. This was previously in MLP patch but
        # that inflated decoder_layer timing (and thus the ×80 extrapolation).
        _state["verify_count"] += 1
        if _state["verify_count"] % 10 == 0:
            _flush()
        return result
    LlamaDecoderLayer.forward = _profiled_decoder_forward

    # Patch 7: LogitsProcessor.forward (verify mode only)
    if _has_logits_processor:
        def _profiled_logits_forward(self, input_ids, hidden_states, lm_head,
                                     logits_metadata, aux_hidden_states=None,
                                     hidden_states_before_norm=None):
            # logits_metadata is a ForwardBatch before conversion inside forward()
            is_verify = False
            try:
                if hasattr(logits_metadata, 'forward_mode'):
                    is_verify = logits_metadata.forward_mode.is_target_verify()
            except Exception:
                pass
            if not is_verify:
                return _orig_logits_forward(self, input_ids, hidden_states, lm_head,
                                            logits_metadata, aux_hidden_states,
                                            hidden_states_before_norm)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _orig_logits_forward(self, input_ids, hidden_states, lm_head,
                                          logits_metadata, aux_hidden_states,
                                          hidden_states_before_norm)
            end.record()
            _state["logits_verify_events"].append((start, end))
            return result
        LogitsProcessor.forward = _profiled_logits_forward

    # Patch 8: EAGLEWorker.forward_draft_extend_after_decode — Phase 3
    if _has_eagle_worker:
        def _profiled_draft_extend(self, batch):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _orig_draft_extend(self, batch)
            end.record()
            _state["draft_extend_events"].append((start, end))
            return result
        EAGLEWorker.forward_draft_extend_after_decode = _profiled_draft_extend

    # Patch 9: EagleVerifyInput.verify — tree verification logic
    if _has_eagle_verify:
        def _profiled_tree_verify(self, batch, logits_output,
                                  token_to_kv_pool_allocator, page_size,
                                  vocab_mask=None):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _orig_verify(self, batch, logits_output,
                                  token_to_kv_pool_allocator, page_size,
                                  vocab_mask)
            end.record()
            _state["tree_verify_events"].append((start, end))
            return result
        EagleVerifyInput.verify = _profiled_tree_verify

    # Patch 10: EAGLEWorker.verify — full Phase 2 (target forward + tree verify + overhead)
    if _has_eagle_worker:
        def _profiled_eagle_verify(self, batch, spec_info):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _orig_eagle_verify(self, batch, spec_info)
            end.record()
            _state["eagle_verify_events"].append((start, end))
            return result
        EAGLEWorker.verify = _profiled_eagle_verify

    # Patch 11: EAGLEWorker.forward_batch_generation — full iteration (decode only)
    if _has_eagle_worker:
        def _profiled_fbg(self, batch):
            is_extend = (batch.forward_mode.is_extend()
                         or getattr(batch, 'is_extend_in_batch', False))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _orig_fbg(self, batch)
            end.record()
            if not is_extend:
                _state["fbg_events"].append((start, end))
            return result
        EAGLEWorker.forward_batch_generation = _profiled_fbg

    atexit.register(_flush)


def _apply_nvtx_patch(sync=False):
    """Add NVTX range markers to EAGLEWorker phases for nsys profiling.

    Works WITH CUDA graphs enabled. Zero overhead when nsys is not attached.

    Top-level markers: P1_Draft, P2_Verify, P3_DraftExtend, SpecDecode_Iter.
    Sub-phase markers inside verify():
        P2a_Prep        — prepare_for_verify + batch setup
        P2b_TargetFwd   — target model forward (CUDA graph replay)
        P2c_VerifySample — spec_info.verify (acceptance check, includes .tolist() sync)

    Args:
        sync: If True, insert torch.cuda.synchronize() before each range_pop()
              so that NVTX durations reflect actual GPU execution time, not just
              CPU-side kernel launch time.
    """
    import torch

    try:
        from sglang.srt.speculative.eagle_worker import EAGLEWorker
    except ImportError:
        print("[NVTX] EAGLEWorker not found, skipping NVTX patch", flush=True)
        return

    _orig_draft = EAGLEWorker.draft
    _orig_verify = EAGLEWorker.verify
    _orig_fbg = EAGLEWorker.forward_batch_generation

    try:
        _orig_draft_extend = EAGLEWorker.forward_draft_extend_after_decode
        _has_draft_extend = True
    except AttributeError:
        _has_draft_extend = False

    def _nvtx_draft(self, batch):
        torch.cuda.nvtx.range_push("P1_Draft")
        result = _orig_draft(self, batch)
        if sync:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return result
    EAGLEWorker.draft = _nvtx_draft

    # ----- Detailed verify with sub-phase markers -----
    def _nvtx_verify(self, batch, spec_info):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.spec_utils import (
            detect_nan,
            generate_token_bitmask,
        )
        from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1

        torch.cuda.nvtx.range_push("P2_Verify")

        # === P2a: Preparation ===
        torch.cuda.nvtx.range_push("P2a_Prep")
        seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        if sync:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()  # P2a_Prep

        # === P2b: Target model forward ===
        torch.cuda.nvtx.range_push("P2b_TargetFwd")
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        if sync:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()  # P2b_TargetFwd

        # === P2c: Verify / sampling ===
        torch.cuda.nvtx.range_push("P2c_VerifySample")

        vocab_mask = None
        if batch.has_grammar:
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )
            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                batch.sampling_info.vocab_mask = None

        if self.enable_nan_detection:
            detect_nan(logits_output)

        spec_info.hidden_states = logits_output.hidden_states
        res = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        if sync:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()  # P2c_VerifySample

        if sync:
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()  # P2_Verify

        return logits_output, res, model_worker_batch, can_run_cuda_graph
    EAGLEWorker.verify = _nvtx_verify

    if _has_draft_extend:
        def _nvtx_draft_extend(self, batch):
            torch.cuda.nvtx.range_push("P3_DraftExtend")
            result = _orig_draft_extend(self, batch)
            if sync:
                torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return result
        EAGLEWorker.forward_draft_extend_after_decode = _nvtx_draft_extend

    def _nvtx_fbg(self, batch):
        is_extend = (batch.forward_mode.is_extend()
                     or getattr(batch, 'is_extend_in_batch', False))
        if not is_extend:
            torch.cuda.nvtx.range_push("SpecDecode_Iter")
        result = _orig_fbg(self, batch)
        if not is_extend:
            if sync:
                torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        return result
    EAGLEWorker.forward_batch_generation = _nvtx_fbg

    mode = "sync" if sync else "async"
    print(f"[NVTX] NVTX range markers applied to EAGLEWorker (mode={mode})", flush=True)


# ============================================================================
# Scheduler subprocess wrapper (must be module-level for pickling)
# ============================================================================

def _patched_run_scheduler_process(server_args, port_args, gpu_id, tp_rank,
                                   moe_ep_rank, pp_rank, dp_rank, pipe_writer):
    """Scheduler process wrapper that applies monkey-patches before starting.

    Reads configuration from environment variables:
        SGLANG_BENCH_SCRIPTS_DIR: path to scripts/ for imports
        SGLANG_BENCH_FLAGS: comma-separated flags (profile, nvtx)
        SGLANG_ATTN_PROFILE_FILE: path for profiling JSON output
    """
    scripts_dir = os.environ.get("SGLANG_BENCH_SCRIPTS_DIR", "")
    if scripts_dir and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    flags = os.environ.get("SGLANG_BENCH_FLAGS", "")
    profile_file = os.environ.get("SGLANG_ATTN_PROFILE_FILE", "")

    if "profile" in flags and profile_file:
        _apply_profiling_patch(profile_file, tp_rank=tp_rank)
        print(f"[PROFILE] CUDA event profiling enabled in TP rank {tp_rank}", flush=True)

    if "nvtx" in flags and tp_rank == 0:
        use_sync = "nvtx_sync" in flags
        _apply_nvtx_patch(sync=use_sync)

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
    parser.add_argument("--speculative-algorithm", type=str, default="STANDALONE",
                        choices=["STANDALONE", "EAGLE3"],
                        help="Speculative decoding algorithm (default: STANDALONE). "
                             "STANDALONE uses the draft model as a linear chain (topk=1). "
                             "EAGLE3 uses dynamic tree with EAGLE3-specific draft model.")
    parser.add_argument("--tp-size", type=int, default=1)
    # STANDALONE defaults: num_steps controls draft tokens per round, topk=1, num_draft_tokens=num_steps+1
    # EAGLE3 defaults: num_steps=5, topk=8, num_draft_tokens=63
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Draft steps per round (default: auto). "
                             "STANDALONE: number of draft tokens. EAGLE3: tree depth.")
    parser.add_argument("--eagle-topk", type=int, default=None,
                        help="Branching factor per step (default: auto). "
                             "EAGLE3 default: 8, STANDALONE default: 1 (use >1 for tree drafting).")
    parser.add_argument("--num-draft-tokens", type=int, default=None,
                        help="Max parallel verification tokens (default: auto). "
                             "STANDALONE: auto = num_steps+1. EAGLE3: tree width.")
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
    parser.add_argument("--profile-attention", action="store_true",
                        help="Profile attention kernel time during tree verification. "
                             "Instruments RadixAttention.forward (layer 0) with CUDA event timing. "
                             "Requires --disable-cuda-graph.")
    parser.add_argument("--profile-nvtx", action="store_true",
                        help="Add NVTX range markers (P1_Draft, P2_Verify, P3_DraftExtend) "
                             "for nsys profiling. Works WITH CUDA graphs. "
                             "Run with: nsys profile -t cuda,nvtx python benchmark_sglang_eagle3.py ...")
    parser.add_argument("--profile-nvtx-sync", action="store_true",
                        help="Like --profile-nvtx but adds torch.cuda.synchronize() before "
                             "each range_pop(). This makes NVTX durations reflect true GPU time "
                             "instead of CPU-side launch time. Adds ~2-5%% overhead but fixes "
                             "the async CUDA graph replay timing distortion.")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Max prompt tokens for dataset filtering (longbench only)")
    parser.add_argument("--attention-backend", type=str, default=None,
                        choices=["triton", "flashinfer"],
                        help="Force attention backend (default: auto-detect)")
    parser.add_argument("--fly-enabled", action="store_true",
                        help="Enable FLy (Training-Free Loosely Speculative Decoding) "
                             "for greedy verification")
    parser.add_argument("--fly-entropy-threshold", type=float, default=0.3,
                        help="Normalized entropy threshold for FLy deferral (default: 0.3)")
    parser.add_argument("--fly-window-size", type=int, default=6,
                        help="Lookahead window size for FLy verification (default: 6)")
    parser.add_argument("--fly-use-cuda-kernel", action="store_true",
                        help="Use CUDA kernel for FLy verification (requires recompiled sgl-kernel)")
    return parser.parse_args()


def read_profile_results(profile_file, total_time, total_verify_ct, tp_size=4):
    """Read and display attention + MLP profiling results from subprocess JSON.

    Computes latency, FLOPs, bandwidth, and memory metrics.
    Model config: Llama-3.1-70B (num_heads=64, num_kv_heads=8, head_dim=128, 80 layers).
    """
    # Llama-3.1-70B model config
    NUM_LAYERS = 80
    NUM_HEADS = 64       # total query heads
    NUM_KV_HEADS = 8     # total KV heads (GQA)
    HEAD_DIM = 128
    HIDDEN_SIZE = 8192
    INTERMEDIATE_SIZE = 28672
    DTYPE_BYTES = 2      # fp16

    # Per-GPU (TP-split)
    heads_per_gpu = NUM_HEADS // tp_size          # 16
    kv_heads_per_gpu = NUM_KV_HEADS // tp_size    # 2
    intermediate_per_gpu = INTERMEDIATE_SIZE // tp_size  # 7168

    for _ in range(30):
        if os.path.exists(profile_file):
            time.sleep(0.5)
            break
        time.sleep(0.2)

    if not os.path.exists(profile_file):
        print(f"[PROFILE] Warning: profile file not found at {profile_file}")
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
    verify_context_lens = data.get("verify_context_lens", [])
    verify_extend_lens = data.get("verify_extend_lens", [])
    attn_block_verify_times = data.get("attn_block_verify_times_ms", [])
    attn_block_other_times = data.get("attn_block_other_times_ms", [])
    mlp_verify_times = data.get("mlp_verify_times_ms", [])
    mlp_other_times = data.get("mlp_other_times_ms", [])
    draft_times = data.get("draft_times_ms", [])
    decoder_layer_verify_times = data.get("decoder_layer_verify_times_ms", [])
    logits_verify_times = data.get("logits_verify_times_ms", [])
    draft_extend_times = data.get("draft_extend_times_ms", [])
    tree_verify_times = data.get("tree_verify_times_ms", [])
    eagle_verify_times = data.get("eagle_verify_times_ms", [])
    fbg_times = data.get("fbg_times_ms", [])

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

    total_time_ms = total_time * 1000
    iter_time_ms = (total_time / total_verify_ct) * 1000 if total_verify_ct > 0 else 0

    attn_per_iteration_ms = total_verify_attn_ms / total_verify_ct if total_verify_ct > 0 else 0
    attn_pct_of_iteration = (attn_per_iteration_ms / iter_time_ms) * 100 if iter_time_ms > 0 else 0

    print("\n" + "-" * 70)
    print("PROFILING RESULTS (layer 0, per GPU, TP=%d):" % tp_size)
    print(f"  Model: Llama-3.1-70B  heads/gpu={heads_per_gpu}  "
          f"kv_heads/gpu={kv_heads_per_gpu}  head_dim={HEAD_DIM}")
    print(f"  Iteration time:         {iter_time_ms:.4f} ms")
    print(f"  Avg Q tokens per call:  {avg_q_tokens:.0f}")

    # ── Core attention kernel ──
    print(f"\n  CORE ATTENTION KERNEL (RadixAttention, layer 0):")
    print(f"    Verify calls: {len(verify_times)}  |  Other calls: {len(other_times)}")
    print(f"    Avg verify time:   {avg_verify_attn_ms:.4f} ms")
    print(f"    Per iteration:     {attn_per_iteration_ms:.4f} ms")
    print(f"    % of iteration:    {attn_pct_of_iteration:.3f}%")

    result = {
        "verify_attn_calls": len(verify_times),
        "avg_verify_attn_ms": avg_verify_attn_ms,
        "total_verify_attn_ms": total_verify_attn_ms,
        "avg_q_tokens": avg_q_tokens,
        "attn_per_iteration_ms": attn_per_iteration_ms,
        "attn_pct_of_iteration": attn_pct_of_iteration,
    }

    # ── Full attention block ──
    if attn_block_verify_times:
        total_attn_block_ms = sum(attn_block_verify_times)
        avg_attn_block_ms = total_attn_block_ms / len(attn_block_verify_times)
        attn_block_per_iter = total_attn_block_ms / total_verify_ct if total_verify_ct > 0 else 0
        attn_block_pct = (attn_block_per_iter / iter_time_ms) * 100 if iter_time_ms > 0 else 0
        proj_overhead_ms = avg_attn_block_ms - avg_verify_attn_ms

        print(f"\n  FULL ATTENTION BLOCK (QKV proj + RoPE + attn + O proj, layer 0):")
        print(f"    Verify calls: {len(attn_block_verify_times)}")
        print(f"    Avg verify time:   {avg_attn_block_ms:.4f} ms")
        print(f"    Per iteration:     {attn_block_per_iter:.4f} ms")
        print(f"    % of iteration:    {attn_block_pct:.3f}%")
        print(f"    Projection overhead: {proj_overhead_ms:.4f} ms  "
              f"(QKV+O proj, {proj_overhead_ms/avg_attn_block_ms*100:.1f}% of block)"
              if avg_attn_block_ms > 0 else "")

        result.update({
            "avg_attn_block_ms": avg_attn_block_ms,
            "attn_block_per_iter": attn_block_per_iter,
            "attn_block_pct_of_iteration": attn_block_pct,
            "attn_proj_overhead_ms": proj_overhead_ms,
        })

    # ── MLP/FFN block ──
    if mlp_verify_times:
        total_mlp_ms = sum(mlp_verify_times)
        avg_mlp_ms = total_mlp_ms / len(mlp_verify_times)
        mlp_per_iter = total_mlp_ms / total_verify_ct if total_verify_ct > 0 else 0
        mlp_pct = (mlp_per_iter / iter_time_ms) * 100 if iter_time_ms > 0 else 0

        print(f"\n  MLP/FFN BLOCK (gate_up + SiLU + down + allreduce, layer 0):")
        print(f"    Verify calls: {len(mlp_verify_times)}")
        print(f"    Avg verify time:   {avg_mlp_ms:.4f} ms")
        print(f"    Per iteration:     {mlp_per_iter:.4f} ms")
        print(f"    % of iteration:    {mlp_pct:.3f}%")

        # MLP FLOPs per call per GPU (using avg Q tokens)
        # gate_up: 2 × num_tokens × hidden × (2 × intermediate/tp)
        # down:    2 × num_tokens × (intermediate/tp) × hidden
        avg_q = avg_q_tokens
        gate_up_flops = 2 * avg_q * HIDDEN_SIZE * (2 * intermediate_per_gpu)
        down_flops = 2 * avg_q * intermediate_per_gpu * HIDDEN_SIZE
        mlp_flops_per_call = gate_up_flops + down_flops

        # MLP bytes per call per GPU
        # Weights: gate_up (hidden × 2*inter/tp) + down (inter/tp × hidden) × dtype
        gate_up_weight_bytes = HIDDEN_SIZE * 2 * intermediate_per_gpu * DTYPE_BYTES
        down_weight_bytes = intermediate_per_gpu * HIDDEN_SIZE * DTYPE_BYTES
        # Activations: input + gate_up_out + silu_out + output
        act_bytes = avg_q * (HIDDEN_SIZE + 2 * intermediate_per_gpu +
                             intermediate_per_gpu + HIDDEN_SIZE) * DTYPE_BYTES
        mlp_bytes_per_call = gate_up_weight_bytes + down_weight_bytes + act_bytes

        mlp_arith_intensity = mlp_flops_per_call / mlp_bytes_per_call if mlp_bytes_per_call > 0 else 0
        mlp_achieved_tflops = (mlp_flops_per_call * len(mlp_verify_times) / 1e12) / (total_mlp_ms / 1000) if total_mlp_ms > 0 else 0
        mlp_achieved_bw = (mlp_bytes_per_call * len(mlp_verify_times) / 1e9) / (total_mlp_ms / 1000) if total_mlp_ms > 0 else 0

        print(f"    FLOPs per call:    {mlp_flops_per_call:.2e}")
        print(f"    Bytes per call:    {mlp_bytes_per_call / 1024 / 1024:.1f} MB  "
              f"(weights: {(gate_up_weight_bytes + down_weight_bytes) / 1024 / 1024:.1f} MB)")
        print(f"    Arith intensity:   {mlp_arith_intensity:.1f} FLOPs/byte")
        print(f"    Achieved compute:  {mlp_achieved_tflops:.3f} TFLOPS")
        print(f"    Achieved bandwidth:{mlp_achieved_bw:.1f} GB/s")
        if mlp_arith_intensity < 150:
            print(f"    >> Memory-bound (intensity < ~150 for A100)")
        else:
            print(f"    >> Compute-bound (intensity >= ~150 for A100)")

        result.update({
            "avg_mlp_ms": avg_mlp_ms,
            "mlp_per_iter": mlp_per_iter,
            "mlp_pct_of_iteration": mlp_pct,
            "mlp_flops_per_call": mlp_flops_per_call,
            "mlp_bytes_per_call": mlp_bytes_per_call,
            "mlp_arith_intensity": mlp_arith_intensity,
            "mlp_achieved_tflops": mlp_achieved_tflops,
        })

    # ── Context vs Tree/Chain breakdown (attention) ──
    if verify_context_lens and verify_extend_lens and verify_q_tokens:
        avg_context = sum(verify_context_lens) / len(verify_context_lens)
        avg_extend = sum(verify_extend_lens) / len(verify_extend_lens)

        total_context_flop_units = sum(e * c for c, e in zip(verify_context_lens, verify_extend_lens))
        total_tree_flop_units = sum(e * e for e in verify_extend_lens)
        total_flop_units = total_context_flop_units + total_tree_flop_units
        context_pct = (total_context_flop_units / total_flop_units * 100) if total_flop_units > 0 else 0
        tree_pct = (total_tree_flop_units / total_flop_units * 100) if total_flop_units > 0 else 0

        print(f"\n  CONTEXT vs TREE/CHAIN BREAKDOWN:")
        print(f"    Avg context (prefix) len: {avg_context:.0f} tokens")
        print(f"    Avg extend (draft) len:   {avg_extend:.0f} tokens")
        print(f"    Avg total KV per call:    {avg_context + avg_extend:.0f} tokens")
        print(f"    Context region share:     {context_pct:.1f}%  (Q × context)")
        print(f"    Tree/chain region share:  {tree_pct:.1f}%  (Q × draft)")

        result.update({
            "avg_context_len": avg_context,
            "avg_extend_len": avg_extend,
            "context_flop_pct": context_pct,
            "tree_flop_pct": tree_pct,
        })

    # ── Full model estimate (× 80 layers) + combined breakdown ──
    print(f"\n  FULL MODEL ESTIMATE (× {NUM_LAYERS} layers):")
    full_attn_kernel_ms = total_verify_attn_ms * NUM_LAYERS
    full_attn_kernel_pct = (full_attn_kernel_ms / total_time_ms * 100) if total_time_ms > 0 else 0
    print(f"    Core attention:   {full_attn_kernel_ms:.1f} ms  ({full_attn_kernel_pct:.1f}% of total)")

    if attn_block_verify_times:
        full_attn_block_ms = sum(attn_block_verify_times) * NUM_LAYERS
        full_attn_block_pct = (full_attn_block_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        print(f"    Attn block:       {full_attn_block_ms:.1f} ms  ({full_attn_block_pct:.1f}% of total)")
        result["full_attn_block_ms"] = full_attn_block_ms
    else:
        full_attn_block_ms = full_attn_kernel_ms
        full_attn_block_pct = full_attn_kernel_pct

    if mlp_verify_times:
        full_mlp_ms = sum(mlp_verify_times) * NUM_LAYERS
        full_mlp_pct = (full_mlp_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        print(f"    MLP/FFN:          {full_mlp_ms:.1f} ms  ({full_mlp_pct:.1f}% of total)")
        result["full_mlp_ms"] = full_mlp_ms
    else:
        full_mlp_ms = 0
        full_mlp_pct = 0

    model_forward_ms = full_attn_block_ms + full_mlp_ms
    model_forward_pct = (model_forward_ms / total_time_ms * 100) if total_time_ms > 0 else 0
    other_ms = total_time_ms - model_forward_ms
    other_pct = 100 - model_forward_pct

    print(f"    ─────────────────────────────")
    print(f"    Model forward:    {model_forward_ms:.1f} ms  ({model_forward_pct:.1f}% of total)")
    print(f"    Other (sched+draft+verify+layernorm): {other_ms:.1f} ms  ({other_pct:.1f}%)")

    # Per-iteration breakdown
    if total_verify_ct > 0:
        attn_block_iter = full_attn_block_ms / total_verify_ct
        mlp_iter = full_mlp_ms / total_verify_ct
        model_iter = model_forward_ms / total_verify_ct
        other_iter = other_ms / total_verify_ct

        print(f"\n  PER-ITERATION BREAKDOWN (full model, {iter_time_ms:.1f} ms/iter):")
        print(f"    Attn block:  {attn_block_iter:.2f} ms  ({attn_block_iter/iter_time_ms*100:.1f}%)")
        print(f"    MLP/FFN:     {mlp_iter:.2f} ms  ({mlp_iter/iter_time_ms*100:.1f}%)")
        print(f"    Other:       {other_iter:.2f} ms  ({other_iter/iter_time_ms*100:.1f}%)")

        result.update({
            "attn_block_per_iter_full": attn_block_iter,
            "mlp_per_iter_full": mlp_iter,
            "other_per_iter_full": other_iter,
        })

    # ── FULL DECOMPOSITION (per GPU forward pass) ──
    # Use actual GPU forward pass count (from event counts) instead of
    # total_verify_ct (which sums per-request rounds across batched requests).
    num_gpu_passes = max(
        len(draft_times) if draft_times else 0,
        len(decoder_layer_verify_times) if decoder_layer_verify_times else 0,
        len(logits_verify_times) if logits_verify_times else 0,
        1,
    )
    has_decomposition = draft_times or decoder_layer_verify_times or logits_verify_times
    if has_decomposition and num_gpu_passes > 0:
        gpu_pass_ms = (total_time_ms / num_gpu_passes) if num_gpu_passes > 0 else 0
        avg_batch = total_verify_ct / num_gpu_passes if num_gpu_passes > 0 else 1

        print(f"\n  ════════════════════════════════════════")
        print(f"  FULL DECOMPOSITION (per GPU forward pass)")
        print(f"    GPU passes: {num_gpu_passes}  |  Avg batch: {avg_batch:.1f}  |  "
              f"Pass time: {gpu_pass_ms:.1f} ms")

        # Phase 1: Draft (prep + draft_forward + build_tree)
        draft_per_pass = 0
        if draft_times:
            total_draft_ms = sum(draft_times)
            avg_draft_ms = total_draft_ms / len(draft_times)
            draft_per_pass = avg_draft_ms
            draft_pct = (draft_per_pass / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            print(f"\n    PHASE 1 — DRAFT (prep + draft_forward + build_tree):")
            print(f"      Calls: {len(draft_times)}")
            print(f"      Avg time:        {avg_draft_ms:.2f} ms  ({draft_pct:.1f}%)")
            result["draft_per_pass"] = draft_per_pass
            result["draft_pct"] = draft_pct

        # Phase 2: Target verify — broken down into sub-components
        print(f"\n    PHASE 2 — TARGET VERIFY:")

        # Attn block × 80 layers
        attn_per_pass = 0
        core_attn_per_pass = 0
        if attn_block_verify_times:
            avg_attn_l0 = sum(attn_block_verify_times) / len(attn_block_verify_times)
            attn_per_pass = avg_attn_l0 * NUM_LAYERS
            attn_pct = (attn_per_pass / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            print(f"      ATTN BLOCK (layer 0 × {NUM_LAYERS}):")
            print(f"        Avg layer 0:     {avg_attn_l0:.4f} ms")
            print(f"        Full model:      {attn_per_pass:.2f} ms  ({attn_pct:.1f}%)")
            # Core attention kernel sub-breakdown
            if verify_times:
                avg_core_l0 = sum(verify_times) / len(verify_times)
                core_attn_per_pass = avg_core_l0 * NUM_LAYERS
                proj_per_pass = attn_per_pass - core_attn_per_pass
                print(f"        ├─ Core attn:    {avg_core_l0:.4f} ms/layer  →  {core_attn_per_pass:.2f} ms  "
                      f"({core_attn_per_pass/gpu_pass_ms*100:.1f}%)")
                print(f"        └─ QKV+O proj:   {avg_attn_l0 - avg_core_l0:.4f} ms/layer  →  {proj_per_pass:.2f} ms  "
                      f"({proj_per_pass/gpu_pass_ms*100:.1f}%)")

        # MLP × 80 layers
        mlp_per_pass = 0
        if mlp_verify_times:
            avg_mlp_l0 = sum(mlp_verify_times) / len(mlp_verify_times)
            mlp_per_pass = avg_mlp_l0 * NUM_LAYERS
            mlp_pct = (mlp_per_pass / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            print(f"      MLP/FFN (layer 0 × {NUM_LAYERS}):")
            print(f"        Avg layer 0:     {avg_mlp_l0:.4f} ms")
            print(f"        Full model:      {mlp_per_pass:.2f} ms  ({mlp_pct:.1f}%)")

        # RMSNorm (derived from decoder layer - attn_block - mlp)
        norm_per_pass = 0
        if decoder_layer_verify_times and attn_block_verify_times and mlp_verify_times:
            avg_decoder_l0 = sum(decoder_layer_verify_times) / len(decoder_layer_verify_times)
            avg_attn_l0_ = sum(attn_block_verify_times) / len(attn_block_verify_times)
            avg_mlp_l0_ = sum(mlp_verify_times) / len(mlp_verify_times)
            norm_per_layer = avg_decoder_l0 - avg_attn_l0_ - avg_mlp_l0_
            norm_per_pass = norm_per_layer * NUM_LAYERS
            norm_pct = (norm_per_pass / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            print(f"      RMSNORM (decoder_layer - attn - mlp, × {NUM_LAYERS}):")
            print(f"        Decoder layer 0: {avg_decoder_l0:.4f} ms")
            print(f"        Norm per layer:  {norm_per_layer:.4f} ms")
            print(f"        Full model:      {norm_per_pass:.2f} ms  ({norm_pct:.1f}%)")
            result["norm_per_pass"] = norm_per_pass
            result["norm_pct"] = norm_pct

        # lm_head / LogitsProcessor
        logits_per_pass = 0
        if logits_verify_times:
            avg_logits_ms = sum(logits_verify_times) / len(logits_verify_times)
            logits_per_pass = avg_logits_ms
            logits_pct = (logits_per_pass / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            avg_q = avg_q_tokens if avg_q_tokens > 0 else 1
            lm_head_flops = 2 * avg_q * HIDDEN_SIZE * 128256 // tp_size
            lm_head_weight_bytes = HIDDEN_SIZE * 128256 // tp_size * DTYPE_BYTES
            lm_head_intensity = lm_head_flops / lm_head_weight_bytes if lm_head_weight_bytes > 0 else 0
            print(f"      LM_HEAD / LOGITS PROCESSOR:")
            print(f"        Calls: {len(logits_verify_times)}")
            print(f"        Avg time:        {avg_logits_ms:.4f} ms  ({logits_pct:.1f}%)")
            print(f"        Arith intensity: {lm_head_intensity:.1f} FLOPs/byte (Q={avg_q:.0f} tokens)")
            result["logits_per_pass"] = logits_per_pass
            result["logits_pct"] = logits_pct

        # Tree verify logic (spec_info.verify)
        tree_verify_per_pass = 0
        if tree_verify_times:
            avg_tree_verify_ms = sum(tree_verify_times) / len(tree_verify_times)
            tree_verify_per_pass = avg_tree_verify_ms
            tree_verify_pct = (tree_verify_per_pass / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            print(f"      TREE VERIFY (spec_info.verify):")
            print(f"        Calls: {len(tree_verify_times)}")
            print(f"        Avg time:        {avg_tree_verify_ms:.2f} ms  ({tree_verify_pct:.1f}%)")
            result["tree_verify_per_pass"] = tree_verify_per_pass
            result["tree_verify_pct"] = tree_verify_pct

        # Phase 3: Draft extend after decode
        draft_extend_per_pass = 0
        if draft_extend_times:
            avg_draft_extend_ms = sum(draft_extend_times) / len(draft_extend_times)
            draft_extend_per_pass = avg_draft_extend_ms
            draft_extend_pct = (draft_extend_per_pass / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            print(f"\n    PHASE 3 — DRAFT EXTEND (forward_draft_extend_after_decode):")
            print(f"      Calls: {len(draft_extend_times)}")
            print(f"      Avg time:        {avg_draft_extend_ms:.2f} ms  ({draft_extend_pct:.1f}%)")
            result["draft_extend_per_pass"] = draft_extend_per_pass
            result["draft_extend_pct"] = draft_extend_pct

        # Phase 2 total and internal residual
        p2_total_per_pass = 0
        p2_internal_residual = 0
        if eagle_verify_times:
            avg_eagle_verify_ms = sum(eagle_verify_times) / len(eagle_verify_times)
            p2_total_per_pass = avg_eagle_verify_ms
            p2_total_pct = (p2_total_per_pass / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            p2_sub_accounted = attn_per_pass + mlp_per_pass + norm_per_pass + logits_per_pass + tree_verify_per_pass
            p2_internal_residual = p2_total_per_pass - p2_sub_accounted
            p2_ir_pct = (p2_internal_residual / gpu_pass_ms * 100) if gpu_pass_ms > 0 else 0
            print(f"\n      PHASE 2 TOTAL (EAGLEWorker.verify):")
            print(f"        Calls: {len(eagle_verify_times)}")
            print(f"        Avg time:        {avg_eagle_verify_ms:.2f} ms  ({p2_total_pct:.1f}%)")
            print(f"        Sub-accounted:   {p2_sub_accounted:.2f} ms")
            print(f"        Internal resid:  {p2_internal_residual:.2f} ms  ({p2_ir_pct:.1f}%)  "
                  f"= embed + final_norm + prepare_for_verify + batch prep + post-process")
            result["p2_total_per_pass"] = p2_total_per_pass
            result["p2_internal_residual"] = p2_internal_residual

        # forward_batch_generation total, inter-phase, prefill/scheduler
        fbg_per_pass = 0
        inter_phase = 0
        if fbg_times:
            total_fbg_ms = sum(fbg_times)
            avg_fbg_ms = total_fbg_ms / len(fbg_times)
            fbg_per_pass = avg_fbg_ms
            print(f"\n    ITERATION TOTAL (forward_batch_generation, decode only):")
            print(f"      Decode calls: {len(fbg_times)}")
            print(f"      Avg time:        {avg_fbg_ms:.2f} ms")
            print(f"      Total decode:    {total_fbg_ms/1000:.1f} s")
            result["fbg_per_pass"] = fbg_per_pass

            # Per-iteration aligned inter-phase: fbg[i] - draft[i] - verify[i] - extend[i]
            n = min(len(fbg_times), len(draft_times), len(eagle_verify_times))
            if n > 0:
                inter_phase_vals = []
                for i in range(n):
                    ext = draft_extend_times[i] if i < len(draft_extend_times) else 0.0
                    ip = fbg_times[i] - draft_times[i] - eagle_verify_times[i] - ext
                    inter_phase_vals.append(ip)
                inter_phase = sum(inter_phase_vals) / len(inter_phase_vals)
            inter_phase_pct = (inter_phase / fbg_per_pass * 100) if fbg_per_pass > 0 else 0
            print(f"\n    INTER-PHASE (TP context switch + phase scheduling):")
            print(f"      Per pass:        {inter_phase:.2f} ms  ({inter_phase_pct:.1f}%)")
            print(f"      (computed per-iteration aligned, n={n})")
            result["inter_phase_per_pass"] = inter_phase

            # Prefill + scheduler + client overhead
            non_decode_ms = total_time_ms - total_fbg_ms
            non_decode_pct = (non_decode_ms / total_time_ms * 100) if total_time_ms > 0 else 0
            print(f"\n    NON-DECODE OVERHEAD (prefill + scheduler + sampling + client):")
            print(f"      Total:           {non_decode_ms/1000:.1f} s  ({non_decode_pct:.1f}% of wall time)")
            print(f"      Per decode iter: {non_decode_ms/len(fbg_times):.2f} ms  (amortized)")
            result["non_decode_ms"] = non_decode_ms
        else:
            accounted_phases = draft_per_pass + p2_total_per_pass + draft_extend_per_pass
            inter_phase = gpu_pass_ms - accounted_phases
            print(f"\n    UNACCOUNTED (inter-phase + scheduler + prefill):")
            print(f"      Per pass:        {inter_phase:.2f} ms")

        # Summary table — use fbg_per_pass as 100% base
        # For P2 sub-components: use P2 total as authoritative, show ×80 as informational
        # P2 internal residual from P2 total (avoids ×80 inflation from profiling overhead)
        iter_base = fbg_per_pass if fbg_per_pass > 0 else gpu_pass_ms
        print(f"\n    ─── PER DECODE ITERATION ({iter_base:.1f} ms, batch≈{avg_batch:.1f}) ───")
        # Use P2 total directly; layer-0 ×80 shown as sub-breakdown
        p2_display = p2_total_per_pass if p2_total_per_pass > 0 else (
            attn_per_pass + mlp_per_pass + norm_per_pass + logits_per_pass + tree_verify_per_pass)
        components = [
            ("P1 Draft", draft_per_pass),
            ("P2 Verify total", p2_display),
            ("  Attn block ×80", attn_per_pass),
            ("    ├─Core attn", core_attn_per_pass),
            ("    └─QKV+O proj", attn_per_pass - core_attn_per_pass),
            ("  MLP/FFN ×80", mlp_per_pass),
            ("  RMSNorm ×80", norm_per_pass),
            ("  lm_head/logits", logits_per_pass),
            ("  Tree verify", tree_verify_per_pass),
            ("  P2 overhead", p2_internal_residual),
            ("P3 Draft extend", draft_extend_per_pass),
            ("Inter-phase", inter_phase),
        ]
        for name, val in components:
            pct = (val / iter_base * 100) if iter_base > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"      {name:<20s} {val:7.2f} ms  ({pct:5.1f}%)  {bar}")

    result.update({
        "full_model_attn_ms": full_attn_kernel_ms,
        "full_model_attn_pct": full_attn_kernel_pct,
        "model_forward_ms": model_forward_ms,
        "model_forward_pct": model_forward_pct,
    })
    return result


def main():
    args = parse_args()

    need_patch = args.profile_attention or args.profile_nvtx or args.profile_nvtx_sync

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
        if args.profile_nvtx or args.profile_nvtx_sync:
            flags.append("nvtx")
            if args.profile_nvtx_sync:
                flags.append("nvtx_sync")
                print("[NVTX] NVTX markers enabled with cuda.synchronize() — accurate GPU timing")
            else:
                print("[NVTX] NVTX markers enabled — run with nsys to capture")
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

    # Resolve speculative decoding parameters based on algorithm
    algo = args.speculative_algorithm
    if algo == "STANDALONE":
        num_steps = args.num_steps if args.num_steps is not None else 5
        eagle_topk = args.eagle_topk if args.eagle_topk is not None else 1
        num_draft_tokens = args.num_draft_tokens if args.num_draft_tokens is not None else num_steps + 1
    else:  # EAGLE3
        num_steps = args.num_steps if args.num_steps is not None else 5
        eagle_topk = args.eagle_topk if args.eagle_topk is not None else 8
        num_draft_tokens = args.num_draft_tokens if args.num_draft_tokens is not None else 63

    print(f"\nTarget model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"TP size: {args.tp_size}")
    print(f"Algorithm: {algo}")
    fly_info = ""
    if args.fly_enabled:
        kernel_str = "cuda" if args.fly_use_cuda_kernel else "python"
        fly_info = (f", FLy: threshold={args.fly_entropy_threshold}, "
                    f"window={args.fly_window_size}, kernel={kernel_str}")
    print(f"Spec params: num_steps={num_steps}, "
          f"eagle_topk={eagle_topk}, "
          f"num_draft_tokens={num_draft_tokens}{fly_info}")

    # Create SGLang engine
    engine_kwargs = dict(
        model_path=args.target_model,
        speculative_algorithm=algo,
        speculative_draft_model_path=args.draft_model,
        speculative_num_steps=num_steps,
        speculative_eagle_topk=eagle_topk,
        speculative_num_draft_tokens=num_draft_tokens,
        tp_size=args.tp_size,
        max_running_requests=args.max_num_seqs,
        dtype=args.dtype,
        disable_cuda_graph=args.disable_cuda_graph,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        log_level=args.log_level,
        watchdog_timeout=1800,
    )
    if args.fly_enabled:
        engine_kwargs["speculative_fly_enabled"] = True
        engine_kwargs["speculative_fly_entropy_threshold"] = args.fly_entropy_threshold
        engine_kwargs["speculative_fly_window_size"] = args.fly_window_size
        if args.fly_use_cuda_kernel:
            engine_kwargs["speculative_fly_use_cuda_kernel"] = True
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
        print(f"[CONFIG] Forcing attention backend: {args.attention_backend}")

    if need_patch:
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
    total_fly_deferred = 0
    total_fly_accepted = 0
    total_fly_rejected = 0
    total_fly_entropy_gate = 0
    total_fly_depth_insufficient = 0
    has_fly_metrics = False

    for output in outputs:
        completion_tokens = output["meta_info"]["completion_tokens"]
        total_output_tokens += completion_tokens

        if "spec_verify_ct" in output["meta_info"]:
            has_spec_metrics = True
            total_verify_ct += output["meta_info"]["spec_verify_ct"]

        if "spec_accepted_tokens" in output["meta_info"]:
            total_accepted_tokens += output["meta_info"]["spec_accepted_tokens"]

        if "fly_deferred_count" in output["meta_info"]:
            has_fly_metrics = True
            total_fly_deferred += output["meta_info"]["fly_deferred_count"]
            total_fly_accepted += output["meta_info"]["fly_deferred_accepted"]
            total_fly_rejected += output["meta_info"]["fly_deferred_rejected"]
            total_fly_entropy_gate += output["meta_info"]["fly_entropy_gate_rejections"]
            total_fly_depth_insufficient += output["meta_info"].get("fly_depth_insufficient", 0)

    throughput = total_output_tokens / total_time

    # Print results
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS (SGLang {algo})")
    print("=" * 70)
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Algorithm: {algo}, num_steps={num_steps}, topk={eagle_topk}, "
          f"num_draft_tokens={num_draft_tokens}")
    if args.fly_enabled:
        print(f"FLy: enabled, entropy_threshold={args.fly_entropy_threshold}, "
              f"window_size={args.fly_window_size}")
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

    if has_fly_metrics and (total_fly_deferred > 0 or total_fly_entropy_gate > 0
                            or total_fly_depth_insufficient > 0):
        fly_accept_rate = total_fly_accepted / total_fly_deferred * 100 if total_fly_deferred > 0 else 0
        print("-" * 70)
        print("FLY METRICS:")
        print(f"  Deferrals attempted: {total_fly_deferred}")
        print(f"  Deferrals accepted (window passed): {total_fly_accepted}")
        print(f"  Deferrals rejected (window failed): {total_fly_rejected}")
        print(f"  Entropy gate rejections: {total_fly_entropy_gate}")
        print(f"  Depth insufficient (skipped): {total_fly_depth_insufficient}")
        print(f"  Deferral success rate: {fly_accept_rate:.1f}%")

    # Read CUDA event profiling results (periodic dump from subprocess)
    profile_metrics = None
    if profile_file:
        profile_metrics = read_profile_results(profile_file, total_time, total_verify_ct, tp_size=args.tp_size)

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
        "speculative_algorithm": algo,
        "num_steps": num_steps,
        "eagle_topk": eagle_topk,
        "num_draft_tokens": num_draft_tokens,
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

    if has_fly_metrics:
        json_metrics["fly"] = {
            "enabled": args.fly_enabled,
            "entropy_threshold": args.fly_entropy_threshold,
            "window_size": args.fly_window_size,
            "total_deferrals": total_fly_deferred,
            "deferrals_accepted": total_fly_accepted,
            "deferrals_rejected": total_fly_rejected,
            "entropy_gate_rejections": total_fly_entropy_gate,
            "depth_insufficient": total_fly_depth_insufficient,
        }

    if profile_metrics:
        json_metrics["profile"] = profile_metrics

    print(f"\nJSON metrics: {json.dumps(json_metrics)}")

    if engine is not None:
        engine.shutdown()


if __name__ == "__main__":
    main()
