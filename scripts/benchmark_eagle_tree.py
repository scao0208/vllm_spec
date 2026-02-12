"""
Benchmark EAGLE expanded tree speculative decoding (Scenario A).

Compares vLLM's default unified attention kernel vs ea_attn_exp tree attention
kernel for tree verification with large EAGLE-style trees (512+ tokens).

Usage:
    cd scripts
    conda activate vllm-spec

    # Baseline (no spec decode)
    python benchmark_eagle_tree.py \
        --target-model meta-llama/Llama-3.3-70B-Instruct \
        --draft-model meta-llama/Llama-3.1-8B-Instruct \
        --tensor-parallel-size 2 \
        --draft-tensor-parallel-size 2 \
        --tree-size 512 \
        --dataset /path/to/gsm8k

    # With ea_attn_exp kernel
    python benchmark_eagle_tree.py \
        --target-model meta-llama/Llama-3.3-70B-Instruct \
        --draft-model meta-llama/Llama-3.1-8B-Instruct \
        --tensor-parallel-size 2 \
        --draft-tensor-parallel-size 2 \
        --tree-size 512 \
        --use-eagle-kernel \
        --dataset /path/to/gsm8k
"""

import argparse
import json
import time

import torch
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector
from eagle_tree_choices import EAGLE_TREES, get_tree_stats
from datasets import detect_and_load

# Patch vLLM's MAX_SPEC_LEN to support 512+ token trees.
# Default is 128 which is too small for our large tree benchmarks.
import vllm.v1.sample.rejection_sampler as _rs
_rs.MAX_SPEC_LEN = 1024

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
    "Write a recursive function to calculate Fibonacci numbers.",
    "Explain how HTTPS encryption works.",
    "What is the difference between TCP and UDP?",
    "Write a SQL query to find the second highest salary.",
    "Explain the concept of closure in JavaScript.",
    "What are the benefits of using Docker containers?",
    "Write a Python decorator that measures function execution time.",
    "Explain how garbage collection works in Java.",
    "What is the difference between mutex and semaphore?",
    "Write a function to detect a cycle in a linked list.",
]


def patch_tree_attention_with_eagle_kernel():
    """Monkey-patch vLLM's TreeAttentionImpl to use the ea_attn_exp kernel.

    The ea_attn_exp kernel has a two-stage design:
    - Stage 1: Dense attention to past context (no mask loading)
    - Stage 2: Sparse tree attention with mask only for tree region

    This is more efficient than vLLM's unified_attention for large trees (512+)
    because it skips bias loading for past-context KV blocks.
    """
    from eagle_tree_attention import _attention_strided

    from vllm.v1.attention.backends.tree_attn import (
        TreeAttentionImpl,
        TreeAttentionMetadata,
    )
    from vllm import _custom_ops as ops

    original_forward = TreeAttentionImpl.forward

    def patched_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output.fill_(0)

        # Cache KV
        key_cache, value_cache = kv_cache.unbind(0)
        if self.kv_sharing_target_layer_name is None:
            ops.reshape_and_cache_flash(
                key, value, key_cache, value_cache,
                attn_metadata.slot_mapping, self.kv_cache_dtype,
                layer._k_scale, layer._v_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        # Prefill path: use original unified_attention (no tree mask needed)
        if prefill_meta := attn_metadata.prefill_metadata:
            from vllm.v1.attention.ops.triton_unified_attention import unified_attention
            descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, key.shape[1])
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache, v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale, causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        # Decode path: use ea_attn_exp kernel for tree attention
        if decode_meta := attn_metadata.decode_metadata:
            tree_bias = decode_meta.tree_attn_bias
            if tree_bias is not None and decode_meta.max_query_len > 1:
                # Use ea_attn_exp two-stage kernel
                _eagle_tree_decode(
                    self, layer, query[:num_decode_tokens],
                    key_cache, value_cache,
                    output[:num_decode_tokens],
                    decode_meta, _attention_strided,
                )
            else:
                # Single-token decode: use original path
                from vllm.v1.attention.ops.triton_unified_attention import unified_attention
                descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, key.shape[1])
                unified_attention(
                    q=query[:num_decode_tokens],
                    k=key_cache, v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_query_len,
                    seqused_k=decode_meta.seq_lens,
                    max_seqlen_k=decode_meta.max_seq_len,
                    softmax_scale=self.scale, causal=True,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=decode_meta.block_table,
                    softcap=self.logits_soft_cap,
                    q_descale=None,
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
        return output

    def _eagle_tree_decode(self, layer, query, key_cache, value_cache,
                           output, decode_meta, kernel_cls):
        """Run tree attention decode with the ea_attn_exp kernel.

        Gathers KV from paged cache into contiguous tensors, reshapes
        query from vLLM flat format to 4D [B, H, N, D], runs the
        two-stage kernel, then writes output back.
        """
        tree_bias = decode_meta.tree_attn_bias
        block_table = decode_meta.block_table
        seq_lens = decode_meta.seq_lens
        q_start_loc = decode_meta.query_start_loc

        num_seqs = seq_lens.shape[0]
        num_q_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size
        block_size = key_cache.shape[1]  # [num_blocks, block_size, num_kv_heads, head_size]

        # For each sequence, gather KV from paged cache into contiguous tensor
        # and run the kernel
        q_seqlens = torch.diff(q_start_loc)
        max_q_len = int(q_seqlens.max().item())
        max_kv_len = int(seq_lens.max().item())

        # Expand tree_bias to additive mask format [1, 1, tree_len, tree_len]
        # vLLM tree_bias is [tree_len, tree_len] with 0/-inf
        tree_mask_4d = tree_bias[:max_q_len, :max_q_len].unsqueeze(0).unsqueeze(0)

        for seq_idx in range(num_seqs):
            q_start = int(q_start_loc[seq_idx].item())
            q_len = int(q_seqlens[seq_idx].item())
            kv_len = int(seq_lens[seq_idx].item())

            if q_len <= 1:
                # Single-token: skip tree attention
                continue

            # Extract query for this sequence: [q_len, num_q_heads, head_size]
            q_seq = query[q_start:q_start + q_len]
            # Reshape to [1, num_q_heads, q_len, head_size]
            q_4d = q_seq.permute(1, 0, 2).unsqueeze(0).contiguous()

            # Gather KV from paged cache
            num_blocks_needed = (kv_len + block_size - 1) // block_size
            seq_block_table = block_table[seq_idx, :num_blocks_needed]

            # key_cache: [num_blocks, block_size, num_kv_heads, head_size]
            k_blocks = key_cache[seq_block_table]  # [num_blocks_needed, block_size, num_kv_heads, head_size]
            v_blocks = value_cache[seq_block_table]

            # Reshape to contiguous: [1, num_kv_heads, kv_len, head_size]
            k_contig = k_blocks.reshape(-1, num_kv_heads, head_size)[:kv_len]
            v_contig = v_blocks.reshape(-1, num_kv_heads, head_size)[:kv_len]
            k_4d = k_contig.permute(1, 0, 2).unsqueeze(0).contiguous()
            v_4d = v_contig.permute(1, 0, 2).unsqueeze(0).contiguous()

            # Handle GQA: expand KV heads to match Q heads
            if num_q_heads != num_kv_heads:
                repeat_factor = num_q_heads // num_kv_heads
                k_4d = k_4d.repeat_interleave(repeat_factor, dim=1)
                v_4d = v_4d.repeat_interleave(repeat_factor, dim=1)

            # Ensure fp16
            q_4d = q_4d.to(torch.float16)
            k_4d = k_4d.to(torch.float16)
            v_4d = v_4d.to(torch.float16)

            # Slice tree mask for this q_len
            mask = tree_mask_4d[:, :, :q_len, :q_len].to(torch.float16)

            # Run ea_attn_exp kernel
            sm_scale = self.scale
            out_4d = kernel_cls.apply(q_4d, k_4d, v_4d, mask, sm_scale)

            # Write back: [1, num_q_heads, q_len, head_size] -> [q_len, num_q_heads, head_size]
            out_seq = out_4d.squeeze(0).permute(1, 0, 2).contiguous()
            # Flatten to [q_len, num_q_heads * head_size] for output
            output[q_start:q_start + q_len] = out_seq.reshape(q_len, -1)

    TreeAttentionImpl.forward = patched_forward
    print("[EAGLE KERNEL] Patched TreeAttentionImpl.forward with ea_attn_exp kernel")


def extract_metrics(raw_metrics, num_spec_tokens=512):
    """Extract acceptance metrics from vLLM metrics."""
    metrics_dict = {}
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * min(num_spec_tokens, 50)

    for metric in raw_metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(min(len(metric.values), len(acceptance_counts))):
                acceptance_counts[pos] += metric.values[pos]

    metrics_dict["num_drafts"] = num_drafts
    metrics_dict["num_draft_tokens"] = num_draft_tokens
    metrics_dict["num_accepted_tokens"] = num_accepted_tokens
    metrics_dict["acceptance_counts"] = acceptance_counts

    if num_drafts > 0:
        metrics_dict["avg_acceptance_length"] = 1 + (num_accepted_tokens / num_drafts)
        metrics_dict["draft_efficiency"] = (
            num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0
        )
        for i in range(len(acceptance_counts)):
            if acceptance_counts[i] > 0 or i < 5:
                metrics_dict[f"acceptance_rate_pos_{i}"] = acceptance_counts[i] / num_drafts
    else:
        metrics_dict["avg_acceptance_length"] = 1.0
        metrics_dict["draft_efficiency"] = 0.0

    return metrics_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE expanded tree speculative decoding (Scenario A)"
    )
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--tree-size", type=int, default=512, choices=[256, 512, 1024],
                        help="EAGLE tree size (256, 512, or 1024 tokens)")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=None,
                        help="Draft model TP size (defaults to --tensor-parallel-size)")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--use-eagle-kernel", action="store_true",
                        help="Use ea_attn_exp tree attention kernel instead of vLLM unified attention")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--method", type=str, default="draft_model",
                        choices=["draft_model", "eagle", "eagle3"],
                        help="Speculative decoding method")
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply monkey-patch before creating LLM if requested
    if args.use_eagle_kernel:
        patch_tree_attention_with_eagle_kernel()

    # Load prompts
    if args.dataset:
        prompts = detect_and_load(args.dataset, args.num_prompts)
    else:
        prompts = SAMPLE_PROMPTS
        if args.num_prompts is not None:
            prompts = prompts[:args.num_prompts]
    print(f"Running with {len(prompts)} prompts")

    # Get tree
    tree = EAGLE_TREES[args.tree_size]
    stats = get_tree_stats(tree)
    print(f"\nEAGLE Tree (size={args.tree_size}):")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")

    # Setup TP
    tp_size = args.tensor_parallel_size or torch.cuda.device_count()
    draft_tp_size = args.draft_tensor_parallel_size or tp_size

    print(f"\nScenario A: EAGLE Expanded Tree")
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Tree size: {args.tree_size} nodes, depth {stats['max_depth']}")
    print(f"Kernel: {'ea_attn_exp' if args.use_eagle_kernel else 'vllm_unified'}")
    print(f"TP: target={tp_size}, draft={draft_tp_size}")

    spec_config = {
        "method": args.method,
        "model": args.draft_model,
        "num_speculative_tokens": len(tree),
        "speculative_token_tree": str(tree),
        "draft_tensor_parallel_size": draft_tp_size,
    }

    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
        speculative_config=spec_config,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Run
    print(f"\nGenerating {args.max_tokens} tokens per prompt...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_output_tokens / total_time

    metrics = extract_metrics(llm.get_metrics(), len(tree))

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS - Scenario A: EAGLE Expanded Tree")
    print("=" * 70)
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Tree size: {args.tree_size} nodes, depth {stats['max_depth']}")
    print(f"Kernel: {'ea_attn_exp' if args.use_eagle_kernel else 'vllm_unified'}")
    print(f"Num prompts: {len(prompts)}")
    print("-" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print("-" * 70)
    print("SPECULATION METRICS:")
    print(f"  Num drafts: {metrics['num_drafts']}")
    print(f"  Num draft tokens: {metrics['num_draft_tokens']}")
    print(f"  Num accepted tokens: {metrics['num_accepted_tokens']}")
    print(f"  Avg acceptance length: {metrics['avg_acceptance_length']:.2f}")
    print(f"  Draft efficiency: {metrics['draft_efficiency']:.2%}")
    print("-" * 70)
    print("ACCEPTANCE RATE BY POSITION (first 20):")
    for i in range(min(20, len(tree))):
        key = f"acceptance_rate_pos_{i}"
        if key in metrics:
            bar = "#" * int(metrics[key] * 40)
            print(f"  Position {i:3d}: {metrics[key]:6.2%} {bar}")
    print("-" * 70)

    # Speedup metrics
    num_drafts = metrics['num_drafts']
    num_accepted = metrics['num_accepted_tokens']
    time_per_iteration_ms = (total_time / num_drafts) * 1000 if num_drafts > 0 else 0
    tokens_per_iteration = metrics['avg_acceptance_length']
    time_per_token_ms = time_per_iteration_ms / tokens_per_iteration if tokens_per_iteration > 0 else 0

    print("SPEEDUP METRICS:")
    print(f"  Num iterations: {num_drafts}")
    print(f"  Time per iteration: {time_per_iteration_ms:.4f} ms")
    print(f"  Tokens per iteration (avg acceptance): {tokens_per_iteration:.2f}")
    print(f"  Time per token: {time_per_token_ms:.4f} ms")
    print("=" * 70)

    # JSON output
    json_metrics = {
        "scenario": "A_eagle_expanded_tree",
        "tree_size": args.tree_size,
        "tree_depth": stats["max_depth"],
        "kernel": "ea_attn_exp" if args.use_eagle_kernel else "vllm_unified",
        "total_time_s": total_time,
        "total_tokens": total_output_tokens,
        "throughput_tokens_per_s": throughput,
        "time_per_token_ms": time_per_token_ms,
        "num_prompts": len(prompts),
        "num_drafts": num_drafts,
        "num_accepted_tokens": num_accepted,
        "avg_acceptance_length": metrics["avg_acceptance_length"],
        "draft_efficiency": metrics["draft_efficiency"],
        "time_per_iteration_ms": time_per_iteration_ms,
        "tokens_per_iteration": tokens_per_iteration,
    }
    print(f"\nJSON metrics: {json.dumps(json_metrics)}")


if __name__ == "__main__":
    main()
