"""
Benchmark draft model speculative decoding with vLLM.

Use a smaller model (e.g., 8B) as draft to speculate for a larger model (e.g., 70B).

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/benchmark_draft_model.py \
        --target-model /home/dataset_model/model/models--meta-llama--Llama-3.1-70B-Instruct/snapshot
  s/1605565b47bb9346c5515c34102e054115b4f98b \
        --draft-model /home/dataset_model/model/Llama-3.1-8B-Instruct \
        --num-speculative-tokens 15\
        --tensor-parallel-size 4 \
        --draft-tensor-parallel-size 4 \
        --max-tokens 512 \
        --max-num-seqs 2 \
        --dataset /home/dataset_model/dataset/gsm8k \
        --enforce-eager
"""

import argparse
import json
import time

import torch
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector
from choices import mc_sim_8b_12, mc_sim_8b_512
from datasets import detect_and_load

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark draft model speculative decoding with vLLM"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Target (large) model path or HuggingFace ID",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        required=True,
        help="Draft (small) model path or HuggingFace ID",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=5,
        help="Number of tokens to speculate per round (K)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for target model (default: auto)",
    )
    parser.add_argument(
        "--draft-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for draft model (default: 1)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to run (default: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with custom prompts (optional)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (may help with stability)",
    )
    parser.add_argument(
        "--use-tree",
        action="store_true",
        help="Use mc_sim_8b_512 tree structure for speculation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset directory (auto-detects gsm8k, humaneval, mbpp from path name)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per iteration (batch size)",
    )
    return parser.parse_args()


def extract_metrics(raw_metrics, num_spec_tokens: int = 10) -> dict:
    """Extract acceptance metrics from vLLM metrics."""
    metrics_dict = {}
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_spec_tokens

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
            print(f"num_accepted_tokens: {num_accepted_tokens}")
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(min(len(metric.values), num_spec_tokens)):
                acceptance_counts[pos] += metric.values[pos]

    metrics_dict["num_drafts"] = num_drafts
    metrics_dict["num_draft_tokens"] = num_draft_tokens
    metrics_dict["num_accepted_tokens"] = num_accepted_tokens
    metrics_dict["acceptance_counts"] = acceptance_counts

    if num_drafts > 0:
        # avg_acceptance_length = 1 (bonus token) + accepted_tokens / num_drafts
        metrics_dict["avg_acceptance_length"] = 1 + (num_accepted_tokens / num_drafts)
        metrics_dict["draft_efficiency"] = (
            num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0
        )
        for i in range(num_spec_tokens):
            if acceptance_counts[i] > 0 or i < 5:
                metrics_dict[f"acceptance_rate_pos_{i}"] = acceptance_counts[i] / num_drafts
    else:
        metrics_dict["avg_acceptance_length"] = 1.0
        metrics_dict["draft_efficiency"] = 0.0

    return metrics_dict


def main():
    args = parse_args()

    # Load prompts
    if args.dataset:
        prompts = detect_and_load(args.dataset, args.num_prompts)
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        if args.num_prompts is not None:
            prompts = prompts[: args.num_prompts]
    else:
        prompts = SAMPLE_PROMPTS
        if args.num_prompts is not None:
            prompts = prompts[: args.num_prompts]
    print(f"Running with {len(prompts)} prompts")

    # Setup tensor parallel size
    tp_size = args.tensor_parallel_size
    if tp_size is None:
        tp_size = torch.cuda.device_count()
    print(f"Target model tensor parallel size: {tp_size}")
    print(f"Draft model tensor parallel size: {args.draft_tensor_parallel_size}")

    print(f"\nTarget model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Num speculative tokens (K): {args.num_speculative_tokens}")

    # Create speculative config for draft model
    spec_config = {
        "method": "draft_model",
        "model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
    }
    if args.use_tree:
        spec_config["speculative_token_tree"] = str(mc_sim_8b_512)
        print(f"Using tree structure with {len(mc_sim_8b_512)} nodes")

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

    # Run generation
    print(f"\nGenerating {args.max_tokens} tokens per prompt...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    # Calculate stats
    total_time = end_time - start_time
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_output_tokens / total_time

    # Extract speculative decoding metrics
    metrics = extract_metrics(llm.get_metrics(), args.num_speculative_tokens)

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS - Draft Model Speculative Decoding")
    print("=" * 70)
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Num speculative tokens (K): {args.num_speculative_tokens}")
    print(f"Num prompts: {len(prompts)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("-" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print("-" * 70)
    print("SPECULATION METRICS:")
    print(f"  Num drafts (speculation rounds): {metrics['num_drafts']}")
    print(f"  Num draft tokens proposed: {metrics['num_draft_tokens']}")
    print(f"  Num accepted tokens: {metrics['num_accepted_tokens']}")
    print(f"  Avg acceptance length: {metrics['avg_acceptance_length']:.2f}")
    print(f"  Draft efficiency: {metrics['draft_efficiency']:.2%}")
    print("-" * 70)
    print("ACCEPTANCE RATE BY POSITION:")
    for i in range(min(args.num_speculative_tokens, 10)):
        key = f"acceptance_rate_pos_{i}"
        if key in metrics:
            bar = "█" * int(metrics[key] * 40)
            print(f"  Position {i}: {metrics[key]:6.2%} {bar}")
    print("-" * 70)

    # Calculate per-iteration metrics for speedup comparison
    # Each draft round (iteration) produces: 1 bonus token + accepted tokens
    num_drafts = metrics['num_drafts']
    num_accepted = metrics['num_accepted_tokens']
    # Total tokens from spec decoding = accepted + bonus tokens (one per draft)
    total_spec_tokens = num_accepted + num_drafts

    # Time per iteration (draft round)
    time_per_iteration_ms = (total_time / num_drafts) * 1000 if num_drafts > 0 else 0
    # Tokens per iteration (mean acceptance length)
    tokens_per_iteration = metrics['avg_acceptance_length']
    # Time per token = time_per_iteration / tokens_per_iteration
    time_per_token_ms = time_per_iteration_ms / tokens_per_iteration if tokens_per_iteration > 0 else 0

    print("SPEEDUP METRICS:")
    print(f"  Num iterations (draft rounds): {num_drafts}")
    print(f"  Time per iteration: {time_per_iteration_ms:.4f} ms")
    print(f"  Tokens per iteration (avg acceptance): {tokens_per_iteration:.2f}")
    print(f"  Time per token: {time_per_token_ms:.4f} ms")
    print("=" * 70)

    # Output key metrics in JSON format for easy comparison
    json_metrics = {
        "mode": "speculative_decoding",
        "total_time_s": total_time,
        "total_tokens": total_output_tokens,
        "throughput_tokens_per_s": throughput,
        "time_per_token_ms": time_per_token_ms,
        "num_prompts": len(prompts),
        "num_drafts": num_drafts,
        "num_accepted_tokens": num_accepted,
        "avg_acceptance_length": metrics['avg_acceptance_length'],
        "draft_efficiency": metrics['draft_efficiency'],
        "time_per_iteration_ms": time_per_iteration_ms,
        "tokens_per_iteration": tokens_per_iteration,
    }
    print(f"\nJSON metrics: {json.dumps(json_metrics)}")

    # Print sample outputs
    print("\nSAMPLE OUTPUTS (first 2):")
    for i, output in enumerate(outputs[:2]):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input: {prompts[i][:80]}...")
        text = output.outputs[0].text
        print(f"Output ({len(output.outputs[0].token_ids)} tokens): {text[:300]}...")


if __name__ == "__main__":
    main()
