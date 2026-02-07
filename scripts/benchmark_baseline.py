"""
Benchmark baseline inference (no speculative decoding) with vLLM.

Use this to measure pure target model throughput for speedup comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/benchmark_baseline.py \
        --target-model /home/dataset_model/model/models--meta-llama--Llama-3.1-70B-Instruct/snapshot
  s/1605565b47bb9346c5515c34102e054115b4f98b \
        --tensor-parallel-size 4 \
        --max-tokens 512 \
        --max-num-seqs 2 \
        --dataset /home/dataset_model/dataset/gsm8k \
        --enforce-eager
"""

import argparse
import json
import time

import numpy as np
import torch
from vllm import LLM, SamplingParams

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
        description="Benchmark baseline inference (no speculative decoding) with vLLM"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Target model path or HuggingFace ID",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size (default: auto)",
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
    print(f"Tensor parallel size: {tp_size}")

    print(f"\nTarget model: {args.target_model}")
    print("Mode: BASELINE (no speculative decoding)")

    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
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

    # Per-request statistics for variance calculation
    tokens_per_request = [len(o.outputs[0].token_ids) for o in outputs]

    # Time per token (key metric for speedup comparison)
    # In baseline, each forward pass produces 1 token per sequence
    time_per_token_ms = (total_time / total_output_tokens) * 1000

    # Calculate per-request throughput (tokens/time proportional)
    # Note: We can't get exact per-request timing from vLLM batch inference
    # but we can estimate based on token count distribution
    tokens_array = np.array(tokens_per_request)

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS - BASELINE (No Speculative Decoding)")
    print("=" * 70)
    print(f"Target model: {args.target_model}")
    print(f"Num prompts: {len(prompts)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Max num seqs: {args.max_num_seqs}")
    print("-" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print("-" * 70)
    print("PER-TOKEN METRICS (for speedup calculation):")
    print(f"  Time per token: {time_per_token_ms:.4f} ms")
    print(f"  Tokens per request: mean={tokens_array.mean():.1f}, std={tokens_array.std():.1f}")
    print(f"  Tokens per request: min={tokens_array.min()}, max={tokens_array.max()}")
    print("=" * 70)

    # Output key metrics in JSON format for easy comparison
    metrics = {
        "mode": "baseline",
        "total_time_s": total_time,
        "total_tokens": total_output_tokens,
        "throughput_tokens_per_s": throughput,
        "time_per_token_ms": time_per_token_ms,
        "num_prompts": len(prompts),
        "tokens_per_request_mean": float(tokens_array.mean()),
        "tokens_per_request_std": float(tokens_array.std()),
    }
    print(f"\nJSON metrics: {json.dumps(metrics)}")

    # Print sample outputs
    print("\nSAMPLE OUTPUTS (first 2):")
    for i, output in enumerate(outputs[:2]):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input: {prompts[i][:80]}...")
        text = output.outputs[0].text
        print(f"Output ({len(output.outputs[0].token_ids)} tokens): {text[:300]}...")


if __name__ == "__main__":
    main()
