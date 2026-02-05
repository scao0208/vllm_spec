# Speculative decoding by vllm 0.15.0

This repositary provides guidance to deploy classic speculative decoding experiments following the paper "Accelerating Large Language Model Decoding
with Speculative Sampling"(https://arxiv.org/pdf/2302.01318).

## Project Overview

This is a benchmarking toolkit for evaluating **vLLM speculative decoding** performance. It measures speedup achieved when using a smaller "draft" model to speculate tokens for a larger "target" model.

## Running Benchmarks

**Baseline (no speculative decoding):**
```bash
python scripts/benchmark_baseline.py \
    --target-model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 \
    --num-prompts 20 \
    --max-tokens 256 \
    --gsm8k /path/to/gsm8k
```

**Speculative decoding with draft model:**
```bash
python scripts/benchmark_draft_model.py \
    --target-model /path/to/70B-model \
    --draft-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 24 \
    --tensor-parallel-size 4 \
    --draft-tensor-parallel-size 1 \
    --use-tree \
    --max-num-seqs 96 \
    --gsm8k /path/to/gsm8k
```

**Key flags:**
- `--enforce-eager`: Disable CUDA graphs (helps with stability)
- `--use-tree`: Enable tree-based speculation with 512-node structure
- `--gsm8k`: Path to GSM8K dataset (expects parquet at `{path}/main/test-00000-of-00001.parquet`)

## Architecture

Two benchmark scripts with a shared structure:
1. `benchmark_baseline.py` - Pure target model inference, establishes baseline throughput
2. `benchmark_draft_model.py` - Speculative decoding with draft model, extracts acceptance metrics

**Key metrics extracted:**
- Throughput (tokens/s)
- Time per token (ms) - primary comparison metric
- Acceptance rate by position (speculative only)
- Draft efficiency = accepted_tokens / proposed_tokens (speculative only)

**Tree structures:** The draft benchmark includes hardcoded Monte Carlo simulation trees (`mc_sim_8b_12`, `mc_sim_8b_512`) for tree-based speculative decoding. The 512-node tree spans 24 levels.

**Metrics extraction:** Uses vLLM's internal counters:
- `vllm:spec_decode_num_drafts`
- `vllm:spec_decode_num_draft_tokens`
- `vllm:spec_decode_num_accepted_tokens`
- `vllm:spec_decode_num_accepted_tokens_per_pos`