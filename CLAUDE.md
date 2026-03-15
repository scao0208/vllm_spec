# CLAUDE.md - Scenario A: EAGLE Expanded Tree

## Overview

This worktree implements **Scenario A**: benchmarking EAGLE-style expanded tree speculative decoding with the ea_attn_exp two-stage tree attention kernel.

The key idea: EAGLE generates dynamic draft trees with 10+ depth layers and 512+ total tokens. At this scale, the ea_attn_exp kernel's two-stage design (Stage 1: dense past context, Stage 2: sparse tree region) should outperform vLLM's generic unified attention kernel which loads qq_bias for every KV tile.

## Branch

`scenario-a-eagle-tree` (git worktree of vllm_spec)

## Setup

```bash
conda activate vllm-spec
cd scripts
```

## Files

- `eagle_tree_choices.py` - EAGLE expanded tree definitions (256/512/1024 tokens, 10-16 depth)
- `eagle_tree_attention.py` - ea_attn_exp Triton kernel (copied from ea_attn_exp project)
- `benchmark_eagle_tree.py` - Benchmark script comparing vLLM unified vs ea_attn_exp kernel

## Running Benchmarks

```bash
# vLLM unified attention (baseline)
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_eagle_tree.py \
    --target-model meta-llama/Llama-3.3-70B-Instruct \
    --draft-model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --tree-size 512 \
    --dataset /path/to/gsm8k \
    --enforce-eager

# ea_attn_exp tree kernel
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_eagle_tree.py \
    --target-model meta-llama/Llama-3.3-70B-Instruct \
    --draft-model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --tree-size 512 \
    --use-eagle-kernel \
    --dataset /path/to/gsm8k \
    --enforce-eager
```

## Tree Sizes

| Tree | Nodes | Max Depth | Top-K |
|------|-------|-----------|-------|
| eagle_tree_256 | 256 | 12 | 8 |
| eagle_tree_512 | 512 | 14 | 10 |
| eagle_tree_1024 | 1024 | 16 | 12 |

## Metrics Collected

- `avg_acceptance_length` - Mean accepted tokens per speculation round
- `time_per_token_ms` - End-to-end latency per output token
- `draft_efficiency` - Fraction of draft tokens accepted
- `throughput_tokens_per_s` - Output tokens per second
