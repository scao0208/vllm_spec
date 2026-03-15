#!/bin/bash
# Profile: SGLang STANDALONE tree + FLy — LongBench v2 (long context)
# Target: Llama-3.1-70B-Instruct  |  Draft: Llama-3.1-8B-Instruct
# GPUs: 0,1,2,3 (TP=4)
# Tree: num_steps=22, topk=5, num_draft_tokens=512
# FLy: entropy_threshold=0.3, window_size=6
# Dataset: longbench_v2_filtered, max_context_tokens=32000

set -euo pipefail
cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate vllm-spec

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="/home/siyu/Documents/vllm_spec_scenario_b/thirdparty/sglang/python:${PYTHONPATH:-}"

TARGET_MODEL="/home/dataset_model/model/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
DRAFT_MODEL="/home/dataset_model/model/Llama-3.1-8B-Instruct"
DATASET="/home/dataset_model/dataset/longbench_v2_filtered"

python benchmark_sglang_eagle3.py \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --speculative-algorithm STANDALONE \
    --tp-size 4 \
    --eagle-topk 5 \
    --num-steps 22 \
    --num-draft-tokens 512 \
    --max-tokens 512 \
    --num-prompts 20 \
    --max-num-seqs 1 \
    --mem-fraction-static 0.70 \
    --dtype float16 \
    --dataset "$DATASET" \
    --max-context-tokens 32000 \
    --log-level error \
    --fly-enabled \
    --fly-entropy-threshold 0.3 \
    --fly-window-size 6 \
    --profile-attention \
    --disable-cuda-graph \
    2>&1 | tee profile_fly_tree_lb.log
