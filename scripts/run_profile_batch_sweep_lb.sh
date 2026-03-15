#!/bin/bash
# Batch size sweep: profile chain + tree on LongBench v2 (32 prompts)
# Batch sizes: 2, 4, 8, 16
# Target: Llama-3.1-70B-Instruct  |  Draft: Llama-3.1-8B-Instruct
# GPUs: 0,1,2,3 (TP=4)

set -euo pipefail
cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate vllm-spec

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="/home/siyu/Documents/vllm_spec_scenario_b/thirdparty/sglang/python:${PYTHONPATH:-}"

TARGET_MODEL="/home/dataset_model/model/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
DRAFT_MODEL="/home/dataset_model/model/Llama-3.1-8B-Instruct"
DATASET="/home/dataset_model/dataset/longbench_v2_filtered"

NUM_PROMPTS=32

for BATCH_SIZE in 2 4 8 16; do
    echo ""
    echo "================================================================"
    echo "  CHAIN  |  batch_size=$BATCH_SIZE  |  $NUM_PROMPTS prompts"
    echo "================================================================"
    python benchmark_sglang_eagle3.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --speculative-algorithm STANDALONE \
        --tp-size 4 \
        --num-steps 22 \
        --num-draft-tokens 23 \
        --max-tokens 512 \
        --num-prompts $NUM_PROMPTS \
        --max-num-seqs $BATCH_SIZE \
        --mem-fraction-static 0.55 \
        --dtype float16 \
        --dataset "$DATASET" \
        --max-context-tokens 32000 \
        --log-level error \
        --fly-enabled \
        --fly-entropy-threshold 0.3 \
        --fly-window-size 6 \
        --profile-attention \
        --disable-cuda-graph \
        2>&1
done

for BATCH_SIZE in 2 4 8 16; do
    echo ""
    echo "================================================================"
    echo "  TREE  |  batch_size=$BATCH_SIZE  |  $NUM_PROMPTS prompts"
    echo "================================================================"
    python benchmark_sglang_eagle3.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --speculative-algorithm STANDALONE \
        --tp-size 4 \
        --eagle-topk 5 \
        --num-steps 22 \
        --num-draft-tokens 512 \
        --max-tokens 512 \
        --num-prompts $NUM_PROMPTS \
        --max-num-seqs $BATCH_SIZE \
        --mem-fraction-static 0.65 \
        --dtype float16 \
        --dataset "$DATASET" \
        --max-context-tokens 32000 \
        --log-level error \
        --fly-enabled \
        --fly-entropy-threshold 0.3 \
        --fly-window-size 6 \
        --profile-attention \
        --disable-cuda-graph \
        2>&1
done

echo ""
echo "================================================================"
echo "  SWEEP COMPLETE"
echo "================================================================"
