#!/bin/bash
# Monitor GPU availability and launch baseline + unified sparse experiments
# when 4 free GPUs are found (< 1GB used).

set -e

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

TARGET_MODEL="/home/dataset_model/model/Llama-3.1-8B-Instruct"
DRAFT_MODEL="/home/dataset_model/model/models--lmsys--sglang-EAGLE3-LLaMA3.1-Instruct-8B/snapshots/28a53ce8911434c031d7c78392abb26d898ec293"
DATASET="/home/dataset_model/dataset/longbench_v2_filtered"

MIN_FREE_GPUS=4
THRESHOLD_MB=1000  # GPU with < 1GB used is considered free

echo "[MONITOR] Waiting for $MIN_FREE_GPUS free GPUs (< ${THRESHOLD_MB}MB used)..."
echo "[MONITOR] Checking every 30 seconds..."

while true; do
    # Get list of free GPU indices
    FREE_GPUS=()
    while IFS=, read -r idx mem_used; do
        idx=$(echo "$idx" | xargs)
        mem_used=$(echo "$mem_used" | xargs | sed 's/ MiB//')
        if [ "$mem_used" -lt "$THRESHOLD_MB" ]; then
            FREE_GPUS+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)

    echo "[MONITOR] $(date '+%H:%M:%S') Free GPUs: ${FREE_GPUS[*]:-none} (${#FREE_GPUS[@]}/$MIN_FREE_GPUS needed)"

    if [ "${#FREE_GPUS[@]}" -ge "$MIN_FREE_GPUS" ]; then
        # Pick first 4 free GPUs, split into 2 pairs
        GPU_BASELINE="${FREE_GPUS[0]},${FREE_GPUS[1]}"
        GPU_UNIFIED="${FREE_GPUS[2]},${FREE_GPUS[3]}"
        echo ""
        echo "[MONITOR] Found ${#FREE_GPUS[@]} free GPUs! Launching experiments..."
        echo "[MONITOR] Baseline on GPUs: $GPU_BASELINE"
        echo "[MONITOR] Unified on GPUs:  $GPU_UNIFIED"
        echo ""
        break
    fi

    sleep 30
done

cd "$SCRIPTS_DIR"

# Launch baseline
echo "=========================================="
echo "[BASELINE] Starting on GPUs $GPU_BASELINE"
echo "=========================================="
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
CUDA_VISIBLE_DEVICES=$GPU_BASELINE \
nsys profile --trace=cuda,nvtx --force-overwrite=true \
    -o "$PROJECT_DIR/longbench_baseline" \
    python benchmark_sglang_eagle3.py \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --tp-size 2 --profile-nvtx --disable-cuda-graph \
    --max-tokens 4096 --num-prompts 16 \
    --max-context-tokens 16000 --max-num-seqs 4 \
    --dataset "$DATASET" \
    2>&1 | tee "$PROJECT_DIR/longbench_baseline.log"

BASELINE_EXIT=$?
echo ""
echo "[BASELINE] Finished with exit code $BASELINE_EXIT"
echo ""

# Launch unified sparse
echo "=========================================="
echo "[UNIFIED] Starting on GPUs $GPU_UNIFIED"
echo "=========================================="
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
CUDA_VISIBLE_DEVICES=$GPU_UNIFIED \
nsys profile --trace=cuda,nvtx --force-overwrite=true \
    -o "$PROJECT_DIR/longbench_unified" \
    python benchmark_sglang_eagle3.py \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --tp-size 2 --use-unified-sparse --kv-top-k-ratio 0.3 \
    --profile-nvtx --disable-cuda-graph \
    --max-tokens 4096 --num-prompts 16 \
    --max-context-tokens 16000 --max-num-seqs 4 \
    --dataset "$DATASET" \
    2>&1 | tee "$PROJECT_DIR/longbench_unified.log"

UNIFIED_EXIT=$?
echo ""
echo "[UNIFIED] Finished with exit code $UNIFIED_EXIT"

echo ""
echo "=========================================="
echo "Both experiments complete!"
echo "  Baseline log: $PROJECT_DIR/longbench_baseline.log"
echo "  Unified log:  $PROJECT_DIR/longbench_unified.log"
echo "  Baseline nsys: $PROJECT_DIR/longbench_baseline.nsys-rep"
echo "  Unified nsys:  $PROJECT_DIR/longbench_unified.nsys-rep"
echo ""
echo "Analyze with:"
echo "  nsys stats --report nvtxsum $PROJECT_DIR/longbench_baseline.nsys-rep"
echo "  nsys stats --report nvtxsum $PROJECT_DIR/longbench_unified.nsys-rep"
echo "=========================================="
