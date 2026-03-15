#!/bin/bash
# Finetune Llama-3.1-8B on LongBench QA data (DDP, 4 GPUs)
# Usage: bash run_finetune.sh

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup torchrun --nproc_per_node=4 train.py > /tmp/finetune.log 2>&1 &
echo "Training launched (PID: $!). Log: /tmp/finetune.log"
