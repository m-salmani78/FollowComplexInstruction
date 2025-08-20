#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

BASE_MODEL=${1:-google/gemma-2-2b-it}
OUTPUT_DIR=${2:-/teamspace/studios/this_studio/FollowComplexInstruction/dpo_train/output}

TRAIN_FILE=/teamspace/studios/this_studio/FollowComplexInstruction/dpo_train/data/gemma2_dpo_train.jsonl
EVAL_FILE=/teamspace/studios/this_studio/FollowComplexInstruction/dpo_train/data/gemma2_dpo_train.jsonl

python /teamspace/studios/this_studio/FollowComplexInstruction/dpo_train/dpo_train.py \
  --model_name_or_path "$BASE_MODEL" \
  --trust_remote_code \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --max_steps 1000 \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 10 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_prompt_length 1024 \
  --max_length 1024 \
  --report_to none