#!/bin/bash

# Base paths
BASE_DIR="/teamspace/studios/this_studio/FollowComplexInstruction"
GET_DATA_DIR="$BASE_DIR/get_data"
TRAIN_DATA_DIR="$BASE_DIR/dpo_train/data"

# Parameters
TEACHER_BACKEND="vllm"
MODEL="google/gemma-3-27b-it"
RES_PATH="$GET_DATA_DIR/results/checked_res_gemma2_2b_it.jsonl"
IFT_DATA="$TRAIN_DATA_DIR/gemma2_ift_train.jsonl"
DPO_DATA="$TRAIN_DATA_DIR/gemma2_dpo_train.jsonl"

# Model settings
MAX_TOKENS=512
TEMP=0.2
TOP_P=0.9
TP_SIZE=1
DTYPE="auto"
GPU_UTIL=0.9

# Run
python "$GET_DATA_DIR/correct.py" \
  --teacher_backend "$TEACHER_BACKEND" \
  --model_name_or_path "$MODEL" \
  --res_path "$RES_PATH" \
  --ift_data_path "$IFT_DATA" \
  --dpo_data_path "$DPO_DATA" \
  --dtype "$DTYPE" \
  --trust_remote_code \
  --use_chat_template
