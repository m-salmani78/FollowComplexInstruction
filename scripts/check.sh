#!/bin/bash

BASE_DIR="/teamspace/studios/this_studio/FollowComplexInstruction/get_data"
DATA_DIR="$BASE_DIR/data"
RESULTS_DIR="$BASE_DIR/results"

python "$BASE_DIR/check.py" \
    --input_data="$DATA_DIR/data.jsonl" \
    --input_response_data="$RESULTS_DIR/res_gemma2_2b_it.jsonl" \
    --output_dir="$RESULTS_DIR" \
    --output_file_name="checked_res_gemma2_2b_it"
