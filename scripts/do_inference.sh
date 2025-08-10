python /teamspace/studios/this_studio/FollowComplexInstruction/get_data/do_inference.py \
  --data_path=/teamspace/studios/this_studio/FollowComplexInstruction/get_data/data/data.jsonl \
  --res_path=/teamspace/studios/this_studio/FollowComplexInstruction/get_data/data/res_gemma2_2b_it.jsonl \
  --model_path google/gemma-2-2b-it \
  --batch_size 32 \
  --max_model_len 4096 \
  --max_tokens 512 \
  --gpu_memory_utilization 0.9 \
  --dtype bfloat16