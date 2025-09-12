export CUDA_VISIBLE_DEVICES=6,7
export PORT=29503

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_2gpus.yaml \
    --main_process_port $PORT \
    --num_processes=2 src/open_r1/grpo.py \
    --config recipes/Qwen3-4B/grpo/config_tldr_regular_skywork-8b_seed43.yaml \
    --vllm_mode="colocate"