export CUDA_VISIBLE_DEVICES=4,5,6,7
export PORT=29501

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_4gpus.yaml \
    --main_process_port $PORT \
    --num_processes=4 src/open_r1/grpo.py \
    --config recipes/Qwen3-8B/grpo/config_if_regular_skywork-8b_seed42.yaml \
    --vllm_mode="colocate"