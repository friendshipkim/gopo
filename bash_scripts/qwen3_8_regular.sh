export CUDA_VISIBLE_DEVICES=0
export PORT=29503

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_singlegpu.yaml \
    --main_process_port $PORT \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/Qwen3-8B/grpo/config_chat_regular.yaml \
    --vllm_mode="colocate"