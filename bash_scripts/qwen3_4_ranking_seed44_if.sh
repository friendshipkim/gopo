export CUDA_VISIBLE_DEVICES=0,1,2,3
export PORT=29500

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_4gpus.yaml \
    --main_process_port $PORT \
    --num_processes=4 src/open_r1/grpo.py \
    --config recipes/Qwen3-4B/grpo/config_if_ranking_skywork-8b_seed44.yaml \
    --vllm_mode="colocate"