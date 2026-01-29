export CUDA_VISIBLE_DEVICES=4,5,6,7
export PORT=29501
export HF_USERNAME=${HF_USERNAME:?Please set HF_USERNAME environment variable}

# Process YAML config with environment variables
CONFIG_FILE="recipes/Qwen3-1.7B/config_chat_regular_qrm_seed42.yaml"
PROCESSED_CONFIG=$(mktemp)
envsubst < "$CONFIG_FILE" > "$PROCESSED_CONFIG"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_4gpus.yaml \
    --main_process_port $PORT \
    --num_processes=4 src/open_r1/grpo.py \
    --config "$PROCESSED_CONFIG"

rm -f "$PROCESSED_CONFIG"