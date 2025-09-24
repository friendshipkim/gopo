#!/bin/bash

# Script to upload multiple HF model checkpoints at regular intervals
# Usage: ./upload_multiple_checkpoints.sh <total_steps> <interval> [model_base_name] [model_dir_pattern] [repo_prefix]

set -e  # Exit on any error

# Default values
DEFAULT_MODEL_BASE="Qwen3-1.7B-ultrachat-bsz128-regular-seed42-lr2e-6"
DEFAULT_MODEL_DIR_PATTERN="../saved_models/Qwen3-1.7B-ultrachat-bsz128-regular-seed42-lr2e-6"
DEFAULT_REPO_PREFIX="choiqs"

# Function to show usage
show_usage() {
    echo "Usage: $0 <total_steps> <interval> [model_base_name] [model_dir_pattern] [repo_prefix]"
    echo ""
    echo "Parameters:"
    echo "  total_steps     - Total number of training steps (e.g., 100)"
    echo "  interval        - Interval between checkpoints (e.g., 25)"
    echo "  model_base_name - Base name for the model (optional)"
    echo "                   Default: $DEFAULT_MODEL_BASE"
    echo "  model_dir_pattern - Pattern for model directory path (optional)"
    echo "                     Default: $DEFAULT_MODEL_DIR_PATTERN"
    echo "  repo_prefix     - HuggingFace repo prefix (optional)"
    echo "                   Default: $DEFAULT_REPO_PREFIX"
    echo ""
    echo "Example:"
    echo "  $0 100 25"
    echo "  $0 100 25 my-model ../models/my-model myusername"
    echo ""
    echo "This will process checkpoints at: 25, 50, 75, 100"
}

# Check if we have at least the required parameters
if [ $# -lt 2 ]; then
    echo "Error: Missing required parameters"
    echo ""
    show_usage
    exit 1
fi

# Parse arguments
TOTAL_STEPS=$1
INTERVAL=$2
MODEL_BASE_NAME=${3:-$DEFAULT_MODEL_BASE}
MODEL_DIR_PATTERN=${4:-$DEFAULT_MODEL_DIR_PATTERN}
REPO_PREFIX=${5:-$DEFAULT_REPO_PREFIX}

# Validate that total_steps and interval are positive integers
if ! [[ "$TOTAL_STEPS" =~ ^[0-9]+$ ]] || [ "$TOTAL_STEPS" -le 0 ]; then
    echo "Error: total_steps must be a positive integer"
    exit 1
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [ "$INTERVAL" -le 0 ]; then
    echo "Error: interval must be a positive integer"
    exit 1
fi

if [ "$INTERVAL" -gt "$TOTAL_STEPS" ]; then
    echo "Error: interval ($INTERVAL) cannot be greater than total_steps ($TOTAL_STEPS)"
    exit 1
fi

# Export MODEL_NAME for consistency with original script
export MODEL_NAME="$MODEL_BASE_NAME"

# Print configuration
echo "========================================="
echo "Multi-Checkpoint Upload Configuration"
echo "========================================="
echo "Total steps: $TOTAL_STEPS"
echo "Interval: $INTERVAL"
echo "Model base name: $MODEL_BASE_NAME"
echo "Model directory pattern: $MODEL_DIR_PATTERN"
echo "Repository prefix: $REPO_PREFIX"
echo ""

# Calculate and display checkpoints to process
echo "Checkpoints to process:"
checkpoints=()
for ((step=INTERVAL; step<=TOTAL_STEPS; step+=INTERVAL)); do
    checkpoints+=($step)
    echo "  - checkpoint-$step"
done
echo ""

# Confirm before proceeding
read -p "Proceed with uploading ${#checkpoints[@]} checkpoints? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

# Track success/failure
successful_uploads=()
failed_uploads=()

echo ""
echo "========================================="
echo "Starting Upload Process"
echo "========================================="

# Process each checkpoint
for checkpoint_step in "${checkpoints[@]}"; do
    echo ""
    echo "--- Processing checkpoint-$checkpoint_step ---"
    
    MODEL_DIR="$MODEL_DIR_PATTERN/checkpoint-$checkpoint_step"
    REPO_NAME="$REPO_PREFIX/$MODEL_BASE_NAME-checkpoint$checkpoint_step"
    
    echo "Model directory: $MODEL_DIR"
    echo "Repository name: $REPO_NAME"
    
    # Check if model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "❌ Error: Model directory not found: $MODEL_DIR"
        failed_uploads+=("checkpoint-$checkpoint_step (directory not found)")
        continue
    fi
    
    # Run the upload command
    echo "Uploading..."
    if python upload_hf_model.py --model-dir "$MODEL_DIR" --repo-name "$REPO_NAME"; then
        echo "✅ Successfully uploaded checkpoint-$checkpoint_step"
        successful_uploads+=("checkpoint-$checkpoint_step")
    else
        echo "❌ Failed to upload checkpoint-$checkpoint_step"
        failed_uploads+=("checkpoint-$checkpoint_step (upload failed)")
    fi
    
    echo "--- Completed checkpoint-$checkpoint_step ---"
done

echo ""
echo "========================================="
echo "Upload Summary"
echo "========================================="
echo "Total checkpoints processed: ${#checkpoints[@]}"
echo "Successful uploads: ${#successful_uploads[@]}"
echo "Failed uploads: ${#failed_uploads[@]}"
echo ""

if [ ${#successful_uploads[@]} -gt 0 ]; then
    echo "✅ Successfully uploaded:"
    for upload in "${successful_uploads[@]}"; do
        echo "   - $upload"
    done
    echo ""
fi

if [ ${#failed_uploads[@]} -gt 0 ]; then
    echo "❌ Failed uploads:"
    for upload in "${failed_uploads[@]}"; do
        echo "   - $upload"
    done
    echo ""
    echo "You may want to retry the failed uploads manually."
    exit 1
else
    echo "🎉 All checkpoints uploaded successfully!"
fi

echo "========================================="
