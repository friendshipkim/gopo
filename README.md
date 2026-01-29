# GOPO

GRPO/GOPO training framework for language models.

## Installation

```shell
conda create -n rl --python 3.11 && conda activate rl && pip install --upgrade pip
```

Next, install vLLM and FlashAttention:

```shell
pip install vllm==0.8.5.post1
pip install setuptools && pip install flash-attn==2.7.4.post1 --no-build-isolation
```

This will also install PyTorch `v2.6.0` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

```shell
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

## Environment Variables

Set your HuggingFace username for dataset paths:

```shell
export HF_USERNAME=your_username
export HF_TOKEN=your_huggingface_token  # Required for pushing datasets
```

You can add these to your `~/.bashrc` for persistence.

## Data Preprocessing

Preprocessing scripts are located in the `preprocess_data/` folder. These scripts download, process, and push datasets to your HuggingFace Hub.

### UltraChat Dataset (Chat)

```shell
python preprocess_data/preprocess_ultrachat_dataset.py
```

Downloads `HuggingFaceH4/ultrachat_200k`, creates train/val/test splits, and pushes to `$HF_USERNAME/UltraChat-200k`.

### TLDR Dataset (Summarization)

```shell
python preprocess_data/preprocess_tldr_datasets.py
```

Downloads `trl-lib/tldr`, samples validation set, and pushes to `$HF_USERNAME/tldr`.

### Instruction Following Dataset

```shell
python preprocess_data/preprocess_if_datasets.py
```

Merges `allenai/tulu-3-sft-personas-instruction-following` (train) with `google/IFEval` (test), and pushes to `$HF_USERNAME/IF-Datasets-Tulu-IFEval`.

## Training

Training scripts are in the `train_scripts/` folder. These run GRPO/GOPO training with 4 GPUs using recipes from the `recipes/` folder.

### Running Training

```shell
# GRPO training (regular advantage)
bash train_scripts/qwen3_1.7_grpo_chat.sh

# GOPO training (ranking advantage)
bash train_scripts/qwen3_1.7_gopo_chat.sh
```

### Customizing Training

1. **Change GPU assignment**: Edit `CUDA_VISIBLE_DEVICES` in the bash script
2. **Change config**: Modify `CONFIG_FILE` to point to a different recipe

### Available Recipes

Recipes are in `recipes/Qwen3-1.7B/`:

| Config | Dataset | Advantage | Reward Model |
|--------|---------|-----------|--------------|
| `config_chat_regular_qrm_seed42.yaml` | UltraChat | Regular | QRM |
| `config_chat_ranking_qrm_seed42.yaml` | UltraChat | Ranking | QRM |
| `config_tldr_regular_skywork-8b_seed42.yaml` | TLDR | Regular | Skywork-8B |
| `config_tldr_ranking_skywork-8b_seed42.yaml` | TLDR | Ranking | Skywork-8B |
| `config_if_regular_skywork-8b_seed42.yaml` | IF-Datasets | Regular | Skywork-8B |
| `config_if_ranking_skywork-8b_seed42.yaml` | IF-Datasets | Ranking | Skywork-8B |

### Example: Custom Training Run

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PORT=29500
export HF_USERNAME=your_username

CONFIG_FILE="recipes/Qwen3-1.7B/config_tldr_ranking_skywork-8b_seed42.yaml"
PROCESSED_CONFIG=$(mktemp)
envsubst < "$CONFIG_FILE" > "$PROCESSED_CONFIG"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_4gpus.yaml \
    --main_process_port $PORT \
    --num_processes=4 src/open_r1/grpo.py \
    --config "$PROCESSED_CONFIG"

rm -f "$PROCESSED_CONFIG"
```


## Generation

The `generate/` folder contains scripts for generating model completions on evaluation datasets.

### Generate Completions

`generate_completions.py` generates completions from a trained model checkpoint for later evaluation.

```shell
python generate/generate_completions.py \
    --model <path/to/model_checkpoint> \
    --dataset chat \
    --num-prompts 1000 \
    --n-completions 3 \
    --temperature 0.7 \
    --seed 42
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Path to model (HuggingFace path or local directory) |
| `--dataset` | Dataset to use: `chat`, `tldr`, `if`, or `storygen` (auto-detected from model path if omitted) |
| `--num-prompts` | Number of prompts to sample (default: 100) |
| `--n-completions` | Number of completions per prompt (default: 1) |
| `--temperature` | Sampling temperature (default: 0.7) |
| `--top-p` | Top-p nucleus sampling (default: 0.9) |
| `--max-new-tokens` | Maximum tokens to generate (default: 2048) |
| `--seed` | Random seed for reproducibility |
| `--output` | Output path (auto-generated if omitted) |
| `--no-vllm` | Disable vLLM, use transformers instead |
| `--vllm-gpu-memory` | GPU memory utilization for vLLM (default: 0.85) |

#### Output Structure

Completions are saved to `completions_n{N}/<model_name>_temp{T}/` with auto-generated filenames:

```
completions_n3/
└── Qwen3-1.7B-chat-ranking_temp0.7/
    └── Qwen3-1.7B-chat-ranking-checkpoint100_1000prompts_3completions_seed42_temp0.7.json
```


## Evaluation

The `evaluate/` folder contains scripts for comparing model outputs using LLM-as-judge evaluation.

### Bootstrap Judge Evaluation

`bootstrap_judge.py` compares two models' completions using bootstrap sampling to compute win rates with confidence intervals.

```shell
python evaluate/bootstrap_judge.py \
    --completions1 <path/to/model1_completions.json> \
    --completions2 <path/to/model2_completions.json> \
    --judge-model gpt-4o \
    --api-key $OPENAI_API_KEY \
    --N 25 \
    --B 25 \
    --seed 42 \
    --no-ties
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--completions1` | Path to first model's completions (JSON/JSONL) |
| `--completions2` | Path to second model's completions (JSON/JSONL) |
| `--judge-model` | LLM judge model (`gpt-4o`, `gpt-5`, `claude-3-5-sonnet`, etc.) |
| `--api-key` | OpenAI or Anthropic API key (auto-detected by prefix) |
| `--N` | Number of prompts to subsample per bootstrap iteration |
| `--B` | Number of bootstrap iterations |
| `--seed` | Random seed for reproducibility |
| `--no-ties` | Force judge to pick a winner (no ties allowed) |
| `--allow-ties` | Allow ties in evaluation |
| `--completion-index` | Which completion to use from multi-completion files (default: 0) |
| `--output-dir` | Output directory for results (default: `evaluate/`) |

#### Completion File Format

The script accepts JSON files with the following structure:

```json
{
  "meta": {},
  "items": [
    {"prompt": "...", "completions": ["response1", "response2", ...]},
    ...
  ]
}
```

Or JSONL format with one item per line containing `prompt` and `responses` fields.