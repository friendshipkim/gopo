#!/usr/bin/env python3
"""
Compute out-of-sample reward scores for every (prompt, completion) pair
from a completions artifact, using one or more reward models.

Outputs a reusable JSON with the following shape:

{
  "meta": {
    "source_folder": str,
    "checkpoint": int,
    "model_repo": str,
    "num_prompts": int,
    "n_completions": int,
    "split": str,
    "seed": int,
    "reward_models": [str, ...],
    "generated_at": ISO8601 str
  },
  "items": [
    {
      "prompt_index": int,
      "prompt": str,
      "rewards": {
        "<reward_model_id>": [float, ... n_completions],
        "...": [...]
      },
      "completion_count": int,
      "stats": {
        "<reward_model_short>": {"mean": float, "std": float, "min": float, "max": float},
        "...": { ... }
      }
    },
    ... num_prompts
  ]
}

Usage examples:
  - From a specific completions file:
    python evaluate/oos_reward.py \
      --completions-file completions_n30/<model-folder>/<file>.json

  - From a folder + checkpoint (auto-detect JSON):
    python evaluate/oos_reward.py \
      --folder completions_n30/<model-folder> --checkpoint 25

  - Multiple checkpoints:
    python evaluate/oos_reward.py \
      --folder completions_n30/<model-folder> --checkpoints 25 50 75 100

Note: Seed is automatically read from the input completions file metadata.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


REWARD_MODEL_IDS = {
    "skywork": "Skywork/Skywork-Reward-V2-Qwen3-8B",
    "qrm": "friendshipkim/QRM-Llama3.1-8B-v2",
}


def find_completions_file(folder: str, checkpoint: int) -> str:
    """Find a completions JSON file in folder that corresponds to the checkpoint.
    Strategy: pick the first file containing f"checkpoint{checkpoint}_" and "completions".
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    for name in os.listdir(folder):
        if not name.endswith(".json"):
            continue
        if f"checkpoint{checkpoint}_" in name and "completions" in name:
            return os.path.join(folder, name)
    raise FileNotFoundError(
        f"No completions JSON found in {folder} for checkpoint{checkpoint}")


def load_completions(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    items = data.get("items", [])
    # Expect format: {"prompt": str, "completions": [str, ...]}
    for idx, it in enumerate(items):
        if not isinstance(it, dict) or "prompt" not in it or "completions" not in it:
            raise ValueError(f"Item {idx} missing 'prompt' or 'completions'")
        if not isinstance(it["prompt"], str) or not isinstance(it["completions"], list):
            raise ValueError(f"Item {idx} has invalid types")
    return meta, items


def load_reward_model(model_id: str):
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise ImportError("transformers/torch not available. Install required packages.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        num_labels=1,
    )
    return tokenizer, model


def score_prompt_completions(tokenizer, model, prompt: str, completions: List[str]) -> List[float]:
    """Score all completions for a single prompt with given reward model.
    Sequential scoring for clarity and robustness.
    """
    device = next(model.parameters()).device
    scores: List[float] = []
    for completion in completions:
        input_text = prompt + completion
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
            padding_side="right",
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits[0, 0].item()
        scores.append(score)
    return scores


def calc_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if np is None:
        # Fallback without numpy
        m = sum(values) / len(values)
        var = sum((x - m) ** 2 for x in values) / len(values)
        return {"mean": m, "std": var ** 0.5, "min": min(values), "max": max(values)}
    arr = np.array(values, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def derive_output_folder_and_meta(completions_path: str, folder_arg: str | None) -> Tuple[str, str]:
    """Return (model_folder_name, source_folder) for output/meta.
    model_folder_name is the last directory under completions_n*/, used under oos_reward/.
    source_folder is the provided completions folder path if available, else parent dir of file.
    """
    if folder_arg:
        source_folder = folder_arg.rstrip("/")
        model_folder = os.path.basename(source_folder)
        return model_folder, source_folder
    # infer from completions path
    parent = os.path.dirname(completions_path.rstrip("/"))
    model_folder = os.path.basename(parent)
    return model_folder, parent


def extract_checkpoint_from_filename(filename: str) -> int:
    m = re.search(r"checkpoint(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else -1


def process_single_checkpoint(folder: str, checkpoint: int, loaded_models: Dict[str, Tuple[Any, Any]], 
                              reward_models: List[str], output_dir_root: str):
    """Process a single checkpoint and save rewards. Uses pre-loaded reward models."""
    completions_path = find_completions_file(folder, checkpoint)
    
    # Load completions
    meta_in, items = load_completions(completions_path)
    num_prompts = len(items)
    if num_prompts == 0:
        raise ValueError("No items found in completions file")
    n_completions = len(items[0]["completions"]) if isinstance(items[0].get("completions"), list) else 0

    # Get seed from input metadata (used for generation)
    seed = meta_in.get("seed", 42)  # Default to 42 if not found

    # Prepare output folders/meta
    model_folder_name, source_folder = derive_output_folder_and_meta(completions_path, folder)
    output_dir = os.path.join(output_dir_root, model_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build model_repo from filename (best-effort informational)
    base_name = os.path.basename(completions_path)
    model_repo_guess = base_name.split("_")[0] if "checkpoint" in base_name else meta_in.get("model_path", "unknown")

    # Compute rewards
    results: List[Dict[str, Any]] = []
    for idx, it in enumerate(tqdm(items, desc=f"Scoring prompts (checkpoint {checkpoint})")):
        prompt = it["prompt"]
        completions = it["completions"]
        per_model_scores: Dict[str, List[float]] = {}
        for rm, (tok, mdl) in loaded_models.items():
            scores = score_prompt_completions(tok, mdl, prompt, completions)
            per_model_scores[rm] = scores

        # Stats per model
        stats: Dict[str, Dict[str, float]] = {}
        for rm, scores in per_model_scores.items():
            short = "skywork" if "Skywork" in rm or "skywork" in rm else ("qrm" if "QRM" in rm or "qrm" in rm else rm)
            stats[short] = calc_stats(scores)

        results.append({
            "prompt_index": idx,
            "prompt": prompt,
            "rewards": per_model_scores,
            "completion_count": len(completions),
            "stats": stats,
        })

    # Assemble payload
    payload = {
        "meta": {
            "source_folder": source_folder,
            "checkpoint": checkpoint,
            "model_repo": model_repo_guess,
            "num_prompts": num_prompts,
            "n_completions": n_completions,
            "split": meta_in.get("split", "test"),
            "seed": seed,
            "reward_models": list(reward_models),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "items": results,
    }

    # Output filename
    out_name = f"checkpoint{checkpoint}_rewards_n{n_completions}_seed{seed}.json"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved rewards to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute OOS reward arrays for completions")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--completions-file", help="Path to completions JSON file")
    src.add_argument("--folder", help="Path to folder under completions_n*/<model-folder>")
    parser.add_argument("--checkpoint", type=int, help="Single checkpoint number (required if --folder is used)")
    parser.add_argument("--checkpoints", nargs="+", type=int, help="Multiple checkpoint numbers (e.g., 25 50 75)")
    parser.add_argument("--reward-models", nargs="*", default=[REWARD_MODEL_IDS["skywork"], REWARD_MODEL_IDS["qrm"]],
                        help="Reward model HF IDs to use")
    parser.add_argument("--output-dir-root", default="oos_reward", help="Root directory to save outputs")
    args = parser.parse_args()

    # Determine checkpoints to process
    if args.folder:
        if args.checkpoint and args.checkpoints:
            raise ValueError("Cannot use both --checkpoint and --checkpoints")
        if args.checkpoint:
            checkpoints = [args.checkpoint]
        elif args.checkpoints:
            checkpoints = args.checkpoints
        else:
            raise ValueError("Must specify either --checkpoint or --checkpoints when using --folder")
        
        # Load reward models once (they'll be reused for all checkpoints)
        print(f"Loading reward models (will be reused for all {len(checkpoints)} checkpoints)...")
        loaded_models: Dict[str, Tuple[Any, Any]] = {}
        for rm in args.reward_models:
            print(f"Loading reward model: {rm}")
            tok, mdl = load_reward_model(rm)
            loaded_models[rm] = (tok, mdl)
        print("Reward models loaded successfully!\n")
        
        # Process each checkpoint
        print(f"Processing {len(checkpoints)} checkpoint(s): {checkpoints}")
        for checkpoint in checkpoints:
            print(f"\n{'='*80}")
            print(f"Processing checkpoint {checkpoint}")
            print(f"{'='*80}\n")
            try:
                process_single_checkpoint(args.folder, checkpoint, loaded_models, args.reward_models, args.output_dir_root)
            except Exception as e:
                print(f"Error processing checkpoint {checkpoint}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        # Single file mode
        completions_path = args.completions_file
        checkpoint = extract_checkpoint_from_filename(completions_path)
        folder = os.path.dirname(completions_path)
        
        # Load reward models
        print("Loading reward models...")
        loaded_models: Dict[str, Tuple[Any, Any]] = {}
        for rm in args.reward_models:
            print(f"Loading reward model: {rm}")
            tok, mdl = load_reward_model(rm)
            loaded_models[rm] = (tok, mdl)
        
        process_single_checkpoint(folder, checkpoint, loaded_models, args.reward_models, args.output_dir_root)


if __name__ == "__main__":
    main()


