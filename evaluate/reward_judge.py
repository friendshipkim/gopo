#!/usr/bin/env python3
"""
Reward-based judging CLI that uses reward models to judge responses.
Given (prompt, response1, response2), it injects (prompt, response1) and (prompt, response2) 
into a reward model and determines the winner based on higher reward score.

Reward model selection:
- If completion filename contains "tldr" or "if": use skywork-qwen3-8b
- If completion filename contains "chat": use qrm
"""

import os
import json
import argparse
import re
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Reward model imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    print("Warning: transformers package not found. Please install it with: pip install transformers torch")
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None


def read_completions_artifact(path: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Read and validate completions artifact JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    items = data.get("items", [])
    
    # If no items key, try results key (for evaluation files)
    if not items and "results" in data:
        items = data["results"]
    
    # Minimal validation - handle both completions and evaluation file formats
    for idx, it in enumerate(items):
        # Check for evaluation file format first
        if all(k in it for k in ("prompt", "model1_completion", "model2_completion")):
            # Evaluation file format
            if not isinstance(it["prompt"], str) or not isinstance(it["model1_completion"], str) or not isinstance(it["model2_completion"], str):
                raise ValueError(f"Item {idx} fields must be strings")
        # Check for completions file format
        elif all(k in it for k in ("prompt", "completion1", "completion2")):
            # Completions file format
            if not isinstance(it["prompt"], str) or not isinstance(it["completion1"], str) or not isinstance(it["completion2"], str):
                raise ValueError(f"Item {idx} fields must be strings")
        else:
            raise ValueError(f"Item {idx} missing required keys. Expected either (prompt, model1_completion, model2_completion) or (prompt, completion1, completion2)")
    return meta, items


def determine_reward_model(completions_filename: str) -> str:
    """Determine which reward model to use based on filename."""
    filename_lower = completions_filename.lower()
    if "tldr" in filename_lower or "if" in filename_lower:
        return "skywork-qwen3-8b"
    elif "chat" in filename_lower:
        return "qrm"
    else:
        # Default to skywork-qwen3-8b if no specific pattern matches
        return "skywork-qwen3-8b"


def load_reward_model(model_name: str):
    """Load the reward model and tokenizer."""
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise ImportError("Required packages (transformers, torch) not available")
    
    print(f"Loading reward model: {model_name}")
    
    # Model paths - using HuggingFace model IDs
    model_paths = {
        "skywork-qwen3-8b": "Skywork/Skywork-Reward-V2-Qwen3-8B",
        "skywork-qwen3-4b": "Skywork/Skywork-Reward-V2-Qwen3-4B", 
        "qrm": "friendshipkim/QRM-Llama3.1-8B-v2"
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Unknown reward model: {model_name}")
    
    model_path = model_paths[model_name]
    
    # Load tokenizer and model (sequence classification for reward models)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        num_labels=1  # Reward models typically have 1 output (reward score)
    )
    
    return tokenizer, model


def get_reward_score(tokenizer, model, prompt: str, response: str) -> float:
    """Get reward score for a given prompt-response pair."""
    # Format the input for the reward model (concatenate prompt + completion)
    input_text = prompt + response
    
    # Tokenize with right padding and no special tokens (matching GRPO trainer)
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,
        padding=True,
        padding_side="right",
        add_special_tokens=False
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
        # For sequence classification models, the reward score is the first (and only) logit
        reward_score = outputs.logits[0, 0].item()
    
    return reward_score


def call_reward_judge(tokenizer, model, prompt: str, response1: str, response2: str) -> Dict[str, Any]:
    """Judge two responses using reward model scores."""
    # Get reward scores for both responses
    score1 = get_reward_score(tokenizer, model, prompt, response1)
    score2 = get_reward_score(tokenizer, model, prompt, response2)
    
    # Determine winner based on scores
    if score1 > score2:
        winner = "A"
        explanation = f"Response A has higher reward score ({score1:.4f} vs {score2:.4f})"
    elif score2 > score1:
        winner = "B"
        explanation = f"Response B has higher reward score ({score2:.4f} vs {score1:.4f})"
    else:
        winner = "tie"
        explanation = f"Both responses have equal reward scores ({score1:.4f})"
    
    return {
        "winner": winner,
        "explanation": explanation,
        "score1": score1,
        "score2": score2,
        "score_diff": abs(score1 - score2)
    }


def generate_output_filename(completions_filename: str, reward_model: str, seed: int) -> str:
    """Generate output filename based on input filename and parameters."""
    # Remove .json extension and path
    base = os.path.splitext(os.path.basename(completions_filename))[0]
    
    # Add reward model info
    base += f"_{reward_model.replace('-', '_')}"
    
    # Add seed
    base += f"_seed{seed}"
    
    return f"{base}.json"


def main():
    parser = argparse.ArgumentParser(description="Reward-based judging using reward models")
    parser.add_argument("--completions", required=True, help="Path to completions artifact JSON file")
    parser.add_argument("--output_dir", default="reward_evaluation", help="Output directory for results")
    parser.add_argument("--num_votes", type=int, default=1, help="Number of votes per prompt (not used for reward models)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reward_model", help="Override reward model selection (skywork-qwen3-8b or qrm)")
    parser.add_argument("--max_prompts", type=int, help="Maximum number of prompts to process (for testing)")
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    random.seed(args.seed)
    
    # Read completions
    print(f"Reading completions from: {args.completions}")
    meta, items = read_completions_artifact(args.completions)
    
    # Determine reward model
    if args.reward_model:
        reward_model = args.reward_model
    else:
        reward_model = determine_reward_model(args.completions)
    
    print(f"Using reward model: {reward_model}")
    
    # Load reward model
    tokenizer, model = load_reward_model(reward_model)
    
    # Limit prompts if specified
    if args.max_prompts:
        items = items[:args.max_prompts]
        print(f"Processing only first {args.max_prompts} prompts")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each prompt
    results = []
    for idx, item in enumerate(tqdm(items, desc="Processing prompts")):
        prompt = item["prompt"]
        
        # Handle both file formats
        if "model1_completion" in item and "model2_completion" in item:
            # Evaluation file format
            response1 = item["model1_completion"]
            response2 = item["model2_completion"]
        elif "completion1" in item and "completion2" in item:
            # Completions file format
            response1 = item["completion1"]
            response2 = item["completion2"]
        else:
            raise ValueError(f"Item {idx} has unsupported format")
        
        # Get reward-based judgment
        judgment = call_reward_judge(tokenizer, model, prompt, response1, response2)
        
        # Map back to original model names
        if judgment["winner"] == "A":
            final_winner = "model1"
        elif judgment["winner"] == "B":
            final_winner = "model2"
        else:
            final_winner = "tie"
        
        result = {
            "prompt": prompt,
            "model1_completion": response1,
            "model2_completion": response2,
            "reward_score1": judgment["score1"],
            "reward_score2": judgment["score2"],
            "score_difference": judgment["score_diff"],
            "winner": final_winner,
            "explanation": judgment["explanation"]
        }
        
        results.append(result)
    
    # Calculate statistics
    model1_wins = sum(1 for r in results if r["winner"] == "model1")
    model2_wins = sum(1 for r in results if r["winner"] == "model2")
    ties = sum(1 for r in results if r["winner"] == "tie")
    total = len(results)
    
    analysis = {
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "ties": ties,
        "model1_win_rate": model1_wins / total if total > 0 else 0,
        "model2_win_rate": model2_wins / total if total > 0 else 0,
        "tie_rate": ties / total if total > 0 else 0,
        "avg_score_diff": sum(r["score_difference"] for r in results) / total if total > 0 else 0
    }
    
    # Prepare output data
    output_data = {
        "model1_path": meta.get("model1_path", "unknown"),
        "model2_path": meta.get("model2_path", "unknown"),
        "reward_model": reward_model,
        "seed": args.seed,
        "analysis": analysis,
        "results": results
    }
    
    # Save results
    output_filename = generate_output_filename(args.completions, reward_model, args.seed)
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Model1 wins: {model1_wins} ({analysis['model1_win_rate']*100:.1f}%)")
    print(f"Model2 wins: {model2_wins} ({analysis['model2_win_rate']*100:.1f}%)")
    print(f"Ties: {ties} ({analysis['tie_rate']*100:.1f}%)")
    print(f"Average score difference: {analysis['avg_score_diff']:.4f}")


if __name__ == "__main__":
    main()