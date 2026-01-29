#!/usr/bin/env python3
"""
Generation-only CLI that produces a reusable completions artifact JSON
containing single model completions {prompt, completion}.

example usage:
python evaluate/generate_completions.py \
    --model "HF directory" \
    --checkpoint 275 \
    --num-prompts 100 \
    --seed 42
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any, Optional, Tuple
import gc
import torch

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Optional VLLM for fast inference (mirror majority_vote.py behavior)
# Compatible with vLLM 0.8.5.post1+
try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Falling back to standard transformers inference.")
    VLLM_AVAILABLE = False


SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer. Respond in the following format: <think>\n...\n</think>\n"
    "<answer>\n...\n</answer>"
)


def load_validation_prompts(num_prompts: int, seed: int, model_path: str = None, dataset_override: str = None) -> Tuple[List[str], Optional[List[str]], str, str]:
    print(f"Loading {num_prompts} validation prompts...")
    
    # Determine dataset based on model names
    dataset_name = "HuggingFaceH4/ultrachat_200k"  # default
    split_name = "test_sft"
    prompt_column = "messages"
    dataset_type = "ultrachat"
    
    # If dataset is explicitly specified, use it (highest priority)
    if dataset_override:
        dataset_override_lower = dataset_override.lower()
        if dataset_override_lower == "if":
            dataset_name = f"{os.environ['HF_USERNAME']}/IF-Datasets-Tulu-IFEval"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "if"
            print(f"Using explicitly specified IF dataset: {dataset_name}")
        elif dataset_override_lower == "tldr":
            dataset_name = "trl-lib/tldr"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "tldr"
            print(f"Using explicitly specified tldr dataset: {dataset_name}")
        elif dataset_override_lower == "chat" or dataset_override_lower == "ultrachat":
            dataset_name = "HuggingFaceH4/ultrachat_200k"
            split_name = "test_sft"
            prompt_column = "messages"
            dataset_type = "ultrachat"
            print(f"Using explicitly specified ultrachat dataset: {dataset_name}")
        elif dataset_override_lower == "storygen" or dataset_override_lower == "sg":
            dataset_name = f"{os.environ['HF_USERNAME']}/RUCAIBox-Story-Generation-test"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "storygen"
            print(f"Using explicitly specified StoryGen dataset: {dataset_name}")
        else:
            raise ValueError(f"Unknown dataset override: {dataset_override}. Supported values: if, tldr, chat/ultrachat, storygen/sg")
    # Otherwise, determine dataset based on model name (default behavior)
    elif model_path:
        if "tldr" in model_path.lower():
            dataset_name = "trl-lib/tldr"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "tldr"
            print("Detected tldr model, using trl-lib/tldr dataset")
        elif "if" in model_path.lower():
            dataset_name = f"{os.environ['HF_USERNAME']}/IF-Datasets-Tulu-IFEval"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "if"
            print(f"Detected IF model, using {os.environ['HF_USERNAME']}/IF-Datasets-Tulu-IFEval dataset")
        elif "sg" in model_path.lower():
            dataset_name = f"{os.environ['HF_USERNAME']}/RUCAIBox-Story-Generation-test"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "storygen"
            print(f"Detected StoryGen model, using {os.environ['HF_USERNAME']}/RUCAIBox-Story-Generation-test dataset")
        else:
            print("Using default ultrachat_200k dataset")
    
    dataset = load_dataset(dataset_name)
    print(f"Using {split_name} split to ensure no training data overlap")
    
    golden_completions: Optional[List[str]] = None
    
    if prompt_column == "messages":
        # For ultrachat format
        prompts = dataset[split_name]["messages"]
        validation_prompts: List[str] = []
        for example in prompts:
            for message in example:
                if message["role"] == "user":
                    validation_prompts.append(message["content"])
                    break
    else:
        # For prompt-based formats (tldr, if, storygen)
        validation_prompts = list(dataset[split_name]["prompt"])
        
        # For tldr, also load golden completions
        if dataset_type == "tldr":
            golden_completions = list(dataset[split_name]["completion"])
            # Create pairs to keep prompts and completions aligned
            pairs = list(zip(validation_prompts, golden_completions))
            
            # Seed and shuffle pairs together to maintain alignment
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.shuffle(pairs)
            
            # Select N pairs and unpack
            pairs = pairs[:num_prompts]
            validation_prompts, golden_completions = zip(*pairs)
            validation_prompts = list(validation_prompts)
            golden_completions = list(golden_completions)
            print(f"Loaded {len(validation_prompts)} validation prompts with golden completions from {split_name} split of {dataset_name}")
            return validation_prompts, golden_completions, dataset_type, split_name

    # For non-tldr datasets, shuffle and select as before
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(validation_prompts)
    validation_prompts = validation_prompts[:num_prompts]
    print(f"Loaded {len(validation_prompts)} validation prompts from {split_name} split of {dataset_name}")
    return validation_prompts, golden_completions, dataset_type, split_name


def format_prompt(user_message: str, tokenizer, dataset_type: str = "ultrachat") -> str:
    # Use different system prompts based on dataset
    if dataset_type == "tldr":
        system_prompt = "Summarize the following text in 100 words or less"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    elif dataset_type in ["if", "storygen"]:
        # No system prompt for IF and StoryGen datasets
        messages = [
            {"role": "user", "content": user_message},
        ]
    else:
        system_prompt = SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    
    # if tokenizer is None:
    #     return (
    #         f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    #         f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    #     )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt


def generate_completion(prompt: str, model, tokenizer, max_new_tokens: int = 2048, use_vllm: bool = False, sampling_params: Any = None, temperature: float = 0.7, top_p: float = 0.9) -> str:
    if use_vllm:
        outputs = model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_completions_vllm(prompts: List[str], vllm_model, sampling_params: Any, desc: str) -> List[List[str]]:
    """Generate completions using vLLM. Returns list of completion lists (one list per prompt)."""
    completions: List[List[str]] = []
    outputs = vllm_model.generate(prompts, sampling_params)
    for output in outputs:
        # Extract all n completions for this prompt
        prompt_completions = [sample.text for sample in output.outputs]
        completions.append(prompt_completions)
    return completions


def write_completions_artifact(path: str, meta: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "items": items}, f, indent=2)
    print(f"Completions artifact saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate completions artifact for single model")
    parser.add_argument("--model", required=True, help="Path to model directory (HF path or local directory)")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of validation prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-vllm", action="store_true", help="Disable VLLM and use transformers")
    parser.add_argument("--output", required=False, help="Output file or directory for completions JSON. If omitted, saves to completions/<auto-name>.json")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of new tokens to generate (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling, higher values = more random (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold (default: 0.9)")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.85, help="GPU memory utilization for vLLM (default: 0.85)")
    parser.add_argument("--n-completions", type=int, default=1, help="Number of completions to generate per prompt (default: 1)")
    parser.add_argument("--dataset", type=str, default=None, help="Explicitly specify dataset (if, tldr, chat/ultrachat, storygen/sg). If not specified, auto-detects from model path.")
    args = parser.parse_args()

    use_vllm = (not args.no_vllm) and VLLM_AVAILABLE

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load prompts
    prompts, golden_completions, dataset_type, split_name = load_validation_prompts(args.num_prompts, args.seed, args.model, args.dataset)
    
    print("Running in single model mode")

    # Initialize model
    completions = []
    
    if use_vllm:
        print("Using VLLM for fast inference...")
        
        # Load tokenizer
        print("Loading tokenizer for chat template formatting...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
        
        try:
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=args.n_completions  # Generate n completions per prompt
            )
            
            # Load model directly from the specified directory
            print(f"Loading model from: {args.model}")
            vllm_model = LLM(
                model=args.model,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=args.vllm_gpu_memory,
                max_model_len=8192,
                enforce_eager=True,
                dtype="bfloat16",
            )
            
            formatted_prompts = [format_prompt(p, tokenizer, dataset_type) for p in prompts]
            completions = generate_completions_vllm(formatted_prompts, vllm_model, sampling_params, desc="Generating completions")
            
        except Exception as e:
            print(f"ERROR: vLLM failed to initialize")
            print(f"Exception: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise the exception to stop execution
    
    else:
        print("Using standard transformers inference...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
        
        for prompt in tqdm(prompts, desc="Generating completions"):
            p = format_prompt(prompt, tokenizer, dataset_type)
            prompt_completions = []
            for _ in range(args.n_completions):
                c = generate_completion(p, model, tokenizer, max_new_tokens=args.max_new_tokens, use_vllm=False, temperature=args.temperature, top_p=args.top_p)
                prompt_completions.append(c)
            completions.append(prompt_completions)

    items: List[Dict[str, Any]] = []
    for p, c in zip(prompts, completions):
        items.append({"prompt": p, "completions": c})

    meta: Dict[str, Any] = {
        "dataset": dataset_type,
        "split": split_name,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "use_vllm": use_vllm,
        "n_completions": args.n_completions,
        "generation_params": {
            "temperature": args.temperature, 
            "top_p": args.top_p, 
            "max_tokens" if use_vllm else "max_new_tokens": args.max_new_tokens
        },
        "model_path": args.model,
    }

    # Determine output path and auto-generate a descriptive filename without org prefix
    def repo_name(model_path: str) -> str:
        # Use only the repository/directory name (strip organization like "org/" or get last dir)
        return model_path.rstrip("/").split("/")[-1]
    
    def extract_base_model_name(model_path: str, temperature: float = None) -> str:
        # Extract base model name by removing checkpoint suffix
        # e.g., "model-checkpoint25" -> "model"
        import re
        name = repo_name(model_path)
        # Remove patterns like "-checkpoint25", "-checkpoint-25", "_checkpoint25", etc.
        base_name = re.sub(r'[-_]checkpoint[-_]?\d+$', '', name, flags=re.IGNORECASE)
        # Append temperature if provided
        if temperature is not None:
            base_name = f"{base_name}_temp{temperature:.1f}"
        return base_name

    model_name = repo_name(args.model)
    base_model_name = extract_base_model_name(args.model, args.temperature)
    auto_filename = f"{model_name}_{args.num_prompts}prompts_{args.n_completions}completions_seed{args.seed}_temp{args.temperature:.1f}.json"
    
    if args.output:
        # If output looks like a file (endswith .json), use it; otherwise treat as directory
        output_path = args.output if args.output.endswith(".json") else os.path.join(args.output, auto_filename)
    else:
        # Choose directory based on n_completions
        if args.n_completions == 1:
            output_dir = "completions"
        else:
            output_dir = f"completions_n{args.n_completions}"

        # Add subfolder with base model name
        output_dir = os.path.join(output_dir, base_model_name)
        
        output_path = os.path.join(output_dir, auto_filename)

    write_completions_artifact(output_path, meta, items)
    
    # Save golden completions for tldr dataset
    if dataset_type == "tldr" and golden_completions is not None:
        golden_items: List[Dict[str, Any]] = []
        for p, g in zip(prompts, golden_completions):
            golden_items.append({"prompt": p, "completions": [g]})
        
        golden_meta: Dict[str, Any] = {
            "is_golden": True,
            "dataset": dataset_type,
            "split": split_name,
            "num_prompts": args.num_prompts,
            "seed": args.seed,
            "model_path": None,
        }
        
        # Generate golden filename: golden_{num_prompts}prompts_seed{seed}.json
        golden_filename = f"golden_{args.num_prompts}prompts_seed{args.seed}.json"
        
        # Use same directory as model completions
        golden_output_path = os.path.join(os.path.dirname(output_path), golden_filename)
        
        write_completions_artifact(golden_output_path, golden_meta, golden_items)


if __name__ == "__main__":
    main()


