#!/usr/bin/env python3
"""
Generation-only CLI that mirrors the generation stage of evaluate/majority_vote.py
without importing or modifying it. Produces a reusable completions artifact JSON
containing either single model completions {prompt, completion} or dual model completions {prompt, completion1, completion2}.

example usage (dual model):
python evaluate/generate_completions.py \
    --model1 choiqs/Qwen3-1.7B-sg-bsz128-ranking-skywork8b-seed42-lr2e-6-checkpoint200 \
    --model2 choiqs/Qwen3-1.7B-sg-bsz128-regular-skywork8b-seed42-lr2e-6-checkpoint200 \
    --num-prompts 100 \
    --seed 42

example usage (single model):
python evaluate/generate_completions.py \
    --model1 choiqs/Qwen3-1.7B-sg-bsz128-ranking-skywork8b-seed42-lr2e-6-checkpoint200 \
    --num-prompts 100 \
    --seed 42
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any
import gc
import torch
import signal
import threading
import time

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


def timeout_handler(signum, frame):
    """Handle timeout signal by force killing the process"""
    print(f"\nTimeout reached! Force killing process to avoid VLLM cleanup issues...")
    os._exit(0)


def setup_timeout(timeout_seconds=300):
    """Set up a timeout mechanism that will kill the process after timeout_seconds"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    print(f"Timeout set to {timeout_seconds} seconds. Process will be automatically killed if it takes longer.")


def load_validation_prompts(num_prompts: int, seed: int, model1_path: str = None, model2_path: str = None) -> tuple[List[str], str, str]:
    print(f"Loading {num_prompts} validation prompts...")
    
    # Determine dataset based on model names
    dataset_name = "HuggingFaceH4/ultrachat_200k"  # default
    split_name = "test_sft"
    prompt_column = "messages"
    dataset_type = "ultrachat"
    
    # Determine dataset based on model names (check both models if available)
    if model1_path:
        if "tldr" in model1_path.lower():
            dataset_name = "trl-lib/tldr"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "tldr"
            print("Detected tldr model, using trl-lib/tldr dataset")
        elif "if" in model1_path.lower():
            dataset_name = "friendshipkim/IF-Datasets-Tulu-IFEval"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "if"
            print("Detected IF model, using friendshipkim/IF-Datasets-Tulu-IFEval dataset")
        elif "sg" in model1_path.lower():
            dataset_name = "friendshipkim/RUCAIBox-Story-Generation-test"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "storygen"
            print("Detected StoryGen model, using friendshipkim/RUCAIBox-Story-Generation-test dataset")
        else:
            print("Using default ultrachat_200k dataset")
    elif model2_path:
        if "tldr" in model2_path.lower():
            dataset_name = "trl-lib/tldr"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "tldr"
            print("Detected tldr model, using trl-lib/tldr dataset")
        elif "if" in model2_path.lower():
            dataset_name = "friendshipkim/IF-Datasets-Tulu-IFEval"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "if"
            print("Detected IF model, using friendshipkim/IF-Datasets-Tulu-IFEval dataset")
        elif "sg" in model2_path.lower():
            dataset_name = "friendshipkim/RUCAIBox-Story-Generation-test"
            split_name = "test"
            prompt_column = "prompt"
            dataset_type = "storygen"
            print("Detected StoryGen model, using friendshipkim/RUCAIBox-Story-Generation-test dataset")
        else:
            print("Using default ultrachat_200k dataset")
    
    dataset = load_dataset(dataset_name)
    print(f"Using {split_name} split to ensure no training data overlap")
    
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
        # For tldr format
        validation_prompts = list(dataset[split_name]["prompt"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(validation_prompts)
    validation_prompts = validation_prompts[:num_prompts]
    print(f"Loaded {len(validation_prompts)} validation prompts from {split_name} split of {dataset_name}")
    return validation_prompts, dataset_type, split_name


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


def generate_completions_vllm(prompts: List[str], vllm_model, sampling_params: Any, desc: str) -> List[str]:
    completions: List[str] = []
    outputs = vllm_model.generate(prompts, sampling_params)
    for output in outputs:
        completions.append(output.outputs[0].text)
    return completions


def write_completions_artifact(path: str, meta: Dict[str, Any], items: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "items": items}, f, indent=2)
    print(f"Completions artifact saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate completions artifact for majority voting")
    parser.add_argument("--model1", required=True, help="HF path to first model")
    parser.add_argument("--model2", required=False, help="HF path to second model (optional for single model generation)")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of validation prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-vllm", action="store_true", help="Disable VLLM and use transformers")
    parser.add_argument("--output", required=False, help="Output file or directory for completions JSON. If omitted, saves to completions/<auto-name>.json")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of new tokens to generate (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling, higher values = more random (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold (default: 0.9)")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.85, help="GPU memory utilization for vLLM (default: 0.85)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds before force killing process (default: 300)")
    args = parser.parse_args()

    # Set up timeout mechanism to prevent VLLM cleanup hanging
    setup_timeout(args.timeout)

    use_vllm = (not args.no_vllm) and VLLM_AVAILABLE

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load prompts
    prompts, dataset_type, split_name = load_validation_prompts(args.num_prompts, args.seed, args.model1, args.model2)
    
    # Determine if this is single or dual model generation
    single_model = args.model2 is None
    print(f"Running in {'single' if single_model else 'dual'} model mode")

    # Initialize models mirroring majority_vote.py
    completions1 = []
    completions2 = [] if not single_model else None
    
    if use_vllm:
        print("Using VLLM for fast inference with sequential loading...")
        
        # Load tokenizers for chat template formatting
        print("Loading tokenizers for chat template formatting...")
        tokenizer1 = AutoTokenizer.from_pretrained(args.model1, trust_remote_code=True)
        tokenizer2 = AutoTokenizer.from_pretrained(args.model2, trust_remote_code=True) if not single_model else None
        
        try:
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens
            )
            
            # Sequential loading: Load one model at a time (safer and more efficient)
            print(f"Loading model 1: {args.model1}")
            vllm_model1 = LLM(
                model=args.model1,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=args.vllm_gpu_memory,
                max_model_len=8192,
                enforce_eager=True,
                dtype="bfloat16",
            )
            
            formatted_prompts1 = [format_prompt(p, tokenizer1, dataset_type) for p in prompts]
            completions1 = generate_completions_vllm(formatted_prompts1, vllm_model1, sampling_params, desc="Generating with Model 1")
            
            if not single_model:
                print(f"Loading model 2: {args.model2}")
                vllm_model2 = LLM(
                    model=args.model2,
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=args.vllm_gpu_memory,
                    max_model_len=8192,
                    enforce_eager=True,
                    dtype="bfloat16",
                )
                
                formatted_prompts2 = [format_prompt(p, tokenizer2, dataset_type) for p in prompts]
                completions2 = generate_completions_vllm(formatted_prompts2, vllm_model2, sampling_params, desc="Generating with Model 2")
            
            # Save completions immediately after generation (before cleanup)
            print("Saving completions artifact...")
            items: List[Dict[str, str]] = []
            if single_model:
                for p, c1 in zip(prompts, completions1):
                    items.append({"prompt": p, "completion": c1})
            else:
                for p, c1, c2 in zip(prompts, completions1, completions2):
                    items.append({"prompt": p, "completion1": c1, "completion2": c2})

            meta: Dict[str, Any] = {
                "dataset": dataset_type,
                "split": split_name,
                "num_prompts": args.num_prompts,
                "seed": args.seed,
                "use_vllm": use_vllm,
                "generation_params": {
                    "temperature": args.temperature, 
                    "top_p": args.top_p, 
                    "max_tokens" if use_vllm else "max_new_tokens": args.max_new_tokens
                },
                "single_model": single_model,
            }
            
            if single_model:
                meta["model_path"] = args.model1
            else:
                meta["model1_path"] = args.model1
                meta["model2_path"] = args.model2

            # Determine output path and auto-generate a descriptive filename without org prefix
            def repo_name(model_path: str) -> str:
                # Use only the repository name (strip organization like "org/")
                return model_path.split("/")[-1]

            if single_model:
                auto_filename = f"{repo_name(args.model1)}_{args.num_prompts}prompts_seed{args.seed}.json"
            else:
                auto_filename = f"{repo_name(args.model1)}_vs_{repo_name(args.model2)}_{args.num_prompts}prompts_seed{args.seed}.json"
            if args.output:
                # If output looks like a file (endswith .json), use it; otherwise treat as directory
                output_path = args.output if args.output.endswith(".json") else os.path.join(args.output, auto_filename)
            else:
                output_path = os.path.join("completions_checkpoint", auto_filename)

            write_completions_artifact(output_path, meta, items)
            print(f"Completions artifact saved to {output_path}")
            
            # Cancel timeout since we completed successfully
            signal.alarm(0)
            print("Completions saved successfully. Force killing process to avoid VLLM cleanup issues...")
            os._exit(0)
            
        except Exception as e:
            print(f"Error initializing vLLM models: {e}")
            print("Falling back to standard transformers inference...")
            use_vllm = False
    
    else:
        print("Using standard transformers inference...")
        tokenizer1 = AutoTokenizer.from_pretrained(args.model1, trust_remote_code=True)
        model1 = AutoModelForCausalLM.from_pretrained(
            args.model1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if not single_model:
            tokenizer2 = AutoTokenizer.from_pretrained(args.model2, trust_remote_code=True)
            model2 = AutoModelForCausalLM.from_pretrained(
                args.model2,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        completions1 = []
        completions2 = [] if not single_model else None
        
        for prompt in tqdm(prompts, desc="Generating completions"):
            p1 = format_prompt(prompt, tokenizer1, dataset_type)
            c1 = generate_completion(p1, model1, tokenizer1, max_new_tokens=args.max_new_tokens, use_vllm=False, temperature=args.temperature, top_p=args.top_p)
            completions1.append(c1)
            
            if not single_model:
                p2 = format_prompt(prompt, tokenizer2, dataset_type)
                c2 = generate_completion(p2, model2, tokenizer2, max_new_tokens=args.max_new_tokens, use_vllm=False, temperature=args.temperature, top_p=args.top_p)
                completions2.append(c2)

    # File saving is now handled in the VLLM section above
    # This section is only reached for transformers fallback
    items: List[Dict[str, str]] = []
    if single_model:
        for p, c1 in zip(prompts, completions1):
            items.append({"prompt": p, "completion": c1})
    else:
        for p, c1, c2 in zip(prompts, completions1, completions2):
            items.append({"prompt": p, "completion1": c1, "completion2": c2})

    meta: Dict[str, Any] = {
        "dataset": dataset_type,
        "split": split_name,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "use_vllm": use_vllm,
        "generation_params": {
            "temperature": args.temperature, 
            "top_p": args.top_p, 
            "max_tokens" if use_vllm else "max_new_tokens": args.max_new_tokens
        },
        "single_model": single_model,
    }
    
    if single_model:
        meta["model_path"] = args.model1
    else:
        meta["model1_path"] = args.model1
        meta["model2_path"] = args.model2

    # Determine output path and auto-generate a descriptive filename without org prefix
    def repo_name(model_path: str) -> str:
        # Use only the repository name (strip organization like "org/")
        return model_path.split("/")[-1]

    if single_model:
        auto_filename = f"{repo_name(args.model1)}_{args.num_prompts}prompts_seed{args.seed}.json"
    else:
        auto_filename = f"{repo_name(args.model1)}_vs_{repo_name(args.model2)}_{args.num_prompts}prompts_seed{args.seed}.json"
    if args.output:
        # If output looks like a file (endswith .json), use it; otherwise treat as directory
        output_path = args.output if args.output.endswith(".json") else os.path.join(args.output, auto_filename)
    else:
        output_path = os.path.join("completions_checkpoint", auto_filename)

    write_completions_artifact(output_path, meta, items)
    print(f"Completions artifact saved to {output_path}")
    
    # Cancel timeout since we completed successfully
    signal.alarm(0)
    print("Completions saved successfully. Force killing process...")
    os._exit(0)


if __name__ == "__main__":
    main()


