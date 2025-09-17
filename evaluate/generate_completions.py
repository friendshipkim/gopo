#!/usr/bin/env python3
"""
Generation-only CLI that mirrors the generation stage of evaluate/majority_vote.py
without importing or modifying it. Produces a reusable completions artifact JSON
containing ordered triples of {prompt, completion1, completion2}.
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Optional VLLM for fast inference (mirror majority_vote.py behavior)
try:
    from vllm import LLM, SamplingParams
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


def load_validation_prompts(num_prompts: int, seed: int) -> List[str]:
    print(f"Loading {num_prompts} validation prompts...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    split_name = "test_sft"
    print(f"Using {split_name} split to ensure no training data overlap")
    prompts = dataset[split_name]["messages"]

    validation_prompts: List[str] = []
    for example in prompts:
        for message in example:
            if message["role"] == "user":
                validation_prompts.append(message["content"])
                break

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(validation_prompts)
    validation_prompts = validation_prompts[:num_prompts]
    print(f"Loaded {len(validation_prompts)} validation prompts from {split_name} split")
    return validation_prompts


def format_prompt(user_message: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    if tokenizer is None:
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_completion(prompt: str, model, tokenizer, max_new_tokens: int = 1024, use_vllm: bool = False, sampling_params: Any = None) -> str:
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
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_completions_batch(prompts: List[str], vllm_model, sampling_params: Any, desc: str) -> List[str]:
    completions: List[str] = []
    for i in tqdm(range(0, len(prompts), 8), desc=desc):
        batch_prompts = prompts[i : i + 8]
        outputs = vllm_model.generate(batch_prompts, sampling_params)
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
    parser.add_argument("--model2", required=True, help="HF path to second model")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of validation prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-vllm", action="store_true", help="Disable VLLM and use transformers")
    parser.add_argument("--output", required=False, help="Output file or directory for completions JSON. If omitted, saves to completions/<auto-name>.json")
    args = parser.parse_args()

    use_vllm = (not args.no_vllm) and VLLM_AVAILABLE

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load prompts
    prompts = load_validation_prompts(args.num_prompts, args.seed)

    # Initialize models mirroring majority_vote.py
    if use_vllm:
        print("Using VLLM for fast inference...")
        vllm_model1 = LLM(
            model=args.model1,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )
        vllm_model2 = LLM(
            model=args.model2,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)
        formatted_prompts1 = [format_prompt(p, None) for p in prompts]
        formatted_prompts2 = [format_prompt(p, None) for p in prompts]
        completions1 = generate_completions_batch(formatted_prompts1, vllm_model1, sampling_params, desc="Generating with Model 1")
        completions2 = generate_completions_batch(formatted_prompts2, vllm_model2, sampling_params, desc="Generating with Model 2")
    else:
        print("Using standard transformers inference...")
        tokenizer1 = AutoTokenizer.from_pretrained(args.model1, trust_remote_code=True)
        model1 = AutoModelForCausalLM.from_pretrained(
            args.model1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer2 = AutoTokenizer.from_pretrained(args.model2, trust_remote_code=True)
        model2 = AutoModelForCausalLM.from_pretrained(
            args.model2,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        completions1 = []
        completions2 = []
        for prompt in tqdm(prompts, desc="Generating completions"):
            p1 = format_prompt(prompt, tokenizer1)
            c1 = generate_completion(p1, model1, tokenizer1, use_vllm=False)
            p2 = format_prompt(prompt, tokenizer2)
            c2 = generate_completion(p2, model2, tokenizer2, use_vllm=False)
            completions1.append(c1)
            completions2.append(c2)

    items: List[Dict[str, str]] = []
    for p, c1, c2 in zip(prompts, completions1, completions2):
        items.append({"prompt": p, "completion1": c1, "completion2": c2})

    meta: Dict[str, Any] = {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "split": "test_sft",
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "model1_path": args.model1,
        "model2_path": args.model2,
        "use_vllm": use_vllm,
        "generation_params": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 1024},
    }

    # Determine output path and auto-generate a descriptive filename without org prefix
    def repo_name(model_path: str) -> str:
        # Use only the repository name (strip organization like "org/")
        return model_path.split("/")[-1]

    auto_filename = f"{repo_name(args.model1)}_vs_{repo_name(args.model2)}_{args.num_prompts}prompts_seed{args.seed}.json"
    if args.output:
        # If output looks like a file (endswith .json), use it; otherwise treat as directory
        output_path = args.output if args.output.endswith(".json") else os.path.join(args.output, auto_filename)
    else:
        output_path = os.path.join("completions", auto_filename)

    write_completions_artifact(output_path, meta, items)


if __name__ == "__main__":
    main()


