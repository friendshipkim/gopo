#!/usr/bin/env python3
"""
Majority Vote Model Evaluator - Single file implementation
Handles generation + judge evaluation + majority voting workflow
"""

import os
import json
import argparse
import random
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Import VLLM for fast inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Falling back to standard transformers inference.")
    VLLM_AVAILABLE = False

# Import OpenAI and Anthropic
try:
    import openai
except ImportError:
    print("Warning: openai package not found. Please install it with: pip install openai")
    openai = None

try:
    import anthropic
except ImportError:
    print("Warning: anthropic package not found. Please install it with: pip install anthropic")
    anthropic = None


@dataclass
class ComparisonResult:
    """Result of comparing two completions side by side."""
    prompt: str
    completion1: str
    completion2: str
    winner: str  # "model1", "model2", or "tie"
    explanation: str
    model1_name: str
    model2_name: str


class MajorityVoteEvaluator:
    """Single-file evaluator that handles generation + judge evaluation + majority voting"""
    
    def __init__(self, 
                 model1_path: str,
                 model2_path: str, 
                 judge_model: str,
                 api_key: str,
                 num_prompts: int = 100,
                 num_votes: int = 3,
                 allow_ties: bool = True,
                 seed: int = 42,
                 use_vllm: bool = True):
        """
        Initialize the majority vote evaluator.
        
        Args:
            model1_path: HF path to first model
            model2_path: HF path to second model
            judge_model: Judge model (gpt-4o, claude-3-5-sonnet, etc.)
            api_key: OpenAI or Anthropic API key
            num_prompts: Number of validation prompts
            num_votes: Number of votes per comparison
            allow_ties: Whether to allow ties in evaluation
            seed: Random seed for reproducibility
            use_vllm: Whether to use VLLM for faster inference
        """
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.judge_model = judge_model
        self.api_key = api_key
        self.num_prompts = num_prompts
        self.num_votes = num_votes
        self.allow_ties = allow_ties
        self.seed = seed
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Initialize judge
        self._setup_judge()
        
        # Initialize models
        self._setup_models()
        
        # Set up system prompt for chat format
        self.system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    
    def _setup_judge(self):
        """Setup judge API client based on API key type"""
        if self.api_key.startswith("sk-ant-"):
            self.api_type = "anthropic"
            if anthropic is None:
                raise ImportError("anthropic package is required for Claude API. Install with: pip install anthropic")
            self.judge_client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.api_type = "openai"
            if openai is None:
                raise ImportError("openai package is required. Install with: pip install openai")
            self.judge_client = openai.OpenAI(api_key=self.api_key)
        
        print(f"Using {self.api_type.upper()} API with judge model: {self.judge_model}")
    
    def _setup_models(self):
        """Setup generation models"""
        if self.use_vllm:
            print("Using VLLM for fast inference...")
            # Initialize VLLM models
            self.vllm_model1 = LLM(
                model=self.model1_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=8192,
            )
            self.vllm_model2 = LLM(
                model=self.model2_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=8192,
            )
            # Set sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            )
        else:
            print("Using standard transformers inference...")
            # Load both models with transformers
            print(f"Loading model 1: {self.model1_path}")
            self.tokenizer1 = AutoTokenizer.from_pretrained(self.model1_path, trust_remote_code=True)
            self.model1 = AutoModelForCausalLM.from_pretrained(
                self.model1_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"Loading model 2: {self.model2_path}")
            self.tokenizer2 = AutoTokenizer.from_pretrained(self.model2_path, trust_remote_code=True)
            self.model2 = AutoModelForCausalLM.from_pretrained(
                self.model2_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
    
    def load_validation_prompts(self) -> List[str]:
        """Load validation prompts from dataset"""
        print(f"Loading {self.num_prompts} validation prompts...")
        
        # Load UltraChat dataset
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
        
        # Use test_sft split specifically for GRPO models
        split_name = 'test_sft'
        print(f"Using {split_name} split to ensure no training data overlap")
        
        # Get prompts from the test_sft split
        prompts = dataset[split_name]['messages']
        
        # Extract user messages (prompts) - take the first user message from each conversation
        validation_prompts = []
        for example in prompts:
            # Find the first user message
            for message in example:
                if message['role'] == 'user':
                    validation_prompts.append(message['content'])
                    break
        
        # Shuffle and take the specified number
        random.shuffle(validation_prompts)
        validation_prompts = validation_prompts[:self.num_prompts]
        
        print(f"Loaded {len(validation_prompts)} validation prompts from {split_name} split")
        return validation_prompts
    
    def format_prompt(self, user_message: str, tokenizer) -> str:
        """Format prompt for a specific model"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        if tokenizer is None:
            # For VLLM, use a simple format
            return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Apply chat template for transformers
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt
    
    def generate_completion(self, prompt: str, model, tokenizer, max_new_tokens: int = 1024) -> str:
        """Generate completion for a given prompt using specified model"""
        if self.use_vllm:
            # Use VLLM for fast inference
            outputs = model.generate([prompt], self.sampling_params)
            completion = outputs[0].outputs[0].text
        else:
            # Use standard transformers inference
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
                    eos_token_id=tokenizer.eos_token_id
                )
            
            completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return completion
    
    def generate_completions_batch(self, prompts: List[str], model_num: int = 1, batch_size: int = 8) -> List[str]:
        """Generate completions for a batch of prompts using VLLM"""
        if not self.use_vllm:
            raise ValueError("Batch generation only available with VLLM")
        
        model = self.vllm_model1 if model_num == 1 else self.vllm_model2
        completions = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating with Model {model_num}"):
            batch_prompts = prompts[i:i + batch_size]
            outputs = model.generate(batch_prompts, self.sampling_params)
            
            for output in outputs:
                completion = output.outputs[0].text
                completions.append(completion)
        
        return completions
    
    def generate_responses(self, prompts: List[str]) -> List[Dict[str, str]]:
        """Generate responses from both models"""
        print("Generating responses from both models...")
        
        completions = []
        
        if self.use_vllm:
            # Use VLLM batch generation for faster inference
            print("Using VLLM batch generation...")
            
            # Format all prompts
            formatted_prompts1 = [self.format_prompt(prompt, None) for prompt in prompts]
            formatted_prompts2 = [self.format_prompt(prompt, None) for prompt in prompts]
            
            # Generate completions in batches
            completions1 = self.generate_completions_batch(formatted_prompts1, model_num=1)
            completions2 = self.generate_completions_batch(formatted_prompts2, model_num=2)
            
            # Combine results
            for prompt, completion1, completion2 in zip(prompts, completions1, completions2):
                completions.append({
                    "prompt": prompt,
                    "completion1": completion1,
                    "completion2": completion2
                })
        else:
            # Use standard sequential generation
            for prompt in tqdm(prompts, desc="Generating completions"):
                # Generate from model 1
                prompt1 = self.format_prompt(prompt, self.tokenizer1)
                completion1 = self.generate_completion(prompt1, self.model1, self.tokenizer1)
                
                # Generate from model 2
                prompt2 = self.format_prompt(prompt, self.tokenizer2)
                completion2 = self.generate_completion(prompt2, self.model2, self.tokenizer2)
                
                completions.append({
                    "prompt": prompt,
                    "completion1": completion1,
                    "completion2": completion2
                })
        
        print(f"Generated {len(completions)} response pairs")
        return completions
    
    def call_judge_api(self, prompt: str, response1: str, response2: str, allow_ties: bool = True) -> Dict[str, Any]:
        """Single API call to judge with proper handling for each model type"""
        
        # Create comparison prompt
        if allow_ties:
            comparison_prompt = (
                f"You are an expert evaluator. Please compare two responses and determine which one is better.\n\n"
                f"Response A:\n{response1}\n\n"
                f"Response B:\n{response2}\n\n"
                "Please evaluate both responses and determine which one is better. Consider factors such as:\n"
                "- Accuracy and relevance to the (implied) query or task\n"
                "- Completeness of the response\n"
                "- Clarity and coherence\n"
                "- Helpfulness and usefulness\n\n"
                "Please provide:\n"
                "1. Which response is better (A, B, or tie)\n"
                "2. A brief explanation of why you chose that response\n\n"
                "Format your response as:\n"
                "Winner: [A/B/tie]\n"
                "Explanation: [brief explanation]"
            )
        else:
            comparison_prompt = (
                f"You are an expert evaluator. Please compare two responses and determine which one is better.\n\n"
                f"Response A:\n{response1}\n\n"
                f"Response B:\n{response2}\n\n"
                "Please evaluate both responses and determine which one is better. Consider factors such as:\n"
                "- Accuracy and relevance to the (implied) query or task\n"
                "- Completeness of the response\n"
                "- Clarity and coherence\n"
                "- Helpfulness and usefulness\n\n"
                "Please provide:\n"
                "1. Which response is better (A or B) - you must choose one\n"
                "2. A brief explanation of why you chose that response\n\n"
                "Format your response as:\n"
                "Winner: [A/B]\n"
                "Explanation: [brief explanation]"
            )
        
        try:
            if self.api_type == "anthropic":
                # Claude API
                response = self.judge_client.messages.create(
                    model=self.judge_model,
                    max_tokens=300,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": f"You are an expert evaluator. Be fair and objective in your comparison. Provide clear reasoning for your choice.\n\n{comparison_prompt}"}
                    ]
                )
                response_text = response.content[0].text.strip()
                
            # elif "gpt-5" in self.judge_model.lower():
            #     # GPT-5 special handling
            #     response = self.judge_client.chat.completions.create(
            #         model=self.judge_model,
            #         messages=[
            #             {"role": "system", "content": "You are an expert evaluator. Be fair and objective in your comparison. Provide clear reasoning for your choice."},
            #             {"role": "user", "content": comparison_prompt}
            #         ],
            #         max_completion_tokens=16000,
            #         response_format={"type": "text"},
            #         reasoning_effort="medium"
            #     )
            #     response_text = response.choices[0].message.content.strip()
            
            elif "gpt-5" in self.judge_model.lower():
                # Use Responses API workaround for GPT-5 empty output bug
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Be fair and objective in your comparison. Provide clear reasoning for your choice."},
                        {"role": "user", "content": comparison_prompt}
                    ],
                    # temperature=0.1,
                    # text={"format": {"type": "text"}},
                    max_completion_tokens=16000,
                    response_format={"type": "text"},
                    reasoning_effort="medium"
                )
                temperature = 0.1
                response_text = response.choices[0].message.content.strip()
                
            else:
                # Standard OpenAI models
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Be fair and objective in your comparison. Provide clear reasoning for your choice."},
                        {"role": "user", "content": comparison_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                response_text = response.choices[0].message.content.strip()
            
            # Parse the response
            if allow_ties:
                winner_match = re.search(r'Winner:\s*(A|B|tie)', response_text, re.IGNORECASE)
            else:
                winner_match = re.search(r'Winner:\s*(A|B)', response_text, re.IGNORECASE)
            
            # Determine winner
            winner = "tie"
            if winner_match:
                winner_text = winner_match.group(1).lower()
                if winner_text == "a":
                    winner = "model1"
                elif winner_text == "b":
                    winner = "model2"
                else:
                    winner = "tie"
            
            # Extract explanation
            explanation_match = re.search(r'Explanation:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            
            return {
                "winner": winner,
                "explanation": explanation,
                "raw_response": response_text
            }
            
        except Exception as e:
            print(f"Error calling judge API: {e}")
            return {
                "winner": "tie",
                "explanation": f"Error during comparison: {str(e)}",
                "raw_response": ""
            }
    
    def calculate_majority_vote(self, individual_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate majority vote from individual judgments"""
        vote_counts = {"model1": 0, "model2": 0, "tie": 0}
        
        for judgment in individual_judgments:
            vote_counts[judgment["winner"]] += 1
        
        # Determine majority winner
        max_votes = max(vote_counts.values())
        majority_winners = [k for k, v in vote_counts.items() if v == max_votes]
        
        if len(majority_winners) == 1:
            majority_winner = majority_winners[0]
        else:
            # Tie case - use the first judgment as tiebreaker
            majority_winner = individual_judgments[0]["winner"]
        
        return {
            "vote_counts": vote_counts,
            "majority_winner": majority_winner,
            "final_winner": majority_winner
        }
    
    def judge_with_majority_voting(self, completions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Judge each comparison with multiple votes and majority voting"""
        print(f"Starting judge evaluation with {self.num_votes} votes per comparison...")
        
        results = []
        
        for i, completion_data in enumerate(tqdm(completions, desc="Judging with majority voting")):
            prompt = completion_data["prompt"]
            response1 = completion_data["completion1"]
            response2 = completion_data["completion2"]
            
            try:
                # Make multiple judge calls
                individual_judgments = []
                for call_num in range(self.num_votes):
                    judgment = self.call_judge_api(prompt, response1, response2, self.allow_ties)
                    individual_judgments.append({
                        f"call_{call_num + 1}": judgment
                    })
                
                # Calculate majority voting
                judgments_only = [list(j.values())[0] for j in individual_judgments]
                majority_result = self.calculate_majority_vote(judgments_only)
                
                # Create final result
                result = {
                    "prompt": prompt,
                    "model1_completion": response1,
                    "model2_completion": response2,
                    "individual_judgments": individual_judgments,
                    "vote_counts": majority_result["vote_counts"],
                    "majority_winner": majority_result["majority_winner"],
                    "final_winner": majority_result["final_winner"],
                    "final_explanation": f"Majority of {self.num_votes} judges preferred {majority_result['final_winner']}",
                    "model1_name": "model1",
                    "model2_name": "model2"
                }
                
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(completions)} comparisons")
                
            except Exception as e:
                print(f"Error processing comparison {i}: {e}")
                continue
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze vote distribution, win rates, etc."""
        if not results:
            return {}
        
        # Count final winners
        model1_wins = sum(1 for r in results if r.get('final_winner') == 'model1')
        model2_wins = sum(1 for r in results if r.get('final_winner') == 'model2')
        ties = sum(1 for r in results if r.get('final_winner') == 'tie')
        
        # Count individual judgments
        individual_model1_wins = 0
        individual_model2_wins = 0
        individual_ties = 0
        total_individual_judgments = 0
        
        for result in results:
            if 'individual_judgments' in result:
                for judgment in result['individual_judgments']:
                    for call_data in judgment.values():
                        if call_data['winner'] == 'model1':
                            individual_model1_wins += 1
                        elif call_data['winner'] == 'model2':
                            individual_model2_wins += 1
                        else:
                            individual_ties += 1
                        total_individual_judgments += 1
        
        analysis = {
            "total_comparisons": len(results),
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "ties": ties,
            "model1_win_rate": model1_wins / len(results) if len(results) > 0 else 0,
            "model2_win_rate": model2_wins / len(results) if len(results) > 0 else 0,
            "tie_rate": ties / len(results) if len(results) > 0 else 0,
            "individual_judgments": {
                "total_judgments": total_individual_judgments,
                "model1_wins": individual_model1_wins,
                "model2_wins": individual_model2_wins,
                "ties": individual_ties,
                "model1_win_rate": individual_model1_wins / total_individual_judgments if total_individual_judgments > 0 else 0,
                "model2_win_rate": individual_model2_wins / total_individual_judgments if total_individual_judgments > 0 else 0,
                "tie_rate": individual_ties / total_individual_judgments if total_individual_judgments > 0 else 0
            }
        }
        
        return analysis
    
    def extract_model_name(self, model_path: str) -> str:
        """Extract clean model name from HF path"""
        return model_path.split("/")[-1].replace("/", "-")
    
    def clean_judge_name(self, judge_model: str) -> str:
        """Clean judge model name for filename"""
        return judge_model.replace("-", "").replace(".", "")
    
    def generate_output_filename(self) -> str:
        """Generate output filename based on arguments"""
        model1_name = self.extract_model_name(self.model1_path)
        model2_name = self.extract_model_name(self.model2_path)
        judge_name = self.clean_judge_name(self.judge_model)
        
        # Base template
        base_name = f"{model1_name}_vs_{model2_name}_{judge_name}_{self.num_votes}votes_{self.num_prompts}prompts_seed{self.seed}"
        
        # Add tie handling suffix
        if not self.allow_ties:
            base_name += "_noties"
        
        return f"{base_name}.json"
    
    def get_output_path(self, output_dir: str = "majority_evaluation") -> str:
        """Get full output file path"""
        os.makedirs(output_dir, exist_ok=True)
        filename = self.generate_output_filename()
        return os.path.join(output_dir, filename)
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], output_path: str):
        """Save results in the same format as existing JSON files"""
        output_data = {
            "model1_path": self.model1_path,
            "model2_path": self.model2_path,
            "judge_model": self.judge_model,
            "api_type": self.api_type,
            "seed": self.seed,
            "num_judge_calls": self.num_votes,
            "majority_voting": True,
            "allow_ties": self.allow_ties,
            "analysis": analysis,
            "results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("MAJORITY VOTE EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nModel 1: {self.model1_path}")
        print(f"Model 2: {self.model2_path}")
        print(f"Judge Model: {self.judge_model}")
        print(f"Number of votes per comparison: {self.num_votes}")
        print(f"Allow ties: {self.allow_ties}")
        print(f"Total Comparisons: {analysis['total_comparisons']}")
        
        print(f"\nFinal Results (Majority Voting):")
        print(f"  Model 1: {analysis['model1_win_rate']:.1%} ({analysis['model1_wins']} wins)")
        print(f"  Model 2: {analysis['model2_win_rate']:.1%} ({analysis['model2_wins']} wins)")
        print(f"  Ties: {analysis['tie_rate']:.1%} ({analysis['ties']} ties)")
        
        if 'individual_judgments' in analysis:
            print(f"\nIndividual Judgment Results:")
            print(f"  Total individual judgments: {analysis['individual_judgments']['total_judgments']}")
            print(f"  Model 1: {analysis['individual_judgments']['model1_win_rate']:.1%} ({analysis['individual_judgments']['model1_wins']} wins)")
            print(f"  Model 2: {analysis['individual_judgments']['model2_win_rate']:.1%} ({analysis['individual_judgments']['model2_wins']} wins)")
            print(f"  Ties: {analysis['individual_judgments']['tie_rate']:.1%} ({analysis['individual_judgments']['ties']} ties)")
        
        # Determine overall winner
        if analysis['model1_win_rate'] > analysis['model2_win_rate']:
            print(f"\n🏆 Overall Winner: Model 1")
        elif analysis['model2_win_rate'] > analysis['model1_win_rate']:
            print(f"\n🏆 Overall Winner: Model 2")
        else:
            print(f"\n🤝 Overall Result: Tie")
        
        print("="*80)
    
    def run_evaluation(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Complete workflow: generate responses → judge → majority vote"""
        
        # Step 1: Load validation prompts
        prompts = self.load_validation_prompts()
        
        # Step 2: Generate responses from both models
        completions = self.generate_responses(prompts)
        
        # Step 3: Judge with multiple votes and majority voting
        results = self.judge_with_majority_voting(completions)
        
        # Step 4: Analyze and save results
        analysis = self.analyze_results(results)
        
        # Determine output path
        if output_path is None:
            output_path = self.get_output_path()
        
        self.save_results(results, analysis, output_path)
        self.print_summary(results, analysis)
        
        return {
            "results": results,
            "analysis": analysis,
            "output_path": output_path
        }


def main():
    parser = argparse.ArgumentParser(description="Majority Vote Model Evaluation")
    parser.add_argument("--model1", required=True, help="HF path to first model")
    parser.add_argument("--model2", required=True, help="HF path to second model") 
    parser.add_argument("--judge-model", required=True, help="Judge model (gpt-4o, claude-3-5-sonnet, etc.)")
    parser.add_argument("--api-key", required=True, help="OpenAI or Anthropic API key")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of validation prompts")
    parser.add_argument("--num-votes", type=int, default=3, help="Number of votes per comparison")
    parser.add_argument("--allow-ties", action="store_true", help="Allow ties in evaluation")
    parser.add_argument("--no-ties", action="store_true", help="Disable ties in evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="majority_evaluation", help="Output directory")
    parser.add_argument("--output-file", help="Custom output filename (optional)")
    parser.add_argument("--no-vllm", action="store_true", help="Disable VLLM and use transformers")
    
    args = parser.parse_args()
    
    # Handle ties setting
    allow_ties = args.allow_ties and not args.no_ties
    
    # Create evaluator
    evaluator = MajorityVoteEvaluator(
        model1_path=args.model1,
        model2_path=args.model2,
        judge_model=args.judge_model,
        api_key=args.api_key,
        num_prompts=args.num_prompts,
        num_votes=args.num_votes,
        allow_ties=allow_ties,
        seed=args.seed,
        use_vllm=not args.no_vllm
    )
    
    # Determine output path
    if args.output_file:
        output_path = os.path.join(args.output_dir, args.output_file)
    else:
        output_path = evaluator.get_output_path(args.output_dir)
    
    # Run evaluation
    results = evaluator.run_evaluation(output_path)
    
    print(f"\nEvaluation complete! Results saved to: {results['output_path']}")


if __name__ == "__main__":
    main()