#!/usr/bin/env python3
"""
Prompt-randomized judging CLI that mirrors the judging stage of evaluate/majority_vote.py
without importing or modifying it. Reads a completions artifact JSON produced by
generate_completions.py and runs multiple judge calls with majority voting.

This version randomizes the order of completion1 and completion2 per prompt
(consistent across all votes for that prompt) to eliminate systematic bias, 
then maps the results back to the original order.
"""

import os
import json
import argparse
import re
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# OpenAI / Anthropic clients (mirroring majority_vote.py)
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


def read_completions_artifact(path: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    with open(path, "r") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    items = data.get("items", [])
    # Minimal validation
    for idx, it in enumerate(items):
        if not all(k in it for k in ("prompt", "completion1", "completion2")):
            raise ValueError(f"Item {idx} missing required keys")
        if not isinstance(it["prompt"], str) or not isinstance(it["completion1"], str) or not isinstance(it["completion2"], str):
            raise ValueError(f"Item {idx} fields must be strings")
    return meta, items


def clean_judge_name(judge_model: str) -> str:
    return judge_model.replace("-", "").replace(".", "")


def generate_output_filename(model1_path: str, model2_path: str, judge_model: str, num_votes: int, num_prompts: int, seed: int, allow_ties: bool, completions_filename: str = "", randomized: bool = False) -> str:
    def extract_name(model_path: str) -> str:
        return model_path.split("/")[-1].replace("/", "-")
    model1_name = extract_name(model1_path)
    model2_name = extract_name(model2_path)
    judge_name = clean_judge_name(judge_model)
    base = f"{model1_name}_vs_{model2_name}_{judge_name}_{num_votes}votes_{num_prompts}prompts_seed{seed}"
    if not allow_ties:
        base += "_noties"
    if "_omit_thinking" in completions_filename:
        base += "_omit_thinking"
    if "_new_instruct" in completions_filename:
        base += "_new_instruct"
    if "_reverse" in completions_filename:
        base += "_reverse"
    if randomized:
        base += "_prompt_randomized"
    return f"{base}.json"


def setup_judge_client(api_key: str):
    if api_key.startswith("sk-ant-"):
        if anthropic is None:
            raise ImportError("anthropic package is required for Claude API. Install with: pip install anthropic")
        return "anthropic", anthropic.Anthropic(api_key=api_key)
    if openai is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    return "openai", openai.OpenAI(api_key=api_key)


def call_judge(api_type: str, judge_client, judge_model: str, prompt: str, response1: str, response2: str, allow_ties: bool, order_swapped: bool = False) -> Dict[str, Any]:
    # Store original order for mapping back
    original_response1 = response1
    original_response2 = response2
    
    # Use the predetermined order (no randomization here)
    if order_swapped:
        response1, response2 = response2, response1
    
    if allow_ties:
        comparison_prompt = (
            f"You are an expert evaluator. Please compare two responses and determine which one is better.\n\n" 
            f"Question of interest:\n{prompt}\n\n"
            f"Response A:\n{response1}\n\n" 
            f"Response B:\n{response2}\n\n" 
            "Please evaluate both responses based on the following factors:\n" 
            "- Accuracy and relevance to the (implied) query or task\n" 
            "- Clarity and coherence of the content provided\n" 
            "- Helpfulness and usefulness of the information given\n" 
            "- Quality of reasoning and depth of insight in the available content\n\n" 
            "IMPORTANT: Do not penalize responses for being incomplete or truncated. Focus your evaluation on the quality, accuracy, and usefulness of the content that is actually present, regardless of whether the response appears to end mid-sentence or seems incomplete.\n\n" 
            "Please provide:\n" 
            "1. Which response is clearly better (A or B), if you cannot clearly determine which is better, give a tie\n"
            "2. Brief explanation of your reasoning\n\n" 
            "Format your response exactly as:\n" 
            "Winner: [A/B]\n"
            "Explanation: [Your brief explanation focusing on the quality of the available content]"
        )
    else:
        comparison_prompt = (
                f"You are an expert evaluator. Please compare two responses and determine which one is better.\n\n" 
                f"Question of interest:\n{prompt}\n\n"
                f"Response A:\n{response1}\n\n" 
                f"Response B:\n{response2}\n\n" 
                "Please evaluate both responses based on the following factors:\n" 
                "- Accuracy and relevance to the (implied) query or task\n" 
                "- Clarity and coherence of the content provided\n" 
                "- Helpfulness and usefulness of the information given\n" 
                "- Quality of reasoning and depth of insight in the available content\n\n" 
                "IMPORTANT: Do not penalize responses for being incomplete or truncated. Focus your evaluation on the quality, accuracy, and usefulness of the content that is actually present, regardless of whether the response appears to end mid-sentence or seems incomplete.\n\n" 
                "Please provide:\n" 
                "1. Which response is better (A or B) - you must choose one\n" 
                "2. Brief explanation of your reasoning\n\n" 
                "Format your response exactly as:\n" 
                "Winner: [A/B]\n"
                "Explanation: [Your brief explanation focusing on the quality of the available content]"
        )

    try:
        if api_type == "anthropic":
            response = judge_client.messages.create(
                model=judge_model,
                max_tokens=8192,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": comparison_prompt}
                ],
            )
            response_text = response.content[0].text.strip()
        elif "gpt-5" in judge_model.lower():
            response = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "user", "content": comparison_prompt}
                ],
                max_completion_tokens=16000,
                response_format={"type": "text"},
                reasoning_effort="medium",
            )
            response_text = response.choices[0].message.content.strip()
        else:
            response = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "user", "content": comparison_prompt}
                ],
                temperature=0.1,
                max_tokens=16000,
            )
            response_text = response.choices[0].message.content.strip()

        winner_match = re.search(r"Winner:\s*(A|B|tie)" if allow_ties else r"Winner:\s*(A|B)", response_text, re.IGNORECASE)
        judge_winner = "tie"
        if winner_match:
            judge_winner = winner_match.group(1).lower()

        # Re-ask once with a stricter prompt when ties are not allowed and parsing didn't yield A/B
        if not allow_ties and judge_winner not in ("a", "b"):
            strict_prompt = (
                comparison_prompt
                + "\n\nSTRICT REQUIREMENT: You MUST choose either A or B. Do NOT output 'tie' under any circumstances."
                + " Respond EXACTLY in this format (no extra text):\nWinner: [A/B]\nExplanation: [brief reason]"
            )

            if api_type == "anthropic":
                response = judge_client.messages.create(
                    model=judge_model,
                    max_tokens=8192,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": strict_prompt}
                    ],
                )
                response_text = response.content[0].text.strip()
            elif "gpt-5" in judge_model.lower():
                response = judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "user", "content": strict_prompt}
                    ],
                    max_completion_tokens=16000,
                    response_format={"type": "text"},
                    reasoning_effort="medium",
                )
                response_text = response.choices[0].message.content.strip()
            else:
                response = judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "user", "content": strict_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=16000,
                )
                response_text = response.choices[0].message.content.strip()

            winner_match = re.search(r"Winner:\s*(A|B)", response_text, re.IGNORECASE)
            if winner_match:
                judge_winner = winner_match.group(1).lower()
        
        # Map judge's response back to original order
        winner = "tie"
        if judge_winner == "a":
            if order_swapped:
                winner = "model2"  # A was actually original_response2
            else:
                winner = "model1"  # A was original_response1
        elif judge_winner == "b":
            if order_swapped:
                winner = "model1"  # B was actually original_response1
            else:
                winner = "model2"  # B was original_response2
        else:
            winner = "tie"

        explanation_match = re.search(r"Explanation:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        return {
            "winner": winner, 
            "explanation": explanation, 
            "raw_response": response_text,
            "order_swapped": order_swapped,
            "judge_winner": judge_winner
        }
    except Exception as e:
        print(f"Error calling judge API: {e}")
        return {
            "winner": "tie", 
            "explanation": f"Error during comparison: {str(e)}", 
            "raw_response": "",
            "order_swapped": order_swapped,
            "judge_winner": "error"
        }


def calculate_majority_vote(individual_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    vote_counts = {"model1": 0, "model2": 0, "tie": 0}
    for j in individual_judgments:
        vote_counts[j["winner"]] += 1
    max_votes = max(vote_counts.values())
    majority_winners = [k for k, v in vote_counts.items() if v == max_votes]
    majority_winner = majority_winners[0] if len(majority_winners) == 1 else individual_judgments[0]["winner"]
    return {"vote_counts": vote_counts, "majority_winner": majority_winner, "final_winner": majority_winner}


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    model1_wins = sum(1 for r in results if r.get("final_winner") == "model1")
    model2_wins = sum(1 for r in results if r.get("final_winner") == "model2")
    ties = sum(1 for r in results if r.get("final_winner") == "tie")
    individual_model1_wins = 0
    individual_model2_wins = 0
    individual_ties = 0
    total_individual = 0
    for r in results:
        if "individual_judgments" in r:
            for j in r["individual_judgments"]:
                for call_data in j.values():
                    if call_data["winner"] == "model1":
                        individual_model1_wins += 1
                    elif call_data["winner"] == "model2":
                        individual_model2_wins += 1
                    else:
                        individual_ties += 1
                    total_individual += 1
    return {
        "total_comparisons": len(results),
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "ties": ties,
        "model1_win_rate": model1_wins / len(results) if len(results) > 0 else 0,
        "model2_win_rate": model2_wins / len(results) if len(results) > 0 else 0,
        "tie_rate": ties / len(results) if len(results) > 0 else 0,
        "individual_judgments": {
            "total_judgments": total_individual,
            "model1_wins": individual_model1_wins,
            "model2_wins": individual_model2_wins,
            "ties": individual_ties,
            "model1_win_rate": individual_model1_wins / total_individual if total_individual > 0 else 0,
            "model2_win_rate": individual_model2_wins / total_individual if total_individual > 0 else 0,
            "tie_rate": individual_ties / total_individual if total_individual > 0 else 0,
        },
    }


def save_results(output_path: str, model1_path: str, model2_path: str, judge_model: str, api_type: str, seed: int, num_votes: int, allow_ties: bool, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "model1_path": model1_path,
        "model2_path": model2_path,
        "judge_model": judge_model,
        "api_type": api_type,
        "seed": seed,
        "num_judge_calls": num_votes,
        "majority_voting": True,
        "allow_ties": allow_ties,
        "analysis": analysis,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {output_path}")


def print_summary(model1_path: str, model2_path: str, judge_model: str, num_votes: int, allow_ties: bool, analysis: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("PROMPT-RANDOMIZED MAJORITY VOTE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nModel 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Judge Model: {judge_model}")
    print(f"Number of votes per comparison: {num_votes}")
    print(f"Allow ties: {allow_ties}")
    print(f"Total Comparisons: {analysis.get('total_comparisons', 0)}")
    print(f"\nFinal Results (Majority Voting):")
    print(f"  Model 1: {analysis.get('model1_win_rate', 0):.1%} ({analysis.get('model1_wins', 0)} wins)")
    print(f"  Model 2: {analysis.get('model2_win_rate', 0):.1%} ({analysis.get('model2_wins', 0)} wins)")
    print(f"  Ties: {analysis.get('tie_rate', 0):.1%} ({analysis.get('ties', 0)} ties)")
    indiv = analysis.get("individual_judgments", {})
    if indiv:
        print(f"\nIndividual Judgment Results:")
        print(f"  Total individual judgments: {indiv.get('total_judgments', 0)}")
        print(f"  Model 1: {indiv.get('model1_win_rate', 0):.1%} ({indiv.get('model1_wins', 0)} wins)")
        print(f"  Model 2: {indiv.get('model2_win_rate', 0):.1%} ({indiv.get('model2_wins', 0)} wins)")
        print(f"  Ties: {indiv.get('tie_rate', 0):.1%} ({indiv.get('ties', 0)} ties)")
    if analysis.get("model1_win_rate", 0) > analysis.get("model2_win_rate", 0):
        print("\n🏆 Overall Winner: Model 1")
    elif analysis.get("model2_win_rate", 0) > analysis.get("model1_win_rate", 0):
        print("\n🏆 Overall Winner: Model 2")
    else:
        print("\n🤝 Overall Result: Tie")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Judge from a saved completions artifact with per-prompt randomization")
    parser.add_argument("--completions", required=True, help="Path to completions artifact JSON")
    parser.add_argument("--judge-model", required=True, help="Judge model (gpt-4o, claude-3-5-sonnet, etc.)")
    parser.add_argument("--api-key", required=True, help="OpenAI or Anthropic API key")
    parser.add_argument("--num-votes", type=int, default=3, help="Number of votes per comparison")
    parser.add_argument("--allow-ties", action="store_true", help="Allow ties in evaluation")
    parser.add_argument("--no-ties", action="store_true", help="Disable ties in evaluation")
    parser.add_argument("--output-dir", default="random_evaluation", help="Directory to save results JSON")
    args = parser.parse_args()

    meta, items = read_completions_artifact(args.completions)
    allow_ties = args.allow_ties and not args.no_ties

    # Set random seed for reproducible randomization
    seed = meta.get("seed", 42)
    random.seed(seed)
    print(f"Using random seed: {seed} for per-prompt order randomization")

    api_type, judge_client = setup_judge_client(args.api_key)
    print(f"Using {api_type.upper()} API with judge model: {args.judge_model}")

    results: List[Dict[str, Any]] = []
    for idx, it in enumerate(tqdm(items, desc="Judging with per-prompt randomization")):
        prompt = it["prompt"]
        response1 = it["completion1"]
        response2 = it["completion2"]

        # Determine randomization for this prompt (consistent across all votes)
        order_swapped = random.choice([True, False])
        print(f"Prompt {idx+1}: Order {'swapped' if order_swapped else 'original'}")

        individual_judgments: List[Dict[str, Any]] = []
        for call_num in range(args.num_votes):
            j = call_judge(api_type, judge_client, args.judge_model, prompt, response1, response2, allow_ties, order_swapped=order_swapped)
            individual_judgments.append({f"call_{call_num + 1}": j})

        judgments_only = [list(j.values())[0] for j in individual_judgments]
        majority = calculate_majority_vote(judgments_only)
        result = {
            "prompt": prompt,
            "model1_completion": response1,
            "model2_completion": response2,
            "individual_judgments": individual_judgments,
            "vote_counts": majority["vote_counts"],
            "majority_winner": majority["majority_winner"],
            "final_winner": majority["final_winner"],
            "final_explanation": f"Majority of {args.num_votes} judges preferred {majority['final_winner']}",
            "model1_name": "model1",
            "model2_name": "model2",
            "order_swapped": order_swapped,  # Track the randomization for this prompt
        }
        results.append(result)

    analysis = analyze_results(results)
    filename = generate_output_filename(
        meta.get("model1_path", "model1"),
        meta.get("model2_path", "model2"),
        args.judge_model,
        args.num_votes,
        meta.get("num_prompts", len(items)),
        meta.get("seed", 42),
        allow_ties,
        args.completions,
        randomized=True,
    )
    output_path = os.path.join(args.output_dir, filename)
    save_results(
        output_path,
        meta.get("model1_path", "model1"),
        meta.get("model2_path", "model2"),
        args.judge_model,
        api_type,
        meta.get("seed", 42),
        args.num_votes,
        allow_ties,
        results,
        analysis,
    )
    print_summary(meta.get("model1_path", "model1"), meta.get("model2_path", "model2"), args.judge_model, args.num_votes, allow_ties, analysis)


if __name__ == "__main__":
    main()