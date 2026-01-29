#!/usr/bin/env python3
"""
Bootstrap-based LLM-as-judge evaluation with CLT confidence intervals.

This script implements a robust evaluation approach using bootstrap sampling to construct
confidence intervals for win rates. For each bootstrap iteration, it randomly subsamples
N prompts from the full dataset, randomizes the order of completions (consistent across
all votes for each prompt), and runs judge evaluation with majority voting.

The output contains all low-level details from each bootstrap iteration plus statistical
analysis including confidence intervals for win rates.
"""

import os
import json
import argparse
import re
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# NumPy for statistical calculations
try:
    import numpy as np
except ImportError:
    print("Warning: numpy package not found. Please install it with: pip install numpy")
    np = None

# OpenAI / Anthropic clients (mirroring existing judge implementations)
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


def read_single_completions_artifact(path: str, completion_index: int = 0) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Read a single model's completions artifact (prompt + completions/responses pairs).
    
    Args:
        path: Path to the completions file (JSON or JSONL)
        completion_index: Which completion to extract from multi-completion files (default: 0 = first)
    
    Returns:
        Tuple of (metadata dict, items list with 'completion' key)
    """
    # Handle both JSON and JSONL formats
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            # JSONL format: one JSON object per line
            lines = f.readlines()
            items = [json.loads(line) for line in lines if line.strip()]
            meta = {}  # JSONL typically doesn't have metadata
        else:
            # Standard JSON format
            data = json.load(f)
            meta = data.get("meta", {})
            items = data.get("items", [])
    
    # Process each item to extract the requested completion
    for idx, it in enumerate(items):
        if not isinstance(it.get("prompt"), str):
            raise ValueError(f"Item {idx} missing or invalid 'prompt' field")
        
        # Handle multiple possible formats
        completion_extracted = False
        
        # Format 1: "responses" key (from n30 JSONL files)
        if "responses" in it and isinstance(it["responses"], list):
            if len(it["responses"]) <= completion_index:
                raise IndexError(f"Item {idx}: Requested completion_index={completion_index} but only {len(it['responses'])} responses available")
            it["completion"] = it["responses"][completion_index]
            completion_extracted = True
        
        # Format 2: "completions" key (from generate_completions.py)
        elif "completions" in it and isinstance(it["completions"], list):
            if len(it["completions"]) <= completion_index:
                raise IndexError(f"Item {idx}: Requested completion_index={completion_index} but only {len(it['completions'])} completions available")
            it["completion"] = it["completions"][completion_index]
            completion_extracted = True
        
        # Format 3: "completion" key (legacy single completion)
        elif "completion" in it and isinstance(it["completion"], str):
            if completion_index != 0:
                raise ValueError(f"Item {idx}: Requested completion_index={completion_index} but file only has single completion")
            # Already in correct format
            completion_extracted = True
        
        if not completion_extracted:
            raise ValueError(f"Item {idx} missing completion data (expected 'completion', 'completions', or 'responses' key)")
    
    return meta, items


def merge_completions_artifacts(meta1: Dict[str, Any], items1: List[Dict[str, str]], 
                               meta2: Dict[str, Any], items2: List[Dict[str, str]]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Merge two single completions artifacts into paired format, validating prompts match."""
    if len(items1) != len(items2):
        raise ValueError(f"Mismatch in number of items: {len(items1)} vs {len(items2)}")
    
    # Validate that prompts match exactly
    for idx, (item1, item2) in enumerate(zip(items1, items2)):
        if item1["prompt"] != item2["prompt"]:
            raise ValueError(f"Prompt mismatch at index {idx}:\n"
                           f"File 1: {item1['prompt'][:100]}...\n"
                           f"File 2: {item2['prompt'][:100]}...")
    
    # Merge into paired format
    merged_items = []
    for item1, item2 in zip(items1, items2):
        merged_item = {
            "prompt": item1["prompt"],  # Same for both
            "completion1": item1["completion"],
            "completion2": item2["completion"]
        }
        merged_items.append(merged_item)
    
    # Merge metadata, preferring non-None values
    merged_meta = {}
    for key in set(meta1.keys()) | set(meta2.keys()):
        if key in meta1 and key in meta2:
            if meta1[key] != meta2[key]:
                print(f"Warning: Metadata key '{key}' differs between files. Using value from first file.")
            merged_meta[key] = meta1[key]
        elif key in meta1:
            merged_meta[key] = meta1[key]
        else:
            merged_meta[key] = meta2[key]
    
    # Add information about the two source files
    merged_meta["model1_path"] = meta1.get("model_path", "model1")
    merged_meta["model2_path"] = meta2.get("model_path", "model2")
    merged_meta["merged_from_two_files"] = True
    
    return merged_meta, merged_items


def clean_judge_name(judge_model: str) -> str:
    return judge_model.replace("-", "").replace(".", "")


def extract_dataset_from_filename(filename: str) -> str:
    """Extract dataset type from filename (if, chat, or tldr). Returns None if not found."""
    filename_lower = filename.lower()
    if "if" in filename_lower:
        return "if"
    elif "chat" in filename_lower:
        return "chat"
    elif "tldr" in filename_lower:
        return "tldr"
    return None


def extract_type_from_filename(filename: str) -> str:
    """Extract model type from filename (ranking, regular, or baseline)."""
    filename_lower = filename.lower()
    if "ranking" in filename_lower:
        return "ranking"
    elif "regular" in filename_lower:
        return "regular"
    else:
        return "baseline"


def extract_temperature_from_filename(filename: str) -> str:
    """Extract temperature value from filename (e.g., temp0.3, _temp0.3, temperature0.3). Returns None if not found."""
    filename_lower = filename.lower()
    # Look for patterns like _temp0.3, temp0.3, temperature0.3, _temperature0.3
    # Match decimal numbers (e.g., 0.3, 0.7, 1.0)
    patterns = [
        r'_temp([0-9]+\.?[0-9]*)',  # _temp0.3 or _temp1
        r'temp([0-9]+\.?[0-9]*)',   # temp0.3 or temp1 (standalone)
        r'_temperature([0-9]+\.?[0-9]*)',  # _temperature0.3
        r'temperature([0-9]+\.?[0-9]*)',   # temperature0.3 (standalone)
    ]
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            return match.group(1)
    return None


def extract_model_size_from_filename(filename: str) -> str:
    """Extract model size from filename (e.g., Qwen3-4B -> qwen4b, Llama-3.2-3B -> llama3b). Returns None if not found."""
    # Patterns for different model naming conventions
    patterns = [
        # Qwen3-4B, Qwen3-1.7B, Qwen3-8B -> qwen4b, qwen1.7b, qwen8b
        (r'[Qq]wen\d*-(\d+\.?\d*[Bb])', lambda m: f"qwen{m.group(1).lower()}"),
        # Llama-3.2-3B, Llama-3-8B -> llama3b, llama8b
        (r'[Ll]lama[^-]*-(\d+\.?\d*[Bb])', lambda m: f"llama{m.group(1).lower()}"),
        # Generic fallback: ModelName-SizeB -> modelname-sizeb
        (r'([A-Za-z]+\d*)-(\d+\.?\d*[Bb])', lambda m: f"{m.group(1).lower()}{m.group(2).lower()}"),
    ]

    for pattern, formatter in patterns:
        match = re.search(pattern, filename)
        if match:
            return formatter(match)
    return None


def determine_output_directory(model1_path: str, model2_path: str, default_output_dir: str) -> str:
    """Determine output directory based on dataset and model types in filenames.
    
    Args:
        model1_path: Path to first model's completions file
        model2_path: Path to second model's completions file
        default_output_dir: Default output directory if conditions aren't met
    
    Returns:
        Output directory path
    """
    # Extract just the filenames (without path)
    filename1 = os.path.basename(model1_path)
    filename2 = os.path.basename(model2_path)
    
    # Extract dataset from both files
    dataset1 = extract_dataset_from_filename(filename1)
    dataset2 = extract_dataset_from_filename(filename2)
    
    # Determine dataset to use
    if dataset1 is not None and dataset2 is not None:
        # Both have dataset keywords - they must match
        if dataset1 != dataset2:
            raise ValueError(
                f"Dataset mismatch: File 1 has '{dataset1}' but File 2 has '{dataset2}'. "
                f"Both files must have the same dataset type (if, chat, or tldr)."
            )
        dataset = dataset1
    elif dataset1 is not None:
        # Only file1 has dataset keyword
        dataset = dataset1
    elif dataset2 is not None:
        # Only file2 has dataset keyword
        dataset = dataset2
    else:
        # Neither file has dataset keyword - use default directory
        return default_output_dir
    
    # Extract model types
    type1 = extract_type_from_filename(filename1)
    type2 = extract_type_from_filename(filename2)

    # Extract temperatures
    temp1 = extract_temperature_from_filename(filename1)
    temp2 = extract_temperature_from_filename(filename2)

    # Extract model size
    model_size1 = extract_model_size_from_filename(filename1)
    model_size2 = extract_model_size_from_filename(filename2)

    # Use model size if available (prefer first file, fallback to second)
    model_size = model_size1 or model_size2

    # Build subfolder: {model_size}-{dataset}-{type1}-vs-{type2} or {dataset}-{type1}-vs-{type2}
    if model_size:
        subfolder = f"{model_size}-{dataset}-{type1}-vs-{type2}"
    else:
        subfolder = f"{dataset}-{type1}-vs-{type2}"
    
    # Add temperature information if available
    if temp1 is not None and temp2 is not None:
        if temp1 == temp2:
            subfolder += f"-temp{temp1}"
        else:
            subfolder += f"-temp{temp1}-temp{temp2}"
    elif temp1 is not None:
        subfolder += f"-temp{temp1}"
    elif temp2 is not None:
        subfolder += f"-temp{temp2}"
    
    return os.path.join(default_output_dir, subfolder)


def generate_output_filename(model1_path: str, model2_path: str, judge_model: str,
                           N: int, B: int, seed: int, allow_ties: bool,
                           completion_index: int = 0) -> str:
    def extract_name(model_path: str) -> str:
        # Get filename without directory, remove .jsonl/.json extension
        name = model_path.split("/")[-1].replace("/", "-")
        if name.endswith(".jsonl"):
            name = name[:-6]
        elif name.endswith(".json"):
            name = name[:-5]
        
        # Remove training hyperparameters: bsz, lr, warmup
        name = re.sub(r'-bsz\d+-', '-', name)
        name = re.sub(r'-lr[0-9e.-]+-', '-', name)
        name = re.sub(r'-warmup\d+-', '-', name)
        
        # Remove prompts and completions information
        # Pattern: _NUMBERprompts_NUMBERcompletions_seedNUMBER or _NUMBERcompletions
        name = re.sub(r'_\d+prompts_\d+completions_seed\d+', '', name)
        name = re.sub(r'_\d+completions', '', name)
        name = re.sub(r'_\d+prompts', '', name)
        
        # Clean up any double dashes that might result
        name = re.sub(r'--+', '-', name)
        name = name.strip('-')
        
        return name
    model1_name = extract_name(model1_path)
    model2_name = extract_name(model2_path)
    judge_name = clean_judge_name(judge_model)
    
    # Extract temperatures from full paths (not just filenames, to catch temp in directory names)
    temp1 = extract_temperature_from_filename(model1_path)
    temp2 = extract_temperature_from_filename(model2_path)
    
    base = f"{model1_name}_vs_{model2_name}_{judge_name}"
    
    # Add temperature information if available
    if temp1 is not None and temp2 is not None:
        if temp1 == temp2:
            base += f"_temp{temp1}"
        else:
            base += f"_temp{temp1}-temp{temp2}"
    elif temp1 is not None:
        base += f"_temp{temp1}"
    elif temp2 is not None:
        base += f"_temp{temp2}"
    
    base += f"_N{N}_B{B}_seed{seed}"
    if completion_index != 0:
        base += f"_ind{completion_index}"
    if not allow_ties:
        base += "_noties"
    base += "_bootstrap"
    return f"{base}.json"


def parse_survey_response(response_text: str, order_swapped: bool, allow_ties: bool) -> Dict[str, Any]:
    """Parse the structured survey response to extract criterion evaluations."""
    criteria = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    criterion_evaluations = {}
    
    # Set regex pattern based on whether ties are allowed
    winner_pattern = r"([ABTie]+)" if allow_ties else r"([AB])"
    
    # Parse each criterion
    for i, criterion in enumerate(criteria, 1):
        # Look for the criterion section (asterisks optional for robustness)
        pattern = rf"\*?\*?{i}\.\s*{criterion.title()}\*?\*?\s*\n.*?Winner:\s*{winner_pattern}.*?Justification:\s*([^\n]+(?:\n(?!\*?\*?{i+1}\.)[^\n]*)*)"
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            winner_raw = match.group(1).strip().upper()
            justification = match.group(2).strip()
            
            # Map winner to model names
            if winner_raw == "A":
                winner = "model2" if order_swapped else "model1"
            elif winner_raw == "B":
                winner = "model1" if order_swapped else "model2"
            else:
                # Should not happen with correct regex, but handle defensively
                if allow_ties:
                    winner = "tie"
                else:
                    # This shouldn't happen with correct regex, but if it does, treat as parsing failure
                    winner = None
            
            criterion_evaluations[criterion] = {
                "winner": winner,
                "parsing_failed": False,
                "justification": justification
            }
        else:
            # Parsing failed - set to None instead of defaulting to model1
            criterion_evaluations[criterion] = {
                "winner": None,
                "parsing_failed": True,
                "justification": "Failed to parse criterion evaluation"
            }
    
    # Calculate survey_winner from successfully parsed criteria
    successful_criteria = [eval_result for eval_result in criterion_evaluations.values() if not eval_result.get("parsing_failed", False)]
    
    if len(successful_criteria) == 0:
        # All criteria failed parsing
        survey_winner = None
        survey_calculation = {
            "model1_wins": 0,
            "model2_wins": 0,
            "tie_count": 0,
            "model1_score": 0.0,
            "model2_score": 0.0,
            "successful_criteria_count": 0
        }
    else:
        # Count wins from successfully parsed criteria
        model1_wins = sum(1 for eval_result in successful_criteria if eval_result["winner"] == "model1")
        model2_wins = sum(1 for eval_result in successful_criteria if eval_result["winner"] == "model2")
        tie_count = sum(1 for eval_result in successful_criteria if eval_result["winner"] == "tie")
        
        # Calculate scores (ties count as 0.5 each)
        model1_score = model1_wins + (tie_count * 0.5)
        model2_score = model2_wins + (tie_count * 0.5)
        
        # Determine survey winner
        if model1_score > model2_score:
            survey_winner = "model1"
        elif model2_score > model1_score:
            survey_winner = "model2"
        else:
            survey_winner = "tie"
        
        survey_calculation = {
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "tie_count": tie_count,
            "model1_score": model1_score,
            "model2_score": model2_score,
            "successful_criteria_count": len(successful_criteria)
        }
    
    # Parse overall recommendation (asterisks optional for robustness)
    overall_winner_pattern = r"([ABTie]+)" if allow_ties else r"([AB])"
    overall_pattern = rf"\*?\*?Overall Recommendation:\*?\*?\s*.*?\[?{overall_winner_pattern}\]?.*?(?:Explain your reasoning\.|Reasoning:)\s*([^\n]+(?:\n[^\n]*)*)"
    overall_match = re.search(overall_pattern, response_text, re.IGNORECASE | re.DOTALL)
    
    if overall_match:
        overall_winner_raw = overall_match.group(1).strip().upper()
        overall_justification = overall_match.group(2).strip()
        
        # Map overall winner to model names
        # Note: Overall winner does not allow ties (only model1 or model2 are valid)
        if overall_winner_raw == "A":
            overall_winner = "model2" if order_swapped else "model1"
        elif overall_winner_raw == "B":
            overall_winner = "model1" if order_swapped else "model2"
        else:
            # If regex matched but value is not A or B, treat as parsing failure
            overall_winner = None
        
        # If overall_winner is None, treat as parsing failure (even if regex matched)
        # Overall winner should only be model1 or model2
        if overall_winner is None:
            overall_parsing_failed = True
        else:
            overall_parsing_failed = False
    else:
        # Parsing failed - set to None (do NOT use criterion fallback)
        overall_winner = None
        overall_justification = "Failed to parse overall recommendation"
        overall_parsing_failed = True
    
    return {
        "criterion_evaluations": criterion_evaluations,
        "overall_winner": overall_winner,
        "overall_parsing_failed": overall_parsing_failed,
        "overall_justification": overall_justification,
        "survey_winner": survey_winner,
        "survey_calculation": survey_calculation
    }


def setup_judge_client(api_key: str):
    if api_key.startswith("sk-ant-"):
        if anthropic is None:
            raise ImportError("anthropic package is required for Claude API. Install with: pip install anthropic")
        return "anthropic", anthropic.Anthropic(api_key=api_key)
    if openai is None:
        raise ImportError("openai package is required. Install with: pip install openai")
    return "openai", openai.OpenAI(api_key=api_key)


def call_judge(api_type: str, judge_client, judge_model: str, prompt: str, response1: str, response2: str, 
               allow_ties: bool, order_swapped: bool = False) -> Dict[str, Any]:
    """Call the judge API with survey-based evaluation across 5 criteria."""
    # Store original order for mapping back
    original_response1 = response1
    original_response2 = response2
    
    # Use the predetermined order (no randomization here)
    if order_swapped:
        response1, response2 = response2, response1
    
    if allow_ties:
        winner_format = "[A/B/Tie]"
        choice_instruction = ""
    else:
        winner_format = "[A/B]"
        choice_instruction = " - YOU MUST CHOOSE ONE"
    
    comparison_prompt = f"""You are an expert evaluator tasked with comparing two responses to a given question. You will evaluate each response across five specific dimensions and determine which response performs better for each dimension.

**Question:**
{prompt}

**Response A:**
{response1}

**Response B:**
{response2}

---

Please evaluate both responses on the following five dimensions:

1. **Helpfulness**: How well does the response satisfy what the question asks for? Does it address the core needs of the prompt?

2. **Correctness**: Does the response contain factually accurate and relevant information? Are there any hallucinations, errors, or false information?

3. **Coherence**: Is the response clear, logical, and self-consistent? Does it flow well and make sense?

4. **Complexity**: What is the level of intellectual depth and sophistication? Consider vocabulary, sentence structure, and whether the response demonstrates basic or expert-level understanding.

5. **Verbosity**: Is the response appropriately concise or detailed relative to what the question asks for? Is it too brief, too verbose, or just right?

---

For each dimension, provide:
- Your assessment of Response A
- Your assessment of Response B
- Which response is better for this dimension ({winner_format.replace('[', '').replace(']', '')}){choice_instruction}
- A brief justification for your choice

Format your evaluation as follows:

**1. Helpfulness**
- Response A: [evaluation]
- Response B: [evaluation]
- Winner: {winner_format}
- Justification: [explanation]

**2. Correctness**
- Response A: [evaluation]
- Response B: [evaluation]
- Winner: {winner_format}
- Justification: [explanation]

**3. Coherence**
- Response A: [evaluation]
- Response B: [evaluation]
- Winner: {winner_format}
- Justification: [explanation]

**4. Complexity**
- Response A: [evaluation]
- Response B: [evaluation]
- Winner: {winner_format}
- Justification: [explanation]

**5. Verbosity**
- Response A: [evaluation]
- Response B: [evaluation]
- Winner: {winner_format}
- Justification: [explanation]

**Overall Recommendation:**
Based on the five dimensions above, which response would you recommend overall? {winner_format}{choice_instruction} - Explain your reasoning."""

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

        # Parse the structured survey response
        parsed_result = parse_survey_response(response_text, order_swapped, allow_ties)
        parsed_result["raw_response"] = response_text
        parsed_result["order_swapped"] = order_swapped
        
        return parsed_result
    except Exception as e:
        print(f"Error calling judge API: {e}")
        # On error, mark all as parsing failures
        return {
            "criterion_evaluations": {
                "helpfulness": {"winner": None, "parsing_failed": True, "justification": f"Error: {str(e)}"},
                "correctness": {"winner": None, "parsing_failed": True, "justification": f"Error: {str(e)}"},
                "coherence": {"winner": None, "parsing_failed": True, "justification": f"Error: {str(e)}"},
                "complexity": {"winner": None, "parsing_failed": True, "justification": f"Error: {str(e)}"},
                "verbosity": {"winner": None, "parsing_failed": True, "justification": f"Error: {str(e)}"}
            },
            "overall_winner": None,
            "overall_parsing_failed": True,
            "overall_justification": f"Error during evaluation: {str(e)}",
            "survey_winner": None,
            "survey_calculation": {
                "model1_wins": 0,
                "model2_wins": 0,
                "tie_count": 0,
                "model1_score": 0.0,
                "model2_score": 0.0,
                "successful_criteria_count": 0
            },
            "raw_response": "",
            "order_swapped": order_swapped,
            "parse_error": str(e)
        }


def calculate_majority_vote(individual_judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate majority vote from individual judgments."""
    vote_counts = {"model1": 0, "model2": 0, "tie": 0}
    for j in individual_judgments:
        vote_counts[j["winner"]] += 1
    max_votes = max(vote_counts.values())
    majority_winners = [k for k, v in vote_counts.items() if v == max_votes]
    majority_winner = majority_winners[0] if len(majority_winners) == 1 else individual_judgments[0]["winner"]
    return {"vote_counts": vote_counts, "majority_winner": majority_winner, "final_winner": majority_winner}


def analyze_bootstrap_results(bootstrap_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze bootstrap results to calculate win rate distributions and confidence intervals."""
    if not bootstrap_results:
        return {}
    
    if np is None:
        raise ImportError("numpy is required for statistical analysis. Please install it with: pip install numpy")
    
    # Calculate confidence intervals using percentile method
    def calculate_ci(data, confidence=0.95):
        alpha = 1 - confidence
        lower = np.percentile(data, (alpha/2) * 100)
        upper = np.percentile(data, (1 - alpha/2) * 100)
        return [float(lower), float(upper)]
    
    # Extract overall winner win rates (exclude iterations with 0 valid comparisons)
    model1_overall_win_rates = []
    model2_overall_win_rates = []
    overall_exclusion_rates = []
    
    # Extract survey winner win rates (exclude iterations with 0 valid comparisons)
    model1_survey_win_rates = []
    model2_survey_win_rates = []
    survey_tie_rates = []
    survey_exclusion_rates = []
    
    overall_iterations_excluded = 0
    survey_iterations_excluded = 0
    
    for iteration_result in bootstrap_results:
        analysis = iteration_result["analysis"]
        overall_analysis = analysis["overall_analysis"]
        survey_analysis = analysis["survey_analysis"]
        
        # Overall winner statistics - only include iterations with valid comparisons > 0
        if overall_analysis["valid_comparisons"] > 0:
            model1_overall_win_rates.append(overall_analysis["model1_win_rate"])
            model2_overall_win_rates.append(overall_analysis["model2_win_rate"])
            total = analysis["total_comparisons"]
            overall_exclusion_rates.append(overall_analysis["excluded"] / total if total > 0 else 0)
        else:
            overall_iterations_excluded += 1
        
        # Survey winner statistics - only include iterations with valid comparisons > 0
        if survey_analysis["valid_comparisons"] > 0:
            model1_survey_win_rates.append(survey_analysis["model1_win_rate"])
            model2_survey_win_rates.append(survey_analysis["model2_win_rate"])
            survey_tie_rates.append(survey_analysis["tie_rate"])
            total = analysis["total_comparisons"]
            survey_exclusion_rates.append(survey_analysis["excluded"] / total if total > 0 else 0)
        else:
            survey_iterations_excluded += 1
    
    # Convert to numpy arrays (may be empty if all iterations excluded)
    model1_overall_win_rates = np.array(model1_overall_win_rates) if model1_overall_win_rates else np.array([])
    model2_overall_win_rates = np.array(model2_overall_win_rates) if model2_overall_win_rates else np.array([])
    overall_exclusion_rates = np.array(overall_exclusion_rates) if overall_exclusion_rates else np.array([])
    
    model1_survey_win_rates = np.array(model1_survey_win_rates) if model1_survey_win_rates else np.array([])
    model2_survey_win_rates = np.array(model2_survey_win_rates) if model2_survey_win_rates else np.array([])
    survey_tie_rates = np.array(survey_tie_rates) if survey_tie_rates else np.array([])
    survey_exclusion_rates = np.array(survey_exclusion_rates) if survey_exclusion_rates else np.array([])
    
    # Helper function to calculate stats safely (handles empty arrays)
    def safe_stats(data, include_ci_99=True):
        if len(data) == 0:
            result = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "ci_95": [0.0, 0.0],
                "raw_values": []
            }
            if include_ci_99:
                result["ci_99"] = [0.0, 0.0]
            return result
        result = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "ci_95": calculate_ci(data, 0.95),
            "raw_values": data.tolist()
        }
        if include_ci_99:
            result["ci_99"] = calculate_ci(data, 0.99)
        return result
    
    return {
        "bootstrap_iterations": len(bootstrap_results),
        "overall_winner_analysis": {
            "iterations_with_valid_comparisons": len(model1_overall_win_rates),
            "iterations_excluded": overall_iterations_excluded,
            "model1_win_rate_distribution": safe_stats(model1_overall_win_rates, include_ci_99=True),
            "model2_win_rate_distribution": safe_stats(model2_overall_win_rates, include_ci_99=True),
            "exclusion_rate_distribution": safe_stats(overall_exclusion_rates, include_ci_99=False)
        },
        "survey_winner_analysis": {
            "iterations_with_valid_comparisons": len(model1_survey_win_rates),
            "iterations_excluded": survey_iterations_excluded,
            "model1_win_rate_distribution": safe_stats(model1_survey_win_rates, include_ci_99=True),
            "model2_win_rate_distribution": safe_stats(model2_survey_win_rates, include_ci_99=True),
            "tie_rate_distribution": safe_stats(survey_tie_rates, include_ci_99=True),
            "exclusion_rate_distribution": safe_stats(survey_exclusion_rates, include_ci_99=False)
        }
    }


def run_bootstrap_evaluation(meta1: Dict[str, Any], items1: List[Dict[str, str]], 
                           meta2: Dict[str, Any], items2: List[Dict[str, str]], 
                           N: int, B: int, judge_model: str, api_type: str, judge_client, 
                           allow_ties: bool, seed: int) -> List[Dict[str, Any]]:
    """Run bootstrap evaluation with B iterations of N subsamples each."""
    
    if np is None:
        raise ImportError("numpy is required for bootstrap sampling. Please install it with: pip install numpy")
    
    # Merge completions into paired format
    meta, all_items = merge_completions_artifacts(meta1, items1, meta2, items2)
    total_prompts = len(all_items)
    
    if N > total_prompts:
        raise ValueError(f"Subsample size N ({N}) cannot be larger than total prompts ({total_prompts})")
    
    print(f"Running bootstrap evaluation: {B} iterations of {N} subsamples from {total_prompts} total prompts")
    
    # Set random seed for reproducible bootstrap sampling
    random.seed(seed)
    np.random.seed(seed)
    
    bootstrap_results = []
    
    for b in tqdm(range(B), desc="Bootstrap iterations"):
        # Randomly sample N prompts with replacement
        subsample_indices = np.random.choice(total_prompts, size=N, replace=True)
        subsample_items = [all_items[i] for i in subsample_indices]
        
        print(f"\nBootstrap iteration {b+1}/{B}: Evaluating {N} subsampled prompts")
        
        # Run evaluation on this subsample
        iteration_results = []
        for idx, item in enumerate(tqdm(subsample_items, desc=f"  Iteration {b+1} evaluations", leave=False)):
            prompt = item["prompt"]
            response1 = item["completion1"]
            response2 = item["completion2"]
            
            # Randomize order for this prompt
            order_swapped = random.choice([True, False])
            
            # Single judgment (no multiple votes)
            judgment = call_judge(api_type, judge_client, judge_model, prompt, response1, response2, 
                                allow_ties, order_swapped=order_swapped)
            
            result = {
                "prompt": prompt,
                "model1_completion": response1,
                "model2_completion": response2,
                "judgment": judgment,
                "overall_winner": judgment.get("overall_winner"),
                "overall_parsing_failed": judgment.get("overall_parsing_failed", False),
                "survey_winner": judgment.get("survey_winner"),
                "survey_calculation": judgment.get("survey_calculation", {}),
                "criterion_evaluations": judgment.get("criterion_evaluations", {}),
                "model1_name": "model1",
                "model2_name": "model2",
                "order_swapped": order_swapped,
                "subsample_index": int(subsample_indices[idx])
            }
            iteration_results.append(result)
        
        # Analyze this iteration's results - separate for overall and survey
        # Overall winner analysis (exclude parsing failures)
        valid_overall = [r for r in iteration_results if not r.get("overall_parsing_failed", False)]
        model1_overall_wins = sum(1 for r in valid_overall if r.get("overall_winner") == "model1")
        model2_overall_wins = sum(1 for r in valid_overall if r.get("overall_winner") == "model2")
        overall_excluded = len(iteration_results) - len(valid_overall)
        
        # Survey winner analysis (exclude None)
        valid_survey = [r for r in iteration_results if r.get("survey_winner") is not None]
        model1_survey_wins = sum(1 for r in valid_survey if r.get("survey_winner") == "model1")
        model2_survey_wins = sum(1 for r in valid_survey if r.get("survey_winner") == "model2")
        survey_ties = sum(1 for r in valid_survey if r.get("survey_winner") == "tie")
        survey_excluded = len(iteration_results) - len(valid_survey)
        
        iteration_analysis = {
            "total_comparisons": len(iteration_results),
            "overall_analysis": {
                "valid_comparisons": len(valid_overall),
                "excluded": overall_excluded,
                "model1_wins": model1_overall_wins,
                "model2_wins": model2_overall_wins,
                "model1_win_rate": model1_overall_wins / len(valid_overall) if len(valid_overall) > 0 else 0,
                "model2_win_rate": model2_overall_wins / len(valid_overall) if len(valid_overall) > 0 else 0,
            },
            "survey_analysis": {
                "valid_comparisons": len(valid_survey),
                "excluded": survey_excluded,
                "model1_wins": model1_survey_wins,
                "model2_wins": model2_survey_wins,
                "ties": survey_ties,
                "model1_win_rate": model1_survey_wins / len(valid_survey) if len(valid_survey) > 0 else 0,
                "model2_win_rate": model2_survey_wins / len(valid_survey) if len(valid_survey) > 0 else 0,
                "tie_rate": survey_ties / len(valid_survey) if len(valid_survey) > 0 else 0,
            }
        }
        
        bootstrap_result = {
            "iteration": b,
            "subsample_indices": subsample_indices.tolist(),
            "analysis": iteration_analysis,
            "results": iteration_results
        }
        
        bootstrap_results.append(bootstrap_result)
        
        print(f"  Overall - Model 1: {iteration_analysis['overall_analysis']['model1_win_rate']:.1%} ({iteration_analysis['overall_analysis']['model1_wins']} wins, {overall_excluded} excluded)")
        print(f"  Overall - Model 2: {iteration_analysis['overall_analysis']['model2_win_rate']:.1%} ({iteration_analysis['overall_analysis']['model2_wins']} wins)")
        print(f"  Survey - Model 1: {iteration_analysis['survey_analysis']['model1_win_rate']:.1%} ({iteration_analysis['survey_analysis']['model1_wins']} wins, {survey_excluded} excluded)")
        print(f"  Survey - Model 2: {iteration_analysis['survey_analysis']['model2_win_rate']:.1%} ({iteration_analysis['survey_analysis']['model2_wins']} wins)")
        print(f"  Survey - Ties: {iteration_analysis['survey_analysis']['tie_rate']:.1%} ({iteration_analysis['survey_analysis']['ties']} ties)")
    
    return bootstrap_results


def save_bootstrap_results(output_path: str, model1_path: str, model2_path: str, judge_model: str, 
                          api_type: str, seed: int, N: int, B: int, allow_ties: bool, 
                          bootstrap_results: List[Dict[str, Any]], bootstrap_analysis: Dict[str, Any], 
                          completion_index: int = 0) -> None:
    """Save comprehensive bootstrap results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    payload = {
        "bootstrap_config": {
            "model1_path": model1_path,
            "model2_path": model2_path,
            "judge_model": judge_model,
            "api_type": api_type,
            "seed": seed,
            "subsample_size_N": N,
            "bootstrap_iterations_B": B,
            "allow_ties": allow_ties,
            "completion_index_used": completion_index,
            "methodology": "bootstrap_sampling_with_randomization_single_vote"
        },
        "bootstrap_analysis": bootstrap_analysis,
        "bootstrap_results": bootstrap_results
    }
    
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Bootstrap results saved to {output_path}")


def print_bootstrap_summary(model1_path: str, model2_path: str, judge_model: str, 
                           N: int, B: int, allow_ties: bool, 
                           bootstrap_analysis: Dict[str, Any]) -> None:
    """Print comprehensive summary of bootstrap evaluation results."""
    print("\n" + "=" * 100)
    print("BOOTSTRAP LLM-AS-JUDGE EVALUATION SUMMARY")
    print("=" * 100)
    print(f"\nModel 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Judge Model: {judge_model}")
    print(f"Bootstrap Configuration:")
    print(f"  - Subsample size (N): {N}")
    print(f"  - Bootstrap iterations (B): {B}")
    print(f"  - Allow ties: {allow_ties}")
    
    overall_analysis = bootstrap_analysis["overall_winner_analysis"]
    survey_analysis = bootstrap_analysis["survey_winner_analysis"]
    
    # Overall winner statistics
    print(f"\n{'='*100}")
    print("OVERALL WINNER ANALYSIS (from explicit Overall Recommendation)")
    print(f"{'='*100}")
    
    model1_overall_dist = overall_analysis["model1_win_rate_distribution"]
    model2_overall_dist = overall_analysis["model2_win_rate_distribution"]
    overall_exclusion_dist = overall_analysis["exclusion_rate_distribution"]
    
    print(f"\nModel 1 Overall Win Rate:")
    print(f"  Mean: {model1_overall_dist['mean']:.3f} ± {model1_overall_dist['std']:.3f}")
    print(f"  95% CI: [{model1_overall_dist['ci_95'][0]:.3f}, {model1_overall_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{model1_overall_dist['ci_99'][0]:.3f}, {model1_overall_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{model1_overall_dist['min']:.3f}, {model1_overall_dist['max']:.3f}]")
    
    print(f"\nModel 2 Overall Win Rate:")
    print(f"  Mean: {model2_overall_dist['mean']:.3f} ± {model2_overall_dist['std']:.3f}")
    print(f"  95% CI: [{model2_overall_dist['ci_95'][0]:.3f}, {model2_overall_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{model2_overall_dist['ci_99'][0]:.3f}, {model2_overall_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{model2_overall_dist['min']:.3f}, {model2_overall_dist['max']:.3f}]")
    
    print(f"\nOverall Parsing Failure Rate:")
    print(f"  Mean: {overall_exclusion_dist['mean']:.3f} ± {overall_exclusion_dist['std']:.3f}")
    print(f"  95% CI: [{overall_exclusion_dist['ci_95'][0]:.3f}, {overall_exclusion_dist['ci_95'][1]:.3f}]")
    print(f"  Range: [{overall_exclusion_dist['min']:.3f}, {overall_exclusion_dist['max']:.3f}]")
    
    # Determine overall winner
    if model1_overall_dist['mean'] > model2_overall_dist['mean']:
        print(f"\n🏆 Overall Winner: Model 1 (mean win rate: {model1_overall_dist['mean']:.3f})")
    elif model2_overall_dist['mean'] > model1_overall_dist['mean']:
        print(f"\n🏆 Overall Winner: Model 2 (mean win rate: {model2_overall_dist['mean']:.3f})")
    else:
        print(f"\n🤝 Overall Result: Tie (both models: {model1_overall_dist['mean']:.3f})")
    
    # Survey winner statistics
    print(f"\n{'='*100}")
    print("SURVEY WINNER ANALYSIS (from criterion majority)")
    print(f"{'='*100}")
    
    model1_survey_dist = survey_analysis["model1_win_rate_distribution"]
    model2_survey_dist = survey_analysis["model2_win_rate_distribution"]
    survey_tie_dist = survey_analysis["tie_rate_distribution"]
    survey_exclusion_dist = survey_analysis["exclusion_rate_distribution"]
    
    print(f"\nModel 1 Survey Win Rate:")
    print(f"  Mean: {model1_survey_dist['mean']:.3f} ± {model1_survey_dist['std']:.3f}")
    print(f"  95% CI: [{model1_survey_dist['ci_95'][0]:.3f}, {model1_survey_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{model1_survey_dist['ci_99'][0]:.3f}, {model1_survey_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{model1_survey_dist['min']:.3f}, {model1_survey_dist['max']:.3f}]")
    
    print(f"\nModel 2 Survey Win Rate:")
    print(f"  Mean: {model2_survey_dist['mean']:.3f} ± {model2_survey_dist['std']:.3f}")
    print(f"  95% CI: [{model2_survey_dist['ci_95'][0]:.3f}, {model2_survey_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{model2_survey_dist['ci_99'][0]:.3f}, {model2_survey_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{model2_survey_dist['min']:.3f}, {model2_survey_dist['max']:.3f}]")
    
    print(f"\nSurvey Tie Rate:")
    print(f"  Mean: {survey_tie_dist['mean']:.3f} ± {survey_tie_dist['std']:.3f}")
    print(f"  95% CI: [{survey_tie_dist['ci_95'][0]:.3f}, {survey_tie_dist['ci_95'][1]:.3f}]")
    print(f"  99% CI: [{survey_tie_dist['ci_99'][0]:.3f}, {survey_tie_dist['ci_99'][1]:.3f}]")
    print(f"  Range: [{survey_tie_dist['min']:.3f}, {survey_tie_dist['max']:.3f}]")
    
    print(f"\nSurvey Exclusion Rate (all criteria failed):")
    print(f"  Mean: {survey_exclusion_dist['mean']:.3f} ± {survey_exclusion_dist['std']:.3f}")
    print(f"  95% CI: [{survey_exclusion_dist['ci_95'][0]:.3f}, {survey_exclusion_dist['ci_95'][1]:.3f}]")
    print(f"  Range: [{survey_exclusion_dist['min']:.3f}, {survey_exclusion_dist['max']:.3f}]")
    
    # Determine survey winner
    if model1_survey_dist['mean'] > model2_survey_dist['mean']:
        print(f"\n🏆 Survey Winner: Model 1 (mean win rate: {model1_survey_dist['mean']:.3f})")
    elif model2_survey_dist['mean'] > model1_survey_dist['mean']:
        print(f"\n🏆 Survey Winner: Model 2 (mean win rate: {model2_survey_dist['mean']:.3f})")
    else:
        print(f"\n🤝 Survey Result: Tie (both models: {model1_survey_dist['mean']:.3f})")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap-based LLM-as-judge evaluation with confidence intervals")
    parser.add_argument("--completions1", required=True, help="Path to first model's completions artifact JSON/JSONL")
    parser.add_argument("--completions2", required=True, help="Path to second model's completions artifact JSON/JSONL")
    parser.add_argument("--judge-model", required=True, help="Judge model (gpt-4o, claude-3-5-sonnet, etc.)")
    parser.add_argument("--api-key", required=True, help="OpenAI or Anthropic API key")
    parser.add_argument("--N", type=int, required=True, help="Subsample size for each bootstrap iteration")
    parser.add_argument("--B", type=int, required=True, help="Number of bootstrap iterations")
    parser.add_argument("--allow-ties", action="store_true", help="Allow ties in evaluation")
    parser.add_argument("--no-ties", action="store_true", help="Disable ties in evaluation")
    parser.add_argument("--output-dir", default="evaluate", help="Directory to save results JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--completion-index", type=int, default=0, help="Which completion to use from multi-completion files (0=first, 1=second, etc.)")
    args = parser.parse_args()

    # Load both completion files
    print(f"Loading completions from: {args.completions1}")
    print(f"Using completion index: {args.completion_index} (0=first, 1=second, etc.)")
    meta1, items1 = read_single_completions_artifact(args.completions1, completion_index=args.completion_index)
    print(f"Loaded {len(items1)} completions from first file")
    
    print(f"Loading completions from: {args.completions2}")
    meta2, items2 = read_single_completions_artifact(args.completions2, completion_index=args.completion_index)
    print(f"Loaded {len(items2)} completions from second file")
    
    allow_ties = args.allow_ties and not args.no_ties
    
    # Setup judge client
    api_type, judge_client = setup_judge_client(args.api_key)
    print(f"Using {api_type.upper()} API with judge model: {args.judge_model}")
    
    # Run bootstrap evaluation
    bootstrap_results = run_bootstrap_evaluation(
        meta1, items1, meta2, items2, 
        args.N, args.B, args.judge_model, api_type, judge_client,
        allow_ties, args.seed
    )
    
    # Analyze bootstrap results
    bootstrap_analysis = analyze_bootstrap_results(bootstrap_results)
    
    # Determine output directory based on filenames
    output_dir = determine_output_directory(args.completions1, args.completions2, args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Generate output filename and save results
    filename = generate_output_filename(
        args.completions1,
        args.completions2,
        args.judge_model,
        args.N,
        args.B,
        args.seed,
        allow_ties,
        completion_index=args.completion_index
    )
    output_path = os.path.join(output_dir, filename)
    
    save_bootstrap_results(
        output_path,
        args.completions1,
        args.completions2,
        args.judge_model,
        api_type,
        args.seed,
        args.N,
        args.B,
        allow_ties,
        bootstrap_results,
        bootstrap_analysis,
        completion_index=args.completion_index
    )
    
    # Print summary
    print_bootstrap_summary(
        meta1.get("model_path", "model1"),
        meta2.get("model_path", "model2"),
        args.judge_model,
        args.N,
        args.B,
        allow_ties,
        bootstrap_analysis
    )


if __name__ == "__main__":
    main()
