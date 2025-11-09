#!/usr/bin/env python3
"""
Plot reward trajectories with confidence intervals for ranking vs regular training regimes.

For each checkpoint:
1. For each completion index (0-29), average rewards across all prompts
2. From the 30 averaged values, compute mean and confidence interval (mean ± 1.96*std)
3. Plot trajectories with confidence bands for both ranking and regular subfolders
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
BASE_DIR = Path(__file__).parent
RANKING_FOLDER = "Qwen3-1.7B-if-bsz128-ts750-ranking-skywork8b-seed42-lr1e-6-warmup10"
REGULAR_FOLDER = "Qwen3-1.7B-if-bsz128-ts750-regular-skywork8b-seed42-lr1e-6-warmup10"
FIGURES_DIR = BASE_DIR / "figures"
SKYWORK_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-8B"
QRM_MODEL = "friendshipkim/QRM-Llama3.1-8B-v2"

# Confidence interval multiplier (for 95% CI assuming normality)
CI_MULTIPLIER = 1.96


def extract_checkpoint_number(filename: str) -> int:
    """Extract checkpoint number from filename like 'checkpoint75_rewards_n30_seed42.json'"""
    # Extract number after 'checkpoint' and before '_rewards'
    return int(filename.split('checkpoint')[1].split('_rewards')[0])


def load_checkpoint_data(folder_path: Path, reward_model: str) -> Dict[int, Dict]:
    """
    Load all checkpoint JSON files from a folder.
    
    Returns:
        Dict mapping checkpoint number to data containing:
        - 'checkpoint': checkpoint number
        - 'averaged_rewards': list of 30 values (one per completion index, averaged across prompts)
    """
    checkpoint_data = {}
    
    if not folder_path.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return checkpoint_data
    
    # Find all JSON files
    json_files = list(folder_path.glob("checkpoint*_rewards_n30_seed42.json"))
    
    for json_file in json_files:
        try:
            checkpoint_num = extract_checkpoint_number(json_file.name)
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract rewards for the specified model
            num_prompts = data['meta']['num_prompts']
            n_completions = data['meta']['n_completions']
            
            # Initialize: for each completion index, collect rewards across all prompts
            rewards_by_index = [[] for _ in range(n_completions)]
            
            # Extract rewards for each prompt
            for item in data['items']:
                prompt_rewards = item['rewards'][reward_model]
                if len(prompt_rewards) != n_completions:
                    print(f"Warning: Expected {n_completions} rewards, got {len(prompt_rewards)} in {json_file}")
                    continue
                
                for idx in range(n_completions):
                    rewards_by_index[idx].append(prompt_rewards[idx])
            
            # Average across prompts for each completion index
            averaged_rewards = [np.mean(rewards_by_index[idx]) for idx in range(n_completions)]
            
            checkpoint_data[checkpoint_num] = {
                'checkpoint': checkpoint_num,
                'averaged_rewards': averaged_rewards
            }
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return checkpoint_data


def compute_trajectory_stats(checkpoint_data: Dict[int, Dict]) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Compute trajectory statistics (mean, upper CI, lower CI) for each checkpoint.
    
    Returns:
        checkpoints: sorted list of checkpoint numbers
        means: mean of the 30 averaged reward values for each checkpoint
        upper_bounds: upper CI bound (mean + 1.96*std) for each checkpoint
        lower_bounds: lower CI bound (mean - 1.96*std) for each checkpoint
    """
    checkpoints = sorted(checkpoint_data.keys())
    means = []
    upper_bounds = []
    lower_bounds = []
    
    for ckpt in checkpoints:
        averaged_rewards = checkpoint_data[ckpt]['averaged_rewards']
        
        # Compute mean and std of the 30 averaged values
        mean_val = np.mean(averaged_rewards)
        std_val = np.std(averaged_rewards, ddof=1)  # Sample std
        
        # Confidence interval
        ci_half_width = CI_MULTIPLIER * std_val
        upper_bound = mean_val + ci_half_width
        lower_bound = mean_val - ci_half_width
        
        means.append(mean_val)
        upper_bounds.append(upper_bound)
        lower_bounds.append(lower_bound)
    
    return checkpoints, means, upper_bounds, lower_bounds


def plot_trajectory(
    ranking_data: Dict[int, Dict],
    regular_data: Dict[int, Dict],
    reward_model: str,
    output_filename: str,
    ylabel: str
):
    """
    Plot reward trajectories with confidence intervals for ranking and regular regimes.
    """
    # Compute statistics for both regimes
    ranking_checkpoints, ranking_means, ranking_upper, ranking_lower = compute_trajectory_stats(ranking_data)
    regular_checkpoints, regular_means, regular_upper, regular_lower = compute_trajectory_stats(regular_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot ranking trajectory with confidence band
    ax.plot(ranking_checkpoints, ranking_means, 'o-', color='#2E86AB', label='Ranking', linewidth=2, markersize=4)
    ax.fill_between(ranking_checkpoints, ranking_lower, ranking_upper, color='#2E86AB', alpha=0.2)
    
    # Plot regular trajectory with confidence band
    ax.plot(regular_checkpoints, regular_means, 's-', color='#A23B72', label='Regular', linewidth=2, markersize=4)
    ax.fill_between(regular_checkpoints, regular_lower, regular_upper, color='#A23B72', alpha=0.2)
    
    # Formatting
    ax.set_xlabel('Checkpoint', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Reward Trajectories: {ylabel}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(exist_ok=True)
    output_path = FIGURES_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    
    plt.close()


def main():
    """Main function to generate both figures."""
    ranking_folder = BASE_DIR / RANKING_FOLDER
    regular_folder = BASE_DIR / REGULAR_FOLDER
    
    print("Loading ranking data...")
    ranking_skywork = load_checkpoint_data(ranking_folder, SKYWORK_MODEL)
    ranking_qrm = load_checkpoint_data(ranking_folder, QRM_MODEL)
    
    print("Loading regular data...")
    regular_skywork = load_checkpoint_data(regular_folder, SKYWORK_MODEL)
    regular_qrm = load_checkpoint_data(regular_folder, QRM_MODEL)
    
    print(f"\nFound {len(ranking_skywork)} ranking checkpoints")
    print(f"Found {len(regular_skywork)} regular checkpoints")
    
    # Generate Figure 1: Skywork rewards
    print("\nGenerating Figure 1: Skywork rewards...")
    plot_trajectory(
        ranking_skywork,
        regular_skywork,
        SKYWORK_MODEL,
        'skywork_reward_trajectories.png',
        'Average Skywork Reward'
    )
    
    # Also save as PDF
    plot_trajectory(
        ranking_skywork,
        regular_skywork,
        SKYWORK_MODEL,
        'skywork_reward_trajectories.pdf',
        'Average Skywork Reward'
    )
    
    # Generate Figure 2: QRM rewards
    print("\nGenerating Figure 2: QRM rewards...")
    plot_trajectory(
        ranking_qrm,
        regular_qrm,
        QRM_MODEL,
        'qrm_reward_trajectories.png',
        'Average QRM Reward'
    )
    
    # Also save as PDF
    plot_trajectory(
        ranking_qrm,
        regular_qrm,
        QRM_MODEL,
        'qrm_reward_trajectories.pdf',
        'Average QRM Reward'
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()






