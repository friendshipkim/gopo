#!/usr/bin/env python3
"""
Plot win/lose ratios for Rank vs Z-score 300 comparisons across all tasks
Based on the data from the evaluation results table
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the table (excluding rank50 vs z-score300 as requested)
checkpoints = [100, 150, 200, 250, 300]

# Win/lose ratios (win / lose) for all three tasks
chat_ratios = [
    45.6 / 54.4,  # Rank 100 vs Z-score 300
    52.6 / 47.0,  # Rank 150 vs Z-score 300
    61.0 / 39.0,  # Rank 200 vs Z-score 300
    61.4 / 38.6,  # Rank 250 vs Z-score 300
    60.4 / 39.6   # Rank 300 vs Z-score 300
]

tldr_ratios = [
    45.6 / 54.0,  # Rank 100 vs Z-score 300
    45.6 / 54.0,  # Rank 150 vs Z-score 300
    50.4 / 49.6,  # Rank 200 vs Z-score 300
    55.0 / 45.0,  # Rank 250 vs Z-score 300
    52.6 / 47.2   # Rank 300 vs Z-score 300
]

ifeval_ratios = [
    47.8 / 49.6,  # Rank 100 vs Z-score 300
    51.0 / 46.4,  # Rank 150 vs Z-score 300
    58.2 / 39.2,  # Rank 200 vs Z-score 300
    53.4 / 44.6,  # Rank 250 vs Z-score 300
    54.6 / 43.6   # Rank 300 vs Z-score 300
]

# Create the plot
plt.figure(figsize=(14, 10))
plt.plot(checkpoints, chat_ratios, 'o-', linewidth=3, markersize=12, color='blue', label='Chat (UltraChat)')
plt.plot(checkpoints, tldr_ratios, 's-', linewidth=3, markersize=12, color='red', label='Summarization (TLDR)')
plt.plot(checkpoints, ifeval_ratios, '^-', linewidth=3, markersize=12, color='green', label='IFeval')

# Add horizontal line at y=1 (equal performance)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Equal Performance')

# Customize the plot (even larger fonts)
plt.xlabel('Rank Checkpoint', fontsize=22)
plt.ylabel('Win/Lose Ratio', fontsize=22)
plt.title('Rank vs Z-score 300: Win/Lose Ratio by Checkpoint\n(Out-of-sample Reward Judge - All Tasks)', fontsize=26)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=18)

# Add value labels on points
for i, (x, y) in enumerate(zip(checkpoints, chat_ratios)):
    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,15), ha='center', color='blue', fontsize=16)
for i, (x, y) in enumerate(zip(checkpoints, tldr_ratios)):
    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,-20), ha='center', color='red', fontsize=16)
for i, (x, y) in enumerate(zip(checkpoints, ifeval_ratios)):
    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,15), ha='center', color='green', fontsize=16)

# Set x-axis ticks
plt.xticks(checkpoints, fontsize=20)
plt.yticks(fontsize=20)

# Add some padding
plt.tight_layout()

# Save the plot
plt.savefig('/root/gopo/figure/rank_vs_zscore_winlose_ratio_out_of_sample_reward_judge.png', dpi=300, bbox_inches='tight')
plt.show()

print("Win/Lose ratio plot saved as: rank_vs_zscore_winlose_ratio_out_of_sample_reward_judge.png")
print("\nData points:")
print("Checkpoint | Chat    | TLDR    | IFeval")
print("-" * 40)
for i, (checkpoint, chat, tldr, ifeval) in enumerate(zip(checkpoints, chat_ratios, tldr_ratios, ifeval_ratios)):
    print(f"    {checkpoint:3d}   | {chat:.3f}  | {tldr:.3f}  | {ifeval:.3f}")