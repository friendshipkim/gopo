#!/usr/bin/env python3
"""
Plot win/lose ratios for Qwen3-1.7B Rank vs Z-score 300 comparisons across all tasks
Using GPT-4o as judge - Based on the actual evaluation results table data
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the table (excluding rank50 vs z-score300 as requested)
checkpoints = [100, 150, 200, 250, 300]

# Win/lose ratios (win / lose) for all three tasks from the actual data
# Chat (UltraChat) task - Qwen3-1.7B win rates
chat_ratios = [
    47.4 / 52.6,  # Rank 100 vs Z-score 300: 47.4% win, 52.6% lose
    48.0 / 52.0,  # Rank 150 vs Z-score 300: 48.0% win, 52.0% lose
    52.6 / 47.4,  # Rank 200 vs Z-score 300: 52.6% win, 47.4% lose
    51.8 / 48.2,  # Rank 250 vs Z-score 300: 51.8% win, 48.2% lose
    50.8 / 49.2   # Rank 300 vs Z-score 300: 50.8% win, 49.2% lose
]

# Summarization (TLDR) task - Qwen3-1.7B win rates
tldr_ratios = [
    57.8 / 42.2,  # Rank 100 vs Z-score 300: 57.8% win, 42.2% lose
    65.6 / 34.4,  # Rank 150 vs Z-score 300: 65.6% win, 34.4% lose
    54.4 / 45.6,  # Rank 200 vs Z-score 300: 54.4% win, 45.6% lose
    58.0 / 42.0,  # Rank 250 vs Z-score 300: 58.0% win, 42.0% lose
    55.8 / 44.2   # Rank 300 vs Z-score 300: 55.8% win, 44.2% lose
]

# IFeval task - Qwen3-1.7B win rates
ifeval_ratios = [
    47.8 / 55.2,  # Rank 100 vs Z-score 300: 47.8% win, 55.2% lose
    53.4 / 46.4,  # Rank 150 vs Z-score 300: 53.4% win, 46.4% lose
    55.4 / 45.4,  # Rank 200 vs Z-score 300: 55.4% win, 45.4% lose
    55.2 / 44.8,  # Rank 250 vs Z-score 300: 55.2% win, 44.8% lose
    51.2 / 48.0   # Rank 300 vs Z-score 300: 51.2% win, 48.0% lose
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
plt.title('Qwen3-1.7B Rank vs Z-score 300: Win/Lose Ratio by Checkpoint\n(GPT-4o Judge - All Tasks)', fontsize=26)
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
plt.savefig('/root/gopo/figure/qwen3_rank_vs_zscore_winlose_ratio_gpt4o_judge.png', dpi=300, bbox_inches='tight')
plt.show()

print("Qwen3-1.7B Win/Lose ratio plot saved as: qwen3_rank_vs_zscore_winlose_ratio_gpt4o_judge.png")
print("\nData points:")
print("Checkpoint | Chat    | TLDR    | IFeval")
print("-" * 40)
for i, (checkpoint, chat, tldr, ifeval) in enumerate(zip(checkpoints, chat_ratios, tldr_ratios, ifeval_ratios)):
    print(f"    {checkpoint:3d}   | {chat:.3f}  | {tldr:.3f}  | {ifeval:.3f}")

print("\nWin rates (Qwen3-1.7B vs Z-score 300):")
print("Checkpoint | Chat % | TLDR % | IFeval %")
print("-" * 40)
chat_win_rates = [47.4, 48.0, 52.6, 51.8, 50.8]
tldr_win_rates = [57.8, 65.6, 54.4, 58.0, 55.8]
ifeval_win_rates = [47.8, 53.4, 55.4, 55.2, 51.2]
for i, (checkpoint, chat, tldr, ifeval) in enumerate(zip(checkpoints, chat_win_rates, tldr_win_rates, ifeval_win_rates)):
    print(f"    {checkpoint:3d}   | {chat:5.1f}% | {tldr:5.1f}% | {ifeval:5.1f}%")