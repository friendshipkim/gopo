#!/usr/bin/env python3
"""
Plot KL divergence values for Rank vs Z-score 300 comparisons across all tasks
Based on the data from the evaluation results table
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the table (excluding rank50 vs z-score300 as requested)
checkpoints = [100, 150, 200, 250, 300]

# KL divergence values for all three tasks
# Chat (UltraChat) task
chat_kl_rank = [0.0278, 0.0519, 0.0738, 0.1018, 0.1125]
chat_kl_zscore = [0.0207, 0.0350, 0.0445, 0.0527, 0.0596]

# Summarization (TLDR) task
tldr_kl_rank = [0.2302, 0.3162, 0.3856, 0.4268, 0.4681]
tldr_kl_zscore = [0.2214, 0.3415, 0.3721, 0.3939, 0.4237]

# IFeval task
ifeval_kl_rank = [0.0165, 0.0252, 0.0309, 0.0352, 0.0372]
ifeval_kl_zscore = [0.0126, 0.0184, 0.0211, 0.0221, 0.0220]

# Create the plot
plt.figure(figsize=(14, 10))

# Create subplots for each task
fig, axes = plt.subplots(2, 2, figsize=(24, 20))
fig.suptitle('KL Divergence Comparison: Rank vs Z-score Models (All Tasks)', fontsize=28)

# Chat (UltraChat) task
ax1 = axes[0, 0]
ax1.plot(checkpoints, chat_kl_rank, 'o-', linewidth=3, markersize=10, color='blue', label='Rank Model')
ax1.plot(checkpoints, chat_kl_zscore, 'o--', linewidth=3, markersize=10, color='blue', label='Z-score Model')
ax1.set_title('Chat (UltraChat)', fontsize=20)
ax1.set_xlabel('Checkpoint', fontsize=18)
ax1.set_ylabel('KL Divergence', fontsize=18)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=16)
ax1.set_xticks(checkpoints)
ax1.tick_params(axis='both', which='major', labelsize=18)

# Horizontal line at Z-score model's KL value for checkpoint 300
ax1.axhline(y=chat_kl_zscore[checkpoints.index(300)], color='black', linestyle=':', linewidth=2)

# Summarization (TLDR) task
ax2 = axes[0, 1]
ax2.plot(checkpoints, tldr_kl_rank, 's-', linewidth=3, markersize=10, color='red', label='Rank Model')
ax2.plot(checkpoints, tldr_kl_zscore, 's--', linewidth=3, markersize=10, color='red', label='Z-score Model')
ax2.set_title('Summarization (TLDR)', fontsize=20)
ax2.set_xlabel('Checkpoint', fontsize=18)
ax2.set_ylabel('KL Divergence', fontsize=18)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=16)
ax2.set_xticks(checkpoints)
ax2.tick_params(axis='both', which='major', labelsize=18)

# Horizontal line at Z-score model's KL value for checkpoint 300
ax2.axhline(y=tldr_kl_zscore[checkpoints.index(300)], color='black', linestyle=':', linewidth=2)

# IFeval task
ax3 = axes[1, 0]
ax3.plot(checkpoints, ifeval_kl_rank, '^-', linewidth=3, markersize=10, color='green', label='Rank Model')
ax3.plot(checkpoints, ifeval_kl_zscore, '^--', linewidth=3, markersize=10, color='green', label='Z-score Model')
ax3.set_title('IFeval', fontsize=20)
ax3.set_xlabel('Checkpoint', fontsize=18)
ax3.set_ylabel('KL Divergence', fontsize=18)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=16)
ax3.set_xticks(checkpoints)
ax3.tick_params(axis='both', which='major', labelsize=18)

# Horizontal line at Z-score model's KL value for checkpoint 300
ax3.axhline(y=ifeval_kl_zscore[checkpoints.index(300)], color='black', linestyle=':', linewidth=2)

# Combined plot
ax4 = axes[1, 1]
ax4.plot(checkpoints, chat_kl_rank, 'o-', linewidth=2, markersize=8, color='blue', label='Chat Rank')
ax4.plot(checkpoints, chat_kl_zscore, 'o--', linewidth=2, markersize=8, color='blue', label='Chat Z-score')
ax4.plot(checkpoints, tldr_kl_rank, 's-', linewidth=2, markersize=8, color='red', label='TLDR Rank')
ax4.plot(checkpoints, tldr_kl_zscore, 's--', linewidth=2, markersize=8, color='red', label='TLDR Z-score')
ax4.plot(checkpoints, ifeval_kl_rank, '^-', linewidth=2, markersize=8, color='green', label='IFeval Rank')
ax4.plot(checkpoints, ifeval_kl_zscore, '^--', linewidth=2, markersize=8, color='green', label='IFeval Z-score')
ax4.set_title('All Tasks Combined', fontsize=20)
ax4.set_xlabel('Checkpoint', fontsize=18)
ax4.set_ylabel('KL Divergence', fontsize=18)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=14)
ax4.set_xticks(checkpoints)
ax4.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()

# Save the plot
plt.savefig('/root/gopo/figure/rank_vs_zscore_kl_divergence.png', dpi=300, bbox_inches='tight')
plt.show()

print("KL divergence plot saved as: rank_vs_zscore_kl_divergence.png")
print("\nKL divergence values:")
print("Checkpoint | Chat Rank | Chat Z-score | TLDR Rank | TLDR Z-score | IFeval Rank | IFeval Z-score")
print("-" * 90)
for i, (checkpoint, cr, cz, tr, tz, ir, iz) in enumerate(zip(checkpoints, chat_kl_rank, chat_kl_zscore, tldr_kl_rank, tldr_kl_zscore, ifeval_kl_rank, ifeval_kl_zscore)):
    print(f"    {checkpoint:3d}   |   {cr:.4f}   |    {cz:.4f}    |   {tr:.4f}   |    {tz:.4f}    |   {ir:.4f}   |    {iz:.4f}")