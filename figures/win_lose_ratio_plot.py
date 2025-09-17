import matplotlib.pyplot as plt
import numpy as np
import sys

# Chat dataset (original)
chat_data = {
    'gpt5': {
        'Qwen3-1.7B': (0.6, 0.4),
        'Llama3.2-3B': (0.58, 0.41),
        'Qwen3-4B': (0.47, 0.53),
        'Qwen3-8B': None  # No data available (dash in table)
    },
    'gpt4-o': {
        'Qwen3-1.7B': (0.84, 0.16),
        'Llama3.2-3B': (0.76, 0.22),
        'Qwen3-4B': (0.73, 0.25),
        'Qwen3-8B': None  # No data available (dash in table)
    },
    'sonnet-3.5': {
        'Qwen3-1.7B': (0.90, 0.09),
        'Llama3.2-3B': (0.52, 0.35),
        'Qwen3-4B': (0.82, 0.06),
        'Qwen3-8B': None  # No data available (dash in table)
    }
}

# TLDR dataset (new)
tldr_data = {
    'gpt5': {
        'Qwen3-1.7B': (0.54, 0.45),
        'Llama3.2-3B': None,  # No data available (dash in table)
        'Qwen3-4B': (0.49, 0.48),
        'Qwen3-8B': (0.57, 0.4)
    },
    'gpt4-o': {
        'Qwen3-1.7B': None,  # No data available (dash in table)
        'Llama3.2-3B': None,  # No data available (dash in table)
        'Qwen3-4B': (0.71, 0.25),
        'Qwen3-8B': (0.77, 0.21)
    },
    'sonnet-3.5': {
        'Qwen3-1.7B': (0.78, 0.15),
        'Llama3.2-3B': None,  # No data available (dash in table)
        'Qwen3-4B': (0.69, 0.17),
        'Qwen3-8B': None  # No data available (dash in table)
    }
}

# Models and judges
models = ['Qwen3-1.7B', 'Llama3.2-3B', 'Qwen3-4B', 'Qwen3-8B']
judges = ['gpt5', 'gpt4-o', 'sonnet-3.5']

# Calculate win/lose ratios and their log values for both datasets
def calculate_ratios(data):
    ratios = {}
    log_ratios = {}
    for judge in judges:
        ratios[judge] = {}
        log_ratios[judge] = {}
        for model in models:
            data_pair = data[judge][model]
            if data_pair is not None:
                win_rate, lose_rate = data_pair
                if lose_rate > 0:  # Avoid division by zero
                    ratio = win_rate / lose_rate
                    ratios[judge][model] = ratio
                    log_ratios[judge][model] = np.log(ratio)
                else:
                    ratios[judge][model] = None  # Infinite ratio case
                    log_ratios[judge][model] = None
            else:
                ratios[judge][model] = None
                log_ratios[judge][model] = None
    return ratios, log_ratios

# Calculate ratios for both datasets
chat_ratios, chat_log_ratios = calculate_ratios(chat_data)
tldr_ratios, tldr_log_ratios = calculate_ratios(tldr_data)

# Set up the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Colors and markers for each judge
judge_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
judge_markers = ['o', 's', '^']

# Line styles for datasets
line_styles = ['-', '--']  # solid for chat, dashed for tldr
dataset_names = ['Chat', 'TLDR']

# Create line plots for each judge and dataset
for dataset_idx, (ratios, log_ratios, line_style) in enumerate(zip([chat_ratios, tldr_ratios], 
                                                                   [chat_log_ratios, tldr_log_ratios], 
                                                                   line_styles)):
    for j, judge in enumerate(judges):
        x_values = []
        y_values = []
        
        for i, model in enumerate(models):
            log_ratio = log_ratios[judge][model]
            
            if log_ratio is not None:
                x_values.append(i)  # Model index
                y_values.append(log_ratio)
        
        if x_values:  # Only plot if there are data points
            # Plot line with markers
            label = f'{judge} ({dataset_names[dataset_idx]})' if dataset_idx == 0 else None
            ax.plot(x_values, y_values, color=judge_colors[j], marker=judge_markers[j], 
                    linestyle=line_style, linewidth=3, markersize=10, alpha=0.8, label=label)

# Set x-axis ticks and labels
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=14)

# Add horizontal line at log(1) = 0 (neutral performance)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Neutral (log=0)')

# Customize the plot
ax.set_xlabel('Models', fontsize=16)
ax.set_ylabel('Log(Win/Lose Ratio)', fontsize=16)
ax.set_title('Model Performance: Log(Win/Lose Ratios) by Different Judges - Chat vs TLDR Datasets', fontsize=18, fontweight='bold')

# Add legend
ax.legend(loc='upper right', fontsize=12)

ax.grid(True, alpha=0.3, axis='y')

# Set y-axis limits based on log ratios from both datasets
all_log_ratios = []
for log_ratios in [chat_log_ratios, tldr_log_ratios]:
    all_log_ratios.extend([log_ratios[j][m] for j in judges for m in models if log_ratios[j][m] is not None])

if all_log_ratios:
    min_log = min(all_log_ratios)
    max_log = max(all_log_ratios)
    ax.set_ylim(min_log - 0.3, max_log + 0.5)
else:
    ax.set_ylim(-1, 1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('win_lose_ratio_combined.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Log(Win/Lose ratio) line plot created and saved as 'win_lose_ratio_combined.png'")
print("\nWin/Lose Ratios and Log Ratios Summary:")
print("\n=== CHAT DATASET ===")
for judge in judges:
    print(f"\n{judge}:")
    for model in models:
        ratio = chat_ratios[judge][model]
        log_ratio = chat_log_ratios[judge][model]
        if ratio is not None:
            print(f"  {model}: Ratio={ratio:.2f}, Log={log_ratio:.2f}")
        else:
            print(f"  {model}: No data")

print("\n=== TLDR DATASET ===")
for judge in judges:
    print(f"\n{judge}:")
    for model in models:
        ratio = tldr_ratios[judge][model]
        log_ratio = tldr_log_ratios[judge][model]
        if ratio is not None:
            print(f"  {model}: Ratio={ratio:.2f}, Log={log_ratio:.2f}")
        else:
            print(f"  {model}: No data")