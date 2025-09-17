import matplotlib.pyplot as plt
import numpy as np
import sys

# Set the mode - can be changed here or via command line argument
# Command line: python bar_plot_visualization.py chat  or  python bar_plot_visualization.py tldr
if len(sys.argv) > 1:
    MODE = sys.argv[1].lower()
else:
    MODE = 'chat'  # Default mode

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

# Select dataset based on mode
if MODE == 'chat':
    data = chat_data
    dataset_name = "Chat"
elif MODE == 'tldr':
    data = tldr_data
    dataset_name = "TLDR"
else:
    print(f"Invalid mode: {MODE}. Please use 'chat' or 'tldr'")
    sys.exit(1)

# Models (including Qwen3-8B but will show as blank)
models = ['Qwen3-1.7B', 'Llama3.2-3B', 'Qwen3-4B', 'Qwen3-8B']
judges = ['gpt5', 'gpt4-o', 'sonnet-3.5']

# Set up the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Set the width of the bars and spacing
bar_width = 0.12
group_width = 0.8  # Total width for each model group
spacing = 0.2  # Space between model groups

# Calculate positions for each model group
num_models = len(models)
x_positions = []
judge_labels = []
model_labels = []

for i, model in enumerate(models):
    # Calculate base position for this model group
    base_x = i * (group_width + spacing)
    
    # Create positions for win/lose bars for each judge
    for j, judge in enumerate(judges):
        # Position for win bar
        win_x = base_x + j * (bar_width * 2 + 0.05)
        # Position for lose bar  
        lose_x = win_x + bar_width
        
        x_positions.append((win_x, lose_x))
        
        # Add judge label position (center of the win/lose pair)
        judge_labels.append((win_x + bar_width/2, judge))
    
    # Add model label (center of the entire model group)
    model_labels.append((base_x + group_width/2, model))

# Colors
win_color = '#800000'  # Maroon
lose_color = '#000080'  # Navy
judge_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # For judge labels

# Create bars
bar_idx = 0
for i, model in enumerate(models):
    for j, judge in enumerate(judges):
        data_pair = data[judge][model]
        win_x, lose_x = x_positions[bar_idx]
        
        if data_pair is not None:
            win_rate, lose_rate = data_pair
            
            # Create win and lose bars
            ax.bar(win_x, win_rate, bar_width, color=win_color, alpha=0.8, 
                   label='Win' if i == 0 and j == 0 else "")
            ax.bar(lose_x, lose_rate, bar_width, color=lose_color, alpha=0.8,
                   label='Lose' if i == 0 and j == 0 else "")
            
            # Add value labels
            ax.text(win_x, win_rate + 0.01, f'{win_rate:.2f}', 
                   ha='center', va='bottom', fontsize=12)
            ax.text(lose_x, lose_rate + 0.01, f'{lose_rate:.2f}', 
                   ha='center', va='bottom', fontsize=12)
        else:
            # For 8B model, create invisible bars to maintain spacing
            ax.bar(win_x, 0, bar_width, color='white', alpha=0, edgecolor='none')
            ax.bar(lose_x, 0, bar_width, color='white', alpha=0, edgecolor='none')
        
        bar_idx += 1

# Customize the plot
ax.set_xlabel('Models', fontsize=16)
ax.set_ylabel('Rate', fontsize=16)
ax.set_title(f'Model Performance: Win vs Lose Rates by Different Judges - {dataset_name} Dataset', fontsize=18, fontweight='bold')

# Add judge labels under each bar pair
for pos, judge in judge_labels:
    ax.text(pos, -0.05, judge, ha='center', va='top', fontsize=12, 
            rotation=0, color=judge_colors[judges.index(judge)])

# Add model labels at the bottom
for pos, model in model_labels:
    ax.text(pos, -0.12, model, ha='center', va='top', fontsize=14, 
            fontweight='bold', rotation=0)

# Remove default x-axis ticks and labels
ax.set_xticks([])
ax.set_xticklabels([])

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=win_color, label='Win Rate'),
    Patch(facecolor=lose_color, label='Lose Rate')
]
ax.legend(handles=legend_elements, loc='upper right')

ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(-0.15, 1.0)  # Extended to accommodate labels

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(f'model_performance_comparison_{MODE}.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print(f"Bar plot created and saved as 'model_performance_comparison_{MODE}.png'")
print("\nData summary:")
for judge in judges:
    print(f"\n{judge}:")
    for model in models:
        value_pair = data[judge][model]
        if value_pair is not None:
            win_rate, lose_rate = value_pair
            print(f"  {model}: Win={win_rate:.2f}, Lose={lose_rate:.2f}")
        else:
            print(f"  {model}: No data")