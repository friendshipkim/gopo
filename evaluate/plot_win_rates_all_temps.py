#!/usr/bin/env python3
"""
Plot ranking win rates across all temperatures as bar charts.

Creates two separate figures:
1. Survey win rates (grouped bars by checkpoint, one bar per temperature)
2. Overall win rates (grouped bars by checkpoint, one bar per temperature)

Usage:
    python plot_win_rates_all_temps.py --task tldr --model qwen4b --N 15 --B 15 [--output-dir figures] [--format pdf]
    python plot_win_rates_all_temps.py --task chat --model qwen4b --N 25 --B 25 --chat-evaluator skywork

Examples:
    # Plot 4B model with N15_B15 config
    python plot_win_rates_all_temps.py --task tldr --model qwen4b --N 15 --B 15

    # Plot 1.7B model with N30_B30 config, specific temperatures
    python plot_win_rates_all_temps.py --task tldr --model qwen1.7b --N 30 --B 30 --temperatures 0.1 0.5 0.9

    # Plot multiple models with per-model configs
    python plot_win_rates_all_temps.py --task tldr --model-configs qwen1.7b:30:30 qwen4b:25:25 --temperatures 0.5

    # Plot chat task with skywork evaluator
    python plot_win_rates_all_temps.py --task chat --model qwen4b --N 25 --B 25 --chat-evaluator skywork

    # Plot chat task with qrm evaluator
    python plot_win_rates_all_temps.py --task chat --model qwen4b --N 25 --B 25 --chat-evaluator qrm
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    # Use matplotlib's built-in mathtext renderer (doesn't require LaTeX installation)
    plt.rcParams['mathtext.default'] = 'regular'
except ImportError:
    print("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy")
    exit(1)

# Import shared plot configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_config as pc

# Import functions from existing plot_win_rates.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_win_rates import (
    extract_checkpoint_from_filename,
    load_win_rate_data,
    parse_model_name
)


def extract_completion_index_from_filename(filename: str) -> int:
    """Extract completion index from filename.

    Args:
        filename: Filename to parse

    Returns:
        Completion index (0 if no _ind pattern found, otherwise the number after _ind)
    """
    # Look for _ind{N}_ pattern
    match = re.search(r'_ind(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0  # Default to index 0 (no _ind in filename)


def format_model_name(model: str) -> str:
    """Format model name for display in titles.

    Examples:
        'qwen4b' -> 'Qwen4B'
        'qwen1.7b' -> 'Qwen1.7B'
    """
    if model is None:
        return None
    # Handle patterns like 'qwen4b' or 'qwen1.7b'
    import re
    match = re.match(r'(qwen)(\d+\.?\d*)(b)', model, re.IGNORECASE)
    if match:
        return f"Qwen{match.group(2)}B"
    return model


def load_temperature_folder_data(folder_path: str, N: Optional[int] = None, B: Optional[int] = None,
                                  chat_evaluator: Optional[str] = None, task: Optional[str] = None,
                                  max_steps: Optional[int] = None,
                                  aggregate_indices: bool = False) -> Tuple[Dict[int, Dict[str, Any]], str, str]:
    """Load all checkpoint data from a temperature folder.

    Args:
        folder_path: Path to the temperature folder
        N: Filter by N value in filename (e.g., 30 for N30). If None, no filtering.
        B: Filter by B value in filename (e.g., 30 for B30). If None, no filtering.
        chat_evaluator: Filter by evaluator type for chat tasks ('skywork' or 'qrm'). Only applies if task is 'chat' or 'ultrachat'.
        task: Task name (e.g., 'tldr', 'chat', 'ultrachat'). Used to determine if chat_evaluator filtering should apply.
        max_steps: Maximum checkpoint step to include. If None, no filtering.
        aggregate_indices: If True, aggregate data across all completion indices (ind0, ind1, ind2, etc.)
                          and include min/max for confidence intervals.

    Returns:
        Tuple of (data_by_checkpoint, model1_name, model2_name)
        When aggregate_indices=True, each checkpoint's data includes '_values' lists and '_ci_low'/'_ci_high' keys.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    model1_name = None
    model2_name = None

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    if not json_files:
        raise ValueError(f"No JSON files found in {folder_path}")

    # Filter by N and B if specified (skip when aggregating indices to allow mixed N/B)
    if N is not None and B is not None and not aggregate_indices:
        pattern = f'_N{N}_B{B}_'
        json_files = [f for f in json_files if pattern in f]
        if not json_files:
            raise ValueError(f"No JSON files matching N{N}_B{B} found in {folder_path}")

    # Filter by chat evaluator if specified and task is chat/ultrachat
    if chat_evaluator is not None and task is not None:
        task_lower = task.lower()
        if task_lower in ['chat', 'ultrachat']:
            if chat_evaluator.lower() == 'skywork':
                # Match 'skywork' (will match both 'skywork' and 'skywork8b')
                json_files = [f for f in json_files if 'skywork' in f.lower()]
            elif chat_evaluator.lower() == 'qrm':
                # Match 'qrm' explicitly OR files without 'skywork' (default is qrm)
                json_files = [f for f in json_files if 'qrm' in f.lower() or 'skywork' not in f.lower()]
            else:
                raise ValueError(f"Invalid chat_evaluator '{chat_evaluator}'. Must be 'skywork' or 'qrm'")

            if not json_files:
                raise ValueError(f"No JSON files matching evaluator '{chat_evaluator}' found in {folder_path}")

    if aggregate_indices:
        # Collect data by checkpoint AND completion index
        data_by_checkpoint_and_index = {}  # checkpoint -> {index -> win_rate_data}

        for filename in json_files:
            checkpoint = extract_checkpoint_from_filename(filename)
            if checkpoint is None:
                continue

            # Skip checkpoints beyond max_steps
            if max_steps is not None and checkpoint > max_steps:
                continue

            comp_index = extract_completion_index_from_filename(filename)
            json_path = os.path.join(folder_path, filename)
            win_rate_data = load_win_rate_data(json_path)

            if win_rate_data is not None:
                if checkpoint not in data_by_checkpoint_and_index:
                    data_by_checkpoint_and_index[checkpoint] = {}
                data_by_checkpoint_and_index[checkpoint][comp_index] = win_rate_data
                if model1_name is None:
                    model1_name = win_rate_data.get('model1_name', 'model1')
                    model2_name = win_rate_data.get('model2_name', 'model2')

        if not data_by_checkpoint_and_index:
            raise ValueError(f"No valid data found in {folder_path}")

        # Aggregate across indices for each checkpoint
        data_by_checkpoint = {}
        for checkpoint, index_data in data_by_checkpoint_and_index.items():
            indices = sorted(index_data.keys())
            num_indices = len(indices)

            # Collect values across indices for each metric
            aggregated = {
                'num_indices': num_indices,
                'indices_used': indices,
            }

            # Metrics to aggregate
            metrics = ['survey_mean_model1', 'survey_mean_model2', 'overall_mean_model1', 'overall_mean_model2']

            for metric in metrics:
                values = [index_data[idx].get(metric, 0.0) for idx in indices]
                aggregated[metric] = np.mean(values)
                aggregated[f'{metric}_values'] = values
                aggregated[f'{metric}_ci_low'] = np.min(values)
                aggregated[f'{metric}_ci_high'] = np.max(values)
                aggregated[f'{metric}_std'] = np.std(values) if num_indices > 1 else 0.0

            # Copy model names
            first_data = index_data[indices[0]]
            aggregated['model1_name'] = first_data.get('model1_name', 'model1')
            aggregated['model2_name'] = first_data.get('model2_name', 'model2')

            data_by_checkpoint[checkpoint] = aggregated

        return data_by_checkpoint, model1_name, model2_name

    else:
        # Original behavior: just load files (last one wins for each checkpoint)
        data_by_checkpoint = {}

        for filename in json_files:
            checkpoint = extract_checkpoint_from_filename(filename)
            if checkpoint is None:
                continue

            # Skip checkpoints beyond max_steps
            if max_steps is not None and checkpoint > max_steps:
                continue

            json_path = os.path.join(folder_path, filename)
            win_rate_data = load_win_rate_data(json_path)

            if win_rate_data is not None:
                data_by_checkpoint[checkpoint] = win_rate_data
                if model1_name is None:
                    model1_name = win_rate_data.get('model1_name', 'model1')
                    model2_name = win_rate_data.get('model2_name', 'model2')

        if not data_by_checkpoint:
            raise ValueError(f"No valid data found in {folder_path}")

        return data_by_checkpoint, model1_name, model2_name


def collect_data_across_temperatures(base_dir: str, task: str,
                                     temperatures: List[float],
                                     model: Optional[str] = None,
                                     N: Optional[int] = None,
                                     B: Optional[int] = None,
                                     chat_evaluator: Optional[str] = None,
                                     max_steps: Optional[int] = None,
                                     aggregate_indices: bool = False) -> Dict[float, Tuple[Dict[int, Dict[str, Any]], str, str]]:
    """Collect win rate data across all temperatures.

    Args:
        base_dir: Base directory containing temperature folders
        task: Task name (e.g., 'tldr')
        temperatures: List of temperatures to process
        model: Model prefix (e.g., 'qwen4b'). If provided, folder pattern is {model}-{task}-...
        N: Filter by N value in filename
        B: Filter by B value in filename
        chat_evaluator: Filter by evaluator type for chat tasks ('skywork' or 'qrm'). Only applies if task is 'chat' or 'ultrachat'.
        max_steps: Maximum checkpoint step to include. If None, no filtering.
        aggregate_indices: If True, aggregate data across all completion indices.

    Returns:
        Dict mapping temperature -> (data_by_checkpoint, model1_name, model2_name)
    """
    results = {}

    for temp in temperatures:
        # Build folder name based on whether model prefix is provided
        if model:
            folder_name = f'{model}-{task}-ranking-vs-regular-temp{temp}'
        else:
            folder_name = f'{task}-ranking-vs-regular-temp{temp}'
        temp_folder = os.path.join(base_dir, folder_name)

        print(f"\nProcessing temperature {temp}...")

        try:
            data_by_checkpoint, model1_name, model2_name = load_temperature_folder_data(
                temp_folder, N=N, B=B, chat_evaluator=chat_evaluator, task=task,
                max_steps=max_steps, aggregate_indices=aggregate_indices)
            print(f"  Found {len(data_by_checkpoint)} checkpoints")
            print(f"  Models: {model1_name} vs {model2_name}")
            if aggregate_indices:
                # Print info about indices found
                for ckpt in sorted(data_by_checkpoint.keys())[:1]:  # Just first checkpoint
                    ckpt_data = data_by_checkpoint[ckpt]
                    print(f"  Aggregating across {ckpt_data.get('num_indices', 1)} indices: {ckpt_data.get('indices_used', [0])}")
            results[temp] = (data_by_checkpoint, model1_name, model2_name)

        except Exception as e:
            print(f"  Error processing temperature {temp}: {e}")
            import traceback
            traceback.print_exc()

    return results


def plot_all_temperatures(data_by_temp: Dict[float, Tuple[Dict[int, Dict[str, Any]], str, str]],
                         task: str,
                         output_dir: str,
                         format: str = 'pdf',
                         model: Optional[str] = None,
                         N: Optional[int] = None,
                         B: Optional[int] = None,
                         chat_evaluator: Optional[str] = None,
                         aggregate_indices: bool = False,
                         error_bar_type: str = 'minmax'):
    """Create bar plots for survey and overall win rates across temperatures.

    Args:
        data_by_temp: Dict mapping temperature -> (data_by_checkpoint, model1_name, model2_name)
        task: Task name (e.g., 'tldr')
        output_dir: Directory to save plots
        format: Output format ('pdf')
        model: Model prefix for output filename (e.g., 'qwen4b')
        N: N value for output filename
        B: B value for output filename
        chat_evaluator: Evaluator type for chat tasks ('skywork' or 'qrm'). Included in filename if specified.
        aggregate_indices: If True, show error bars across completion indices.
        error_bar_type: Type of error bars - 'minmax' for min/max range, 'ci' for 95% confidence interval.
    """
    temperatures = sorted(data_by_temp.keys())

    if not temperatures:
        print("No data to plot")
        return

    # Green color (same as best bar in plot_best_worst_multi_model.py)
    bar_color = '#27ae60'

    # Create separate figures for survey and overall, for each temperature
    for temp in temperatures:
        data_by_checkpoint, model1_name, model2_name = data_by_temp[temp]

        # Determine which model is ranking
        if model1_name == 'ranking':
            is_model1_ranking = True
        elif model2_name == 'ranking':
            is_model1_ranking = False
        else:
            print(f"Warning: Could not identify ranking model for temp {temp}. Using model1.")
            is_model1_ranking = True

        checkpoints = sorted(data_by_checkpoint.keys())

        if not checkpoints:
            print(f"No checkpoints found for temp {temp}")
            continue

        for metric_name in ['survey', 'overall']:
            fig, ax = plt.subplots(figsize=pc.FIGSIZE)

            # Extract win rates for each checkpoint
            means = []
            yerr_low_list = []
            yerr_high_list = []
            has_error_bars = False

            for checkpoint in checkpoints:
                win_data = data_by_checkpoint[checkpoint]
                if is_model1_ranking:
                    mean_key = f'{metric_name}_mean_model1'
                else:
                    mean_key = f'{metric_name}_mean_model2'

                mean_val = win_data.get(mean_key, 0.0)
                means.append(mean_val)

                # Check for error bar data (only present when aggregate_indices=True)
                if aggregate_indices:
                    if error_bar_type == 'ci':
                        # 95% confidence interval: mean ± 1.96 * std / sqrt(n)
                        std_key = f'{mean_key}_std'
                        num_indices = win_data.get('num_indices', 1)
                        std_val = win_data.get(std_key, 0.0)
                        if num_indices > 1 and std_val > 0:
                            ci_margin = 1.96 * std_val / np.sqrt(num_indices)
                            yerr_low_list.append(ci_margin)
                            yerr_high_list.append(ci_margin)
                            has_error_bars = True
                        else:
                            yerr_low_list.append(0.0)
                            yerr_high_list.append(0.0)
                    else:  # minmax
                        ci_low_key = f'{mean_key}_ci_low'
                        ci_high_key = f'{mean_key}_ci_high'
                        if ci_low_key in win_data and ci_high_key in win_data:
                            yerr_low_list.append(mean_val - win_data[ci_low_key])
                            yerr_high_list.append(win_data[ci_high_key] - mean_val)
                            has_error_bars = True
                        else:
                            yerr_low_list.append(0.0)
                            yerr_high_list.append(0.0)

            # Calculate bar positions
            x = np.arange(len(checkpoints))
            bar_width = 0.6

            if has_error_bars and aggregate_indices:
                yerr = [yerr_low_list, yerr_high_list]
                ax.bar(x, means, bar_width,
                       color=bar_color, alpha=0.8, edgecolor='black', linewidth=1.0,
                       yerr=yerr, capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})
            else:
                ax.bar(x, means, bar_width,
                       color=bar_color, alpha=0.8, edgecolor='black', linewidth=1.0)

            # Formatting
            ax.set_xlabel('Training Steps', fontsize=pc.FONT_SIZE_AXIS_LABEL, fontweight=pc.FONT_WEIGHT_AXIS_LABEL)
            ax.set_ylabel('Win Rate', fontsize=pc.FONT_SIZE_AXIS_LABEL, fontweight=pc.FONT_WEIGHT_AXIS_LABEL)

            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in checkpoints], fontsize=pc.FONT_SIZE_TICK)
            ax.tick_params(axis='y', labelsize=pc.FONT_SIZE_TICK)

            ax.grid(True, alpha=pc.GRID_ALPHA, axis='y')

            # Add horizontal reference line at 0.5
            ax.axhline(y=0.5, color='black', linestyle=pc.REFLINE_STYLE, linewidth=pc.REFLINE_WIDTH, alpha=pc.REFLINE_ALPHA, zorder=1)

            # Set y-axis limits
            ax.set_ylim([0.45, 0.65])

            plt.subplots_adjust(left=pc.SUBPLOT_LEFT, right=pc.SUBPLOT_RIGHT,
                                top=pc.SUBPLOT_TOP, bottom=pc.SUBPLOT_BOTTOM)

            # Build output filename: model_task_temp{X}_metric.pdf
            # Include evaluator for chat tasks: model_task_evaluator_temp{X}_metric.pdf
            filename_parts = []
            if model:
                filename_parts.append(model)
            filename_parts.append(task)
            # Add evaluator to filename if specified for chat tasks
            if chat_evaluator is not None and task.lower() in ['chat', 'ultrachat']:
                filename_parts.append(chat_evaluator)
            if N is not None and B is not None:
                filename_parts.append(f'N{N}_B{B}')
            # Format temperature: e.g., 0.5 -> "temp0.5"
            filename_parts.append(f'temp{temp}')
            filename_parts.append(metric_name)
            # Add 'agg_ci' or 'agg_minmax' suffix if aggregating indices
            if aggregate_indices:
                filename_parts.append(f'agg_{error_bar_type}')
            output_filename = f'{"_".join(filename_parts)}.{format}'
            output_path = os.path.join(output_dir, output_filename)

            plt.savefig(output_path, dpi=pc.DPI)
            plt.close()

            print(f"\n{metric_name.capitalize()} plot (temp {temp}) saved to {output_path}")


def plot_multi_model_temperatures(all_model_data: Dict[str, Dict[float, Tuple[Dict[int, Dict[str, float]], str, str]]],
                                   task: str,
                                   output_dir: str,
                                   temperatures: List[float],
                                   format: str = 'pdf',
                                   model_configs: Optional[Dict[str, Dict[str, int]]] = None,
                                   chat_evaluator: Optional[str] = None):
    """Create bar plots for multiple models across temperatures.

    Args:
        all_model_data: Dict mapping model -> (Dict mapping temperature -> (data_by_checkpoint, model1_name, model2_name))
        task: Task name (e.g., 'tldr')
        output_dir: Directory to save plots
        temperatures: List of temperatures
        format: Output format ('pdf')
        model_configs: Dict mapping model -> {N, B} for filename
        chat_evaluator: Evaluator type for chat tasks ('skywork' or 'qrm'). Included in filename if specified.
    """
    models = list(all_model_data.keys())
    temperatures = sorted(temperatures)

    if not models or not temperatures:
        print("No data to plot")
        return

    # Color palette for different temperatures
    temp_colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))

    # Model markers for differentiation
    model_markers = {
        'qwen1.7b': 'o',
        'qwen4b': 'x',
        'qwen8b': 'D',
    }

    # Create separate figures for survey and overall
    for metric_name in ['survey', 'overall']:
        fig, ax = plt.subplots(figsize=pc.FIGSIZE)

        # Collect all checkpoints across all models and temperatures
        all_checkpoints = set()
        for model in models:
            for temp in temperatures:
                if temp in all_model_data[model]:
                    data_by_checkpoint, _, _ = all_model_data[model][temp]
                    all_checkpoints.update(data_by_checkpoint.keys())
        checkpoints = sorted(all_checkpoints)

        if not checkpoints:
            print(f"No checkpoints found for {metric_name}")
            continue

        # Bar positioning: group by checkpoint, within each group bars for each (model, temp) combo
        num_combos = len(models) * len(temperatures)
        group_width = 0.8
        bar_width = group_width / num_combos

        combo_idx = 0
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_temp_handles = []
        legend_model_handles = []

        for model_idx, model in enumerate(models):
            for temp_idx, temp in enumerate(temperatures):
                if temp not in all_model_data[model]:
                    combo_idx += 1
                    continue

                data_by_checkpoint, model1_name, model2_name = all_model_data[model][temp]

                # Determine which model is ranking
                if model1_name == 'ranking':
                    is_model1_ranking = True
                elif model2_name == 'ranking':
                    is_model1_ranking = False
                else:
                    is_model1_ranking = True

                # Extract win rates for each checkpoint
                means = []
                for checkpoint in checkpoints:
                    if checkpoint in data_by_checkpoint:
                        win_data = data_by_checkpoint[checkpoint]
                        if is_model1_ranking:
                            mean_key = f'{metric_name}_mean_model1'
                        else:
                            mean_key = f'{metric_name}_mean_model2'
                        means.append(win_data.get(mean_key, 0.0))
                    else:
                        means.append(0.0)

                color = temp_colors[temp_idx]

                # Calculate bar positions
                x = np.arange(len(checkpoints))
                offset = (combo_idx - (num_combos - 1) / 2) * bar_width
                bars = ax.bar(x + offset, means, bar_width,
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

                # Add model marker on top of each bar
                marker = model_markers.get(model, 'o')
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        marker_kwargs = {'marker': marker, 's': 50, 'zorder': 6}
                        if marker == 'x':
                            marker_kwargs.update({'color': 'black', 'linewidths': 1.5})
                        else:
                            marker_kwargs.update({'facecolors': 'none', 'edgecolors': 'black', 'linewidths': 1.5})
                        ax.scatter(bar.get_x() + bar.get_width() / 2, height, **marker_kwargs)

                combo_idx += 1

        # Build legend
        # Temperature colors
        for temp_idx, temp in enumerate(temperatures):
            legend_temp_handles.append(Patch(facecolor=temp_colors[temp_idx], edgecolor='black', label=f'Temp {temp:.1f}'))

        # Model markers
        for model in models:
            marker = model_markers.get(model, 'o')
            display_model = format_model_name(model)
            legend_model_handles.append(
                Line2D([0], [0], marker=marker, color='w', markerfacecolor='none',
                       markeredgecolor='black', markeredgewidth=1.5, markersize=8,
                       label=display_model, linestyle='None')
            )

        # Formatting
        ax.set_xlabel('Training Steps (Checkpoint)', fontsize=pc.FONT_SIZE_AXIS_LABEL, fontweight=pc.FONT_WEIGHT_AXIS_LABEL)
        ax.set_ylabel(f'GOPO {metric_name.capitalize()} Win Rate', fontsize=pc.FONT_SIZE_AXIS_LABEL, fontweight=pc.FONT_WEIGHT_AXIS_LABEL)

        title = f'GOPO Win Rates ({task.upper()})'
        ax.set_title(title, fontsize=pc.FONT_SIZE_TITLE, fontweight=pc.FONT_WEIGHT_TITLE, pad=20)

        ax.set_xticks(np.arange(len(checkpoints)))
        ax.set_xticklabels([str(c) for c in checkpoints], fontsize=pc.FONT_SIZE_TICK)
        ax.tick_params(axis='y', labelsize=pc.FONT_SIZE_TICK)
        ax.legend(handles=legend_temp_handles + legend_model_handles,
                  fontsize=pc.FONT_SIZE_LEGEND, loc='best', framealpha=0.9)
        ax.grid(True, alpha=pc.GRID_ALPHA, axis='y')

        # Add horizontal reference line at 0.5
        ax.axhline(y=0.5, color='black', linestyle=pc.REFLINE_STYLE, linewidth=pc.REFLINE_WIDTH, alpha=pc.REFLINE_ALPHA, zorder=1)

        # Set y-axis limits
        ax.set_ylim([0.4, 0.8])

        plt.subplots_adjust(left=pc.SUBPLOT_LEFT, right=pc.SUBPLOT_RIGHT,
                            top=pc.SUBPLOT_TOP, bottom=pc.SUBPLOT_BOTTOM)

        # Build output filename
        # Include evaluator for chat tasks: task_evaluator_multi_model_all_temps_metric.pdf
        filename_parts = [task]
        if chat_evaluator is not None and task.lower() in ['chat', 'ultrachat']:
            filename_parts.append(chat_evaluator)
        filename_parts.extend(['multi_model', 'all_temps', metric_name])
        output_filename = f'{"_".join(filename_parts)}.{format}'
        output_path = os.path.join(output_dir, output_filename)

        plt.savefig(output_path, dpi=pc.DPI)
        plt.close()

        print(f"\n{metric_name.capitalize()} plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot ranking win rates across all temperatures as bar charts.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--task', '-t', type=str, default='tldr',
                       help='Task name (default: tldr)')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Single model prefix (e.g., "qwen1.7b", "qwen4b").')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Multiple model prefixes (e.g., qwen1.7b qwen4b)')
    parser.add_argument('--model-configs', nargs='+', default=None,
                       help='Per-model N/B in format model:N:B (e.g., qwen1.7b:30:30 qwen4b:25:25)')
    parser.add_argument('--N', type=int, default=None,
                       help='Filter by N value in filename (e.g., 30 for N30)')
    parser.add_argument('--B', type=int, default=None,
                       help='Filter by B value in filename (e.g., 30 for B30)')
    parser.add_argument('--temperatures', nargs='+', type=float,
                       default=[0.1, 0.3, 0.5, 0.7, 0.9],
                       help='List of temperatures to analyze (default: 0.1 0.3 0.5 0.7 0.9)')
    parser.add_argument('--chat-evaluator', type=str, default=None,
                       choices=['skywork', 'qrm'],
                       help='Filter by evaluator type for chat tasks: "skywork" or "qrm". Only applies when task is "chat" or "ultrachat".')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for plots (default: ./figures)')
    parser.add_argument('--format', '-f', type=str, default='pdf',
                       choices=['pdf'],
                       help='Output format (default: pdf)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum training step (checkpoint) to include in plots (default: no limit)')
    parser.add_argument('--aggregate-indices', action='store_true',
                       help='Aggregate data across all completion indices (ind0, ind1, ind2, etc.) '
                            'and show error bars')
    parser.add_argument('--error-bar-type', type=str, default='minmax',
                       choices=['minmax', 'ci'],
                       help='Type of error bars when aggregating: "minmax" (min/max range) or "ci" (95%% confidence interval). Default: minmax')

    args = parser.parse_args()

    # Determine base directory and output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_dir, 'figures')

    os.makedirs(output_dir, exist_ok=True)

    # Parse per-model configs if provided
    model_configs = {}
    if args.model_configs:
        for entry in args.model_configs:
            parts = entry.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid model config '{entry}'. Expected format model:N:B")
            model_name = parts[0]
            model_N = int(parts[1])
            model_B = int(parts[2])
            model_configs[model_name] = {'N': model_N, 'B': model_B}

    # Determine models to process
    if args.model_configs:
        models = list(model_configs.keys())
    elif args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        models = None

    # Multi-model mode
    if models and len(models) > 1:
        print(f"Multi-model mode: {models}")
        if args.aggregate_indices:
            print("Aggregating across completion indices")
        all_model_data = {}
        for model in models:
            model_N = model_configs.get(model, {}).get('N', args.N)
            model_B = model_configs.get(model, {}).get('B', args.B)
            print(f"\nCollecting data for model '{model}' with N={model_N}, B={model_B}")
            data_by_temp = collect_data_across_temperatures(
                base_dir, args.task, args.temperatures,
                model=model, N=model_N, B=model_B, chat_evaluator=args.chat_evaluator,
                max_steps=args.max_steps, aggregate_indices=args.aggregate_indices
            )
            if data_by_temp:
                all_model_data[model] = data_by_temp

        if not all_model_data:
            print("Error: No valid data found")
            return 1

        print(f"\nGenerating multi-model plots...")
        plot_multi_model_temperatures(
            all_model_data, args.task, output_dir, args.temperatures,
            format=args.format, model_configs=model_configs, chat_evaluator=args.chat_evaluator
        )
    else:
        # Single model mode
        model = models[0] if models else None
        model_N = model_configs.get(model, {}).get('N', args.N) if model else args.N
        model_B = model_configs.get(model, {}).get('B', args.B) if model else args.B

        model_str = model if model else "(none)"
        nb_str = f"N{model_N}_B{model_B}" if model_N is not None and model_B is not None else "(all)"
        agg_str = " (aggregating indices)" if args.aggregate_indices else ""
        print(f"Collecting data for task '{args.task}', model '{model_str}', config '{nb_str}' across temperatures: {args.temperatures}{agg_str}")

        data_by_temp = collect_data_across_temperatures(
            base_dir, args.task, args.temperatures,
            model=model, N=model_N, B=model_B, chat_evaluator=args.chat_evaluator,
            max_steps=args.max_steps, aggregate_indices=args.aggregate_indices
        )

        if not data_by_temp:
            print("Error: No valid data found")
            return 1

        print(f"\nGenerating plots...")
        plot_all_temperatures(
            data_by_temp, args.task, output_dir, format=args.format,
            model=model, N=model_N, B=model_B, chat_evaluator=args.chat_evaluator,
            aggregate_indices=args.aggregate_indices, error_bar_type=args.error_bar_type
        )

    print(f"\nDone! Plots saved to {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
