#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_task(arg: str) -> Tuple[str, List[float], List[float]]:
    """Parse a task argument of the form Label|w1,w2,...|l1,l2,..."""
    try:
        label, win_csv, lose_csv = arg.split("|")
        win = [float(x) for x in win_csv.split(",") if x]
        lose = [float(x) for x in lose_csv.split(",") if x]
        return label.strip(), win, lose
    except Exception as e:
        raise SystemExit(f"Invalid --task format: '{arg}'. Expected Label|w1,w2,...|l1,l2,...  ({e})")


def win_lose_ratio(win: List[float], lose: List[float]) -> List[float]:
    """Return elementwise win/lose ratio (ignoring ties). Uses small epsilon for 0 lose."""
    ratios = []
    for w, l in zip(win, lose):
        denom = l if l != 0 else 1e-9
        ratios.append(w / denom)
    return ratios


def plot_win_lines(title: str, sizes: List[str], tasks: List[Tuple[str, List[float], List[float]]], out_name: str | None) -> None:
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '>', '<']
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (5, 1, 1, 1))]
    for idx, (label, win, lose) in enumerate(tasks):
        if not (len(sizes) == len(win) == len(lose)):
            raise SystemExit(f"Task '{label}' lengths mismatch with sizes: sizes={len(sizes)} win={len(win)} lose={len(lose)}")
        ratio = win_lose_ratio(win, lose)
        ax.plot(
            x,
            ratio,
            marker=markers[idx % len(markers)],
            markersize=8.5,
            linewidth=1.5,
            linestyle=linestyles[idx % len(linestyles)],
            label=label,
        )

    # Remove title; enlarge axis labels and ticks
    # ax.set_title(title)
    ax.set_xlabel("Qwen3 model size", fontsize=16)
    ax.set_ylabel(r"$\mathbf{win}_{\mathrm{gopo}}$", fontsize=16)
    ax.set_xticks(x, sizes)
    # Make tick label sizes match the axis label sizing intent
    ax.tick_params(axis='both', which='major', labelsize=16)
    # Set y-axis to start at 0.5
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(title="Task", frameon=False, fontsize=12, title_fontsize=12)

    fig.tight_layout()

    # Always save under figure/
    os.makedirs("figure", exist_ok=True)
    if out_name:
        out_file = os.path.join("figure", os.path.basename(out_name))
    else:
        safe_title = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in title.strip().replace(" ", "_").lower())
        out_file = os.path.join("figure", f"{safe_title}.png")
    fig.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot win ratios (ignoring ties) vs model size for multiple tasks")
    parser.add_argument("--title", default="Seed 42: Win ratio by model size (ignore ties)")
    parser.add_argument("--sizes", nargs="+", default=["1.7B", "4B", "8B"], help="Model size labels in order")
    parser.add_argument("--task", action="append", required=True, help="Task spec: Label|w1,w2,...|l1,l2,...  (repeat for multiple tasks)")
    parser.add_argument("--out", default=None, help="Output filename; saved under figure/")
    args = parser.parse_args()

    tasks: List[Tuple[str, List[float], List[float]]] = [parse_task(t) for t in args.task]
    plot_win_lines(args.title, args.sizes, tasks, args.out)


if __name__ == "__main__":
    main()




# Example
# python figure/plot_seed42_lines.py \
#   --title "Win ratio vs size (seed 42)" \
#   --sizes 1.7B 4B 8B \
#   --task "UltraChat|0.742,0.560,0.504|0.258,0.438,0.494" \
#   --task "TLDR|0.524,0.582,0.454|0.472,0.418,0.544" \
#   --task "IFeval|0.552,0.508,0.498|0.426,0.480,0.484" \
#   --out seed42_lines.pdf