#!/usr/bin/env python3
import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def parse_float_list(values: List[str]) -> List[float]:
    return [float(v) for v in values]


def plot_ratios_by_model_size(title: str, model_sizes: List[str], win: List[float], lose: List[float], tie: List[float], out_path: str | None) -> None:
    x = np.arange(len(model_sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, win,  width, label="Win",  color="#2ca02c")  # green
    ax.bar(x,         lose, width, label="Lose", color="#d62728")  # red
    ax.bar(x + width, tie,  width, label="Tie",  color="#7f7f7f")  # gray

    ax.set_title(title)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Ratio")
    ax.set_xticks(x, model_sizes)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    # Always save under figure/ directory
    os.makedirs("figure", exist_ok=True)
    fig.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot seed-42 win/lose/tie ratios by model size")
    parser.add_argument("--title", default="Seed 42: win/lose/tie by model size")
    parser.add_argument("--sizes", nargs="+", default=["1.7B", "4B", "8B"], help="List of model sizes labels in order")
    parser.add_argument("--win", nargs="+", required=True, help="Win ratios, space-separated (e.g. 0.52 0.58 0.45)")
    parser.add_argument("--lose", nargs="+", required=True, help="Lose ratios, space-separated")
    parser.add_argument("--tie", nargs="+", required=True, help="Tie ratios, space-separated")
    parser.add_argument("--out", default=None, help="Filename or path; will be saved under figure/")

    args = parser.parse_args()

    win = parse_float_list(args.win)
    lose = parse_float_list(args.lose)
    tie = parse_float_list(args.tie)

    if not (len(args.sizes) == len(win) == len(lose) == len(tie)):
        raise SystemExit("sizes, win, lose, and tie must have the same length")

    # Basic sanity: ratios should be within [0,1]
    for arr, name in [(win, "win"), (lose, "lose"), (tie, "tie")]:
        for v in arr:
            if v < 0.0 or v > 1.0:
                raise SystemExit(f"{name} ratio out of range [0,1]: {v}")

    # Resolve output path: always inside figure/
    if args.out:
        out_name = os.path.basename(args.out)
    else:
        # Derive a filename from the title
        safe_title = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in args.title.strip().replace(" ", "_").lower())
        out_name = f"{safe_title}.png"
    out_path = os.path.join("figure", out_name)

    plot_ratios_by_model_size(args.title, args.sizes, win, lose, tie, out_path)


if __name__ == "__main__":
    main()

