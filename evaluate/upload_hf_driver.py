#!/usr/bin/env python3
"""
Sequential driver that calls evaluate/upload_hf_model.py for a range of checkpoints.

This keeps behavior consistent with the existing uploader while automating
uploads one-by-one. It optionally skips checkpoints whose repos already exist.
"""

import argparse
import os
import subprocess
import sys
from typing import List

try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None  # type: ignore


DEFAULT_BASE_DIR = \
    "/root/gopo/saved_models/Qwen3-1.7B-if-bsz128-ts500-regular-skywork8b-seed42-lr1e-6-warmup10"

REPO_BASENAME = \
    "Qwen3-1.7B-if-bsz128-ts500-regular-skywork8b-seed42-lr1e-6-warmup10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially upload checkpoints by invoking upload_hf_model.py",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing checkpoint-* folders",
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        default="choiqs",
        help="Hugging Face username/organization for repo owner",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=275,
        help="First checkpoint number to upload (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=500,
        help="Last checkpoint number to upload (inclusive)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=25,
        help="Step between checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing uploads",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Force upload even if repo already exists",
    )
    return parser.parse_args()


def build_checkpoint_numbers(start: int, end: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("--step must be > 0")
    if start > end:
        raise ValueError("--start must be <= --end")
    values: List[int] = []
    v = start
    while v <= end:
        values.append(v)
        v += step
    return values


def repo_exists(repo_id: str) -> bool:
    if HfApi is None:
        return False
    try:
        api = HfApi()
        _ = api.repo_info(repo_id)
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()

    checkpoints = build_checkpoint_numbers(args.start, args.end, args.step)
    print(f"Planned checkpoints: {checkpoints}")

    # Derive repo basename from the provided base directory so repo names
    # reflect the source model path, e.g., saved_models/<basename>.
    repo_basename = os.path.basename(os.path.normpath(args.base_dir))

    uploader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "upload_hf_model.py"))
    if not os.path.exists(uploader_path):
        print(f"Error: uploader not found at {uploader_path}")
        return 1

    any_failures = False
    for n in checkpoints:
        ckpt_dir = os.path.join(args.base_dir, f"checkpoint-{n}")
        if not os.path.isdir(ckpt_dir):
            print(f"[SKIP] Missing checkpoint directory: {ckpt_dir}")
            continue

        repo_name = f"{args.hf_username}/{repo_basename}-checkpoint{n}"

        if not args.no_skip_existing and repo_exists(repo_name):
            print(f"[SKIP] Repo already exists: https://huggingface.co/{repo_name}")
            continue

        cmd = [
            sys.executable,
            uploader_path,
            "--model-dir",
            ckpt_dir,
            "--repo-name",
            repo_name,
        ]

        print("\n============================================================")
        print(f"Uploading checkpoint-{n}")
        print(f"  Model dir  : {ckpt_dir}")
        print(f"  Repo       : {repo_name}")
        print("  Command    :", " ".join(cmd))
        print("============================================================")

        if args.dry_run:
            continue

        # Inherit environment so HF tokens are available
        try:
            result = subprocess.run(cmd, env=os.environ.copy(), check=True)
            if result.returncode == 0:
                print(f"[OK] Uploaded checkpoint-{n} → {repo_name}")
            else:
                any_failures = True
                print(f"[FAIL] Uploader exited with code {result.returncode} for checkpoint-{n}")
        except subprocess.CalledProcessError as e:
            any_failures = True
            print(f"[ERROR] Upload failed for checkpoint-{n}: {e}")

    print("\nAll done.")
    return 0 if not any_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())


