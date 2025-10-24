#!/usr/bin/env bash
# delete_global_steps.sh
# Deletes global_step{N} subfolders inside saved_models/{model_name}/checkpoint-{N}
# and removes all non-directory files directly under saved_models/{model_name}

set -Eeuo pipefail

DRY_RUN=false
LIST_ONLY=false

usage() {
  cat <<EOF
Usage: $(basename "$0") <model_name> [--dry-run] [--list-only]

Arguments:
  model_name     Name of the model folder inside saved_models/
Options:
  --dry-run      Show what would be deleted, but do not delete.
  --list-only    Only list checkpoint-* directories found; no deletions.

Example:
  $(basename "$0") Qwen3-1.7B-if-bsz128-ts500-ranking-skywork8b-seed42-lr1e-6-warmup10 --dry-run
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODEL_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true; shift ;;
    --list-only)
      LIST_ONLY=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      shift ;;
  esac
done

MODEL_DIR="saved_models/$MODEL_NAME"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Error: Directory '$MODEL_DIR' not found." >&2
  exit 1
fi

cd "$MODEL_DIR"

shopt -s nullglob
found_any=false

echo "=== Checking checkpoint-* directories ==="

for dir in checkpoint-*; do
  [[ -d "$dir" ]] || continue
  found_any=true

  if [[ "$dir" =~ ^checkpoint-([0-9]+)$ ]]; then
    step_num="${BASH_REMATCH[1]}"
    target="$dir/global_step${step_num}"

    if $LIST_ONLY; then
      echo "[FOUND] $dir"
      continue
    fi

    if [[ -d "$target" ]]; then
      if $DRY_RUN; then
        echo "[DRY-RUN] Would delete: $target"
      else
        echo "[DELETE] $target"
        rm -rf -- "$target"
      fi
    else
      echo "[SKIP] No matching subdir: $target"
    fi
  else
    echo "[SKIP] '$dir' doesn't match 'checkpoint-<digits>' exactly."
  fi
done

if ! $found_any; then
  echo "No 'checkpoint-*' directories found under: $MODEL_DIR"
fi

echo
echo "=== Cleaning up non-directory files in $MODEL_DIR ==="

for item in *; do
  [[ -e "$item" ]] || continue
  if [[ ! -d "$item" ]]; then
    if $LIST_ONLY; then
      echo "[FOUND-FILE] $item"
    elif $DRY_RUN; then
      echo "[DRY-RUN] Would delete file: $item"
    else
      echo "[DELETE-FILE] $item"
      rm -f -- "$item"
    fi
  fi
done
