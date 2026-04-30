#!/usr/bin/env bash
set -euo pipefail

# Stable LS baseline on Tiny-ImageNet using plain torchvision ImageFolder
# dataloaders instead of the FFCV+Lightning stack.
#
# Usage (inside gpu-interactive on a GPU node):
#   bash ~/project/scripts/Claude/train_resnet50_ls_tiny.sh --conda-env tinyffcv

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="tinyffcv"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if ! hostname | grep -Eq '^gpu-'; then
  echo "Run this from an active gpu-interactive shell on a GPU node." >&2
  exit 1
fi

echo "[plain-tiny] LS wrapper"
echo "  env:    ${CONDA_ENV}"
echo "  output: ${HOME}/project/outputs/plain_ls_tiny"
echo "  entry:  ${HERE}/train_resnet50_tiny_plain.py"

exec conda run --no-capture-output -n "${CONDA_ENV}" python "${HERE}/train_resnet50_tiny_plain.py" \
  --loss-type ls \
  --label-smoothing 0.1 \
  --data-path "${HOME}/data/tiny_imagenet" \
  --output-dir "${HOME}/project/outputs/plain_ls_tiny" \
  --experiment-name resnet50_ls_tiny \
  --epochs 90 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --workers 2 \
  --lr 0.2 \
  --amp none \
  --resume latest \
  "${EXTRA_ARGS[@]}"
