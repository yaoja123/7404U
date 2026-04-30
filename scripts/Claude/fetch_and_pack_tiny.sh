#!/usr/bin/env bash
# Tiny-ImageNet pipeline: HF download -> ImageFolder.
# Optional legacy step: ImageFolder -> FFCV.
#
# The current primary training flow uses plain torchvision ImageFolder
# dataloaders, so FFCV writing is disabled by default.
#
# Run on GPU farm inside tmux/screen:
#   tmux new-session -d -s tiny_data 'bash ~/project/scripts/Claude/fetch_and_pack_tiny.sh'
#   tmux attach -t tiny_data

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLAUDE_DIR="${ROOT_DIR}/scripts/Claude"
FFCV_WRITER="${ROOT_DIR}/maxsup_repo/Conv/ffcv/create_data/write_imagenet.py"

HF_REPO="${HF_REPO:-zh-plus/tiny-imagenet}"
HF_DIR="${HF_DIR:-${HOME}/data/tiny_imagenet_hf}"
IMAGEFOLDER_DIR="${IMAGEFOLDER_DIR:-${HOME}/data/tiny_imagenet}"
FFCV_DIR="${FFCV_DIR:-${HOME}/data/tiny_imagenet_ffcv}"
CONDA_ENV="${CONDA_ENV:-tinyffcv}"
PYTHON_BIN="${PYTHON_BIN:-conda run -n ${CONDA_ENV} python}"
HF_BIN="${HF_BIN:-conda run -n ${CONDA_ENV} hf}"
FFCV_PYTHON_BIN="${FFCV_PYTHON_BIN:-conda run -n ${CONDA_ENV} python}"
WRITE_FFCV="${WRITE_FFCV:-0}"

MAX_RESOLUTION="${MAX_RESOLUTION:-224}"     # upsample 64 -> 224 at FFCV write
JPEG_QUALITY="${JPEG_QUALITY:-90}"
COMPRESS_PROB="${COMPRESS_PROB:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DELETE_INTERMEDIATE="${DELETE_INTERMEDIATE:-0}"   # 1 -> delete HF parquet + ImageFolder after FFCV

TRAIN_FFCV="${FFCV_DIR}/train_${MAX_RESOLUTION}_${COMPRESS_PROB}_${JPEG_QUALITY}.ffcv"
VAL_FFCV="${FFCV_DIR}/val_${MAX_RESOLUTION}_${COMPRESS_PROB}_${JPEG_QUALITY}.ffcv"

mkdir -p "${HF_DIR}" "${IMAGEFOLDER_DIR}" "${FFCV_DIR}"

echo "[step 1/3] Download ${HF_REPO} -> ${HF_DIR}"
if [[ -f "${HF_DIR}/.complete" ]]; then
    echo "  already downloaded (found ${HF_DIR}/.complete)"
else
    ${HF_BIN} download "${HF_REPO}" --repo-type dataset --local-dir "${HF_DIR}"
    touch "${HF_DIR}/.complete"
fi

echo
echo "[step 2/3] Unpack parquet -> ImageFolder at ${IMAGEFOLDER_DIR}"
${PYTHON_BIN} "${CLAUDE_DIR}/unpack_tiny_imagenet.py" \
    --hf-dir "${HF_DIR}" \
    --out-dir "${IMAGEFOLDER_DIR}"

echo
if [[ "${WRITE_FFCV}" != "1" ]]; then
    echo "[step 3/3] Skip FFCV write (WRITE_FFCV=${WRITE_FFCV})"
    echo "  current training wrappers use ImageFolder directly."
    echo "  set WRITE_FFCV=1 only if you explicitly want the legacy FFCV files."
    echo
    echo "Done."
    echo "  IMAGEFOLDER_DIR=${IMAGEFOLDER_DIR}"
    echo
    echo "Next: launch training with one of"
    echo "  bash scripts/Claude/train_resnet50_ce_tiny.sh --conda-env tinyffcv"
    echo "  bash scripts/Claude/train_resnet50_ls_tiny.sh --conda-env tinyffcv"
    echo "  bash scripts/Claude/train_resnet50_maxsup_tiny.sh --conda-env tinyffcv"
    exit 0
fi

echo "[step 3/3] Write FFCV files -> ${FFCV_DIR}"
if ! ${FFCV_PYTHON_BIN} -c "import ffcv" >/dev/null 2>&1; then
    cat >&2 <<EOF
FFCV is not importable from:
  ${FFCV_PYTHON_BIN}

This optional legacy step writes .ffcv files with the author's FFCV pipeline.
The current Tiny-ImageNet wrappers do not need FFCV. If you only want training,
rerun without WRITE_FFCV=1.

If you do want .ffcv files, fix one of:
  1. Install / repair FFCV in the chosen env, then rerun with WRITE_FFCV=1.
  2. Override the writer interpreter explicitly, e.g.
     WRITE_FFCV=1 FFCV_PYTHON_BIN="conda run -n <your-ffcv-env> python" \\
       bash scripts/Claude/fetch_and_pack_tiny.sh
EOF
    exit 1
fi

write_ffcv() {
    local split="$1"          # train | val
    local src="${IMAGEFOLDER_DIR}/${split}"
    local dst="$2"

    if [[ -f "${dst}" ]]; then
        echo "  [skip] already exists: ${dst}"
        return
    fi

    echo "  writing ${split} -> ${dst}"
    (
        cd "$(dirname "${FFCV_WRITER}")"
        ${FFCV_PYTHON_BIN} "$(basename "${FFCV_WRITER}")" \
            --cfg.dataset imagenet \
            --cfg.split "${split}" \
            --cfg.data_dir "${src}" \
            --cfg.write_path "${dst}" \
            --cfg.max_resolution "${MAX_RESOLUTION}" \
            --cfg.num_workers "${NUM_WORKERS}" \
            --cfg.jpeg_quality "${JPEG_QUALITY}" \
            --cfg.compress_probability "${COMPRESS_PROB}"
    )
}

write_ffcv train "${TRAIN_FFCV}"
write_ffcv val   "${VAL_FFCV}"

if [[ "${DELETE_INTERMEDIATE}" == "1" ]]; then
    echo
    echo "DELETE_INTERMEDIATE=1 -> removing HF parquet and ImageFolder"
    rm -rf "${HF_DIR}" "${IMAGEFOLDER_DIR}"
fi

echo
echo "Done."
echo "  IMAGEFOLDER_DIR=${IMAGEFOLDER_DIR}"
echo "  TRAIN_FFCV=${TRAIN_FFCV}"
echo "  VAL_FFCV=${VAL_FFCV}"
echo
echo "Next: launch training with one of"
echo "  bash scripts/Claude/train_resnet50_ce_tiny.sh --conda-env tinyffcv"
echo "  bash scripts/Claude/train_resnet50_ls_tiny.sh --conda-env tinyffcv"
echo "  bash scripts/Claude/train_resnet50_maxsup_tiny.sh --conda-env tinyffcv"
