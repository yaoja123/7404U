# Claude-authored training scripts — CE / LS / MaxSup ResNet-50

Purpose: produce our own CE, LS (alpha=0.1), and MaxSup ResNet-50 checkpoints
for Tiny-ImageNet, then build a paper-style Grad-CAM comparison figure and a
live Streamlit demo.

Important: the current Tiny-ImageNet training flow uses the stable plain
PyTorch trainer in `train_resnet50_tiny_plain.py`. The earlier
FFCV + Lightning path repeatedly crashed on the farm after epoch boundaries, so
it is kept only as legacy reference.

See [PLAN.md](PLAN.md) for the full rationale.

## Tiny-ImageNet quickstart (current primary flow)

Used because ImageNet-1K hit farm quota / OOM issues. Tiny (200 classes,
64x64, ~240 MB raw) is lightweight enough for the farm and still gives usable
Grad-CAM heatmaps after resizing to 224 at train / inference time.

```bash
# 0. Sync scripts to the farm. Copy the contents, not a nested Claude dir.
rsync -av "/Users/jayden/Desktop/7404 comp/project/scripts/Claude/" \
  u3651420@gpu2gate1.cs.hku.hk:~/project/scripts/Claude/

# 1. One-time env setup
conda run -n tinyffcv python -c "import torch, torchvision"
conda run -n tinyffcv pip install grad-cam streamlit pyarrow
conda run -n tinyffcv hf auth login

# 2. Download + unpack Tiny-ImageNet (~30-60 min on farm)
tmux new-session -d -s tiny_data 'bash ~/project/scripts/Claude/fetch_and_pack_tiny.sh'
tmux attach -t tiny_data
#   outputs: ~/data/tiny_imagenet/{train,val}

# 3. On the gateway, start tmux and enter a GPU shell
tmux new -s ce_tiny
gpu-interactive

# 4. Inside the GPU node, run a 1-epoch smoke test with a live progress bar
bash ~/project/scripts/Claude/train_resnet50_ce_tiny.sh \
  --conda-env tinyffcv \
  --resume none \
  --epochs 1 \
  --output-dir ~/project/outputs/plain_ce_tiny_smoke \
  --experiment-name resnet50_ce_tiny_smoke

# 5. Full runs
bash ~/project/scripts/Claude/train_resnet50_ce_tiny.sh --conda-env tinyffcv
bash ~/project/scripts/Claude/train_resnet50_ls_tiny.sh --conda-env tinyffcv
bash ~/project/scripts/Claude/train_resnet50_maxsup_tiny.sh --conda-env tinyffcv
# detach with Ctrl+B d; reattach with: tmux attach -t ce_tiny

# 6. Static paper-style Grad-CAM figure
conda run -n tinyffcv python scripts/Claude/gradcam_compare.py \
    --images demo/*.jpg \
    --ce-ckpt     outputs/plain_ce_tiny/best.pth \
    --ls-ckpt     outputs/plain_ls_tiny/best.pth \
    --maxsup-ckpt outputs/plain_maxsup_tiny/best.pth \
    --output tmp/gradcam_paper_style.png

# 7. Live Streamlit demo
conda run -n tinyffcv streamlit run scripts/Claude/live_demo.py -- \
    --ce-ckpt     outputs/plain_ce_tiny/best.pth \
    --ls-ckpt     outputs/plain_ls_tiny/best.pth \
    --maxsup-ckpt outputs/plain_maxsup_tiny/best.pth
# From your laptop:
#   ssh -L 8501:localhost:8501 u3651420@gpu2gate1.cs.hku.hk
# then open http://localhost:8501
```

If you see `maxsup_repo/Conv/ffcv/main.py` in the logs, you are still running an
old script copy on the farm.

## ImageNet-1K flow (legacy — kept for reference)

The scripts below target the full ImageNet-1K recipe. They were written before
the pivot to Tiny-ImageNet and are retained in case farm quota frees up.

## What's in this directory

| File | Purpose |
|---|---|
| `PLAN.md` | Full plan & rationale (copy of the approved plan). |
| `run_ffcv_hku.sh` | Legacy Slurm launcher for the old FFCV path. |
| **Tiny-ImageNet pipeline (primary)** | |
| `unpack_tiny_imagenet.py` | Direct-pyarrow reader: HF parquet shards → ImageFolder layout (200 wnid subdirs). |
| `fetch_and_pack_tiny.sh` | Orchestrator: `hf download` → unpack Tiny-ImageNet into ImageFolder. |
| `train_resnet50_tiny_plain.py` | Stable plain PyTorch Tiny-ImageNet trainer with CE / LS / MaxSup. |
| `train_resnet50_ce_tiny.sh` | Wrapper — CE on Tiny-ImageNet. |
| `train_resnet50_ls_tiny.sh` | Wrapper — LS (α=0.1) on Tiny-ImageNet. |
| `train_resnet50_maxsup_tiny.sh` | Wrapper — MaxSup (`loss_type=ms`) on Tiny-ImageNet. |
| `gradcam_compare.py` | Paper-style static Grad-CAM figure across CE / LS / MaxSup. |
| `live_demo.py` | Streamlit demo — upload or webcam capture, three-way Grad-CAM comparison. |
| **Shared** | |
| `lightning_ckpt_to_statedict.py` | Legacy Lightning conversion helper. Not needed for the plain Tiny trainer outputs. |
| **ImageNet-1K (legacy)** | |
| `train_resnet50_ce_4080.sh` | One-line wrapper — CE baseline, 4080-tuned, ImageNet-1K. |
| `train_resnet50_ls_4080.sh` | One-line wrapper — LS (α=0.1), 4080-tuned, ImageNet-1K. |

## Recipe (matches author, single-4080 scaled)

| Knob | Value | Source |
|---|---|---|
| Architecture | ResNet-50 + BlurPool | `maxsup_repo/Conv/ffcv/model.py` |
| Optimizer | SGD, momentum=0.9, wd=1e-4 (BN/bias excluded) | author `config.toml` |
| LR | **0.2** | linear-scaled from 0.8 × 128/512 |
| Batch | 128 (single GPU) | fits on 4080 BF16 |
| Schedule | StepLR step=30 γ=0.1 | author `config.toml` |
| Epochs | 90 | author `config.toml` |
| Precision | bf16-mixed | author `main.py` |
| Dataloader | FFCV (`train_500_0.50_90.ffcv`) | author |
| Augmentation | RandomResizedCrop 224 + HFlip + ImageNet norm | author |

Only the loss function differs between the CE and LS runs.

## One-time setup on the GPU farm

### 1. FFCV env

```bash
conda run -n 7606 python -c "import ffcv, lightning, torch; print(ffcv.__version__, lightning.__version__, torch.__version__)"
```

Make sure the `7606` conda env (or whatever `--conda-env` you pass) has
`ffcv`, `lightning`, `torchvision`, `torchmetrics`, and `python-dotenv`.

### 2. Convert ImageNet to .ffcv (one-time, ~2-3h)

```bash
cd ~/project/maxsup_repo/Conv/ffcv/create_data
python write_imagenet.py --split train --data-dir ~/data/imagenet --write-path ~/data/imagenet_ffcv/train_500_0.50_90.ffcv
python write_imagenet.py --split val   --data-dir ~/data/imagenet --write-path ~/data/imagenet_ffcv/val_500_0.50_90.ffcv
```

Resulting files: ~110 GB for train, ~5 GB for val.

## Running training

Both runs take ~1.5–2 days on a single 4080. Always run inside `tmux` on the
gateway node so an SSH drop doesn't kill the Slurm `srun --pty` session.

### CE baseline

```bash
tmux new -s ce_train
bash scripts/Claude/train_resnet50_ce_4080.sh \
  --train-ffcv ~/data/imagenet_ffcv/train_500_0.50_90.ffcv \
  --val-ffcv   ~/data/imagenet_ffcv/val_500_0.50_90.ffcv
# Ctrl+B d  to detach.  tmux attach -t ce_train  to resume viewing.
```

Output goes to `outputs/ffcv_ce/`:
- `checkpoints/last.ckpt` — rolling latest (survives preemption).
- `checkpoints/resnet50_ce_4080-epXX-Val_Acc_Top1=YY.YY.ckpt` — best-by-top1.
- `logs/resnet50_ce_4080/version_0/` — TensorBoard logs.

### LS baseline

```bash
tmux new -s ls_train
bash scripts/Claude/train_resnet50_ls_4080.sh \
  --train-ffcv ~/data/imagenet_ffcv/train_500_0.50_90.ffcv \
  --val-ffcv   ~/data/imagenet_ffcv/val_500_0.50_90.ffcv
```

Output goes to `outputs/ffcv_ls/`.

### Resuming after preemption / disconnect

Re-run the same command. `--resume latest` (the default) picks up from
`outputs/ffcv_{ce,ls}/checkpoints/last.ckpt`.

## Post-training: extract clean weights

Lightning saves with `model.` key prefix; strip it so the state_dict loads
into a plain torchvision `resnet50`:

```bash
conda run -n 7606 python scripts/Claude/lightning_ckpt_to_statedict.py \
    outputs/ffcv_ce/checkpoints/last.ckpt \
    weights/resnet50_ce.pth

conda run -n 7606 python scripts/Claude/lightning_ckpt_to_statedict.py \
    outputs/ffcv_ls/checkpoints/last.ckpt \
    weights/resnet50_ls.pth
```

Note: the recipe wraps stride-2 convs with BlurPool, so downstream code must
apply the same wrapper before `load_state_dict`:

```python
import torchvision.models as tvm
from scripts.train_resnet50_ce import apply_blurpool   # already in repo

m = tvm.resnet50(num_classes=1000)
apply_blurpool(m)
state = torch.load('weights/resnet50_ce.pth', map_location='cpu')
m.load_state_dict(state['state_dict'])
```

## Expected final numbers (paper Table 2)

| Variant | Top-1 | Top-5 |
|---|---|---|
| Baseline (CE) | 76.3% | 93.1% |
| LS (α=0.1) | 76.4% | 93.1% |
| MaxSup (author-released) | 76.9% | 93.4% |

Our runs should land within ~0.3% of these at epoch 90. Any larger gap → check
LR scaling (try grad-accum to 512 to keep lr=0.8 as a fallback).

## Troubleshooting

- **VRAM OOM on 4080**: drop to `--batch-size 96` and set `--lr 0.15`
  (= 0.8 × 96/512). Linear scale always follows batch.
- **Loss diverges after epoch 1**: LR too high; halve it. Or use grad-accum to
  match the author's global batch=512 and keep lr=0.8 untouched.
- **FFCV import error**: you're not in the right pixi/conda env. See setup.
- **Slurm kills job at 24h**: re-run — resume is automatic via `last.ckpt`.
  Consider requesting a longer time limit or a non-`debug` partition.
