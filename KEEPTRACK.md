# Keep Track

Use this file as the single quick status sheet for the Tiny-ImageNet CE / LS / MaxSup project.

## Source Of Truth

- Code source of truth: local git repo
- Training artifacts source of truth: GPU farm
- Do not commit large farm artifacts:
  - `*.pth`
  - `*.ckpt`
  - generated Grad-CAM images in `tmp/`
  - temporary demo candidate images

## Final Checkpoints On Farm

- CE:
  - `/userhome/cs/u3651420/project/outputs/plain_ce_tiny_fresh/best.pth`
- LS:
  - `/userhome/cs/u3651420/project/outputs/plain_ls_tiny_full/best.pth`
- MaxSup:
  - `/userhome/cs/u3651420/project/outputs/plain_maxsup_tiny_full/best.pth`

Training completion status:

- [x] CE finished 90 epochs
- [x] LS finished 90 epochs
- [x] MaxSup finished 90 epochs

## Farm Runtime Files

- Class mapping JSON:
  - `/userhome/cs/u3651420/project/scripts/Claude/tiny_imagenet_wnid_names.json`
- Optional label source file:
  - `/userhome/cs/u3651420/project/scripts/Claude/words.txt`

## Sync Checklist

Mark these whenever you change code locally and copy it to the farm.

- [ ] `scripts/Claude/gradcam_compare.py` synced to farm
- [ ] `scripts/Claude/live_demo.py` synced to farm
- [ ] `scripts/Claude/train_resnet50_tiny_plain.py` synced to farm
- [ ] `scripts/Claude/tiny_imagenet_wnid_names.json` synced to farm
- [ ] `scripts/Claude/words.txt` synced to farm

## Grad-CAM Status

Current known state:

- [x] Grad-CAM script runs
- [x] Three final checkpoints load
- [x] Ground-truth title support added locally
- [x] `--target-mode top1|gt` added locally
- [ ] Farm copy confirmed to include latest Grad-CAM changes
- [ ] Human-readable class names confirmed
- [ ] Final poster-quality examples selected
- [ ] Final paper-style Grad-CAM figure exported

Current comparison caveat:

- `top1` mode is diagnostic
- `gt` mode is better for fair three-model comparison

## Demo Status

- [ ] Streamlit demo runs on GPU node
- [ ] Local port forwarding verified
- [ ] Upload image works
- [ ] Three-model Grad-CAM inference works in demo
- [ ] Top-5 predictions render correctly

## Recommended Next Step

1. Sync latest `gradcam_compare.py` to farm.
2. Run Grad-CAM in `--target-mode gt`.
3. Confirm labels are readable.
4. Pick final examples.
5. Run Streamlit demo on GPU node.

## Useful Farm Checks

Check running jobs:

```bash
squeue -u $USER
```

Check final checkpoints:

```bash
ls -lh \
  ~/project/outputs/plain_ce_tiny_fresh/best.pth \
  ~/project/outputs/plain_ls_tiny_full/best.pth \
  ~/project/outputs/plain_maxsup_tiny_full/best.pth
```

Check latest epoch state:

```bash
python - <<'PY'
import torch
for p in [
    '/userhome/cs/u3651420/project/outputs/plain_ls_tiny_full/latest.pth',
    '/userhome/cs/u3651420/project/outputs/plain_maxsup_tiny_full/latest.pth',
]:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    print(p, ck.get('epoch'), ck.get('epoch_complete'), ck.get('best_acc1'))
PY
```
