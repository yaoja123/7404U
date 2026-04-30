# Handoff Status

Use this file to brief another agent or ChatGPT quickly.

## Project Goal

This project compares three Tiny-ImageNet ResNet-50 models trained with different objectives:

- CE
- Label Smoothing
- MaxSup

The main deliverables are:

- a fair paper-style Grad-CAM comparison figure
- a Streamlit live demo showing the three models side by side

## Current Stage

The project is no longer blocked on training.

Current stage:

- result presentation
- Grad-CAM comparison refinement
- live demo deployment and validation

## Training Status

All three Tiny-ImageNet runs are complete.

Final checkpoints on the GPU farm:

- CE:
  - `/userhome/cs/u3651420/project/outputs/plain_ce_tiny_fresh/best.pth`
- LS:
  - `/userhome/cs/u3651420/project/outputs/plain_ls_tiny_full/best.pth`
- MaxSup:
  - `/userhome/cs/u3651420/project/outputs/plain_maxsup_tiny_full/best.pth`

Training completion was confirmed from farm checkpoints:

- CE finished 90 epochs
- LS finished 90 epochs
- MaxSup finished 90 epochs

For LS and MaxSup, `latest.pth` was checked and showed:

- `epoch = 89`
- `epoch_complete = True`

This corresponds to 90 finished epochs because epoch counting is zero-based.

## Training Pipeline

Tiny-ImageNet no longer uses the original FFCV + Lightning route on the farm.

That route was abandoned because of repeated instability:

- ffcv / cupy dependency issues
- Python 3.13 environment issues
- CUDA asserts
- Lightning teardown crashes / segfaults

The stable training path is now:

- `/Users/jayden/Desktop/7404 comp/project/scripts/Claude/train_resnet50_tiny_plain.py`

Wrapper scripts:

- `/Users/jayden/Desktop/7404 comp/project/scripts/Claude/train_resnet50_ce_tiny.sh`
- `/Users/jayden/Desktop/7404 comp/project/scripts/Claude/train_resnet50_ls_tiny.sh`
- `/Users/jayden/Desktop/7404 comp/project/scripts/Claude/train_resnet50_maxsup_tiny.sh`

This is a stable plain PyTorch controlled comparison of objectives, not a byte-identical reproduction of the original MaxSup paper training stack.

## Grad-CAM Status

Main script:

- `/Users/jayden/Desktop/7404 comp/project/scripts/Claude/gradcam_compare.py`

Current supported Grad-CAM modes:

- `--target-mode top1`
- `--target-mode gt`

Meaning:

- `top1`:
  - each model explains its own top-1 prediction
  - useful for diagnosis
- `gt`:
  - all three models explain the same ground-truth class
  - better for fair paper-style comparison

Confirmed status:

- three checkpoints load successfully
- Grad-CAM runs successfully
- latest farm copy includes the current Grad-CAM script changes
- `top1` mode smoke-tested successfully on the GPU farm
- `gt` mode smoke-tested successfully on the GPU farm
- original-image panel can display inferred ground truth

Ground-truth inference currently comes from Tiny-ImageNet-style image paths:

- parent folder wnid such as `.../val/n02480495/...`
- or filename prefix such as `n02480495_00014.JPEG`

## Label / Class-Name Status

Current class mapping JSON on the farm:

- `/userhome/cs/u3651420/project/scripts/Claude/tiny_imagenet_wnid_names.json`

Important caveat:

- label mapping is only partially human-readable right now
- in many cases it can show wnids like `n02480495`
- it may still not show full English class names like `orangutan`

Reason:

- a proper Tiny-ImageNet `words.txt` file has not yet been recovered successfully on the farm

So current figure labels are improved relative to `class_49`, but still may not be fully human-friendly.

## What The Current Figures Mean

If a figure shows MaxSup confidence lower than CE on one image, that does not automatically mean MaxSup is worse overall.

Important interpretation rules:

- the percentage in each panel title is single-image softmax confidence
- it is not validation accuracy
- `top1` mode is not a fair same-target comparison if the models predict different classes
- `gt` mode is better for final comparison figures

So:

- `top1` figures are diagnostic
- `gt` figures are better candidates for final poster/report figures

## Demo Status

Main demo script:

- `/Users/jayden/Desktop/7404 comp/project/scripts/Claude/live_demo.py`

Confirmed status:

- Streamlit demo runs on a GPU node
- image upload works
- three-model Grad-CAM inference works in the app
- top-5 predictions render correctly

Current expected deployment model:

- run Streamlit on a GPU node
- access it locally via SSH port forwarding

Do not treat the gateway as the runtime for Grad-CAM or the live demo.
Heavy inference should run on a GPU compute node, not on `gpu2gate1`.

Port-forwarding from a local laptop/browser is still worth treating as a separate validation item unless it has been explicitly re-checked.

## Farm Workflow Notes

Important operational rule:

- tmux on the gateway is different from tmux on a GPU node
- a tmux session created on a GPU node can only be attached after ssh'ing back into that same GPU node

Useful rule of thumb:

- local repo is the code source of truth
- GPU farm stores runtime artifacts and deployment copies

The farm working directory `~/project` is currently a mixed runtime directory, not a clean git working tree.
If code updates are needed on the farm, prefer:

- a clean farm checkout for git operations
- then `rsync` selected code into the runtime `~/project`

## Most Important Remaining Tasks

1. Improve label readability further if proper Tiny-ImageNet class names can be restored.
2. Select better final example images instead of random diagnostic batches.
3. Generate final Grad-CAM figures in `--target-mode gt`.
4. Verify local port forwarding for the Streamlit demo if needed.
5. Keep local code and farm deployment copies synchronized.

## Short Summary

Training is complete and stable for CE, LS, and MaxSup.
Grad-CAM now supports both diagnostic `top1` mode and fairer `gt` mode, and both have been tested successfully on the GPU farm.
The Streamlit demo has also been validated on the GPU farm.
The project is now in the final presentation phase, with the main remaining work being better labels, better example selection, final figure generation, and polished demo access workflow.
