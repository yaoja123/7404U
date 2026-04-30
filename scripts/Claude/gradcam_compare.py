#!/usr/bin/env python3
"""Paper-style Grad-CAM comparison for CE / LS / MaxSup ResNet-50 on Tiny-ImageNet.

Loads three checkpoints, applies the paper's BlurPool wrapper, runs Grad-CAM at
`model.layer4[-1]`, and renders a grid:

    rows    = input images
    columns = [original, CE heatmap, LS heatmap, MaxSup heatmap]

Each heatmap panel is annotated with the model's top-1 prediction + confidence.

Usage:
    conda run -n tinyffcv python scripts/Claude/gradcam_compare.py \
        --images demo/*.jpg \
        --ce-ckpt outputs/plain_ce_tiny/best.pth \
        --ls-ckpt outputs/plain_ls_tiny/best.pth \
        --maxsup-ckpt outputs/plain_maxsup_tiny/best.pth \
        --output tmp/gradcam_paper_style.png

Dependencies:
    pip install grad-cam  (provides `pytorch_grad_cam`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
INPUT_SIZE = 224


class BlurPoolConv2d(torch.nn.Module):
    """Copy of maxsup_repo/Conv/ffcv/model.py::BlurPoolConv2d so we can
    reconstruct the same module graph that Lightning checkpoints were saved with.
    """

    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        blurred = F.conv2d(
            x, self.blur_filter, stride=1, padding=(1, 1),
            groups=self.conv.in_channels,
        )
        return self.conv(blurred.contiguous())


def apply_blurpool(mod: torch.nn.Module) -> None:
    """Mirror of maxsup_repo/Conv/ffcv/model.py::Net._apply_blurpool (L55-67)."""
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and max(child.stride) > 1 and child.in_channels >= 16:
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", nargs="+", type=Path, required=True)
    p.add_argument("--ce-ckpt", type=Path, required=True)
    p.add_argument("--ls-ckpt", type=Path, required=True)
    p.add_argument("--maxsup-ckpt", type=Path, required=True)
    p.add_argument("--class-names", type=Path, default=None,
                   help="Optional JSON mapping idx(str) -> {wnid, name} or idx(str)->name.")
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--target-mode", choices=["top1", "gt"], default="top1",
                   help="Grad-CAM target selection: each model's own top-1, or shared ground-truth class.")
    p.add_argument("--output", type=Path, default=Path("tmp/gradcam_paper_style.png"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def build_model(ckpt_path: Path, num_classes: int, device: str) -> torch.nn.Module:
    model = tvm.resnet50(num_classes=num_classes)
    apply_blurpool(model)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[{ckpt_path.name}] unexpected keys ({len(unexpected)}): first few = {unexpected[:3]}")
    if missing:
        print(f"[{ckpt_path.name}] missing keys ({len(missing)}): first few = {missing[:3]}")
    model.eval().to(device)
    return model


def load_class_metadata(path: Path | None, num_classes: int) -> tuple[list[str], dict[int, str], dict[str, str]]:
    names = [f"class_{i}" for i in range(num_classes)]
    idx_to_wnid: dict[int, str] = {}
    wnid_to_name: dict[str, str] = {}
    if path is None:
        return names, idx_to_wnid, wnid_to_name

    raw = json.loads(path.read_text())
    # Support either {idx: name} or {idx: {name, wnid}} or a list.
    if isinstance(raw, list):
        names = list(raw)
        return names, idx_to_wnid, wnid_to_name

    for k, v in raw.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        if isinstance(v, dict):
            wnid = str(v.get("wnid", "")).strip()
            name = str(v.get("name") or wnid or names[idx]).strip()
            names[idx] = name
            if wnid:
                idx_to_wnid[idx] = wnid
                wnid_to_name[wnid] = name
        else:
            names[idx] = str(v)
    return names, idx_to_wnid, wnid_to_name


def infer_ground_truth_label(image_path: Path, wnid_to_name: dict[str, str]) -> str | None:
    parent = image_path.parent.name
    if parent.startswith("n"):
        return wnid_to_name.get(parent, parent)

    stem = image_path.stem
    if "_" in stem:
        wnid = stem.split("_", 1)[0]
        if wnid.startswith("n"):
            return wnid_to_name.get(wnid, wnid)
    return None


def infer_ground_truth_wnid(image_path: Path) -> str | None:
    parent = image_path.parent.name
    if parent.startswith("n"):
        return parent

    stem = image_path.stem
    if "_" in stem:
        wnid = stem.split("_", 1)[0]
        if wnid.startswith("n"):
            return wnid
    return None


def preprocess(image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(image_path).convert("RGB")
    img = TF.resize(img, INPUT_SIZE, antialias=True)
    img = TF.center_crop(img, INPUT_SIZE)
    rgb_float = np.asarray(img, dtype=np.float32) / 255.0
    normalized = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, rgb_float


def run_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    device: str,
    target_idx: int | None = None,
) -> tuple[np.ndarray, int, float, int]:
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    cam_target_idx = int(pred.item()) if target_idx is None else int(target_idx)

    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        targets = [ClassifierOutputTarget(cam_target_idx)]
        grayscale = cam(input_tensor=input_tensor, targets=targets)[0]

    overlay = show_cam_on_image(rgb_float, grayscale, use_rgb=True)
    return overlay, int(pred.item()), float(conf.item()), cam_target_idx


METHODS = [
    ("ce", "CE"),
    ("ls", "Label Smoothing"),
    ("maxsup", "MaxSup"),
]


def main() -> None:
    args = parse_args()
    class_names, idx_to_wnid, wnid_to_name = load_class_metadata(args.class_names, args.num_classes)
    wnid_to_idx = {wnid: idx for idx, wnid in idx_to_wnid.items()}

    ckpts = {
        "ce": args.ce_ckpt,
        "ls": args.ls_ckpt,
        "maxsup": args.maxsup_ckpt,
    }
    print("Loading models...")
    models = {k: build_model(v, args.num_classes, args.device) for k, v in ckpts.items()}

    rows = len(args.images)
    cols = 1 + len(METHODS)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)

    for r, image_path in enumerate(args.images):
        input_tensor, rgb_float = preprocess(image_path)
        gt_label = infer_ground_truth_label(image_path, wnid_to_name)
        gt_wnid = infer_ground_truth_wnid(image_path)
        shared_target_idx: int | None = None
        if args.target_mode == "gt":
            if gt_wnid is None:
                raise ValueError(
                    f"Could not infer ground-truth wnid from image path: {image_path}. "
                    "Expected Tiny-ImageNet-style parent folder or filename prefix."
                )
            if gt_wnid not in wnid_to_idx:
                raise ValueError(
                    f"Ground-truth wnid {gt_wnid} was inferred from {image_path}, "
                    "but it is not present in the provided class mapping JSON."
                )
            shared_target_idx = wnid_to_idx[gt_wnid]

        axes[r, 0].imshow(rgb_float)
        original_title = image_path.name
        if gt_label:
            original_title = f"{image_path.name}\nGT: {gt_label}"
        axes[r, 0].set_title(original_title, fontsize=10)
        axes[r, 0].axis("off")

        for c, (key, pretty) in enumerate(METHODS, start=1):
            overlay, pred_idx, conf, cam_target_idx = run_gradcam(
                models[key],
                input_tensor,
                rgb_float,
                args.device,
                target_idx=shared_target_idx,
            )
            axes[r, c].imshow(overlay)
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            if args.target_mode == "gt":
                target_label = class_names[cam_target_idx] if cam_target_idx < len(class_names) else str(cam_target_idx)
                title = (
                    f"{pretty} [GT CAM]\n"
                    f"pred: {pred_label} ({conf * 100:.1f}%)\n"
                    f"target: {target_label}"
                )
            else:
                title = f"{pretty}\n{pred_label} ({conf * 100:.1f}%)"
            axes[r, c].set_title(title, fontsize=10)
            axes[r, c].axis("off")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()
