#!/usr/bin/env python3
"""Streamlit live demo: three-way CE vs LS vs MaxSup Grad-CAM comparison.

Accepts file upload OR webcam capture. Runs the three Tiny-ImageNet models,
renders a four-panel figure (original + three heatmaps) and a top-5 table per
model.

Launch:
    streamlit run scripts/Claude/live_demo.py -- \
        --ce-ckpt weights/resnet50_ce_tiny.pth \
        --ls-ckpt weights/resnet50_ls_tiny.pth \
        --maxsup-ckpt weights/resnet50_maxsup_tiny.pth \
        --class-names scripts/Claude/tiny_imagenet_wnid_names.json

From a laptop, SSH-forward:
    ssh -L 8501:localhost:8501 u3651420@gpu2gate1.cs.hku.hk
then open http://localhost:8501 in the browser.

Dependencies:
    pip install streamlit grad-cam
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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

METHODS = [
    ("ce", "CE"),
    ("ls", "Label Smoothing"),
    ("maxsup", "MaxSup"),
]


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels)
        return self.conv(blurred.contiguous())


def apply_blurpool(mod: torch.nn.Module) -> None:
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and max(child.stride) > 1 and child.in_channels >= 16:
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def parse_cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ce-ckpt", type=Path, required=True)
    p.add_argument("--ls-ckpt", type=Path, required=True)
    p.add_argument("--maxsup-ckpt", type=Path, required=True)
    p.add_argument("--class-names", type=Path, default=None)
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Streamlit may inject its own args after "--"; ignore unknown.
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


@st.cache_resource(show_spinner="Loading models...")
def load_models(
    ce_path: str,
    ls_path: str,
    maxsup_path: str,
    num_classes: int,
    device: str,
) -> dict[str, torch.nn.Module]:
    def build(path: str) -> torch.nn.Module:
        model = tvm.resnet50(num_classes=num_classes)
        apply_blurpool(model)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sd = payload.get("state_dict", payload)
        model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        return model

    return {
        "ce": build(ce_path),
        "ls": build(ls_path),
        "maxsup": build(maxsup_path),
    }


@st.cache_resource
def load_class_names(path: str | None, num_classes: int) -> list[str]:
    if not path:
        return [f"class_{i}" for i in range(num_classes)]
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, list):
        return list(raw)
    names = [f"class_{i}" for i in range(num_classes)]
    for k, v in raw.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        if isinstance(v, dict):
            names[idx] = v.get("name") or v.get("wnid") or names[idx]
        else:
            names[idx] = str(v)
    return names


def preprocess(pil_image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    img = pil_image.convert("RGB")
    img = TF.resize(img, INPUT_SIZE, antialias=True)
    img = TF.center_crop(img, INPUT_SIZE)
    rgb_float = np.asarray(img, dtype=np.float32) / 255.0
    normalized = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, rgb_float


def gradcam_and_topk(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_float: np.ndarray,
    device: str,
    k: int = 5,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=k)
        top_pairs = list(zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()))

    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        targets = [ClassifierOutputTarget(top_pairs[0][0])]
        grayscale = cam(input_tensor=input_tensor, targets=targets)[0]

    overlay = show_cam_on_image(rgb_float, grayscale, use_rgb=True)
    return overlay, top_pairs


def render_figure(
    rgb_float: np.ndarray,
    overlays: dict[str, np.ndarray],
    top1: dict[str, tuple[str, float]],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 1 + len(METHODS), figsize=(3.2 * (1 + len(METHODS)), 3.6))
    axes[0].imshow(rgb_float)
    axes[0].set_title("Input", fontsize=11)
    axes[0].axis("off")
    for i, (key, pretty) in enumerate(METHODS, start=1):
        axes[i].imshow(overlays[key])
        label, conf = top1[key]
        axes[i].set_title(f"{pretty}\n{label} ({conf*100:.1f}%)", fontsize=11)
        axes[i].axis("off")
    plt.tight_layout()
    return fig


def main() -> None:
    cli = parse_cli_args()
    st.set_page_config(page_title="MaxSup vs LS vs CE — Grad-CAM", layout="wide")
    st.title("MaxSup / Label Smoothing / CE — Grad-CAM comparison")
    st.caption(
        "ResNet-50 trained on Tiny-ImageNet. "
        "Upload an image or capture one from your webcam."
    )

    class_names = load_class_names(str(cli.class_names) if cli.class_names else None, cli.num_classes)
    models = load_models(
        str(cli.ce_ckpt), str(cli.ls_ckpt), str(cli.maxsup_ckpt),
        cli.num_classes, cli.device,
    )

    with st.sidebar:
        st.header("Input")
        mode = st.radio("Source", ["Upload", "Webcam"], horizontal=True)
        uploaded: io.BytesIO | None = None
        if mode == "Upload":
            uploaded = st.file_uploader("Image file", type=["png", "jpg", "jpeg", "webp", "bmp"])
        else:
            uploaded = st.camera_input("Take a picture")
        st.markdown(f"**Device:** `{cli.device}`")
        st.markdown("**Models:**")
        st.code(f"CE     -> {cli.ce_ckpt}\nLS     -> {cli.ls_ckpt}\nMaxSup -> {cli.maxsup_ckpt}")

    if uploaded is None:
        st.info("Upload or capture an image to run inference.")
        return

    pil = Image.open(uploaded)
    input_tensor, rgb_float = preprocess(pil)

    overlays: dict[str, np.ndarray] = {}
    top1: dict[str, tuple[str, float]] = {}
    topk_tables: dict[str, pd.DataFrame] = {}

    progress = st.progress(0.0, text="Running inference...")
    for i, (key, pretty) in enumerate(METHODS):
        overlay, pairs = gradcam_and_topk(models[key], input_tensor, rgb_float, cli.device)
        overlays[key] = overlay
        idx, conf = pairs[0]
        top1[key] = (class_names[idx] if idx < len(class_names) else str(idx), conf)
        topk_tables[pretty] = pd.DataFrame(
            [(class_names[j] if j < len(class_names) else str(j), float(p)) for j, p in pairs],
            columns=["class", "prob"],
        )
        progress.progress((i + 1) / len(METHODS), text=f"Done: {pretty}")

    progress.empty()
    fig = render_figure(rgb_float, overlays, top1)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Top-5 per model")
    cols = st.columns(len(METHODS))
    for col, (_, pretty) in zip(cols, METHODS):
        col.markdown(f"**{pretty}**")
        df = topk_tables[pretty].copy()
        df["prob"] = df["prob"].map(lambda x: f"{x*100:.2f}%")
        col.dataframe(df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
