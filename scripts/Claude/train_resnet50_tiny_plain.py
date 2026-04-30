#!/usr/bin/env python3
"""Stable Tiny-ImageNet ResNet-50 trainer without FFCV/Lightning.

This script exists as a pragmatic fallback for environments where the author's
FFCV + Lightning stack is unstable. It keeps the core comparison intact:
ResNet-50 + BlurPool, same optimizer family, same LR schedule, and the same CE /
LS / MaxSup losses.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
FFCV_DIR = REPO_ROOT / "maxsup_repo" / "Conv" / "ffcv"
if str(FFCV_DIR) not in sys.path:
    sys.path.insert(0, str(FFCV_DIR))

from losses import MaxSuppression  # noqa: E402


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n


class BlurPoolConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        default_filter = torch.tensor(
            [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]],
            dtype=conv.weight.dtype,
        ) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
        )
        return self.conv(blurred.contiguous())


def apply_blurpool(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and max(child.stride) > 1 and child.in_channels >= 16:
            setattr(module, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


class ProgressBar:
    def __init__(self, total: int, prefix: str) -> None:
        self.total = max(total, 1)
        self.prefix = prefix
        self.start_time = time.time()
        self.last_len = 0

    def update(self, step: int, **metrics: float) -> None:
        width = min(28, max(10, shutil.get_terminal_size((100, 20)).columns - 72))
        ratio = min(max(step / self.total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.time() - self.start_time
        eta = elapsed / step * (self.total - step) if step else 0.0
        metric_str = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        message = (
            f"\r{self.prefix} [{bar}] {step:>4}/{self.total:<4} "
            f"eta={eta:>6.1f}s {metric_str}"
        )
        pad = max(0, self.last_len - len(message))
        sys.stdout.write(message + (" " * pad))
        sys.stdout.flush()
        self.last_len = len(message)

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    with torch.no_grad():
        maxk = min(max(topk), logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results: list[float] = []
        for k in topk:
            k = min(k, logits.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(correct_k.mul_(100.0 / target.size(0)).item())
        return results


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader[Any], DataLoader[Any], int]:
    train_dir = Path(args.data_path) / "train"
    val_dir = Path(args.data_path) / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageFolder data at {train_dir} and {val_dir}.")

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.train_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(args.val_resize_size),
            transforms.CenterCrop(args.val_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    if train_ds.classes != val_ds.classes:
        raise ValueError("Train/val class folders do not match.")

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, len(train_ds.classes)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None, num_classes=num_classes)
    apply_blurpool(model)
    return model


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss_type == "ce":
        return nn.CrossEntropyLoss()
    if args.loss_type == "ls":
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.loss_type == "ms":
        return MaxSuppression(
            begin_lambda=args.ms_begin_lambda,
            end_lambda=args.ms_end_lambda,
            epochs=args.epochs,
        )
    raise ValueError(f"Unsupported loss type: {args.loss_type}")


def resolve_amp_dtype(device: torch.device, amp_mode: str) -> torch.dtype | None:
    if device.type != "cuda" or amp_mode == "none":
        return None
    if amp_mode == "fp16":
        return torch.float16
    if amp_mode == "bf16":
        return torch.bfloat16
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_autocast(device: torch.device, amp_dtype: torch.dtype | None) -> contextlib.AbstractContextManager[Any]:
    if amp_dtype is None:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def build_grad_scaler(device: torch.device, amp_dtype: torch.dtype | None) -> torch.amp.GradScaler | None:
    if device.type != "cuda" or amp_dtype != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def build_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.SGD(
        [
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": decay_params, "weight_decay": weight_decay},
        ],
        lr=lr,
        momentum=momentum,
    )


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def atomic_copy(src: Path, dst: Path) -> None:
    tmp_path = dst.with_name(f"{dst.name}.tmp")
    shutil.copy2(src, tmp_path)
    os.replace(tmp_path, dst)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    epoch_complete: bool,
    step_in_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    best_acc1: float,
    args: argparse.Namespace,
) -> None:
    state_dict = model.state_dict()
    state = {
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "step_in_epoch": step_in_epoch,
        "model": state_dict,
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc1": best_acc1,
        "args": vars(args),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    atomic_torch_save(state, path)


def append_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    epochs: int,
    amp_dtype: torch.dtype | None,
    should_exit: dict[str, bool],
    save_partial_checkpoint: Any,
    mid_epoch_save_interval: int,
) -> dict[str, float]:
    model.train()
    if hasattr(criterion, "set_current_epoch"):
        criterion.set_current_epoch(epoch)
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    bar = ProgressBar(len(loader), prefix=f"Train {epoch + 1}/{epochs}")

    for step, (images, target) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with get_autocast(device, amp_dtype):
            logits = model(images)
            loss = criterion(logits, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc5 = topk_accuracy(logits, target)
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1, batch_size)
        acc5_meter.update(acc5, batch_size)
        bar.update(step, loss=loss_meter.avg, acc1=acc1_meter.avg, lr=optimizer.param_groups[0]["lr"])

        if mid_epoch_save_interval > 0 and step % mid_epoch_save_interval == 0:
            save_partial_checkpoint(step)

        if should_exit["flag"]:
            save_partial_checkpoint(step)
            bar.close()
            return {
                "train_loss": loss_meter.avg,
                "train_acc1": acc1_meter.avg,
                "train_acc5": acc5_meter.avg,
                "interrupted": True,
                "steps_done": step,
            }

    bar.close()
    return {
        "train_loss": loss_meter.avg,
        "train_acc1": acc1_meter.avg,
        "train_acc5": acc5_meter.avg,
        "interrupted": False,
        "steps_done": len(loader),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    bar = ProgressBar(len(loader), prefix=f"Val   {epoch + 1}/{epochs}")

    ce_criterion = nn.CrossEntropyLoss()
    for step, (images, target) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        with get_autocast(device, amp_dtype):
            logits = model(images)
            loss = ce_criterion(logits, target)
        acc1, acc5 = topk_accuracy(logits, target)
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1, batch_size)
        acc5_meter.update(acc5, batch_size)
        bar.update(step, loss=loss_meter.avg, acc1=acc1_meter.avg)

    bar.close()
    return {
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Tiny-ImageNet ResNet-50 trainer.")
    parser.add_argument("--data-path", default=str(Path.home() / "data" / "tiny_imagenet"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loss-type", choices=["ce", "ls", "ms"], required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", default="latest", help="Checkpoint path, 'latest', or 'none'")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--mid-epoch-save-interval", type=int, default=200)
    parser.add_argument("--amp", default="none", choices=["auto", "bf16", "fp16", "none"])
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--train-crop-size", type=int, default=224)
    parser.add_argument("--val-resize-size", type=int, default=256)
    parser.add_argument("--val-crop-size", type=int, default=224)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--ms-begin-lambda", type=float, default=0.1)
    parser.add_argument("--ms-end-lambda", type=float, default=0.2)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    parser.add_argument("--std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = output_dir / "latest.pth"
    best_ckpt = output_dir / "best.pth"
    log_path = output_dir / "train_log.jsonl"

    seed_everything(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    should_exit = {"flag": False}

    def _handle_signal(signum: int, _frame: Any) -> None:
        should_exit["flag"] = True
        print(f"\nReceived signal {signum}. Saving latest checkpoint before exit...", flush=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_signal)

    train_loader, val_loader, num_classes = build_dataloaders(args)
    model = build_model(num_classes).to(device)
    criterion = build_criterion(args)

    optimizer = build_optimizer(model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    amp_dtype = resolve_amp_dtype(device, args.amp)
    scaler = build_grad_scaler(device, amp_dtype)

    start_epoch = 0
    best_acc1 = 0.0
    resume_path = None
    if args.resume == "latest":
        resume_path = latest_ckpt if latest_ckpt.exists() else None
    elif args.resume and args.resume != "none":
        resume_path = Path(args.resume)

    if resume_path:
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("epoch_complete", True):
            start_epoch = int(checkpoint["epoch"]) + 1
        else:
            start_epoch = int(checkpoint["epoch"])
        best_acc1 = float(checkpoint.get("best_acc1", 0.0))

    config_payload = {
        "event": "config",
        "experiment_name": args.experiment_name,
        "loss_type": args.loss_type,
        "data_path": str(args.data_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "amp": args.amp,
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
        "resume_from": str(resume_path) if resume_path else None,
    }
    print(json.dumps(config_payload, indent=2))
    append_log(log_path, config_payload)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        def save_partial_checkpoint(step_in_epoch: int) -> None:
            save_checkpoint(
                latest_ckpt,
                epoch=epoch,
                epoch_complete=False,
                step_in_epoch=step_in_epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_acc1=best_acc1,
                args=args,
            )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            epoch=epoch,
            epochs=args.epochs,
            amp_dtype=amp_dtype,
            should_exit=should_exit,
            save_partial_checkpoint=save_partial_checkpoint,
            mid_epoch_save_interval=args.mid_epoch_save_interval,
        )
        if train_metrics["interrupted"]:
            interrupted_payload = {
                "event": "signal_exit",
                "epoch": epoch,
                "steps_done": train_metrics["steps_done"],
                "train_loss": train_metrics["train_loss"],
                "train_acc1": train_metrics["train_acc1"],
                "train_acc5": train_metrics["train_acc5"],
            }
            print(json.dumps(interrupted_payload))
            append_log(log_path, interrupted_payload)
            sys.exit(0)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            amp_dtype=amp_dtype,
        )
        scheduler.step()

        elapsed = time.time() - epoch_start
        metrics = {
            "event": "epoch_end",
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 2),
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
        }
        print(json.dumps(metrics))
        append_log(log_path, metrics)

        is_best = val_metrics["val_acc1"] > best_acc1
        best_acc1 = max(best_acc1, val_metrics["val_acc1"])
        save_checkpoint(
            latest_ckpt,
            epoch=epoch,
            epoch_complete=True,
            step_in_epoch=len(train_loader),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_acc1=best_acc1,
            args=args,
        )

        if is_best:
            atomic_copy(latest_ckpt, best_ckpt)
            print(f"Saved new best checkpoint: {best_ckpt} (acc1={best_acc1:.3f})")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            periodic_ckpt = output_dir / f"epoch_{epoch + 1:03d}.pth"
            atomic_copy(latest_ckpt, periodic_ckpt)

    print(f"Training finished. Best val acc1: {best_acc1:.3f}")


if __name__ == "__main__":
    main()
