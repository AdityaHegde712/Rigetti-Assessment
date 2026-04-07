"""
ml/model.py
-----------
Model definition and all core functions:
  build_model        - EfficientNet-B0 with replaced classification head
  train_one_epoch    - one full pass over the training loader
  evaluate           - loss + accuracy on any loader (val / test)
  test               - full inference pipeline: predictions.csv, confusion
                       matrix, per-class accuracy, report.txt
  save_model         - checkpoint (weights + full training state)
  load_model         - restore model (+ optionally full state) from checkpoint

The test() function optionally accepts a pre-loaded model; if omitted it
loads best.pth (falling back to latest.pth) from run_dir automatically.
"""

from __future__ import annotations

import csv
import matplotlib
matplotlib.use("Agg")           # headless – must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from ml.config import Config


# Model builder 
# Example of a custom CNN architecture (commented out)
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=5):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             # Block 1: 3 -> 16 | 224x224 -> 112x112
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # Block 2: 16 -> 32 | 112x112 -> 56x56
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # Block 3: 32 -> 64 | 56x56 -> 28x28
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # Block 4: 64 -> 128 | 28x28 -> 14x14
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Output: 128 x 7 x 7
#         self.classifier = nn.Sequential(
#             # First Aggressive Reduction: 6,272 -> 1,024
#             nn.Linear(128 * 7 * 7, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             # Second Aggressive Reduction: 1,024 -> 256
#             nn.Linear(1024, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             # Final Classification: 256 -> 5
#             nn.Linear(256, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


def build_model(cfg: Config, pretrained: bool = True) -> nn.Module:
    """Return MobileNet-V3-Small with a task-specific classification head."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, cfg.num_classes)
    return model


# Training 

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """One full training pass.

    Returns:
        (avg_loss, accuracy)  both as plain floats.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    pbar = tqdm(loader, desc="Training", leave=True)
    for images, labels, _ in pbar:          # _ = relative path strings
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:                # AMP path (CUDA only)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:                                 # standard fp32 path
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * images.size(0)
        _, predicted = outputs.max(1)
        batch_correct = predicted.eq(labels).sum().item()
        correct += batch_correct
        n += labels.size(0)

        pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "acc": f"{batch_correct / labels.size(0):.2%}"
        })

    return total_loss / n, correct / n


# Evaluation 

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute loss and accuracy on *any* loader without updating weights.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=True)
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_loss = loss.item()
            total_loss += batch_loss * images.size(0)
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            correct += batch_correct
            n += labels.size(0)

            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "acc": f"{batch_correct / labels.size(0):.2%}"
            })

    return total_loss / n, correct / n


# Test & reporting 

def test(
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    run_dir: Path,
    model: Optional[nn.Module] = None,
) -> Dict:
    """Full inference pipeline on the test loader.

    If *model* is None the function loads best.pth (or latest.pth) from
    run_dir automatically.

    Outputs written to run_dir:
      • predictions.csv     (also copied to project root)
      • confusion_matrix.png
      • report.txt

    Returns:
        {"accuracy": float, "per_class": {class_name: float}}
    """
    if model is None:
        ckpt = run_dir / "best.pth"
        if not ckpt.exists():
            ckpt = run_dir / "latest.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
        model, _ = load_model(ckpt, cfg, device)

    model.eval()
    all_preds:  List[int] = []
    all_labels: List[int] = []
    all_paths:  List[str] = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    cm = confusion_matrix(
        all_labels, all_preds, labels=list(range(cfg.num_classes))
    )
    per_class: Dict[str, float] = {
        cls: (cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0)
        for i, cls in enumerate(cfg.classes)
    }

    _write_predictions(all_paths, all_preds, cfg, run_dir)
    _save_confusion_matrix(cm, cfg.classes, run_dir)
    _write_report(accuracy, per_class, run_dir)

    # Console summary 
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print(sep)
    print("  Per-class Accuracy:")
    for cls, acc in per_class.items():
        print(f"    {cls:<12}: {acc * 100:.2f}%")
    print(f"{sep}\n")

    return {"accuracy": accuracy, "per_class": per_class}


# Private output helpers 

def _write_predictions(
    paths: List[str],
    preds: List[int],
    cfg: Config,
    run_dir: Path,
) -> None:
    rows = [("image_path", "predicted_label")] + [
        (p, cfg.classes[pred]) for p, pred in zip(paths, preds)
    ]
    for dest in (
        run_dir / "predictions.csv",
        cfg.project_root / "predictions.csv",
    ):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"  predictions.csv  → {dest}")


def _save_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    run_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    dest = run_dir / "confusion_matrix.png"
    plt.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"  confusion_matrix.png → {dest}")


def _write_report(
    accuracy: float,
    per_class: Dict[str, float],
    run_dir: Path,
) -> None:
    sep = "=" * 52
    lines = [
        sep,
        f"Test Accuracy : {accuracy * 100:.2f}%",
        sep,
        "Per-Class Accuracy:",
    ] + [f"  {cls:<12}: {acc * 100:.2f}%" for cls, acc in per_class.items()] + [sep]
    dest = run_dir / "report.txt"
    dest.write_text("\n".join(lines) + "\n")
    print(f"  report.txt           → {dest}")


# Checkpoint I/O 

def save_model(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    cfg: Config,
    scaler: Optional[torch.amp.GradScaler] = None,
    filename: str = "checkpoint.pth",
) -> None:
    """Save complete training state for resume-capable checkpoints."""
    checkpoint = {
        "epoch":               epoch,
        "val_loss":            val_loss,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "scheduler_state_dict":scheduler.state_dict(),
        "scaler_state_dict":   scaler.state_dict() if scaler is not None else None,
        "num_classes":         cfg.num_classes,
        "classes":             cfg.classes,
    }
    torch.save(checkpoint, run_dir / filename)


def load_model(
    checkpoint_path: Path,
    cfg: Config,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    """Restore model from a checkpoint.

    Returns:
        (model, checkpoint_dict)
        The raw dict is returned so callers can optionally restore
        optimiser / scheduler state for training resumption.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
