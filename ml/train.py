"""
ml/train.py
-----------
End-to-end training script.

Usage (from project root):
    python ml/train.py

Each invocation creates a fresh timestamped run directory:
    ml/runs/<YYYYMMDD_HHMMSS>/
        metrics.csv          — epoch-by-epoch metrics
        best.pth             — checkpoint with lowest val_loss
        latest.pth           — checkpoint from the last completed epoch
        predictions.csv      — written after training completes
        confusion_matrix.png
        report.txt

predictions.csv is also copied to the project root for submission.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# Allow `python ml/train.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import Config
from ml.dataset import get_dataloaders
from ml.model import (
    build_model,
    evaluate,
    load_model,
    save_model,
    test,
    train_one_epoch,
)


# Helpers 
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device : GPU — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device : CPU  (CUDA not available)")
    return device


def _create_run_dir(cfg: Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _init_metrics_csv(run_dir: Path) -> Path:
    path = run_dir / "metrics.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"]
        )
    return path


def _log_metrics(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
             f"{train_acc:.6f}", f"{val_acc:.6f}"]
        )


# Main 
def main() -> None:
    cfg = Config()
    device = _get_device()

    # Data 
    print("\nLoading dataset …")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    print(
        f"  train={len(train_loader.dataset):,}  "
        f"val={len(val_loader.dataset):,}  "
        f"test={len(test_loader.dataset):,}"
    )

    # Model 
    model = build_model(cfg, pretrained=True).to(device)

    # Optimiser & scheduler 
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed precision (CUDA only) 
    use_amp = cfg.use_amp and (device.type == "cuda")
    # scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f"Mixed precision AMP : {'enabled' if use_amp else 'disabled'}")

    # Run directory 
    run_dir      = _create_run_dir(cfg)
    metrics_path = _init_metrics_csv(run_dir)
    print(f"Run directory       : {run_dir}\n")

    # Training loop 
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        _log_metrics(
            metrics_path, epoch,
            train_loss, val_loss, train_acc, val_acc
        )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%"
        )

        # Always save latest so training can be inspected / resumed
        save_model(
            run_dir, model, optimizer, scheduler, epoch, val_loss, cfg,
            scaler=scaler, filename="latest.pth",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            save_model(
                run_dir, model, optimizer, scheduler, epoch, val_loss, cfg,
                scaler=scaler, filename="best.pth",
            )
            print(f"  ✔ New best  val_loss={val_loss:.4f}  →  best.pth")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"\n  Early stopping after {epoch} epochs "
                      f"(no improvement for {cfg.patience} epochs).")
                break
        print() # Extra space between epochs

    # Test evaluation 
    print("\nRunning test evaluation on best checkpoint …")
    best_model, _ = load_model(run_dir / "best.pth", cfg, device)
    test(test_loader, device, cfg, run_dir, model=best_model)


if __name__ == "__main__":
    main()
