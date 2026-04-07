"""
ml/visualize_logs.py
--------------------
Utility script to visualize training metrics (loss and accuracy) from
a run's metrics.csv file.

Usage:
    python ml/visualize_logs.py ml/runs/<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Allow execution from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def visualize_run(run_dir: Path) -> None:
    """Read metrics.csv from run_dir and save 2x2 training curves PNG."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found.")
        return

    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_accuracy"]))
            val_acc.append(float(row["val_accuracy"]))

    if not epochs:
        print(f"Error: No data found in {metrics_path}.")
        return

    # ── Plotting ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Metrics — Run: {run_dir.name}", fontsize=16)

    # Top-Left: Train Loss
    axes[0, 0].plot(epochs, train_loss, color="tab:blue", linewidth=2)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.7)

    # Top-Right: Val Loss
    axes[0, 1].plot(epochs, val_loss, color="tab:red", linewidth=2)
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, linestyle="--", alpha=0.7)

    # Bottom-Left: Train Accuracy
    axes[1, 0].plot(epochs, [a * 100 for a in train_acc], color="tab:green", linewidth=2)
    axes[1, 0].set_title("Training Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.7)

    # Bottom-Right: Val Accuracy
    axes[1, 1].plot(epochs, [a * 100 for a in val_acc], color="tab:orange", linewidth=2)
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = run_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize metrics from a run folder.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the timestamped run folder (e.g., ml/runs/20260407_120000).",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"Error: {args.run_dir} is not a directory.")
        sys.exit(1)

    visualize_run(args.run_dir)


if __name__ == "__main__":
    main()
