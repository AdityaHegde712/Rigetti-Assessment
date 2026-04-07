"""
ml/test.py
----------
Standalone inference script.  Loads the best (or latest) checkpoint from
the most recent run directory and evaluates on the test set.

Usage (from project root):
    python ml/test.py                          # uses most recent run
    python ml/test.py --run-dir ml/runs/<ts>   # explicit run directory
    python ml/test.py --checkpoint path/to/checkpoint.pth

Outputs (written to the resolved run directory):
    predictions.csv       (also copied to project root)
    confusion_matrix.png
    report.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import Config
from ml.dataset import get_dataloaders
from ml.model import load_model, test


# CLI 

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained defect-classifier on the test split."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific run directory (default: most recent run).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a specific .pth checkpoint (overrides --run-dir lookup).",
    )
    return parser.parse_args()


# Helpers 

def _resolve_run_dir(cfg: Config, args: argparse.Namespace) -> Path:
    """Return the run directory to use for outputs and checkpoint lookup."""
    if args.run_dir is not None:
        if not args.run_dir.is_dir():
            raise NotADirectoryError(f"run-dir not found: {args.run_dir}")
        return args.run_dir

    if not cfg.runs_dir.exists() or not any(cfg.runs_dir.iterdir()):
        raise FileNotFoundError(
            f"No runs found in {cfg.runs_dir}. Train a model first with "
            "`python ml/train.py`."
        )
    # Pick most recent (runs are named by timestamp)
    return sorted(cfg.runs_dir.iterdir())[-1]


def _resolve_checkpoint(run_dir: Path, args: argparse.Namespace) -> Path:
    """Return the checkpoint (.pth) file to load."""
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        return args.checkpoint

    for name in ("best.pth", "latest.pth"):
        ckpt = run_dir / name
        if ckpt.exists():
            return ckpt

    raise FileNotFoundError(
        f"No checkpoint found in {run_dir}. "
        "Expected best.pth or latest.pth."
    )


# Main 

def main() -> None:
    args = _parse_args()
    cfg  = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Resolve paths 
    run_dir = _resolve_run_dir(cfg, args)
    ckpt    = _resolve_checkpoint(run_dir, args)
    print(f"Run dir    : {run_dir}")
    print(f"Checkpoint : {ckpt}")

    # Data (only test split is used) 
    print("\nLoading test dataset …")
    _, _, test_loader = get_dataloaders(cfg)
    print(f"  test={len(test_loader.dataset):,} images")

    # Model 
    model, _ = load_model(ckpt, cfg, device)

    # Inference & reporting 
    test(test_loader, device, cfg, run_dir, model=model)


if __name__ == "__main__":
    main()
