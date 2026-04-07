"""
ml/config.py
------------
Central configuration for the defect-classification pipeline.
All hyperparameters and path settings live here; nothing is hard-coded
elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """Pipeline-wide configuration.

    Paths are derived from the file location at runtime so the project
    can be cloned anywhere without changes.
    """

    # Paths (resolved in __post_init__) 
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    dataset_dir: Optional[Path] = None
    train_dir: Optional[Path] = None
    test_dir: Optional[Path] = None
    runs_dir: Optional[Path] = None

    # Classes (alphabetical → stable label indices) 
    classes: List[str] = field(
        default_factory=lambda: ["crack", "hole", "normal", "rust", "scratch"]
    )
    num_classes: int = 5

    # Image 
    image_size: int = 224

    # DataLoader 
    batch_size: int = 32
    num_workers: int = 1
    pin_memory: bool = True

    # Optimiser 
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Scheduler 
    epochs: int = 100

    # Early stopping 
    patience: int = 10

    # Data split 
    # Fraction of train/ held out as validation (stratified)
    val_split: float = 0.2   # 80 / 20

    # Mixed precision 
    # Activated only when CUDA is available; silently disabled on CPU.
    use_amp: bool = True

    # Reproducibility 
    seed: int = 42

    # 
    def __post_init__(self) -> None:
        if self.dataset_dir is None:
            self.dataset_dir = self.project_root / "dataset"
        if self.train_dir is None:
            self.train_dir = self.dataset_dir / "train"
        if self.test_dir is None:
            self.test_dir = self.dataset_dir / "test"
        if self.runs_dir is None:
            self.runs_dir = self.project_root / "ml" / "runs"
