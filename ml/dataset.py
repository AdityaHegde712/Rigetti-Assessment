"""
ml/dataset.py
-------------
Dataset class and DataLoader factory for the surface-defect pipeline.

Directory layout expected:
    dataset/
        train/
            crack/  hole/  normal/  rust/  scratch/
        test/
            crack/  hole/  normal/  rust/  scratch/

The train/ directory is split 80/20 (stratified) into train and validation.
The test/ directory is kept as the held-out test set.

Every __getitem__ returns (image_tensor, label_int, rel_path_str) where
rel_path_str is relative to cfg.dataset_dir, e.g. "test/crack/img_001.png".
pin_memory=True is safe here: PyTorch's default collate passes Python strings
through without attempting to pin them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.config import Config


# Internal helpers 

def _collect_samples(
    root: Path, classes: List[str]
) -> List[Tuple[Path, int]]:
    """Return sorted (abs_path, label_idx) pairs for all images under root."""
    samples: List[Tuple[Path, int]] = []
    for idx, cls in enumerate(classes):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {cls_dir}")
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_path in sorted(cls_dir.glob(ext)):
                samples.append((img_path, idx))
    return samples


# Transforms 

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(train: bool) -> transforms.Compose:
    # if train:
    #     return transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip(),
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    #         transforms.ToTensor(),
    #         transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    #     ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# Dataset 

class DefectDataset(Dataset):
    """Surface-defect image dataset.

    Args:
        samples:     list of (abs_path, label_idx) pairs.
        dataset_dir: root used to compute relative paths for predictions.csv.
        transform:   torchvision transform applied to each image.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        dataset_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.samples = samples
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, str]:
        abs_path, label = self.samples[idx]
        image = Image.open(abs_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # Forward-slash for cross-platform consistency in CSV output
        rel_path = str(abs_path.relative_to(self.dataset_dir)).replace("\\", "/")
        return image, label, rel_path


# DataLoader factory 

def _make_loader(
    dataset: DefectDataset, cfg: Config, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


def get_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders.

    * train/  → 80 % train  +  20 % validation  (stratified, seeded)
    * test/   → held-out test set (unchanged)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    all_train = _collect_samples(cfg.train_dir, cfg.classes)
    test_samples = _collect_samples(cfg.test_dir, cfg.classes)

    labels = [s[1] for s in all_train]
    train_samples, val_samples = train_test_split(
        all_train,
        test_size=cfg.val_split,
        stratify=labels,
        random_state=cfg.seed,
    )

    train_ds = DefectDataset(
        train_samples, cfg.dataset_dir, build_transforms(train=True)
    )
    val_ds = DefectDataset(
        val_samples, cfg.dataset_dir, build_transforms(train=False)
    )
    test_ds = DefectDataset(
        test_samples, cfg.dataset_dir, build_transforms(train=False)
    )

    return (
        _make_loader(train_ds, cfg, shuffle=True),
        _make_loader(val_ds,   cfg, shuffle=False),
        _make_loader(test_ds,  cfg, shuffle=False),
    )
