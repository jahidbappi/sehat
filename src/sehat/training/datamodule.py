"""Lightning data module built on the Sehat manifest contract.

Consumes manifests via ``sehat.data.manifest.load_manifest`` (owned by the
data-layer worker) and adds image decoding, transforms, and class-imbalance
sampling on top. Requires torch, torchvision, and Pillow at run time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sehat.training.config import TrainConfig

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - legacy package name
    import pytorch_lightning as pl

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["ManifestDataModule"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestDataModule(pl.LightningDataModule):
    """Serves train/val/test loaders from a Sehat manifest CSV.

    The manifest is filtered to ``config.disease`` when it carries a
    ``disease`` column, so a single multi-disease manifest can back both
    baseline configs. Relative ``image_path`` entries are resolved against
    the manifest file's directory.
    """

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self._datasets: dict[str, _ManifestDataset] = {}
        self._train_weights: list[float] = []

    def setup(self, stage: str | None = None) -> None:
        from sehat.data.manifest import load_manifest

        frame = load_manifest(self.config.manifest_path)
        if self.config.disease and "disease" in frame.columns:
            frame = frame[frame["disease"] == self.config.disease]
        base_dir = Path(self.config.manifest_path).resolve().parent

        wanted = ("train", "val", "test") if stage in (None, "fit", "test") else (stage,)
        for split in wanted:
            split_frame = frame[frame["split"] == split]
            train = split == "train"
            self._datasets[split] = _ManifestDataset(
                split_frame,
                base_dir,
                _build_transforms(self.config.image_size, train=train),
            )
        self._train_weights = _class_weights(self._datasets["train"].labels)

    def train_dataloader(self) -> DataLoader:
        dataset = self._datasets["train"]
        sampler = (
            WeightedRandomSampler(
                self._train_weights, num_samples=len(self._train_weights), replacement=True
            )
            if self.config.weighted_sampler
            else None
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._eval_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader("test")

    def _eval_dataloader(self, split: str) -> DataLoader:
        return DataLoader(
            self._datasets[split],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )


class _ManifestDataset(Dataset):
    """Decodes manifest rows into ``(image_tensor, label)`` pairs."""

    def __init__(self, frame: pd.DataFrame, base_dir: Path, transform: Any) -> None:
        self._paths = [_resolve_path(p, base_dir) for p in frame["image_path"]]
        self.labels = [float(v) for v in frame["label"]]
        self._transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        image = Image.open(self._paths[index]).convert("RGB")
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return self._transform(image), label


def _build_transforms(image_size: int, *, train: bool) -> Any:
    from torchvision import transforms

    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def _class_weights(labels: list[float]) -> list[float]:
    """Per-sample weights inversely proportional to class frequency."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return [1.0] * len(labels)
    weight_for = {0.0: len(labels) / (2.0 * negatives), 1.0: len(labels) / (2.0 * positives)}
    return [weight_for[label] for label in labels]


def _resolve_path(image_path: str, base_dir: Path) -> Path:
    path = Path(image_path)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path
