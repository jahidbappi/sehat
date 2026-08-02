"""Training pipeline for Sehat screening models.

:class:`TrainConfig` and :func:`load_backbone_from_ckpt` are importable
without torch/Lightning; the Lightning module and data module are resolved
lazily on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sehat.models.factory import load_backbone_from_ckpt
from sehat.training.config import TrainConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sehat.training.datamodule import ManifestDataModule
    from sehat.training.module import SehatLitModule

__all__ = [
    "ManifestDataModule",
    "SehatLitModule",
    "TrainConfig",
    "load_backbone_from_ckpt",
]


def __getattr__(name: str) -> object:
    if name == "SehatLitModule":
        from sehat.training.module import SehatLitModule

        return SehatLitModule
    if name == "ManifestDataModule":
        from sehat.training.datamodule import ManifestDataModule

        return ManifestDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
