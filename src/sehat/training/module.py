"""PyTorch Lightning module for binary chest-X-ray screening.

Requires torch and Lightning; imported lazily via
``sehat.training.__getattr__`` so the ``sehat.training`` package stays
importable without them.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sehat.models.factory import build_model
from sehat.training.config import TrainConfig

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - legacy package name
    import pytorch_lightning as pl

__all__ = ["SehatLitModule"]


class SehatLitModule(pl.LightningModule):
    """Lightning wrapper around a factory-built backbone.

    Trains with ``BCEWithLogitsLoss`` (optionally ``pos_weight``-weighted
    for class imbalance) and logs epoch-level ``train_loss`` / ``val_loss``
    / ``test_loss`` plus accuracy. The backbone is stored as ``self.model``
    so checkpoint state-dict keys carry the ``model.`` prefix consumed by
    :func:`sehat.models.factory.load_backbone_from_ckpt`; ``arch`` and
    ``num_classes`` are saved into the checkpoint hyperparameters.
    """

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.model = build_model(
            config.arch,
            num_classes=1,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
        pos_weight = torch.tensor([config.pos_weight]) if config.pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.save_hyperparameters({**config.to_dict(), "num_classes": 1})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, 1)``."""
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images).squeeze(1)
        loss = self.criterion(logits, labels)
        accuracy = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean()
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        self.log(
            f"{stage}_acc",
            accuracy,
            prog_bar=stage == "val",
            on_step=False,
            on_epoch=True,
            batch_size=labels.size(0),
        )
        return loss
