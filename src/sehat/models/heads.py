"""Classification heads for transfer-learning backbones.

This module requires PyTorch; it is imported lazily by
:func:`sehat.models.factory.build_model` so that the rest of the
``sehat.models`` package stays importable without torch installed.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """Dropout + linear projection producing raw logits.

    Used as the replacement head on torchvision backbones for binary
    clinical screening (``num_classes=1`` yields a single logit per image,
    consumed by ``BCEWithLogitsLoss``).
    """

    def __init__(self, in_features: int, num_classes: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map pooled backbone features of shape ``(B, in_features)`` to logits ``(B, num_classes)``."""
        return self.fc(self.drop(x))
