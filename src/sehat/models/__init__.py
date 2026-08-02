"""Model construction and calibration for Sehat screening models.

The public API is importable without torch installed; torch is only
required when actually calling :func:`build_model` /
:func:`load_backbone_from_ckpt` or touching :class:`ClassificationHead`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sehat.models.calibration import TemperatureScaler
from sehat.models.factory import build_model, load_backbone_from_ckpt, supported_archs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sehat.models.heads import ClassificationHead

__all__ = [
    "ClassificationHead",
    "TemperatureScaler",
    "build_model",
    "load_backbone_from_ckpt",
    "supported_archs",
]


def __getattr__(name: str) -> object:
    # ClassificationHead lives in heads.py, which imports torch eagerly;
    # resolve it lazily so `import sehat.models` stays torch-free.
    if name == "ClassificationHead":
        from sehat.models.heads import ClassificationHead

        return ClassificationHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
