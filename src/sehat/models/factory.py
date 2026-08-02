"""Model factory for transfer-learning backbones.

Provides :func:`build_model`, the single entry point other Sehat modules
use to construct classification models, plus
:func:`load_backbone_from_ckpt` for restoring a trained backbone from a
Lightning checkpoint (used by the export pipeline).

Torch/torchvision are imported lazily inside functions so this module can
be imported — and its argument validation exercised — in environments
without PyTorch installed (e.g. docs builds, lightweight CI checks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = ["build_model", "load_backbone_from_ckpt", "supported_archs"]

#: Architectures supported by :func:`build_model`, mapped to their
#: torchvision constructor names.
_ARCH_REGISTRY: dict[str, str] = {
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "resnet50": "resnet50",
}

_CKPT_MODEL_PREFIX = "model."


def supported_archs() -> tuple[str, ...]:
    """Return the architecture names accepted by :func:`build_model`."""
    return tuple(_ARCH_REGISTRY)


def build_model(
    arch: str,
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> torch.nn.Module:
    """Build a torchvision backbone with a replaced classification head.

    Args:
        arch: One of ``efficientnet_b0``, ``convnext_tiny``, ``resnet50``.
        num_classes: Number of output logits; ``1`` (default) for binary
            screening with ``BCEWithLogitsLoss``.
        pretrained: Load torchvision's default ImageNet weights.
        dropout: Dropout probability in the classification head.

    Returns:
        A ``torch.nn.Module`` producing logits of shape ``(B, num_classes)``.

    Raises:
        ValueError: If ``arch`` is unsupported or arguments are invalid.
            Raised before any torch import so it works torch-free.
        ImportError: If torch/torchvision are not installed.
    """
    if arch not in _ARCH_REGISTRY:
        raise ValueError(
            f"Unsupported arch {arch!r}; supported: {', '.join(_ARCH_REGISTRY)}"
        )
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")

    from sehat.models.heads import ClassificationHead

    models, torch = _require_torchvision()
    weights = "DEFAULT" if pretrained else None
    constructor = getattr(models, _ARCH_REGISTRY[arch])
    model = constructor(weights=weights)

    if arch == "efficientnet_b0":
        # classifier: Sequential(Dropout, Linear)
        in_features = int(model.classifier[-1].in_features)
        model.classifier = ClassificationHead(in_features, num_classes, dropout)
    elif arch == "convnext_tiny":
        # classifier: Sequential(LayerNorm, Flatten, Linear) — keep norm+flatten.
        in_features = int(model.classifier[-1].in_features)
        model.classifier = torch.nn.Sequential(
            *list(model.classifier[:-1]),
            ClassificationHead(in_features, num_classes, dropout),
        )
    elif arch == "resnet50":
        in_features = int(model.fc.in_features)
        model.fc = ClassificationHead(in_features, num_classes, dropout)

    return model


def load_backbone_from_ckpt(
    path: str,
    *,
    map_location: str = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """Rebuild a trained model from a Lightning ``.ckpt`` checkpoint.

    The architecture and head size are read from the checkpoint's saved
    hyperparameters (``arch`` / ``num_classes``); weights are taken from the
    Lightning ``state_dict`` with the ``model.`` prefix stripped. The model
    is returned in eval mode on ``map_location``, ready for export or
    inference.

    Args:
        path: Path to the Lightning checkpoint file.
        map_location: Device string passed to ``torch.load``.
        strict: Enforce exact state-dict key matching.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the checkpoint does not contain a usable state dict.
        ImportError: If torch is not installed.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    import torch

    ckpt: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"No state_dict found in checkpoint: {path}")

    model_state = {
        key[len(_CKPT_MODEL_PREFIX):]: value
        for key, value in state_dict.items()
        if key.startswith(_CKPT_MODEL_PREFIX)
    }
    if not model_state:
        model_state = dict(state_dict)

    hparams = ckpt.get("hyper_parameters") or {}
    arch = hparams.get("arch") or _infer_arch_from_state(model_state)
    num_classes = int(hparams.get("num_classes", 1))

    model = build_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(model_state, strict=strict)
    model.eval()
    return model


def _require_torchvision() -> tuple[Any, Any]:
    """Import torch and torchvision with an actionable error message."""
    try:
        import torch
        from torchvision import models
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "build_model requires torch and torchvision; "
            "install them with `pip install torch torchvision`."
        ) from exc
    return models, torch


def _infer_arch_from_state(state_dict: dict[str, Any]) -> str:
    """Best-effort architecture detection from state-dict key shapes.

    Only a fallback for checkpoints that predate hyperparameter saving;
    Lightning checkpoints produced by ``sehat.training`` always record
    ``arch`` explicitly.
    """
    keys = tuple(state_dict)
    if any(key.startswith("layer1.") or key.startswith("conv1.") for key in keys):
        return "resnet50"
    if any(".block." in key for key in keys):
        # EfficientNet nests one extra index inside block (block.0.0.weight);
        # ConvNeXt blocks are a single module (block.0.weight).
        if any(key.split(".block.", 1)[1].count(".") >= 2 for key in keys if ".block." in key):
            return "efficientnet_b0"
        return "convnext_tiny"
    raise ValueError(
        "Could not infer architecture from checkpoint; "
        "re-train with sehat.training so `arch` is stored in the checkpoint."
    )
