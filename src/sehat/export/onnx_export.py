"""Export trained Sehat screening models to ONNX.

The exported graph takes a single float32 NCHW tensor named ``images``
(normalized with the serving preprocessing statistics) and emits raw logits
named ``logits`` — the serving layer applies the sigmoid, so do not bake one
into the model before exporting.

This module is importable without torch installed; torch is imported lazily
inside :func:`export_onnx`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Default spatial input size — must match ``preprocessing.image_size`` in
#: ``configs/serve/default.yaml``.
DEFAULT_IMAGE_SIZE = 224
#: Default ONNX opset — must match ``export.opset`` in the serve config.
DEFAULT_OPSET = 17

#: Canonical input/output tensor names shared with the serving engine.
INPUT_NAME = "images"
OUTPUT_NAME = "logits"


def export_onnx(
    model: Any,
    out_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
) -> Path:
    """Export a torch model to ONNX and verify the artifact.

    Args:
        model: A ``torch.nn.Module`` mapping an NCHW float32 batch to logits.
        out_path: Destination ``.onnx`` file (parent directories are created).
        image_size: Height/width of the square dummy input used for tracing.
        opset: ONNX opset version to target.
        dynamic_batch: When True, the batch dimension is dynamic so the same
            artifact serves single images and batches.

    Returns:
        The path to the written ``.onnx`` file.

    Raises:
        RuntimeError: If the exported artifact fails ONNX checker validation.
    """
    import torch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    was_training = getattr(model, "training", False)
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    dynamic_axes = {INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}} if dynamic_batch else None
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out),
            opset_version=opset,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
        )
    if was_training:
        model.train()

    _verify(out)
    return out


def _verify(path: Path) -> None:
    """Sanity-check the exported graph with the ONNX checker when available."""
    try:
        import onnx
    except ImportError:
        return
    onnx.checker.check_model(str(path))
