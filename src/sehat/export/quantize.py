"""INT8 dynamic quantization for exported Sehat ONNX models.

Dynamic quantization keeps activations in float32 and quantizes weights to
INT8 at load time, which needs no calibration dataset — important here
because calibration X-rays may not be redistributable. It typically cuts
model size ~4x and speeds up CPU inference, helping the <2s/image target on
low-end clinic hardware.

Importable without onnxruntime; the dependency is loaded lazily inside
:func:`quantize_int8`.
"""

from __future__ import annotations

from pathlib import Path

#: Suffix appended to produce the quantized artifact name
#: (``model.onnx`` -> ``model.int8.onnx``).
INT8_SUFFIX = ".int8.onnx"


def int8_path_for(onnx_path: str | Path) -> Path:
    """Return the conventional INT8 output path for an ONNX artifact."""
    path = Path(onnx_path)
    return path.with_name(f"{path.stem}{INT8_SUFFIX}")


def quantize_int8(
    onnx_path: str | Path,
    out_path: str | Path | None = None,
    *,
    per_channel: bool = True,
) -> Path:
    """Quantize an ONNX model to INT8 with dynamic quantization.

    Args:
        onnx_path: Source float32 ``.onnx`` model.
        out_path: Destination path; defaults to ``<stem>.int8.onnx`` next to
            the source.
        per_channel: Quantize convolution weights per output channel for
            better accuracy at a small size cost.

    Returns:
        The path to the quantized ``*.int8.onnx`` artifact.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(onnx_path)
    if not src.is_file():
        raise FileNotFoundError(f"ONNX model not found: {src}")
    dst = int8_path_for(src) if out_path is None else Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    return dst
