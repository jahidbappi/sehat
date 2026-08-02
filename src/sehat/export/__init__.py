"""Sehat export package: ONNX export, INT8 quantization, CPU benchmarking.

All heavy dependencies (torch, onnx, onnxruntime, numpy) are imported lazily
inside functions, so this package is importable with the standard library
alone. Re-exports use PEP 562 lazy attribute access so that
``python -m sehat.export.benchmark`` does not execute submodules twice.
"""

from typing import Any

__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_OPSET",
    "INPUT_NAME",
    "OUTPUT_NAME",
    "benchmark_latency",
    "export_onnx",
    "int8_path_for",
    "quantize_int8",
]


_ONNX_EXPORT_NAMES = frozenset(
    {"DEFAULT_IMAGE_SIZE", "DEFAULT_OPSET", "INPUT_NAME", "OUTPUT_NAME", "export_onnx"}
)


def __getattr__(name: str) -> Any:
    if name in _ONNX_EXPORT_NAMES:
        from sehat.export import onnx_export

        return getattr(onnx_export, name)
    if name == "benchmark_latency":
        from sehat.export.benchmark import benchmark_latency

        return benchmark_latency
    if name in ("int8_path_for", "quantize_int8"):
        from sehat.export import quantize

        return getattr(quantize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
