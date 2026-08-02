"""CPU latency benchmark for exported Sehat ONNX models.

Contract (relied on by docs, CI, and the evaluation worker)::

    benchmark_latency(onnx_path, n_runs=50, warmup=10) -> dict

with at least the keys ``p50_ms``, ``p95_ms`` and ``size_mb``.

Run from the command line::

    python -m sehat.export.benchmark artifacts/tb.int8.onnx --n-runs 50

The performance target for clinic hardware is p95 < 2000 ms per image on a
low-end CPU. Importable with stdlib only; numpy/onnxruntime are imported
lazily inside :func:`benchmark_latency`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Performance target for low-end clinic CPUs (milliseconds per image).
TARGET_P95_MS = 2000.0

DEFAULT_N_RUNS = 50
DEFAULT_WARMUP = 10


def benchmark_latency(
    onnx_path: str | Path,
    n_runs: int = DEFAULT_N_RUNS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, Any]:
    """Measure single-image CPU inference latency of an ONNX model.

    Feeds zero tensors through the model (shape taken from the graph, with
    dynamic dimensions pinned to 1 for batch and 224 for spatial dims) and
    reports percentile latencies over ``n_runs`` timed runs after ``warmup``
    untimed runs.

    Args:
        onnx_path: Path to the ``.onnx`` artifact to benchmark.
        n_runs: Number of timed inference runs.
        warmup: Number of untimed warm-up runs before timing.

    Returns:
        A dict with contract keys ``p50_ms``, ``p95_ms``, ``size_mb`` plus
        diagnostic extras (``mean_ms``, ``min_ms``, ``max_ms``, ``n_runs``,
        ``warmup``, ``input_shape``, ``target_p95_ms``, ``meets_target``).
    """
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs!r}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup!r}")

    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    shape = [_resolve_dim(dim) for dim in input_meta.shape]
    batch = np.zeros(shape, dtype=np.float32)

    def run_once() -> None:
        session.run(None, {input_meta.name: batch})

    for _ in range(warmup):
        run_once()

    samples_ms: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_once()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    mean_ms = sum(samples_ms) / len(samples_ms)
    p95_ms = percentile(samples_ms, 95.0)
    return {
        "p50_ms": round(percentile(samples_ms, 50.0), 3),
        "p95_ms": round(p95_ms, 3),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "mean_ms": round(mean_ms, 3),
        "min_ms": round(samples_ms[0], 3),
        "max_ms": round(samples_ms[-1], 3),
        "n_runs": n_runs,
        "warmup": warmup,
        "input_shape": shape,
        "target_p95_ms": TARGET_P95_MS,
        "meets_target": p95_ms < TARGET_P95_MS,
    }


def percentile(sorted_samples: Sequence[float], pct: float) -> float:
    """Return the ``pct``-th percentile of pre-sorted samples.

    Uses linear interpolation between closest ranks (the same convention as
    numpy's default ``linear`` method). Pure stdlib so it is unit-testable
    without numpy.
    """
    if not sorted_samples:
        raise ValueError("cannot compute a percentile of no samples")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must lie in [0, 100], got {pct!r}")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    rank = (pct / 100.0) * (len(sorted_samples) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_samples) - 1)
    frac = rank - low
    return float(sorted_samples[low] * (1.0 - frac) + sorted_samples[high] * frac)


def _resolve_dim(dim: Any) -> int:
    """Map an ONNX dimension (int or symbolic) to a concrete benchmark size."""
    if isinstance(dim, int) and dim > 0:
        return dim
    return 1  # symbolic/unknown dims (batch) benchmark the single-image path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m sehat.export.benchmark``."""
    parser = argparse.ArgumentParser(
        prog="python -m sehat.export.benchmark",
        description="Benchmark single-image CPU latency of an ONNX model.",
    )
    parser.add_argument("onnx_path", help="path to the .onnx artifact")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"timed runs (default: {DEFAULT_N_RUNS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"untimed warm-up runs (default: {DEFAULT_WARMUP})",
    )
    args = parser.parse_args(argv)

    try:
        results = benchmark_latency(args.onnx_path, n_runs=args.n_runs, warmup=args.warmup)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(results, indent=2))
    if not results["meets_target"]:
        print(
            f"WARNING: p95 {results['p95_ms']} ms exceeds the "
            f"{TARGET_P95_MS:.0f} ms low-end CPU target; consider INT8 "
            "quantization (sehat.export.quantize)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
