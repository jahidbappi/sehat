"""Calibration analysis: reliability-diagram data and temperature scaling.

Reliability-diagram data is computed in pure numpy. Post-hoc calibration is
delegated to ``sehat.models.calibration.TemperatureScaler`` (owned by the
models worker); that import is guarded so this module stays importable — and
the reliability-diagram functionality stays usable — on machines without
torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sehat.eval.metrics import expected_calibration_error

__all__ = [
    "TemperatureCalibrator",
    "reliability_diagram_data",
    "temperature_scaling_available",
]

try:  # Guarded: sehat.models.calibration may require torch (Worker 2 owns it).
    from sehat.models.calibration import TemperatureScaler as _TemperatureScaler

    _HAS_TEMPERATURE_SCALER = True
except Exception:
    _TemperatureScaler = None  # type: ignore[assignment]
    _HAS_TEMPERATURE_SCALER = False


def temperature_scaling_available() -> bool:
    """Return True when ``sehat.models.calibration.TemperatureScaler`` is importable."""
    return _HAS_TEMPERATURE_SCALER


def reliability_diagram_data(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float],
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute per-bin statistics needed to draw a reliability diagram.

    Uses equal-width bins over [0, 1], identical to
    :func:`sehat.eval.metrics.expected_calibration_error`, so the reported
    ``ece`` is always consistent with the bin contents.

    Args:
        y_true: binary ground-truth labels in {0, 1}.
        y_prob: predicted probabilities of the positive class.
        n_bins: number of equal-width bins (15 by default).

    Returns:
        ``{"bins": [{"lower", "upper", "count", "mean_confidence",
        "accuracy", "gap"}...], "ece": float, "n_bins": int, "n": int}``
        where ``gap = accuracy - mean_confidence`` (positive = underconfident)
        and empty bins have NaN statistics.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"y_true and y_prob length mismatch: {y.shape[0]} vs {p.shape[0]}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    bins: list[dict[str, float | int]] = []
    for b in range(n_bins):
        in_bin = bin_index == b
        count = int(in_bin.sum())
        if count:
            mean_conf = float(p[in_bin].mean())
            acc = float(y[in_bin].mean())
        else:
            mean_conf = acc = float("nan")
        bins.append(
            {
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "gap": acc - mean_conf if count else float("nan"),
            }
        )

    return {
        "bins": bins,
        "ece": expected_calibration_error(y, p, n_bins=n_bins),
        "n_bins": int(n_bins),
        "n": int(y.shape[0]),
    }


class TemperatureCalibrator:
    """Thin wrapper around ``sehat.models.calibration.TemperatureScaler``.

    The wrapper adapts to the scaler's API surface (``fit``/``calibrate``/
    ``predict_proba`` conventions) so evaluation code has one stable entry
    point regardless of the models-side implementation details. Raw
    numpy arrays are accepted and converted lazily; torch is only ever
    touched inside the scaler itself.

    Raises:
        RuntimeError: on construction when the scaler cannot be imported
            (e.g. torch unavailable on this machine).
    """

    def __init__(self) -> None:
        if not _HAS_TEMPERATURE_SCALER:
            raise RuntimeError(
                "TemperatureCalibrator requires sehat.models.calibration."
                "TemperatureScaler, which could not be imported (torch may "
                "be unavailable on this machine)."
            )
        self._scaler = _TemperatureScaler()

    def fit(self, logits: Any, y_true: Any) -> TemperatureCalibrator:
        """Fit the temperature on validation logits and labels.

        Args:
            logits: raw model logits, shape (N,) or (N, C).
            y_true: integer labels, shape (N,).

        Returns:
            self, for chaining.
        """
        fit = getattr(self._scaler, "fit", None) or self._scaler.fit_temperature
        if fit is None:
            raise RuntimeError("TemperatureScaler exposes neither fit nor fit_temperature")
        fit(logits, y_true)
        return self

    def calibrate(self, logits: Any) -> np.ndarray:
        """Apply the learned temperature and return calibrated probabilities.

        Args:
            logits: raw model logits with the same layout used in ``fit``.

        Returns:
            Calibrated probabilities as a numpy array.
        """
        for name in ("predict_proba", "calibrate", "transform", "__call__"):
            fn = getattr(self._scaler, name, None)
            if callable(fn):
                out = fn(logits)
                break
        else:
            raise RuntimeError(
                "TemperatureScaler exposes no callable among predict_proba, "
                "calibrate, transform, __call__"
            )
        if hasattr(out, "detach"):  # torch tensor
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=np.float64)
