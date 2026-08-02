"""Post-hoc probability calibration via temperature scaling.

Temperature scaling (Guo et al., 2017) divides logits by a single learned
temperature before the sigmoid/softmax, improving probability calibration
without changing ranking metrics such as AUROC — important for clinical
screening where predicted risk must be interpretable.

The optimiser is pure stdlib (golden-section search on log-temperature),
so fitting works with or without torch installed and accepts lists,
tuples, numpy arrays, or torch tensors.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = ["TemperatureScaler"]

#: Inputs accepted by :meth:`TemperatureScaler.fit` / ``calibrate``.
ArrayLike = Union[Iterable[float], "torch.Tensor"]

_LOG_T_BOUNDS = (math.log(1e-3), math.log(1e3))
_SEARCH_ITERS = 200
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


class TemperatureScaler:
    """Single-parameter temperature scaler for binary logits.

    Attributes:
        temperature: Learned temperature; ``1.0`` (identity) before fitting.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._fitted

    def fit(self, logits: ArrayLike, labels: ArrayLike) -> TemperatureScaler:
        """Learn the temperature minimising binary cross-entropy (NLL).

        Args:
            logits: Raw model logits, shape ``(N,)`` or ``(N, 1)``.
            labels: Binary labels in ``{0, 1}``, same length as ``logits``.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: On empty/mismatched inputs or non-binary labels.
        """
        z = _as_float_list(logits)
        y = _as_float_list(labels)
        if not z:
            raise ValueError("Cannot fit TemperatureScaler on empty inputs")
        if len(z) != len(y):
            raise ValueError(
                f"logits and labels must have equal length, got {len(z)} and {len(y)}"
            )
        if any(v not in (0.0, 1.0) for v in y):
            raise ValueError("labels must be binary (0 or 1)")

        def nll(log_t: float) -> float:
            t = math.exp(log_t)
            return _mean_nll(z, y, t)

        lo, hi = _LOG_T_BOUNDS
        c = hi - _GOLDEN * (hi - lo)
        d = lo + _GOLDEN * (hi - lo)
        fc, fd = nll(c), nll(d)
        for _ in range(_SEARCH_ITERS):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - _GOLDEN * (hi - lo)
                fc = nll(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + _GOLDEN * (hi - lo)
                fd = nll(d)

        self.temperature = math.exp((lo + hi) / 2.0)
        self._fitted = True
        return self

    def calibrate(self, logits: ArrayLike) -> ArrayLike:
        """Scale logits by the learned temperature (identity if unfit).

        Returns a torch tensor when given one, otherwise a list of floats.
        """
        if _is_torch_tensor(logits):
            return logits / self.temperature
        return [z / self.temperature for z in _as_float_list(logits)]


def _mean_nll(logits: list[float], labels: list[float], temperature: float) -> float:
    """Mean binary NLL of ``logits / temperature`` with a stable softplus."""
    total = 0.0
    for z, y in zip(logits, labels):
        x = z / temperature
        total += max(x, 0.0) + math.log1p(math.exp(-abs(x))) - y * x
    return total / len(logits)


def _is_torch_tensor(value: object) -> bool:
    return hasattr(value, "detach") and type(value).__module__.split(".")[0] == "torch"


def _as_float_list(values: ArrayLike) -> list[float]:
    """Coerce list/tuple/numpy/torch input to a flat list of floats."""
    if hasattr(values, "detach"):  # torch tensor
        values = values.detach().cpu().tolist()  # type: ignore[union-attr]
    elif hasattr(values, "tolist") and not isinstance(values, (list, tuple)):
        values = values.tolist()  # type: ignore[union-attr]
    flat: list[float] = []
    for item in values:  # type: ignore[union-attr]
        if isinstance(item, (list, tuple)):
            flat.extend(float(v) for v in item)
        else:
            flat.append(float(item))
    return flat
