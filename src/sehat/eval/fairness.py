"""Subgroup fairness evaluation with deterministic bootstrap confidence intervals.

A screening model that is accurate on average can still fail specific
patient populations (sex, age band, clinical site). This module computes
per-subgroup AUROC and sensitivity at the clinical operating point
(95% specificity), each with seeded bootstrap 95% CIs, and raises explicit
disparity flags when a subgroup underperforms the pooled metric.

Only ``numpy`` and the stdlib are imported at module top level.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from sehat.eval.metrics import auroc, sensitivity_at_specificity

__all__ = ["AGE_BANDS", "age_band", "subgroup_report"]

# Inclusive (lower, upper) age bounds and labels for derived age bands.
AGE_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 14, "0-14"),
    (15, 39, "15-39"),
    (40, 64, "40-64"),
    (65, 200, "65+"),
)

_LABEL_KEYS = ("label", "y_true")
_SCORE_KEYS = ("score", "y_score", "prob", "y_prob")


def age_band(age: int | float) -> str:
    """Map a numeric age to a categorical band label.

    Args:
        age: patient age in years.

    Returns:
        Band label such as ``"40-64"``; ``"unknown"`` for missing/invalid ages.
    """
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value) or value < 0:
        return "unknown"
    for lower, upper, label in AGE_BANDS:
        if lower <= value <= upper:
            return label
    return "unknown"


def _get_field(record: Mapping[str, Any], candidates: Sequence[str], kind: str) -> Any:
    for key in candidates:
        if key in record:
            return record[key]
    raise KeyError(f"record is missing a {kind} field (tried {list(candidates)}): {record!r}")


def _normalise_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    """Extract aligned (y, score) arrays; derive ``age_band`` from ``age``."""
    if len(records) == 0:
        raise ValueError("records must be non-empty")
    y = np.empty(len(records), dtype=np.float64)
    s = np.empty(len(records), dtype=np.float64)
    enriched: list[Mapping[str, Any]] = []
    for i, rec in enumerate(records):
        y[i] = float(_get_field(rec, _LABEL_KEYS, "label"))
        s[i] = float(_get_field(rec, _SCORE_KEYS, "score"))
        if "age_band" not in rec and "age" in rec:
            rec = {**rec, "age_band": age_band(rec["age"])}
        enriched.append(rec)
    return y, s, enriched


def _bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile 95% CI via case resampling; degenerate resamples are skipped.

    Deterministic for a fixed ``rng`` state. Returns ``(nan, nan)`` when
    fewer than two finite bootstrap replicates could be computed.
    """
    n = y.shape[0]
    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            value = fn(y[idx], s[idx])
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
    return (float(lo), float(hi))


def _safe_metric(
    fn: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, s: np.ndarray
) -> float:
    try:
        value = fn(y, s)
    except ValueError:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def subgroup_report(
    records: Sequence[Mapping[str, Any]],
    group_cols: Sequence[str] = ("sex", "age_band", "site"),
    target_specificity: float = 0.95,
    disparity_margin: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
    min_n: int = 2,
) -> dict[str, Any]:
    """Compute pooled and per-subgroup metrics with bootstrap CIs and disparity flags.

    Each record must contain a label (``label`` or ``y_true``) and a score
    (``score``, ``y_score``, ``prob`` or ``y_prob``) plus the grouping
    columns. When ``age_band`` is requested but absent, it is derived from a
    numeric ``age`` field via :func:`age_band`.

    Args:
        records: sequence of per-case mappings.
        group_cols: record fields to stratify on.
        target_specificity: operating point for the sensitivity metric.
        disparity_margin: a subgroup is flagged when its metric falls below
            the pooled metric by more than this absolute margin.
        n_boot: bootstrap replicates per CI (deterministic under ``seed``).
        seed: seed for ``numpy.random.default_rng``; identical seeds give
            bit-identical reports.
        min_n: subgroups smaller than this are marked ``insufficient_data``
            and excluded from disparity flagging.

    Returns:
        A dict of the form::

            {
              "pooled": {"auroc": float, "auroc_ci": [lo, hi],
                         "sens_at_95spec": float, "sens_at_95spec_ci": [lo, hi],
                         "n": int},
              "subgroups": {
                "<col>": {
                  "<level>": {"auroc": float, "auroc_ci": [lo, hi],
                              "sens_at_95spec": float,
                              "sens_at_95spec_ci": [lo, hi], "n": int,
                              "insufficient_data": bool,
                              "disparity": {"auroc": bool, "sens_at_95spec": bool}},
                  ...
                },
                ...
              },
              "config": {"group_cols": [...], "target_specificity": float,
                         "disparity_margin": float, "n_boot": int, "seed": int},
            }

        Metrics that are undefined (e.g. a single-class subgroup) are NaN.
    """
    y, s, enriched = _normalise_records(records)
    rng = np.random.default_rng(seed)

    def sens_fn(yy: np.ndarray, ss: np.ndarray) -> float:
        return float(sensitivity_at_specificity(yy, ss, target_specificity))

    pooled = {
        "auroc": _safe_metric(auroc, y, s),
        "auroc_ci": list(_bootstrap_ci(y, s, auroc, n_boot, rng)),
        "sens_at_95spec": _safe_metric(sens_fn, y, s),
        "sens_at_95spec_ci": list(_bootstrap_ci(y, s, sens_fn, n_boot, rng)),
        "n": int(y.shape[0]),
    }

    subgroups: dict[str, dict[str, dict[str, Any]]] = {}
    for col in group_cols:
        levels: dict[str, list[int]] = {}
        for i, rec in enumerate(enriched):
            value = rec.get(col, "unknown")
            levels.setdefault(str(value), []).append(i)

        col_report: dict[str, dict[str, Any]] = {}
        for level in sorted(levels):
            idx = np.asarray(levels[level], dtype=np.int64)
            y_g, s_g = y[idx], s[idx]
            n_g = int(idx.shape[0])
            single_class = np.unique(y_g).shape[0] < 2
            insufficient = single_class or n_g < min_n

            auroc_g = _safe_metric(auroc, y_g, s_g)
            sens_g = _safe_metric(sens_fn, y_g, s_g)
            auroc_ci = _bootstrap_ci(y_g, s_g, auroc, n_boot, rng)
            sens_ci = _bootstrap_ci(y_g, s_g, sens_fn, n_boot, rng)

            def _underperforms(
                metric: float, pooled_metric: float, insufficient: bool = insufficient
            ) -> bool:
                if insufficient or not math.isfinite(metric) or not math.isfinite(pooled_metric):
                    return False
                return bool(metric < pooled_metric - disparity_margin)

            col_report[level] = {
                "auroc": auroc_g,
                "auroc_ci": list(auroc_ci),
                "sens_at_95spec": sens_g,
                "sens_at_95spec_ci": list(sens_ci),
                "n": n_g,
                "insufficient_data": insufficient,
                "disparity": {
                    "auroc": _underperforms(auroc_g, pooled["auroc"]),
                    "sens_at_95spec": _underperforms(sens_g, pooled["sens_at_95spec"]),
                },
            }
        subgroups[str(col)] = col_report

    return {
        "pooled": pooled,
        "subgroups": subgroups,
        "config": {
            "group_cols": [str(c) for c in group_cols],
            "target_specificity": float(target_specificity),
            "disparity_margin": float(disparity_margin),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "min_n": int(min_n),
        },
    }
