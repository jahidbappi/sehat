"""Pure-numpy evaluation metrics for binary clinical screening.

This module is deliberately dependency-light: only ``numpy`` is imported at
module top level so that the metric test-suite runs on bare machines (no
torch, no sklearn, no pandas). Every function validates its inputs, handles
tied scores explicitly, and documents edge-case behaviour.

All metrics assume a binary label space where ``1`` denotes the positive
(disease) class and scores are monotone in the probability of disease.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "auroc",
    "average_precision",
    "brier_score",
    "expected_calibration_error",
    "sensitivity_at_specificity",
    "specificity_at_sensitivity",
]

# Tolerance used when comparing achieved specificity/sensitivity against a
# target operating point, guarding against floating-point division noise.
_TOL = 1e-12


def _as_1d_float_array(values: np.ndarray | list[float], name: str) -> np.ndarray:
    """Coerce input to a contiguous 1-D float64 array."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_binary(
    y_true: np.ndarray | list[float],
    y_score: np.ndarray | list[float],
    *,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels/scores and return them as aligned 1-D arrays.

    Raises:
        ValueError: if lengths differ or labels are not binary. When
            ``require_both_classes`` is True (needed for ranking metrics, but
            not for calibration metrics), also if only one class is present.
    """
    y = _as_1d_float_array(y_true, "y_true")
    s = _as_1d_float_array(y_score, "y_score")
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"y_true and y_score length mismatch: {y.shape[0]} vs {s.shape[0]}")
    unique = np.unique(y)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError(f"y_true must be binary {{0, 1}}, got values {unique.tolist()}")
    if require_both_classes and unique.shape[0] < 2:
        raise ValueError("y_true contains a single class; metric is undefined")
    return y, s


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of ``x`` with ties assigned their mean rank (ascending)."""
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def auroc(y_true: np.ndarray | list[float], y_score: np.ndarray | list[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Equivalent to the probability that a uniformly chosen positive has a
    higher score than a uniformly chosen negative; ties count as half.
    Handles tied scores exactly (no staircase bias).

    Args:
        y_true: binary ground-truth labels in {0, 1}.
        y_score: continuous scores, higher = more likely positive.

    Returns:
        AUROC in [0, 1].

    Raises:
        ValueError: if inputs are invalid or only one class is present.
    """
    y, s = _validate_binary(y_true, y_score)
    ranks = _average_ranks(s)
    pos = y == 1.0
    n_pos = float(pos.sum())
    n_neg = float(y.shape[0]) - n_pos
    u = float(ranks[pos].sum()) - n_pos * (n_pos + 1.0) / 2.0
    return u / (n_pos * n_neg)


def average_precision(y_true: np.ndarray | list[float], y_score: np.ndarray | list[float]) -> float:
    """Average precision (area under the precision-recall step function).

    Uses the step-function interpolation (``sum_n (R_n - R_{n-1}) * P_n``),
    matching ``sklearn.metrics.average_precision_score`` rather than the
    trapezoidal PR area.

    Args:
        y_true: binary ground-truth labels in {0, 1}.
        y_score: continuous scores, higher = more likely positive.

    Returns:
        Average precision in (0, 1].
    """
    y, s = _validate_binary(y_true, y_score)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    # Evaluate precision/recall once per distinct threshold (last index of
    # each run of equal scores) so ties are handled exactly.
    distinct = np.flatnonzero(np.diff(s_sorted) != 0.0)
    cutoff = np.concatenate([distinct, [y_sorted.shape[0] - 1]])
    tps = np.cumsum(y_sorted)[cutoff]
    fps = (cutoff + 1.0) - tps
    precision = tps / (tps + fps)
    recall = tps / float(y_sorted.sum())
    recall_delta = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(recall_delta * precision))


def _roc_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) including the degenerate (0, 0) origin point."""
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    distinct = np.flatnonzero(np.diff(s_sorted) != 0.0)
    cutoff = np.concatenate([distinct, [y_sorted.shape[0] - 1]])
    tps = np.cumsum(y_sorted)[cutoff]
    fps = np.cumsum(1.0 - y_sorted)[cutoff]
    n_pos = float(y_sorted.sum())
    n_neg = float(y_sorted.shape[0]) - n_pos
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    thresholds = np.concatenate([[math.inf], s_sorted[cutoff]])
    return fpr, tpr, thresholds


def sensitivity_at_specificity(
    y_true: np.ndarray | list[float],
    y_score: np.ndarray | list[float],
    target_specificity: float = 0.95,
    return_threshold: bool = False,
) -> float | tuple[float, float]:
    """Sensitivity at the highest-scoring threshold achieving a target specificity.

    This is the clinically meaningful operating point for screening: in
    low-resource TB/pneumonia triage, specificity must stay high to avoid
    overwhelming confirmatory-testing capacity, and we report the
    sensitivity that remains available there.

    Among all ROC thresholds whose specificity >= ``target_specificity`` the
    maximum sensitivity is returned (ties resolve to the highest threshold,
    i.e. the most conservative operating point). If no threshold reaches the
    target, the point of maximum achieved specificity is used instead.

    Args:
        y_true: binary ground-truth labels in {0, 1}.
        y_score: continuous scores, higher = more likely positive.
        target_specificity: required specificity in [0, 1]; 0.95 by default.
        return_threshold: if True, also return the decision threshold.

    Returns:
        Sensitivity at the operating point, or ``(sensitivity, threshold)``.
    """
    if not 0.0 <= target_specificity <= 1.0:
        raise ValueError(f"target_specificity must be in [0, 1], got {target_specificity}")
    y, s = _validate_binary(y_true, y_score)
    fpr, tpr, thresholds = _roc_points(y, s)
    specificity = 1.0 - fpr
    mask = specificity >= target_specificity - _TOL
    if not bool(mask.any()):
        # Target unreachable: fall back to the most specific point available.
        idx = int(np.argmax(specificity))
    else:
        masked_tpr = np.where(mask, tpr, -np.inf)
        idx = int(np.argmax(masked_tpr))
    sens = float(tpr[idx])
    if return_threshold:
        return sens, float(thresholds[idx])
    return sens


def specificity_at_sensitivity(
    y_true: np.ndarray | list[float],
    y_score: np.ndarray | list[float],
    target_sensitivity: float = 0.95,
    return_threshold: bool = False,
) -> float | tuple[float, float]:
    """Specificity at the highest-scoring threshold achieving a target sensitivity.

    Symmetric counterpart to :func:`sensitivity_at_specificity`. If no
    threshold reaches the target sensitivity, the most sensitive point is
    used instead.

    Args:
        y_true: binary ground-truth labels in {0, 1}.
        y_score: continuous scores, higher = more likely positive.
        target_sensitivity: required sensitivity in [0, 1]; 0.95 by default.
        return_threshold: if True, also return the decision threshold.

    Returns:
        Specificity at the operating point, or ``(specificity, threshold)``.
    """
    if not 0.0 <= target_sensitivity <= 1.0:
        raise ValueError(f"target_sensitivity must be in [0, 1], got {target_sensitivity}")
    y, s = _validate_binary(y_true, y_score)
    fpr, tpr, thresholds = _roc_points(y, s)
    specificity = 1.0 - fpr
    mask = tpr >= target_sensitivity - _TOL
    if not bool(mask.any()):
        idx = int(np.argmax(tpr))
    else:
        masked_spec = np.where(mask, specificity, -np.inf)
        idx = int(np.argmax(masked_spec))
    spec = float(specificity[idx])
    if return_threshold:
        return spec, float(thresholds[idx])
    return spec


def expected_calibration_error(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error with equal-width bins over [0, 1].

    ECE = sum_b (|b| / N) * |acc(b) - conf(b)|, where conf(b) is the mean
    predicted probability in bin b and acc(b) the empirical positive rate.
    Probabilities are clipped into [0, 1]; bin 0 includes 0.0 and the last
    bin includes 1.0.

    Args:
        y_true: binary ground-truth labels in {0, 1}.
        y_prob: predicted probabilities of the positive class.
        n_bins: number of equal-width bins (15 by default).

    Returns:
        ECE in [0, 1]; 0 indicates perfect calibration under this binning.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    y, p = _validate_binary(y_true, y_prob, require_both_classes=False)
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    n = float(y.shape[0])
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_index == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        acc = float(y[in_bin].mean())
        conf = float(p[in_bin].mean())
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_true: np.ndarray | list[float], y_prob: np.ndarray | list[float]) -> float:
    """Brier score: mean squared error between probabilities and outcomes.

    Lower is better; a perfect probabilistic forecaster scores 0, and a
    constant base-rate forecaster scores p(1 - p).

    Args:
        y_true: binary ground-truth labels in {0, 1}.
        y_prob: predicted probabilities of the positive class (clipped to
            [0, 1] defensively).

    Returns:
        Brier score in [0, 1].
    """
    y, p = _validate_binary(y_true, y_prob, require_both_classes=False)
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((p - y) ** 2))
