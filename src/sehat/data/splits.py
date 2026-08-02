"""Leakage-safe train/val/test splitting strategies.

Medical ML fails in two classic ways that naive splitting hides:

1. **Patient leakage** — images of the same person in train and test inflate
   metrics. All splitting here is grouped by ``patient_id``: a patient never
   appears in two splits.
2. **Site overfitting** — a model tuned on one hospital can collapse on the
   next. :func:`site_holdout_split` reserves an entire site as
   ``test_external`` for honest out-of-distribution evaluation.

The pandas entry points (:func:`patient_level_split`, :func:`site_holdout_split`)
delegate to pure-stdlib cores (:func:`stratified_group_assignment`,
:func:`holdout_assignment`) so the splitting logic is fully testable without
heavy dependencies, and reusable by the stdlib-only CLI.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_TEST_FRAC",
    "DEFAULT_VAL_FRAC",
    "holdout_assignment",
    "patient_level_split",
    "patient_strata",
    "site_holdout_split",
    "stratified_group_assignment",
]

DEFAULT_SEED = 42
DEFAULT_VAL_FRAC = 0.1
DEFAULT_TEST_FRAC = 0.1

_POSITIVE = "positive"
_NEGATIVE = "negative"


def _check_fractions(val_frac: float, test_frac: float) -> None:
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if not 0.0 <= test_frac < 1.0:
        raise ValueError(f"test_frac must be in [0, 1), got {test_frac}")
    if val_frac + test_frac >= 1.0:
        raise ValueError(f"val_frac + test_frac must be < 1, got {val_frac + test_frac}")


def patient_strata(records: Iterable[tuple[str, int]]) -> dict[str, str]:
    """Resolve one stratum per patient for stratified splitting.

    ``records`` yields ``(patient_id, label)`` pairs. A patient is ``positive``
    if *any* of their images is positive — splitting on the worst-case label
    keeps diseased patients represented in every split.
    """
    strata: dict[str, str] = {}
    for patient_id, label in records:
        if label == 1 or patient_id not in strata:
            strata[patient_id] = _POSITIVE if label == 1 else _NEGATIVE
    return strata


def stratified_group_assignment(
    group_strata: Mapping[str, str],
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Assign each group key to ``train``/``val``/``test``, stratified by stratum.

    Deterministic for a given seed regardless of input ordering: keys are
    sorted before shuffling. Counts are rounded per stratum, so very small
    strata may legitimately yield an empty ``val`` or ``test``.
    """
    _check_fractions(val_frac, test_frac)
    rng = random.Random(seed)
    by_stratum: dict[str, list[str]] = {}
    for key, stratum in group_strata.items():
        by_stratum.setdefault(stratum, []).append(key)

    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        keys = sorted(by_stratum[stratum])
        rng.shuffle(keys)
        n_test = round(len(keys) * test_frac)
        n_val = round(len(keys) * val_frac)
        for key in keys[:n_test]:
            assignment[key] = "test"
        for key in keys[n_test : n_test + n_val]:
            assignment[key] = "val"
        for key in keys[n_test + n_val :]:
            assignment[key] = "train"
    return assignment


def holdout_assignment(
    records: Iterable[tuple[str, str, int]],
    holdout_site: str,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Assign patients to splits with one entire site held out as ``test_external``.

    ``records`` yields ``(patient_id, site, label)`` triples. A patient with
    any row at ``holdout_site`` is external — even rows recorded at other
    sites — to guarantee zero patient overlap between training and the
    external evaluation set. Remaining patients get a stratified
    train/val/test assignment.
    """
    labels: dict[str, int] = {}
    external: set[str] = set()
    for patient_id, site, label in records:
        if site == holdout_site:
            external.add(patient_id)
        labels[patient_id] = max(labels.get(patient_id, 0), label)

    strata = {
        patient_id: (_POSITIVE if has_positive else _NEGATIVE)
        for patient_id, has_positive in labels.items()
        if patient_id not in external
    }
    assignment = dict.fromkeys(sorted(external), "test_external")
    assignment.update(
        stratified_group_assignment(strata, val_frac=val_frac, test_frac=test_frac, seed=seed)
    )
    return assignment


def _require_pandas() -> pd:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "pandas is required for this function; install the data extra: "
            "pip install 'sehat[data]'"
        ) from exc
    return pd


def patient_level_split(
    df: pd.DataFrame,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``split`` assigned per patient (no leakage).

    Splitting is stratified so positive patients are represented in train,
    val, and test. Any existing ``split`` values are overwritten.
    """
    _require_pandas()
    for column in ("patient_id", "label"):
        if column not in df.columns:
            raise ValueError(f"manifest frame is missing required column {column!r}")
    strata = patient_strata(zip(df["patient_id"].astype(str), df["label"].astype(int), strict=True))
    assignment = stratified_group_assignment(
        strata, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    out = df.copy()
    out["split"] = [assignment[patient_id] for patient_id in out["patient_id"].astype(str)]
    return out


def site_holdout_split(df: pd.DataFrame, holdout_site: str) -> pd.DataFrame:
    """Return a copy of ``df`` with ``holdout_site`` rows as ``test_external``.

    The remaining sites are split per patient with the default fractions and
    seed. Any existing ``split`` values are overwritten.
    """
    _require_pandas()
    for column in ("patient_id", "site", "label"):
        if column not in df.columns:
            raise ValueError(f"manifest frame is missing required column {column!r}")
    records = zip(
        df["patient_id"].astype(str),
        df["site"].astype(str),
        df["label"].astype(int),
        strict=True,
    )
    assignment = holdout_assignment(records, holdout_site)
    out = df.copy()
    out["split"] = [assignment[patient_id] for patient_id in out["patient_id"].astype(str)]
    return out
