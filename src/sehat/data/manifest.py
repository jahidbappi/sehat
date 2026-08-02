"""Unified dataset manifest: schema, validation, and I/O.

Every dataset Sehat trains on is normalized into a single CSV manifest with
exactly these columns, in this order::

    image_path,label,disease,site,patient_id,split,sex,age

Field contract
--------------
- ``image_path``: path to the image, relative to the data root; non-empty.
- ``label``: ``1`` (disease present) or ``0`` (normal).
- ``disease``: ``tb``, ``pneumonia``, or ``normal``. Must agree with ``label``:
  ``label=1`` requires ``tb``/``pneumonia``, ``label=0`` requires ``normal``.
- ``site``: origin dataset/hospital (e.g. ``shenzhen``, ``montgomery``, ``nih``);
  used for external-site validation.
- ``patient_id``: de-identified patient key; the unit of leakage-safe splitting.
- ``split``: ``train``, ``val``, ``test``, or ``test_external`` (rows from a
  site the model never trained on).
- ``sex``: ``M``, ``F``, or ``unknown``.
- ``age``: integer years 0-120, or empty / ``-1`` (``AGE_UNKNOWN``) when unknown.

The validation core is pure stdlib so it can run anywhere (CI, clinic laptops,
packaging pipelines) without heavy dependencies. ``load_manifest`` /
``write_manifest`` are the pandas entry points used by the training stack and
import pandas lazily.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "AGE_UNKNOWN",
    "COLUMNS",
    "DISEASES",
    "SEXES",
    "SPLITS",
    "ManifestValidationError",
    "iter_manifest_rows",
    "load_manifest",
    "validate_manifest_csv",
    "validate_rows",
    "write_manifest",
    "write_manifest_csv",
]

COLUMNS: tuple[str, ...] = (
    "image_path",
    "label",
    "disease",
    "site",
    "patient_id",
    "split",
    "sex",
    "age",
)
"""Canonical manifest column order. Every consumer may rely on this order."""

DISEASES = frozenset({"tb", "pneumonia", "normal"})
SPLITS = frozenset({"train", "val", "test", "test_external"})
SEXES = frozenset({"M", "F", "unknown"})
AGE_UNKNOWN = -1
"""Sentinel age value used in place of an empty cell once loaded into a DataFrame."""

_POSITIVE_DISEASES = frozenset({"tb", "pneumonia"})
_MAX_REPORTED_ERRORS = 25


class ManifestValidationError(ValueError):
    """Raised when one or more manifest rows violate the schema.

    Carries the full list of human-readable errors on ``.errors``; the string
    form shows the first ``_MAX_REPORTED_ERRORS`` so CI logs stay readable.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        shown = "\n".join(errors[:_MAX_REPORTED_ERRORS])
        message = f"manifest validation failed with {len(errors)} error(s):\n{shown}"
        extra = len(errors) - _MAX_REPORTED_ERRORS
        if extra > 0:
            message += f"\n... and {extra} more"
        super().__init__(message)


def _validate_row(row: Mapping[str, object], row_number: int) -> list[str]:
    """Return all schema violations for a single row (empty list if valid)."""

    def cell(name: str) -> str:
        value = row.get(name)
        return "" if value is None else str(value).strip()

    missing = [name for name in COLUMNS if name not in row]
    if missing:
        return [f"row {row_number}: missing column(s): {', '.join(missing)}"]
    if None in row:
        return [f"row {row_number}: more fields than the header declares"]

    errors: list[str] = []

    def err(field: str, message: str) -> None:
        errors.append(f"row {row_number}: {field}: {message}")

    if not cell("image_path"):
        err("image_path", "must be non-empty")

    label_raw = cell("label")
    label: int | None = None
    if label_raw not in {"0", "1"}:
        err("label", f"must be 0 or 1, got {label_raw!r}")
    else:
        label = int(label_raw)

    disease = cell("disease")
    if disease not in DISEASES:
        err("disease", f"must be one of {sorted(DISEASES)}, got {disease!r}")

    if not cell("site"):
        err("site", "must be non-empty")
    if not cell("patient_id"):
        err("patient_id", "must be non-empty")

    split = cell("split")
    if split not in SPLITS:
        err("split", f"must be one of {sorted(SPLITS)}, got {split!r}")

    if cell("sex") not in SEXES:
        err("sex", f"must be one of {sorted(SEXES)}, got {cell('sex')!r}")

    age_raw = cell("age")
    if age_raw not in {"", str(AGE_UNKNOWN)}:
        try:
            age = int(age_raw)
        except ValueError:
            err("age", f"must be an integer 0-120 or empty, got {age_raw!r}")
        else:
            if not 0 <= age <= 120:
                err("age", f"must be between 0 and 120, got {age}")

    if label is not None and disease in DISEASES:
        if label == 1 and disease not in _POSITIVE_DISEASES:
            err("disease", f"label=1 requires tb or pneumonia, got {disease!r}")
        if label == 0 and disease != "normal":
            err("disease", f"label=0 requires normal, got {disease!r}")

    return errors


def validate_rows(rows: Iterable[Mapping[str, object]]) -> None:
    """Validate manifest rows, raising :class:`ManifestValidationError` on any violation.

    ``rows`` is any iterable of mappings keyed by column name (e.g. dataset
    records before writing, or ``csv.DictReader`` output). Row numbering in
    error messages starts at 2, assuming row 1 is the CSV header.
    """
    errors: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        errors.extend(_validate_row(row, row_number))
    if errors:
        raise ManifestValidationError(errors)


def iter_manifest_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield manifest rows as plain dicts using only the stdlib.

    Unlike :func:`load_manifest`, this does not validate; pair it with
    :func:`validate_manifest_csv` when the input is untrusted.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def validate_manifest_csv(path: str | Path) -> None:
    """Validate a manifest CSV on disk without any third-party dependency."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != COLUMNS:
            raise ManifestValidationError(
                [
                    f"row 1: header must be exactly {list(COLUMNS)}, got {list(header)}",
                ]
            )
        validate_rows(reader)


def write_manifest_csv(rows: Iterable[Mapping[str, object]], path: str | Path) -> Path:
    """Validate ``rows`` and write them as a manifest CSV using only the stdlib."""
    rows = list(rows)
    validate_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _require_pandas() -> pd:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "pandas is required for this function; install the data extra: "
            "pip install 'sehat[data]'"
        ) from exc
    return pd


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a manifest CSV into a validated, typed ``pandas.DataFrame``.

    Validates the full schema first and raises :class:`ManifestValidationError`
    listing every bad row. The returned frame has columns in canonical order,
    ``label`` as ``int64``, ``age`` as ``int64`` with unknowns mapped to
    :data:`AGE_UNKNOWN`, and all other columns as strings.
    """
    pd = _require_pandas()
    path = Path(path)
    validate_manifest_csv(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype("int64")
    unknown_ages = {"", str(AGE_UNKNOWN)}
    df["age"] = df["age"].map(lambda raw: AGE_UNKNOWN if raw in unknown_ages else int(raw))
    return df[list(COLUMNS)]


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate ``df`` and write it as a manifest CSV.

    Unknown ages (:data:`AGE_UNKNOWN` / null) are serialized as empty cells so
    the file round-trips through :func:`load_manifest` unchanged.
    """
    _require_pandas()
    df = df[list(COLUMNS)].copy()
    df["age"] = df["age"].map(lambda value: "" if value in {AGE_UNKNOWN, None} else str(int(value)))
    rows = [{name: str(row[name]) for name in COLUMNS} for _, row in df.iterrows()]
    return write_manifest_csv(rows, path)
