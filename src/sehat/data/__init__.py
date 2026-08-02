"""Data layer: versioned public datasets, one unified manifest, leakage-safe splits.

Quick tour::

    from sehat.data.manifest import load_manifest
    from sehat.data.splits import patient_level_split, site_holdout_split

    df = load_manifest("data/manifest.csv")          # validated, typed
    df = patient_level_split(df)                     # no patient in two splits
    df = site_holdout_split(df, holdout_site="montgomery")  # external validation

The ``sehat-data`` CLI performs the same steps without any third-party
dependency. Raw data lives under the gitignored ``data/`` directory and is
versioned with DVC when available (see :mod:`sehat.data.versioning`).
"""

from sehat.data.manifest import (
    AGE_UNKNOWN,
    COLUMNS,
    DISEASES,
    SEXES,
    SPLITS,
    ManifestValidationError,
    load_manifest,
    validate_manifest_csv,
    validate_rows,
    write_manifest,
    write_manifest_csv,
)
from sehat.data.splits import (
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
    patient_level_split,
    site_holdout_split,
)

__all__ = [
    "AGE_UNKNOWN",
    "COLUMNS",
    "DEFAULT_SEED",
    "DEFAULT_TEST_FRAC",
    "DEFAULT_VAL_FRAC",
    "DISEASES",
    "SEXES",
    "SPLITS",
    "ManifestValidationError",
    "load_manifest",
    "patient_level_split",
    "site_holdout_split",
    "validate_manifest_csv",
    "validate_rows",
    "write_manifest",
    "write_manifest_csv",
]
