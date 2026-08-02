"""Optional YAML configuration for dataset sources and split defaults.

Config files live in ``configs/data/`` at the repository root and are entirely
optional — sensible defaults are baked into every plugin. They exist so that
mirror URLs, checksums, and subset sizes can be updated without touching code,
which matters when NIH changes its hosting (it has, twice).

PyYAML is imported lazily; without it, defaults are used and only an explicit
``--config`` flag fails loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

__all__ = ["REPO_ROOT", "dataset_overrides", "load_yaml", "split_defaults"]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
"""Repository checkout root (``src/sehat/data/config.py`` -> 3 up)."""

DEFAULT_DATASETS_CONFIG = REPO_ROOT / "configs" / "data" / "datasets.yaml"
DEFAULT_SPLITS_CONFIG = REPO_ROOT / "configs" / "data" / "splits.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, importing PyYAML lazily."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyYAML is required to read config files; install the data extra: "
            "pip install 'sehat[data]'"
        ) from exc
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a YAML mapping at the top level of {path}")
    return loaded


def _load_defaultable(path: Path, explicit: bool) -> dict[str, Any]:
    """Load a config file, tolerating absence only when it was not explicitly requested.

    Without an explicit ``--config`` flag, a missing file *or* a missing PyYAML
    install quietly means "no overrides" — this keeps the stdlib-only CLI truly
    dependency-free. An explicitly passed config always fails loudly.
    """
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"config not found: {path}")
        return {}
    try:
        return load_yaml(path)
    except ImportError:
        if explicit:
            raise
        logger.warning("PyYAML not installed; ignoring %s and using built-in defaults", path)
        return {}


def dataset_overrides(name: str, config_path: str | Path | None = None) -> dict[str, Any]:
    """Return per-dataset overrides (urls, checksums, subset limits).

    With no explicit ``config_path``, the repository's
    ``configs/data/datasets.yaml`` is used when present; a missing file simply
    means no overrides.
    """
    path = Path(config_path) if config_path else DEFAULT_DATASETS_CONFIG
    config = _load_defaultable(path, explicit=config_path is not None)
    overrides = config.get(name, {})
    if not isinstance(overrides, dict):
        raise ValueError(f"expected a mapping for dataset {name!r} in {path}")
    return overrides


def split_defaults(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return split strategy defaults from ``configs/data/splits.yaml`` if present."""
    path = Path(config_path) if config_path else DEFAULT_SPLITS_CONFIG
    return _load_defaultable(path, explicit=config_path is not None)
