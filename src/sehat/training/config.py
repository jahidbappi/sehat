"""Training configuration schema.

Deliberately implemented with stdlib ``dataclasses`` (not pydantic) so the
schema can be imported and unit-tested without third-party dependencies.
YAML loading is supported via :meth:`TrainConfig.from_yaml`, which guards
its ``yaml`` import and raises an actionable error when PyYAML is absent;
:meth:`TrainConfig.from_dict` is the dependency-free path used by tests.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

__all__ = ["TrainConfig"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Validated configuration for a Sehat training run.

    Groups:
        Model: ``arch``, ``pretrained``, ``dropout``, ``image_size``.
        Data: ``manifest_path``, ``disease``, ``batch_size``,
            ``num_workers``, ``weighted_sampler``, ``pos_weight``.
        Optimisation: ``lr``, ``weight_decay``, ``epochs``, ``patience``,
            ``seed``.
        Tracking/artifacts: ``mlflow_experiment``, ``mlflow_tracking_uri``,
            ``checkpoint_dir``.

    ``pos_weight`` and ``weighted_sampler`` are the two class-imbalance
    levers; use one or the other for a given run (both enabled would
    double-count the minority class).
    """

    # Model
    arch: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2
    image_size: int = 224
    # Data
    manifest_path: str = "data/manifest.csv"
    disease: str = "tb"
    batch_size: int = 32
    num_workers: int = 4
    weighted_sampler: bool = True
    pos_weight: float | None = None
    # Optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 5
    seed: int = 42
    # Tracking / artifacts
    mlflow_experiment: str = "sehat"
    mlflow_tracking_uri: str | None = None
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        if not self.arch:
            raise ValueError("arch must be a non-empty string")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if not self.manifest_path:
            raise ValueError("manifest_path must be a non-empty string")
        if not self.disease:
            raise ValueError("disease must be a non-empty string")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.pos_weight is not None and self.pos_weight <= 0.0:
            raise ValueError(f"pos_weight must be positive when set, got {self.pos_weight}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if not self.mlflow_experiment:
            raise ValueError("mlflow_experiment must be a non-empty string")
        if not self.checkpoint_dir:
            raise ValueError("checkpoint_dir must be a non-empty string")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        """Build a config from a mapping, rejecting unknown keys.

        Types are coerced leniently where safe (e.g. ``int`` accepted for
        ``float`` fields) and validated strictly otherwise.

        Raises:
            ValueError: On unknown field names.
            TypeError: On values that cannot be coerced to the field type.
        """
        field_map = {f.name: f for f in fields(cls)}
        unknown = sorted(set(data) - set(field_map))
        if unknown:
            raise ValueError(
                f"Unknown TrainConfig field(s): {', '.join(unknown)}. "
                f"Valid fields: {', '.join(sorted(field_map))}"
            )
        hints = get_type_hints(cls)
        kwargs = {name: _coerce(hints[name], value, name) for name, value in data.items()}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load a config from a YAML file.

        Raises:
            RuntimeError: If PyYAML is not installed.
            ValueError: If the file does not contain a mapping.
        """
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Loading YAML configs requires PyYAML (`pip install pyyaml`). "
                "In stdlib-only environments use TrainConfig.from_dict instead."
            ) from exc
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"YAML config must be a mapping, got {type(data).__name__} in {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict (round-trips via ``from_dict``)."""
        return dataclasses.asdict(self)


def _coerce(hint: Any, value: Any, name: str) -> Any:
    """Coerce ``value`` to the annotated type of field ``name``."""
    if get_origin(hint) is Union:
        args = [arg for arg in get_args(hint) if arg is not type(None)]
        if value is None:
            if len(args) < len(get_args(hint)):
                return None
            raise TypeError(f"Field {name!r} may not be null")
        if len(args) == 1:
            return _coerce(args[0], value, name)
        raise TypeError(f"Field {name!r} has an unsupported union type: {hint}")
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"Field {name!r} expects a bool, got {value!r}")
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Field {name!r} expects an int, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Field {name!r} expects a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(f"Field {name!r} expects a string, got {value!r}")
        return value
    raise TypeError(f"Field {name!r} has an unsupported annotation: {hint}")
