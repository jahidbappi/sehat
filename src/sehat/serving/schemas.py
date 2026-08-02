"""Request/response schemas for the Sehat inference service.

This module is intentionally stdlib-only (``dataclasses``) so it can be
imported in environments without FastAPI or pydantic installed — e.g. the
PWA/build tooling, contract tests, and lightweight CI jobs. ``app.py``
(which runs only where FastAPI is installed) consumes these dataclasses
directly and serializes them via :meth:`to_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Mandatory disclaimer attached to every clinical response. The wording is
#: contractual — the PWA and docs render it verbatim — so do not edit it
#: without a coordinated change across the repo.
DISCLAIMER: Final[str] = (
    "Decision-support only. Not a medical diagnosis. Confirm with a qualified radiologist."
)

#: Canonical label ordering for the binary screening head (index 0/1).
DEFAULT_LABELS: Final[tuple[str, str]] = ("negative", "positive")


class SchemaValidationError(ValueError):
    """Raised when a response payload fails schema-level validation."""


@dataclass(frozen=True, slots=True)
class HealthResponse:
    """Liveness payload for ``GET /healthz``.

    Deliberately free of model state: orchestrators must be able to restart
    an unhealthy container even when no checkpoint is configured.
    """

    status: str = "ok"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable mapping."""
        return {"status": self.status}


@dataclass(frozen=True, slots=True)
class MetadataResponse:
    """Model metadata payload for ``GET /metadata``."""

    model_version: str
    labels: tuple[str, ...] = DEFAULT_LABELS
    disclaimer: str = DISCLAIMER

    def __post_init__(self) -> None:
        if not self.model_version:
            raise SchemaValidationError("model_version must be non-empty")
        if not self.labels:
            raise SchemaValidationError("labels must be non-empty")
        object.__setattr__(self, "labels", tuple(self.labels))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping (labels as a list)."""
        return {
            "model_version": self.model_version,
            "labels": list(self.labels),
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True, slots=True)
class PredictionResponse:
    """Screening result payload for ``POST /predict``.

    ``probability`` is the model's P(positive) after sigmoid, ``label`` is
    the class name selected by comparing ``probability`` to ``threshold``,
    and ``latency_ms`` is end-to-end handler time (decode, preprocess,
    inference) for the request.
    """

    probability: float
    label: str
    threshold: float
    latency_ms: float
    disclaimer: str = DISCLAIMER

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise SchemaValidationError(f"probability must lie in [0, 1], got {self.probability!r}")
        if not 0.0 < self.threshold < 1.0:
            raise SchemaValidationError(f"threshold must lie in (0, 1), got {self.threshold!r}")
        if self.latency_ms < 0.0:
            raise SchemaValidationError(f"latency_ms must be non-negative, got {self.latency_ms!r}")
        if not self.label:
            raise SchemaValidationError("label must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping with the contract key order."""
        return {
            "probability": self.probability,
            "label": self.label,
            "threshold": self.threshold,
            "latency_ms": self.latency_ms,
            "disclaimer": self.disclaimer,
        }


#: Fields of the ``POST /predict`` response, in contract order. Other workers
#: (PWA, docs) may assert against this tuple.
PREDICTION_FIELDS: Final[tuple[str, ...]] = (
    "probability",
    "label",
    "threshold",
    "latency_ms",
    "disclaimer",
)
