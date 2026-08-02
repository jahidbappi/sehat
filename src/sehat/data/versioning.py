"""Dataset versioning: DVC when available, checksum sidecars everywhere.

Reproducibility in a clinic-forked world means being able to answer "exactly
which bytes did this model train on?" DVC gives that for the raw data (pointer
files are committed; payloads live in a remote). Where DVC is not installed —
a common case offline — every produced artifact still gets a ``.sha256``
sidecar so integrity is always verifiable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from sehat.data.download import sha256_file

__all__ = ["dvc_add", "dvc_available", "verify_sidecar", "write_sidecar"]

logger = logging.getLogger(__name__)


def dvc_available() -> bool:
    """Return True if the ``dvc`` executable is on PATH."""
    return shutil.which("dvc") is not None


def write_sidecar(path: str | Path) -> Path:
    """Write ``<path>.sha256`` containing the file's digest; returns its path."""
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def verify_sidecar(path: str | Path) -> bool:
    """Verify a file against its ``.sha256`` sidecar. False if either is missing/bad."""
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not path.exists() or not sidecar.exists():
        return False
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    return sha256_file(path) == expected


def dvc_add(paths: Iterable[str | Path], cwd: str | Path | None = None) -> bool:
    """Track ``paths`` with DVC. Returns False (with guidance) if DVC is absent."""
    if not dvc_available():
        logger.info(
            "dvc not installed; skipping version tracking for %s. "
            "Install it (pip install dvc) and run `dvc add` on the data directory.",
            [str(p) for p in paths],
        )
        return False
    targets = [str(p) for p in paths]
    subprocess.run(["dvc", "add", *targets], cwd=cwd, check=True)
    logger.info("tracked with dvc: %s", targets)
    return True
