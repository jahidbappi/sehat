"""Resumable, checksummed downloads over HTTP(S) and ``file://`` — stdlib only.

Clinics and CI alike need downloads that survive flaky connections and prove
their integrity. Files download to a ``.part`` sibling and are atomically
renamed on success; interrupted HTTP downloads resume via ``Range`` requests;
finished files are verified against an optional SHA-256 and never re-downloaded.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

__all__ = ["CHUNK_SIZE", "ChecksumMismatchError", "download_file", "sha256_file", "verify_checksum"]

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 20  # 1 MiB


class ChecksumMismatchError(IOError):
    """Raised when a downloaded file does not match its expected SHA-256."""


def sha256_file(path: str | Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: str | Path, sha256: str) -> bool:
    """Return True if the file's SHA-256 matches ``sha256`` (case-insensitive)."""
    return sha256_file(path) == sha256.lower()


def download_file(
    url: str,
    dest: str | Path,
    *,
    sha256: str | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``dest`` with resume, atomic rename, and verification.

    - If ``dest`` already exists and matches ``sha256``, it is kept as-is.
    - Interrupted HTTP downloads resume from the ``.part`` file when the server
      honors ``Range``; otherwise the download restarts cleanly.
    - When ``sha256`` is given and the finished download does not match,
      :class:`ChecksumMismatchError` is raised and the ``.part`` file is kept
      for inspection.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if dest.exists():
        if sha256 is not None and verify_checksum(dest, sha256):
            logger.info("already have verified file %s; skipping download", dest)
            return dest
        if sha256 is None:
            logger.info("already have %s (no checksum configured); skipping download", dest)
            return dest
        logger.warning("existing %s failed checksum; re-downloading", dest)

    resume_from = part.stat().st_size if part.exists() else 0
    headers = {}
    if resume_from and url.startswith(("http://", "https://")):
        headers["Range"] = f"bytes={resume_from}-"
        logger.info("resuming %s from byte %d", url, resume_from)

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if resume_from and status != 206:
            logger.info("server did not honor Range; restarting download of %s", url)
            resume_from = 0
        mode = "ab" if resume_from else "wb"
        with part.open(mode) as handle:
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)

    if sha256 is not None and not verify_checksum(part, sha256):
        raise ChecksumMismatchError(
            f"checksum mismatch for {url}: expected sha256 {sha256}, "
            f"got {sha256_file(part)}; kept partial file at {part}"
        )

    part.replace(dest)
    logger.info("downloaded %s -> %s", url, dest)
    return dest
