"""Base interfaces for raw dataset plugins.

A dataset plugin knows how to do three things with one public source:

1. :meth:`RawDataset.download` — fetch the publisher's archives, resumably and
   checksummed, into ``raw_dir``.
2. :meth:`RawDataset.extract` — unpack them (idempotently) into an extracted dir.
3. :meth:`RawDataset.iter_records` — walk the extracted files and yield unified
   manifest records (dicts keyed by :data:`sehat.data.manifest.COLUMNS` with an
   empty ``split``; splits are assigned later by :mod:`sehat.data.splits`).

Raw data always lives under the gitignored ``data/`` directory — never inside
the package.
"""

from __future__ import annotations

import logging
import tarfile
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sehat.data.download import download_file

__all__ = ["DatasetSpec", "ManifestRecord", "RawDataset", "filename_from_url"]

logger = logging.getLogger(__name__)

ManifestRecord = dict[str, str]
"""One unified manifest row (column name -> raw string value)."""


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata describing a public dataset source."""

    name: str
    site: str
    disease: str
    urls: tuple[str, ...]
    sha256: tuple[str | None, ...] = field(default_factory=tuple)
    homepage: str = ""
    license: str = ""
    citation: str = ""

    def checksums(self) -> tuple[str | None, ...]:
        """Return one optional checksum per URL (padded with None if unset)."""
        return self.sha256 + (None,) * (len(self.urls) - len(self.sha256))


def filename_from_url(url: str, fallback: str) -> str:
    """Best-effort filename for a download URL."""
    name = url.split("?", maxsplit=1)[0].rsplit("/", maxsplit=1)[-1]
    return name or fallback


class RawDataset(ABC):
    """Fetch, unpack, and normalize one public dataset into manifest records."""

    spec: DatasetSpec

    def __init__(self, overrides: Mapping[str, object] | None = None) -> None:
        """Apply config overrides (urls, checksums) on top of the baked-in spec."""
        overrides = overrides or {}
        urls = tuple(str(u) for u in overrides.get("urls", self.spec.urls))  # type: ignore[arg-type]
        raw_sums = overrides.get("sha256", self.spec.sha256)
        sha256 = tuple(None if s is None else str(s) for s in raw_sums)  # type: ignore[union-attr]
        self.spec = DatasetSpec(
            name=self.spec.name,
            site=self.spec.site,
            disease=self.spec.disease,
            urls=urls,
            sha256=sha256,
            homepage=self.spec.homepage,
            license=self.spec.license,
            citation=self.spec.citation,
        )

    def download(self, raw_dir: str | Path) -> list[Path]:
        """Download all source archives into ``raw_dir``; returns local paths."""
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for url, sha256 in zip(self.spec.urls, self.spec.checksums(), strict=True):
            dest = raw_dir / filename_from_url(url, fallback=f"{self.spec.name}.bin")
            paths.append(download_file(url, dest, sha256=sha256))
        return paths

    def extract(self, raw_dir: str | Path) -> Path:
        """Extract downloaded archives into ``raw_dir/<name>/extracted`` (idempotent)."""
        raw_dir = Path(raw_dir)
        out_dir = raw_dir / self.spec.name / "extracted"
        marker = out_dir / ".extracted_ok"
        if marker.exists():
            logger.info("%s already extracted at %s", self.spec.name, out_dir)
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for archive in raw_dir.iterdir():
            if not archive.is_file() or archive.name.endswith(".part"):
                continue
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(out_dir)
                logger.info("extracted %s", archive)
            elif tarfile.is_tarfile(archive):
                with tarfile.open(archive) as tf:
                    tf.extractall(out_dir, filter="data")
                logger.info("extracted %s", archive)
        marker.touch()
        return out_dir

    @abstractmethod
    def iter_records(self, extracted_dir: str | Path) -> Iterator[ManifestRecord]:
        """Yield unified manifest records for every usable image under ``extracted_dir``."""

    def records(self, raw_dir: str | Path) -> Iterator[ManifestRecord]:
        """Full pipeline: download -> extract -> yield records."""
        self.download(raw_dir)
        return self.iter_records(self.extract(raw_dir))
