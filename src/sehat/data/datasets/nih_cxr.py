"""NIH ChestX-ray14 subset (pneumonia + normal).

ChestX-ray14 holds ~112k frontal X-rays from ~30k patients. The full release is
~45 GB across 12 archives hosted on NIH Box; mirrors exist on Kaggle and AWS
Open Data. Because Box links are account-bound, source URLs are **config-driven**
(``configs/data/datasets.yaml``) rather than baked in: the first URL must be the
``Data_Entry_2017.csv`` metadata file, followed by image archive URLs.

To keep the subset small and honest:

- only the first ``max_archives`` archives are downloaded (default 1);
- only single-label ``Pneumonia`` rows become positives and ``No Finding`` rows
  become normals — multi-label pathology rows are excluded rather than mislabeled;
- at most ``max_per_class`` rows per class are kept;
- rows whose image is not present in the downloaded archives are skipped.

Cite Wang et al., 2017 (CVPR) when using this data.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path

from sehat.data.datasets.base import DatasetSpec, ManifestRecord, RawDataset

__all__ = ["NIHChestXrayDataset", "record_from_metadata_row"]

logger = logging.getLogger(__name__)

METADATA_FILENAME = "Data_Entry_2017.csv"


def record_from_metadata_row(row: Mapping[str, str]) -> ManifestRecord | None:
    """Map one ``Data_Entry_2017.csv`` row to a manifest record, or None to skip.

    Kept pure (no filesystem) so the labeling policy is unit-testable: single
    ``Pneumonia`` -> positive, ``No Finding`` -> normal, anything else is
    excluded. Ages outside 0-120 (the source CSV contains a few impossible
    values) are treated as unknown.
    """
    findings = set((row.get("Finding Labels") or "").split("|"))
    if findings == {"Pneumonia"}:
        label, disease = "1", "pneumonia"
    elif findings == {"No Finding"}:
        label, disease = "0", "normal"
    else:
        return None

    try:
        age = int(row.get("Patient Age") or "")
        age_raw = str(age) if 0 <= age <= 120 else ""
    except ValueError:
        age_raw = ""

    sex = (row.get("Patient Gender") or "").strip().upper()
    image = (row.get("Image Index") or "").strip()
    patient_id = (row.get("Patient ID") or "").strip()
    if not image or not patient_id:
        return None

    return {
        "image_path": image,
        "label": label,
        "disease": disease,
        "site": "nih",
        "patient_id": patient_id,
        "split": "",
        "sex": sex if sex in {"M", "F"} else "unknown",
        "age": age_raw,
    }


class NIHChestXrayDataset(RawDataset):
    """Config-driven NIH ChestX-ray14 subset downloader and parser."""

    spec = DatasetSpec(
        name="nih_cxr",
        site="nih",
        disease="pneumonia",
        urls=(),  # required via config: metadata CSV first, then archive URLs
        homepage="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        license="CC0 / public domain (NIH); attribution required",
        citation=(
            "Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: "
            "Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised "
            "Classification and Localization of Common Thorax Diseases. CVPR 2017."
        ),
    )

    def __init__(self, overrides: Mapping[str, object] | None = None) -> None:
        super().__init__(overrides)
        overrides = overrides or {}
        self.max_archives = int(overrides.get("max_archives", 1))
        self.max_per_class = int(overrides.get("max_per_class", 5000))

    def download(self, raw_dir: str | Path) -> list[Path]:
        """Download the metadata CSV plus at most ``max_archives`` image archives."""
        if not self.spec.urls:
            raise ValueError(
                "nih_cxr has no built-in URLs; configure them in configs/data/datasets.yaml "
                "(metadata CSV first, then archive URLs). See the module docstring."
            )
        limited = DatasetSpec(
            name=self.spec.name,
            site=self.spec.site,
            disease=self.spec.disease,
            urls=self.spec.urls[: 1 + self.max_archives],
            sha256=self.spec.checksums()[: 1 + self.max_archives],
            homepage=self.spec.homepage,
            license=self.spec.license,
            citation=self.spec.citation,
        )
        original = self.spec
        self.spec = limited
        try:
            return super().download(raw_dir)
        finally:
            self.spec = original

    def iter_records(self, extracted_dir: str | Path) -> Iterator[ManifestRecord]:
        """Yield records for metadata rows whose image exists in the downloaded subset."""
        extracted_dir = Path(extracted_dir)
        metadata = next(extracted_dir.rglob(METADATA_FILENAME), None)
        if metadata is None:
            raise FileNotFoundError(
                f"{METADATA_FILENAME} not found under {extracted_dir}; download the dataset first"
            )
        available = {path.name for path in extracted_dir.rglob("*.png")}
        kept_per_class = {"1": 0, "0": 0}
        with metadata.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record = record_from_metadata_row(row)
                if record is None or record["image_path"] not in available:
                    continue
                if kept_per_class[record["label"]] >= self.max_per_class:
                    continue
                kept_per_class[record["label"]] += 1
                yield record
        logger.info(
            "nih_cxr subset: %d positives, %d normals",
            kept_per_class["1"],
            kept_per_class["0"],
        )
