"""Tests for sehat.data.datasets — parsing logic, no network access.

Synthetic fixtures mimic the publishers' on-disk layouts so every record
parser is tested without downloading a byte. Runnable with either ``pytest``
or ``python -m unittest``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sehat.data.datasets import get_dataset, list_datasets
from sehat.data.datasets._nlm import parse_clinical_reading
from sehat.data.datasets.montgomery import MontgomeryDataset
from sehat.data.datasets.nih_cxr import NIHChestXrayDataset, record_from_metadata_row
from sehat.data.datasets.shenzhen import ShenzhenDataset
from sehat.data.manifest import validate_rows


def build_nlm_layout(root: Path, prefix: str) -> Path:
    """Create a minimal NLM TB dataset layout: 2 positives, 1 normal, 1 mask."""
    images = root / "CXR_png"
    readings = root / "ClinicalReadings"
    images.mkdir(parents=True)
    readings.mkdir(parents=True)
    for stem in (f"{prefix}_0001_1", f"{prefix}_0002_1", f"{prefix}_0003_0"):
        (images / f"{stem}.png").write_bytes(b"\x89PNG fake")
    (images / f"{prefix}_0001_1_mask.png").write_bytes(b"\x89PNG mask")
    (readings / f"{prefix}_0001_1.txt").write_text(
        "Patient Sex: M\nPatient Age: 042\nTB positive findings\n", encoding="utf-8"
    )
    (readings / f"{prefix}_0002_1.txt").write_text(
        "Patient Sex: F\nPatient Age: 007\n", encoding="utf-8"
    )
    return root


class ClinicalReadingTest(unittest.TestCase):
    def test_parses_sex_and_age(self) -> None:
        meta = parse_clinical_reading("Patient Sex: F\nPatient Age: 033\n")
        self.assertEqual(meta, {"sex": "F", "age": "33"})

    def test_missing_fields_become_unknown(self) -> None:
        self.assertEqual(parse_clinical_reading("no metadata here"), {"sex": "unknown", "age": ""})

    def test_impossible_age_dropped(self) -> None:
        self.assertEqual(parse_clinical_reading("Age: 999")["age"], "")


class NLMTBRecordsTest(unittest.TestCase):
    def check_dataset(self, dataset: object, prefix: str, site: str) -> None:
        with TemporaryDirectory() as tmp:
            root = build_nlm_layout(Path(tmp), prefix)
            records = list(dataset.iter_records(root))  # type: ignore[attr-defined]
        self.assertEqual(len(records), 3)  # mask skipped
        validate_rows([{**record, "split": "train"} for record in records])
        by_patient = {record["patient_id"]: record for record in records}
        positive = by_patient[f"{prefix}_0001"]
        self.assertEqual((positive["label"], positive["disease"]), ("1", "tb"))
        self.assertEqual((positive["sex"], positive["age"]), ("M", "42"))
        normal = by_patient[f"{prefix}_0003"]
        self.assertEqual((normal["label"], normal["disease"]), ("0", "normal"))
        self.assertEqual((normal["sex"], normal["age"]), ("unknown", ""))
        self.assertTrue(all(record["site"] == site for record in records))

    def test_shenzhen(self) -> None:
        self.check_dataset(ShenzhenDataset(), "CHNCXR", "shenzhen")

    def test_montgomery(self) -> None:
        self.check_dataset(MontgomeryDataset(), "MCUCXR", "montgomery")


class NIHMetadataPolicyTest(unittest.TestCase):
    def row(self, findings: str, **overrides: str) -> dict[str, str]:
        row = {
            "Image Index": "00000001_000.png",
            "Finding Labels": findings,
            "Patient ID": "42",
            "Patient Age": "58",
            "Patient Gender": "F",
        }
        row.update(overrides)
        return row

    def test_single_label_pneumonia_is_positive(self) -> None:
        record = record_from_metadata_row(self.row("Pneumonia"))
        assert record is not None
        self.assertEqual((record["label"], record["disease"]), ("1", "pneumonia"))

    def test_no_finding_is_normal(self) -> None:
        record = record_from_metadata_row(self.row("No Finding"))
        assert record is not None
        self.assertEqual((record["label"], record["disease"]), ("0", "normal"))

    def test_multilabel_and_other_findings_excluded(self) -> None:
        self.assertIsNone(record_from_metadata_row(self.row("Pneumonia|Edema")))
        self.assertIsNone(record_from_metadata_row(self.row("Mass")))

    def test_impossible_age_and_bad_sex_normalized(self) -> None:
        record = record_from_metadata_row(
            self.row("Pneumonia", **{"Patient Age": "999", "Patient Gender": "?"})
        )
        assert record is not None
        self.assertEqual((record["age"], record["sex"]), ("", "unknown"))

    def test_missing_identifiers_skipped(self) -> None:
        self.assertIsNone(record_from_metadata_row(self.row("Pneumonia", **{"Image Index": ""})))
        self.assertIsNone(record_from_metadata_row(self.row("Pneumonia", **{"Patient ID": ""})))

    def test_iter_records_requires_images_and_caps_classes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images" / "00000001_000.png").write_bytes(b"\x89PNG fake")
            (root / "Data_Entry_2017.csv").write_text(
                "Image Index,Finding Labels,Patient ID,Patient Age,Patient Gender\n"
                "00000001_000.png,Pneumonia,7,58,F\n"
                "00000002_000.png,Pneumonia,8,60,M\n",  # image absent -> skipped
                encoding="utf-8",
            )
            records = list(NIHChestXrayDataset().iter_records(root))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["patient_id"], "7")

    def test_unconfigured_download_fails_loudly(self) -> None:
        with TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            NIHChestXrayDataset().download(tmp)


class RegistryTest(unittest.TestCase):
    def test_registry_lists_all_datasets(self) -> None:
        self.assertEqual(list_datasets(), ["montgomery", "nih_cxr", "shenzhen"])

    def test_get_dataset(self) -> None:
        self.assertIsInstance(get_dataset("shenzhen"), ShenzhenDataset)
        with self.assertRaises(ValueError):
            get_dataset("unknown_dataset")


if __name__ == "__main__":
    unittest.main()
