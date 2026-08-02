"""Tests for sehat.data.manifest — schema, validation, and I/O.

Stdlib-first: pandas-dependent cases skip cleanly when pandas is unavailable.
Runnable with either ``pytest`` or ``python -m unittest``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sehat.data.manifest import (
    AGE_UNKNOWN,
    COLUMNS,
    ManifestValidationError,
    iter_manifest_rows,
    validate_manifest_csv,
    validate_rows,
    write_manifest_csv,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on environment
    pd = None


def valid_row(**overrides: object) -> dict[str, str]:
    row = {
        "image_path": "CXR_png/CHNCXR_0001_1.png",
        "label": "1",
        "disease": "tb",
        "site": "shenzhen",
        "patient_id": "CHNCXR_0001",
        "split": "train",
        "sex": "M",
        "age": "42",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


class ValidateRowsTest(unittest.TestCase):
    def test_valid_rows_pass(self) -> None:
        validate_rows(
            [valid_row(), valid_row(age=""), valid_row(sex="unknown", label="0", disease="normal")]
        )

    def test_row_numbering_starts_at_two(self) -> None:
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_rows([valid_row(label="7")])
        self.assertIn("row 2:", str(ctx.exception))

    def test_label_must_be_binary(self) -> None:
        with self.assertRaises(ManifestValidationError):
            validate_rows([valid_row(label="2")])

    def test_disease_vocabulary(self) -> None:
        with self.assertRaises(ManifestValidationError):
            validate_rows([valid_row(disease="covid")])

    def test_split_vocabulary(self) -> None:
        with self.assertRaises(ManifestValidationError):
            validate_rows([valid_row(split="dev")])
        validate_rows([valid_row(split="test_external")])

    def test_sex_vocabulary(self) -> None:
        with self.assertRaises(ManifestValidationError):
            validate_rows([valid_row(sex="X")])

    def test_age_bounds_and_unknown(self) -> None:
        validate_rows(
            [
                valid_row(age="0"),
                valid_row(age="120"),
                valid_row(age=""),
                valid_row(age=str(AGE_UNKNOWN)),
            ]
        )
        for bad in ("-5", "121", "forty-two"):
            with self.assertRaises(ManifestValidationError, msg=f"age={bad}"):
                validate_rows([valid_row(age=bad)])

    def test_label_disease_consistency(self) -> None:
        with self.assertRaises(ManifestValidationError):
            validate_rows([valid_row(label="1", disease="normal")])
        with self.assertRaises(ManifestValidationError):
            validate_rows([valid_row(label="0", disease="tb")])

    def test_missing_column_reported(self) -> None:
        row = valid_row()
        del row["patient_id"]
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_rows([row])
        self.assertIn("patient_id", str(ctx.exception))

    def test_multiple_errors_aggregated(self) -> None:
        with self.assertRaises(ManifestValidationError) as ctx:
            validate_rows([valid_row(label="9", split="nope", site="")])
        self.assertGreaterEqual(len(ctx.exception.errors), 3)


class ManifestCsvTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        rows = [valid_row(), valid_row(label="0", disease="normal", age="")]
        with TemporaryDirectory() as tmp:
            path = write_manifest_csv(rows, Path(tmp) / "manifest.csv")
            validate_manifest_csv(path)
            loaded = list(iter_manifest_rows(path))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(tuple(loaded[0].keys()), COLUMNS)

    def test_header_must_match_exactly(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("image_path,label\nfoo.png,1\n", encoding="utf-8")
            with self.assertRaises(ManifestValidationError):
                validate_manifest_csv(path)

    def test_write_validates_before_writing(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            with self.assertRaises(ManifestValidationError):
                write_manifest_csv([valid_row(split="")], path)
            self.assertFalse(path.exists())


@unittest.skipIf(pd is None, "pandas not installed")
class PandasIOTest(unittest.TestCase):
    def test_load_manifest_types_and_age_unknown(self) -> None:
        from sehat.data.manifest import load_manifest

        rows = [valid_row(age="35"), valid_row(age="")]
        with TemporaryDirectory() as tmp:
            path = write_manifest_csv(rows, Path(tmp) / "manifest.csv")
            df = load_manifest(path)
        self.assertEqual(list(df.columns), list(COLUMNS))
        self.assertEqual(df["label"].tolist(), [1, 1])
        self.assertEqual(df["age"].tolist(), [35, AGE_UNKNOWN])

    def test_load_manifest_raises_on_bad_rows(self) -> None:
        from sehat.data.manifest import load_manifest

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text(
                ",".join(COLUMNS) + "\n" + ",".join(valid_row(split="oops").values()) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ManifestValidationError):
                load_manifest(path)

    def test_write_manifest_roundtrip(self) -> None:
        from sehat.data.manifest import load_manifest, write_manifest

        rows = [valid_row(age=""), valid_row(age="35")]
        with TemporaryDirectory() as tmp:
            first = write_manifest_csv(rows, Path(tmp) / "a.csv")
            df = load_manifest(first)
            second = Path(tmp) / "b.csv"
            write_manifest(df, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
