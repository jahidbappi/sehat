"""Tests for sehat.eval.fairness and the report contract — unittest + numpy only.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sehat.eval.fairness import age_band, subgroup_report
from sehat.eval.report import evaluate_predictions, load_eval_config

CONTRACT_KEYS = {"auroc", "average_precision", "sensitivity_at_95spec", "ece", "brier", "subgroups"}


def _make_records(seed: int = 0, n_per_sex: int = 60) -> list[dict]:
    """Sex-stratified synthetic cohort with equal, strong separability."""
    rng = np.random.default_rng(seed)
    records = []
    for i, sex in enumerate(("F", "M")):
        y = np.array([0, 1] * (n_per_sex // 2), dtype=float)
        rng.shuffle(y)
        s = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.1, n_per_sex), 0, 1)
        for j in range(n_per_sex):
            records.append(
                {
                    "label": float(y[j]),
                    "score": float(s[j]),
                    "sex": sex,
                    "age": int(rng.integers(10, 80)),
                    "site": "rural" if i == 0 else "urban",
                }
            )
    return records


def _make_disparate_records(seed: int = 1) -> list[dict]:
    """Site A separates well; site B is near chance -> AUROC disparity expected."""
    rng = np.random.default_rng(seed)
    records = []
    for site, quality in (("A", 0.6), ("B", 0.02)):
        y = np.array([0, 1] * 50, dtype=float)
        rng.shuffle(y)
        s = np.clip(0.5 + quality * (y - 0.5) + rng.normal(0, 0.15, 100), 0, 1)
        for j in range(100):
            records.append({"label": float(y[j]), "score": float(s[j]), "site": site})
    return records


class TestAgeBand(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(age_band(5), "0-14")
        self.assertEqual(age_band(30), "15-39")
        self.assertEqual(age_band(50), "40-64")
        self.assertEqual(age_band(90), "65+")

    def test_boundaries(self) -> None:
        self.assertEqual(age_band(14), "0-14")
        self.assertEqual(age_band(15), "15-39")
        self.assertEqual(age_band(65), "65+")

    def test_invalid(self) -> None:
        self.assertEqual(age_band(-1), "unknown")
        self.assertEqual(age_band(float("nan")), "unknown")
        self.assertEqual(age_band("n/a"), "unknown")


class TestSubgroupReport(unittest.TestCase):
    def test_structure_and_group_cols(self) -> None:
        records = _make_records()
        result = subgroup_report(records, n_boot=50, seed=0)
        self.assertIn("pooled", result)
        self.assertIn("subgroups", result)
        self.assertEqual(set(result["subgroups"]), {"sex", "age_band", "site"})
        self.assertEqual(set(result["subgroups"]["sex"]), {"F", "M"})
        entry = result["subgroups"]["sex"]["F"]
        for key in (
            "auroc",
            "auroc_ci",
            "sens_at_95spec",
            "sens_at_95spec_ci",
            "n",
            "insufficient_data",
            "disparity",
        ):
            self.assertIn(key, entry)
        self.assertEqual(entry["n"], 60)
        self.assertEqual(result["pooled"]["n"], 120)

    def test_deterministic_under_seed(self) -> None:
        records = _make_records()
        a = subgroup_report(records, n_boot=100, seed=42)
        b = subgroup_report(records, n_boot=100, seed=42)
        self.assertEqual(
            json.dumps(a, default=str, sort_keys=True),
            json.dumps(b, default=str, sort_keys=True),
        )

    def test_different_seeds_give_same_point_estimates(self) -> None:
        # Point estimates are exact (no sampling); only CIs use the seed.
        records = _make_records()
        a = subgroup_report(records, n_boot=20, seed=1)
        b = subgroup_report(records, n_boot=20, seed=2)
        self.assertAlmostEqual(
            a["subgroups"]["sex"]["F"]["auroc"], b["subgroups"]["sex"]["F"]["auroc"]
        )
        self.assertAlmostEqual(a["pooled"]["auroc"], b["pooled"]["auroc"])

    def test_bootstrap_ci_brackets_point_estimate(self) -> None:
        records = _make_records(n_per_sex=100)
        result = subgroup_report(records, group_cols=("sex",), n_boot=200, seed=3)
        entry = result["subgroups"]["sex"]["F"]
        lo, hi = entry["auroc_ci"]
        self.assertLessEqual(lo, entry["auroc"])
        self.assertGreaterEqual(hi, entry["auroc"])

    def test_disparity_flag_fires(self) -> None:
        records = _make_disparate_records()
        result = subgroup_report(
            records, group_cols=("site",), disparity_margin=0.05, n_boot=50, seed=0
        )
        site_b = result["subgroups"]["site"]["B"]
        self.assertTrue(site_b["disparity"]["auroc"])
        self.assertLess(site_b["auroc"], result["pooled"]["auroc"] - 0.05)
        site_a = result["subgroups"]["site"]["A"]
        self.assertFalse(site_a["disparity"]["auroc"])

    def test_no_disparity_when_balanced(self) -> None:
        records = _make_records()
        result = subgroup_report(
            records, group_cols=("sex",), disparity_margin=0.05, n_boot=50, seed=0
        )
        self.assertFalse(result["subgroups"]["sex"]["F"]["disparity"]["auroc"])
        self.assertFalse(result["subgroups"]["sex"]["M"]["disparity"]["auroc"])

    def test_degenerate_subgroup(self) -> None:
        records = _make_records()
        # A single-class subgroup: AUROC undefined -> NaN, no disparity flag.
        for rec in records:
            if rec["sex"] == "M":
                rec["label"] = 1.0
        result = subgroup_report(records, group_cols=("sex",), n_boot=20, seed=0)
        m = result["subgroups"]["sex"]["M"]
        self.assertTrue(math.isnan(m["auroc"]))
        self.assertTrue(m["insufficient_data"])
        self.assertFalse(m["disparity"]["auroc"])

    def test_age_band_derived_from_age(self) -> None:
        records = [
            {"label": 0, "score": 0.1, "age": 8},
            {"label": 1, "score": 0.9, "age": 12},
            {"label": 0, "score": 0.2, "age": 70},
            {"label": 1, "score": 0.8, "age": 75},
        ]
        result = subgroup_report(records, group_cols=("age_band",), n_boot=5, seed=0)
        self.assertEqual(set(result["subgroups"]["age_band"]), {"0-14", "65+"})

    def test_alternative_field_names(self) -> None:
        records = [
            {"y_true": 0, "y_score": 0.1, "site": "A"},
            {"y_true": 1, "y_score": 0.9, "site": "A"},
            {"y_true": 0, "y_score": 0.2, "site": "B"},
            {"y_true": 1, "y_score": 0.8, "site": "B"},
        ]
        result = subgroup_report(records, group_cols=("site",), n_boot=5, seed=0)
        self.assertAlmostEqual(result["pooled"]["auroc"], 1.0)

    def test_empty_records_raise(self) -> None:
        with self.assertRaises(ValueError):
            subgroup_report([], n_boot=5)


class TestEvaluatePredictionsContract(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.n = 120
        self.y = np.array([0, 1] * (self.n // 2), dtype=float)
        rng.shuffle(self.y)
        self.s = np.clip(0.2 + 0.6 * self.y + rng.normal(0, 0.1, self.n), 0, 1)
        self.groups = {
            "sex": ["F" if i % 2 == 0 else "M" for i in range(self.n)],
            "age": [int(a) for a in rng.integers(5, 85, self.n)],
            "site": ["rural" if i % 3 == 0 else "urban" for i in range(self.n)],
        }

    def test_contract_keys_without_groups(self) -> None:
        report = evaluate_predictions(self.y, self.s, n_boot=20)
        self.assertEqual(set(report.keys()), CONTRACT_KEYS)
        self.assertEqual(report["subgroups"], {})
        for key in ("auroc", "average_precision", "sensitivity_at_95spec", "ece", "brier"):
            self.assertIsInstance(report[key], float)
            self.assertTrue(0.0 <= report[key] <= 1.0)

    def test_subgroup_schema(self) -> None:
        report = evaluate_predictions(self.y, self.s, groups=self.groups, n_boot=20, seed=0)
        subgroups = report["subgroups"]
        self.assertIn("sex:F", subgroups)
        self.assertIn("sex:M", subgroups)
        self.assertTrue(any(k.startswith("site:") for k in subgroups))
        self.assertTrue(any(k.startswith("age_band:") for k in subgroups))
        for entry in subgroups.values():
            self.assertEqual(set(entry.keys()), {"auroc", "sens_at_95spec", "n"})
            self.assertIsInstance(entry["n"], int)

    def test_subgroup_counts_sum_to_total(self) -> None:
        report = evaluate_predictions(self.y, self.s, groups=self.groups, n_boot=10, seed=0)
        sex_total = sum(e["n"] for k, e in report["subgroups"].items() if k.startswith("sex:"))
        self.assertEqual(sex_total, self.n)

    def test_writes_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_predictions(self.y, self.s, groups=self.groups, out_dir=tmp, n_boot=20, seed=0)
            json_path = Path(tmp) / "eval_report.json"
            md_path = Path(tmp) / "eval_report.md"
            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            data = json.loads(json_path.read_text())
            self.assertEqual(set(data.keys()), CONTRACT_KEYS)
            self.assertIn("## Subgroup fairness", md_path.read_text())

    def test_json_serialises_nan_as_null(self) -> None:
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        groups = {"sex": ["F", "F", "M", "M", "M", "M"]}
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_predictions(y, s, groups=groups, out_dir=tmp, n_boot=10, seed=0)
            raw = (Path(tmp) / "eval_report.json").read_text()
            self.assertNotIn("NaN", raw)
            json.loads(raw)  # strict parse must succeed

    def test_determinism_end_to_end(self) -> None:
        a = evaluate_predictions(self.y, self.s, groups=self.groups, n_boot=50, seed=11)
        b = evaluate_predictions(self.y, self.s, groups=self.groups, n_boot=50, seed=11)
        self.assertEqual(
            json.dumps(a, default=str, sort_keys=True),
            json.dumps(b, default=str, sort_keys=True),
        )

    def test_group_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions(self.y, self.s, groups={"sex": ["F"]}, n_boot=5)

    def test_model_card_written_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluate_predictions(
                self.y,
                self.s,
                groups=self.groups,
                out_dir=tmp,
                n_boot=10,
                seed=0,
                metadata={"model_name": "tb-screen-v1", "arch": "densenet121"},
            )
            card = (Path(tmp) / "model_card.md").read_text()
            self.assertIn("tb-screen-v1", card)
            self.assertIn("densenet121", card)
            self.assertIn("AUROC", card)


class TestLoadEvalConfig(unittest.TestCase):
    def test_repo_config_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "eval" / "tb_eval.yaml"
        cfg = load_eval_config(cfg_path)
        self.assertEqual(cfg["disease"], "TB")
        self.assertEqual(cfg["split"], "test")
        self.assertAlmostEqual(cfg["target_specificity"], 0.95)
        self.assertEqual(cfg["group_cols"], ["sex", "age_band", "site"])
        self.assertEqual(cfg["n_boot"], 1000)
        self.assertEqual(cfg["seed"], 0)
        self.assertFalse(cfg["pretrained"])


if __name__ == "__main__":
    unittest.main()
