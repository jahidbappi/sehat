"""Tests for sehat.data.splits — leakage safety and determinism.

The pure-stdlib cores are exercised directly; pandas wrappers skip cleanly
when pandas is unavailable. Runnable with either ``pytest`` or
``python -m unittest``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sehat.data.splits import (
    holdout_assignment,
    patient_strata,
    stratified_group_assignment,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on environment
    pd = None


def synthetic_strata(n_positive: int = 50, n_negative: int = 50) -> dict[str, str]:
    strata = {f"pos_{i:04d}": "positive" for i in range(n_positive)}
    strata.update({f"neg_{i:04d}": "negative" for i in range(n_negative)})
    return strata


class PatientStrataTest(unittest.TestCase):
    def test_positive_if_any(self) -> None:
        records = [("p1", 0), ("p1", 1), ("p2", 0), ("p2", 0)]
        self.assertEqual(patient_strata(records), {"p1": "positive", "p2": "negative"})


class StratifiedGroupAssignmentTest(unittest.TestCase):
    def test_partitions_every_key_exactly_once(self) -> None:
        strata = synthetic_strata()
        assignment = stratified_group_assignment(strata, val_frac=0.2, test_frac=0.2, seed=7)
        self.assertEqual(set(assignment), set(strata))
        self.assertEqual(len(assignment), len(strata))
        self.assertLessEqual(set(assignment.values()), {"train", "val", "test"})

    def test_fractions_approximately_honored(self) -> None:
        assignment = stratified_group_assignment(
            synthetic_strata(500, 500), val_frac=0.2, test_frac=0.2, seed=7
        )
        counts = {"train": 0, "val": 0, "test": 0}
        for split in assignment.values():
            counts[split] += 1
        self.assertEqual(counts["val"], 200)
        self.assertEqual(counts["test"], 200)
        self.assertEqual(counts["train"], 600)

    def test_stratified_both_strata_represented(self) -> None:
        assignment = stratified_group_assignment(
            synthetic_strata(50, 50), val_frac=0.2, test_frac=0.2, seed=7
        )
        for split in ("val", "test"):
            strata_in_split = {k[:3] for k, s in assignment.items() if s == split}
            self.assertEqual(strata_in_split, {"pos", "neg"})

    def test_deterministic_given_seed_and_input_order(self) -> None:
        strata = synthetic_strata(20, 20)
        forward = stratified_group_assignment(strata, 0.2, 0.2, seed=123)
        reversed_input = dict(reversed(list(strata.items())))
        backward = stratified_group_assignment(reversed_input, 0.2, 0.2, seed=123)
        different_seed = stratified_group_assignment(strata, 0.2, 0.2, seed=124)
        self.assertEqual(forward, backward)
        self.assertNotEqual(forward, different_seed)

    def test_zero_fractions_assign_all_train(self) -> None:
        assignment = stratified_group_assignment(synthetic_strata(5, 5), 0.0, 0.0, seed=1)
        self.assertEqual(set(assignment.values()), {"train"})

    def test_invalid_fractions_raise(self) -> None:
        with self.assertRaises(ValueError):
            stratified_group_assignment(synthetic_strata(), val_frac=-0.1)
        with self.assertRaises(ValueError):
            stratified_group_assignment(synthetic_strata(), val_frac=0.6, test_frac=0.6)


class HoldoutAssignmentTest(unittest.TestCase):
    def records(self) -> list[tuple[str, str, int]]:
        return (
            [(f"shenzhen_pos_{i}", "shenzhen", 1) for i in range(30)]
            + [(f"shenzhen_neg_{i}", "shenzhen", 0) for i in range(30)]
            + [(f"montgomery_{i}", "montgomery", i % 2) for i in range(10)]
        )

    def test_holdout_site_is_external(self) -> None:
        assignment = holdout_assignment(self.records(), "montgomery", seed=3)
        external = {patient for patient, split in assignment.items() if split == "test_external"}
        self.assertEqual(external, {f"montgomery_{i}" for i in range(10)})
        others = {split for patient, split in assignment.items() if patient not in external}
        self.assertLessEqual(others, {"train", "val", "test"})

    def test_patient_at_both_sites_goes_external(self) -> None:
        records = [("p1", "shenzhen", 1), ("p1", "montgomery", 1)]
        assignment = holdout_assignment(records, "montgomery")
        self.assertEqual(assignment["p1"], "test_external")

    def test_deterministic(self) -> None:
        first = holdout_assignment(self.records(), "montgomery", seed=3)
        second = holdout_assignment(self.records(), "montgomery", seed=3)
        self.assertEqual(first, second)


@unittest.skipIf(pd is None, "pandas not installed")
class PandasWrappersTest(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        rows = {
            "patient_id": [f"p{i:03d}" for i in range(100)],
            "label": [i % 2 for i in range(100)],
            "site": ["shenzhen" if i < 90 else "montgomery" for i in range(100)],
            "split": [""] * 100,
        }
        return pd.DataFrame(rows)

    def test_patient_level_split_no_leakage(self) -> None:
        from sehat.data.splits import patient_level_split

        out = patient_level_split(self.frame(), val_frac=0.2, test_frac=0.2, seed=11)
        by_patient = out.groupby("patient_id")["split"].nunique()
        self.assertTrue((by_patient == 1).all())
        self.assertLessEqual(set(out["split"]), {"train", "val", "test"})

    def test_site_holdout_split(self) -> None:
        from sehat.data.splits import site_holdout_split

        out = site_holdout_split(self.frame(), holdout_site="montgomery")
        external = set(out.loc[out["split"] == "test_external", "patient_id"])
        self.assertEqual(external, {f"p{i:03d}" for i in range(90, 100)})
        internal = out.loc[out["split"] != "test_external", "site"].unique()
        self.assertEqual(list(internal), ["shenzhen"])


if __name__ == "__main__":
    unittest.main()
