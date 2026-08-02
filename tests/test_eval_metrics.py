"""Tests for sehat.eval.metrics — stdlib unittest + numpy only.

Run with: python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sehat.eval.metrics import (
    auroc,
    average_precision,
    brier_score,
    expected_calibration_error,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)


class TestAUROC(unittest.TestCase):
    def test_known_value(self) -> None:
        # Classic worked example: AUROC = 0.75.
        y = [0, 0, 1, 1]
        s = [0.1, 0.4, 0.35, 0.8]
        self.assertAlmostEqual(auroc(y, s), 0.75)

    def test_perfect_and_reversed(self) -> None:
        y = [0, 0, 1, 1]
        self.assertAlmostEqual(auroc(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(auroc(y, [0.8, 0.9, 0.1, 0.2]), 0.0)

    def test_all_scores_tied_gives_half(self) -> None:
        y = [0, 0, 1, 1]
        s = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(auroc(y, s), 0.5)

    def test_partial_ties(self) -> None:
        # Positive tied with a negative counts as half a win.
        y = [0, 1, 1]
        s = [0.5, 0.5, 0.9]
        # wins: 1 (0.9 beats 0.5) + 0.5 (tie) = 1.5 over 2 pairs -> 0.75
        self.assertAlmostEqual(auroc(y, s), 0.75)

    def test_accepts_lists_and_arrays(self) -> None:
        y = np.array([0, 1])
        s = np.array([0.2, 0.8])
        self.assertAlmostEqual(auroc(y, s), auroc([0, 1], [0.2, 0.8]))

    def test_single_class_raises(self) -> None:
        with self.assertRaises(ValueError):
            auroc([1, 1, 1], [0.1, 0.2, 0.3])

    def test_non_binary_labels_raise(self) -> None:
        with self.assertRaises(ValueError):
            auroc([0, 2, 1], [0.1, 0.2, 0.3])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            auroc([0, 1], [0.1, 0.2, 0.3])

    def test_non_finite_raises(self) -> None:
        with self.assertRaises(ValueError):
            auroc([0, 1], [0.1, float("nan")])


class TestAveragePrecision(unittest.TestCase):
    def test_known_value(self) -> None:
        # Order by score: 0.8(P), 0.4(N), 0.35(P), 0.1(N)
        # AP = 0.5 * 1.0 + 0.5 * (2/3) = 5/6
        y = [0, 0, 1, 1]
        s = [0.1, 0.4, 0.35, 0.8]
        self.assertAlmostEqual(average_precision(y, s), 5.0 / 6.0)

    def test_perfect_ranking(self) -> None:
        y = [0, 0, 1, 1]
        s = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(average_precision(y, s), 1.0)

    def test_tied_scores(self) -> None:
        # All scores tied: precision at full recall = prevalence.
        y = [0, 0, 1, 1]
        s = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(average_precision(y, s), 0.5)


class TestOperatingPoints(unittest.TestCase):
    def setUp(self) -> None:
        self.y = [0, 0, 1, 1]
        self.s = [0.1, 0.4, 0.35, 0.8]

    def test_sensitivity_at_95_specificity(self) -> None:
        # Only threshold 0.8 reaches spec >= 0.95 (spec=1.0); there sens=0.5.
        self.assertAlmostEqual(
            sensitivity_at_specificity(self.y, self.s, target_specificity=0.95), 0.5
        )

    def test_sensitivity_at_specificity_returns_threshold(self) -> None:
        sens, thr = sensitivity_at_specificity(
            self.y, self.s, target_specificity=0.95, return_threshold=True
        )
        self.assertAlmostEqual(sens, 0.5)
        self.assertAlmostEqual(thr, 0.8)

    def test_sensitivity_at_50_specificity(self) -> None:
        # At threshold 0.35: spec=0.5, sens=1.0 -> best sens with spec>=0.5.
        self.assertAlmostEqual(
            sensitivity_at_specificity(self.y, self.s, target_specificity=0.5), 1.0
        )

    def test_specificity_at_95_sensitivity(self) -> None:
        # sens>=0.95 requires threshold <= 0.35; best spec there is 0.5.
        self.assertAlmostEqual(
            specificity_at_sensitivity(self.y, self.s, target_sensitivity=0.95), 0.5
        )

    def test_specificity_at_sensitivity_returns_threshold(self) -> None:
        spec, thr = specificity_at_sensitivity(
            self.y, self.s, target_sensitivity=0.95, return_threshold=True
        )
        self.assertAlmostEqual(spec, 0.5)
        self.assertAlmostEqual(thr, 0.35)

    def test_invalid_target_raises(self) -> None:
        with self.assertRaises(ValueError):
            sensitivity_at_specificity(self.y, self.s, target_specificity=1.5)
        with self.assertRaises(ValueError):
            specificity_at_sensitivity(self.y, self.s, target_sensitivity=-0.1)

    def test_perfect_model(self) -> None:
        y = [0, 0, 1, 1]
        s = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(sensitivity_at_specificity(y, s, 0.95), 1.0)
        self.assertAlmostEqual(specificity_at_sensitivity(y, s, 0.95), 1.0)


class TestCalibrationMetrics(unittest.TestCase):
    def test_ece_perfectly_calibrated(self) -> None:
        # Single occupied bin with accuracy == mean confidence.
        y = [0] * 50 + [1] * 50
        p = [0.5] * 100
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.0)

    def test_ece_known_miscalibration(self) -> None:
        # All positives predicted at 0.5 -> ECE = |1 - 0.5| = 0.5.
        y = [1] * 100
        p = [0.5] * 100
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.5)

    def test_ece_perfect_probabilities(self) -> None:
        y = [0, 0, 1, 1]
        p = [0.0, 0.0, 1.0, 1.0]
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.0)

    def test_ece_weighted_by_bin_mass(self) -> None:
        # Two bins: [0,0.5) perfectly calibrated, [0.5,1] off by 0.4 on 1/3 of data.
        y = [0, 0, 0, 0, 1, 1]
        p = [0.1, 0.1, 0.2, 0.2, 0.6, 0.6]
        # bin [0,0.5): conf 0.15, acc 0 -> gap 0.15, mass 4/6
        # bin [0.5,1]: conf 0.6, acc 1 -> gap 0.4, mass 2/6
        expected = (4 / 6) * 0.15 + (2 / 6) * 0.4
        self.assertAlmostEqual(expected_calibration_error(y, p, n_bins=2), expected)

    def test_ece_invalid_bins(self) -> None:
        with self.assertRaises(ValueError):
            expected_calibration_error([0, 1], [0.1, 0.9], n_bins=0)

    def test_brier_known_value(self) -> None:
        y = [0, 1]
        p = [0.25, 0.75]
        self.assertAlmostEqual(brier_score(y, p), 0.0625)

    def test_brier_perfect(self) -> None:
        self.assertAlmostEqual(brier_score([0, 1], [0.0, 1.0]), 0.0)

    def test_brier_worst(self) -> None:
        self.assertAlmostEqual(brier_score([0, 1], [1.0, 0.0]), 1.0)

    def test_brier_clips_out_of_range_probs(self) -> None:
        self.assertAlmostEqual(brier_score([1], [1.5]), 0.0)


class TestRandomisedSanity(unittest.TestCase):
    def test_auroc_in_unit_interval_on_random_data(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        s = rng.random(500)
        value = auroc(y, s)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_informative_scores_beat_chance(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 2000)
        s = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.15, 2000), 0, 1)
        self.assertGreater(auroc(y, s), 0.8)


if __name__ == "__main__":
    unittest.main()
