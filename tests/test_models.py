"""Stdlib-only tests for sehat.models.

torch/torchvision are not required: the factory validates arguments before
importing them, and TemperatureScaler's optimiser is pure Python. Tests
that would need torch are skipped when it is unavailable.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sehat.models import (
    TemperatureScaler,
    build_model,
    load_backbone_from_ckpt,
    supported_archs,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _mean_nll(logits: list[float], labels: list[float], temperature: float) -> float:
    total = 0.0
    for z, y in zip(logits, labels, strict=True):
        x = z / temperature
        total += max(x, 0.0) + math.log1p(math.exp(-abs(x))) - y * x
    return total / len(logits)


class TestFactory(unittest.TestCase):
    def test_supported_archs(self) -> None:
        archs = supported_archs()
        self.assertEqual(set(archs), {"efficientnet_b0", "convnext_tiny", "resnet50"})

    def test_unknown_arch_raises_before_torch_import(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_model("not_a_real_arch")
        self.assertIn("not_a_real_arch", str(ctx.exception))

    def test_invalid_num_classes_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_model("resnet50", num_classes=0)

    def test_invalid_dropout_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_model("resnet50", dropout=1.5)

    @unittest.skipIf(TORCH_AVAILABLE, "torch is installed")
    def test_missing_torch_raises_import_error(self) -> None:
        with self.assertRaises(ImportError):
            build_model("resnet50")

    def test_load_backbone_missing_ckpt_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_backbone_from_ckpt("/nonexistent/path/model.ckpt")


class TestTemperatureScaler(unittest.TestCase):
    def test_unfit_scaler_is_identity(self) -> None:
        scaler = TemperatureScaler()
        self.assertFalse(scaler.is_fitted)
        self.assertEqual(scaler.temperature, 1.0)
        self.assertEqual(scaler.calibrate([1.5, -2.0]), [1.5, -2.0])

    def test_fit_recovers_true_temperature(self) -> None:
        rng = random.Random(7)
        true_temperature = 3.0
        logits = [rng.uniform(-4.0, 4.0) for _ in range(4000)]
        labels = [
            1.0 if rng.random() < 1.0 / (1.0 + math.exp(-z / true_temperature)) else 0.0
            for z in logits
        ]
        scaler = TemperatureScaler().fit(logits, labels)
        self.assertTrue(scaler.is_fitted)
        self.assertAlmostEqual(scaler.temperature, true_temperature, delta=0.75)

    def test_fit_reduces_nll(self) -> None:
        logits = [4.0, -4.0, 4.0, -4.0, 3.0, -3.0]
        labels = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        scaler = TemperatureScaler().fit(logits, labels)
        self.assertLessEqual(
            _mean_nll(logits, labels, scaler.temperature),
            _mean_nll(logits, labels, 1.0),
        )

    def test_calibrate_divides_by_temperature(self) -> None:
        scaler = TemperatureScaler()
        scaler.temperature = 2.5
        self.assertEqual(scaler.calibrate([5.0, -2.5]), [2.0, -1.0])

    def test_accepts_tuple_and_nested_inputs(self) -> None:
        scaler = TemperatureScaler().fit((1.0, -1.0), (1, 0))
        calibrated = scaler.calibrate([[2.0], [-2.0]])
        self.assertEqual(len(calibrated), 2)

    def test_fit_returns_self(self) -> None:
        scaler = TemperatureScaler()
        self.assertIs(scaler.fit([0.5, -0.5], [1, 0]), scaler)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([1.0, 2.0], [1])

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([], [])

    def test_non_binary_labels_raise(self) -> None:
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([1.0, -1.0], [1, 2])


if __name__ == "__main__":
    unittest.main()
