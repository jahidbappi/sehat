"""Stdlib-only contract tests for sehat.serving.schemas.

Run with:  python3 -m unittest tests.test_serving_schemas -v
These tests must pass without fastapi/pydantic/numpy installed.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sehat.serving.schemas import (  # noqa: E402
    DISCLAIMER,
    PREDICTION_FIELDS,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
    SchemaValidationError,
)

EXPECTED_DISCLAIMER = (
    "Decision-support only. Not a medical diagnosis. Confirm with a qualified radiologist."
)


class TestDisclaimer(unittest.TestCase):
    def test_exact_contract_wording(self) -> None:
        self.assertEqual(DISCLAIMER, EXPECTED_DISCLAIMER)

    def test_disclaimer_is_a_single_sentence_triplet(self) -> None:
        self.assertTrue(DISCLAIMER.endswith("."))
        self.assertNotIn("\n", DISCLAIMER)


class TestHealthResponse(unittest.TestCase):
    def test_default_status_ok(self) -> None:
        self.assertEqual(HealthResponse().to_dict(), {"status": "ok"})


class TestMetadataResponse(unittest.TestCase):
    def test_to_dict_shape(self) -> None:
        payload = MetadataResponse(model_version="1.2.3").to_dict()
        self.assertEqual(set(payload), {"model_version", "labels", "disclaimer"})
        self.assertEqual(payload["model_version"], "1.2.3")
        self.assertEqual(payload["labels"], ["negative", "positive"])
        self.assertEqual(payload["disclaimer"], DISCLAIMER)

    def test_labels_serialized_as_list(self) -> None:
        payload = MetadataResponse(model_version="0.1.0", labels=("normal", "tb")).to_dict()
        self.assertIsInstance(payload["labels"], list)
        self.assertEqual(payload["labels"], ["normal", "tb"])

    def test_empty_version_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            MetadataResponse(model_version="")

    def test_empty_labels_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            MetadataResponse(model_version="0.1.0", labels=())


class TestPredictionResponse(unittest.TestCase):
    def _make(self, **overrides: object) -> PredictionResponse:
        kwargs: dict[str, object] = {
            "probability": 0.87,
            "label": "positive",
            "threshold": 0.5,
            "latency_ms": 123.4,
        }
        kwargs.update(overrides)
        return PredictionResponse(**kwargs)  # type: ignore[arg-type]

    def test_to_dict_matches_contract_keys_and_order(self) -> None:
        payload = self._make().to_dict()
        self.assertEqual(tuple(payload.keys()), PREDICTION_FIELDS)
        self.assertEqual(
            PREDICTION_FIELDS,
            ("probability", "label", "threshold", "latency_ms", "disclaimer"),
        )

    def test_to_dict_values(self) -> None:
        payload = self._make().to_dict()
        self.assertEqual(payload["probability"], 0.87)
        self.assertEqual(payload["label"], "positive")
        self.assertEqual(payload["threshold"], 0.5)
        self.assertEqual(payload["latency_ms"], 123.4)
        self.assertEqual(payload["disclaimer"], DISCLAIMER)

    def test_probability_bounds(self) -> None:
        self._make(probability=0.0)
        self._make(probability=1.0)
        for bad in (-0.01, 1.01, float("nan")):
            with self.assertRaises(SchemaValidationError, msg=f"{bad}"):
                self._make(probability=bad)

    def test_threshold_bounds(self) -> None:
        for bad in (0.0, 1.0, -0.5):
            with self.assertRaises(SchemaValidationError, msg=f"{bad}"):
                self._make(threshold=bad)

    def test_negative_latency_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            self._make(latency_ms=-0.1)

    def test_empty_label_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            self._make(label="")


class TestAppFactoryContract(unittest.TestCase):
    """Guard the create_app signature without importing fastapi."""

    def test_create_app_signature(self) -> None:
        source = (REPO_ROOT / "src" / "sehat" / "serving" / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        funcs = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn("create_app", funcs)
        fn = funcs["create_app"]
        self.assertEqual(len(fn.args.args), 1)
        self.assertEqual(fn.args.args[0].arg, "model_path")
        self.assertIsNotNone(fn.args.args[0].annotation)
        self.assertIsNotNone(fn.returns, "create_app must annotate -> FastAPI")
        self.assertEqual(len(fn.args.defaults), 1, "model_path must default to None")
        self.assertIsInstance(fn.args.defaults[0], ast.Constant)
        self.assertIsNone(fn.args.defaults[0].value)


if __name__ == "__main__":
    unittest.main()
