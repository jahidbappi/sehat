"""Stdlib-only contract tests for sehat.export and the serve config.

Run with:  python3 -m unittest tests.test_export_config -v
These tests must pass without torch/onnx/onnxruntime/numpy/PyYAML installed.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sehat.export import benchmark as benchmark_mod  # noqa: E402
from sehat.export import onnx_export, quantize  # noqa: E402
from sehat.serving.config import (  # noqa: E402
    ENV_MODEL_PATH,
    ConfigError,
    ServeConfig,
    load_serve_config,
    parse_simple_yaml,
)

CONFIG_PATH = REPO_ROOT / "configs" / "serve" / "default.yaml"


class TestServeConfigFile(unittest.TestCase):
    """The shipped YAML must parse with the stdlib subset parser and match
    the export/benchmark contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_serve_config(CONFIG_PATH, env={})

    def test_config_file_exists(self) -> None:
        self.assertTrue(CONFIG_PATH.is_file())

    def test_required_sections_present(self) -> None:
        raw = parse_simple_yaml(CONFIG_PATH.read_text(encoding="utf-8"))
        for section in (
            "service",
            "model",
            "preprocessing",
            "inference",
            "export",
            "benchmark",
            "limits",
        ):
            self.assertIn(section, raw)

    def test_export_section_matches_onnx_export_constants(self) -> None:
        self.assertEqual(self.cfg.opset, onnx_export.DEFAULT_OPSET)
        self.assertEqual(self.cfg.image_size, onnx_export.DEFAULT_IMAGE_SIZE)
        self.assertTrue(self.cfg.dynamic_batch)

    def test_benchmark_section_matches_benchmark_signature(self) -> None:
        sig = inspect.signature(benchmark_mod.benchmark_latency)
        self.assertEqual(sig.parameters["n_runs"].default, self.cfg.benchmark_runs)
        self.assertEqual(sig.parameters["warmup"].default, self.cfg.benchmark_warmup)
        self.assertEqual((self.cfg.benchmark_runs, self.cfg.benchmark_warmup), (50, 10))

    def test_preprocessing_is_imagenet_224(self) -> None:
        self.assertEqual(self.cfg.image_size, 224)
        self.assertEqual(self.cfg.mean, (0.485, 0.456, 0.406))
        self.assertEqual(self.cfg.std, (0.229, 0.224, 0.225))

    def test_binary_labels_negative_first(self) -> None:
        self.assertEqual(self.cfg.labels, ("negative", "positive"))

    def test_threshold_in_unit_interval(self) -> None:
        self.assertGreater(self.cfg.threshold, 0.0)
        self.assertLess(self.cfg.threshold, 1.0)

    def test_env_model_path_override(self) -> None:
        cfg = load_serve_config(CONFIG_PATH, env={ENV_MODEL_PATH: "/tmp/m.onnx"})
        self.assertEqual(cfg.model_path, "/tmp/m.onnx")

    def test_explicit_override_beats_env(self) -> None:
        cfg = load_serve_config(
            CONFIG_PATH,
            model_path_override="/tmp/explicit.onnx",
            env={ENV_MODEL_PATH: "/tmp/env.onnx"},
        )
        self.assertEqual(cfg.model_path, "/tmp/explicit.onnx")

    def test_missing_explicit_path_raises(self) -> None:
        with self.assertRaises(ConfigError):
            load_serve_config(REPO_ROOT / "configs" / "serve" / "nope.yaml", env={})


class TestSubsetYamlParser(unittest.TestCase):
    def test_scalars_and_nesting(self) -> None:
        data = parse_simple_yaml(
            "a: 1\n"
            "b: 2.5\n"
            "c: true\n"
            "d: null\n"
            'e: "quoted # not a comment"\n'
            "f: [x, y, 3]\n"
            "g:\n"
            "  h: deep\n"
            "  i: [1, 2]\n"
        )
        self.assertEqual(
            data,
            {
                "a": 1,
                "b": 2.5,
                "c": True,
                "d": None,
                "e": "quoted # not a comment",
                "f": ["x", "y", 3],
                "g": {"h": "deep", "i": [1, 2]},
            },
        )

    def test_comments_and_blank_lines_ignored(self) -> None:
        data = parse_simple_yaml("# header\n\nkey: value  # trailing\n")
        self.assertEqual(data, {"key": "value"})

    def test_block_sequences_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            parse_simple_yaml("items:\n  - one\n")

    def test_invalid_config_values_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            ServeConfig(threshold=1.5)
        with self.assertRaises(ConfigError):
            ServeConfig(std=(1.0, 0.0, 1.0))
        with self.assertRaises(ConfigError):
            ServeConfig(labels=())


class TestExportContracts(unittest.TestCase):
    def test_benchmark_latency_signature(self) -> None:
        sig = inspect.signature(benchmark_mod.benchmark_latency)
        params = list(sig.parameters.values())
        self.assertEqual([p.name for p in params], ["onnx_path", "n_runs", "warmup"])
        self.assertEqual(params[1].default, 50)
        self.assertEqual(params[2].default, 10)

    def test_export_onnx_signature(self) -> None:
        sig = inspect.signature(onnx_export.export_onnx)
        params = sig.parameters
        self.assertEqual(
            list(params), ["model", "out_path", "image_size", "opset", "dynamic_batch"]
        )
        self.assertEqual(params["image_size"].default, 224)
        self.assertEqual(params["opset"].default, 17)
        self.assertIs(params["dynamic_batch"].default, True)

    def test_int8_path_naming(self) -> None:
        self.assertEqual(
            quantize.int8_path_for("artifacts/tb.onnx"),
            Path("artifacts/tb.int8.onnx"),
        )
        self.assertEqual(quantize.int8_path_for(Path("/m/model.onnx")).name, "model.int8.onnx")

    def test_percentile_linear_interpolation(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(benchmark_mod.percentile(samples, 0.0), 1.0)
        self.assertEqual(benchmark_mod.percentile(samples, 100.0), 4.0)
        self.assertAlmostEqual(benchmark_mod.percentile(samples, 50.0), 2.5)
        self.assertAlmostEqual(benchmark_mod.percentile(samples, 95.0), 3.85)

    def test_percentile_edge_cases(self) -> None:
        self.assertEqual(benchmark_mod.percentile([7.0], 95.0), 7.0)
        with self.assertRaises(ValueError):
            benchmark_mod.percentile([], 50.0)
        with self.assertRaises(ValueError):
            benchmark_mod.percentile([1.0], 101.0)

    def test_benchmark_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            benchmark_mod.benchmark_latency(REPO_ROOT / "no-such-model.onnx")


if __name__ == "__main__":
    unittest.main()
