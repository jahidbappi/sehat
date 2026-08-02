"""Stdlib-only tests for sehat.training.config.TrainConfig.

TrainConfig is a plain dataclass; these tests exercise from_dict /
from_yaml / validation without torch, Lightning, or pydantic.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sehat.training.config import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TestTrainConfigDefaults(unittest.TestCase):
    def test_defaults_construct(self) -> None:
        config = TrainConfig()
        self.assertEqual(config.arch, "efficientnet_b0")
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 20)
        self.assertEqual(config.seed, 42)
        self.assertIsNone(config.pos_weight)
        self.assertEqual(config.mlflow_experiment, "sehat")
        self.assertEqual(config.checkpoint_dir, "checkpoints")

    def test_frozen(self) -> None:
        config = TrainConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.lr = 1.0  # type: ignore[misc]

    def test_round_trip_via_dict(self) -> None:
        config = TrainConfig(arch="convnext_tiny", lr=1e-3, pos_weight=2.5, epochs=7)
        self.assertEqual(TrainConfig.from_dict(config.to_dict()), config)


class TestFromDict(unittest.TestCase):
    def test_full_dict(self) -> None:
        config = TrainConfig.from_dict(
            {
                "arch": "resnet50",
                "pretrained": False,
                "dropout": 0.1,
                "image_size": 320,
                "manifest_path": "data/m.csv",
                "disease": "pneumonia",
                "batch_size": 16,
                "num_workers": 0,
                "weighted_sampler": False,
                "pos_weight": 3,
                "lr": 1e-4,
                "weight_decay": 0.0,
                "epochs": 5,
                "patience": 2,
                "seed": 7,
                "mlflow_experiment": "sehat-test",
                "mlflow_tracking_uri": "http://localhost:5000",
                "checkpoint_dir": "ckpt",
            }
        )
        self.assertEqual(config.arch, "resnet50")
        self.assertEqual(config.pos_weight, 3.0)  # int coerced to float
        self.assertIsInstance(config.lr, float)

    def test_unknown_field_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            TrainConfig.from_dict({"not_a_field": 1})
        self.assertIn("not_a_field", str(ctx.exception))

    def test_bool_not_accepted_as_int(self) -> None:
        with self.assertRaises(TypeError):
            TrainConfig.from_dict({"batch_size": True})

    def test_string_not_accepted_as_float(self) -> None:
        with self.assertRaises(TypeError):
            TrainConfig.from_dict({"lr": "fast"})

    def test_float_not_accepted_as_int(self) -> None:
        with self.assertRaises(TypeError):
            TrainConfig.from_dict({"epochs": 2.5})

    def test_nullable_fields_accept_none(self) -> None:
        config = TrainConfig.from_dict({"pos_weight": None, "mlflow_tracking_uri": None})
        self.assertIsNone(config.pos_weight)
        self.assertIsNone(config.mlflow_tracking_uri)


class TestValidation(unittest.TestCase):
    def _assert_invalid(self, **overrides: object) -> None:
        with self.assertRaises(ValueError):
            TrainConfig(**overrides)  # type: ignore[arg-type]

    def test_invalid_values(self) -> None:
        for overrides in (
            {"arch": ""},
            {"dropout": 1.0},
            {"image_size": 0},
            {"manifest_path": ""},
            {"disease": ""},
            {"batch_size": 0},
            {"num_workers": -1},
            {"pos_weight": 0.0},
            {"pos_weight": -2.0},
            {"lr": 0.0},
            {"weight_decay": -1e-4},
            {"epochs": 0},
            {"patience": 0},
            {"seed": -1},
            {"mlflow_experiment": ""},
            {"checkpoint_dir": ""},
        ):
            with self.subTest(overrides=overrides):
                self._assert_invalid(**overrides)


class TestFromYaml(unittest.TestCase):
    @unittest.skipUnless(YAML_AVAILABLE, "PyYAML not installed")
    def test_yaml_round_trip(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("arch: resnet50\nlr: 0.001\nepochs: 3\npos_weight: 2.0\n")
            path = handle.name
        config = TrainConfig.from_yaml(path)
        self.assertEqual(config.arch, "resnet50")
        self.assertEqual(config.lr, 0.001)
        self.assertEqual(config.pos_weight, 2.0)

    @unittest.skipUnless(YAML_AVAILABLE, "PyYAML not installed")
    def test_repo_configs_are_valid(self) -> None:
        expected = {
            "tb_baseline.yaml": ("tb", "efficientnet_b0"),
            "pneumonia_baseline.yaml": ("pneumonia", "convnext_tiny"),
        }
        for filename, (disease, arch) in expected.items():
            with self.subTest(config=filename):
                config = TrainConfig.from_yaml(REPO_ROOT / "configs" / "train" / filename)
                self.assertEqual(config.disease, disease)
                self.assertEqual(config.arch, arch)

    @unittest.skipIf(YAML_AVAILABLE, "PyYAML is installed")
    def test_missing_pyyaml_raises_actionable_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            TrainConfig.from_yaml("anything.yaml")
        self.assertIn("pyyaml", str(ctx.exception).lower())


class TestLazyImports(unittest.TestCase):
    def test_training_package_imports_without_torch(self) -> None:
        import sehat.training

        self.assertIs(sehat.training.TrainConfig, TrainConfig)
        self.assertTrue(callable(sehat.training.load_backbone_from_ckpt))

    def test_train_entrypoint_module_imports_without_torch(self) -> None:
        import sehat.training.train

        self.assertTrue(callable(sehat.training.train.main))

    @unittest.skipIf(TORCH_AVAILABLE, "torch is installed")
    def test_lit_module_attr_raises_import_error_without_torch(self) -> None:
        import sehat.training

        with self.assertRaises(ImportError):
            _ = sehat.training.SehatLitModule


if __name__ == "__main__":
    unittest.main()
