"""Training entrypoint.

Usage:
    python -m sehat.training.train --config configs/train/tb_baseline.yaml

Heavy dependencies (torch, Lightning, MLflow) are imported lazily inside
functions so this module — and the config schema it uses — stays importable
in minimal environments. MLflow tracking is used when the package is
installed and silently falls back to Lightning's CSV logger otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sehat.training.config import TrainConfig

__all__ = ["main", "run_training"]


def run_training(config: TrainConfig) -> str:
    """Train a model per ``config`` and return the best checkpoint path.

    Reproducibility: seeds Lightning (which covers Python/NumPy/torch) and
    enables deterministic algorithms. Artifacts: Lightning ``.ckpt`` files
    under ``config.checkpoint_dir`` (best-by-val-loss plus ``last.ckpt``).
    """
    pl = _import_lightning()
    from sehat.training.datamodule import ManifestDataModule
    from sehat.training.module import SehatLitModule

    pl.seed_everything(config.seed, workers=True)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f"{config.disease}-{config.arch}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.patience,
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        callbacks=[checkpoint, early_stopping],
        logger=_build_logger(config),
        accelerator="auto",
        devices="auto",
        deterministic=True,
    )
    trainer.fit(
        SehatLitModule(config),
        datamodule=ManifestDataModule(config),
    )
    return str(checkpoint.best_model_path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint; returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="sehat.training.train",
        description="Train a Sehat chest-X-ray screening model.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a training YAML config (see configs/train/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from the config file.",
    )
    args = parser.parse_args(argv)

    config = TrainConfig.from_yaml(args.config)
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)

    best_ckpt = run_training(config)
    print(f"Training complete. Best checkpoint: {best_ckpt}")
    return 0


def _import_lightning() -> Any:
    try:
        import lightning.pytorch as pl
    except ImportError:
        try:
            import pytorch_lightning as pl
        except ImportError as exc:
            raise ImportError(
                "Training requires PyTorch Lightning (`pip install lightning`)."
            ) from exc
    return pl


def _build_logger(config: TrainConfig) -> Any:
    """Return an MLflow logger when mlflow is installed, else a CSV logger."""
    pl = _import_lightning()
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return pl.loggers.CSVLogger(save_dir=config.checkpoint_dir, name="csv_logs")
    kwargs: dict[str, Any] = {"experiment_name": config.mlflow_experiment}
    if config.mlflow_tracking_uri:
        kwargs["tracking_uri"] = config.mlflow_tracking_uri
    return pl.loggers.MLFlowLogger(**kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
