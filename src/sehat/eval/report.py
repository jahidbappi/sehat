"""Evaluation orchestration: predictions -> metrics -> report artefacts.

Public contract (relied upon by other workers):

- :func:`evaluate_predictions` computes the full metrics dictionary and,
  when ``out_dir`` is given, writes ``eval_report.json`` + ``eval_report.md``.
- ``eval_report.json`` schema::

      {
        "auroc": float,
        "average_precision": float,
        "sensitivity_at_95spec": float,
        "ece": float,
        "brier": float,
        "subgroups": {"<col>:<level>": {"auroc": float|null,
                                         "sens_at_95spec": float|null,
                                         "n": int}}
      }

  Undefined metrics (e.g. single-class subgroups) are serialised as null.
- :func:`evaluate_checkpoint` runs a saved checkpoint against a test
  manifest and produces the same artefacts plus a rendered model card.
  torch/pandas are imported lazily inside that function only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sehat.eval import fairness, model_card
from sehat.eval.calibration import reliability_diagram_data
from sehat.eval.metrics import (
    auroc,
    average_precision,
    brier_score,
    expected_calibration_error,
    sensitivity_at_specificity,
)

__all__ = ["evaluate_checkpoint", "evaluate_predictions", "load_eval_config"]

REPORT_JSON_NAME = "eval_report.json"
REPORT_MD_NAME = "eval_report.md"
MODEL_CARD_NAME = "model_card.md"

_GROUP_COLS_DEFAULT: tuple[str, ...] = ("sex", "age_band", "site")


def _json_safe(value: Any) -> Any:
    """Recursively convert NaN/inf floats to None and numpy scalars to Python."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records_from_groups(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    n = y_true.shape[0]
    records: list[dict[str, Any]] = []
    lengths = {name: len(values) for name, values in groups.items()}
    if len(set(lengths.values())) > 1 or (lengths and next(iter(lengths.values())) != n):
        raise ValueError(f"every groups column must have length {n}, got lengths {lengths}")
    for i in range(n):
        rec: dict[str, Any] = {"label": float(y_true[i]), "score": float(y_score[i])}
        for name, values in groups.items():
            rec[name] = values[i]
        records.append(rec)
    return records


def evaluate_predictions(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    groups: Mapping[str, Sequence[Any]] | None = None,
    out_dir: str | Path | None = None,
    *,
    target_specificity: float = 0.95,
    ece_bins: int = 15,
    disparity_margin: float = 0.05,
    n_boot: int = 1000,
    seed: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the full evaluation metrics dictionary from predictions.

    Args:
        y_true: binary ground-truth labels in {0, 1}.
        y_score: predicted probabilities/scores of the positive class.
        groups: optional mapping of column name -> per-case values
            (e.g. ``{"sex": [...], "age": [...], "site": [...]}``). A numeric
            ``age`` column is banded automatically for subgroup analysis.
        out_dir: when given, ``eval_report.json`` and ``eval_report.md`` (and
            a model card when ``metadata`` is provided) are written there.
        target_specificity: operating point for sensitivity (default 0.95).
        ece_bins: number of bins for the ECE / reliability diagram.
        disparity_margin: subgroup disparity flag margin.
        n_boot: bootstrap replicates for subgroup CIs (seeded).
        seed: bootstrap seed; identical seeds reproduce CIs bit-for-bit.
        metadata: extra key/value pairs embedded in the markdown report and
            forwarded to the model card.

    Returns:
        The contract metrics dictionary with keys ``auroc``,
        ``average_precision``, ``sensitivity_at_95spec``, ``ece``, ``brier``
        and ``subgroups`` (``{"<col>:<level>": {"auroc", "sens_at_95spec",
        "n"}}``). Undefined metrics are NaN in memory and null on disk.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"y_true and y_score length mismatch: {y.shape[0]} vs {s.shape[0]}")

    report: dict[str, Any] = {
        "auroc": auroc(y, s),
        "average_precision": average_precision(y, s),
        "sensitivity_at_95spec": sensitivity_at_specificity(
            y, s, target_specificity=target_specificity
        ),
        "ece": expected_calibration_error(y, s, n_bins=ece_bins),
        "brier": brier_score(y, s),
        "subgroups": {},
    }

    fairness_result: dict[str, Any] | None = None
    if groups:
        records = _records_from_groups(y, s, groups)
        group_cols = tuple(
            col for col in ("sex", "age_band", "site") if col in groups or col == "age_band"
        )
        fairness_result = fairness.subgroup_report(
            records,
            group_cols=group_cols,
            target_specificity=target_specificity,
            disparity_margin=disparity_margin,
            n_boot=n_boot,
            seed=seed,
        )
        for col, levels in fairness_result["subgroups"].items():
            for level, entry in levels.items():
                report["subgroups"][f"{col}:{level}"] = {
                    "auroc": entry["auroc"],
                    "sens_at_95spec": entry["sens_at_95spec"],
                    "n": entry["n"],
                }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / REPORT_JSON_NAME).write_text(
            json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8"
        )
        (out / REPORT_MD_NAME).write_text(
            _render_markdown_report(
                report,
                fairness_result=fairness_result,
                y=y,
                s=s,
                ece_bins=ece_bins,
                metadata=dict(metadata or {}),
            ),
            encoding="utf-8",
        )
        if metadata:
            card_metadata = dict(metadata)
            detailed_subgroups = _detailed_subgroups(fairness_result)
            card_metrics = {**report, "subgroups": detailed_subgroups}
            model_card.write_model_card(out / MODEL_CARD_NAME, card_metrics, card_metadata)
    return report


def _detailed_subgroups(
    fairness_result: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Flatten fairness output with CIs + flags for the model card."""
    detailed: dict[str, dict[str, Any]] = {}
    if not fairness_result:
        return detailed
    for col, levels in fairness_result["subgroups"].items():
        for level, entry in levels.items():
            detailed[f"{col}:{level}"] = {
                "auroc": entry["auroc"],
                "auroc_ci": entry["auroc_ci"],
                "sens_at_95spec": entry["sens_at_95spec"],
                "sens_at_95spec_ci": entry["sens_at_95spec_ci"],
                "n": entry["n"],
                "insufficient_data": entry["insufficient_data"],
                "disparity": entry["disparity"],
            }
    return detailed


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "N/A" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def _render_markdown_report(
    report: dict[str, Any],
    *,
    fairness_result: dict[str, Any] | None,
    y: np.ndarray,
    s: np.ndarray,
    ece_bins: int,
    metadata: dict[str, Any],
) -> str:
    lines = ["# Sehat Evaluation Report", ""]
    if metadata:
        lines.append("## Run metadata")
        for key, value in metadata.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
    lines += [
        f"- **Cases evaluated:** {y.shape[0]} (positives: {int(y.sum())})",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| AUROC | {_fmt(report['auroc'])} |",
        f"| Average precision | {_fmt(report['average_precision'])} |",
        f"| Sensitivity @ 95% specificity | {_fmt(report['sensitivity_at_95spec'])} |",
        f"| Expected calibration error ({ece_bins} bins) | {_fmt(report['ece'])} |",
        f"| Brier score | {_fmt(report['brier'])} |",
        "",
    ]

    if fairness_result is not None:
        lines += ["## Subgroup fairness", ""]
        pooled = fairness_result["pooled"]
        lines.append(
            f"Pooled AUROC {_fmt(pooled['auroc'])}; pooled sensitivity @ 95% "
            f"specificity {_fmt(pooled['sens_at_95spec'])}; disparity margin "
            f"{fairness_result['config']['disparity_margin']:.2f} "
            f"(bootstrap n={fairness_result['config']['n_boot']}, "
            f"seed={fairness_result['config']['seed']})."
        )
        lines.append("")
        lines.append(model_card.subgroup_markdown_table(_detailed_subgroups(fairness_result)))
        flagged = [
            name
            for name, entry in _detailed_subgroups(fairness_result).items()
            if any(entry["disparity"].values())
        ]
        lines.append("")
        if flagged:
            lines.append("**Disparity flags raised for:** " + ", ".join(f"`{f}`" for f in flagged))
        else:
            lines.append("No subgroup disparity flags raised at the configured margin.")
        lines.append("")

    reliability = reliability_diagram_data(y, s, n_bins=ece_bins)
    lines += [
        "## Calibration (reliability diagram data)",
        "",
        "| Bin | Count | Mean confidence | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for b in reliability["bins"]:
        if b["count"] == 0:
            continue
        lines.append(
            f"| [{b['lower']:.2f}, {b['upper']:.2f}] | {b['count']} | "
            f"{b['mean_confidence']:.4f} | {b['accuracy']:.4f} |"
        )
    lines += ["", "---", "*Generated by `sehat.eval.report`.*", ""]
    return "\n".join(lines)


def load_eval_config(path: str | Path) -> dict[str, Any]:
    """Load an eval YAML config.

    Uses PyYAML when installed; otherwise falls back to a small parser for
    the flat/nested-by-two-space-indent subset used by ``configs/eval/*.yaml``
    (scalars, inline ``[a, b]`` lists, comments, booleans/numbers).
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return dict(data or {})
    except ImportError:
        return _minimal_yaml_load(text)


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"unsupported config line: {raw!r}")
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_parse_scalar(v.strip()) for v in inner.split(",")] if inner else []
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def evaluate_checkpoint(
    ckpt_path: str | Path,
    manifest_path: str | Path,
    out_dir: str | Path = "eval_outputs",
    *,
    config: Mapping[str, Any] | str | Path | None = None,
    disease: str | None = None,
    split: str = "test",
    arch: str | None = None,
    num_classes: int = 1,
    pretrained: bool = False,
    batch_size: int = 32,
    device: str | None = None,
    seed: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint on a manifest split and write all artefacts.

    Heavy dependencies (torch, pandas, the models/training packages owned by
    other workers) are imported lazily so that importing ``sehat.eval`` never
    requires them.

    Args:
        ckpt_path: path to a training checkpoint, loadable via
            ``sehat.training.checkpoint.load_backbone_from_ckpt``.
        manifest_path: CSV/parquet manifest with columns ``image_path,
            label, disease, site, patient_id, split, sex, age``.
        out_dir: directory for ``eval_report.json`` / ``eval_report.md`` /
            ``model_card.md``.
        config: optional eval config (mapping or path to a YAML file such as
            ``configs/eval/tb_eval.yaml``). Keys used: ``disease``, ``split``,
            ``arch``, ``num_classes``, ``batch_size``, ``target_specificity``,
            ``ece_bins``, ``disparity_margin``, ``n_boot``, ``seed``.
        disease: override disease filter (else config, else all rows).
        split: manifest split to evaluate (default ``"test"``).
        arch: model architecture for ``build_model`` (else config).
        num_classes: classifier head size for ``build_model``.
        pretrained: forwarded to ``build_model``.
        batch_size: inference batch size.
        device: torch device string; auto-detected when None.
        seed: bootstrap seed for subgroup CIs.
        metadata: extra metadata forwarded to report + model card.

    Returns:
        The contract metrics dictionary (see :func:`evaluate_predictions`).
    """
    cfg: dict[str, Any] = {}
    if isinstance(config, (str, Path)):
        cfg = load_eval_config(config)
    elif isinstance(config, Mapping):
        cfg = dict(config)

    disease = disease or cfg.get("disease")
    split = str(cfg.get("split", split))
    arch = arch or cfg.get("arch")
    num_classes = int(cfg.get("num_classes", num_classes))
    batch_size = int(cfg.get("batch_size", batch_size))
    seed = int(cfg.get("seed", seed))

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("evaluate_checkpoint requires pandas to read the manifest") from exc
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("evaluate_checkpoint requires torch for inference") from exc
    try:
        from sehat.models.factory import build_model
        from sehat.training.checkpoint import load_backbone_from_ckpt
    except ImportError as exc:  # pragma: no cover - depends on other workers
        raise ImportError(
            "evaluate_checkpoint requires sehat.models.factory.build_model and "
            "sehat.training.checkpoint.load_backbone_from_ckpt (Workers 1/2)"
        ) from exc

    manifest_path = Path(manifest_path)
    if manifest_path.suffix in (".parquet", ".pq"):
        manifest = pd.read_parquet(manifest_path)
    else:
        manifest = pd.read_csv(manifest_path)
    frame = manifest[manifest["split"] == split]
    if disease is not None and "disease" in frame.columns:
        frame = frame[frame["disease"] == disease]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"manifest {manifest_path} has no rows for split={split!r}, disease={disease!r}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch=arch, num_classes=num_classes, pretrained=pretrained)
    model = load_backbone_from_ckpt(str(ckpt_path), model=model)
    model.to(device).eval()

    scores = _infer_scores(model, frame, batch_size=batch_size, device=device, torch=torch)

    y_true = frame["label"].to_numpy(dtype=np.float64)
    groups: dict[str, Any] = {}
    for col in ("sex", "age", "site"):
        if col in frame.columns:
            groups[col] = frame[col].tolist()

    run_metadata: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "manifest": str(manifest_path),
        "disease": disease or "all",
        "split": split,
        "arch": arch or "N/A",
        "n_eval": int(frame.shape[0]),
        "device": device,
    }
    run_metadata.update(dict(metadata or {}))

    return evaluate_predictions(
        y_true,
        scores,
        groups=groups or None,
        out_dir=out_dir,
        target_specificity=float(cfg.get("target_specificity", 0.95)),
        ece_bins=int(cfg.get("ece_bins", 15)),
        disparity_margin=float(cfg.get("disparity_margin", 0.05)),
        n_boot=int(cfg.get("n_boot", 1000)),
        seed=seed,
        metadata=run_metadata,
    )


def _infer_scores(
    model: Any, frame: Any, *, batch_size: int, device: str, torch: Any
) -> np.ndarray:
    """Run inference over ``frame['image_path']`` and return positive-class scores.

    Uses the project dataset pipeline when available (``sehat.data``); falls
    back to a standard RGB resize/center-crop/ImageNet-normalise transform so
    the evaluator works against any checkpoint.
    """
    transform = _default_eval_transform(torch)
    paths = frame["image_path"].tolist()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors = [_load_image(p, transform) for p in batch_paths]
            batch = torch.stack(tensors).to(device)
            logits = model(batch)
            if logits.ndim == 2 and logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)[:, -1]
            else:
                probs = torch.sigmoid(logits.reshape(-1))
            scores.extend(probs.detach().cpu().numpy().astype(np.float64).tolist())
    return np.asarray(scores, dtype=np.float64)


def _default_eval_transform(torch: Any) -> Any:
    try:
        from sehat.data.transforms import eval_transform  # type: ignore

        return eval_transform()
    except Exception:
        pass
    try:
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "no eval transform available: sehat.data.transforms not found and "
            "torchvision is not installed"
        ) from exc


def _load_image(path: str, transform: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Pillow is required to load images for inference") from exc
    with Image.open(path) as img:
        return transform(img.convert("RGB"))
