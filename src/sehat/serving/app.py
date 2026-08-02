"""FastAPI inference service for Project Sehat chest-X-ray screening.

This module is only imported where the serving dependencies (fastapi,
uvicorn, pillow, numpy, and onnxruntime or torch) are installed — e.g. the
Docker image. Pure schema/config logic lives in :mod:`sehat.serving.schemas`
and :mod:`sehat.serving.config`, which stay stdlib-only.

Run locally::

    SEHAT_MODEL_PATH=artifacts/tb.onnx python -m sehat.serving
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile

from sehat.serving.config import ServeConfig, load_serve_config
from sehat.serving.schemas import (
    DISCLAIMER,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
)

logger = logging.getLogger("sehat.serving")

#: Suffixes routed to the ONNX Runtime engine; anything else is treated as a
#: torch checkpoint and served through the torch fallback engine.
ONNX_SUFFIXES = (".onnx",)


class _Engine(Protocol):
    """Minimal inference-engine interface used by the predict endpoint."""

    name: str

    def predict_logit(self, batch: Any) -> float:
        """Return the raw (pre-sigmoid) logit for a single preprocessed image."""
        ...


class _OnnxEngine:
    """ONNX Runtime CPU engine — the production path (<2s/image target)."""

    name = "onnxruntime"

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict_logit(self, batch: Any) -> float:
        outputs = self._session.run(None, {self._input_name: batch})
        return _first_scalar(outputs[0])


class _TorchEngine:
    """Torch fallback engine for ``*.ckpt``/``*.pt`` checkpoints.

    Loads weights through the training contract
    (:func:`sehat.training.checkpoint.load_backbone_from_ckpt`) and falls
    back to :func:`sehat.models.factory.build_model` plus a raw
    ``torch.load`` state-dict when the checkpoint helper is unavailable.
    """

    name = "torch"

    def __init__(self, model_path: Path, arch: str) -> None:
        import torch

        self._torch = torch
        self._model = self._load_model(model_path, arch)
        self._model.eval()

    def _load_model(self, model_path: Path, arch: str) -> Any:
        from sehat.models.factory import build_model

        try:
            from sehat.training.checkpoint import load_backbone_from_ckpt
        except ImportError:
            load_backbone_from_ckpt = None

        if load_backbone_from_ckpt is not None:
            model = load_backbone_from_ckpt(str(model_path))
            if model is not None:
                return model
            logger.warning(
                "load_backbone_from_ckpt returned None for %s; falling back to "
                "build_model(%r) + torch.load",
                model_path,
                arch,
            )

        model = build_model(arch, num_classes=1, pretrained=False)
        state = self._torch.load(str(model_path), map_location="cpu")
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        model.load_state_dict(state)
        return model

    def predict_logit(self, batch: Any) -> float:
        tensor = self._torch.from_numpy(batch)
        with self._torch.no_grad():
            output = self._model(tensor)
        return _first_scalar(output)


def _first_scalar(output: Any) -> float:
    """Reduce a (1, 1)/(1,)-shaped model output to a Python float."""
    if hasattr(output, "detach"):  # torch tensor
        output = output.detach().cpu().numpy()
    if hasattr(output, "reshape"):  # numpy array
        return float(output.reshape(-1)[0])
    return float(output)


def _sigmoid(logit: float) -> float:
    """Numerically stable logistic sigmoid."""
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


def _preprocess(image_bytes: bytes, cfg: ServeConfig) -> Any:
    """Decode and normalize an uploaded X-ray into an NCHW float32 batch.

    Mirrors the training-time validation transforms: RGB conversion, resize
    to ``image_size`` (bilinear), scale to [0, 1], then standardize with the
    configured channel mean/std (ImageNet statistics by default).
    """
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB").resize((cfg.image_size, cfg.image_size), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.asarray(cfg.mean, dtype=np.float32)
    std = np.asarray(cfg.std, dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(arr, dtype=np.float32)


def create_app(model_path: str | None = None) -> FastAPI:
    """Build the Sehat inference application.

    Args:
        model_path: Optional path to an ``.onnx`` artifact (served via ONNX
            Runtime) or a torch checkpoint (served via the torch fallback).
            Overrides ``SEHAT_MODEL_PATH`` and ``model.path`` from the YAML
            config.

    Returns:
        A configured :class:`fastapi.FastAPI` instance exposing
        ``GET /healthz``, ``GET /metadata`` and ``POST /predict``.
    """
    cfg = load_serve_config(model_path_override=model_path)
    app = FastAPI(
        title="Sehat — TB & Pneumonia X-ray Screening",
        version=cfg.model_version,
        description=(
            f"Offline-capable clinical decision-support API for low-resource clinics. {DISCLAIMER}"
        ),
    )

    state: dict[str, Any] = {"engine": None, "lock": threading.Lock()}

    def get_engine() -> _Engine:
        if state["engine"] is None:
            with state["lock"]:
                if state["engine"] is None:
                    state["engine"] = _build_engine(cfg)
        return state["engine"]

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return HealthResponse().to_dict()

    @app.get("/metadata", tags=["model"])
    def metadata() -> dict[str, Any]:
        return MetadataResponse(model_version=cfg.model_version, labels=cfg.labels).to_dict()

    @app.post("/predict", tags=["model"])
    async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
        started = time.perf_counter()
        engine = get_engine()  # raises 503 when no model is configured

        max_bytes = cfg.max_upload_mb * 1024 * 1024
        payload = await file.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds the {cfg.max_upload_mb} MB limit",
            )
        if not payload:
            raise HTTPException(status_code=400, detail="empty upload")

        try:
            batch = _preprocess(payload, cfg)
        except Exception as exc:  # PIL raises many error types for bad images
            raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

        try:
            logit = engine.predict_logit(batch)
        except Exception as exc:
            logger.exception("inference failed")
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

        probability = _sigmoid(logit)
        label = cfg.labels[1] if probability >= cfg.threshold else cfg.labels[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return PredictionResponse(
            probability=round(probability, 6),
            label=label,
            threshold=cfg.threshold,
            latency_ms=round(latency_ms, 3),
        ).to_dict()

    return app


def _build_engine(cfg: ServeConfig) -> _Engine:
    """Instantiate the inference engine for the configured model artifact."""
    if not cfg.model_path:
        raise HTTPException(
            status_code=503,
            detail=("no model configured; set SEHAT_MODEL_PATH or pass create_app(model_path=...)"),
        )
    path = Path(cfg.model_path)
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"model not found: {path}")
    if path.suffix.lower() in ONNX_SUFFIXES:
        logger.info("serving %s via onnxruntime", path)
        return _OnnxEngine(path)
    logger.info("serving %s via torch fallback (arch=%s)", path, cfg.arch)
    return _TorchEngine(path, cfg.arch)
