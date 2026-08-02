"""Serving configuration loading (stdlib-only).

The canonical config lives at ``configs/serve/default.yaml``. PyYAML is used
when installed; otherwise a deliberately small YAML-subset parser
(:func:`parse_simple_yaml`) handles the flat/nested-mapping structure used by
that file. This keeps config loading importable and testable with nothing
beyond the standard library.

Resolution order for the config path:

1. explicit ``path`` argument
2. ``SEHAT_CONFIG`` environment variable
3. ``configs/serve/default.yaml`` relative to the current working directory
4. ``configs/serve/default.yaml`` relative to the repository root (src layout)
5. built-in defaults (no file read)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Environment variable holding the path to the model artifact to serve.
ENV_MODEL_PATH = "SEHAT_MODEL_PATH"
#: Environment variable overriding the decision threshold.
ENV_THRESHOLD = "SEHAT_THRESHOLD"
#: Environment variable pointing at an alternate YAML config file.
ENV_CONFIG = "SEHAT_CONFIG"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "serve" / "default.yaml"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ConfigError(ValueError):
    """Raised when the serving configuration is missing or invalid."""


class YamlSubsetError(ConfigError):
    """Raised when the fallback parser meets YAML outside its subset."""


@dataclass(frozen=True, slots=True)
class ServeConfig:
    """Typed view over ``configs/serve/default.yaml``.

    Attributes mirror the YAML sections: ``service``/``model`` describe the
    deployment, ``preprocessing`` must match the training-time validation
    transforms, ``inference`` holds the decision threshold, and ``export`` /
    ``benchmark`` pin the ONNX contract shared with :mod:`sehat.export`.
    """

    service_name: str = "sehat-serving"
    model_path: str | None = None
    model_version: str = "0.1.0"
    arch: str = "resnet18"
    labels: tuple[str, ...] = ("negative", "positive")
    image_size: int = 224
    mean: tuple[float, float, float] = _IMAGENET_MEAN
    std: tuple[float, float, float] = _IMAGENET_STD
    threshold: float = 0.5
    opset: int = 17
    dynamic_batch: bool = True
    benchmark_runs: int = 50
    benchmark_warmup: int = 10
    max_upload_mb: int = 15

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ConfigError(f"threshold must lie in (0, 1), got {self.threshold!r}")
        if self.image_size <= 0:
            raise ConfigError(f"image_size must be positive, got {self.image_size!r}")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ConfigError("mean/std must each have exactly 3 channel values")
        if any(s <= 0 for s in self.std):
            raise ConfigError("std values must be strictly positive")
        if self.opset <= 0:
            raise ConfigError(f"opset must be positive, got {self.opset!r}")
        if self.benchmark_runs <= 0 or self.benchmark_warmup < 0:
            raise ConfigError("benchmark_runs must be > 0 and benchmark_warmup >= 0")
        if self.max_upload_mb <= 0:
            raise ConfigError(f"max_upload_mb must be positive, got {self.max_upload_mb!r}")
        if not self.labels:
            raise ConfigError("labels must be non-empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ServeConfig:
        """Build a config from the parsed YAML mapping.

        Unknown sections/keys are ignored so the file can grow without
        breaking older readers; missing keys fall back to dataclass defaults.
        """
        if not isinstance(data, Mapping):
            raise ConfigError("top-level YAML document must be a mapping")

        def section(name: str) -> Mapping[str, Any]:
            value = data.get(name, {})
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ConfigError(f"section {name!r} must be a mapping")
            return value

        service, model = section("service"), section("model")
        preproc, infer = section("preprocessing"), section("inference")
        export, bench, limits = section("export"), section("benchmark"), section("limits")

        def vec3(raw: Any, name: str) -> tuple[float, float, float]:
            try:
                values = tuple(float(v) for v in raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{name} must be a list of 3 numbers") from exc
            if len(values) != 3:
                raise ConfigError(f"{name} must have exactly 3 values")
            return values  # type: ignore[return-value]

        labels_raw = model.get("labels", ("negative", "positive"))
        if not isinstance(labels_raw, (list, tuple)):
            raise ConfigError("model.labels must be a list of strings")

        return cls(
            service_name=str(service.get("name", "sehat-serving")),
            model_path=model.get("path"),
            model_version=str(model.get("version", "0.1.0")),
            arch=str(model.get("arch", "resnet18")),
            labels=tuple(str(label) for label in labels_raw),
            image_size=int(preproc.get("image_size", 224)),
            mean=vec3(preproc.get("mean", _IMAGENET_MEAN), "preprocessing.mean"),
            std=vec3(preproc.get("std", _IMAGENET_STD), "preprocessing.std"),
            threshold=float(infer.get("threshold", 0.5)),
            opset=int(export.get("opset", 17)),
            dynamic_batch=bool(export.get("dynamic_batch", True)),
            benchmark_runs=int(bench.get("n_runs", 50)),
            benchmark_warmup=int(bench.get("warmup", 10)),
            max_upload_mb=int(limits.get("max_upload_mb", 15)),
        )


def load_serve_config(
    path: str | os.PathLike[str] | None = None,
    *,
    model_path_override: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ServeConfig:
    """Load the serving config, applying environment overrides.

    ``model_path_override`` (the ``create_app(model_path=...)`` argument)
    wins over ``SEHAT_MODEL_PATH``, which wins over ``model.path`` in YAML.
    """
    env = os.environ if env is None else env
    resolved = _resolve_config_path(path, env)
    if resolved is None:
        cfg = ServeConfig()
    else:
        text = Path(resolved).read_text(encoding="utf-8")
        cfg = ServeConfig.from_mapping(load_yaml_text(text, source=str(resolved)))

    model_path = model_path_override or env.get(ENV_MODEL_PATH) or cfg.model_path
    threshold_raw = env.get(ENV_THRESHOLD)
    if model_path == cfg.model_path and threshold_raw is None:
        return cfg
    return ServeConfig(
        **{
            **{f: getattr(cfg, f) for f in ServeConfig.__dataclass_fields__},
            "model_path": model_path,
            "threshold": float(threshold_raw) if threshold_raw is not None else cfg.threshold,
        }
    )


def _resolve_config_path(
    path: str | os.PathLike[str] | None, env: Mapping[str, str]
) -> Path | None:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    elif env.get(ENV_CONFIG):
        candidates.append(Path(env[ENV_CONFIG]))
    else:
        candidates.append(Path.cwd() / "configs" / "serve" / "default.yaml")
        candidates.append(DEFAULT_CONFIG_PATH)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if path is not None or env.get(ENV_CONFIG):
        raise ConfigError(f"config file not found: {candidates[0]}")
    return None


# ---------------------------------------------------------------------------
# Minimal YAML-subset parser (used only when PyYAML is unavailable)
# ---------------------------------------------------------------------------


def load_yaml_text(text: str, *, source: str = "<string>") -> Any:
    """Parse YAML text with PyYAML when available, else the subset parser."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return parse_simple_yaml(text, source=source)
    return yaml.safe_load(text)


def parse_simple_yaml(text: str, *, source: str = "<string>") -> dict[str, Any]:
    """Parse a small, documented subset of YAML into nested dicts.

    Supported: nested mappings by 2+ space indentation, scalars (``null``,
    booleans, ints, floats, quoted/plain strings), inline lists
    (``[a, b, c]``), and ``#`` comments. Not supported: block sequences
    (``- item``), anchors, multi-line strings, flow mappings.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise YamlSubsetError(f"{source}:{lineno}: tabs are not supported")
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- ") or stripped == "-":
            raise YamlSubsetError(
                f"{source}:{lineno}: block sequences are not supported; "
                "use inline lists like [a, b]"
            )
        key, sep, value = stripped.partition(":")
        if not sep or not key.strip():
            raise YamlSubsetError(f"{source}:{lineno}: expected 'key: value'")
        key = key.strip().strip("\"'")
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value, source=source, lineno=lineno)
    return root


def _strip_comment(line: str) -> str:
    quote: str | None = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _parse_scalar(raw: str, *, source: str, lineno: int) -> Any:
    text = raw.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("null", "~", "none", ""):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith("["):
        if not text.endswith("]"):
            raise YamlSubsetError(f"{source}:{lineno}: unterminated inline list")
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [
            _parse_scalar(part, source=source, lineno=lineno) for part in _split_inline_list(inner)
        ]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_inline_list(inner: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for ch in inner:
        if quote:
            if ch == quote:
                quote = None
            current.append(ch)
        elif ch in "\"'":
            quote = ch
            current.append(ch)
        elif ch == ",":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [part.strip() for part in parts]
