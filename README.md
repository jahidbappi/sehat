# Sehat

> AI disease screening for the four billion people a radiologist never visits.

[![CI](https://github.com/jahidbappi/sehat/actions/workflows/ci.yml/badge.svg)](https://github.com/jahidbappi/sehat/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Sehat** (সেহাত / صحت / सेहत — "health") is an open-source platform that lets a rural
clinic with a cheap laptop and no internet connection screen chest X-rays for
**tuberculosis** and **pneumonia** in seconds. It is decision support — it flags
suspected cases for human confirmation. It never claims to diagnose.

## The problem

Tuberculosis kills roughly 1.3 million people every year. It is curable.
The bottleneck is not treatment — it is diagnosis. Chest X-ray screening requires
a radiologist, and most of the world does not have one. Pneumonia is the leading
infectious killer of children under five. Same bottleneck.

The science to help already exists. The public datasets exist. What does not exist
is an open, production-grade, offline-first system that a clinic can actually run.
That is the gap Sehat fills.

## What Sehat does

- **Screens** chest X-rays for TB and pneumonia with calibrated probability scores,
  not just binary guesses.
- **Runs offline** on low-end hardware via quantized ONNX/TFLite exports.
- **Proves it generalizes**: models are evaluated on hospitals they never trained on
  (site-held-out validation) and across sex/age subgroups — because medical ML that
  only works in the lab helps no one.
- **Is honest about its limits**: every release ships with a model card, a data card,
  and subgroup performance reports.

## Architecture

```
public datasets ──► unified manifest ──► training ──► evaluation ──► model registry
 (NIH, NLM TB)      (versioned, DVC)     (Lightning)  (fairness,        (MLflow)
                                                         calibration)
                                                                │
                          ┌─────────────────────────────────────┤
                          ▼                                     ▼
                  edge export (ONNX/TFLite)           cloud serving (FastAPI)
                          │                                     │
                  offline clinic PWA                    NGO / hospital API
```

## Quickstart

```bash
# Install (data-layer tooling)
pip install -e ".[dev,data]"

# Download a public TB dataset (resumable, checksummed)
sehat-data download --dataset shenzhen --data-dir data

# Build the unified manifest with leakage-safe splits
sehat-data build-manifest --data-dir data --out data/manifest.csv

# Validate any manifest against the schema
sehat-data verify --manifest data/manifest.csv
```

Training, evaluation, edge export, and serving live in the companion modules
`sehat.training`, `sehat.eval`, `sehat.export`, and `sehat.serving` — install the
`train` / `serve` / `export` extras and see each module's documentation.

## Web app (`app/`)

A static, buildless offline-first PWA for clinics — no npm, no build step:

```bash
# Export a quantized model for the app (after training)
python -m sehat.export.onnx_export --checkpoint artifacts/best.ckpt --out app/model/model.int8.onnx

# Serve the app (service worker requires localhost or https)
cd app && python3 -m http.server 8080
```

The app runs the INT8 ONNX model **in the browser** via onnxruntime-web
(WebAssembly) — X-rays never leave the device. A remote mode can instead call a
running `sehat.serving` API (`POST /predict`). See `app/README.md` for details.
Deployable as-is on Vercel (static, no build) — see `app/README.md`.

## Repository layout

```
src/sehat/
├── data/         # downloaders, unified manifest, leakage-safe splits, DVC hooks
├── models/       # network architectures and calibration
├── training/     # Lightning training pipeline, experiment tracking
├── eval/         # fairness, calibration, external-site validation
├── export/       # ONNX/TFLite quantization + benchmarks
└── serving/      # FastAPI inference service
configs/
├── data/         # dataset sources and split strategy
app/              # offline-first clinic PWA
docs/             # model card, data card, ethics
tests/            # unit + regression tests
```

## Ethics and safety

Sehat is **decision support, not a medical device**. It is designed to prioritize
cases for qualified health workers, never to replace them. Every model ships with
documented failure modes, subgroup performance, and a clear statement of what it
must not be used for. See `docs/` for the full ethics statement.

## Contributing

This project exists to be used where it matters most. Contributions welcome —
see `CONTRIBUTING.md`.

## License

MIT. Datasets remain the property of their publishers and are subject to
their own licenses and citation requirements (see `configs/data/datasets.yaml`).
