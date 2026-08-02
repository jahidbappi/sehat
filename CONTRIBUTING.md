# Contributing to Project Sehat

Thank you for your interest in contributing. Project Sehat builds
offline-capable clinical **decision-support** screening tools for tuberculosis
and pneumonia, aimed at low-resource clinics. Because the software touches
health care, we hold contributions to a high bar for correctness, honesty about
limitations, and reproducibility — and a low bar for process. No CLA, no DCO:
by submitting a pull request you agree that your contribution is licensed under
the project's [MIT License](LICENSE).

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

---

## Development setup

The project uses a src-layout Python package (`sehat`) defined in
`pyproject.toml`. Python >= 3.10 is required.

### Option A: uv (recommended)

```bash
git clone https://github.com/<org>/sehat.git
cd sehat
uv sync --all-extras
uv run pre-commit install
```

### Option B: pip

```bash
git clone https://github.com/<org>/sehat.git
cd sehat
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Pre-commit runs on every commit and enforces the formatting and lint rules
below. Run it against the whole tree any time with:

```bash
pre-commit run --all-files
```

## Code style

- **[Ruff](https://docs.astral.sh/ruff/)** handles both linting and formatting.
  Configuration lives in `pyproject.toml`; do not add per-file ignore comments
  without a written justification in the PR.
- Type hints are required on all public functions in `src/sehat/`.
- Docstrings: one-line summary for simple functions, full argument/return
  documentation for anything that touches data, models, or metrics.
- No dead code, no commented-out blocks, no `print()` debugging left behind —
  use the `logging` module.

## Running tests

```bash
pytest                  # full suite
pytest tests/ -k data   # subset by name
pytest --cov=sehat      # with coverage
```

All new behavior needs tests. Bug fixes should land with a regression test that
fails on the unfixed code.

## Running the pieces locally

```bash
# Train the TB baseline (requires prepared data manifests; see docs/data_card.md)
python -m sehat.training.train --config configs/train/tb_baseline.yaml

# Evaluate (writes eval_report.json and eval_report.md)
python -m sehat.eval.evaluate --run <mlflow_run_id>

# Serve the model
SEHAT_MODEL_PATH=artifacts/model.onnx python -m sehat.serving
# -> http://localhost:$SEHAT_PORT/healthz (SEHAT_PORT defaults to 8000)

# Benchmark an exported ONNX model
python -m sehat.export.benchmark artifacts/model.onnx
```

## Pull request process

1. **Open an issue first** for anything non-trivial (new datasets, new metrics,
   model changes). Small fixes (typos, docs, obvious bugs) can go straight to a
   PR.
2. **Fork and branch.** Branch names: `feat/<slug>`, `fix/<slug>`, or
   `docs/<slug>`.
3. **Keep PRs small and single-purpose.** A PR that changes data handling
   should not also refactor the training loop.
4. **Fill out the PR template checklist.** It includes documentation and
   model-card gates — they are not optional when they apply.
5. **CI must be green**: pre-commit (ruff), the test suite, and any packaging
   checks.
6. **One approval from a maintainer** is required to merge. Maintainers may
   request changes; please respond to every comment, even if only to explain
   why no change is needed.

### Special rules for model- or metric-affecting changes

If your change alters training data, model architecture, thresholds, or any
reported metric, the PR **must** also:

- regenerate `eval_report.json` / `eval_report.md` and describe the delta in
  the PR body,
- update the rendered model card (see `docs/model_card_template.md` and
  `sehat.eval.model_card`),
- state explicitly whether subgroup fairness metrics (sex / age / site) moved
  outside their bootstrap confidence intervals.

Changes that worsen calibration (ECE) or external-site performance without a
documented, justified reason will not be merged.

## What to work on

- Issues labeled [`good first issue`] are scoped for newcomers.
- The [roadmap](docs/roadmap.md) shows where the project is heading; aligning a
  contribution with the current milestone is the fastest path to merge.
- Documentation improvements are always welcome and are reviewed with the same
  care as code — this project is read by clinicians as well as engineers.

## A note on scope

Project Sehat produces **decision-support software, not a medical device** (see
[docs/ethics.md](docs/ethics.md)). Contributions that blur that line — for
example, UI copy that reads as a diagnosis — will be asked to change.

[`good first issue`]: https://github.com/<org>/sehat/labels/good%20first%20issue
