# Roadmap — Project Sehat

Project Sehat is developed in milestones. Each milestone has explicit exit
criteria ("gates"), and — true to the project's values — the gates are about
**evidence**, not features. A milestone is done when its gate passes, not when
its code lands.

Status legend: ✅ done · 🚧 in progress · ⬜ not started

> This roadmap describes direction, not deadlines. Dates are omitted
> deliberately: the project ships when the evaluation gates pass.

---

## M1 — Data + training baseline ⬜

**Goal:** a fully reproducible TB screening baseline, end to end.

- Versioned data manifests for Shenzhen + Montgomery TB (~800 films), NIH
  ChestX-ray14, and RSNA Pneumonia, with patient-level splits and one TB site
  held out for external validation (`src/sehat/data/`).
- Lightning training pipeline driven by checked-in configs, with MLflow
  tracking (`python -m sehat.training.train --config configs/train/tb_baseline.yaml`).
- Every run reproducible from config + manifest hash + git commit.

**Exit gate:** two independent runs of the baseline config produce the same
test metrics within bootstrap CI; the run is traceable end to end (see the
traceability chain in [architecture.md](architecture.md)).

## M2 — Evaluation gates ⬜

**Goal:** evaluation becomes a gate that models must pass to be released.

- Metrics suite: AUROC, sensitivity @ 95% specificity, ECE, and subgroup
  fairness over sex / age / site with bootstrap confidence intervals —
  computed on the internal test set **and** the site-held-out external set.
- `eval_report.json` / `eval_report.md` produced for every run
  (`src/sehat/eval/`).
- `sehat.eval.model_card` renders [model_card_template.md](model_card_template.md)
  from the report; every released model ships a filled card.

**Exit gate:** a model that regresses on external-site AUROC, calibration
(ECE), or any measured subgroup beyond its bootstrap CI is automatically
blocked from export. No exceptions without a written justification in the
model card.

## M3 — Edge export ⬜

**Goal:** models run at interactive speed on CPU-only clinic hardware.

- ONNX export with INT8 quantization (`src/sehat/export/`).
- Quantized model re-validated against the full M2 gate — quantization is not
  allowed to silently move metrics.
- Latency benchmark (`python -m sehat.export.benchmark <model.onnx>`)
  reporting p50 / p95 on reference commodity hardware.

**Exit gate:** INT8 artifact passes the M2 gate **and** meets the latency
budget of the clinic PWA (interactive scoring on a mid-range CPU, reported as
p50/p95 in the model card).

## M4 — Clinic PWA ⬜

**Goal:** a usable, offline-capable screening workflow for real clinics.

- FastAPI serving (`python -m sehat.serving`, configured via `SEHAT_MODEL_PATH`
  and `SEHAT_PORT`, default 8000) with `/healthz`, `/metadata`, and `/predict`.
- Progressive web app that caches its shell for offline use, talks to the local
  serving instance, and keeps all patient data on-device (see
  [ethics.md](ethics.md)).
- Copy and UX reviewed so outputs read as decision support, never diagnosis;
  `/metadata` surfaced in-app so users can always see which model they are
  running and when it was evaluated.

**Exit gate:** full workflow — image in, calibrated score shown — works with
the network cable unplugged; a usability pass with at least one target-setting
health worker is documented.

## M5 — Additional diseases ⬜

**Goal:** extend screening beyond TB and pneumonia, carefully.

- Candidate findings prioritized by (a) burden in low-resource settings,
  (b) availability of public labeled data with usable terms, and (c) evidence
  that a single-film classifier can be reliable enough for triage.
- Each new finding gets its own data-card section, evaluation gate, and model
  card. Nothing ships by piggybacking on the TB evidence base.

**Exit gate:** each added finding meets the same M2/M3 gates as the TB
baseline, on its own site-held-out evaluation.

---

## How to influence the roadmap

- Open an issue describing the use case (deployment setting, disease, hardware
  constraints). Concrete field needs outrank abstract improvements.
- Contributions aligned with the current milestone are the fastest to merge —
  see [CONTRIBUTING.md](../CONTRIBUTING.md).
- Suggesting a dataset? Check [data_card.md](data_card.md) first: public,
  de-identified, research-usable terms, and metadata sufficient for subgroup
  evaluation are hard requirements.
