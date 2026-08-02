<!--
Thanks for contributing to Project Sehat. Small, single-purpose PRs merge
fastest. Delete any section that does not apply — but never delete the
checklist; tick items honestly and explain any unchecked box in the body.
-->

## Summary

<!-- What does this PR change and why? Link the issue it addresses, e.g. "Fixes #123". -->

## Changes

<!--
Bullet list of the concrete changes. Include config paths, module names,
and any changes to public interfaces (API responses, config keys, manifests).
-->

## Test plan

<!--
Commands you ran and their results, e.g. `pytest`, `pre-commit run --all-files`,
`python -m sehat.export.benchmark artifacts/model.onnx`. For metric-affecting
changes, paste the before/after from eval_report.json.
-->

## Checklist

**Quality**

- [ ] `pre-commit run --all-files` passes (ruff lint + format)
- [ ] `pytest` passes, and new behavior has tests (bug fixes include a regression test)
- [ ] Public functions in `src/sehat/` have type hints and docstrings
- [ ] Docs updated for any changed behavior (README, `docs/`, docstrings)

**Model & metrics** *(required if this PR touches training, data, thresholds, export, or evaluation — otherwise tick "N/A")*

- [ ] N/A — this PR cannot change any metric or model artifact
- [ ] `eval_report.json` / `eval_report.md` regenerated; the delta is described above
- [ ] **Model card updated if metrics change** (rendered via `sehat.eval.model_card` from `docs/model_card_template.md`)
- [ ] Subgroup fairness metrics (sex / age / site) reviewed; any movement outside bootstrap CIs is explained
- [ ] External-site results are reported separately (not pooled with internal results)
- [ ] INT8/ONNX artifact re-validated against the eval gate if the export path changed

**Ethics & safety**

- [ ] No PHI, clinic patient data, or non-public datasets introduced
- [ ] User-facing copy remains decision-support-only (no diagnostic claims); see `docs/ethics.md`
- [ ] New datasets come with `docs/data_card.md` provenance, licensing notes, and bias disclosure
