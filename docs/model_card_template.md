# Model Card — <model_name>

<!--
  TEMPLATE — rendered by `sehat.eval.model_card`.

  Placeholders use the form <snake_case_token>. The renderer substitutes them
  from eval_report.json and the MLflow run record. Any token left in angle
  brackets after rendering means the value was unavailable and MUST be filled
  in by hand or the card must not be published. Do not delete sections; mark
  genuinely inapplicable ones "N/A" with a one-line reason.

  Style rules for authors:
  - Report numbers with their bootstrap 95% confidence intervals, always.
  - Never pool internal and external-site results into a single number.
  - If a metric regressed relative to the previous released model, say so
    explicitly in "Limitations".
-->

**Model version:** <model_version>
**Release date:** <release_date>
**MLflow run ID:** <mlflow_run_id>
**Training manifest hash:** <manifest_hash>
**Contact:** Project Sehat Contributors (via GitHub issues)

## Model details

- **Architecture:** <architecture> (e.g. ResNet-family classifier, single frontal chest X-ray input)
- **Task:** binary screening score per finding — TB and/or pneumonia
- **Input:** frontal (PA or AP) chest radiograph, preprocessed per
  [data_card.md](data_card.md) (grayscale, resized, normalized)
- **Output:** a calibrated risk score in [0, 1] per finding, plus metadata;
  **not** a diagnosis
- **Export artifact:** ONNX, INT8-quantized (`<onnx_artifact_name>`)
- **License:** MIT (model weights and code); see dataset terms below for the
  data the model was trained on

## Intended use

- **Primary use:** decision-support **screening/triage aid** for TB and
  pneumonia on chest X-rays in low-resource clinics, operated by trained
  healthcare workers as one input among many (symptoms, history, sputum tests,
  clinical judgement).
- **Primary users:** clinicians and trained health workers in settings where
  radiologist access is limited or absent.
- **Deployment context:** offline-capable clinic PWA backed by the local
  FastAPI service (`python -m sehat.serving`); no internet required at
  inference time.

## Out-of-scope uses

This model must **not** be used:

- as a standalone diagnostic, or to rule disease **in or out** without
  confirmatory testing (e.g. sputum NAAT/culture for TB);
- on populations or image types not represented in the evaluation data:
  pediatric patients (<age_floor> years), lateral-view films, CT scans, or
  smartphone photographs of films;
- for any finding other than the one(s) listed under "Task" (e.g. it does not
  detect COVID-19, malignancy, or fractures);
- as a basis for withholding care — a low score must never delay a clinically
  indicated workup;
- in any workflow where its output is shown to patients as a definitive result.

## Training data

- **Sources:** public datasets only — Shenzhen and Montgomery TB collections
  (~800 films combined), NIH ChestX-ray14, RSNA Pneumonia Detection Challenge.
  Full provenance, licensing, and known biases: [data_card.md](data_card.md).
- **Manifest:** versioned manifest `<manifest_id>` (hash `<manifest_hash>`),
  patient-level train/validation/test split; site `<external_site_name>` held
  out entirely for external validation.
- **No PHI:** all training data is from public, de-identified research
  collections; no clinic patient data is used in training.
- **Preprocessing:** see [data_card.md](data_card.md#preprocessing).

## Evaluation results

Evaluation follows the gate defined in [architecture.md](architecture.md):
internal test split **and** a site-held-out external set, with bootstrap 95%
CIs. Numbers below are rendered from `eval_report.json`
(run `<mlflow_run_id>`).

### Discrimination

| Metric | Internal test | External site (<external_site_name>) |
| --- | --- | --- |
| AUROC (TB) | <auroc_tb_internal> (<auroc_tb_internal_ci>) | <auroc_tb_external_site> (<auroc_tb_external_site_ci>) |
| Sensitivity @ 95% specificity (TB) | <sens_at_95spec_tb_internal> (<sens_at_95spec_tb_internal_ci>) | <sens_at_95spec_tb_external_site> (<sens_at_95spec_tb_external_site_ci>) |
| AUROC (pneumonia) | <auroc_pna_internal> (<auroc_pna_internal_ci>) | <auroc_pna_external_site> (<auroc_pna_external_site_ci>) |
| Sensitivity @ 95% specificity (pneumonia) | <sens_at_95spec_pna_internal> (<sens_at_95spec_pna_internal_ci>) | <sens_at_95spec_pna_external_site> (<sens_at_95spec_pna_external_site_ci>) |

### Calibration

| Metric | Internal test | External site |
| --- | --- | --- |
| ECE (TB) | <ece_tb_internal> | <ece_tb_external_site> |
| ECE (pneumonia) | <ece_pna_internal> | <ece_pna_external_site> |

Calibration method: <calibration_method>, fitted on the validation split only.
Reliability diagrams: <reliability_diagram_paths>.

### Subgroup performance (fairness)

AUROC with bootstrap 95% CIs, per subgroup. Differences whose CIs exclude the
overall estimate are flagged in "Limitations".

| Subgroup | n | TB AUROC (95% CI) | Pneumonia AUROC (95% CI) |
| --- | --- | --- | --- |
| Sex: female | <n_sex_female> | <auroc_tb_sex_female> (<auroc_tb_sex_female_ci>) | <auroc_pna_sex_female> (<auroc_pna_sex_female_ci>) |
| Sex: male | <n_sex_male> | <auroc_tb_sex_male> (<auroc_tb_sex_male_ci>) | <auroc_pna_sex_male> (<auroc_pna_sex_male_ci>) |
| Age: <age_bin_1_label> | <n_age_bin_1> | <auroc_tb_age_bin_1> (<auroc_tb_age_bin_1_ci>) | <auroc_pna_age_bin_1> (<auroc_pna_age_bin_1_ci>) |
| Age: <age_bin_2_label> | <n_age_bin_2> | <auroc_tb_age_bin_2> (<auroc_tb_age_bin_2_ci>) | <auroc_pna_age_bin_2> (<auroc_pna_age_bin_2_ci>) |
| Age: <age_bin_3_label> | <n_age_bin_3> | <auroc_tb_age_bin_3> (<auroc_tb_age_bin_3_ci>) | <auroc_pna_age_bin_3> (<auroc_pna_age_bin_3_ci>) |
| Site: internal | <n_site_internal> | <auroc_tb_site_internal> (<auroc_tb_site_internal_ci>) | <auroc_pna_site_internal> (<auroc_pna_site_internal_ci>) |
| Site: external (<external_site_name>) | <n_site_external> | <auroc_tb_site_external> (<auroc_tb_site_external_ci>) | <auroc_pna_site_external> (<auroc_pna_site_external_ci>) |

### Edge performance

| Metric | Value |
| --- | --- |
| p50 inference latency (CPU, INT8) | <p50_latency_ms> ms |
| p95 inference latency (CPU, INT8) | <p95_latency_ms> ms |
| Benchmark hardware | <benchmark_hardware> |

Measured with `python -m sehat.export.benchmark <onnx_artifact_name>`.

## Limitations

- **Site shift:** performance on the held-out site is lower than internal
  performance (see tables above). A new clinic's equipment, protocol, and
  population will differ again; treat external-site numbers, not internal ones,
  as the upper bound on real-world performance.
- **Demographic skew:** the public training collections over-represent certain
  age bands and clinical populations (see [data_card.md](data_card.md)); the
  pediatric population is not represented.
- **Small TB corpora:** the combined Shenzhen + Montgomery TB corpora contain
  ~800 films; external TB validation therefore has wide confidence intervals.
- **Label noise:** NIH ChestX-ray14 labels were mined from radiology reports
  and contain a non-trivial error rate; RSNA pneumonia labels cover only the
  "lung opacity" proxy, not confirmed pneumonia.
- **Calibration drift:** calibration was measured on static public test sets;
  prevalence shifts at deployment sites will move the operating point. Local
  re-validation is required before relying on absolute scores.
- <model_specific_limitation>

## Ethical considerations

- **Decision support, never diagnosis.** Outputs are triage aids for trained
  health workers. See [ethics.md](ethics.md) for the full position, including
  failure modes and the rationale for site-held-out evaluation.
- **Privacy:** in offline deployments, images and scores never leave the clinic
  device. No PHI is used in training or telemetry.
- **Dual risk:** false negatives can delay TB treatment (an infectious-disease
  risk to the patient and community); false positives consume scarce
  confirmatory-testing capacity. The operating threshold was chosen with both
  harms in view and is reported above, not hidden.
- **Equity:** subgroup metrics are part of the release gate; regressions
  outside bootstrap CIs block release (see M2 in [roadmap.md](roadmap.md)).

## Disclaimer

**Project Sehat is research software. It is not a medical device, has not been
cleared or approved by any regulatory authority (FDA, CE, CDSCO, or otherwise),
and must not be used as the sole basis for any clinical decision.** Always
confirm screening results with qualified clinical judgement and appropriate
diagnostic testing, and comply with local regulations before any clinical use.

## Feedback

Report issues, unexpected behavior, or bias concerns via the project's GitHub
issue tracker. Field reports from deployment settings are especially valuable
and will be reflected in future versions of this card.
