# Data Card — Project Sehat

This card documents every dataset used to train and evaluate Project Sehat
models: provenance, licensing and usage terms, known biases, and preprocessing.
Project Sehat trains **exclusively on public, de-identified research
datasets** — no PHI, no clinic data, no scraped images.

For how these datasets flow into splits and training, see
[architecture.md](architecture.md).

## Summary table

| Dataset | Role | Size (approx.) | Labels | Access |
| --- | --- | --- | --- | --- |
| Shenzhen Hospital TB (NLM) | TB training + internal test | ~660 frontal films (one site) | TB positive / normal | Public download from NIH/NLM |
| Montgomery County TB (NLM) | TB **site-held-out external validation** | ~140 frontal films (one site) | TB positive / normal | Public download from NIH/NLM |
| NIH ChestX-ray14 | Pretraining / auxiliary findings | ~112k frontal films, ~30k patients | 14 findings, NLP-mined from reports | Public, after agreeing to NIH terms |
| RSNA Pneumonia Detection Challenge | Pneumonia training + evaluation | ~30k frontal films (DICOM, subset of ChestX-ray14 re-annotated) | Lung opacity / normal (+ "not normal") | Kaggle competition download |

Combined, the dedicated TB corpora contain **~800 films** — small by deep
learning standards. This is a first-order limitation and is reflected in the
width of our confidence intervals; see the [model card](model_card_template.md).

## Sources and usage terms

> **Honest note on licensing.** None of these datasets ships with a standard
> open-source license text. Each is distributed for research use under its own
> terms, summarized below from the distributors' pages. **This summary is not
> legal advice.** If you redistribute derived artifacts (e.g. preprocessed
> tensors), verify the current terms at the source first. Project Sehat itself
> distributes manifests and hashes — never the images.

### Shenzhen & Montgomery TB datasets

- **Publisher:** U.S. National Library of Medicine (NLM), in collaboration with
  Shenzhen No. 3 People's Hospital (China) and the Montgomery County Department
  of Health (Maryland, USA).
- **Citation:** Jaeger S. et al., "Two public chest X-ray datasets for
  computer-aided screening of pulmonary diseases," *Quantitative Imaging in
  Medicine and Surgery*, 2014.
- **Terms:** made publicly available by NLM for research purposes, with the
  above citation requested. De-identified; no PHI.
- **Our usage:** training, internal testing (patient-level splits), and
  site-held-out external validation consistent with that research purpose.

### NIH ChestX-ray14

- **Publisher:** NIH Clinical Center.
- **Citation:** Wang X. et al., "ChestX-ray8: Hospital-scale Chest X-ray
  Database and Benchmarks…," *CVPR*, 2017.
- **Terms:** released by NIH for research use; the download page requires
  acknowledging NIH's data-use terms. Labels were extracted from radiology
  reports via NLP and are known to be noisy (the publisher and follow-up
  literature both note label error; treat individual labels as
  approximate).
- **Our usage:** representation learning/pretraining and auxiliary supervision.
  We do not report headline TB/pneumonia screening claims against these labels.

### RSNA Pneumonia Detection Challenge

- **Publisher:** Radiological Society of North America (RSNA), hosted on Kaggle
  (2018). Images are a re-annotated subset of NIH ChestX-ray14.
- **Terms:** Kaggle competition rules — free for research and competition use;
  see the competition page for the current text.
- **Labels caveat:** the positive class is *pneumonia-like lung opacity as
  localized by radiologists*, not microbiologically confirmed pneumonia. We
  treat it as a proxy and say so wherever results are reported.

## Known biases and dataset pathologies

These are known to the project and actively mitigated or measured — not
discovered after the fact:

1. **Site bias.** Each TB dataset comes from a single site with its own
   acquisition hardware, protocol, and patient mix. Models can learn the site
   (e.g. film markers, exposure characteristics) instead of the disease. This
   is the primary reason our evaluation **holds one entire site out** for
   external validation (see [ethics.md](ethics.md)) — a model that has only
   seen Shenzhen films is not trusted until it performs on Montgomery's.
2. **Demographic skew.** ChestX-ray14 derives from one U.S. clinical system and
   skews toward older adults; the TB corpora come from China and the U.S. and
   under-represent children and some age bands. Sex/age metadata is incomplete
   in places. Consequence: subgroup metrics are part of the release gate, and
   subgroups with too few samples are reported with wide CIs rather than
   silently dropped.
3. **Label noise.** ChestX-ray14's NLP-mined labels have a documented error
   rate; RSNA's opacity labels are a proxy for pneumonia. We mitigate by using
   these sets for pretraining/auxiliary tasks rather than as ground truth for
   headline claims.
4. **Spectrum effects.** "Normal" in these corpora means "no flagged finding in
   a hospital picture archive," which differs from "healthy screening
   volunteer." Prevalence and case mix will not match any deployment clinic;
   absolute calibration must be re-validated locally.
5. **Scale imbalance.** ~800 dedicated TB films vs. ~112k ChestX-ray14 images.
   We report per-dataset provenance in every manifest so no one can mistake a
   big number for a well-measured one.

## Preprocessing

Applied uniformly at manifest-consumption time (implemented in
`src/sehat/data/`, versioned with the manifest hash):

1. Convert to grayscale; apply the dataset's native intensity range.
2. Resize to the model input resolution (see the training config, e.g.
   `configs/train/tb_baseline.yaml`) with aspect-preserving letterboxing —
   no anamorphic stretching, which distorts cardiothoracic ratios.
3. Normalize intensities (per-image standardization; training-set statistics
   stored in the manifest).
4. Attach metadata used for subgroup evaluation: patient ID (split grouping
   only), site, sex, and age band where available. Patient IDs are used
   **only** to keep one patient in one split — they are never features.
5. Test-time and external-site preprocessing is byte-identical to training
   preprocessing; any change produces a new manifest hash.

## Retention, privacy, and deletion

- All datasets are public and de-identified by their publishers; the project
  holds no identifiable data and therefore has nothing to delete on request.
- If any publisher withdraws or restricts a dataset, we will stop using it,
  note the change here, and regenerate affected manifests and model cards.

## Maintenance

This card is updated whenever a dataset version, split policy, or preprocessing
step changes. The manifest hash on a rendered model card links a model to the
exact data versions described here at training time.
