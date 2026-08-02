# Ethics — Project Sehat

Tuberculosis kills roughly **1.3 million people every year**, and the
bottleneck is rarely treatment — it is **access to diagnosis**. Chest X-ray
screening can triage patients toward confirmatory testing in places that have
an X-ray machine but no radiologist. That is the gap Project Sehat exists to
narrow. It is also a setting where a careless ML system can do real harm, so
this document states our positions plainly: what the software is, how it fails,
and what we do about those failures.

## Position 1: Decision support, never diagnosis

Project Sehat produces **screening/triage decision-support software**. Its
output is a calibrated risk score to help a trained health worker decide who
needs confirmatory testing (for TB: sputum NAAT or culture) — not a diagnosis,
and never a reason to deny care.

This position is enforced in the product, not just in prose:

- The serving API (`python -m sehat.serving`) returns scores and calibration
  context; there is no "diagnosis" field in the response schema.
- The clinic PWA's copy is reviewed to ensure it cannot be read as a definitive
  result. Contributions that blur this line are rejected at review (see
  [CONTRIBUTING.md](../CONTRIBUTING.md)).
- Reported metrics are screening metrics (sensitivity at fixed specificity,
  calibration), not diagnostic claims.

## Position 2: We name our failure modes

The honest answer to "does it work?" is "it depends where, and on whom." We
track three failure modes explicitly.

### Site shift

A model trained on films from one hospital learns that hospital — its machine,
exposure settings, patient mix, even film markers — not just the disease.
Deployed at a different site, performance drops. **Mitigation:** our release
gate requires evaluation on an entire dataset site held out from training
(e.g. train on Shenzhen, test on Montgomery). We report external-site metrics
separately and never pool them with internal ones, because the external number
is the only honest estimate of what a new clinic will see.

### Demographic bias

The public datasets skew by site, age, and sex (documented in
[data_card.md](data_card.md)). A screening model that under-performs for women
or for older patients systematically mis-triages them. **Mitigation:** subgroup
metrics (sex, age band, site) with bootstrap confidence intervals are part of
the release gate, not a footnote. A regression outside the CIs blocks release,
and subgroups too small to measure are disclosed rather than averaged away.

### Calibration drift

A risk score is only useful if "0.8" reliably means "about 80%." Prevalence
differs between the test set and any real clinic, so calibration measured on
public data will not transfer perfectly. **Mitigation:** we report ECE on both
internal and external sets, fit any recalibration on validation data only, and
state in the model card that absolute scores require local re-validation before
clinical reliance.

## Why site-held-out evaluation (and not just random splits)

Random image-level or even patient-level splits overestimate real-world
performance because train and test come from the same distribution — the same
machine, protocol, and population. The deployment scenario for Project Sehat is
precisely the opposite: a clinic we have never seen. Holding out an entire
*site* is the closest public-data proxy for that scenario. It costs us headline
metrics — external-site AUROC is lower, and we say so in every model card —
but a lower honest number is worth more than a higher imaginary one.

## Privacy

- **No PHI in development.** Training and evaluation use only public,
  de-identified research datasets (see [data_card.md](data_card.md)). No
  identifiable patient data enters the repository, manifests, or experiment
  tracking.
- **No PHI leaves the device in offline mode.** The intended deployment is a
  local FastAPI service plus an offline-capable PWA: the X-ray is processed on
  the clinic's own machine, and neither the image nor the score is transmitted
  anywhere. There is no telemetry in the serving path.
- **Metadata minimality.** Patient IDs in manifests exist only to guarantee
  patient-level splits and are never model features.

## Regulatory disclaimer

> **Project Sehat is research software. It is not a medical device and has not
> been cleared, approved, or certified by the FDA, a European notified body
> (CE/MDR), CDSCO, or any other regulatory authority. It must not be used as
> the sole basis for the diagnosis, prevention, or treatment of any disease.
> Clinical deployment may be regulated in your jurisdiction; it is the
> deployer's responsibility to comply with local law and to obtain any required
> approvals. Nothing in this repository constitutes medical advice.**

## What we ask of users and contributors

- Deploy only with trained health workers in the loop and confirmatory testing
  available.
- Re-validate (discrimination and calibration) on local data before relying on
  the scores, and report the results back — field evidence is how this project
  gets more honest.
- Report bias or unexpected failure via the issue tracker; these reports are
  treated with the same seriousness as crashes.

## Feedback and governance

Ethics-relevant changes (new data sources, new findings, threshold changes,
deployment-copy changes) require the same review as code and must update the
model card when metrics move (see the PR template). This document is versioned
with the code; material changes are called out in release notes.
