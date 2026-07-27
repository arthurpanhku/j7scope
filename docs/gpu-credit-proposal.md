# J7Scope GPU Cloud Credit Proposal

## Project title

J7Scope: Reproducible Cross-Lingual Validation of the Jacobian-Lens Global Workspace

## Abstract

J7Scope is an open-source interpretability project testing whether the
verbalizable “global workspace” identified by Jacobian-lens methods is shared
across Chinese and English. We will reproduce the published pretrained
Jacobian lens on a known Qwen case, then fit and evaluate a bilingual lens on
open-weight Qwen models. The project separates GPU-dependent capture from a
zero-GPU static replay platform: every experiment produces a versioned trace
with model revision, estimator definition, Jacobian provenance, shuffled-pair
null baselines, confidence intervals, and deep links to individual tokens.

Cloud credits will fund a finite validation and capture campaign rather than
an ongoing service. Deliverables will include (1) an upstream reproduction
report, (2) cross-lingual overlap/CKA/SVCCA results with null and same-language
controls, (3) a curated bilingual trace dataset, and (4) a public Zenodo
release with DOI. Code is Apache-2.0 and all small artifacts, configurations,
and methodology documentation will be publicly available.

## Research question

Do semantically matched Chinese and English prompts converge on a shared
Jacobian-lens subspace beyond shuffled-pair and same-language baselines?

## Work plan

1. Reproduce the published Qwen3.5-4B pretrained-lens walkthrough and archive
   resolved model/lens revisions and layer-level readouts.
2. Lock the estimator definition by comparing position-local and published
   future-target reductions.
3. Fit Qwen2.5-7B on a generic, language-balanced corpus.
4. Run M1 on deception and manipulation probes, then expand to all registered
   concept categories with same-language controls.
5. Capture bilingual traces and publish code, metrics, data cards, and a
   versioned dataset with DOI.

## Requested resources

- Requested credit: USD 1,000.
- GPU: NVIDIA A100 40/80 GB, L40S 48 GB, or equivalent.
- Planned usage: up to 200 GPU-hours, adjusted after the upstream reproduction
  and a measured ten-prompt fitting benchmark.
- Storage: 200 GB temporary model/cache storage and under 20 GB durable
  experiment artifacts.
- Compute is batchable and will be stopped between experiments.

## Reproducibility and public benefit

Every run records the exact model revision, upstream code commit, lens
snapshot, estimator, corpus identifier, layer, seed, dtype, and GPU. Results
will be released through the public J7Scope repository and Zenodo. The static
viewer requires no ongoing GPU resources, so the credit creates a durable
public research artifact rather than a service with recurring compute cost.

## Data and safety

The project uses public open-weight models and public, authored, or synthetic
probe text. It does not process personal, medical, financial, or otherwise
sensitive user data. Model licenses and upstream Apache-2.0 attribution will
be preserved.

## Timeline

- Weeks 1–2: upstream reproduction and estimator lock.
- Weeks 3–5: first bilingual M1 batch and robustness controls.
- Weeks 6–8: expanded concepts, trace capture, data card, and Zenodo release.
