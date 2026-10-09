# Phase 1 — Initial repository audit

**Status:** static source/documentation audit; not a complete execution audit.  
**Scope:** static inspection of selected source files and documentation. GitHub Actions unit/integration tests and the final engineering V&V workflow succeeded on the PR #17 audit commit (`1724396219e4d3324fb73680f45a527d1f26bfac`) and on the PR #18 feature-contract test commit (`1d34b4ed1eca09d41fd2a3e0f770e6af1c0f1252`). PR #18 is merged. This is not a complete line-by-line or physical-device validation.

## 1. Scientific evidence levels

Use these labels consistently in the project:

- **Implemented:** code exists in the repository.
- **Tested:** automated tests verify specified software behaviour.
- **Evaluated:** a defined protocol has produced reported metrics.
- **Experimentally supported:** traceable acquisitions from a physical scanner support the conclusion.
- **Validated for a use case:** evidence is adequate for a narrowly defined intended use.

A code path or passing unit test must not be described as experimental validation.

## 2. Initial module inventory

| Component | Observed responsibility | Current evidence visible from source | Main gap / next action |
|---|---|---|---|
| `src/acquisition/loader.py` | Loads grayscale images, resizes them and supports experimental/public/demo sources. | Source-aware loading and explicit synthetic demo phantom are implemented. | Audit metadata completeness, original-image preservation, resize policy and source counts on the actual available files. |
| `src/image_analysis/digital_signature.py` | Defines the versioned Digital Signature and canonical feature order. | `SIGNATURE_VERSION = 2.1`; `FEATURE_ORDER` currently lists 23 entries, including image and physics-derived fields. | Document the distinct roles: the canonical `DigitalSignature` vector has 23 features; `StatisticalBaseline.DEFAULT_FEATURES` is a 13-feature subset; the project's 11-feature list is historical. Keep these counts labelled by role/version and test vector order and subset compatibility. |
| `src/image_analysis/edge_quality.py` | Computes fixed Canny, median-adaptive Canny and gradient-percentile edge-density candidates. | Three definitions are implemented; PR #16 adds a digital-perturbation sensitivity audit. | Report sensitivity by transformation and image family; do not call one definition universally superior without evidence. |
| `src/validation/repeatability.py` | Repeats synthetic Gaussian-noise perturbations and summarizes feature variability. | Implementation is explicitly simulation-based. | Keep it labelled as a simulated noise/sensitivity estimate; never treat it as physical scanner repeatability. |
| `src/validation/experimental_repeatability.py` | Summarizes feature variability in supplied observed acquisitions and optionally by session. | Descriptive statistics and input checks are implemented. | Validate metadata/protocol quality and test with traceable real repeated acquisitions before making device-specific claims. |
| `src/validation/audit.py` and `scripts/run_vv_audit.py` | Check acquisition contracts, IDs, simulation lineage, reproducibility and degradation response. | A reproducible software/data-contract audit path exists. | Inspect all checks and run it against a documented dataset; a PASS remains software/simulation evidence, not proof of physical validity. |
| `src/ai/fusion.py` | Combines statistical, AI, quality and physical evidence with fixed weights. | Formula and thresholds are explicit; outputs are documented as evidence, not probabilities. | Treat weights/thresholds as provisional until calibrated on representative independent data; test disagreement and edge cases. |
| `src/digital_twin/state_engine.py` | Maps statistical, fusion, health and temporal drift evidence to one canonical state. | Deterministic states and validation are implemented. AI/fusion high evidence alone maps at most to `EARLY_DRIFT`, not `HIGH_DEVIATION`. | Add/verify transition, persistence, missing-evidence and hysteresis tests against the project’s state policy. |
| `src/prediction/` and AI model modules | Forecasting and AI components are documented as existing project capabilities. | Documentation describes baselines, a Random Forest ensemble and empirical P05/P95 ensemble spread. | Inventory exact entry points and confirm actual target construction, temporal splits, baseline comparison, leakage controls and interval coverage. Ensemble spread is not automatically a calibrated prediction interval. |
| `dashboard/app.py` | Streamlit presentation layer reads artifacts from `outputs/`. | Dashboard loads summary/state/feature artifacts and reports when pipeline artifacts are absent. | Trace every displayed value to its source artifact; label demo/simulated data and prevent illustrative values from appearing as measured results. |
| `docs/DATA_CONTRACT.md`, `docs/VALIDATION_MATRIX.md`, AI governance docs | Define provenance, validation levels and model-claim boundaries. | The documentation explicitly separates public, simulated and experimental evidence. | Align feature counts, commands, file names and claims with the actual implementation; keep one authoritative source of truth. |

## 3. Audit findings and risks

### P0 — Resolve before claiming a stable scientific pipeline

1. **Feature-schema clarity:** `digital_signature.py` defines 23 canonical features, while `src/signature/baseline.py` defines a 13-feature default subset and the project plan lists 11 historical features. These counts serve different roles and must not be presented as interchangeable. A regression test now checks canonical vector order and compatibility of the 13-feature subset.
2. **Evidence-source separation:** every output and metric must retain `public`, `simulated` or `experimental` provenance. Do not aggregate them into a single unlabeled score.
3. **Lineage-aware splits:** parent images and their digitally transformed derivatives must remain in the same partition. Session- or sequence-related observations must not leak across temporal train/calibration/test boundaries.
4. **State policy:** retain the conservative rule that AI/fusion evidence alone cannot establish `HIGH_DEVIATION`. The rationale and contributing evidence must be inspectable.

### P1 — Required before interpreting predictive performance

5. Verify that the forecasting benchmark uses ordered sequences and compares each candidate with persistence and other simple baselines on identical forecast origins.
6. Report per-feature and aggregate MAE/RMSE/MASE as applicable, plus uncertainty and test-set definitions. Do not infer scanner dynamics from unordered public image collections.
7. Evaluate prediction intervals on held-out temporal data using coverage and width; describe ensemble P05/P95 as ensemble spread until calibrated.
8. Calibrate drift thresholds on separate nominal data and report false alarms, detection rate and delay together, including uncertainty and scenario strata.
9. Trace dashboard fields to real pipeline outputs and show source, model/version and limitations.

### P2 — Experimental evidence

10. Collect authorized, traceable repeated acquisitions under a documented protocol, ideally with a suitable phantom and recorded settings.
11. Estimate within-session and between-session variability separately from digitally simulated noise.
12. Do not deliberately degrade clinical equipment. Any physical variation must be approved, safe and documented.

## 4. Focused AI, drift and state-engine review (static)

The following findings come from source inspection, not from a completed benchmark:

- **Unsupervised anomaly assessment:** `UltrasoundAIEngine.fit_reference()` fits an ensemble of Isolation Forest models and computes an empirical rank against reference anomaly scores. The score is a relative anomaly-evidence percentile for that reference set, not a calibrated probability of failure.
- **Vote rate and confidence:** the anomaly vote fraction and ensemble agreement describe model votes. The backward-compatible `anomaly_probability` property explicitly returns the vote fraction; downstream UI must not label it as a calibrated probability. The confidence formula is a heuristic, not validated calibration.
- **Supervised classifier:** `fit_supervised()` uses a Random Forest classifier and `predict_class()` exposes `predict_proba`. These outputs are model scores unless probability calibration and held-out reliability are demonstrated.
- **Predictive regression:** `fit_predictive_ensemble()` fits Random Forest regressors on the same supplied history. `predict_with_uncertainty()` uses the 5th and 95th percentiles of the ensemble member predictions and defines confidence as (1/(1+spread)). This is an ensemble-spread interval and a scale-dependent heuristic, not calibrated predictive coverage or a probability of correctness.
- **Explainability:** `anomaly_feature_contributions()` replaces each feature with its reference mean and measures score change. This is a perturbation sensitivity ranking, not causal attribution; correlated features can make the ranking unstable.
- **Fusion:** fixed weights (35% statistical, 35% AI, 15% image quality and 15% physical evidence) and fixed state thresholds are transparent engineering defaults. The code comments appropriately say they need representative validation before deployment.
- **State engine:** `src/digital_twin/state_engine.py` makes a deterministic state from statistical, fusion, health and temporal drift evidence. Statistical or health high-deviation evidence can trigger `HIGH_DEVIATION`; fusion high evidence alone leads to `EARLY_DRIFT`. The numeric confidence is an average of heuristic terms, not a calibrated probability.
- **Temporal drift:** `src/drift/engine.py` applies a configured monitor to the historical Mahalanobis-squared series. Source inspection alone does not establish false-alarm rate, detection delay, or sensitivity on SCAN A.

### Required follow-up tests/evaluation

1. Verify target construction and feature/label alignment for each AI task.
2. Ensure train/calibration/test partitions are ordered by time where appropriate and grouped by acquisition lineage, parent image, session or sequence to prevent leakage.
3. Compare predictive models with persistence and other simple baselines at identical forecast origins.
4. Measure held-out regression errors and prediction-interval coverage/width; do not report ensemble spread as calibrated uncertainty.
5. Evaluate anomaly thresholds and drift monitors on separate nominal and perturbed datasets; report false alarms and detection delay by scenario.
6. Test state precedence, contradictory evidence, absent drift data, persistent drift, and boundary values.
7. Check that dashboard labels distinguish score, vote fraction, heuristic confidence, calibrated probability (if ever validated), and physical evidence.

## 5. Phase alignment

| Project phase | Initial audit assessment | Exit criterion |
|---|---|---|
| Phase 0 — Stabilization | The configured unit/integration-test and final engineering V&V jobs succeeded on PR #17 and PR #18 commits; PR #18 is merged. | Confirm repository-wide checks and test coverage are adequate; CI success is software evidence only, not physical validation. |
| Phase 1 — Repository audit | **In progress.** Feature-contract tests and a focused static AI/drift review are recorded; this is not a complete execution audit of every module. | Complete module inventory, run tests/V&V, reconcile claims and publish prioritized issues. |
| Phase 2 — Data contract and quality | Contract and audit components exist. | Reproducible data-quality report with source counts, duplicates, missing metadata and lineage-leakage checks. |
| Phase 3 — Features and signal behaviour | Signature, physics and edge-feature modules exist; PR #16 adds one robustness analysis. | Versioned feature catalogue and controlled sensitivity/redundancy report. |
| Phase 4 — Predictive benchmark | Forecasting infrastructure is documented and baseline-related PRs exist. | Executed leakage-aware benchmark on valid ordered sequences with a held-out temporal test. |
| Phases 5–10 | Partial architecture/documentation exists; completion depends on evidence and execution. | Separate acceptance criteria for uncertainty, fusion/state, dashboard, simulations, real acquisitions and final report. |

## 6. Recommended next actions

1. Verify CI for the new feature-contract regression test.
2. Keep the 23-feature canonical signature, 13-feature default baseline subset and 11 historical features clearly distinguished in docs and tests.
3. Finish the module-by-module audit, including AI, drift, prediction, explainability, metadata/provenance, tests and workflows.
4. Generate a source-aware data inventory and verify lineage-safe partitions.
5. Only then run or extend the predictive benchmark using ordered sequences and fixed baseline comparisons.

## Evidence boundary

This document is a static inspection of selected source files and documentation. Configured CI workflows passed on the cited PR #17 and PR #18 commits, but this does **not** mean the complete repository has been audited, predictive performance has been evaluated, model uncertainty is calibrated, or SCAN A has been experimentally validated.
