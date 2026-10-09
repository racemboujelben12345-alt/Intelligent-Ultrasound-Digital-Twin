# Phase 1 — Initial repository audit

**Status:** static source/documentation audit; not a complete execution audit.  
**Scope:** inspected selected source files and documentation on `main` after PR #16, then checked the GitHub Actions runs associated with this audit PR commit. The CI unit/integration-test job and final V&V job both completed successfully on commit `1724396219e4d3324fb73680f45a527d1f26bfac`. This is evidence of those configured checks passing on that commit, not a complete line-by-line audit or physical-device validation.

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
| `src/image_analysis/digital_signature.py` | Defines the versioned Digital Signature and canonical feature order. | `SIGNATURE_VERSION = 2.1`; `FEATURE_ORDER` currently lists 23 entries, including image and physics-derived fields. | Reconcile the feature contract with documentation that describes 11 historical features or a 13-feature signature. Establish one authoritative schema and test its version/hash. |
| `src/image_analysis/edge_quality.py` | Computes fixed Canny, median-adaptive Canny and gradient-percentile edge-density candidates. | Three definitions are implemented; PR #16 adds a digital-perturbation sensitivity audit. | Report sensitivity by transformation and image family; do not call one definition universally superior without evidence. |
| `src/validation/repeatability.py` | Repeats synthetic Gaussian-noise perturbations and summarizes feature variability. | Implementation is explicitly simulation-based. | Keep it labelled as a simulated noise/sensitivity estimate; never treat it as physical scanner repeatability. |
| `src/validation/experimental_repeatability.py` | Summarizes feature variability in supplied observed acquisitions and optionally by session. | Descriptive statistics and input checks are implemented. | Validate metadata/protocol quality and test with traceable real repeated acquisitions before making device-specific claims. |
| `src/validation/audit.py` and `scripts/run_vv_audit.py` | Check acquisition contracts, IDs, simulation lineage, reproducibility and degradation response. | A reproducible software/data-contract audit path exists. | Inspect all checks and run it against a documented dataset; a PASS remains software/simulation evidence, not proof of physical validity. |
| `src/ai/fusion.py` | Combines statistical, AI, quality and physical evidence with fixed weights. | Formula and thresholds are explicit; scores are documented as evidence, not probabilities. | Treat weights/thresholds as provisional until calibrated on representative independent data; test disagreement and edge cases. |
| `src/digital_twin/state_engine.py` | Maps statistical, fusion, health and temporal drift evidence to one canonical state. | Deterministic states and validation are implemented. AI/fusion high evidence alone maps at most to `EARLY_DRIFT`, not `HIGH_DEVIATION`. | Add/verify transition, persistence, missing-evidence and hysteresis tests against the project’s state policy. |
| `src/prediction/` and AI model modules | Forecasting and AI components are documented as existing project capabilities. | Documentation describes baselines, a Random Forest ensemble and empirical P05/P95 ensemble spread. | Inventory exact entry points and confirm actual target construction, temporal splits, baseline comparison, leakage controls and interval coverage. Ensemble spread is not automatically a calibrated prediction interval. |
| `dashboard/app.py` | Streamlit presentation layer reads artifacts from `outputs/`. | Dashboard loads summary/state/feature artifacts and reports when pipeline artifacts are absent. | Trace every displayed value to its source artifact; label demo/simulated data and prevent illustrative values from appearing as measured results. |
| `docs/DATA_CONTRACT.md`, `docs/VALIDATION_MATRIX.md`, AI governance docs | Define provenance, validation levels and model-claim boundaries. | The documentation explicitly separates public, simulated and experimental evidence. | Align feature counts, commands, file names and claims with the actual implementation; keep one authoritative source of truth. |

## 3. Audit findings and risks

### P0 — Resolve before claiming a stable scientific pipeline

1. **Feature-schema inconsistency:** `digital_signature.py` lists 23 canonical feature names, while other project documents refer to 11 historical features or a 13-feature signature. Determine the actual vector contract and update documentation/tests together.
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

## 4. Phase alignment

| Project phase | Initial audit assessment | Exit criterion |
|---|---|---|
| Phase 0 — Stabilization | GitHub Actions reports the configured unit/integration-test job and final engineering V&V job succeeded on audit commit `1724396219e4d3324fb73680f45a527d1f26bfac`. | Confirm those checks are the required gates for the current project; separately verify whether `compileall` and `git diff --check` are configured/executed, since their execution is not established by the job summaries retrieved. |
| Phase 1 — Repository audit | **Started; not complete.** This document is a static first pass, not a line-by-line audit of every module. | Complete module inventory, run tests/V&V, reconcile claims and publish prioritized issues. |
| Phase 2 — Data contract and quality | Contract and audit components exist. | Reproducible data-quality report with source counts, duplicates, missing metadata and lineage-leakage checks. |
| Phase 3 — Features and signal behaviour | Signature, physics and edge-feature modules exist; PR #16 adds one robustness analysis. | Versioned feature catalogue and controlled sensitivity/redundancy report. |
| Phase 4 — Predictive benchmark | Forecasting infrastructure is documented and baseline-related PRs exist. | Executed leakage-aware benchmark on valid ordered sequences with a held-out temporal test. |
| Phases 5–10 | Partial architecture/documentation exists; completion depends on evidence and execution. | Separate acceptance criteria for uncertainty, fusion/state, dashboard, simulations, real acquisitions and final report. |

## 5. Recommended next actions

1. CI and final V&V were confirmed successful for the current audit commit. Next, inspect the workflow definitions and logs to verify exact test coverage and whether `compileall` and `git diff --check` are included.
2. Reconcile the Digital Signature feature schema (23 vs 13 vs 11) and add a test that the documented schema matches the produced vector.
3. Finish the module-by-module audit, including AI, drift, prediction, explainability, metadata/provenance, tests and workflows.
4. Generate a source-aware data inventory and verify lineage-safe partitions.
5. Only then run or extend the predictive benchmark using ordered sequences and fixed baseline comparisons.

## Evidence boundary

This document is a static inspection of selected source files and documentation. It does **not** claim that tests passed, the complete repository has been audited, the predictive models have been evaluated, or SCAN A has been experimentally validated.
