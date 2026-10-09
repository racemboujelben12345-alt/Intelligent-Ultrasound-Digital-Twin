# Scientific and Technical Audit — Baseline Review

**Status:** preliminary repository audit  
**Reference branch:** `main`  
**Purpose:** establish what the current code demonstrably implements before extending predictive AI.

This document is a source-code and test-inventory review. It is not a fresh local execution of the test suite, a dataset audit, or experimental validation of a physical ultrasound scanner. A module is not considered scientifically validated merely because it exists or has unit tests.

## 1. Evidence maturity vocabulary

| Level | Meaning | Evidence required |
|---|---|---|
| Implemented | Code for the capability exists | Source inspection |
| Tested | Defined software behavior has automated checks | Relevant passing test run |
| Evaluated | Performance measured under a stated protocol | Reproducible benchmark, split definition and metrics |
| Experimentally supported | Traceable physical acquisitions support the specific claim | Documented acquisition protocol and analysis |
| Validated for intended use | Evidence supports a narrowly defined operational use | Intended-use-specific validation and acceptance criteria |

These levels are not interchangeable. The current audit labels implementation and test coverage from repository inspection; it does not upgrade a capability to evaluated or experimentally validated without separate evidence.

## 2. Module inventory

| Module / area | Observed responsibility | Existing tests or evidence found in repository | Limits / next audit action |
|---|---|---|---|
| `src/acquisition/models.py`, `src/acquisition/loader.py` | Canonical acquisition object; source, session, timestamp, image and simulation-lineage metadata; image loading | `tests/test_data_contract.py`, validation/audit tests and V&V workflow are present | Audit real dataset manifests, duplicate detection, missing metadata handling, timestamp ordering and grouping. Confirm public acquisitions can be represented without fabricated device/session values. |
| `src/image_analysis/digital_signature.py` | Builds a quantitative image signature and connects image features with physics-derived proxies | Signature, physics, simulation and contrast-integrity test files exist | Inventory each feature definition, units/range, preprocessing dependence, redundancy and robustness. A display-image metric is not automatically a calibrated acoustic measurement. |
| `src/image_analysis/edge_quality.py` | Compares fixed Canny, adaptive Canny and gradient-percentile edge-density definitions | Root script `edge_quality_evaluation.py` exists; relevant test coverage must be confirmed | Fixed Canny 50/150 is a candidate definition, not a universal threshold. Compare on controlled, representative data and document sensitivity to scaling and normalization. |
| `src/physics/` and physics-informed documentation | Computes physics-linked quantities and metadata consistency proxies | `tests/test_physics_engine.py`, `tests/test_physics_informed_model.py`, and counterfactual tests exist | Verify assumptions, units, input availability and fallback behavior. The reference equipment manual is not a specification or validation dataset for the university scanner. |
| `src/ai/intelligence.py` | Isolation Forest anomaly evidence, supervised Random Forest classification, predictive regression ensemble, model/inference records and feature contribution outputs | `tests/test_ai_intelligence.py`, `tests/test_explainability.py` and fusion tests exist | Confirm every trained target has a defensible label and split. Ensemble spread / P05-P95 is not automatically calibrated uncertainty. Anomaly vote rate is explicitly not a calibrated probability. |
| `src/ai/fusion.py`, `src/digital_twin/health.py` | Combines evidence and calculates engineering-health/confidence indicators | `tests/test_ai_fusion.py`, `tests/test_twin_health.py`, evidence-fusion tests exist | Check dependence between fused inputs to avoid double-counting correlated features/evidence. Engineering scores are not probabilities of hardware failure. |
| `src/digital_twin/state_engine.py` | Deterministic mapping of statistical, fusion, health and drift evidence to four Twin states | `tests/test_state_engine.py` explicitly checks that high fusion evidence alone does not force `HIGH_DEVIATION` | Expand transition tests for invalid/low-quality data, high uncertainty, domain shift, contradictory evidence, persistence, hysteresis and recovery. Review all state branches against the documented policy. |
| `src/drift/` | Existing drift calibration, control charts and temporal history analysis | `tests/test_drift_calibration.py`, `test_drift_engine.py`, `test_drift_monitor.py`, history integration tests exist | Separate detector calibration from evaluation. Report false alarms per observation/time and detection delay on pre-specified scenarios. |
| `src/prediction/engine.py`, `trend.py` | Fits a trend to existing Twin-history signals and reports a next-point trend estimate | `tests/test_prediction_engine.py`, `tests/test_trend_analyzer.py` exist | This is not yet a general feature-forecasting benchmark across ordered acquisitions. Verify trend uncertainty, horizon, temporal validation and baseline comparison. |
| `src/prediction/forecasting_evaluation.py` | MAE/RMSE/optional MASE, relative MAE against persistence and expanding-window rolling-origin evaluation for one ordered series | `tests/test_forecasting_evaluation.py` checks hand-calculated metrics, observed-prefix behavior and edge cases | This is an evaluation harness, not a trained forecaster. It does not automatically create independent sequences or group/session splits. Add baseline comparison and multi-series orchestration only after defining valid temporal inputs. |
| `src/prediction/sequential_drift.py` | Robust reference scaling with EWMA/CUSUM-style sequential warnings | `tests/test_sequential_drift.py` exists | Thresholds need calibration for cadence and false-alarm burden. An alarm is statistical evidence, not a physical cause. |
| `src/prediction/uncertainty_calibration.py` | Split-conformal interval radius and empirical coverage/width/tail-miss metrics | `tests/test_uncertainty_calibration.py` exists | Exchangeability can fail under temporal dependence, drift and domain shift. Ensure the predictor is fixed before calibration and the test set remains independent. |
| `src/validation/repeatability.py` | Perturbs an image digitally and measures signature sensitivity | `tests/test_repeatability.py` exists | Must be described as simulated perturbation sensitivity, not physical scanner repeatability. |
| `src/validation/experimental_repeatability.py` | Descriptive variability summaries for supplied observed acquisitions | `tests/test_experimental_repeatability.py` exists | Interpretation depends on session grouping and acquisition protocol. It does not prove general scanner repeatability on its own. |
| `src/validation/audit.py` and `scripts/run_vv_audit.py` | Acquisition checks, reproducibility/severity checks and lineage-aware partition helpers | `tests/test_evidence_vv.py`, `tests/test_multisample_validation.py`, `tests/test_signature_simulation.py` exist | Confirm each check is exercised by a negative test, and inspect whether all data-loading/training paths actually use the audit gates. |
| Dashboard, root scripts and generated reports | User-facing view and exploratory public-data analysis scripts exist | Root scripts include BUSI/public-statistics/feature-quality analyses; CI runs pytest and a separate V&V workflow | Audit dashboard data flow to ensure illustrative values cannot be mistaken for measured results. Convert important exploratory analyses into reproducible scripts with provenance and saved metric summaries. |
| CI | `.github/workflows/ci.yml` runs `python -m pytest tests -q` on Python 3.11; `.github/workflows/vv.yml` runs the V&V audit | Workflow files are present | A workflow definition is not proof that the current commit passed. Record commit SHA, run URL, conclusion and test count for each validation claim. |

## 3. Confirmed strengths

1. Data provenance and synthetic-parent lineage are represented in the acquisition contract.
2. Public, simulated and experimental evidence are explicitly intended to remain separate.
3. The state engine has a conservative rule against promoting high AI-fusion evidence alone to `HIGH_DEVIATION`.
4. Predictive evaluation has a persistence baseline and rolling-origin machinery that passes only an observed history prefix to the forecaster.
5. Interval calibration reports coverage and width and documents assumptions/limitations.
6. Documentation explicitly limits claims about hardware faults, calibrated acoustic measurements and physical validation.

These are architectural strengths; they do not establish predictive superiority or physical-system validity.

## 4. Highest-priority gaps

### P0 — Establish a reproducible baseline
- Record the exact `main` commit SHA and the latest successful CI/V&V run IDs before each milestone.
- Run tests and V&V against that exact commit; retain the raw logs and test count.
- Produce an inventory of implemented modules, corresponding tests and untested branches.

### P1 — Define the valid temporal unit
- Define one series as a sequence of comparable, ordered acquisitions with documented timestamps/order and grouping.
- Do not turn BUSI or USSimAndSegm image collections into scanner time series without genuine temporal links.
- Specify feature, prediction horizon, acquisition/session grouping, training window and evaluation targets before fitting a model.

### P1 — Build a baseline comparison protocol
- Compare persistence, moving-average or exponential-smoothing baselines, and a trend/linear baseline on identical rolling-origin targets.
- Report MAE, RMSE, MASE where defined and relative MAE to persistence.
- Report results per feature and per independent sequence, not only a pooled average.
- Do not introduce deep sequence models until independent sequences and baseline results justify them.

### P1 — Verify uncertainty under the intended split
- Freeze the predictor before calibration.
- Keep calibration and final test observations separate.
- Report empirical coverage and interval width on an independent test set.
- Add explicit temporal/domain-shift warnings and an abstention/OOD policy before displaying intervals as dependable operational uncertainty.

### P2 — Connect prediction to early warning carefully
- Define the engineering threshold from a justified protocol, not by optimizing on test data.
- Evaluate lead time, false alarms per unit time/observation, missed events and threshold-crossing classification only where labels support it.
- Keep detection-now, forecasted deviation and event-risk prediction as separate outputs.

### P2 — Test evidence fusion and state transitions
- Add cases for invalid inputs, high uncertainty, model disagreement, domain shift, persistent and transient drift, recovery/hysteresis and conflicting evidence.
- Check whether correlated features or dependent models are counted more than once in the fused evidence.
- Ensure every displayed state includes a traceable rationale and evidence identifiers.

### P3 — Strengthen feature and physics audit
- Create a feature catalogue with formula, unit/range, preprocessing, expected sensitivities, known confounders and tests.
- Quantify redundancy and stability; do not remove features from correlation alone.
- Distinguish measured metadata, model-derived proxies and unknown values.
- Treat fixed edge thresholds and physics assumptions as candidates requiring validation.

### P3 — Dashboard and reporting integrity
- Label public, simulated and experimental results separately in every chart/report.
- Ensure all displayed predictions and intervals come from the selected model output, not static illustrative values.
- Show acquisition order/session, model version, provenance, uncertainty and warnings alongside forecasts.

## 5. Next implementation milestone

The next code milestone should be a **baseline model comparison layer** built on the existing `evaluate_rolling_origin` API, not another complex AI model.

Proposed acceptance criteria:
1. Persistence and at least one simple trend/smoothing forecaster are evaluated on the same target indices.
2. The API rejects non-finite, too-short and misaligned inputs.
3. Every score is traceable to the feature, horizon, origin indices and source sequence.
4. A synthetic ordered sequence with known behavior tests the evaluation mechanics; results are labelled simulated.
5. Tests demonstrate that no future observation is passed to a forecaster.
6. Documentation states that success on a synthetic sequence validates software behavior only.
7. No claim of performance on a real ultrasound system is made without experimental sequences.

## 6. Explicit non-claims

This audit does not claim:
- that a physical ultrasound system has been experimentally validated;
- that the project predicts hardware failures;
- that the existing ensemble uncertainty is calibrated;
- that public image datasets establish scanner dynamics;
- that any predictive model outperforms persistence on real acquisition sequences;
- that a green CI run proves scientific validity.

## 7. Audit conclusion

The repository already contains meaningful infrastructure for image signatures, physics-derived proxies, anomaly evidence, state fusion, drift monitoring, rolling-origin evaluation, uncertainty calibration and data lineage. The main scientific bottleneck is now **not the absence of more model families**. It is the need for a clean temporal evaluation protocol, explicit baseline comparisons, audited state transitions and evidence-labelled reporting.

Proceed in this order: verify the current CI/V&V baseline → finish the code/test map → implement simple forecasting baselines → evaluate on valid ordered sequences → only then connect forecasts and uncertainty to the Twin state engine.
