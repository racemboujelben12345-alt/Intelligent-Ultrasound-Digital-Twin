# Predictive AI research plan

## Research objective

Develop and evaluate a predictive, uncertainty-aware framework for monitoring
the *observable behaviour* of ultrasound image acquisitions over time. The
first target is forecasting selected image-quality features across ordered,
comparable acquisitions. Later tasks may include early drift warnings and
threshold-crossing risk if sufficient labelled temporal evidence becomes
available.

This is an engineering research prototype. It is not a clinical diagnostic
system, and forecast errors do not by themselves establish a physical device
fault.

## Evidence classes

Keep these evidence classes separate in data contracts, reports and charts:

1. **Public data**: useful for image-feature methodology and external-domain
   experiments. Unordered image collections do not automatically provide
   temporal sequences of a scanner.
2. **Simulated data**: useful for testing known trends, controlled drift and
   software behaviour. Synthetic success is not experimental validation.
3. **Experimental data**: traceable acquisitions from a physical ultrasound
   system, with protocol and acquisition metadata. Required for claims about
   the tested physical setup.

Never pool these classes without an explicit, justified experiment and
stratified reporting.

## Prediction tasks

### Task A — Feature forecasting (first implementation)

For each chosen feature, predict the value at a specified future horizon from
a sequence of ordered acquisitions. Start with the persistence baseline
(the latest value), then compare moving averages or exponential smoothing,
autoregressive models and tabular ML models with lag features. Consider TCN or
LSTM models only if the number and independence of temporal sequences justify
their complexity.

Report MAE, RMSE and MASE where defined. A model should not be called useful
unless it improves on an appropriate baseline on held-out temporal targets.

### Task B — Early drift warning

Forecast deviations or monitor sequential residuals using methods such as
EWMA and CUSUM. Evaluate detection delay, false alarms per observation or
time unit, and missed events on pre-specified scenarios. Do not translate a
statistical warning into a hardware-failure prediction without labelled,
representative failure evidence.

### Task C — Threshold-crossing risk

Estimate whether a feature or monitored statistic will cross a pre-defined
engineering limit within a stated horizon. This task needs a documented
threshold and enough labelled positive and negative temporal examples. If
probabilities are reported, assess calibration and Brier score in addition to
precision-recall metrics.

### Task D — Uncertainty and out-of-domain handling

Evaluate prediction intervals using empirical coverage and average width on
held-out data. Explore conformal prediction only when its calibration
assumptions and split design are defensible. Add out-of-distribution checks for
unseen acquisition conditions and report when the model should abstain.

### Task E — Hybrid physics/statistics + AI

Compare a statistical reference, an AI-only predictor and a hybrid residual
model under the same splits. Acquisition settings can be model inputs when
measured and documented. Physics-informed claims require explicit physical
constraints or equations and evidence that they improve useful performance.

## First code milestone

`src/prediction/forecasting_evaluation.py` provides:

- MAE and RMSE;
- MASE when the training series has a non-zero naive scale;
- relative MAE versus a persistence baseline on matching targets;
- expanding-window rolling-origin evaluation;
- a persistence forecaster and target/origin bookkeeping.

This is an evaluation harness, not a trained predictive model. It evaluates a
single ordered feature series. It does not establish that public image
collections form a time series, perform group/session splitting automatically,
or validate a scanner physically.

## Evaluation protocol

1. Define the feature, acquisition order, prediction horizon and target before
   training.
2. Check data provenance, duplicates, missing metadata and acquisition lineage.
3. Split by time and, where appropriate, by session, phantom, device or
   independent sequence. Related frames or derived images must not leak across
   partitions.
4. Fit preprocessing, feature selection and model parameters using training
   data only.
5. Compare every candidate with persistence and another justified simple
   baseline.
6. Keep a final test period untouched until model and hyperparameters are fixed.
7. Report MAE/RMSE/MASE, uncertainty coverage and width, false-alert burden,
   and per-domain performance where data supports them.
8. Record software version, data identifiers, parameters, seeds, splits,
   metrics and failure cases.
9. Report null or negative results. Do not cherry-pick features or horizons.
10. Keep public, simulated and experimental results distinct.

## Planned implementation sequence

1. Stabilize the existing test and V&V workflows.
2. Audit current prediction, anomaly, fusion and state-engine modules before
   adding duplicate implementations.
3. Extend this evaluation harness to support grouped temporal splits and
   multi-feature reports, with tests for leakage and alignment.
4. Add baseline forecasting models and benchmark them on a clearly defined
   ordered dataset.
5. Add calibrated uncertainty and out-of-domain warnings.
6. Add residual-based drift warning and threshold-risk evaluation only after
   the task and labels are defined.
7. Integrate outputs into the Unified Twin State Engine with explicit evidence
   gates and auditable state transitions.
8. Add dashboard visualizations for observed values, forecasts, intervals,
   residuals, model disagreement and data provenance.
9. Run controlled simulation studies; report them as simulations.
10. Conduct authorized, traceable physical acquisitions when available, then
    evaluate generalization only within the protocol's evidence limits.

## Acceptance criteria for a predictive model

- Beats or meaningfully complements persistence on held-out temporal data.
- Has no detected temporal, group or lineage leakage.
- Has reproducible metrics and a documented evaluation split.
- Reports uncertainty honestly; no empirical ensemble spread is called a
  calibrated prediction interval without coverage evaluation.
- Exposes data-quality, domain-shift and model-disagreement warnings.
- Does not automatically update the nominal baseline from suspicious data.
- Does not claim clinical utility, physical fault diagnosis or remaining useful
  life without task-specific validation.


## Simple baseline benchmark implementation

`src/prediction/baseline_forecasters.py` exposes four fixed comparison methods:

- **Persistence**: repeat the latest observed value.
- **Trailing moving average**: repeat the mean of the latest configured window.
- **Simple exponential smoothing**: update a level with a preselected alpha and repeat it over the forecast horizon.
- **Linear trend**: fit least-squares linear regression to the observed prefix and extrapolate.

Use `compare_baseline_forecasters(...)` to evaluate these methods through the same expanding-window rolling-origin harness. Each method is scored on the same target indices and reports MAE, RMSE, MASE when defined, and relative MAE to persistence. Hyperparameters are fixed inputs; the helper does not tune them or automatically choose a winner. If tuning is later added, it must happen inside training-only temporal validation, leaving the final test period untouched.

These methods are deliberately transparent baselines. They should first be exercised on a controlled synthetic sequence with known behaviour, then on traceable ordered acquisitions from a single comparable domain. Do not use unordered BUSI or USSimAndSegm images as if they were a physical scanner time series. A synthetic linear-trend success checks implementation behaviour only; it does not validate scanner prediction.


## Multi-series benchmark reporting

`src/prediction/benchmark_report.py` aggregates per-model rolling-origin results across named, ordered feature series. Supply a mapping from series name to the per-model evaluations returned by `compare_baseline_forecasters(...)`.

Within each series, the report checks that candidate models share the same target indices, actual values, horizon, step and persistence predictions. It then reports per-series MAE/RMSE/MASE, relative MAE to persistence where the persistence error is non-zero, and wins/ties/losses against persistence. The primary cross-series summary is the macro mean of per-series relative MAE, so a high-magnitude feature does not dominate simply because of its units. Raw MAE/RMSE summaries are descriptive and should not be used to rank models across differently scaled features.

This helper does not validate provenance, infer acquisition order, or split sessions/devices. The caller must provide independent, comparable ordered sequences and avoid leakage from related acquisitions. Report the number of series and per-series results alongside any aggregate; a macro score from a small or homogeneous set is not evidence of generalization. A zero-error persistence baseline has no defined relative-error ratio and is counted as a tie when the candidate is equally accurate.


## Controlled synthetic drift validation

`src/validation/synthetic_drift_scenarios.py` provides deterministic,
seeded one-dimensional scenarios for checking the current EWMA/CUSUM monitor:

- **nominal**: stable observations without a known change point;
- **gradual_drift**: a gradual positive shift starting at a known index;
- **abrupt_shift**: a step change at a known index;
- **noisy_nominal**: nominal mean with increased noise, to probe sensitivity
  to variability changes.

`run_synthetic_drift_validation(...)` returns the sequential monitor output
and metrics. The report includes alarm count, false-alarm observation fraction
before the change (or across the full sequence when no change is specified),
whether an alarm occurred at or after the change, and detection delay in
observations. The false-alarm fraction is not a calibrated per-hour rate; a
real-time rate requires timestamps and a defined exposure period. Thresholds
must be pre-specified or calibrated on separate nominal scenarios, not tuned
on the same evaluation scenarios.

Use multiple seeds and vary noise, drift magnitude, onset and optional
seasonality before drawing conclusions. Report missed changes and false alarms,
not only successful detections. The generator is a software test fixture, not
a validated acoustic or scanner physics simulator. It does not establish
physical repeatability, device-fault prediction, clinical utility or real-world
false-alarm performance. Keep simulated outcomes separate from public-data
experiments and authorized physical acquisitions.


## Repeated synthetic drift evaluation grid

`src/validation/drift_validation_suite.py` runs the existing monitor across a
pre-specified grid of scenario families, random seeds, noise levels and drift
magnitudes. It reports per-trial metrics plus per-family detection rate, mean
false-alarm observation fraction, and mean/median detection delay among detected
trials. The report records the seeds and fixed monitor parameters for
reproducibility and JSON-friendly export.

The monitor thresholds are held fixed throughout one grid; this helper does
not optimize them on the evaluation scenarios. If threshold calibration is
introduced later, use a separate calibration set of nominal scenarios, freeze
the thresholds, and evaluate on independent seeds/scenarios. Include missed
changes and false alarms, not only detected changes. Detection delay is
conditional on detection and therefore must be interpreted alongside detection
rate. The false-alarm fraction is per observation, not per hour or per session.

The grid is a software-level robustness study on simplified synthetic
sequences. It is not a calibrated physical model of ultrasound propagation,
scanner electronics, probe wear, or acoustic phantoms. It cannot establish
physical repeatability, real-world alarm burden, clinical utility, or
hardware-failure prediction. Keep its results separate from public-image
experiments and traceable physical acquisitions.


## Nominal-only threshold calibration and independent evaluation

`src/validation/drift_threshold_calibration.py` separates threshold selection
from evaluation:

- `calibrate_drift_thresholds(...)` accepts nominal/no-change calibration
  sequences only, evaluates a pre-specified list of EWMA/CUSUM threshold pairs,
  and selects the least conservative supplied pair whose **sequence-level**
  false-alarm rate is at or below the target. Sequence-level false-alarm rate
  means the proportion of nominal sequences with at least one alarm; it is not
  a per-hour or per-acquisition guarantee.
- At least five calibration sequences with distinct seeds are required. If no
  candidate meets the target, calibration fails explicitly rather than
  silently choosing an unsuitable threshold.
- `evaluate_calibrated_thresholds(...)` freezes the selected parameters and
  rejects evaluation scenarios whose seeds overlap with calibration seeds.
  It reports nominal sequence false-alarm rate, change detection rate, mean
  false-alarm observation fraction, and detection delay among detected changes.

Choose the candidate grid and target before looking at evaluation results. The
small empirical calibration set does not establish a statistical guarantee of
future false-alarm rates, particularly under temporal dependence or domain
shift. This is a synthetic software-validation protocol only; it does not
validate ultrasound hardware, acoustic physics, clinical utility, or failure
prediction. For physical claims, repeat the protocol on authorized, traceable
acquisitions with independent sessions and a pre-registered operating target.


### Confidence-aware threshold selection

The nominal calibration helper reports a Wilson confidence interval for each
candidate's sequence-level false-alarm proportion. A candidate is eligible
only when the **upper confidence bound**, not just the observed point estimate,
is at or below the pre-specified target. This helps expose the uncertainty
caused by small calibration samples. The interval relies on independent
sequence-level Bernoulli outcomes and should not be interpreted as a guarantee
under temporal dependence or domain shift. If no candidate qualifies, collect
more independent nominal sequences or revise the operating target and candidate
grid before inspecting the independent evaluation results. Do not tune against
the evaluation set.
