# Sequential drift monitoring

## Purpose

`src/prediction/sequential_drift.py` provides two statistical early-warning
baselines for an ordered feature sequence:

- **EWMA** smooths standardized deviations and can highlight persistent changes.
- **CUSUM** accumulates positive or negative standardized evidence and can
  highlight sustained shifts.

The module is deliberately model-agnostic. It consumes one numeric reference
sample and a later ordered sequence for one feature. It does not identify the
physical cause of a shift, predict hardware failure, or establish clinical
meaning.

## Reference and scaling

The reference centre is the median. Scale is estimated from
(1.4826 \times MAD), with sample standard deviation as a fallback when MAD is
numerically zero. If the reference is effectively constant, the module raises
an error rather than fabricating a standardized scale.

## Threshold calibration

The default values are starting parameters for software demonstrations, not
universal ultrasound QA thresholds:

- EWMA smoothing factor: 0.2
- EWMA absolute threshold: 3.0 standardized units
- CUSUM allowance: 0.5 standardized units
- CUSUM threshold: 5.0 accumulated units

Before interpreting warnings, calibrate thresholds using independent nominal
sequences that represent the intended acquisition cadence and settings.
Report false alarms per acquisition, session or unit time, and evaluate
detection delay and missed detections on pre-specified changes. If the data
does not support this calibration, label thresholds as exploratory.

## Recommended validation

1. Test deterministic cases with no shift and with controlled positive and
   negative shifts.
2. Use separate reference and monitoring sequences.
3. Test sensitivity to outliers, scale changes and non-stationarity.
4. Calibrate thresholds on training/validation sequences only.
5. Freeze thresholds before evaluating held-out sequences.
6. Report simulated perturbation experiments separately from real
   acquisitions.
7. Compare EWMA/CUSUM warnings with the Digital Twin's statistical and AI
   evidence; preserve disagreement as information rather than hiding it.

A sequential warning is evidence of statistical deviation relative to a
reference. It is not proof of hardware failure. Changes in probe, settings,
phantom, operator, acquisition protocol or image preprocessing can also change
the measured feature distribution.
