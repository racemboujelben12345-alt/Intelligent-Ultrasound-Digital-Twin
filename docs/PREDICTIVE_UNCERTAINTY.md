# Predictive uncertainty: split-conformal intervals

## Goal

A point forecast alone hides the range of errors the model can make. The
module `src/prediction/uncertainty_calibration.py` adds a transparent
split-conformal baseline around an already fixed point predictor.

It provides:
- calibration of a symmetric interval radius from absolute calibration residuals;
- finite-sample rank calculation `ceil((n + 1) * (1 - alpha))`, clipped to the available calibration sample count;
- empirical test coverage, average/median interval width, and lower/upper miss rates.

## Required data separation

Use three distinct roles where data volume permits:

1. **Training**: fit the point predictor and all preprocessing.
2. **Calibration**: calculate the interval radius; do not refit or tune the predictor here.
3. **Test**: evaluate coverage and width once the pipeline is fixed.

Do not calibrate and evaluate on the same observations. For a time series, a strictly later test segment is preferable to random shuffling, but chronological order alone does not guarantee conformal validity when residuals are dependent or the distribution drifts.

## Interpretation and limitations

The target nominal coverage is not a promise that every future window will achieve that coverage. Standard split-conformal coverage relies on exchangeability between calibration and test examples. Sequential ultrasound acquisitions can be autocorrelated; a change in settings, probe, phantom, device or preprocessing can also shift the residual distribution. Therefore report empirical coverage and width by time/session/domain where data volume permits, and treat results under distribution shift as exploratory.

A very wide interval can attain high coverage but may be useless. Always report coverage together with interval width and compare against a simple reference. Do not describe ensemble spread as a calibrated interval unless it has been evaluated this way. This method quantifies predictive uncertainty; it does not explain the physical cause of an observed change or predict hardware failure.

## Next validation steps

- Evaluate multiple nominal coverage levels (for example, 80%, 90%, 95%) only with enough independent calibration data.
- Plot empirical coverage versus nominal coverage on held-out sequences.
- Inspect interval width and lower/upper misses by feature and acquisition domain.
- Stress-test simulated drift and domain shifts, clearly labelling them as simulations.
- Freeze model and calibration choices before the final held-out evaluation.
