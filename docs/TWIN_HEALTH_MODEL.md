# Twin Health & Evidence Confidence Model

## Purpose

The Digital Twin exposes two separate engineering indicators:

1. **Twin Health Index** — a bounded condition indicator derived from observable
   image quality and multivariate statistical deviation.
2. **Evidence Confidence** — a bounded score describing the maturity and
   completeness of the evidence supporting the assessment.

They are deliberately separated.

> Health is not confidence, and confidence is not probability of failure.

## Health model

The current transparent model is:

- anomaly component = 100 × exp(-D / D_ref)
- health index = 0.60 × quality score + 0.40 × anomaly component

D is the Mahalanobis distance and D_ref is configurable, with a default of
3.0. The result is clipped to [0, 100].

### Engineering states

| Health Index | State |
|---:|---|
| 85–100 | NOMINAL |
| 70–84.999 | WATCH |
| 50–69.999 | EARLY_DRIFT |
| 0–49.999 | HIGH_DEVIATION |

These states describe the observed Digital Twin state. They do not establish
that a physical ultrasound system has failed.

## Evidence confidence

Evidence confidence combines:

- baseline strength: 40%;
- feature coverage: 30%;
- provenance/source maturity: 30%.

Baseline strength saturates at 30 observations. Feature coverage compares
available features with the expected feature count.

For the current dashboard prototype, source-maturity guidance is:

- real experimental acquisition: 100;
- public reference: 80;
- controlled simulation: 60.

These values are engineering defaults and must be revised when a formal
experimental validation protocol establishes source-specific evidence quality.

## Explainability

The assessment reports dominant evidence such as:

- quality;
- statistical_deviation;
- limited_baseline;
- source_maturity.

This prevents the single health number from hiding the underlying evidence.

## V&V boundary

The health layer is covered by unit tests for:

- bounded outputs;
- deterministic behavior;
- decreasing anomaly component with increasing statistical distance;
- evidence-confidence degradation with weaker evidence;
- state classification.

It does **not** validate physical ultrasound-system behavior.

The following claims remain outside the scope of this model:

- confirmed hardware failure;
- clinical diagnosis;
- failure probability;
- remaining useful life (RUL);
- physical acceptance thresholds.

Those require dedicated experimental characterization and longitudinal
device-specific validation.
