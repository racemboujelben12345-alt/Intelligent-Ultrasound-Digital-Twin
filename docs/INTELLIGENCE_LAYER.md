# Intelligent Ultrasound Digital Twin — Intelligence Layer

## Purpose

The intelligence layer converts a single analysed ultrasound acquisition into a
structured engineering assessment without conflating statistical evidence with
physical diagnosis.

## State hierarchy

```
Digital Signature
      ↓
Statistical Detection
      ↓
Digital Twin State
      ├── quality_score
      ├── Mahalanobis distance
      ├── feature contributions
      └── temporal history
      ↓
Twin Health Assessment
      ├── Health Index
      └── Evidence Confidence
      ↓
Drift / Trend / Prediction
```

### Two independent indicators

**Twin Health Index** is a bounded engineering condition indicator based on
observable image quality and multivariate statistical deviation.

**Evidence Confidence** describes how mature and complete the supporting
evidence is. It is intentionally independent from health.

Neither value is:
- a probability of failure;
- a clinical score;
- proof of a hardware defect;
- a remaining-useful-life estimate.

## Source-aware evidence

Current engineering defaults are:

| Source | Evidence score |
|---|---:|
| experimental | 100 |
| public | 80 |
| simulated | 60 |
| legacy SCAN A | 100 |

These are governance defaults, not experimentally validated device-specific
weights. They must be recalibrated when controlled physical acquisitions are
available.

## Controlled simulation lineage

Every registered virtual perturbation contains:

- parent acquisition ID;
- scenario name;
- degradation family;
- severity;
- random seed;
- simulation version;
- SHA-256 hash of the generated image.

This creates an auditable chain:

```
parent acquisition
      ↓
registered scenario
      ↓
severity + seed
      ↓
simulation version
      ↓
derived image
      ↓
output hash
```

The same parent, scenario, severity, seed and simulator version must reproduce
the same image exactly.

## Validation boundary

Software V&V can establish that the implementation is deterministic, bounded,
traceable and sensitive to controlled digital perturbations.

It cannot establish that a particular blur, noise or contrast perturbation
corresponds quantitatively to a real ultrasound hardware fault.

Physical validation therefore remains a separate maturity level requiring
controlled acquisitions, repeatability analysis, device-specific baselines and
authorized perturbation protocols.

## Engineering maturity

The project should be presented using the following evidence hierarchy:

1. **Software validity** — implementation behaves as specified.
2. **Simulation validity** — controlled digital perturbations produce expected
   computational responses.
3. **Public benchmark validity** — methodology generalizes to independent public
   ultrasound data.
4. **Experimental validity** — repeated acquisitions from the real system
   characterize device-specific behavior.
5. **Operational validity** — longitudinal monitoring demonstrates usefulness in
   the intended engineering workflow.

The project must not claim level 4 or 5 from levels 1–3 alone.
