# Unified Evidence Fusion — Verification & Validation

## Purpose

This layer verifies that the Digital Twin can combine statistical, AI, physics, causal, and optional counterfactual evidence without presenting simulation outputs as proof of a physical SCAN A fault.

## Evidence directions

All production fusion inputs use the same convention:
- `0`: no anomaly evidence
- `1`: strong anomaly evidence

Therefore physical consistency is converted as:
`physics_anomaly = 1 - physical_consistency_score`

A high physical consistency score is evidence against an anomaly, not evidence for one.

## Controlled simulation V&V

`src/validation/evidence_vv.py` provides a reproducible harness:
1. Build a reference digital signature.
2. Apply one controlled digital degradation.
3. Recompute the signature.
4. Calculate the signature delta.
5. Rank qualitative causal hypotheses.
6. Fuse the available evidence.
7. Record disagreement and provenance through the simulation scenario.

Supported simulations are deliberately treated as digital perturbations. They do not establish that the same mechanism occurs in a physical scanner.

## Tests

The V&V tests verify:
- deterministic output for the same image/scenario/seed;
- non-zero signature response to controlled perturbation;
- severity traceability;
- bounded causal agreement, fusion, and disagreement scores.

## Validation hierarchy

This layer provides software and simulation verification. It does not replace:
1. repeated baseline acquisitions;
2. phantom measurements;
3. probe-specific characterization;
4. authorized controlled perturbations;
5. comparison against independent QA measurements.

Only those higher-level experiments can support SCAN A-specific physical claims.

## Important limitation

The current causal engine uses qualitative feature-direction hypotheses. A high causal agreement score means that observed feature directions are consistent with a hypothesis. It is not a probability that the mechanism is the true cause.

Likewise, the unified fusion score is an engineering evidence score, not a calibrated probability of equipment failure and not a clinical diagnostic score.