# Final V&V and Robustness Workflow

## One-command audit

```bash
python scripts/run_final_vv.py --n 10 --repeats 20 --seed 42
```

The final audit combines:
- software/data-contract audit;
- deterministic controlled-degradation checks;
- severity-response verification;
- repeated-acquisition noise-floor estimation;
- sensitivity/robustness matrix;
- robustness margin relative to `3 × repeatability noise`;
- JSON artifacts under `outputs/vv/`.

## Interpretation

`robustness_margin_l2 > 0` means the simulated signature change exceeds the selected repeatability noise threshold. It is engineering evidence that the digital perturbation is distinguishable from the simulated repeatability floor.

## Scientific boundary

The audit verifies software behavior and controlled simulations. It does not prove a hardware fault, scanner degradation, clinical effect, or SCAN A-specific mechanism. SCAN A conclusions require traceable repeated acquisitions, phantom QA, probe-specific characterization, and authorized controlled perturbations.

## Final architecture

```text
SCAN A / ultrasound acquisition
        ↓
Provenance + metadata
        ↓
Digital Signature + physics
        ↓
Statistical Baseline
        ↓
Mahalanobis anomaly evidence
        ↓
AI ensemble evidence
        ↓
Causal hypothesis ranking
        ↓
Evidence Fusion
        ↓
Unified Twin State
        ↓
Temporal Drift / Trend
        ↓
Repeatability Noise Floor
        ↓
Sensitivity + Robustness V&V
        ↓
Engineering Report / Dashboard
```