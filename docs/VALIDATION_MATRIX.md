# Validation Matrix

| Question | Dataset | Method | Metric | Meaning |
|---|---|---|---|---|
| Feature stability | repeated nominal | repeated extraction | CV / ICC | measurement/software stability |
| Baseline calibration | nominal + holdout | statistical detector | false-positive rate | nominal calibration |
| Degradation sensitivity | synthetic | severity sweep | detection + monotonic trend | response to injected perturbation |
| Generalization | public train/holdout | fixed split | balanced metrics | methodological robustness |
| Physical relevance | experimental | repeated acquisition | repeatability + agreement | experimental evidence |
| Device transfer | device-specific | external validation | external performance | case-study evidence |

## Reporting rule
Never combine public, synthetic and experimental results into one unlabeled score. Every result identifies its source layer.

## Synthetic experiment
Select nominal parents, generate severity levels, preserve parent IDs, use fixed seeds, compute the complete feature vector, verify expected feature directions, and report failure cases.

## Experimental experiment
Acquire repeated nominal observations before controlled variation. Unsafe or unapproved physical perturbations are not performed; simulation is used instead.


## Automated V&V gate

The repository now provides a reproducible audit layer at
`src/validation/audit.py` and a command-line runner at
`scripts/run_vv_audit.py`.

The audit checks:

1. canonical acquisition contract;
2. unique acquisition identifiers;
3. complete simulation lineage;
4. deterministic seeded simulation;
5. non-trivial sensitivity to degradation severity;
6. parent-family leakage between train and holdout.

Example:

```bash
python scripts/run_vv_audit.py --n 10
```

### Interpretation rule

A PASS means that the software/data-contract property was verified.
It does **not** mean that a simulated degradation corresponds to a physical
ultrasound-system failure.

Physical validity still requires experimental reference measurements and
device-specific validation.
