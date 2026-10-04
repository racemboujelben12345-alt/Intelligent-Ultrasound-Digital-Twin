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
