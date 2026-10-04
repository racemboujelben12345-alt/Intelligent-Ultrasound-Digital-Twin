# Expert Architecture — Intelligent Ultrasound Digital Twin

## Objective
Build a traceable computational representation of the observable imaging behaviour of an ultrasound system, monitor deviations from a nominal reference state, and study controlled virtual degradations.

## Architecture contract
1. Acquisition — provenance, ordering and source selection.
2. Image/signal analysis — convert an acquisition into measurable observables.
3. Digital Signature — stable feature schema and versioning.
4. Statistical Baseline — model nominal variability.
5. Quality Assessment — quantify image-quality dimensions.
6. Anomaly Detection — compare a new signature to the nominal model.
7. Digital Twin State — represent the current observed state.
8. Drift Monitoring — detect persistent temporal change.
9. Simulation — inject controlled observable degradations.
10. AI Intelligence — ensemble anomaly evidence, validated-state classification and predictive uncertainty.
11. Intelligence Fusion — combine statistical deviation, AI evidence and quality with explicit weights and disagreement penalty.
12. Drift/Prediction — quantify temporal direction without claiming physical RUL.
13. Validation — compare outputs with explicit ground truth.
13. Reporting/Dashboard — present results without changing analytical decisions.

## Why this is a Digital Twin prototype
The project is not simply image classification. Each ultrasound acquisition is treated as an observation of an imaging-system process. The Digital Signature provides a compact state vector, the baseline represents nominal behaviour, and history/drift layers monitor its evolution.

The current implementation is a **data-driven Digital Twin prototype**. It can later be strengthened with device telemetry, probe identity, gain/TGC, depth, frequency, focus, phantom/reference measurements and other equipment metadata.

## Data hierarchy
### Level A — public data
Use public ultrasound datasets to develop and benchmark image-analysis and statistical components.

### Level B — controlled simulation
Apply reproducible image-domain degradations and verify coherent responses.

### Level C — experimental ultrasound data
Acquire repeated measurements from a physical ultrasound system when available and use them for calibration and repeatability validation.

### Level D — device-specific validation
If data from a named device becomes available, treat it as a case study/validation layer rather than redefining the generic Digital Twin around one model.

Software degradation is not the same thing as a confirmed hardware fault.

## Recommended metadata
timestamp, device identifier, probe identifier, operator/session, preset, frequency, gain, depth, focus, dynamic range, phantom/sample, sequence number, environmental notes and file checksum.

## Validation strategy
Report feature stability, repeatability, false-positive rate, anomaly detection performance, response versus simulated degradation severity, temporal consistency and computational runtime.

Never claim physical-fault detection unless the ground truth represents that physical fault.

## Future extensions
Metadata conditioning, ROI-aware ultrasound features, A-scan signal support, multivariate control charts, calibrated uncertainty, domain-shift monitoring, fault-scenario library, experiment tracking and model/version registry.


## AI intelligence contract

The AI layer is a first-class engineering component, not a decorative classifier.

### Unsupervised
An Isolation Forest ensemble operates on the canonical Digital Signature and
reports anomaly evidence, ensemble agreement and confidence.

### Supervised
A Random Forest classifier is available only when validated engineering labels
or controlled simulation classes exist.

### Predictive
A Random Forest regression ensemble estimates a validated temporal engineering
target and exposes an empirical P05/P95 interval.

### Explainability and governance
AI feature sensitivity is estimated by leave-one-feature-out perturbation.
Each fitted task has a model card with version, feature contract, source,
hyperparameters, seeds and timestamp. Inference records can hash the input
signature for auditability.

### Evidence fusion
Statistical Mahalanobis evidence, AI anomaly evidence and image-quality evidence
are fused transparently. Disagreement between statistical and AI signals reduces
confidence. The fused score is an engineering evidence index and is not a
physical-failure probability.

## End-to-end contract

`Acquisition → Provenance → Image Analysis → Digital Signature → Quality →
Statistical Baseline → AI Ensemble → Evidence Fusion → Twin State →
History/Drift → Prediction → Simulation → V&V → Reporting`

No downstream dashboard component is allowed to silently redefine the analytical
state. Analytical decisions remain in `src/`; the dashboard is presentation.
