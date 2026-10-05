# Physics + AI Architecture

## Purpose

The Digital Twin is an engineering monitoring system for an ultrasound scanner. It combines image-derived evidence, ultrasound physics, statistical deviation, machine-learning anomaly evidence, temporal drift and provenance.

## Evidence layers

1. **Acquisition / provenance**
   - source: experimental, public reference or simulated
   - acquisition metadata: frequency, depth, probe/preset when available
   - lineage and reproducibility

2. **Image signature**
   - intensity distribution
   - contrast and entropy
   - gradients and edge density
   - sharpness
   - speckle proxy
   - uniformity

3. **Physics-aware signature**
   - wavelength: lambda = c / f
   - nominal axial resolution proxy: lambda / 2
   - relative attenuation/depth profile proxy
   - axial intensity slope
   - depth uniformity
   - near-field energy ratio
   - metadata/physics consistency

The implementation uses a conventional soft-tissue reference speed of 1540 m/s when scanner metadata does not provide a value. Frequency and depth should be supplied by the acquisition protocol whenever possible.

## AI layer

The existing AI engine uses an ensemble of Isolation Forest models for unsupervised anomaly evidence. Its output is an empirical anomaly rank/vote evidence, **not a calibrated probability of hardware failure**.

The feature space now includes both image and physics-aware descriptors. This prevents the AI layer from relying exclusively on generic computer-vision statistics.

Future validated datasets can additionally train:
- supervised equipment-state classification;
- temporal predictive models;
- model-calibrated uncertainty;
- physics-informed representation learning.

## Evidence fusion

The unified intelligence score combines:

- statistical Mahalanobis evidence;
- AI anomaly evidence;
- acquisition quality evidence;
- physics-inconsistency evidence.

Physics evidence is defined as:

physical_evidence = 1 - physical_consistency_score

Therefore:
- 0 = physically consistent;
- 1 = stronger physics/metadata inconsistency evidence.

The fusion weights are explicit engineering defaults and must be recalibrated on representative validation data before any operational deployment.

## Temporal layer

The Digital Twin tracks:
- Mahalanobis distance and D²;
- quality trajectory;
- EWMA/CUSUM drift signals;
- persistent drift;
- trend direction.

The temporal layer describes statistical trajectory. It does not claim to predict a physical failure date or remaining useful life.

## Validation strategy

A professional validation campaign should progressively use:

1. repeated baseline acquisitions under controlled settings;
2. phantom-based QA measurements;
3. controlled parameter perturbations;
4. known image-quality degradations;
5. independent sessions/days/operators;
6. held-out validation;
7. calibration and uncertainty evaluation.

For ultrasound QA, relevant engineering quantities include uniformity, depth of penetration, axial/lateral resolution and distance accuracy. These should be measured with an appropriate tissue-mimicking phantom when the project moves from image-only proxies to physical equipment validation.

## Scientific boundary

Image-derived attenuation and resolution quantities in this software are labelled as proxies unless the acquisition contains calibrated spatial and acoustic metadata. They must not be presented as absolute acoustic measurements without calibration.

The system is intended for equipment monitoring and research. It is not a clinical diagnostic system and does not establish a hardware fault by itself.
