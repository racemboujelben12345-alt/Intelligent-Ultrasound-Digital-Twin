# AI Intelligence Architecture

## Objective
The AI layer is an engineering intelligence layer over the canonical ultrasound Digital Signature. It is deliberately separated from physical diagnosis.

## Multi-model architecture

Ultrasound Image
      ↓
Digital Signature (13 features)
      ↓
AI ENSEMBLE: multiple Isolation Forest models
      ↓
Anomaly Score + Anomaly Probability + Ensemble Agreement + Evidence Confidence
      ↓
NOMINAL / WATCH / ANOMALY

Multiple random seeds reduce dependence on a single stochastic model realization.

## Supervised mode
When validated engineering labels exist, a Random Forest classifier can be trained on the same canonical feature contract. It exposes predicted class, class probabilities and feature importance.
Labels must correspond to defensible engineering states or controlled simulation classes. They must not be presented as clinical diagnosis or physical-failure labels unless independently validated.

## Predictive AI
A Random Forest regression ensemble can model a future engineering indicator from longitudinal observations. It exposes an ensemble mean and an empirical 5th–95th percentile interval.
This interval represents model disagreement, not a calibrated statistical coverage guarantee. Calibration against held-out temporal data is required before claiming coverage.

## AI + statistical fusion
Digital Signature → Mahalanobis statistical detector
Digital Signature → AI ensemble
Statistical evidence + AI evidence → Digital Twin State → Twin Health Index
The AI engine does not replace the statistical baseline. Agreement between independent methods is stronger evidence than either model alone; disagreement is itself useful evidence and should reduce confidence.

## Validation strategy
AI validation must include:
1. acquisition-family leakage prevention
2. train/validation/test separation
3. temporal holdout for longitudinal prediction
4. seed sensitivity
5. class imbalance analysis
6. false-positive and false-negative reporting
7. calibration evaluation when probabilities are presented
8. robustness across public and experimental sources
9. controlled degradation sensitivity
10. external validation where data permit

## Scientific boundary
An AI output such as ANOMALY with high model probability means strong computational anomaly evidence under the trained/reference distribution.
It does not mean a hardware-failure probability, physical component defect probability, clinical diagnosis, or remaining-useful-life estimate.
Physical interpretation requires device-specific experimental validation.