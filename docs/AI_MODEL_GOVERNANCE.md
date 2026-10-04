# AI Model Governance

## Purpose

The AI layer is an engineering intelligence component of the Intelligent Ultrasound Digital Twin. It provides anomaly evidence, validated-state classification when labels exist, and predictive estimates with empirical ensemble uncertainty.

It does **not** establish physical device failure, clinical diagnosis, or calibrated failure probability.

## Model families

| Task | Model | Output | Evidence boundary |
|---|---|---|---|
| Unsupervised anomaly | Isolation Forest ensemble | anomaly evidence + ensemble agreement | reference-distribution deviation |
| Supervised state | Random Forest classifier | class + probability | only validated labels/classes |
| Prediction | Random Forest regression ensemble | estimate + P05/P95 empirical interval | validated temporal target |

## Provenance

Every fitted task records:
- model version;
- task;
- feature contract;
- training sample count;
- declared training source;
- random seeds;
- hyperparameters;
- UTC fit timestamp;
- SHA-256 feature-contract hash.

An anomaly inference can additionally be represented by an AIInferenceRecord containing acquisition ID, source, input hash, model version and output values.

## Explainability

Unsupervised anomaly explanation uses leave-one-feature-out sensitivity: one Digital Signature feature is replaced by its reference mean and the change in anomaly evidence is measured. This is a **sensitivity attribution**, not a causal explanation.

Supervised Random Forest exposes feature importance for engineering-state classification.

## Validation roadmap

1. acquisition-family/group split to prevent parent/simulation leakage;
2. temporal holdout;
3. cross-source evaluation;
4. severity sensitivity under controlled degradation;
5. seed stability;
6. false-positive / false-negative analysis;
7. calibration assessment for supervised probabilities;
8. prediction error and interval coverage for temporal targets;
9. ablation: statistical-only vs AI-only vs fused intelligence.

Until these experiments are executed on representative data, AI confidence and anomaly scores must be described as model evidence rather than calibrated probabilities.

## Scientific boundary

Evidence maturity remains hierarchical:

software validity → simulation validity → public benchmark validity → experimental validity → operational/longitudinal validity

Passing an AI software test does not imply that an ultrasound system has been physically validated.
