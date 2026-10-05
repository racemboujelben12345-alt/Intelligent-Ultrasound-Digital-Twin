# Physics-to-Image Causal Engine

## Purpose

The causal map formalizes engineering hypotheses connecting a possible physical/acquisition mechanism to observable image changes and Digital Signature features.

It is used for:

- simulation scenario design;
- explainability;
- evidence organization;
- validation planning;
- future counterfactual Digital Twin experiments.

It is **not** a diagnostic rule and does not prove a component failure.

## Causal chain

```
mechanism / acquisition condition
        ↓
physical effect
        ↓
expected image manifestation
        ↓
measurable signature features
        ↓
statistical / AI evidence
        ↓
Twin-state interpretation
```

## Current hypotheses

| Mechanism | Physical effect | Main observables |
|---|---|---|
| Focus shift / defocus | Beam-width / focal-response change | Sharpness, edges, gradients |
| Electronic noise increase | Noise floor increase | Intensity dispersion, CV, speckle proxy |
| Sensitivity / element response change | Local echo response change | Uniformity, intensity, edges |
| Attenuation-like depth roll-off | Stronger depth-dependent signal decrease | Depth slope, attenuation proxy, depth uniformity |
| Pulse / bandwidth change | Effective spatial pulse length change | Axial-resolution proxy |
| Acquisition timing inconsistency | PRF/depth inconsistency | PRF margin, physical consistency |

## Scientific boundary

The direction of an effect is a hypothesis until it is verified for the actual SCAN A system, probe, phantom and acquisition protocol.

For example, a reduction in image sharpness can have several causes: focus, motion, processing, beamforming, compression or other acquisition conditions. Therefore the causal map must be combined with metadata, repeated acquisitions and independent evidence.

## Validation path

1. Validate each mapping with controlled software perturbations.
2. Measure repeatability on a fixed phantom.
3. Repeat across sessions and probes.
4. Compare predicted directional changes with measured changes.
5. Only then promote a mapping from a hypothesis to a validated engineering relationship.
