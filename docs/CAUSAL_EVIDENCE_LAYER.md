# Causal Evidence Layer

## Purpose

The causal evidence layer ranks documented engineering hypotheses by how
consistent the observed Digital Signature movement is with their expected
feature directions.

It is intentionally **not a diagnostic engine**.

The reasoning chain is:

    observed acquisition
        -> digital signature
        -> temporal/baseline delta
        -> directional evidence
        -> causal hypothesis ranking
        -> evidence fusion
        -> Digital Twin state

## Evidence categories

For each hypothesis, every available feature is classified as:

- **supported**: observed movement agrees with the documented direction;
- **contradicted**: observed movement opposes the documented direction;
- **unavailable**: the feature was not observed or remained within tolerance.

The agreement score is:

    supported / (supported + contradicted)

This is an engineering evidence score, not a probability of hardware failure.

## Scientific boundary

The causal mappings remain hypotheses until validated on the actual SCAN A
system, probe, acquisition protocol and controlled phantom experiments.

A high agreement score means:

> the observed feature movement is directionally consistent with the
> documented mechanism.

It does **not** mean:

> the mechanism is proven to be the physical cause.

## Recommended validation

1. Acquire repeated baseline images using a fixed phantom and fixed protocol.
2. Measure repeatability of each signature feature.
3. Apply one controlled perturbation at a time when technically and safely
   authorized.
4. Measure feature deltas.
5. Compare measured directions with the hypothesis map.
6. Record false-support and contradiction cases.
7. Only then calibrate scanner/probe-specific causal weights.

## Integration

The layer is designed to sit after baseline/drift analysis and before final
evidence fusion. It can therefore provide a second, interpretable evidence
stream alongside statistical and AI anomaly scores.
