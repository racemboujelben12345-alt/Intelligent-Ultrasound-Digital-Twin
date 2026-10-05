"""Counterfactual Digital Twin engine.

Creates controlled virtual alternatives from an observed acquisition and
compares their Digital Signatures. Counterfactuals are simulation evidence,
not measurements of a real device.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from src.image_analysis.digital_signature import FEATURE_ORDER, DigitalSignature, build_digital_signature_from_image
from src.simulation.degradation import apply_degradation, DegradationType


@dataclass(frozen=True)
class CounterfactualResult:
    scenario: str
    severity: float
    observed_source: str
    observed_signature_version: str
    counterfactual_signature_version: str
    changed_features: tuple[tuple[str, float], ...]
    l2_signature_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


def compare_signatures(
    observed: DigitalSignature,
    counterfactual: DigitalSignature,
) -> CounterfactualResult:
    """Compare two signatures using the canonical feature vector."""
    a = observed.to_vector()
    b = counterfactual.to_vector()
    delta = b - a
    names = FEATURE_ORDER
    changed = tuple(
        (name, float(value))
        for name, value in zip(names, delta)
        if abs(float(value)) > 1e-12
    )
    return CounterfactualResult(
        scenario="comparison",
        severity=0.0,
        observed_source=observed.source,
        observed_signature_version=observed.signature_version,
        counterfactual_signature_version=counterfactual.signature_version,
        changed_features=changed,
        l2_signature_delta=float(np.linalg.norm(delta)),
    )


def run_counterfactual(
    image: np.ndarray,
    observed: DigitalSignature,
    degradation_type: DegradationType | str,
    severity: float,
    *,
    seed: int = 42,
    params: dict | None = None,
) -> CounterfactualResult:
    """Generate one reproducible virtual scenario and compare signatures."""
    if not 0.0 <= float(severity) <= 1.0:
        raise ValueError("severity must be in [0, 1].")

    result = apply_degradation(
        image,
        degradation_type,
        float(severity),
        seed=seed,
    )
    counterfactual_image = result.image
    signature = build_digital_signature_from_image(
        counterfactual_image,
        source="simulated",
        params=params,
    )
    comparison = compare_signatures(observed, signature)
    return CounterfactualResult(
        scenario=str(getattr(degradation_type, "value", degradation_type)),
        severity=float(severity),
        observed_source=observed.source,
        observed_signature_version=observed.signature_version,
        counterfactual_signature_version=signature.signature_version,
        changed_features=comparison.changed_features,
        l2_signature_delta=comparison.l2_signature_delta,
    )
