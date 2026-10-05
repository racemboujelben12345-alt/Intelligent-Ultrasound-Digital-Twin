"""Multi-source evidence fusion for the ultrasound Digital Twin.

Combines independent evidence streams without interpreting the result as a
clinical diagnosis or a calibrated probability of hardware failure.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class EvidenceFusionResult:
    statistical: float
    ai: float
    physics: float
    causal: float
    counterfactual: float
    fusion_score: float
    disagreement: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def fuse_evidence(
    *,
    statistical: float,
    ai: float,
    physics: float,
    causal: float,
    counterfactual: float,
    weights: dict[str, float] | None = None,
) -> EvidenceFusionResult:
    """Fuse normalized evidence scores with a weighted mean and disagreement."""
    values = {
        "statistical": float(statistical),
        "ai": float(ai),
        "physics": float(physics),
        "causal": float(causal),
        "counterfactual": float(counterfactual),
    }
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values.values()):
        raise ValueError("All evidence scores must be finite and in [0,1].")

    if weights is None:
        weights = {name: 1.0 for name in values}
    if set(weights) != set(values):
        raise ValueError("weights must contain exactly the five evidence sources.")
    if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights.values()):
        raise ValueError("Weights must be finite and non-negative.")

    total_weight = sum(float(w) for w in weights.values())
    if total_weight <= 0:
        raise ValueError("At least one evidence weight must be positive.")

    fusion = sum(values[k] * float(weights[k]) for k in values) / total_weight
    mean = sum(values.values()) / len(values)
    variance = sum((v - mean) ** 2 for v in values.values()) / len(values)
    disagreement = min(1.0, math.sqrt(variance) * 2.0)

    # Confidence rises with evidence level and falls with disagreement.
    confidence = max(0.0, min(1.0, fusion * (1.0 - disagreement)))

    return EvidenceFusionResult(
        statistical=values["statistical"],
        ai=values["ai"],
        physics=values["physics"],
        causal=values["causal"],
        counterfactual=values["counterfactual"],
        fusion_score=float(fusion),
        disagreement=float(disagreement),
        confidence=float(confidence),
    )
