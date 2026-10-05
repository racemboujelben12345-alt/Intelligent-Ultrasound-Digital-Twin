"""Causal evidence scoring for Digital Twin observations.

The scorer compares observed feature movement against qualitative causal
hypotheses. It is evidence ranking, not fault diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isfinite

from src.physics.causal_mapping import expected_feature_directions


@dataclass(frozen=True)
class CausalEvidence:
    mechanism: str
    agreement_score: float
    supported_features: tuple[str, ...]
    contradicted_features: tuple[str, ...]
    unavailable_features: tuple[str, ...]
    evidence_level: str = "hypothesis"

    def to_dict(self) -> dict:
        return asdict(self)


def _sign(delta: float, tolerance: float) -> int:
    if abs(delta) <= tolerance:
        return 0
    return 1 if delta > 0 else -1


def score_causal_evidence(
    mechanism: str,
    observed_delta: dict[str, float],
    *,
    tolerance: float = 1e-8,
) -> CausalEvidence:
    """Score directional agreement for one mechanism."""
    if tolerance < 0 or not isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative.")

    directions = expected_feature_directions(mechanism)
    if not directions:
        raise KeyError(f"No directional hypothesis for: {mechanism}")

    supported = []
    contradicted = []
    unavailable = []

    for feature, direction in directions.items():
        if feature not in observed_delta:
            unavailable.append(feature)
            continue

        value = float(observed_delta[feature])
        if not isfinite(value):
            raise ValueError(f"Non-finite delta for {feature}.")

        sign = _sign(value, tolerance)
        if sign == 0:
            unavailable.append(feature)
            continue

        expected = 1 if direction == "increase" else -1
        if direction == "change":
            supported.append(feature)
        elif sign == expected:
            supported.append(feature)
        else:
            contradicted.append(feature)

    total = len(supported) + len(contradicted)
    score = 0.0 if total == 0 else len(supported) / total

    return CausalEvidence(
        mechanism=mechanism,
        agreement_score=float(score),
        supported_features=tuple(supported),
        contradicted_features=tuple(contradicted),
        unavailable_features=tuple(unavailable),
    )


def rank_causal_evidence(
    observed_delta: dict[str, float],
) -> tuple[CausalEvidence, ...]:
    """Rank all documented hypotheses by directional agreement."""
    mechanisms = (
        "focus_shift_or_defocus",
        "electronic_noise_increase",
        "sensitivity_or_element_response_change",
        "attenuation_like_depth_rolloff",
        "pulse_or_bandwidth_change",
        "acquisition_timing_inconsistency",
    )
    results = tuple(
        score_causal_evidence(m, observed_delta)
        for m in mechanisms
    )
    return tuple(
        sorted(
            results,
            key=lambda item: item.agreement_score,
            reverse=True,
        )
    )
