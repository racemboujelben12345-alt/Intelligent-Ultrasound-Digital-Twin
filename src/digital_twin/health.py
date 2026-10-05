"""Engineering health and evidence-confidence assessment for the Digital Twin.

This module separates health_index from confidence_score. Neither is a
probability of failure, a clinical metric, or proof of physical hardware fault.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite


HEALTH_STATES = ("NOMINAL", "WATCH", "EARLY_DRIFT", "HIGH_DEVIATION")


@dataclass(frozen=True)
class HealthModelConfig:
    """Explicit configuration for the heuristic engineering-health mapping."""
    quality_weight: float = 0.60
    anomaly_weight: float = 0.40
    nominal_min: float = 85.0
    watch_min: float = 70.0
    early_drift_min: float = 50.0
    reference_distance: float = 3.0

    def validate(self) -> None:
        if self.quality_weight < 0 or self.anomaly_weight < 0:
            raise ValueError("Health weights must be non-negative.")
        if abs((self.quality_weight + self.anomaly_weight) - 1.0) > 1e-9:
            raise ValueError("Health weights must sum to 1.")
        if not (0 <= self.early_drift_min <= self.watch_min <= self.nominal_min <= 100):
            raise ValueError("Health thresholds must be ordered and within [0,100].")
        if self.reference_distance <= 0:
            raise ValueError("reference_distance must be > 0.")


@dataclass(frozen=True)
class TwinHealthAssessment:
    """Transparent, bounded assessment of the current Digital Twin state."""

    health_index: float
    confidence_score: float
    quality_component: float
    anomaly_component: float
    health_state: str
    dominant_evidence: tuple[str, ...]
    interpretation: str
    version: str = "1.0"

    def validate(self) -> None:
        for name in (
            "health_index",
            "confidence_score",
            "quality_component",
            "anomaly_component",
        ):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be in [0, 100].")

        if self.health_state not in HEALTH_STATES:
            raise ValueError(
                f"Invalid health_state: {self.health_state}. "
                f"Allowed: {HEALTH_STATES}"
            )

        if not self.dominant_evidence:
            raise ValueError("dominant_evidence cannot be empty.")

        if not self.version:
            raise ValueError("version is required.")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def anomaly_component(
    mahalanobis_distance: float,
    reference_distance: float = 3.0,
) -> float:
    """Map statistical distance to a bounded monitoring component.

    This is a configurable engineering mapping, not a calibrated probability.
    """
    if not isfinite(mahalanobis_distance) or mahalanobis_distance < 0:
        raise ValueError("mahalanobis_distance must be finite and >= 0.")
    if not isfinite(reference_distance) or reference_distance <= 0:
        raise ValueError("reference_distance must be finite and > 0.")

    return _clip(100.0 * exp(-mahalanobis_distance / reference_distance))


def evidence_confidence(
    *,
    baseline_observations: int,
    feature_count: int,
    expected_feature_count: int,
    provenance_score: float,
) -> float:
    """Estimate evidence quality, not statistical certainty."""
    if baseline_observations < 0:
        raise ValueError("baseline_observations must be >= 0.")
    if feature_count < 0:
        raise ValueError("feature_count must be >= 0.")
    if expected_feature_count <= 0:
        raise ValueError("expected_feature_count must be > 0.")

    baseline_component = _clip(100.0 * min(baseline_observations, 30) / 30)
    feature_component = _clip(
        100.0 * min(feature_count, expected_feature_count)
        / expected_feature_count
    )
    provenance_component = _clip(provenance_score)

    return _clip(
        0.40 * baseline_component
        + 0.30 * feature_component
        + 0.30 * provenance_component
    )


def assess_twin_health(
    *,
    quality_score: float,
    mahalanobis_distance: float,
    baseline_observations: int,
    feature_count: int,
    expected_feature_count: int,
    provenance_score: float,
    reference_distance: float = 3.0,
    model_config: HealthModelConfig | None = None,
) -> TwinHealthAssessment:
    """Build a transparent Digital Twin engineering-health assessment.

    Health = 0.60 * quality + 0.40 * anomaly_component.

    Confidence is deliberately independent from health.
    """
    if not isfinite(quality_score) or not 0.0 <= quality_score <= 100.0:
        raise ValueError("quality_score must be in [0, 100].")

    config = model_config or HealthModelConfig(reference_distance=reference_distance)
    config.validate()
    q = float(quality_score)
    a = anomaly_component(mahalanobis_distance, config.reference_distance)
    health = _clip(config.quality_weight * q + config.anomaly_weight * a)

    confidence = evidence_confidence(
        baseline_observations=baseline_observations,
        feature_count=feature_count,
        expected_feature_count=expected_feature_count,
        provenance_score=provenance_score,
    )

    if health >= config.nominal_min:
        state = "NOMINAL"
    elif health >= config.watch_min:
        state = "WATCH"
    elif health >= config.early_drift_min:
        state = "EARLY_DRIFT"
    else:
        state = "HIGH_DEVIATION"

    evidence = []
    if q <= a:
        evidence.append("quality")
    if a < q:
        evidence.append("statistical_deviation")
    if baseline_observations < 10:
        evidence.append("limited_baseline")
    if provenance_score < 80:
        evidence.append("source_maturity")
    if not evidence:
        evidence.append("balanced_evidence")

    assessment = TwinHealthAssessment(
        health_index=round(health, 3),
        confidence_score=round(confidence, 3),
        quality_component=round(q, 3),
        anomaly_component=round(a, 3),
        health_state=state,
        dominant_evidence=tuple(evidence),
        interpretation=(
            "Engineering indicator derived from observable image quality and "
            "multivariate statistical deviation; not a hardware-failure diagnosis."
        ),
    )
    assessment.validate()
    return assessment
