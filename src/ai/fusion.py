"""Transparent fusion of statistical and AI evidence.

The fusion score is an engineering evidence index. It is intentionally not
called a probability of physical failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ai.intelligence import AIAnomalyAssessment


@dataclass(frozen=True)
class IntelligenceFusion:
    """Auditable multi-source intelligence assessment."""

    fused_score: float
    confidence: float
    statistical_evidence: float
    ai_evidence: float
    quality_evidence: float
    agreement: float
    state: str

    def validate(self) -> None:
        values = (
            self.fused_score, self.confidence,
            self.statistical_evidence, self.ai_evidence,
            self.quality_evidence, self.agreement,
        )
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Fusion contains non-finite values.")
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError("Fusion scores must be in [0, 1].")
        if self.state not in {"NOMINAL", "WATCH", "HIGH_EVIDENCE"}:
            raise ValueError("Invalid fusion state.")


def fuse_intelligence(
    *,
    mahalanobis_squared: float,
    critical_threshold: float,
    quality_score: float,
    ai: AIAnomalyAssessment,
) -> IntelligenceFusion:
    """Fuse independent evidence streams with explicit contributions.

    Weights are intentionally fixed and inspectable. They are engineering
    defaults, not statistically learned probabilities. They should be
    recalibrated on representative validation data before deployment.
    """
    if not np.isfinite(mahalanobis_squared) or mahalanobis_squared < 0:
        raise ValueError("mahalanobis_squared must be finite and >= 0.")
    if not np.isfinite(critical_threshold) or critical_threshold <= 0:
        raise ValueError("critical_threshold must be finite and > 0.")
    if not 0.0 <= quality_score <= 100.0:
        raise ValueError("quality_score must be in [0, 100].")

    ai.validate()
    statistical = float(np.clip(mahalanobis_squared / critical_threshold, 0.0, 1.0))
    ai_evidence = float(np.clip(ai.anomaly_score, 0.0, 1.0))
    quality_evidence = float(np.clip(1.0 - quality_score / 100.0, 0.0, 1.0))

    agreement = float(np.clip(
        1.0 - abs(statistical - ai_evidence),
        0.0, 1.0,
    ))
    raw = (
        0.45 * statistical
        + 0.40 * ai_evidence
        + 0.15 * quality_evidence
    )
    confidence = float(np.clip(
        0.60 * ai.confidence + 0.40 * agreement,
        0.0, 1.0,
    ))
    fused = float(np.clip(raw * (0.75 + 0.25 * agreement), 0.0, 1.0))

    state = (
        "HIGH_EVIDENCE" if fused >= 0.75
        else "WATCH" if fused >= 0.40
        else "NOMINAL"
    )
    result = IntelligenceFusion(
        fused_score=fused,
        confidence=confidence,
        statistical_evidence=statistical,
        ai_evidence=ai_evidence,
        quality_evidence=quality_evidence,
        agreement=agreement,
        state=state,
    )
    result.validate()
    return result
