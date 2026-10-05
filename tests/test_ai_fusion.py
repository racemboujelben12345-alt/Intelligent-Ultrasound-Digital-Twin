import numpy as np

from src.ai.fusion import fuse_intelligence
from src.ai.intelligence import AIAnomalyAssessment


def _ai(score=0.8, confidence=0.9):
    return AIAnomalyAssessment(
        anomaly_score=score,
        anomaly_vote_rate=0.7,
        confidence=confidence,
        ensemble_agreement=0.8,
        model_scores=(0.8, 0.82, 0.79),
        state="ANOMALY",
    )


def test_fusion_is_bounded_and_exposes_contributions():
    result = fuse_intelligence(
        mahalanobis_squared=8.0,
        critical_threshold=10.0,
        quality_score=70.0,
        ai=_ai(),
    )
    result.validate()
    assert 0.0 <= result.fused_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert np.isclose(
        result.fused_score,
        np.clip(
            (0.35 * 0.8 + 0.35 * 0.8 + 0.15 * 0.3 + 0.15 * 0.0)
            * (0.75 + 0.25 * 1.0),
            0.0, 1.0,
        ),
    )


def test_fusion_penalizes_disagreement():
    result = fuse_intelligence(
        mahalanobis_squared=0.1,
        critical_threshold=10.0,
        quality_score=95.0,
        ai=_ai(score=0.95, confidence=0.9),
    )
    assert result.agreement < 1.0
    assert result.confidence < 0.9
