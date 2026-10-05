import pytest

from src.ai.fusion import IntelligenceFusion
from src.digital_twin.health import assess_twin_health
from src.digital_twin.state_engine import decide_twin_state


def _fusion(state="NOMINAL", score=0.2, confidence=0.9):
    return IntelligenceFusion(
        fused_score=score,
        confidence=confidence,
        statistical_evidence=score,
        ai_evidence=score,
        quality_evidence=0.1,
        agreement=0.9,
        state=state,
    )


@pytest.mark.parametrize(
    ("statistical", "fusion_state", "health_state", "expected"),
    [
        ("NOMINAL", "NOMINAL", "NOMINAL", "NOMINAL"),
        ("NOMINAL", "WATCH", "NOMINAL", "WATCH"),
        ("EARLY_DRIFT", "WATCH", "EARLY_DRIFT", "EARLY_DRIFT"),
        ("HIGH_DEVIATION", "HIGH_EVIDENCE", "HIGH_DEVIATION", "HIGH_DEVIATION"),
    ],
)
def test_unified_state_precedence(statistical, fusion_state, health_state, expected):
    health = assess_twin_health(
        quality_score={"NOMINAL": 100, "WATCH": 80, "EARLY_DRIFT": 65, "HIGH_DEVIATION": 20}[health_state],
        mahalanobis_distance={"NOMINAL": 0, "WATCH": 1, "EARLY_DRIFT": 2, "HIGH_DEVIATION": 8}[health_state],
        baseline_observations=30,
        feature_count=10,
        expected_feature_count=10,
        provenance_score=100,
    )
    decision = decide_twin_state(
        statistical_state=statistical,
        fusion=_fusion(fusion_state, 0.8 if fusion_state != "NOMINAL" else 0.1),
        health=health,
        drift=None,
    )
    assert decision.state == expected
    decision.validate()


def test_unified_state_is_conservative_with_persistent_drift():
    from src.drift.engine import DriftAnalysis
    from src.drift.control_charts import DriftObservation

    observation = DriftObservation(
        index=2, signal=4.0, z_score=3.0, ewma=2.0,
        cusum_positive=5.0, cusum_negative=0.0,
        point_alarm=True, ewma_alarm=True, cusum_alarm=True,
        consecutive_alarms=3, persistent_alarm=True,
        status="PERSISTENT_DRIFT",
    )
    drift = DriftAnalysis(observations=(observation,))
    health = assess_twin_health(
        quality_score=95,
        mahalanobis_distance=0.5,
        baseline_observations=30,
        feature_count=10,
        expected_feature_count=10,
        provenance_score=100,
    )
    decision = decide_twin_state(
        statistical_state="NOMINAL",
        fusion=_fusion(),
        health=health,
        drift=drift,
    )
    assert decision.state == "EARLY_DRIFT"
    assert "persistent temporal drift detected" in decision.rationale
