import pytest

from src.digital_twin.health import (
    assess_twin_health,
    anomaly_component,
    evidence_confidence,
)


def test_health_is_bounded_and_separates_confidence():
    result = assess_twin_health(
        quality_score=90,
        mahalanobis_distance=0.5,
        baseline_observations=30,
        feature_count=10,
        expected_feature_count=10,
        provenance_score=100,
    )
    assert 0 <= result.health_index <= 100
    assert 0 <= result.confidence_score <= 100
    assert result.health_index > 85
    assert result.confidence_score == 100


def test_anomaly_component_decreases_with_distance():
    assert anomaly_component(0.5) > anomaly_component(4.0)


def test_health_is_deterministic():
    kwargs = dict(
        quality_score=82,
        mahalanobis_distance=2.0,
        baseline_observations=12,
        feature_count=8,
        expected_feature_count=10,
        provenance_score=80,
    )
    assert assess_twin_health(**kwargs) == assess_twin_health(**kwargs)


def test_confidence_is_lower_with_weak_evidence():
    strong = evidence_confidence(
        baseline_observations=30,
        feature_count=10,
        expected_feature_count=10,
        provenance_score=100,
    )
    weak = evidence_confidence(
        baseline_observations=2,
        feature_count=4,
        expected_feature_count=10,
        provenance_score=50,
    )
    assert strong > weak


@pytest.mark.parametrize(
    "quality,distance,expected",
    [
        (100, 0.0, "NOMINAL"),
        (80, 1.0, "WATCH"),
        (65, 2.0, "EARLY_DRIFT"),
        (20, 8.0, "HIGH_DEVIATION"),
    ],
)
def test_health_state_mapping(quality, distance, expected):
    result = assess_twin_health(
        quality_score=quality,
        mahalanobis_distance=distance,
        baseline_observations=30,
        feature_count=10,
        expected_feature_count=10,
        provenance_score=100,
    )
    assert result.health_state == expected
