import numpy as np
import pytest

from src.prediction.sequential_drift import monitor_sequential_drift
from src.validation.synthetic_drift_scenarios import (
    evaluate_drift_detection,
    generate_synthetic_drift_scenario,
    run_synthetic_drift_validation,
)


def test_scenario_generation_is_reproducible_and_tracks_known_change():
    first = generate_synthetic_drift_scenario("abrupt_shift", seed=17)
    second = generate_synthetic_drift_scenario("abrupt_shift", seed=17)

    assert first == second
    assert first.change_index == 60
    assert len(first.reference) == 120
    assert len(first.observations) == 120
    assert np.mean(first.observations[70:]) > np.mean(first.observations[:50])


@pytest.mark.parametrize(
    ("kind", "expected_change"),
    [
        ("nominal", None),
        ("noisy_nominal", None),
        ("gradual_drift", 60),
        ("abrupt_shift", 60),
    ],
)
def test_supported_scenarios_have_correct_ground_truth(kind, expected_change):
    scenario = generate_synthetic_drift_scenario(kind, seed=3)
    assert scenario.change_index == expected_change
    assert np.all(np.isfinite(scenario.reference))
    assert np.all(np.isfinite(scenario.observations))


def test_abrupt_shift_is_detected_and_delay_is_non_negative():
    scenario = generate_synthetic_drift_scenario(
        "abrupt_shift", seed=42, drift_magnitude=3.0, change_index=50
    )
    _, metrics = run_synthetic_drift_validation(scenario)

    assert metrics.detected
    assert metrics.detection_delay is not None
    assert metrics.detection_delay >= 0
    assert 0.0 <= metrics.false_alarm_fraction <= 1.0
    assert "not evidence" in metrics.interpretation


def test_nominal_metrics_treat_all_alarms_as_false_alarm_observations():
    scenario = generate_synthetic_drift_scenario("nominal", seed=7)
    monitor_result = monitor_sequential_drift(
        scenario.reference, scenario.observations,
        ewma_threshold=1e6, cusum_threshold=1e6,
    )
    metrics = evaluate_drift_detection(scenario, monitor_result)

    assert not metrics.detected
    assert metrics.detection_delay is None
    assert metrics.false_alarm_count == 0
    assert metrics.false_alarm_fraction == 0.0
    assert metrics.to_dict()["scenario_name"] == "nominal"


def test_metric_evaluation_rejects_mismatched_result_length():
    scenario = generate_synthetic_drift_scenario("nominal", observation_count=20)
    wrong_length_result = monitor_sequential_drift(scenario.reference, [0.0] * 19)
    with pytest.raises(ValueError, match="length must match"):
        evaluate_drift_detection(scenario, wrong_length_result)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_count": 4},
        {"observation_count": 0},
        {"noise_std": 0},
        {"drift_magnitude": -1},
        {"seasonality_amplitude": -1},
    ],
)
def test_generator_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic_drift_scenario("nominal", **kwargs)


def test_change_index_is_not_allowed_for_nominal_scenario():
    with pytest.raises(ValueError, match="must be None"):
        generate_synthetic_drift_scenario("nominal", change_index=10)


def test_scenario_metadata_is_json_friendly():
    scenario = generate_synthetic_drift_scenario("gradual_drift", seed=9)
    payload = scenario.to_dict()
    assert payload["name"] == "gradual_drift"
    assert payload["change_index"] == 60
    assert isinstance(payload["observations"], tuple)
