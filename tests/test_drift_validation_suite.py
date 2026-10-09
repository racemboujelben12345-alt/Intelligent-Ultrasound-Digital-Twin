import pytest

from src.validation.drift_validation_suite import evaluate_synthetic_drift_grid


def test_grid_is_reproducible_and_aggregates_scenario_families():
    kwargs = dict(
        seeds=(2, 5),
        noise_levels=(0.25,),
        drift_magnitudes=(1.5, 3.0),
        observation_count=60,
        reference_count=60,
    )
    first = evaluate_synthetic_drift_grid(**kwargs)
    second = evaluate_synthetic_drift_grid(**kwargs)

    assert first == second
    assert [item.scenario_name for item in first.aggregates] == [
        "nominal", "gradual_drift", "abrupt_shift", "noisy_nominal"
    ]
    assert len(first.trial_metrics) == 2 + 4 + 4 + 2
    assert all(0.0 <= item.detection_rate <= 1.0 for item in first.aggregates)
    assert all(0.0 <= item.mean_false_alarm_fraction <= 1.0 for item in first.aggregates)


def test_grid_reports_delay_only_for_detected_change_trials():
    report = evaluate_synthetic_drift_grid(
        kinds=("nominal", "abrupt_shift"),
        seeds=(4,),
        noise_levels=(0.2,),
        drift_magnitudes=(5.0,),
        observation_count=80,
        reference_count=80,
        ewma_threshold=2.0,
        cusum_threshold=3.0,
    )
    nominal, shifted = report.aggregates
    assert nominal.detection_rate == 0.0
    assert nominal.mean_detection_delay is None
    assert shifted.detection_rate == 1.0
    assert shifted.mean_detection_delay is not None
    assert shifted.detected_trials == 1


def test_grid_json_friendly_report_preserves_configuration():
    report = evaluate_synthetic_drift_grid(
        kinds=("nominal",),
        seeds=(13,),
        noise_levels=(0.3,),
        drift_magnitudes=(2.0,),
        observation_count=40,
        reference_count=40,
    )
    payload = report.to_dict()
    assert payload["seeds"] == [13]
    assert payload["monitor_parameters"]["cusum_threshold"] == 5.0
    assert len(payload["trial_metrics"]) == 1
    assert "Synthetic algorithmic evaluation only" in payload["interpretation"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"noise_levels": ()},
        {"noise_levels": (0.0,)},
        {"drift_magnitudes": (-1.0,)},
        {"change_fraction": 0.0},
        {"kinds": ("unknown",)},
        {"observation_count": 1},
    ],
)
def test_grid_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        evaluate_synthetic_drift_grid(**kwargs)
