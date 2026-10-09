import pytest

from src.validation.drift_sensitivity_analysis import evaluate_drift_sensitivity


def test_sensitivity_report_stratifies_noise_and_change_magnitude():
    report = evaluate_drift_sensitivity(
        kinds=("nominal", "abrupt_shift"),
        seeds=(3, 7),
        noise_levels=(0.2, 0.5),
        drift_magnitudes=(1.0, 3.0),
        reference_count=40,
        observation_count=40,
    )
    # Nominal: 2 noise cells x 1 magnitude; abrupt: 2 noise x 2 magnitudes.
    assert len(report.cells) == 6
    assert {c.noise_std for c in report.cells} == {0.2, 0.5}
    assert {c.drift_magnitude for c in report.cells} == {1.0, 3.0}
    assert all(c.trial_count == 2 for c in report.cells)
    assert all(0 <= c.detection_rate <= 1 for c in report.cells)
    assert all(0 <= c.mean_false_alarm_fraction <= 1 for c in report.cells)


def test_sensitivity_report_is_reproducible_and_json_friendly():
    kwargs = dict(
        kinds=("abrupt_shift",),
        seeds=(9, 12),
        noise_levels=(0.3,),
        drift_magnitudes=(2.0,),
        reference_count=30,
        observation_count=30,
    )
    first = evaluate_drift_sensitivity(**kwargs)
    assert first == evaluate_drift_sensitivity(**kwargs)
    payload = first.to_dict()
    assert payload["cells"][0]["scenario_name"] == "abrupt_shift"
    assert payload["monitor_parameters"]["ewma_threshold"] == 3.0
    assert "Synthetic" in payload["interpretation"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"noise_levels": (0.0,)},
        {"drift_magnitudes": ()},
        {"kinds": ("unknown",)},
        {"change_fraction": 1.0},
        {"observation_count": 1},
    ],
)
def test_sensitivity_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        evaluate_drift_sensitivity(**kwargs)
