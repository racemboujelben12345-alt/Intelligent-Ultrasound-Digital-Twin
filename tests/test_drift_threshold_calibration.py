import pytest

from src.validation.drift_threshold_calibration import (
    calibrate_drift_thresholds,
    evaluate_calibrated_thresholds,
)
from src.validation.synthetic_drift_scenarios import generate_synthetic_drift_scenario


def _nominal(seed):
    return generate_synthetic_drift_scenario(
        "nominal", seed=seed, reference_count=80, observation_count=80, noise_std=0.3
    )


def test_calibration_uses_nominal_data_and_evaluates_disjoint_seeds():
    calibration_scenarios = [_nominal(seed) for seed in (1, 2, 3, 4, 5, 6)]
    calibration = calibrate_drift_thresholds(
        calibration_scenarios,
        candidates=((2.0, 3.0), (3.0, 5.0), (4.0, 7.0)),
        target_false_alarm_rate=1.0,
    )
    assert set(calibration.calibration_seed_ids) == {1, 2, 3, 4, 5, 6}
    assert calibration.selected_ewma_threshold > 0
    assert calibration.selected_cusum_threshold > 0
    assert all(0.0 <= score.false_alarm_rate_lower <= score.false_alarm_rate_upper <= 1.0
               for score in calibration.candidate_scores)
    assert all(score.false_alarm_rate_upper <= calibration.target_false_alarm_rate
               for score in calibration.candidate_scores
               if (score.ewma_threshold, score.cusum_threshold) ==
               (calibration.selected_ewma_threshold, calibration.selected_cusum_threshold))
    assert calibration.to_dict()["candidate_scores"]

    evaluation_scenarios = [
        generate_synthetic_drift_scenario(
            "abrupt_shift", seed=seed, reference_count=80, observation_count=80,
            noise_std=0.3, drift_magnitude=3.0,
        )
        for seed in (101, 102, 103)
    ]
    result = evaluate_calibrated_thresholds(calibration, evaluation_scenarios)
    assert set(result.evaluation_seed_ids).isdisjoint(result.calibration_seed_ids)
    assert result.change_detection_rate is not None
    assert result.ewma_threshold == calibration.selected_ewma_threshold
    assert len(result.to_dict()["trial_metrics"]) == 3


def test_calibration_rejects_changed_scenarios_and_reused_seeds():
    with pytest.raises(ValueError, match="nominal/no-change"):
        calibrate_drift_thresholds(
            [_nominal(seed) for seed in (1, 2, 3, 4)] + [
                generate_synthetic_drift_scenario("abrupt_shift", seed=5)
            ],
            candidates=((3.0, 5.0),),
        )
    with pytest.raises(ValueError, match="distinct seeds"):
        calibrate_drift_thresholds(
            [_nominal(seed) for seed in (1, 2, 3, 4)] + [_nominal(4)],
            candidates=((3.0, 5.0),),
        )


def test_calibration_fails_when_no_candidate_meets_target():
    scenarios = [
        generate_synthetic_drift_scenario(
            "nominal", seed=seed, reference_count=80, observation_count=80, noise_std=0.3
        )
        for seed in (11, 12, 13, 14, 15)
    ]
    with pytest.raises(ValueError, match="No candidate meets"):
        calibrate_drift_thresholds(
            scenarios, candidates=((0.01, 0.01),), target_false_alarm_rate=0.0
        )


def test_evaluation_rejects_seed_overlap():
    calibration = calibrate_drift_thresholds(
        [_nominal(seed) for seed in (1, 2, 3, 4, 5)],
        candidates=((3.0, 5.0),),
        target_false_alarm_rate=1.0,
    )
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_calibrated_thresholds(calibration, [_nominal(5)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_false_alarm_rate": -0.1},
        {"target_false_alarm_rate": 1.1},
        {"ewma_alpha": 0.0},
        {"cusum_allowance": -0.1},
        {"confidence_level": 1.0},
    ],
)
def test_calibration_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        calibrate_drift_thresholds(
            [_nominal(seed) for seed in (21, 22, 23, 24, 25)],
            candidates=((3.0, 5.0),),
            **kwargs,
        )
