import pytest

from src.validation.drift_calibration_sample_size import required_nominal_sequences


def test_zero_alarm_plan_for_five_percent_target_at_95_percent_confidence():
    assert required_nominal_sequences(0.05, confidence_level=0.95) == 73


def test_higher_confidence_requires_at_least_as_many_sequences():
    usual = required_nominal_sequences(0.05, confidence_level=0.95)
    stronger = required_nominal_sequences(0.05, confidence_level=0.99)
    assert stronger > usual


def test_assumed_false_alarms_increase_planned_sample_size():
    zero = required_nominal_sequences(0.05, assumed_false_alarm_sequences=0)
    one = required_nominal_sequences(0.05, assumed_false_alarm_sequences=1)
    assert one > zero


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_false_alarm_rate": 0.0},
        {"target_false_alarm_rate": 1.0},
        {"target_false_alarm_rate": 0.05, "confidence_level": 1.0},
        {"target_false_alarm_rate": 0.05, "assumed_false_alarm_sequences": -1},
        {"target_false_alarm_rate": 0.05, "max_sequences": 0},
    ],
)
def test_invalid_planning_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        required_nominal_sequences(**kwargs)


def test_fails_when_maximum_is_too_small():
    with pytest.raises(ValueError, match="No sample size"):
        required_nominal_sequences(0.05, max_sequences=10)
