import numpy as np
import pytest

from src.prediction.sequential_drift import monitor_sequential_drift


def _reference():
    return np.linspace(-1.0, 1.0, 101)


def test_nominal_like_observations_do_not_trigger_large_thresholds():
    result = monitor_sequential_drift(
        _reference(),
        [0.0, 0.1, -0.1, 0.05],
        ewma_threshold=3.0,
        cusum_threshold=5.0,
    )
    assert result.combined_alarm_indices == ()
    assert result.first_alarm_index is None
    assert result.reference_count == 101
    assert result.observation_count == 4


def test_persistent_shift_is_detected_by_sequential_monitors():
    result = monitor_sequential_drift(
        _reference(),
        [4.0] * 8,
        ewma_threshold=3.0,
        cusum_threshold=5.0,
    )
    assert result.combined_alarm_indices
    assert result.first_alarm_index is not None
    assert result.first_alarm_index < 8
    assert len(result.ewma) == 8
    assert len(result.cusum_positive) == 8


def test_result_serialization_includes_first_alarm():
    result = monitor_sequential_drift(_reference(), [5.0] * 6)
    payload = result.to_dict()
    assert payload["first_alarm_index"] == result.first_alarm_index
    assert "not proof" in payload["interpretation"]


def test_rejects_constant_reference_because_scale_is_undefined():
    with pytest.raises(ValueError, match="variability is too small"):
        monitor_sequential_drift([1.0] * 8, [1.0, 2.0])


def test_rejects_invalid_inputs_and_thresholds():
    with pytest.raises(ValueError, match="at least 5"):
        monitor_sequential_drift([1.0, 2.0, 3.0, 4.0], [1.0])
    with pytest.raises(ValueError, match="ewma_alpha"):
        monitor_sequential_drift(_reference(), [1.0], ewma_alpha=0.0)
    with pytest.raises(ValueError, match="finite"):
        monitor_sequential_drift(_reference(), [np.nan])
