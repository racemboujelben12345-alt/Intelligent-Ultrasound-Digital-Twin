import numpy as np
import pytest

from src.prediction.baseline_forecasters import (
    compare_baseline_forecasters,
    exponential_smoothing_forecaster,
    linear_trend_forecaster,
    moving_average_forecaster,
)


def test_moving_average_uses_only_trailing_window():
    forecast = moving_average_forecaster(window=3)
    np.testing.assert_allclose(forecast(np.array([100.0, 2.0, 4.0, 6.0]), 2), [4.0, 4.0])


def test_exponential_smoothing_alpha_one_equals_last_observation():
    forecast = exponential_smoothing_forecaster(alpha=1.0)
    np.testing.assert_allclose(forecast(np.array([1.0, 4.0, 9.0]), 3), [9.0, 9.0, 9.0])


def test_linear_trend_recovers_linear_series_without_looking_ahead():
    forecast = linear_trend_forecaster(np.array([1.0, 3.0, 5.0, 7.0]), 2)
    np.testing.assert_allclose(forecast, [9.0, 11.0])


def test_comparison_uses_identical_targets_for_every_baseline():
    result = compare_baseline_forecasters(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        initial_train_size=4,
        horizon=1,
        step=1,
        moving_average_window=3,
        smoothing_alpha=0.3,
    )
    assert set(result) == {
        "persistence",
        "moving_average_3",
        "exponential_smoothing_0.3",
        "linear_trend",
    }
    target_indices = {evaluation.target_indices for evaluation in result.values()}
    assert target_indices == {(4, 5, 6, 7)}
    assert result["linear_trend"].model_metrics.mae == pytest.approx(0.0)
    assert result["linear_trend"].model_metrics.mae < result["persistence"].model_metrics.mae


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_moving_average_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        moving_average_forecaster(window=window)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, float("nan")])
def test_exponential_smoothing_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        exponential_smoothing_forecaster(alpha=alpha)
