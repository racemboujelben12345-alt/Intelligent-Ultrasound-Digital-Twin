import numpy as np
import pytest

from src.prediction.forecasting_evaluation import (
    evaluate_forecasts,
    evaluate_rolling_origin,
    persistence_forecaster,
)


def test_metrics_match_hand_calculated_values():
    result = evaluate_forecasts(
        [1.0, 3.0, 2.0],
        [2.0, 2.0, 4.0],
        training_series=[1.0, 2.0, 4.0],
        baseline_mae=2.0,
    )
    assert result.n_forecasts == 3
    assert result.mae == pytest.approx(4.0 / 3.0)
    assert result.rmse == pytest.approx(np.sqrt(2.0))
    assert result.mase == pytest.approx((4.0 / 3.0) / 1.5)
    assert result.relative_mae_to_baseline == pytest.approx(2.0 / 3.0)


def test_rolling_origin_only_passes_observed_prefix_to_forecaster():
    series = np.arange(1.0, 11.0)
    observed_lengths = []

    def forecast(history, horizon):
        observed_lengths.append(len(history))
        # Linear extrapolation from the last two observed points.
        slope = history[-1] - history[-2]
        return history[-1] + slope * np.arange(1, horizon + 1)

    result = evaluate_rolling_origin(
        series,
        initial_train_size=5,
        horizon=2,
        step=1,
        forecaster=forecast,
    )

    assert result.origin_train_sizes == (5, 6, 7, 8)
    assert result.target_indices == (6, 7, 8, 9)
    assert observed_lengths == [5, 6, 7, 8]
    assert result.model_metrics.mae == pytest.approx(0.0)
    assert result.model_metrics.mae < result.persistence_metrics.mae


def test_persistence_baseline_repeats_last_observation():
    np.testing.assert_array_equal(
        persistence_forecaster(np.array([2.0, 5.0]), 3),
        np.array([5.0, 5.0, 5.0]),
    )


def test_rolling_origin_rejects_insufficient_data():
    with pytest.raises(ValueError, match="Not enough observations"):
        evaluate_rolling_origin([1, 2, 3], initial_train_size=2, horizon=2)


def test_metrics_reject_misaligned_or_nonfinite_values():
    with pytest.raises(ValueError, match="same length"):
        evaluate_forecasts([1, 2], [1])
    with pytest.raises(ValueError, match="finite"):
        evaluate_forecasts([1, np.nan], [1, 2])


def test_constant_training_series_has_undefined_mase():
    result = evaluate_forecasts(
        [3.0, 4.0], [3.0, 3.0], training_series=[2.0, 2.0, 2.0]
    )
    assert result.mase is None
