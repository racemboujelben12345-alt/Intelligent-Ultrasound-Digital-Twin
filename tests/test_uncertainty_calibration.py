import numpy as np
import pytest

from src.prediction.uncertainty_calibration import (
    calibrate_split_conformal,
    evaluate_prediction_intervals,
)


def test_split_conformal_uses_calibration_residuals_and_expected_rank():
    model = calibrate_split_conformal(
        calibration_actual=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        calibration_predictions=[1, 2, 2, 4, 4, 6, 7, 6, 9],
        alpha=0.2,
    )
    # Absolute residuals are [0, 0, 1, 0, 1, 0, 0, 2, 0].
    # ceil((9 + 1) * (1 - .2)) = 8, so the 8th ordered residual is 1.
    assert model.calibration_count == 9
    assert model.quantile_rank_1based == 8
    assert model.absolute_residual_quantile == pytest.approx(1.0)


def test_prediction_intervals_are_symmetric_about_point_prediction():
    model = calibrate_split_conformal([0, 2, 4, 6, 8], [0, 1, 2, 3, 4], alpha=0.2)
    lower, upper = model.predict_interval([10.0, 20.0])
    np.testing.assert_allclose(lower, [6.0, 16.0])
    np.testing.assert_allclose(upper, [14.0, 24.0])


def test_interval_metrics_calculate_coverage_width_and_tail_misses():
    result = evaluate_prediction_intervals(
        actual=[0.0, 1.0, 2.0, 4.0],
        lower=[0.0, 0.0, 1.0, 1.0],
        upper=[1.0, 2.0, 3.0, 3.0],
        nominal_coverage=0.8,
    )
    assert result.empirical_coverage == pytest.approx(0.75)
    assert result.mean_width == pytest.approx(1.75)
    assert result.lower_miss_rate == pytest.approx(0.0)
    assert result.upper_miss_rate == pytest.approx(0.25)


def test_rejects_misaligned_nonfinite_and_invalid_intervals():
    with pytest.raises(ValueError, match="same length"):
        calibrate_split_conformal([1, 2], [1])
    with pytest.raises(ValueError, match="alpha"):
        calibrate_split_conformal([1, 2], [1, 2], alpha=1.0)
    with pytest.raises(ValueError, match="finite"):
        calibrate_split_conformal([1, np.nan], [1, 2])
    with pytest.raises(ValueError, match="lower interval"):
        evaluate_prediction_intervals([1], [2], [1], nominal_coverage=0.9)


def test_quantile_rank_clips_to_available_calibration_samples():
    model = calibrate_split_conformal([1, 3, 5], [0, 0, 0], alpha=0.01)
    assert model.quantile_rank_1based == 3
    assert model.absolute_residual_quantile == pytest.approx(5.0)
