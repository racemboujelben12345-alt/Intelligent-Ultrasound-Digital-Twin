"""Split-conformal prediction intervals with explicit calibration separation.

The implementation provides a simple, distribution-light baseline for
interval calibration around an existing point predictor. Its coverage
guarantees rely on exchangeability of calibration and test examples; temporal
dependence, domain shift and adaptive monitoring can invalidate that premise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConformalIntervalModel:
    """Calibrated symmetric interval radius for a fixed predictor and alpha."""

    alpha: float
    calibration_count: int
    absolute_residual_quantile: float
    quantile_rank_1based: int
    method: str = "split_conformal_absolute_residual"

    def predict_interval(
        self, point_predictions: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = _finite_vector(point_predictions, "point_predictions")
        radius = self.absolute_residual_quantile
        return predictions - radius, predictions + radius

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class IntervalMetrics:
    """Empirical test-set quality of prediction intervals."""

    n_intervals: int
    nominal_coverage: float
    empirical_coverage: float
    mean_width: float
    median_width: float
    lower_miss_rate: float
    upper_miss_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def calibrate_split_conformal(
    calibration_actual: Sequence[float] | np.ndarray,
    calibration_predictions: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.1,
) -> ConformalIntervalModel:
    """Calibrate a symmetric absolute-residual interval on held-out calibration data.

    The point predictor must already be fixed; do not fit or tune it on this
    calibration set. Test outcomes must not be used during calibration.
    The finite-sample rank is ceil((n + 1) * (1 - alpha)), clipped to n.
    """
    actual = _finite_vector(calibration_actual, "calibration_actual")
    predicted = _finite_vector(calibration_predictions, "calibration_predictions")
    if actual.shape != predicted.shape:
        raise ValueError("Calibration actuals and predictions must have the same length.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1.")

    residuals = np.abs(actual - predicted)
    n = int(residuals.size)
    rank = min(n, int(np.ceil((n + 1) * (1.0 - alpha))))
    # The rank is 1-based; partition at rank-1 to avoid full sorting.
    quantile = float(np.partition(residuals, rank - 1)[rank - 1])
    return ConformalIntervalModel(
        alpha=float(alpha),
        calibration_count=n,
        absolute_residual_quantile=quantile,
        quantile_rank_1based=rank,
    )


def evaluate_prediction_intervals(
    actual: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    *,
    nominal_coverage: float,
) -> IntervalMetrics:
    """Measure empirical coverage, width and lower/upper miss rates on test data."""
    y = _finite_vector(actual, "actual")
    lo = _finite_vector(lower, "lower")
    hi = _finite_vector(upper, "upper")
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("Actuals, lower bounds and upper bounds must have the same length.")
    if not np.isfinite(nominal_coverage) or not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be strictly between 0 and 1.")
    if np.any(lo > hi):
        raise ValueError("Every lower interval bound must be <= its upper bound.")

    below = y < lo
    above = y > hi
    widths = hi - lo
    return IntervalMetrics(
        n_intervals=int(y.size),
        nominal_coverage=float(nominal_coverage),
        empirical_coverage=float(np.mean(~(below | above))),
        mean_width=float(np.mean(widths)),
        median_width=float(np.median(widths)),
        lower_miss_rate=float(np.mean(below)),
        upper_miss_rate=float(np.mean(above)),
    )
