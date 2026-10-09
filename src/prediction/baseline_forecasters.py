"""Simple, reproducible forecasting baselines for ordered feature series.

These baselines are comparison points, not evidence of scanner fault prediction.
Use only a single meaningful, ordered acquisition sequence per evaluation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.prediction.forecasting_evaluation import (
    Forecaster,
    RollingOriginEvaluation,
    evaluate_rolling_origin,
    persistence_forecaster,
)

def _history(history: np.ndarray, horizon: int) -> tuple[np.ndarray, int]:
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("history must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values.")
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    return values, int(horizon)


def moving_average_forecaster(window: int = 5) -> Forecaster:
    """Return a trailing moving-average forecaster, flat over the horizon."""
    if isinstance(window, bool) or int(window) != window or window < 1:
        raise ValueError("window must be a positive integer.")
    window = int(window)

    def forecast(history: np.ndarray, horizon: int) -> np.ndarray:
        values, steps = _history(history, horizon)
        level = float(np.mean(values[-window:]))
        return np.full(steps, level, dtype=float)

    return forecast


def exponential_smoothing_forecaster(alpha: float = 0.3) -> Forecaster:
    """Return simple exponential smoothing with a fixed, preselected alpha."""
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1].")

    def forecast(history: np.ndarray, horizon: int) -> np.ndarray:
        values, steps = _history(history, horizon)
        level = float(values[0])
        for value in values[1:]:
            level = alpha * float(value) + (1.0 - alpha) * level
        return np.full(steps, level, dtype=float)

    return forecast


def linear_trend_forecaster(history: np.ndarray, horizon: int) -> np.ndarray:
    """Fit a least-squares linear trend on observed history and extrapolate."""
    values, steps = _history(history, horizon)
    if values.size < 2:
        raise ValueError("linear trend requires at least two observations.")
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, deg=1)
    future_x = np.arange(values.size, values.size + steps, dtype=float)
    return slope * future_x + intercept


def compare_baseline_forecasters(
    series: np.ndarray | list[float] | tuple[float, ...],
    *,
    initial_train_size: int,
    horizon: int = 1,
    step: int = 1,
    moving_average_window: int = 5,
    smoothing_alpha: float = 0.3,
) -> dict[str, RollingOriginEvaluation]:
    """Evaluate simple baselines on exactly the same rolling-origin targets.

    Hyperparameters are fixed before evaluation; this helper does not tune them.
    The caller must define valid temporal/group splits and ensure the input is
    one ordered, comparable feature series. The output includes persistence,
    trailing moving average, simple exponential smoothing and linear trend.
    """
    forecasters: dict[str, Forecaster] = {
        "persistence": persistence_forecaster,
        f"moving_average_{int(moving_average_window)}": moving_average_forecaster(
            moving_average_window
        ),
        f"exponential_smoothing_{smoothing_alpha:g}": exponential_smoothing_forecaster(
            smoothing_alpha
        ),
        "linear_trend": linear_trend_forecaster,
    }
    return {
        name: evaluate_rolling_origin(
            series,
            initial_train_size=initial_train_size,
            horizon=horizon,
            step=step,
            forecaster=forecaster,
        )
        for name, forecaster in forecasters.items()
    }
