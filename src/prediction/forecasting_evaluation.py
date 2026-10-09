"""Leakage-aware rolling-origin evaluation for univariate forecasts.

This module evaluates forecasts for a single, ordered feature series (for
example, entropy across sequential acquisitions). It deliberately does not
train a model or assume that an image collection is a time series. Callers
must provide a meaningful acquisition order and define the forecasting task.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

Forecaster = Callable[[np.ndarray, int], Sequence[float] | np.ndarray]


@dataclass(frozen=True)
class ForecastMetrics:
    """Point-forecast metrics for one aligned set of targets."""

    n_forecasts: int
    mae: float
    rmse: float
    mase: float | None
    relative_mae_to_baseline: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class RollingOriginEvaluation:
    """Predictions produced at sequential origins without future-data access."""

    horizon: int
    step: int
    origin_train_sizes: tuple[int, ...]
    target_indices: tuple[int, ...]
    actual: tuple[float, ...]
    predicted: tuple[float, ...]
    persistence_predicted: tuple[float, ...]
    model_metrics: ForecastMetrics
    persistence_metrics: ForecastMetrics

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "step": self.step,
            "origin_train_sizes": list(self.origin_train_sizes),
            "target_indices": list(self.target_indices),
            "actual": list(self.actual),
            "predicted": list(self.predicted),
            "persistence_predicted": list(self.persistence_predicted),
            "model_metrics": self.model_metrics.to_dict(),
            "persistence_metrics": self.persistence_metrics.to_dict(),
        }


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def evaluate_forecasts(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    training_series: Sequence[float] | np.ndarray | None = None,
    baseline_mae: float | None = None,
) -> ForecastMetrics:
    """Compute MAE, RMSE, optional MASE and optional relative MAE.

    MASE uses the mean absolute one-step difference in the supplied training
    series as its scale. It is undefined for a constant training series.
    baseline_mae must be measured on the exact same targets as y_pred.
    """
    actual = _finite_vector(y_true, "y_true")
    predicted = _finite_vector(y_pred, "y_pred")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if baseline_mae is not None and (
        not np.isfinite(baseline_mae) or baseline_mae < 0
    ):
        raise ValueError("baseline_mae must be finite and non-negative.")

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    mase = None
    if training_series is not None:
        train = _finite_vector(training_series, "training_series")
        if train.size >= 2:
            scale = float(np.mean(np.abs(np.diff(train))))
            if scale > np.finfo(float).eps:
                mase = mae / scale

    relative = None
    if baseline_mae is not None and baseline_mae > np.finfo(float).eps:
        relative = mae / baseline_mae

    return ForecastMetrics(
        n_forecasts=int(actual.size),
        mae=mae,
        rmse=rmse,
        mase=None if mase is None else float(mase),
        relative_mae_to_baseline=None if relative is None else float(relative),
    )


def persistence_forecaster(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast every future step as the latest observed value."""
    if history.ndim != 1 or history.size == 0:
        raise ValueError("history must be a non-empty one-dimensional array.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    return np.full(horizon, float(history[-1]), dtype=float)


def evaluate_rolling_origin(
    series: Sequence[float] | np.ndarray,
    *,
    initial_train_size: int,
    horizon: int = 1,
    step: int = 1,
    forecaster: Forecaster | None = None,
) -> RollingOriginEvaluation:
    """Evaluate a forecaster with expanding-window rolling-origin testing.

    At each origin, the callback receives only the observed prefix and the
    requested horizon. The last value of each horizon forecast is scored
    against the corresponding future target. The persistence baseline uses
    the same origins and targets. MASE is scaled using the initial training
    prefix so model and baseline results are comparable.

    The input must be one ordered series for one feature and one comparable
    acquisition domain. Group/session splits and data-quality checks must be
    handled by the caller before using this evaluator.
    """
    values = _finite_vector(series, "series")
    if isinstance(initial_train_size, bool) or initial_train_size < 2:
        raise ValueError("initial_train_size must be an integer of at least 2.")
    if int(initial_train_size) != initial_train_size:
        raise ValueError("initial_train_size must be an integer.")
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if isinstance(step, bool) or int(step) != step or step < 1:
        raise ValueError("step must be a positive integer.")
    initial_train_size, horizon, step = int(initial_train_size), int(horizon), int(step)
    if initial_train_size + horizon > values.size:
        raise ValueError("Not enough observations for the initial train prefix and horizon.")

    predict = forecaster or persistence_forecaster
    origins: list[int] = []
    target_indices: list[int] = []
    actual: list[float] = []
    predicted: list[float] = []
    persistence_predicted: list[float] = []

    for origin in range(initial_train_size, values.size - horizon + 1, step):
        history = values[:origin].copy()
        forecast = _finite_vector(predict(history.copy(), horizon), "forecaster output")
        if forecast.size != horizon:
            raise ValueError(
                f"forecaster must return exactly {horizon} values; got {forecast.size}."
            )
        baseline = persistence_forecaster(history, horizon)
        target_index = origin + horizon - 1
        origins.append(origin)
        target_indices.append(target_index)
        actual.append(float(values[target_index]))
        predicted.append(float(forecast[-1]))
        persistence_predicted.append(float(baseline[-1]))

    if not actual:
        raise ValueError("No rolling-origin forecasts could be evaluated.")

    train_scale = values[:initial_train_size]
    baseline_metrics = evaluate_forecasts(
        actual, persistence_predicted, training_series=train_scale
    )
    model_metrics = evaluate_forecasts(
        actual,
        predicted,
        training_series=train_scale,
        baseline_mae=baseline_metrics.mae,
    )
    return RollingOriginEvaluation(
        horizon=horizon,
        step=step,
        origin_train_sizes=tuple(origins),
        target_indices=tuple(target_indices),
        actual=tuple(actual),
        predicted=tuple(predicted),
        persistence_predicted=tuple(persistence_predicted),
        model_metrics=model_metrics,
        persistence_metrics=baseline_metrics,
    )
