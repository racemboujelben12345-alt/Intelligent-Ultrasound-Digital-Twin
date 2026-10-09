"""Aggregate baseline forecasts across independent ordered feature series.

Raw errors are retained as descriptive summaries; cross-feature comparison uses
per-series MAE relative to persistence so differently scaled features are not
implicitly treated as interchangeable. This report does not create valid time
series or perform group/session splitting for the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from src.prediction.forecasting_evaluation import RollingOriginEvaluation


@dataclass(frozen=True)
class SeriesBenchmarkResult:
    """Per-series result for one forecasting method."""
    series_name: str
    n_forecasts: int
    mae: float
    rmse: float
    mase: float | None
    persistence_mae: float
    relative_mae_to_persistence: float | None
    comparison_to_persistence: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelBenchmarkSummary:
    """Macro-aggregated model results across the supplied feature series."""
    model_name: str
    series_count: int
    forecast_count: int
    mean_mae: float
    mean_rmse: float
    mean_mase: float | None
    relative_mae_series_count: int
    macro_mean_relative_mae_to_persistence: float | None
    wins_vs_persistence: int
    ties_vs_persistence: int
    losses_vs_persistence: int
    per_series: tuple[SeriesBenchmarkResult, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["per_series"] = [item.to_dict() for item in self.per_series]
        return result


@dataclass(frozen=True)
class MultiSeriesBenchmarkReport:
    """Validated cross-series summary; lower relative MAE is better."""
    series_count: int
    model_summaries: tuple[ModelBenchmarkSummary, ...]
    primary_comparison_metric: str = "macro_mean_relative_mae_to_persistence"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["model_summaries"] = [item.to_dict() for item in self.model_summaries]
        return result


def build_multi_series_benchmark_report(
    results_by_series: Mapping[str, Mapping[str, RollingOriginEvaluation]],
    *,
    comparison_tolerance: float = 1e-12,
) -> MultiSeriesBenchmarkReport:
    """Aggregate same-target rolling-origin comparisons across ordered series.

    Input shape: series name -> model name -> rolling-origin evaluation.
    All models within a series must use identical targets, actual values,
    horizons, and persistence predictions. Model names must be consistent
    across series. Each series contributes equally to the macro relative-MAE
    score when its persistence MAE is non-zero; raw MAE/RMSE are descriptive
    only because feature scales may differ. No hyperparameter tuning occurs.
    """
    if not isinstance(results_by_series, Mapping) or not results_by_series:
        raise ValueError("results_by_series must be a non-empty mapping.")
    if not np.isfinite(comparison_tolerance) or comparison_tolerance < 0:
        raise ValueError("comparison_tolerance must be finite and non-negative.")

    expected_models: set[str] | None = None
    per_model: dict[str, list[SeriesBenchmarkResult]] = {}
    for series_name, evaluations in results_by_series.items():
        if not isinstance(series_name, str) or not series_name.strip():
            raise ValueError("Every series name must be a non-empty string.")
        if not isinstance(evaluations, Mapping) or not evaluations:
            raise ValueError(f"Series {series_name!r} must contain model evaluations.")
        model_names = set(evaluations)
        if any(not isinstance(name, str) or not name.strip() for name in model_names):
            raise ValueError("Every model name must be a non-empty string.")
        if expected_models is None:
            expected_models = model_names
            per_model = {name: [] for name in sorted(model_names)}
        elif model_names != expected_models:
            raise ValueError("Every series must contain the same model names.")

        if any(not isinstance(value, RollingOriginEvaluation) for value in evaluations.values()):
            raise ValueError("Every model result must be a RollingOriginEvaluation.")

        first = next(iter(evaluations.values()))
        for model_name, evaluation in evaluations.items():
            if (
                evaluation.target_indices != first.target_indices
                or evaluation.actual != first.actual
                or evaluation.horizon != first.horizon
                or evaluation.step != first.step
                or evaluation.persistence_predicted != first.persistence_predicted
                or evaluation.persistence_metrics != first.persistence_metrics
            ):
                raise ValueError(
                    f"Model {model_name!r} is not aligned with other models for "
                    f"series {series_name!r}."
                )

            model_mae = evaluation.model_metrics.mae
            persistence_mae = evaluation.persistence_metrics.mae
            relative = (
                model_mae / persistence_mae
                if persistence_mae > np.finfo(float).eps
                else None
            )
            tolerance = comparison_tolerance * max(1.0, persistence_mae, model_mae)
            if model_mae < persistence_mae - tolerance:
                comparison = "win"
            elif model_mae > persistence_mae + tolerance:
                comparison = "loss"
            else:
                comparison = "tie"
            per_model[model_name].append(
                SeriesBenchmarkResult(
                    series_name=series_name,
                    n_forecasts=evaluation.model_metrics.n_forecasts,
                    mae=model_mae,
                    rmse=evaluation.model_metrics.rmse,
                    mase=evaluation.model_metrics.mase,
                    persistence_mae=persistence_mae,
                    relative_mae_to_persistence=relative,
                    comparison_to_persistence=comparison,
                )
            )

    summaries: list[ModelBenchmarkSummary] = []
    for model_name, series_results in sorted(per_model.items()):
        relative_values = [
            item.relative_mae_to_persistence
            for item in series_results
            if item.relative_mae_to_persistence is not None
        ]
        mase_values = [item.mase for item in series_results if item.mase is not None]
        summaries.append(
            ModelBenchmarkSummary(
                model_name=model_name,
                series_count=len(series_results),
                forecast_count=sum(item.n_forecasts for item in series_results),
                mean_mae=float(np.mean([item.mae for item in series_results])),
                mean_rmse=float(np.mean([item.rmse for item in series_results])),
                mean_mase=float(np.mean(mase_values)) if mase_values else None,
                relative_mae_series_count=len(relative_values),
                macro_mean_relative_mae_to_persistence=(
                    float(np.mean(relative_values)) if relative_values else None
                ),
                wins_vs_persistence=sum(item.comparison_to_persistence == "win" for item in series_results),
                ties_vs_persistence=sum(item.comparison_to_persistence == "tie" for item in series_results),
                losses_vs_persistence=sum(item.comparison_to_persistence == "loss" for item in series_results),
                per_series=tuple(series_results),
            )
        )
    return MultiSeriesBenchmarkReport(
        series_count=len(results_by_series),
        model_summaries=tuple(summaries),
    )
