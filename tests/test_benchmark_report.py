import numpy as np
import pytest

from src.prediction.baseline_forecasters import compare_baseline_forecasters
from src.prediction.benchmark_report import build_multi_series_benchmark_report


def _evaluate(series):
    return compare_baseline_forecasters(
        series,
        initial_train_size=5,
        horizon=1,
        step=1,
        moving_average_window=3,
        smoothing_alpha=0.3,
    )


def test_report_aggregates_multiple_series_without_pooling_target_values():
    results = {
        "linear_feature": _evaluate([1, 2, 3, 4, 5, 6, 7, 8]),
        "scaled_linear_feature": _evaluate([10, 20, 30, 40, 50, 60, 70, 80]),
    }
    report = build_multi_series_benchmark_report(results)
    assert report.series_count == 2
    summaries = {item.model_name: item for item in report.model_summaries}
    trend = summaries["linear_trend"]
    assert trend.wins_vs_persistence == 2
    assert trend.losses_vs_persistence == 0
    assert trend.macro_mean_relative_mae_to_persistence == pytest.approx(0.0)
    assert trend.relative_mae_series_count == 2
    assert trend.forecast_count == 6


def test_report_exposes_per_series_metrics_and_json_friendly_dict():
    report = build_multi_series_benchmark_report({"feature": _evaluate([1, 2, 3, 4, 5, 6, 7])})
    item = next(x for x in report.model_summaries if x.model_name == "persistence")
    assert item.per_series[0].comparison_to_persistence == "tie"
    payload = report.to_dict()
    assert payload["series_count"] == 1
    assert payload["model_summaries"][0]["per_series"][0]["series_name"] == "feature"


def test_report_rejects_inconsistent_model_sets():
    with pytest.raises(ValueError, match="same model names"):
        build_multi_series_benchmark_report({
            "a": _evaluate([1, 2, 3, 4, 5, 6, 7]),
            "b": {"persistence": _evaluate([2, 3, 4, 5, 6, 7, 8])["persistence"]},
        })


def test_report_rejects_target_misalignment_within_series():
    good = _evaluate([1, 2, 3, 4, 5, 6, 7])
    bad = _evaluate([1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="not aligned"):
        build_multi_series_benchmark_report({"feature": {**good, "linear_trend": bad["linear_trend"]}})


def test_zero_persistence_error_omits_relative_ratio():
    results = _evaluate([3, 3, 3, 3, 3, 3, 3])
    report = build_multi_series_benchmark_report({"constant_feature": results})
    trend = next(x for x in report.model_summaries if x.model_name == "linear_trend")
    assert trend.relative_mae_series_count == 0
    assert trend.macro_mean_relative_mae_to_persistence is None
    assert trend.ties_vs_persistence == 1
