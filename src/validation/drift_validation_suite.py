"""Monte Carlo-style evaluation of sequential drift monitoring on synthetic data.

This module summarizes algorithmic behaviour across pre-specified scenario
grids. It is not a physical ultrasound-system simulator or validation study.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from src.validation.synthetic_drift_scenarios import (
    DriftDetectionMetrics,
    generate_synthetic_drift_scenario,
    run_synthetic_drift_validation,
)


@dataclass(frozen=True)
class DriftScenarioAggregate:
    """Aggregate metrics for one scenario family across repeated trials."""

    scenario_name: str
    trial_count: int
    detection_rate: float
    mean_false_alarm_fraction: float
    mean_detection_delay: float | None
    median_detection_delay: float | None
    detected_trials: int
    interpretation: str = (
        "Synthetic scenario-grid summary only; not validated physical "
        "scanner performance or hardware-failure prediction."
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DriftGridReport:
    """Per-trial and per-family summaries for a fixed, explicit configuration."""

    trial_metrics: tuple[DriftDetectionMetrics, ...]
    aggregates: tuple[DriftScenarioAggregate, ...]
    seeds: tuple[int, ...]
    noise_levels: tuple[float, ...]
    drift_magnitudes: tuple[float, ...]
    monitor_parameters: dict[str, float]
    interpretation: str = (
        "Synthetic algorithmic evaluation only. No physical-system or clinical "
        "claim is supported by this report."
    )

    def to_dict(self) -> dict:
        return {
            "trial_metrics": [metric.to_dict() for metric in self.trial_metrics],
            "aggregates": [aggregate.to_dict() for aggregate in self.aggregates],
            "seeds": list(self.seeds),
            "noise_levels": list(self.noise_levels),
            "drift_magnitudes": list(self.drift_magnitudes),
            "monitor_parameters": dict(self.monitor_parameters),
            "interpretation": self.interpretation,
        }


def evaluate_synthetic_drift_grid(
    *,
    kinds: Sequence[str] = ("nominal", "gradual_drift", "abrupt_shift", "noisy_nominal"),
    seeds: Sequence[int] = (11, 23, 37, 41, 53),
    noise_levels: Sequence[float] = (0.2, 0.35, 0.6),
    drift_magnitudes: Sequence[float] = (1.0, 2.5, 4.0),
    reference_count: int = 120,
    observation_count: int = 120,
    change_fraction: float = 0.5,
    seasonality_amplitude: float = 0.0,
    ewma_alpha: float = 0.2,
    ewma_threshold: float = 3.0,
    cusum_allowance: float = 0.5,
    cusum_threshold: float = 5.0,
) -> DriftGridReport:
    """Evaluate fixed EWMA/CUSUM parameters over a pre-specified scenario grid.

    Nominal/noisy-nominal trials are run once per seed/noise level; drift trials
    additionally cross the specified magnitudes. The same monitor thresholds
    are applied throughout the grid. This function deliberately does not tune
    thresholds from the evaluation outcomes.
    """
    allowed = {"nominal", "gradual_drift", "abrupt_shift", "noisy_nominal"}
    kinds = tuple(kinds)
    seeds = tuple(int(seed) for seed in seeds)
    noise_levels = tuple(float(level) for level in noise_levels)
    drift_magnitudes = tuple(float(magnitude) for magnitude in drift_magnitudes)
    if not kinds or any(kind not in allowed for kind in kinds):
        raise ValueError(f"kinds must be non-empty and drawn from {sorted(allowed)}.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique.")
    if not noise_levels or any(not np.isfinite(v) or v <= 0 for v in noise_levels):
        raise ValueError("noise_levels must be non-empty, finite and positive.")
    if not drift_magnitudes or any(not np.isfinite(v) or v <= 0 for v in drift_magnitudes):
        raise ValueError("drift_magnitudes must be non-empty, finite and positive.")
    if not 0.0 < change_fraction < 1.0:
        raise ValueError("change_fraction must be strictly between 0 and 1.")
    if reference_count < 5 or observation_count < 2:
        raise ValueError("reference_count must be >= 5 and observation_count >= 2.")
    if not np.isfinite(seasonality_amplitude) or seasonality_amplitude < 0:
        raise ValueError("seasonality_amplitude must be finite and non-negative.")

    change_index = min(observation_count - 1, max(1, int(observation_count * change_fraction)))
    trial_metrics: list[DriftDetectionMetrics] = []
    by_kind: dict[str, list[DriftDetectionMetrics]] = {kind: [] for kind in kinds}

    for kind in kinds:
        magnitudes = drift_magnitudes if kind in {"gradual_drift", "abrupt_shift"} else (drift_magnitudes[0],)
        for seed in seeds:
            for noise_std in noise_levels:
                for magnitude in magnitudes:
                    scenario = generate_synthetic_drift_scenario(
                        kind,
                        reference_count=reference_count,
                        observation_count=observation_count,
                        seed=seed,
                        noise_std=noise_std,
                        drift_magnitude=magnitude,
                        change_index=change_index if kind in {"gradual_drift", "abrupt_shift"} else None,
                        seasonality_amplitude=seasonality_amplitude,
                    )
                    _, metrics = run_synthetic_drift_validation(
                        scenario,
                        ewma_alpha=ewma_alpha,
                        ewma_threshold=ewma_threshold,
                        cusum_allowance=cusum_allowance,
                        cusum_threshold=cusum_threshold,
                    )
                    trial_metrics.append(metrics)
                    by_kind[kind].append(metrics)

    aggregates: list[DriftScenarioAggregate] = []
    for kind in kinds:
        trials = by_kind[kind]
        delays = [m.detection_delay for m in trials if m.detection_delay is not None]
        aggregates.append(
            DriftScenarioAggregate(
                scenario_name=kind,
                trial_count=len(trials),
                detection_rate=float(np.mean([m.detected for m in trials])),
                mean_false_alarm_fraction=float(np.mean([m.false_alarm_fraction for m in trials])),
                mean_detection_delay=float(np.mean(delays)) if delays else None,
                median_detection_delay=float(np.median(delays)) if delays else None,
                detected_trials=len(delays),
            )
        )

    return DriftGridReport(
        trial_metrics=tuple(trial_metrics),
        aggregates=tuple(aggregates),
        seeds=seeds,
        noise_levels=noise_levels,
        drift_magnitudes=drift_magnitudes,
        monitor_parameters={
            "ewma_alpha": float(ewma_alpha),
            "ewma_threshold": float(ewma_threshold),
            "cusum_allowance": float(cusum_allowance),
            "cusum_threshold": float(cusum_threshold),
            "change_fraction": float(change_fraction),
        },
    )
