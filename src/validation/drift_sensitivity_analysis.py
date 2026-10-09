"""Stratified sensitivity analysis for sequential drift monitors.

This module reports outcomes separately by scenario family, noise level, and
change magnitude so pooled averages do not conceal difficult conditions.
All generated data are synthetic and do not establish physical-system validity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from src.validation.synthetic_drift_scenarios import (
    generate_synthetic_drift_scenario,
    run_synthetic_drift_validation,
)


@dataclass(frozen=True)
class DriftSensitivityCell:
    """Summary for one fixed scenario/noise/magnitude configuration."""

    scenario_name: str
    noise_std: float
    drift_magnitude: float
    trial_count: int
    detected_trials: int
    detection_rate: float
    mean_false_alarm_fraction: float
    mean_detection_delay_detected: float | None
    interpretation: str = (
        "Synthetic stratified sensitivity result only; not physical scanner "
        "performance or hardware-failure prediction."
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DriftSensitivityReport:
    """Stratified outcomes and the frozen monitor configuration."""

    cells: tuple[DriftSensitivityCell, ...]
    seeds: tuple[int, ...]
    monitor_parameters: dict[str, float]
    change_fraction: float
    interpretation: str = (
        "Synthetic algorithmic sensitivity analysis only; no physical or "
        "clinical claim is supported."
    )

    def to_dict(self) -> dict:
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "seeds": list(self.seeds),
            "monitor_parameters": dict(self.monitor_parameters),
            "change_fraction": float(self.change_fraction),
            "interpretation": self.interpretation,
        }


def evaluate_drift_sensitivity(
    *,
    kinds: Sequence[str] = ("nominal", "noisy_nominal", "gradual_drift", "abrupt_shift"),
    seeds: Sequence[int] = (11, 23, 37, 41, 53),
    noise_levels: Sequence[float] = (0.2, 0.35, 0.6),
    drift_magnitudes: Sequence[float] = (1.0, 2.5, 4.0),
    reference_count: int = 120,
    observation_count: int = 120,
    change_fraction: float = 0.5,
    ewma_alpha: float = 0.2,
    ewma_threshold: float = 3.0,
    cusum_allowance: float = 0.5,
    cusum_threshold: float = 5.0,
) -> DriftSensitivityReport:
    """Evaluate fixed monitor settings and return non-pooled configuration cells.

    Each cell uses the same list of distinct seeds. Mean detection delay is
    conditional on detected change trials and should be interpreted together
    with the detection rate. No thresholds are tuned on these outcomes.
    """
    allowed = {"nominal", "noisy_nominal", "gradual_drift", "abrupt_shift"}
    kinds = tuple(kinds)
    seeds = tuple(int(s) for s in seeds)
    noise_levels = tuple(float(v) for v in noise_levels)
    drift_magnitudes = tuple(float(v) for v in drift_magnitudes)
    if not kinds or any(k not in allowed for k in kinds):
        raise ValueError(f"kinds must be non-empty and drawn from {sorted(allowed)}.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique.")
    if not noise_levels or any(not np.isfinite(v) or v <= 0 for v in noise_levels):
        raise ValueError("noise_levels must be non-empty, finite and positive.")
    if not drift_magnitudes or any(not np.isfinite(v) or v <= 0 for v in drift_magnitudes):
        raise ValueError("drift_magnitudes must be non-empty, finite and positive.")
    if reference_count < 5 or observation_count < 2:
        raise ValueError("reference_count must be >= 5 and observation_count >= 2.")
    if not np.isfinite(change_fraction) or not 0.0 < change_fraction < 1.0:
        raise ValueError("change_fraction must be finite and strictly between 0 and 1.")

    change_index = min(observation_count - 1, max(1, int(observation_count * change_fraction)))
    cells: list[DriftSensitivityCell] = []
    for kind in kinds:
        magnitudes = drift_magnitudes if kind in {"gradual_drift", "abrupt_shift"} else (drift_magnitudes[0],)
        for noise in noise_levels:
            for magnitude in magnitudes:
                metrics = []
                for seed in seeds:
                    scenario = generate_synthetic_drift_scenario(
                        kind,
                        reference_count=reference_count,
                        observation_count=observation_count,
                        seed=seed,
                        noise_std=noise,
                        drift_magnitude=magnitude,
                        change_index=change_index if kind in {"gradual_drift", "abrupt_shift"} else None,
                    )
                    _, metric = run_synthetic_drift_validation(
                        scenario,
                        ewma_alpha=ewma_alpha,
                        ewma_threshold=ewma_threshold,
                        cusum_allowance=cusum_allowance,
                        cusum_threshold=cusum_threshold,
                    )
                    metrics.append(metric)
                detected_delays = [m.detection_delay for m in metrics if m.detection_delay is not None]
                cells.append(
                    DriftSensitivityCell(
                        scenario_name=kind,
                        noise_std=noise,
                        drift_magnitude=magnitude,
                        trial_count=len(metrics),
                        detected_trials=len(detected_delays),
                        detection_rate=float(np.mean([m.detected for m in metrics])),
                        mean_false_alarm_fraction=float(np.mean([m.false_alarm_fraction for m in metrics])),
                        mean_detection_delay_detected=(
                            float(np.mean(detected_delays)) if detected_delays else None
                        ),
                    )
                )
    return DriftSensitivityReport(
        cells=tuple(cells),
        seeds=seeds,
        monitor_parameters={
            "ewma_alpha": float(ewma_alpha),
            "ewma_threshold": float(ewma_threshold),
            "cusum_allowance": float(cusum_allowance),
            "cusum_threshold": float(cusum_threshold),
        },
        change_fraction=float(change_fraction),
    )
