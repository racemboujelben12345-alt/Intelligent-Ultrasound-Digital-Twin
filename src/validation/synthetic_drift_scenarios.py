"""Deterministic synthetic scenarios for validating sequential drift monitors.

Synthetic data checks algorithmic behaviour under known changes. It is not
physical ultrasound-system validation and does not model all real acquisition
mechanisms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from src.prediction.sequential_drift import SequentialDriftResult, monitor_sequential_drift

ScenarioKind = Literal["nominal", "gradual_drift", "abrupt_shift", "noisy_nominal"]


@dataclass(frozen=True)
class SyntheticDriftScenario:
    """A reproducible one-dimensional reference and ordered observation series."""

    name: str
    reference: tuple[float, ...]
    observations: tuple[float, ...]
    change_index: int | None
    seed: int
    description: str

    def to_dict(self) -> dict:
        """Return JSON-friendly scenario metadata and values."""
        return asdict(self)


@dataclass(frozen=True)
class DriftDetectionMetrics:
    """Alarm burden and detection delay for one known synthetic change point.

    An alarm is an observation index flagged by the monitor. The false-alarm
    fraction is the fraction of pre-change (or all, if no change exists)
    observations that are flagged. Detection delay is measured in observations
    from the known change index to the first alarm at or after that index.
    """

    scenario_name: str
    observation_count: int
    change_index: int | None
    alarm_count: int
    false_alarm_count: int
    false_alarm_fraction: float
    detected: bool
    detection_delay: int | None
    first_alarm_index: int | None
    interpretation: str = (
        "Synthetic algorithmic evaluation only; not evidence of physical "
        "scanner performance or hardware-failure prediction."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def generate_synthetic_drift_scenario(
    kind: ScenarioKind,
    *,
    reference_count: int = 120,
    observation_count: int = 120,
    seed: int = 42,
    noise_std: float = 0.35,
    drift_magnitude: float = 2.5,
    change_index: int | None = None,
    seasonality_amplitude: float = 0.0,
) -> SyntheticDriftScenario:
    """Generate a reproducible nominal or drift scenario with known truth.

    The reference is nominal Gaussian noise around zero. Observations can be
    nominal, gradually drifting, abruptly shifted, or nominal with increased
    noise. Optional sinusoidal variation is added to observations only. For
    changed scenarios, change_index is the first affected observation.
    """
    allowed = {"nominal", "gradual_drift", "abrupt_shift", "noisy_nominal"}
    if kind not in allowed:
        raise ValueError(f"kind must be one of {sorted(allowed)}.")
    if reference_count < 5:
        raise ValueError("reference_count must be at least 5.")
    if observation_count < 1:
        raise ValueError("observation_count must be positive.")
    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be finite and positive.")
    if not np.isfinite(drift_magnitude) or drift_magnitude <= 0:
        raise ValueError("drift_magnitude must be finite and positive.")
    if not np.isfinite(seasonality_amplitude) or seasonality_amplitude < 0:
        raise ValueError("seasonality_amplitude must be finite and non-negative.")

    has_change = kind in {"gradual_drift", "abrupt_shift"}
    if change_index is None:
        change_index = max(1, observation_count // 2) if has_change else None
    if has_change and (change_index is None or not 0 <= change_index < observation_count):
        raise ValueError("change_index must be within observations for drift scenarios.")
    if not has_change and change_index is not None:
        raise ValueError("change_index must be None for scenarios without a mean shift.")

    rng = np.random.default_rng(seed)
    reference = rng.normal(0.0, noise_std, reference_count)
    observation_noise = noise_std * (2.5 if kind == "noisy_nominal" else 1.0)
    observations = rng.normal(0.0, observation_noise, observation_count)
    t = np.arange(observation_count, dtype=float)
    if seasonality_amplitude:
        period = max(8.0, observation_count / 4.0)
        observations += seasonality_amplitude * np.sin(2.0 * np.pi * t / period)

    if kind == "gradual_drift":
        ramp_length = max(1, observation_count - int(change_index))
        ramp = np.arange(observation_count - int(change_index), dtype=float) / ramp_length
        observations[int(change_index):] += drift_magnitude * ramp
        description = "Gradual positive drift begins at the known change index."
    elif kind == "abrupt_shift":
        observations[int(change_index):] += drift_magnitude
        description = "Abrupt positive mean shift begins at the known change index."
    elif kind == "noisy_nominal":
        description = "Nominal mean with elevated observation noise; no mean-change point."
    else:
        description = "Stable nominal observations; no change point."

    return SyntheticDriftScenario(
        name=kind,
        reference=tuple(float(v) for v in reference),
        observations=tuple(float(v) for v in observations),
        change_index=change_index,
        seed=int(seed),
        description=description,
    )


def evaluate_drift_detection(
    scenario: SyntheticDriftScenario,
    result: SequentialDriftResult,
) -> DriftDetectionMetrics:
    """Score alarm burden and delay against the scenario's known change point."""
    if result.observation_count != len(scenario.observations):
        raise ValueError("monitor result length must match scenario observations.")
    alarms = result.combined_alarm_indices
    if any(index < 0 or index >= result.observation_count for index in alarms):
        raise ValueError("monitor result contains an out-of-range alarm index.")

    boundary = scenario.change_index
    if boundary is None:
        pre_change_alarms = alarms
        denominator = result.observation_count
        detection_delay = None
        detected = False
    else:
        pre_change_alarms = tuple(index for index in alarms if index < boundary)
        denominator = boundary
        post_change_alarms = tuple(index for index in alarms if index >= boundary)
        detected = bool(post_change_alarms)
        detection_delay = post_change_alarms[0] - boundary if detected else None

    false_fraction = len(pre_change_alarms) / denominator if denominator else 0.0
    return DriftDetectionMetrics(
        scenario_name=scenario.name,
        observation_count=result.observation_count,
        change_index=boundary,
        alarm_count=len(alarms),
        false_alarm_count=len(pre_change_alarms),
        false_alarm_fraction=float(false_fraction),
        detected=detected,
        detection_delay=detection_delay,
        first_alarm_index=result.first_alarm_index,
    )


def run_synthetic_drift_validation(
    scenario: SyntheticDriftScenario,
    *,
    ewma_alpha: float = 0.2,
    ewma_threshold: float = 3.0,
    cusum_allowance: float = 0.5,
    cusum_threshold: float = 5.0,
) -> tuple[SequentialDriftResult, DriftDetectionMetrics]:
    """Run the current EWMA/CUSUM monitor and score its synthetic outcome."""
    result = monitor_sequential_drift(
        scenario.reference,
        scenario.observations,
        ewma_alpha=ewma_alpha,
        ewma_threshold=ewma_threshold,
        cusum_allowance=cusum_allowance,
        cusum_threshold=cusum_threshold,
    )
    return result, evaluate_drift_detection(scenario, result)
