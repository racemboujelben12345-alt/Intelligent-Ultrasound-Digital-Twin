"""Sequential drift monitoring with robust reference scaling.

EWMA and CUSUM are statistical early-warning baselines. Thresholds must be
calibrated for the acquisition cadence and acceptable false-alarm burden.
They do not identify a physical cause or prove a device fault.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SequentialDriftResult:
    """Per-observation standardized residuals and sequential warning signals."""

    reference_count: int
    observation_count: int
    reference_center: float
    reference_scale: float
    ewma: tuple[float, ...]
    cusum_positive: tuple[float, ...]
    cusum_negative: tuple[float, ...]
    ewma_alarm_indices: tuple[int, ...]
    cusum_alarm_indices: tuple[int, ...]
    combined_alarm_indices: tuple[int, ...]
    ewma_threshold: float
    cusum_allowance: float
    cusum_threshold: float
    interpretation: str = (
        "Statistical warning relative to the supplied reference; not proof "
        "of hardware failure or a causal diagnosis."
    )

    @property
    def first_alarm_index(self) -> int | None:
        return self.combined_alarm_indices[0] if self.combined_alarm_indices else None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["first_alarm_index"] = self.first_alarm_index
        return result


def _as_finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def monitor_sequential_drift(
    reference: Sequence[float],
    observations: Sequence[float],
    *,
    ewma_alpha: float = 0.2,
    ewma_threshold: float = 3.0,
    cusum_allowance: float = 0.5,
    cusum_threshold: float = 5.0,
) -> SequentialDriftResult:
    """Monitor a new ordered sequence against a separate reference sample.

    The reference centre is its median. Scale is 1.4826 * MAD, falling back
    to standard deviation only when MAD is numerically zero. A constant
    reference is rejected because a meaningful standardized residual cannot
    be obtained from it.

    Alarm thresholds are configuration parameters, not universal constants.
    Calibrate them using held-out nominal sequences and report false alarms
    per acquisition/session or unit time before interpreting the output.
    """
    baseline = _as_finite_vector(reference, "reference")
    current = _as_finite_vector(observations, "observations")
    if baseline.size < 5:
        raise ValueError("reference must contain at least 5 observations.")
    if not 0 < ewma_alpha <= 1:
        raise ValueError("ewma_alpha must be in (0, 1].")
    if not np.isfinite(ewma_threshold) or ewma_threshold <= 0:
        raise ValueError("ewma_threshold must be finite and positive.")
    if not np.isfinite(cusum_allowance) or cusum_allowance < 0:
        raise ValueError("cusum_allowance must be finite and non-negative.")
    if not np.isfinite(cusum_threshold) or cusum_threshold <= 0:
        raise ValueError("cusum_threshold must be finite and positive.")

    center = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - center)))
    scale = 1.4826 * mad
    if scale <= np.finfo(float).eps:
        scale = float(np.std(baseline, ddof=1))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("reference variability is too small to estimate a scale.")

    z_values = (current - center) / scale
    ewma_values: list[float] = []
    positive_values: list[float] = []
    negative_values: list[float] = []
    ewma_alarms: list[int] = []
    cusum_alarms: list[int] = []
    ewma_state = 0.0
    positive_state = 0.0
    negative_state = 0.0

    for index, z_value in enumerate(z_values):
        ewma_state = ewma_alpha * float(z_value) + (1.0 - ewma_alpha) * ewma_state
        positive_state = max(0.0, positive_state + float(z_value) - cusum_allowance)
        negative_state = max(0.0, negative_state - float(z_value) - cusum_allowance)
        ewma_values.append(ewma_state)
        positive_values.append(positive_state)
        negative_values.append(negative_state)
        if abs(ewma_state) >= ewma_threshold:
            ewma_alarms.append(index)
        if positive_state >= cusum_threshold or negative_state >= cusum_threshold:
            cusum_alarms.append(index)

    combined = tuple(sorted(set(ewma_alarms).union(cusum_alarms)))
    return SequentialDriftResult(
        reference_count=int(baseline.size),
        observation_count=int(current.size),
        reference_center=center,
        reference_scale=scale,
        ewma=tuple(ewma_values),
        cusum_positive=tuple(positive_values),
        cusum_negative=tuple(negative_values),
        ewma_alarm_indices=tuple(ewma_alarms),
        cusum_alarm_indices=tuple(cusum_alarms),
        combined_alarm_indices=combined,
        ewma_threshold=float(ewma_threshold),
        cusum_allowance=float(cusum_allowance),
        cusum_threshold=float(cusum_threshold),
    )
