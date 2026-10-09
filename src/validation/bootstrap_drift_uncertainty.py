"""Bootstrap uncertainty summaries for independent synthetic drift trials.

Intervals describe variability across the supplied trial-level values. They do
not establish physical ultrasound-system performance or account for dependence
between trials, scenario selection, or domain shift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapMeanInterval:
    """Percentile-bootstrap interval for a sample mean."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    sample_size: int
    resamples: int
    seed: int
    interpretation: str = (
        "Bootstrap uncertainty across supplied trial-level values only; "
        "not physical-system validation."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> BootstrapMeanInterval:
    """Estimate a mean and percentile-bootstrap interval with replacement.

    The independent sampling unit must be the trial (not each observation
    within a time series). Use pre-specified, independently generated trials;
    do not use this interval to hide dependence or scenario-selection bias.
    """
    data = np.asarray(tuple(values), dtype=float)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("values must contain at least two trial-level values.")
    if not np.all(np.isfinite(data)):
        raise ValueError("values must all be finite.")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 100:
        raise ValueError("n_resamples must be an integer >= 100.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(n_resamples, data.size))
    means = data[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return BootstrapMeanInterval(
        estimate=float(data.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        sample_size=int(data.size),
        resamples=n_resamples,
        seed=seed,
    )
