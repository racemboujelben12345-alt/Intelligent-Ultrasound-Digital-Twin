"""Planning aid for nominal drift-calibration sample sizes.

This module plans the number of independent nominal sequences needed for a
Wilson upper confidence bound on the sequence-level false-alarm probability.
It is a planning calculation, not evidence of achieved system performance.
"""
from __future__ import annotations

from statistics import NormalDist


def _wilson_upper(successes: int, trials: int, confidence_level: float) -> float:
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half_width = z * ((p * (1.0 - p) / trials) + z * z / (4.0 * trials * trials)) ** 0.5 / denominator
    return min(1.0, center + half_width)


def required_nominal_sequences(
    target_false_alarm_rate: float,
    *,
    confidence_level: float = 0.95,
    assumed_false_alarm_sequences: int = 0,
    max_sequences: int = 100_000,
) -> int:
    """Return the smallest sequence count whose Wilson upper bound meets target.

    The assumed alarm count is held fixed while the planned total sequence count
    increases. Use zero to answer: "How many independent nominal sequences would
    be needed if no sequence alarms?" The function assumes independent
    sequence-level Bernoulli outcomes and does not estimate a real-world rate.
    """
    if not 0.0 < target_false_alarm_rate < 1.0:
        raise ValueError("target_false_alarm_rate must be in (0, 1).")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
    if isinstance(assumed_false_alarm_sequences, bool) or not isinstance(assumed_false_alarm_sequences, int):
        raise ValueError("assumed_false_alarm_sequences must be a non-negative integer.")
    if assumed_false_alarm_sequences < 0:
        raise ValueError("assumed_false_alarm_sequences must be a non-negative integer.")
    if isinstance(max_sequences, bool) or not isinstance(max_sequences, int) or max_sequences < 1:
        raise ValueError("max_sequences must be a positive integer.")
    if assumed_false_alarm_sequences >= max_sequences:
        raise ValueError("max_sequences must exceed assumed_false_alarm_sequences.")

    for n in range(max(1, assumed_false_alarm_sequences), max_sequences + 1):
        if _wilson_upper(assumed_false_alarm_sequences, n, confidence_level) <= target_false_alarm_rate:
            return n
    raise ValueError(
        f"No sample size up to {max_sequences} meets the target under the "
        "specified assumed false-alarm count and confidence level."
    )
