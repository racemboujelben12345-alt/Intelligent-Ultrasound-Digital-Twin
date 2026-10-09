"""Separate nominal-only calibration and independent evaluation of drift thresholds.

Threshold selection is performed only on supplied no-change calibration
scenarios. Evaluation scenarios must use disjoint seeds; this is a synthetic
algorithmic protocol, not physical ultrasound-system validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

from src.validation.synthetic_drift_scenarios import (
    DriftDetectionMetrics,
    SyntheticDriftScenario,
    run_synthetic_drift_validation,
)




def _wilson_interval(successes: int, trials: int, confidence_level: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion without SciPy."""
    z = NormalDist().inv_cdf((1.0 + float(confidence_level)) / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half_width = z * np.sqrt((p * (1.0 - p) / trials) + z * z / (4.0 * trials * trials)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))

@dataclass(frozen=True)
class ThresholdCandidateScore:
    """Nominal-only alarm burden for one fixed threshold pair."""

    ewma_threshold: float
    cusum_threshold: float
    false_alarm_sequence_rate: float
    mean_false_alarm_observation_fraction: float
    alarmed_sequences: int
    false_alarm_rate_lower: float
    false_alarm_rate_upper: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DriftThresholdCalibration:
    """Frozen thresholds selected against a nominal-only calibration set."""

    ewma_alpha: float
    cusum_allowance: float
    target_false_alarm_rate: float
    confidence_level: float
    selected_ewma_threshold: float
    selected_cusum_threshold: float
    calibration_seed_ids: tuple[int, ...]
    candidate_scores: tuple[ThresholdCandidateScore, ...]
    interpretation: str = (
        "Thresholds calibrated on synthetic nominal sequences only. "
        "No physical-system or clinical validation is implied."
    )

    def to_dict(self) -> dict:
        return {
            "ewma_alpha": self.ewma_alpha,
            "cusum_allowance": self.cusum_allowance,
            "target_false_alarm_rate": self.target_false_alarm_rate,
            "confidence_level": self.confidence_level,
            "selected_ewma_threshold": self.selected_ewma_threshold,
            "selected_cusum_threshold": self.selected_cusum_threshold,
            "calibration_seed_ids": list(self.calibration_seed_ids),
            "candidate_scores": [score.to_dict() for score in self.candidate_scores],
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class CalibratedDriftEvaluation:
    """Independent-seed evaluation using a frozen calibration result."""

    trial_metrics: tuple[DriftDetectionMetrics, ...]
    nominal_sequence_false_alarm_rate: float | None
    nominal_false_alarm_rate_lower: float | None
    nominal_false_alarm_rate_upper: float | None
    change_detection_rate: float | None
    change_detection_rate_lower: float | None
    change_detection_rate_upper: float | None
    confidence_level: float
    mean_false_alarm_observation_fraction: float
    mean_detection_delay: float | None
    calibration_seed_ids: tuple[int, ...]
    evaluation_seed_ids: tuple[int, ...]
    ewma_threshold: float
    cusum_threshold: float
    interpretation: str = (
        "Independent-seed synthetic evaluation only; not physical ultrasound "
        "performance or hardware-failure prediction."
    )

    def to_dict(self) -> dict:
        return {
            "trial_metrics": [metric.to_dict() for metric in self.trial_metrics],
            "nominal_sequence_false_alarm_rate": self.nominal_sequence_false_alarm_rate,
            "nominal_false_alarm_rate_lower": self.nominal_false_alarm_rate_lower,
            "nominal_false_alarm_rate_upper": self.nominal_false_alarm_rate_upper,
            "change_detection_rate": self.change_detection_rate,
            "change_detection_rate_lower": self.change_detection_rate_lower,
            "change_detection_rate_upper": self.change_detection_rate_upper,
            "confidence_level": self.confidence_level,
            "mean_false_alarm_observation_fraction": self.mean_false_alarm_observation_fraction,
            "mean_detection_delay": self.mean_detection_delay,
            "calibration_seed_ids": list(self.calibration_seed_ids),
            "evaluation_seed_ids": list(self.evaluation_seed_ids),
            "ewma_threshold": self.ewma_threshold,
            "cusum_threshold": self.cusum_threshold,
            "interpretation": self.interpretation,
        }


def calibrate_drift_thresholds(
    nominal_scenarios: Sequence[SyntheticDriftScenario],
    candidates: Sequence[tuple[float, float]],
    *,
    target_false_alarm_rate: float = 0.05,
    confidence_level: float = 0.95,
    ewma_alpha: float = 0.2,
    cusum_allowance: float = 0.5,
) -> DriftThresholdCalibration:
    """Select the least conservative supplied threshold pair meeting a nominal FAR.

    The false-alarm rate is the fraction of calibration sequences with one or
    more alarms, not the fraction of individual observations. A Wilson binomial
    confidence interval is reported for each candidate; selection requires its
    upper bound to be at or below the target, not merely the point estimate.
    Candidates are ranked by ascending (EWMA threshold + CUSUM threshold),
    then by EWMA and CUSUM threshold. At least five nominal scenarios with
    distinct seeds are required. If no candidate meets the target, calibration
    fails explicitly. This empirical interval assumes independent Bernoulli
    sequence outcomes and is not a guarantee under dependence or domain shift.
    """
    scenarios = tuple(nominal_scenarios)
    pairs = tuple((float(e), float(c)) for e, c in candidates)
    if len(scenarios) < 5:
        raise ValueError("At least five nominal calibration scenarios are required.")
    if any(s.change_index is not None for s in scenarios):
        raise ValueError("Calibration scenarios must be nominal/no-change only.")
    seeds = tuple(int(s.seed) for s in scenarios)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Calibration scenarios must have distinct seeds.")
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("Candidates must be non-empty and threshold pairs unique.")
    if not np.isfinite(target_false_alarm_rate) or not 0.0 <= target_false_alarm_rate <= 1.0:
        raise ValueError("target_false_alarm_rate must be finite and in [0, 1].")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1].")
    if not np.isfinite(ewma_alpha) or not 0.0 < ewma_alpha <= 1.0:
        raise ValueError("ewma_alpha must be in (0, 1].")
    if not np.isfinite(cusum_allowance) or cusum_allowance < 0:
        raise ValueError("cusum_allowance must be finite and non-negative.")
    if any(not np.isfinite(e) or e <= 0 or not np.isfinite(c) or c <= 0 for e, c in pairs):
        raise ValueError("Candidate thresholds must be finite and positive.")

    scores: list[ThresholdCandidateScore] = []
    for ewma_threshold, cusum_threshold in sorted(pairs, key=lambda pair: (sum(pair), pair[0], pair[1])):
        metrics = []
        for scenario in scenarios:
            _, metric = run_synthetic_drift_validation(
                scenario,
                ewma_alpha=ewma_alpha,
                ewma_threshold=ewma_threshold,
                cusum_allowance=cusum_allowance,
                cusum_threshold=cusum_threshold,
            )
            metrics.append(metric)
        alarmed = sum(metric.alarm_count > 0 for metric in metrics)
        scores.append(
            ThresholdCandidateScore(
                ewma_threshold=ewma_threshold,
                cusum_threshold=cusum_threshold,
                false_alarm_sequence_rate=float(alarmed / len(metrics)),
                mean_false_alarm_observation_fraction=float(
                    np.mean([metric.false_alarm_fraction for metric in metrics])
                ),
                alarmed_sequences=int(alarmed),
                false_alarm_rate_lower=_wilson_interval(alarmed, len(metrics), confidence_level)[0],
                false_alarm_rate_upper=_wilson_interval(alarmed, len(metrics), confidence_level)[1],
            )
        )

    eligible = [score for score in scores if score.false_alarm_rate_upper <= target_false_alarm_rate]
    if not eligible:
        best_rate = min(score.false_alarm_sequence_rate for score in scores)
        raise ValueError(
            "No candidate meets target_false_alarm_rate; "
            f"lowest observed calibration rate was {best_rate:.6g}; no confidence upper bound met the target. Expand the "
            "pre-specified candidate grid or revise the target before evaluation."
        )
    selected = min(
        eligible,
        key=lambda score: (
            score.ewma_threshold + score.cusum_threshold,
            score.ewma_threshold,
            score.cusum_threshold,
        ),
    )
    return DriftThresholdCalibration(
        ewma_alpha=float(ewma_alpha),
        cusum_allowance=float(cusum_allowance),
        target_false_alarm_rate=float(target_false_alarm_rate),
        confidence_level=float(confidence_level),
        selected_ewma_threshold=selected.ewma_threshold,
        selected_cusum_threshold=selected.cusum_threshold,
        calibration_seed_ids=seeds,
        candidate_scores=tuple(scores),
    )


def evaluate_calibrated_thresholds(
    calibration: DriftThresholdCalibration,
    evaluation_scenarios: Sequence[SyntheticDriftScenario],
) -> CalibratedDriftEvaluation:
    """Evaluate frozen thresholds on scenarios whose seeds were not calibrated."""
    scenarios = tuple(evaluation_scenarios)
    if not scenarios:
        raise ValueError("evaluation_scenarios must not be empty.")
    eval_seeds = tuple(int(s.seed) for s in scenarios)
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("Evaluation scenarios must have distinct seeds.")
    overlap = set(calibration.calibration_seed_ids).intersection(eval_seeds)
    if overlap:
        raise ValueError(f"Calibration and evaluation seeds must be disjoint; overlap: {sorted(overlap)}.")

    metrics: list[DriftDetectionMetrics] = []
    for scenario in scenarios:
        _, metric = run_synthetic_drift_validation(
            scenario,
            ewma_alpha=calibration.ewma_alpha,
            ewma_threshold=calibration.selected_ewma_threshold,
            cusum_allowance=calibration.cusum_allowance,
            cusum_threshold=calibration.selected_cusum_threshold,
        )
        metrics.append(metric)

    nominal = [metric for metric in metrics if metric.change_index is None]
    changed = [metric for metric in metrics if metric.change_index is not None]
    delays = [metric.detection_delay for metric in changed if metric.detection_delay is not None]
    nominal_alarmed = sum(metric.alarm_count > 0 for metric in nominal)
    changed_detected = sum(metric.detected for metric in changed)
    nominal_interval = (
        _wilson_interval(nominal_alarmed, len(nominal), calibration.confidence_level)
        if nominal else (None, None)
    )
    detection_interval = (
        _wilson_interval(changed_detected, len(changed), calibration.confidence_level)
        if changed else (None, None)
    )
    return CalibratedDriftEvaluation(
        trial_metrics=tuple(metrics),
        nominal_sequence_false_alarm_rate=(
            float(nominal_alarmed / len(nominal)) if nominal else None
        ),
        nominal_false_alarm_rate_lower=nominal_interval[0],
        nominal_false_alarm_rate_upper=nominal_interval[1],
        change_detection_rate=(
            float(changed_detected / len(changed)) if changed else None
        ),
        change_detection_rate_lower=detection_interval[0],
        change_detection_rate_upper=detection_interval[1],
        confidence_level=calibration.confidence_level,
        mean_false_alarm_observation_fraction=float(
            np.mean([metric.false_alarm_fraction for metric in metrics])
        ),
        mean_detection_delay=float(np.mean(delays)) if delays else None,
        calibration_seed_ids=calibration.calibration_seed_ids,
        evaluation_seed_ids=eval_seeds,
        ewma_threshold=calibration.selected_ewma_threshold,
        cusum_threshold=calibration.selected_cusum_threshold,
    )
