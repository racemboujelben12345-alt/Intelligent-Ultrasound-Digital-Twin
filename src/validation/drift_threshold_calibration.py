"""Separate nominal-only calibration and independent evaluation of drift thresholds.

Threshold selection is performed only on supplied no-change calibration
scenarios. Evaluation scenarios must use disjoint seeds; this is a synthetic
algorithmic protocol, not physical ultrasound-system validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from src.validation.synthetic_drift_scenarios import (
    DriftDetectionMetrics,
    SyntheticDriftScenario,
    run_synthetic_drift_validation,
)


@dataclass(frozen=True)
class ThresholdCandidateScore:
    """Nominal-only alarm burden for one fixed threshold pair."""

    ewma_threshold: float
    cusum_threshold: float
    false_alarm_sequence_rate: float
    mean_false_alarm_observation_fraction: float
    alarmed_sequences: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DriftThresholdCalibration:
    """Frozen thresholds selected against a nominal-only calibration set."""

    ewma_alpha: float
    cusum_allowance: float
    target_false_alarm_rate: float
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
    change_detection_rate: float | None
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
            "change_detection_rate": self.change_detection_rate,
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
    ewma_alpha: float = 0.2,
    cusum_allowance: float = 0.5,
) -> DriftThresholdCalibration:
    """Select the least conservative supplied threshold pair meeting a nominal FAR.

    The false-alarm rate is the fraction of calibration sequences with one or
    more alarms, not the fraction of individual observations. Candidates are
    ranked by ascending (EWMA threshold + CUSUM threshold), then by EWMA and
    CUSUM threshold. At least five nominal scenarios with distinct seeds are
    required. If no candidate meets the target, calibration fails explicitly.
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
            )
        )

    eligible = [score for score in scores if score.false_alarm_sequence_rate <= target_false_alarm_rate]
    if not eligible:
        best_rate = min(score.false_alarm_sequence_rate for score in scores)
        raise ValueError(
            "No candidate meets target_false_alarm_rate; "
            f"lowest observed calibration rate was {best_rate:.6g}. Expand the "
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
    return CalibratedDriftEvaluation(
        trial_metrics=tuple(metrics),
        nominal_sequence_false_alarm_rate=(
            float(np.mean([metric.alarm_count > 0 for metric in nominal])) if nominal else None
        ),
        change_detection_rate=(
            float(np.mean([metric.detected for metric in changed])) if changed else None
        ),
        mean_false_alarm_observation_fraction=float(
            np.mean([metric.false_alarm_fraction for metric in metrics])
        ),
        mean_detection_delay=float(np.mean(delays)) if delays else None,
        calibration_seed_ids=calibration.calibration_seed_ids,
        evaluation_seed_ids=eval_seeds,
        ewma_threshold=calibration.selected_ewma_threshold,
        cusum_threshold=calibration.selected_cusum_threshold,
    )
