"""Experimental repeatability metrics from observed ultrasound images.

Unlike the synthetic-noise estimator in repeatability.py, this module summarizes
actual repeated acquisitions supplied by the caller. Results depend on protocol
and metadata and are not, by themselves, proof of scanner performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.image_analysis.digital_signature import build_digital_signature_from_image


@dataclass(frozen=True)
class ExperimentalRepeatabilityResult:
    """Descriptive feature variability for observed acquisitions."""

    n_acquisitions: int
    n_sessions: int
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    feature_cv: tuple[float, ...]
    within_session_std_mean: float | None
    between_session_mean_range: float | None
    interpretation: str = (
        "Descriptive variability of supplied acquisitions only; not proof of "
        "scanner performance, a fault, or generalizable physical repeatability."
    )

    def to_dict(self) -> dict:
        return {
            "n_acquisitions": self.n_acquisitions,
            "n_sessions": self.n_sessions,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_std": list(self.feature_std),
            "feature_cv": list(self.feature_cv),
            "within_session_std_mean": self.within_session_std_mean,
            "between_session_mean_range": self.between_session_mean_range,
            "interpretation": self.interpretation,
        }


def analyze_experimental_repeatability(
    images: Sequence[np.ndarray],
    *,
    session_ids: Sequence[str] | None = None,
    params: dict | None = None,
) -> ExperimentalRepeatabilityResult:
    """Summarize signatures from at least three observed repeated acquisitions.

    Images should come from a documented, comparable acquisition protocol. If
    session IDs are provided, within-session feature SD and the range of session
    means are reported descriptively. Session summaries require at least two
    sessions with at least two images in each session.
    """
    if len(images) < 3:
        raise ValueError("At least 3 observed acquisitions are required.")

    arrays = [np.asarray(image) for image in images]
    if any(image.ndim != 2 for image in arrays):
        raise ValueError("Every acquisition must be a 2D grayscale image.")
    if any(not np.all(np.isfinite(image)) for image in arrays):
        raise ValueError("Acquisitions must not contain NaN or Inf.")
    if any(image.shape != arrays[0].shape for image in arrays):
        raise ValueError("All images must have the same dimensions.")
    if session_ids is not None and len(session_ids) != len(arrays):
        raise ValueError("session_ids length must match the number of images.")
    if session_ids is not None and any(not str(value).strip() for value in session_ids):
        raise ValueError("Session IDs must be non-empty strings.")

    signatures = [
        build_digital_signature_from_image(image, source="experimental", params=params)
        for image in arrays
    ]
    names = tuple(signatures[0].feature_names)
    if any(tuple(signature.feature_names) != names for signature in signatures):
        raise ValueError("Digital signature feature schemas do not match.")

    matrix = np.asarray([signature.to_vector() for signature in signatures], dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Computed feature matrix contains NaN or Inf.")

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=1)
    scales = np.maximum(np.abs(means), 1e-8)
    cvs = stds / scales

    within_session_std_mean = None
    between_session_mean_range = None
    n_sessions = 1
    if session_ids is not None:
        groups: dict[str, list[int]] = {}
        for index, session in enumerate(session_ids):
            groups.setdefault(str(session), []).append(index)
        n_sessions = len(groups)
        eligible = [matrix[idx] for idx in groups.values() if len(idx) >= 2]
        if len(groups) >= 2 and len(eligible) == len(groups):
            session_stds = np.asarray([group.std(axis=0, ddof=1) for group in eligible])
            session_means = np.asarray([matrix[idx].mean(axis=0) for idx in groups.values()])
            within_session_std_mean = float(session_stds.mean())
            between_session_mean_range = float(np.mean(np.ptp(session_means, axis=0)))

    return ExperimentalRepeatabilityResult(
        n_acquisitions=len(arrays),
        n_sessions=n_sessions,
        feature_names=names,
        feature_mean=tuple(float(value) for value in means),
        feature_std=tuple(float(value) for value in stds),
        feature_cv=tuple(float(value) for value in cvs),
        within_session_std_mean=within_session_std_mean,
        between_session_mean_range=between_session_mean_range,
    )
