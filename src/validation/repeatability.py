"""Repeatability and noise-robustness analysis for Digital Twin signatures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from src.image_analysis.digital_signature import build_digital_signature_from_image
from src.simulation.degradation import DegradationType, apply_degradation

@dataclass(frozen=True)
class RepeatabilityResult:
    n_repeats: int
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_cv: np.ndarray
    mean_relative_variability: float
    max_relative_variability: float
    signature_l2_noise: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ('feature_mean','feature_std','feature_cv'):
            d[key] = d[key].tolist()
        return d

def estimate_repeatability(image: np.ndarray, *, n_repeats: int = 20,
                           noise_severity: float = 0.02, seed: int = 42,
                           params: dict | None = None) -> RepeatabilityResult:
    if n_repeats < 3:
        raise ValueError('n_repeats must be at least 3.')
    if not 0.0 <= noise_severity <= 1.0:
        raise ValueError('noise_severity must be in [0,1].')
    signatures = []
    for i in range(n_repeats):
        degraded = apply_degradation(image, DegradationType.GAUSSIAN_NOISE,
                                      noise_severity, seed=seed + i)
        sig = build_digital_signature_from_image(degraded.image, source='simulated', params=params)
        signatures.append(sig.to_vector())
    X = np.asarray(signatures, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    scale = np.maximum(np.abs(mean), 1e-8)
    cv = std / scale
    centered = X - mean
    l2_noise = float(np.mean(np.linalg.norm(centered, axis=1)))
    names = tuple(sig.feature_names)
    return RepeatabilityResult(n_repeats, names, mean, std, cv,
                              float(np.mean(cv)), float(np.max(cv)), l2_noise)

def robustness_margin(baseline: RepeatabilityResult, observed_delta_l2: float,
                      multiplier: float = 3.0) -> float:
    """Positive margin means the observed change exceeds repeatability noise."""
    if multiplier <= 0:
        raise ValueError('multiplier must be positive.')
    return float(observed_delta_l2 - multiplier * baseline.signature_l2_noise)