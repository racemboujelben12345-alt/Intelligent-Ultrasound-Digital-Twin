"""Robust sensitivity analysis combining controlled degradation and repeatability noise."""

from __future__ import annotations

from dataclasses import asdict
import numpy as np

from src.simulation.degradation import DegradationType
from src.validation.repeatability import estimate_repeatability
from src.validation.sensitivity_matrix import run_sensitivity_matrix

def run_robustness_matrix(image: np.ndarray, *, scenarios=None, severities=(0.1,0.3,0.5,0.7,0.9),
                           repeats: int = 20, noise_severity: float = 0.02, seed: int = 42,
                           multiplier: float = 3.0, params: dict | None = None) -> list[dict]:
    """Run controlled degradations and compare them with the measured simulation noise floor."""
    if multiplier <= 0:
        raise ValueError('multiplier must be positive.')
    repeatability = estimate_repeatability(image, n_repeats=repeats,
                                           noise_severity=noise_severity, seed=seed, params=params)
    rows = run_sensitivity_matrix(image, scenarios=scenarios, severities=severities,
                                  params=params, seed=seed)
    noise_floor = multiplier * repeatability.signature_l2_noise
    for row in rows:
        row['noise_floor_l2'] = float(noise_floor)
        row['robustness_margin_l2'] = float(row['signature_delta_l2'] - noise_floor)
        row['robust_detection_evidence'] = bool(row['robustness_margin_l2'] > 0.0)
    return rows

def summarize_robustness(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('rows cannot be empty.')
    out = {}
    for scenario in sorted({str(r['scenario']) for r in rows}):
        group = [r for r in rows if str(r['scenario']) == scenario]
        margins = np.asarray([r['robustness_margin_l2'] for r in group], dtype=float)
        out[scenario] = {
            'n': len(group),
            'robust_detection_rate': float(np.mean(margins > 0.0)),
            'margin_min': float(margins.min()),
            'margin_max': float(margins.max()),
            'noise_floor_l2': float(group[0]['noise_floor_l2']),
        }
    return out