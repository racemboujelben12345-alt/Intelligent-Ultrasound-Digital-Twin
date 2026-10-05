"""Sensitivity and robustness matrix for controlled Digital Twin simulations."""

from __future__ import annotations

from dataclasses import asdict
import numpy as np

from src.simulation.degradation import DegradationType
from src.validation.evidence_vv import evaluate_controlled_degradation

DEFAULT_SEVERITIES = (0.1, 0.3, 0.5, 0.7, 0.9)

def run_sensitivity_matrix(
    image: np.ndarray,
    *,
    scenarios: tuple[DegradationType, ...] | None = None,
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
    params: dict | None = None,
    seed: int = 42,
) -> list[dict]:
    """Evaluate every scenario/severity pair with deterministic seeds."""
    if scenarios is None:
        scenarios = tuple(DegradationType)
    if not severities:
        raise ValueError('At least one severity is required.')
    if any(not 0.0 <= float(s) <= 1.0 for s in severities):
        raise ValueError('All severities must be in [0,1].')

    rows = []
    for scenario in scenarios:
        for severity in severities:
            record = evaluate_controlled_degradation(
                image, scenario=scenario, severity=float(severity),
                params=params, seed=seed,
            )
            rows.append(asdict(record))
    return rows

def summarize_sensitivity_matrix(rows: list[dict]) -> dict:
    """Return scenario-level robustness indicators."""
    if not rows:
        raise ValueError('rows cannot be empty.')
    summary = {}
    for scenario in sorted({str(r['scenario']) for r in rows}):
        group = [r for r in rows if str(r['scenario']) == scenario]
        deltas = np.asarray([r['signature_delta_l2'] for r in group], dtype=float)
        fusions = np.asarray([r['fusion_score'] for r in group], dtype=float)
        causal = np.asarray([r['causal_agreement'] for r in group], dtype=float)
        summary[scenario] = {
            'n': int(len(group)),
            'delta_min': float(deltas.min()),
            'delta_max': float(deltas.max()),
            'fusion_min': float(fusions.min()),
            'fusion_max': float(fusions.max()),
            'causal_mean': float(causal.mean()),
            'nonzero_response_rate': float(np.mean(deltas > 0.0)),
        }
    return summary