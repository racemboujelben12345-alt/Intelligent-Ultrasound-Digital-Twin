import numpy as np

from src.simulation.degradation import DegradationType
from src.validation.sensitivity_matrix import run_sensitivity_matrix, summarize_sensitivity_matrix

def _image():
    x = np.linspace(0.05, 0.95, 96)
    return np.tile(x, (96, 1))

def test_matrix_covers_all_scenarios_and_severities():
    severities = (0.2, 0.5, 0.8)
    rows = run_sensitivity_matrix(_image(), severities=severities, seed=11)
    assert len(rows) == len(tuple(DegradationType)) * len(severities)
    assert {r['scenario'] for r in rows} == {d.value for d in DegradationType}

def test_matrix_is_reproducible():
    a = run_sensitivity_matrix(_image(), severities=(0.3, 0.7), seed=11)
    b = run_sensitivity_matrix(_image(), severities=(0.3, 0.7), seed=11)
    assert a == b

def test_matrix_summary_is_bounded():
    rows = run_sensitivity_matrix(_image(), severities=(0.2, 0.5, 0.8), seed=11)
    summary = summarize_sensitivity_matrix(rows)
    assert set(summary) == {d.value for d in DegradationType}
    for metrics in summary.values():
        assert 0.0 <= metrics['nonzero_response_rate'] <= 1.0
        assert 0.0 <= metrics['fusion_min'] <= metrics['fusion_max'] <= 1.0
        assert 0.0 <= metrics['causal_mean'] <= 1.0