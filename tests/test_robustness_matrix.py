import numpy as np

from src.validation.robustness_matrix import run_robustness_matrix, summarize_robustness

def _image():
    x = np.linspace(0.05, 0.95, 64)
    return np.tile(x, (64, 1))

def test_robustness_matrix_is_deterministic():
    a = run_robustness_matrix(_image(), severities=(0.2, 0.8), repeats=5, seed=4)
    b = run_robustness_matrix(_image(), severities=(0.2, 0.8), repeats=5, seed=4)
    assert a == b

def test_robustness_summary_is_bounded():
    rows = run_robustness_matrix(_image(), severities=(0.2, 0.5, 0.8), repeats=5, seed=4)
    summary = summarize_robustness(rows)
    for value in summary.values():
        assert 0.0 <= value['robust_detection_rate'] <= 1.0
        assert value['noise_floor_l2'] >= 0.0
        assert value['margin_min'] <= value['margin_max']