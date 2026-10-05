import numpy as np

from src.validation.evidence_vv import evaluate_controlled_degradation
from src.simulation.degradation import DegradationType


def _image():
    x = np.linspace(0.05, 0.95, 128)
    return np.tile(x, (128, 1))


def test_vv_is_reproducible():
    a = evaluate_controlled_degradation(
        _image(), scenario=DegradationType.BLUR, severity=0.7
    )
    b = evaluate_controlled_degradation(
        _image(), scenario=DegradationType.BLUR, severity=0.7
    )
    assert a == b


def test_vv_detects_nonzero_signature_change():
    result = evaluate_controlled_degradation(
        _image(), scenario=DegradationType.CONTRAST_REDUCTION, severity=0.8
    )
    assert result.signature_delta_l2 > 0.0
    assert 0.0 <= result.causal_agreement <= 1.0
    assert 0.0 <= result.fusion_score <= 1.0
    assert 0.0 <= result.disagreement <= 1.0


def test_vv_severity_is_traceable():
    low = evaluate_controlled_degradation(
        _image(), scenario=DegradationType.GAUSSIAN_NOISE, severity=0.2, seed=7
    )
    high = evaluate_controlled_degradation(
        _image(), scenario=DegradationType.GAUSSIAN_NOISE, severity=0.8, seed=7
    )
    assert high.severity > low.severity
    assert high.signature_delta_l2 >= low.signature_delta_l2
