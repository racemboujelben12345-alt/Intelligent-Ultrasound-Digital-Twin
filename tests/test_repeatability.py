import numpy as np
import pytest

from src.validation.repeatability import estimate_repeatability, robustness_margin

def _image():
    x = np.linspace(0.05, 0.95, 64)
    return np.tile(x, (64, 1))

def test_repeatability_is_reproducible_and_bounded():
    a = estimate_repeatability(_image(), n_repeats=6, seed=7)
    b = estimate_repeatability(_image(), n_repeats=6, seed=7)
    assert np.allclose(a.feature_mean, b.feature_mean)
    assert np.all(a.feature_std >= 0)
    assert a.mean_relative_variability >= 0
    assert a.max_relative_variability >= 0
    assert a.signature_l2_noise >= 0

def test_robustness_margin_tracks_noise_floor():
    r = estimate_repeatability(_image(), n_repeats=6, seed=7)
    assert robustness_margin(r, 10 * r.signature_l2_noise) > 0
    assert robustness_margin(r, 0.0) <= 0

def test_invalid_repeat_count():
    with pytest.raises(ValueError):
        estimate_repeatability(_image(), n_repeats=2)