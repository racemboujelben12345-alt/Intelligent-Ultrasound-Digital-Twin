import numpy as np
import pytest

from src.validation.experimental_repeatability import analyze_experimental_repeatability


def _image(offset=0.0):
    x = np.linspace(0.05, 0.95, 64, dtype=np.float32)
    return np.clip(np.tile(x, (64, 1)) + offset, 0.0, 1.0)


def test_experimental_repeatability_uses_observed_images_and_is_reproducible():
    images = [_image(0.0), _image(0.001), _image(-0.001), _image(0.002)]
    first = analyze_experimental_repeatability(images, session_ids=["A", "A", "B", "B"])
    second = analyze_experimental_repeatability(images, session_ids=["A", "A", "B", "B"])

    assert first.n_acquisitions == 4
    assert first.n_sessions == 2
    assert len(first.feature_names) == len(first.feature_std)
    assert np.all(np.asarray(first.feature_std) >= 0)
    assert first.within_session_std_mean is not None
    assert first.between_session_mean_range is not None
    assert first.to_dict() == second.to_dict()
    assert "not proof" in first.interpretation


def test_experimental_repeatability_requires_three_images():
    with pytest.raises(ValueError, match="At least 3"):
        analyze_experimental_repeatability([_image(), _image()])


def test_experimental_repeatability_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same dimensions"):
        analyze_experimental_repeatability(
            [_image(), _image(), np.zeros((32, 32), dtype=np.float32)]
        )


def test_experimental_repeatability_rejects_wrong_session_count():
    with pytest.raises(ValueError, match="session_ids length"):
        analyze_experimental_repeatability([_image(), _image(), _image()], session_ids=["A", "B"])
