"""Regression tests for the Digital Signature feature contract."""

import numpy as np

from src.image_analysis.digital_signature import (
    FEATURE_ORDER,
    build_digital_signature_from_image,
)
from src.signature.baseline import DEFAULT_FEATURES


def _sample_signature():
    image = np.linspace(
        0.05,
        0.95,
        num=128 * 128,
        dtype=np.float32,
    ).reshape(128, 128)
    return build_digital_signature_from_image(
        image,
        source="simulated",
    )


def test_canonical_feature_order_matches_signature_vector():
    signature = _sample_signature()
    values = signature.numeric_values()

    assert len(FEATURE_ORDER) == 23
    assert tuple(values) == FEATURE_ORDER
    assert signature.feature_names == FEATURE_ORDER
    assert signature.to_vector().shape == (len(FEATURE_ORDER),)
    assert np.isfinite(signature.to_vector()).all()


def test_default_baseline_features_are_valid_signature_features():
    signature = _sample_signature()

    assert len(DEFAULT_FEATURES) == 13
    assert len(set(DEFAULT_FEATURES)) == len(DEFAULT_FEATURES)
    assert set(DEFAULT_FEATURES).issubset(set(FEATURE_ORDER))
    assert signature.to_vector(DEFAULT_FEATURES).shape == (
        len(DEFAULT_FEATURES),
    )
