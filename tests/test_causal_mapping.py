from src.physics.causal_mapping import (
    causal_feature_map,
    expected_feature_directions,
    get_causal_hypothesis,
)


def test_causal_hypothesis_is_traceable():
    item = get_causal_hypothesis("focus_shift_or_defocus")
    assert item.physical_effect
    assert "sharpness_laplacian" in item.signature_features
    assert item.evidence_type == "hypothesis"


def test_causal_map_is_deterministic():
    mapping = causal_feature_map()
    assert "electronic_noise_increase" in mapping
    assert "attenuation_like_depth_rolloff" in mapping


def test_directions_are_qualitative_not_numeric():
    directions = expected_feature_directions("acquisition_timing_inconsistency")
    assert directions["prf_depth_margin"] == "decrease"
    assert directions["physical_consistency_score"] == "decrease"
