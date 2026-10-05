"""Physics-to-image causal engineering map.

This module does not claim that an observed image feature proves a hardware fault.
It encodes expected directional relationships used for simulation design,
explainability, and validation planning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalHypothesis:
    mechanism: str
    physical_effect: str
    expected_image_effects: tuple[str, ...]
    signature_features: tuple[str, ...]
    evidence_type: str = "hypothesis"


CAUSAL_HYPOTHESES = (
    CausalHypothesis(
        "focus_shift_or_defocus",
        "beam width increases / focal response changes",
        ("sharpness decreases", "edge response decreases"),
        ("sharpness_laplacian", "edge_density", "gradient_mean"),
    ),
    CausalHypothesis(
        "electronic_noise_increase",
        "noise floor increases",
        ("background dispersion increases", "SNR-related quality decreases"),
        ("std_intensity", "coefficient_variation", "speckle_proxy"),
    ),
    CausalHypothesis(
        "sensitivity_or_element_response_change",
        "local echo sensitivity changes",
        ("spatial non-uniformity increases", "local intensity response changes"),
        ("uniformity", "mean_intensity", "std_intensity", "edge_density"),
    ),
    CausalHypothesis(
        "attenuation_like_depth_rolloff",
        "depth-dependent signal decreases more strongly",
        ("far-field response decreases", "penetration-related response changes"),
        ("attenuation_proxy_db_cm_mhz", "axial_intensity_slope",
         "depth_uniformity", "near_field_energy_ratio"),
    ),
    CausalHypothesis(
        "pulse_or_bandwidth_change",
        "effective spatial pulse length changes",
        ("axial resolution proxy changes"),
        ("axial_resolution_mm", "wavelength_mm"),
    ),
    CausalHypothesis(
        "acquisition_timing_inconsistency",
        "reported PRF conflicts with simple depth timing bound",
        ("metadata/physics consistency warning"),
        ("theoretical_max_prf_hz", "prf_depth_margin",
         "physical_consistency_score"),
    ),
)


def get_causal_hypothesis(mechanism: str) -> CausalHypothesis:
    """Return a documented hypothesis by mechanism name."""
    for item in CAUSAL_HYPOTHESES:
        if item.mechanism == mechanism:
            return item
    raise KeyError(f"Unknown causal mechanism: {mechanism}")


def causal_feature_map() -> dict[str, tuple[str, ...]]:
    """Return mechanism -> affected digital-signature features."""
    return {
        item.mechanism: item.signature_features
        for item in CAUSAL_HYPOTHESES
    }


def expected_feature_directions(mechanism: str) -> dict[str, str]:
    """Return qualitative expected directions where defensible.

    Directions are deliberately qualitative because the project has not yet
    established scanner-specific calibration coefficients.
    """
    mapping = {
        "focus_shift_or_defocus": {
            "sharpness_laplacian": "decrease",
            "edge_density": "decrease",
            "gradient_mean": "decrease",
        },
        "electronic_noise_increase": {
            "std_intensity": "increase",
            "coefficient_variation": "increase",
        },
        "sensitivity_or_element_response_change": {
            "uniformity": "decrease",
        },
        "attenuation_like_depth_rolloff": {
            "attenuation_proxy_db_cm_mhz": "increase",
            "depth_uniformity": "decrease",
        },
        "pulse_or_bandwidth_change": {
            "axial_resolution_mm": "change",
        },
        "acquisition_timing_inconsistency": {
            "prf_depth_margin": "decrease",
            "physical_consistency_score": "decrease",
        },
    }
    return mapping.get(mechanism, {})
