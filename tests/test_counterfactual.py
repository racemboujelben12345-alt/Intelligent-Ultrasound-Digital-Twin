import numpy as np

from src.digital_twin.counterfactual import run_counterfactual
from src.image_analysis.digital_signature import build_digital_signature_from_image
from src.simulation.degradation import DegradationType


def test_counterfactual_is_reproducible():
    image = np.tile(np.linspace(0.1, 0.9, 64), (64, 1))
    params = {"frequency_mhz": 5.0, "depth_mm": 50.0, "prf_hz": 1000.0}

    observed = build_digital_signature_from_image(
        image, source="simulated", params=params
    )

    a = run_counterfactual(
        image, observed, DegradationType.BLUR, 0.6, seed=123, params=params
    )
    b = run_counterfactual(
        image, observed, DegradationType.BLUR, 0.6, seed=123, params=params
    )

    assert a.l2_signature_delta == b.l2_signature_delta
    assert a.changed_features == b.changed_features
    assert a.severity == 0.6


def test_counterfactual_produces_traceable_change():
    image = np.tile(np.linspace(0.1, 0.9, 64), (64, 1))
    observed = build_digital_signature_from_image(
        image, source="simulated", params={"frequency_mhz": 5.0, "depth_mm": 50.0}
    )

    result = run_counterfactual(
        image, observed, DegradationType.BLUR, 0.8, seed=42,
    )

    assert result.scenario == "blur"
    assert result.observed_source == "simulated"
    assert result.l2_signature_delta > 0.0
    assert np.isfinite(result.l2_signature_delta)
