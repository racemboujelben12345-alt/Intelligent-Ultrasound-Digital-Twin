from src.physics.counterfactual import perturb_physics


def test_frequency_counterfactual_changes_wavelength():
    image = [[0.5] * 16 for _ in range(16)]
    params = {"frequency_mhz": 5.0, "depth_mm": 50.0}

    result = perturb_physics(
        image,
        params,
        "frequency_mhz",
        2.0,
    )

    assert result.baseline_value == 5.0
    assert result.counterfactual_value == 10.0
    assert abs(result.wavelength_ratio - 0.5) < 1e-12


def test_depth_counterfactual_changes_prf_margin():
    image = [[0.5] * 16 for _ in range(16)]
    params = {
        "frequency_mhz": 5.0,
        "depth_mm": 50.0,
        "prf_hz": 1000.0,
    }

    result = perturb_physics(
        image,
        params,
        "depth_mm",
        2.0,
    )

    assert result.prf_margin_ratio is not None
    assert abs(result.prf_margin_ratio - 0.5) < 1e-12
