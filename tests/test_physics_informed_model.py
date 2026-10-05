import numpy as np

from src.physics.ultrasound import compute_ultrasound_physics


def test_pulse_duration_drives_axial_resolution():
    image = np.full((128, 128), 0.5, dtype=np.float32)
    profile = compute_ultrasound_physics(
        image,
        params={
            "frequency": 5.0,
            "depth": 50.0,
            "pulse_duration_us": 0.5,
        },
    )

    expected_spl_mm = 1540.0 * 0.5e-3
    assert np.isclose(profile.spatial_pulse_length_mm, expected_spl_mm)
    assert np.isclose(profile.axial_resolution_mm, expected_spl_mm / 2.0)


def test_prf_depth_consistency_is_reported():
    image = np.full((128, 128), 0.5, dtype=np.float32)
    profile = compute_ultrasound_physics(
        image,
        params={
            "frequency": 5.0,
            "depth": 50.0,
            "prf_hz": 1000.0,
        },
    )

    assert profile.theoretical_max_prf_hz is not None
    assert profile.prf_depth_margin is not None
    assert profile.theoretical_max_prf_hz > 1000.0
    assert profile.prf_depth_margin > 1.0
