import numpy as np

from src.physics.ultrasound import compute_ultrasound_physics
from src.image_analysis.digital_signature import build_digital_signature_from_image


def test_physics_profile_is_finite_and_frequency_aware():
    image = np.tile(np.linspace(0.8, 0.2, 128, dtype=np.float32), (128, 1))
    profile = compute_ultrasound_physics(
        image,
        params={"frequency": 7.5, "depth": 60.0},
    )
    assert profile.metadata_complete
    assert profile.wavelength_mm > 0
    assert profile.axial_resolution_mm > 0
    assert 0.0 <= profile.physical_consistency_score <= 1.0


def test_signature_contains_physics_features():
    image = np.random.default_rng(42).uniform(0.1, 0.9, (128, 128)).astype(np.float32)
    signature = build_digital_signature_from_image(
        image,
        source="simulated",
        params={"frequency": 5.0, "depth": 50.0},
    )
    vector = signature.to_vector()
    assert len(vector) >= 20
    assert np.all(np.isfinite(vector))
    assert signature.wavelength_mm > 0
