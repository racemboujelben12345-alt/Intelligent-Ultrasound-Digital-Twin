from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class ReferenceGenerationConfig:
    size: int = 128
    n_samples: int = 40
    seed: int = 42

    base_intensity: float = 0.48
    texture_amplitude: float = 0.12
    speckle_amplitude: float = 0.08
    gaussian_noise: float = 0.015


def generate_reference_image(
    rng: np.random.Generator,
    config: ReferenceGenerationConfig,
) -> np.ndarray:
    """
    Generate one synthetic ultrasound-like image representing
    a normal operating condition.

    This is a validation/reference generator only.
    It is NOT SCAN A data.
    """

    size = config.size

    # Low-frequency anatomical/background structure
    low_frequency = rng.normal(0.0, 1.0, (size, size))
    low_frequency = gaussian_filter(
        low_frequency,
        sigma=rng.uniform(5.0, 9.0),
    )

    low_frequency -= low_frequency.min()
    low_frequency /= low_frequency.max() + 1e-12

    # Medium-scale tissue texture
    texture = rng.normal(0.0, 1.0, (size, size))
    texture = gaussian_filter(
        texture,
        sigma=rng.uniform(1.5, 3.0),
    )

    texture -= texture.mean()
    texture /= texture.std() + 1e-12

    image = (
        config.base_intensity
        + 0.18 * (low_frequency - 0.5)
        + config.texture_amplitude * texture
    )

    # Multiplicative speckle-like component
    speckle = rng.normal(
        0.0,
        config.speckle_amplitude,
        (size, size),
    )

    image = image * (1.0 + speckle)

    # Small acquisition noise
    noise = rng.normal(
        0.0,
        config.gaussian_noise,
        (size, size),
    )

    image = image + noise

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def generate_reference_set(
    config: ReferenceGenerationConfig | None = None,
) -> list[np.ndarray]:
    """
    Generate a reproducible set of normal reference images.
    """

    if config is None:
        config = ReferenceGenerationConfig()

    rng = np.random.default_rng(config.seed)

    return [
        generate_reference_image(rng, config)
        for _ in range(config.n_samples)
    ]