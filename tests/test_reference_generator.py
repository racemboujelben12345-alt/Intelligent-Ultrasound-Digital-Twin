import numpy as np

from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)


def main():
    config = ReferenceGenerationConfig(
        size=128,
        n_samples=40,
        seed=42,
    )

    images = generate_reference_set(config)

    print()
    print("REFERENCE GENERATOR V2")
    print("-" * 60)

    print(f"Images       : {len(images)}")
    print(f"Shape        : {images[0].shape}")
    print(f"Dtype        : {images[0].dtype}")
    print(f"Global min   : {min(image.min() for image in images):.4f}")
    print(f"Global max   : {max(image.max() for image in images):.4f}")

    means = np.array([image.mean() for image in images])
    stds = np.array([image.std() for image in images])

    print()
    print("REFERENCE VARIABILITY")
    print("-" * 60)
    print(f"Mean intensity : {means.mean():.4f} ± {means.std():.4f}")
    print(f"Image std      : {stds.mean():.4f} ± {stds.std():.4f}")

    if len(images) != config.n_samples:
        raise AssertionError("Nombre d'images incorrect.")

    for image in images:
        if image.shape != (config.size, config.size):
            raise AssertionError("Dimensions incorrectes.")

        if image.dtype != np.float32:
            raise AssertionError("dtype incorrect.")

        if not np.isfinite(image).all():
            raise AssertionError("Image contenant NaN ou Inf.")

        if image.min() < 0.0 or image.max() > 1.0:
            raise AssertionError("Image hors de [0,1].")

    if np.std(means) <= 1e-6:
        raise AssertionError(
            "La baseline ne présente pratiquement aucune variabilité."
        )

    if np.mean(stds) <= 1e-3:
        raise AssertionError(
            "Les images sont trop uniformes."
        )

    print()
    print("REFERENCE GENERATOR V2 OK")


if __name__ == "__main__":
    main()