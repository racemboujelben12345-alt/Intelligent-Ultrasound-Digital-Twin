import numpy as np

from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)

from src.simulation.degradation import (
    DegradationType,
    apply_degradation,
)

from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)


def main():

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    image = generate_reference_set(config)[-1]

    print()
    print("=" * 90)
    print("DIGITAL SIGNATURE CONTRAST INTEGRITY TEST")
    print("=" * 90)

    for severity in (0.0, 0.2, 0.5, 0.8, 1.0):

        result = apply_degradation(
            image=image,
            degradation_type=DegradationType.CONTRAST_REDUCTION,
            severity=severity,
            seed=42,
        )

        signature = build_digital_signature_from_image(
            result.image,
            source="simulated",
        )

        print()
        print(f"SEVERITY = {severity:.1f}")
        print("-" * 90)

        print(f"IMAGE std              : {result.image.std():.8f}")
        print(f"SIGNATURE std_intensity : {signature.std_intensity:.8f}")

        print(f"IMAGE mean             : {result.image.mean():.8f}")
        print(f"SIGNATURE mean         : {signature.mean_intensity:.8f}")

        print(f"SIGNATURE rms_contrast : {signature.rms_contrast:.8f}")
        print(f"SIGNATURE dynamic_range: {signature.dynamic_range:.8f}")

        print(f"SIGNATURE entropy      : {signature.entropy:.8f}")
        print(f"SIGNATURE edge_density : {signature.edge_density:.8f}")
        print(
            f"SIGNATURE sharpness    : "
            f"{signature.sharpness_laplacian:.8f}"
        )

        if not np.isclose(
            result.image.std(),
            signature.std_intensity,
            rtol=1e-4,
            atol=1e-6,
        ):
            raise AssertionError(
                "Incohérence entre image.std() "
                "et DigitalSignature.std_intensity."
            )

    print()
    print("=" * 90)
    print("DIGITAL SIGNATURE CONTRAST INTEGRITY TEST OK")
    print("=" * 90)


if __name__ == "__main__":
    main()