import numpy as np

from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)

from src.simulation.scenarios import build_scenario
from src.simulation.runner import run_scenario

from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)


def main():

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    images = generate_reference_set(config)

    image = images[-1]

    scenario = build_scenario("contrast_progression")

    steps = run_scenario(
        image,
        scenario,
    )

    print()
    print("=" * 90)
    print("CONTRAST DEGRADATION → IMAGE → DIGITAL SIGNATURE")
    print("=" * 90)

    print(
        f"{'severity':>8} | "
        f"{'mean':>10} | "
        f"{'std':>10} | "
        f"{'rms_contrast':>14} | "
        f"{'dynamic_range':>14}"
    )

    print("-" * 90)

    for step in steps:

        signature = build_digital_signature_from_image(
            step.result.image,
            source="simulated",
        )

        print(
            f"{step.severity:8.1f} | "
            f"{signature.mean_intensity:10.5f} | "
            f"{signature.std_intensity:10.5f} | "
            f"{signature.rms_contrast:14.5f} | "
            f"{signature.dynamic_range:14.5f}"
        )

    print()
    print("CONTRAST DIAGNOSTIC V2 OK")


if __name__ == "__main__":
    main()