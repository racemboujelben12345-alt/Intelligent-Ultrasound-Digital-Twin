import numpy as np

from src.simulation.scenarios import build_scenario
from src.simulation.runner import run_scenario
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)


def main():
    # Image de départ
    image = np.ones((128, 128), dtype=np.float32) * 0.5

    # Scénario de dégradation
    scenario = build_scenario("speckle_progression")

    # Simulation
    steps = run_scenario(image, scenario)

    print("level | severity | mean | std | speckle")
    print("-" * 50)

    for step in steps:
        signature = build_digital_signature_from_image(
            step.result.image,
            source="simulated",
        )

        print(
            f"{step.level:5d} | "
            f"{step.severity:.1f}      | "
            f"{signature.mean_intensity:.4f} | "
            f"{signature.std_intensity:.4f} | "
            f"{signature.speckle_proxy:.4f}"
        )

    print()
    print("SIMULATION → DIGITAL SIGNATURE V2 OK")


if __name__ == "__main__":
    main()