import numpy as np

from src.simulation.scenarios import build_scenario
from src.simulation.runner import run_scenario
from src.image_analysis.basic_metrics import compute_basic_metrics


def main():
    image = np.ones(
        (128, 128),
        dtype=np.float32,
    ) * 0.5

    scenario = build_scenario(
        "speckle_progression"
    )

    steps = run_scenario(
        image,
        scenario,
    )

    print("severity | std | speckle | entropy")
    print("-" * 42)

    for step in steps:
        metrics = compute_basic_metrics(
            step.result.image
        )

        print(
            f"{step.severity:.1f}      | "
            f"{metrics['std_intensity']:.4f} | "
            f"{metrics['speckle_proxy']:.4f} | "
            f"{metrics['entropy']:.4f}"
        )

    print(
        "SIMULATION + IMAGE ANALYSIS V2 OK"
    )


if __name__ == "__main__":
    main()