import numpy as np

from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)

from src.simulation.degradation import (
    DegradationType,
    apply_degradation,
)

from src.simulation.scenarios import (
    build_scenario,
)

from src.simulation.runner import (
    run_scenario,
)


def main():
    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    image = generate_reference_set(config)[-1]

    print()
    print("=" * 80)
    print("CONTRAST ENGINE ISOLATION TEST")
    print("=" * 80)

    print()
    print("A. DIRECT apply_degradation()")
    print("-" * 80)

    for severity in (0.0, 0.5, 1.0):
        result = apply_degradation(
            image=image,
            degradation_type=DegradationType.CONTRAST_REDUCTION,
            severity=severity,
            seed=42,
        )

        print(
            f"severity={severity:.1f} | "
            f"mean={result.image.mean():.6f} | "
            f"std={result.image.std():.6f} | "
            f"min={result.image.min():.6f} | "
            f"max={result.image.max():.6f}"
        )

    print()
    print("B. SCENARIO + RUNNER")
    print("-" * 80)

    scenario = build_scenario("contrast_progression")

    print(f"Scenario name : {scenario.name}")
    print(f"Degradation   : {scenario.degradation_type}")
    print(f"Severities    : {scenario.severities}")

    steps = run_scenario(
        image=image,
        scenario=scenario,
        seed=42,
    )

    for step in steps:
        print(
            f"severity={step.severity:.1f} | "
            f"mean={step.result.image.mean():.6f} | "
            f"std={step.result.image.std():.6f} | "
            f"min={step.result.image.min():.6f} | "
            f"max={step.result.image.max():.6f}"
        )

    print()
    print("CONTRAST ENGINE ISOLATION TEST OK")


if __name__ == "__main__":
    main()
    