import numpy as np

from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)

from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)

from src.signature.baseline import (
    StatisticalBaseline,
)

from src.simulation.scenarios import (
    build_scenario,
)

from src.simulation.runner import (
    run_scenario,
)


def build_signature_matrix(images):
    signatures = [
        build_digital_signature_from_image(
            image,
            source="simulated",
        )
        for image in images
    ]

    X = np.vstack([
        signature.to_vector()
        for signature in signatures
    ])

    return signatures, X


def main():

    print()
    print("=" * 80)
    print("DIGITAL SIGNATURE → STATISTICAL BASELINE → ANOMALY DETECTION V2")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Generate realistic normal reference data
    # ------------------------------------------------------------------

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    images = generate_reference_set(config)

    print()
    print("1. NORMAL REFERENCE DATA")
    print("-" * 80)
    print(f"Images      : {len(images)}")
    print(f"Image shape : {images[0].shape}")

    # ------------------------------------------------------------------
    # 2. Convert images → digital signatures
    # ------------------------------------------------------------------

    signatures, X_reference = build_signature_matrix(images)

    print()
    print("2. DIGITAL SIGNATURE MATRIX")
    print("-" * 80)
    print(f"Observations : {X_reference.shape[0]}")
    print(f"Features     : {X_reference.shape[1]}")

    # ------------------------------------------------------------------
    # 3. Build statistical normal operating baseline
    # ------------------------------------------------------------------

    baseline = StatisticalBaseline(
        X_reference=X_reference,
        feature_names=signatures[0].numeric_values().keys(),
    )

    print()
    print("3. STATISTICAL BASELINE")
    print("-" * 80)
    print(f"Samples      : {baseline.n_samples}")
    print(f"Features     : {baseline.n_features}")
    print(f"Shrinkage    : {baseline.shrinkage:.4f}")
    print(f"Early        : {baseline.thresholds['early']:.3f}")
    print(f"Significant  : {baseline.thresholds['significant']:.3f}")
    print(f"Critical     : {baseline.thresholds['critical']:.3f}")

    # ------------------------------------------------------------------
    # 4. Normal holdout
    # ------------------------------------------------------------------

    holdout_image = images[-1]

    holdout_signature = build_digital_signature_from_image(
        holdout_image,
        source="simulated",
    )

    holdout_vector = holdout_signature.to_vector()

    holdout_d2 = float(
        baseline.mahalanobis_squared(
            holdout_vector
        )[0]
    )

    holdout_distance = float(
        np.sqrt(max(holdout_d2, 0.0))
    )

    holdout_state = str(
        baseline.classify_distance(
            holdout_d2
        )
    )

    print()
    print("4. NORMAL HOLDOUT")
    print("-" * 80)
    print(f"D²         : {holdout_d2:.4f}")
    print(f"Distance   : {holdout_distance:.4f}")
    print(f"State      : {holdout_state}")

    # ------------------------------------------------------------------
    # 5. Controlled degradation experiments
    # ------------------------------------------------------------------

    scenarios = [
        "speckle_progression",
        "noise_progression",
        "blur_progression",
        "contrast_progression",
        "intensity_progression",
    ]

    for scenario_name in scenarios:

        scenario = build_scenario(scenario_name)

        steps = run_scenario(
            holdout_image,
            scenario,
        )

        print()
        print(f"5. DEGRADATION EXPERIMENT : {scenario_name}")
        print("-" * 80)

        print(
            f"{'severity':>8} | "
            f"{'D²':>12} | "
            f"{'distance':>10} | "
            f"{'state'}"
        )

        print("-" * 80)

        for step in steps:

            signature = build_digital_signature_from_image(
                step.result.image,
                source="simulated",
            )

            vector = signature.to_vector()

            d2 = float(
                baseline.mahalanobis_squared(
                    vector
                )[0]
            )

            distance = float(
                np.sqrt(max(d2, 0.0))
            )

            state = str(
                baseline.classify_distance(
                    d2
                )
            )

            print(
                f"{step.severity:8.1f} | "
                f"{d2:12.4f} | "
                f"{distance:10.4f} | "
                f"{state}"
            )

    print()
    print("=" * 80)
    print("DIGITAL SIGNATURE → BASELINE → ANOMALY V2 OK")
    print("=" * 80)


if __name__ == "__main__":
    main()