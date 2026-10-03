from __future__ import annotations

import numpy as np

from src.anomaly.statistical import (
    StatisticalAnomalyDetector,
)
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.signature.baseline import (
    StatisticalBaseline,
)
from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)


def build_signature_matrix(images):
    signatures = [
        build_digital_signature_from_image(
            image,
            source="simulated",
        )
        for image in images
    ]

    matrix = np.vstack(
        [
            signature.to_vector()
            for signature in signatures
        ]
    )

    return signatures, matrix


def main():

    print()
    print("=" * 100)
    print("STATISTICAL ANOMALY DETECTOR V2")
    print("=" * 100)

    # ================================================================
    # 1. Reference population
    # ================================================================

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=40,
        seed=42,
    )

    images = generate_reference_set(
        config
    )

    signatures, X_reference = (
        build_signature_matrix(images)
    )

    # ================================================================
    # 2. Statistical baseline
    # ================================================================

    baseline = StatisticalBaseline(
        X_reference=X_reference,
        feature_names=(
            signatures[0]
            .numeric_values()
            .keys()
        ),
    )

    detector = StatisticalAnomalyDetector(
        baseline=baseline
    )

    # ================================================================
    # 3. Test normal observation
    # ================================================================

    normal_image = images[-1]

    normal_signature = (
        build_digital_signature_from_image(
            normal_image,
            source="simulated",
        )
    )

    normal_result = detector.detect(
        normal_signature.to_vector()
    )

    print()
    print("NORMAL OBSERVATION")
    print("-" * 100)
    print(
        f"D²         : {normal_result.d2:.6f}"
    )
    print(
        f"D          : {normal_result.distance:.6f}"
    )
    print(
        f"State      : {normal_result.state}"
    )
    print(
        f"Is anomaly : "
        f"{detector.is_anomaly(normal_signature.to_vector())}"
    )

    if normal_result.state != "NOMINAL":
        raise AssertionError(
            "Une observation normale a été "
            "classée comme anormale."
        )

    # ================================================================
    # 4. Feature attribution
    # ================================================================

    contributions = (
        normal_result.feature_contributions
    )

    contribution_sum = sum(
        contributions.values()
    )

    print()
    print("FEATURE CONTRIBUTIONS")
    print("-" * 100)

    for feature, contribution in sorted(
        contributions.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]:

        print(
            f"{feature:30s} : "
            f"{contribution:.6f}"
        )

    print(
        f"Contribution sum : "
        f"{contribution_sum:.6f}"
    )

    if not np.isclose(
        contribution_sum,
        1.0,
        atol=1e-6,
    ):
        raise AssertionError(
            "Les contributions des features "
            "ne somment pas à 1."
        )

    # ================================================================
    # 5. Synthetic anomaly
    # ================================================================

    anomalous_image = (
        normal_image + 0.20
    )

    anomalous_image = np.clip(
        anomalous_image,
        0.0,
        1.0,
    ).astype(np.float32)

    anomalous_signature = (
        build_digital_signature_from_image(
            anomalous_image,
            source="simulated",
        )
    )

    anomalous_result = detector.detect(
        anomalous_signature.to_vector()
    )

    print()
    print("SYNTHETIC ANOMALY")
    print("-" * 100)
    print(
        f"D²         : {anomalous_result.d2:.6f}"
    )
    print(
        f"D          : "
        f"{anomalous_result.distance:.6f}"
    )
    print(
        f"State      : "
        f"{anomalous_result.state}"
    )
    print(
        f"Is anomaly : "
        f"{detector.is_anomaly(anomalous_signature.to_vector())}"
    )

    if not detector.is_anomaly(
        anomalous_signature.to_vector()
    ):
        raise AssertionError(
            "L'anomalie synthétique n'a pas été détectée."
        )

    # ================================================================
    # 6. Final validation
    # ================================================================

    print()
    print("=" * 100)
    print(
        "STATISTICAL ANOMALY DETECTOR V2 OK"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
    