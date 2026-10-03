from __future__ import annotations

import numpy as np

from src.anomaly.statistical import StatisticalAnomalyDetector
from src.digital_twin.factory import build_twin_state_from_detection
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.signature.baseline import StatisticalBaseline
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
    print("ANOMALY DETECTION → DIGITAL TWIN STATE V2")
    print("=" * 100)

    # ================================================================
    # 1. Reference data
    # ================================================================

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=30,
        seed=42,
    )

    images = generate_reference_set(config)

    signatures, X_reference = build_signature_matrix(
        images
    )

    # ================================================================
    # 2. Baseline + detector
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
    # 3. Analyse one acquisition
    # ================================================================

    acquisition = images[-1]

    signature = build_digital_signature_from_image(
        acquisition,
        source="simulated",
    )

    vector = signature.to_vector()

    detection = detector.detect(
        vector
    )

    # ================================================================
    # 4. Build Digital Twin State
    # ================================================================

    twin_state = build_twin_state_from_detection(
        acquisition_id="validation_001",
        source="simulated",
        vector=vector,
        detection=detection,
        baseline=baseline,
    )

    twin_state.validate()

    # ================================================================
    # 5. Display
    # ================================================================

    print()
    print("DETECTION RESULT")
    print("-" * 100)

    print(
        f"D²              : "
        f"{detection.d2:.6f}"
    )

    print(
        f"D               : "
        f"{detection.distance:.6f}"
    )

    print(
        f"State            : "
        f"{detection.state}"
    )

    print()
    print("DIGITAL TWIN STATE")
    print("-" * 100)

    print(
        f"Acquisition      : "
        f"{twin_state.acquisition_id}"
    )

    print(
        f"Source           : "
        f"{twin_state.source}"
    )

    print(
        f"State            : "
        f"{twin_state.state}"
    )

    print(
        f"Mahalanobis D    : "
        f"{twin_state.mahalanobis_distance:.6f}"
    )

    print(
        f"Mahalanobis D²   : "
        f"{twin_state.mahalanobis_squared:.6f}"
    )

    print(
        f"Quality score    : "
        f"{twin_state.quality_score:.6f}"
    )

    print()
    print("DIMENSION SCORES")
    print("-" * 100)

    for name, value in twin_state.dimension_scores.items():
        print(
            f"{name:20s} : {value:.6f}"
        )

    print()
    print("TOP FEATURE CONTRIBUTIONS")
    print("-" * 100)

    for feature, contribution in (
        twin_state.feature_contributions[:5]
    ):
        print(
            f"{feature:30s} : "
            f"{contribution:.6f}"
        )

    # ================================================================
    # 6. Consistency checks
    # ================================================================

    if twin_state.state != detection.state:
        raise AssertionError(
            "L'état du Digital Twin ne correspond pas "
            "au résultat du détecteur."
        )

    if not np.isclose(
        twin_state.mahalanobis_distance,
        detection.distance,
    ):
        raise AssertionError(
            "Incohérence sur la distance de Mahalanobis."
        )

    if not np.isclose(
        twin_state.mahalanobis_squared,
        detection.d2,
    ):
        raise AssertionError(
            "Incohérence sur D²."
        )

    if not twin_state.dimension_scores:
        raise AssertionError(
            "Les dimensions de santé sont absentes."
        )

    if not twin_state.feature_contributions:
        raise AssertionError(
            "Les contributions des features sont absentes."
        )

    print()
    print("=" * 100)
    print(
        "ANOMALY → DIGITAL TWIN STATE V2 OK"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()