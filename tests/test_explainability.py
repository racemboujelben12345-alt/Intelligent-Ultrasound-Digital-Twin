from __future__ import annotations

import numpy as np

from src.anomaly.statistical import (
    StatisticalAnomalyDetector,
)
from src.explainability.attribution import (
    explain_detection,
    explanation_text,
)
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.signature.baseline import (
    StatisticalBaseline,
)
from src.simulation.degradation import (
    apply_degradation,
    DegradationType,
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
    print("=" * 110)
    print("EXPLAINABILITY PIPELINE V2")
    print("=" * 110)

    # ================================================================
    # 1. Reference data
    # ================================================================

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=40,
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
    # 3. Controlled anomaly
    # ================================================================

    normal_image = images[-1]

    degraded_image = apply_degradation(
        image=normal_image,
        degradation_type=DegradationType.INTENSITY_SHIFT,
        severity=0.8,
        seed=42,
    ).image

    signature = build_digital_signature_from_image(
        degraded_image,
        source="simulated",
    )

    detection = detector.detect(
        signature.to_vector()
    )

    print()
    print("DETECTION")
    print("-" * 110)
    print(
        f"D²         : {detection.d2:.6f}"
    )
    print(
        f"D          : {detection.distance:.6f}"
    )
    print(
        f"State      : {detection.state}"
    )

    if detection.state == "NOMINAL":
        raise AssertionError(
            "La dégradation contrôlée devrait produire "
            "un état non-NOMINAL."
        )

    # ================================================================
    # 4. Explain detection
    # ================================================================

    explanation = explain_detection(
        detection,
        top_k=5,
    )

    explanation.validate()

    print()
    print("EXPLANATION")
    print("-" * 110)
    print(
        explanation_text(
            explanation
        )
    )

    # ================================================================
    # 5. Validation
    # ================================================================

    if not explanation.top_features:
        raise AssertionError(
            "Aucune feature explicative retournée."
        )

    if explanation.dominant_feature is None:
        raise AssertionError(
            "La feature dominante est absente."
        )

    if explanation.dominant_contribution <= 0.0:
        raise AssertionError(
            "La contribution dominante doit être > 0."
        )

    contributions = [
        item.contribution
        for item in explanation.top_features
    ]

    if contributions != sorted(
        contributions,
        reverse=True,
    ):
        raise AssertionError(
            "Les contributions ne sont pas triées."
        )

    if any(
        contribution < 0.0
        for contribution in contributions
    ):
        raise AssertionError(
            "Une contribution est négative."
        )

    print()
    print("=" * 110)
    print(
        "EXPLAINABILITY PIPELINE V2 OK"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()