from __future__ import annotations

import numpy as np

from src.digital_twin.history import DigitalTwinHistory
from src.digital_twin.state import build_twin_state
from src.drift.calibration import calibrate_drift_signal
from src.drift.control_charts import DriftMonitorConfig
from src.drift.engine import DriftEngine
from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.signature.baseline import StatisticalBaseline


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
    print("DRIFT ENGINE INTEGRATION V2")
    print("=" * 110)

    # ================================================================
    # 1. Generate normal population
    # ================================================================

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    images = generate_reference_set(config)

    training_images = images[:30]
    calibration_images = images[30:40]

    signatures, X_training = build_signature_matrix(
        training_images
    )

    # ================================================================
    # 2. Build baseline
    # ================================================================

    baseline = StatisticalBaseline(
        X_reference=X_training,
        feature_names=(
            signatures[0]
            .numeric_values()
            .keys()
        ),
    )

    # ================================================================
    # 3. Build temporal calibration
    # ================================================================

    calibration = calibrate_drift_signal(
        baseline=baseline,
        images=calibration_images,
    )

    print()
    print("CALIBRATION")
    print("-" * 110)
    print(
        f"Samples   : {calibration.n_samples}"
    )
    print(
        f"Mean D²   : {calibration.mean_d2:.6f}"
    )
    print(
        f"Std D²    : {calibration.std_d2:.6f}"
    )
    print(
        f"Median D² : {calibration.median_d2:.6f}"
    )
    print(
        f"MAD D²    : {calibration.mad_d2:.6f}"
    )

    # ================================================================
    # 4. Create a Digital Twin History
    # ================================================================

    history = DigitalTwinHistory()

    # Normal temporal sequence
    for index, image in enumerate(
        images[40:43]
    ):

        signature = build_digital_signature_from_image(
            image,
            source="simulated",
        )

        vector = signature.to_vector()

        d2 = float(
            baseline.mahalanobis_squared(
                vector
            )[0]
        )

        distance = float(
            np.sqrt(
                max(d2, 0.0)
            )
        )

        state = str(
            baseline.classify_distance(
                d2
            )
        )

        quality = float(
            baseline.quality_score(
                vector
            )[0]
        )

        dimensions = {
            name: float(
                values[0]
            )
            for name, values in (
                baseline.health_dimensions(
                    vector
                ).items()
            )
        }

        contributions = baseline.feature_contributions(
            vector
        )

        twin_state = build_twin_state(
            acquisition_id=f"normal_{index:03d}",
            source="simulated",
            state=state,
            mahalanobis_distance=distance,
            mahalanobis_squared=d2,
            quality_score=quality,
            dimension_scores=dimensions,
            feature_contributions=contributions,
            timestamp=(
                f"2026-10-03T10:00:{index:02d}+00:00"
            ),
        )

        history.add(
            twin_state
        )

    print()
    print("HISTORY")
    print("-" * 110)
    print(
        f"States      : {len(history)}"
    )
    print(
        f"D² series   : "
        f"{history.mahalanobis_squared_series}"
    )
    print(
        f"State series: "
        f"{history.state_series}"
    )

    # ================================================================
    # 5. Analyse history with DriftEngine
    # ================================================================

    engine = DriftEngine(
        calibration=calibration,
        config=DriftMonitorConfig(
            ewma_lambda=0.20,
            ewma_limit=3.0,
            cusum_k=0.5,
            cusum_h=5.0,
            persistence=3,
        ),
    )

    analysis = engine.analyze_history(
        history
    )

    print()
    print("DRIFT ANALYSIS")
    print("-" * 110)

    for observation in analysis.observations:

        print(
            f"idx={observation.index:02d} | "
            f"D²={observation.signal:10.4f} | "
            f"z={observation.z_score:8.4f} | "
            f"EWMA={observation.ewma:8.4f} | "
            f"CUSUM+={observation.cusum_positive:8.4f} | "
            f"status={observation.status}"
        )

    # ================================================================
    # 6. Validate
    # ================================================================

    if len(
        analysis.observations
    ) != len(history):

        raise AssertionError(
            "Le nombre d'observations de drift "
            "doit correspondre à l'historique."
        )

    if analysis.persistent_drift_detected:
        raise AssertionError(
            "Une séquence normale ne devrait pas "
            "produire un drift persistant."
        )

    if analysis.latest.status not in {
        "STABLE",
        "DRIFT_SIGNAL",
        "PERSISTENT_DRIFT",
    }:
        raise AssertionError(
            "Statut de drift invalide."
        )

    print()
    print("=" * 110)
    print(
        "HISTORY → CALIBRATION → DRIFT ENGINE V2 OK"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
    