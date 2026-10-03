from __future__ import annotations

import numpy as np

from src.anomaly.statistical import (
    StatisticalAnomalyDetector,
)
from src.digital_twin.factory import (
    build_twin_state_from_detection,
)
from src.digital_twin.history import (
    DigitalTwinHistory,
)
from src.drift.control_charts import (
    DriftMonitor,
    DriftMonitorConfig,
)
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.simulation.degradation import (
    apply_degradation,
    DegradationType,
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
    print("=" * 110)
    print(
        "DIGITAL TWIN HISTORY → TEMPORAL DRIFT INTEGRATION V2"
    )
    print("=" * 110)

    # ================================================================
    # 1. Reference population
    # ================================================================

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    images = generate_reference_set(config)

    reference_images = images[:40]
    holdout_images = images[40:]

    signatures, X_reference = build_signature_matrix(
        reference_images
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
    # 3. History
    # ================================================================

    history = DigitalTwinHistory()

    # ================================================================
    # 4. Add normal acquisitions
    # ================================================================

    print()
    print("NORMAL ACQUISITIONS")
    print("-" * 110)

    for index, image in enumerate(
        holdout_images[:3]
    ):

        signature = build_digital_signature_from_image(
            image,
            source="simulated",
        )

        vector = signature.to_vector()

        detection = detector.detect(
            vector
        )

        state = build_twin_state_from_detection(
            acquisition_id=f"normal_{index:03d}",
            source="simulated",
            vector=vector,
            detection=detection,
            baseline=baseline,
            timestamp=(
                f"2026-10-03T10:00:0{index}+00:00"
            ),
        )

        history.add(state)

        print(
            f"{state.acquisition_id:15s} | "
            f"D²={state.mahalanobis_squared:10.4f} | "
            f"D={state.mahalanobis_distance:8.4f} | "
            f"state={state.state}"
        )

    # ================================================================
    # 5. Persistent controlled degradation
    # ================================================================

    degraded_image = apply_degradation(
        image=holdout_images[0],
        degradation_type=DegradationType.INTENSITY_SHIFT,
        severity=0.6,
        seed=42,
    ).image

    print()
    print("PERSISTENT CONTROLLED DEGRADATION")
    print("-" * 110)

    for index in range(5):

        signature = build_digital_signature_from_image(
            degraded_image,
            source="simulated",
        )

        vector = signature.to_vector()

        detection = detector.detect(
            vector
        )

        state = build_twin_state_from_detection(
            acquisition_id=f"degraded_{index:03d}",
            source="simulated",
            vector=vector,
            detection=detection,
            baseline=baseline,
            timestamp=(
                f"2026-10-03T10:01:{index:02d}+00:00"
            ),
        )

        history.add(state)

        print(
            f"{state.acquisition_id:15s} | "
            f"D²={state.mahalanobis_squared:10.4f} | "
            f"D={state.mahalanobis_distance:8.4f} | "
            f"state={state.state}"
        )

    history.validate()

    # ================================================================
    # 6. Extract temporal D² signal
    # ================================================================

    d2_series = history.mahalanobis_squared_series

    print()
    print("TEMPORAL D² SERIES")
    print("-" * 110)

    print(
        " → ".join(
            f"{value:.4f}"
            for value in d2_series
        )
    )

    if len(d2_series) != len(history):
        raise AssertionError(
            "La série D² ne correspond pas à la taille de l'historique."
        )

    # ================================================================
    # 7. Drift monitor
    # ================================================================

    drift_config = DriftMonitorConfig(
        ewma_lambda=0.20,
        ewma_limit=3.0,
        cusum_k=0.5,
        cusum_h=5.0,
        persistence=3,
    )

    drift_monitor = DriftMonitor(
        reference_mean=baseline.reference_distance_mean,
        reference_std=baseline.reference_distance_std,
        config=drift_config,
    )

    observations = drift_monitor.update_many(
        list(d2_series)
    )

    # ================================================================
    # 8. Display drift evolution
    # ================================================================

    print()
    print("TEMPORAL DRIFT MONITOR")
    print("-" * 110)

    print(
        f"{'idx':>4} | "
        f"{'D²':>12} | "
        f"{'z':>9} | "
        f"{'EWMA':>10} | "
        f"{'CUSUM+':>10} | "
        f"{'alarms':>7} | "
        f"{'status'}"
    )

    print("-" * 110)

    for observation in observations:

        print(
            f"{observation.index:4d} | "
            f"{observation.signal:12.4f} | "
            f"{observation.z_score:9.4f} | "
            f"{observation.ewma:10.4f} | "
            f"{observation.cusum_positive:10.4f} | "
            f"{observation.consecutive_alarms:7d} | "
            f"{observation.status}"
        )

    # ================================================================
    # 9. Validation
    # ================================================================

    persistent = [
        observation
        for observation in observations
        if observation.status == "PERSISTENT_DRIFT"
    ]

    if not persistent:
        raise AssertionError(
            "Le drift persistant n'a pas été détecté."
        )

    if history.latest is None:
        raise AssertionError(
            "L'historique devrait contenir des états."
        )

    if history.latest.state != "HIGH_DEVIATION":
        raise AssertionError(
            "La dernière acquisition contrôlée devrait "
            "être HIGH_DEVIATION."
        )

    if not np.isfinite(
        baseline.reference_distance_mean
    ):
        raise AssertionError(
            "La moyenne de référence D² est invalide."
        )

    if not np.isfinite(
        baseline.reference_distance_std
    ):
        raise AssertionError(
            "L'écart-type de référence D² est invalide."
        )

    print()
    print("=" * 110)
    print(
        "HISTORY → D² → EWMA/CUSUM → PERSISTENT DRIFT V2 OK"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()