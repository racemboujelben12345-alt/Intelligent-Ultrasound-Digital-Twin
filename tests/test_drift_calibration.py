from __future__ import annotations

import numpy as np

from src.drift.calibration import calibrate_drift_signal
from src.drift.control_charts import (
    DriftMonitor,
    DriftMonitorConfig,
)
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.signature.baseline import StatisticalBaseline
from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)
from src.validation.splits import split_dataset


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
        "TRAINING → CALIBRATION → TEST DRIFT PIPELINE V2"
    )
    print("=" * 110)

    # ================================================================
    # 1. Generate population
    # ================================================================

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    images = generate_reference_set(
        config
    )

    split = split_dataset(
        images,
        training_size=30,
        calibration_size=10,
        test_size=10,
    )

    print()
    print("DATASET SPLIT")
    print("-" * 110)
    print(f"Training    : {len(split.training)}")
    print(f"Calibration : {len(split.calibration)}")
    print(f"Test        : {len(split.test)}")

    # ================================================================
    # 2. Baseline = TRAINING ONLY
    # ================================================================

    training_signatures, X_training = (
        build_signature_matrix(
            split.training
        )
    )

    baseline = StatisticalBaseline(
        X_reference=X_training,
        feature_names=(
            training_signatures[0]
            .numeric_values()
            .keys()
        ),
    )

    print()
    print("BASELINE")
    print("-" * 110)
    print(
        f"Training samples : {baseline.n_samples}"
    )
    print(
        f"Features         : {baseline.n_features}"
    )
    print(
        f"Shrinkage        : {baseline.shrinkage:.4f}"
    )

    # ================================================================
    # 3. Temporal calibration = CALIBRATION ONLY
    # ================================================================

    calibration = calibrate_drift_signal(
        baseline=baseline,
        images=split.calibration,
    )

    print()
    print("TEMPORAL CALIBRATION")
    print("-" * 110)
    print(
        f"Samples     : {calibration.n_samples}"
    )
    print(
        f"Mean D²     : {calibration.mean_d2:.6f}"
    )
    print(
        f"Std D²      : {calibration.std_d2:.6f}"
    )
    print(
        f"Median D²   : {calibration.median_d2:.6f}"
    )
    print(
        f"MAD D²      : {calibration.mad_d2:.6f}"
    )
    print(
        f"P95 D²      : "
        f"{calibration.percentiles['p95']:.6f}"
    )
    print(
        f"P99 D²      : "
        f"{calibration.percentiles['p99']:.6f}"
    )

    # ================================================================
    # 4. Test set → normal temporal observations
    # ================================================================

    test_d2 = []

    for image in split.test:

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

        test_d2.append(d2)

    test_d2 = np.asarray(
        test_d2,
        dtype=np.float64,
    )

    print()
    print("INDEPENDENT TEST SIGNAL")
    print("-" * 110)

    print(
        f"Test D² mean ± std : "
        f"{np.mean(test_d2):.6f} ± "
        f"{np.std(test_d2):.6f}"
    )

    print(
        "D² values : "
        + " → ".join(
            f"{value:.4f}"
            for value in test_d2
        )
    )

    # ================================================================
    # 5. Drift monitor calibrated on CALIBRATION only
    # ================================================================

    monitor = DriftMonitor(
        reference_mean=calibration.mean_d2,
        reference_std=calibration.std_d2,
        config=DriftMonitorConfig(
            ewma_lambda=0.20,
            ewma_limit=3.0,
            cusum_k=0.5,
            cusum_h=5.0,
            persistence=3,
        ),
    )

    normal_observations = monitor.update_many(
        list(test_d2)
    )

    print()
    print("TEST → DRIFT MONITOR")
    print("-" * 110)

    print(
        f"{'idx':>4} | "
        f"{'D²':>12} | "
        f"{'z':>10} | "
        f"{'EWMA':>10} | "
        f"{'CUSUM+':>10} | "
        f"{'status'}"
    )

    print("-" * 110)

    for observation in normal_observations:

        print(
            f"{observation.index:4d} | "
            f"{observation.signal:12.4f} | "
            f"{observation.z_score:10.4f} | "
            f"{observation.ewma:10.4f} | "
            f"{observation.cusum_positive:10.4f} | "
            f"{observation.status}"
        )

    # ================================================================
    # 6. Independent persistent drift experiment
    # ================================================================

    drift_monitor = DriftMonitor(
        reference_mean=calibration.mean_d2,
        reference_std=calibration.std_d2,
        config=DriftMonitorConfig(
            ewma_lambda=0.20,
            ewma_limit=3.0,
            cusum_k=0.5,
            cusum_h=5.0,
            persistence=3,
        ),
    )

    # Construct a clearly shifted signal using the
    # calibrated normal scale. This is a temporal
    # validation signal, not SCAN A physical data.

    shifted_signal = (
        calibration.mean_d2
        + 8.0 * calibration.std_d2
    )

    drift_sequence = [
        calibration.mean_d2,
        calibration.mean_d2,
        calibration.mean_d2,
        shifted_signal,
        shifted_signal,
        shifted_signal,
        shifted_signal,
    ]

    drift_observations = drift_monitor.update_many(
        drift_sequence
    )

    print()
    print("INDEPENDENT PERSISTENT DRIFT TEST")
    print("-" * 110)

    for observation in drift_observations:

        print(
            f"{observation.index:4d} | "
            f"D²={observation.signal:12.4f} | "
            f"z={observation.z_score:10.4f} | "
            f"EWMA={observation.ewma:10.4f} | "
            f"CUSUM+={observation.cusum_positive:10.4f} | "
            f"{observation.status}"
        )

    # ================================================================
    # 7. Validation
    # ================================================================

    if calibration.n_samples != 10:
        raise AssertionError(
            "Le nombre d'observations de calibration est incorrect."
        )

    if not np.isfinite(
        calibration.mean_d2
    ):
        raise AssertionError(
            "Mean D² de calibration invalide."
        )

    if not np.isfinite(
        calibration.std_d2
    ):
        raise AssertionError(
            "Std D² de calibration invalide."
        )

    if calibration.std_d2 <= 0.0:
        raise AssertionError(
            "La dispersion de calibration doit être > 0."
        )

    persistent = [
        observation
        for observation in drift_observations
        if observation.status == "PERSISTENT_DRIFT"
    ]

    if not persistent:
        raise AssertionError(
            "Le drift persistant n'a pas été détecté."
        )

    print()
    print("=" * 110)
    print(
        "TRAINING → CALIBRATION → TEST DRIFT PIPELINE V2 OK"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()