from __future__ import annotations

import numpy as np

from src.digital_twin.analyzer import DigitalTwinAnalyzer
from src.digital_twin.history import DigitalTwinHistory
from src.drift.calibration import calibrate_drift_signal
from src.drift.control_charts import DriftMonitorConfig
from src.drift.engine import DriftEngine
from src.prediction.engine import TemporalPredictionEngine
from src.prediction.trend import TrendAnalyzer
from src.signature.baseline import StatisticalBaseline
from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.simulation.degradation import (
    apply_degradation,
    DegradationType,
)


def build_feature_matrix(images):

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
    print("=" * 120)
    print("DIGITAL TWIN ANALYZER FULL INTEGRATION V2")
    print("=" * 120)

    # ================================================================
    # 1. Population
    # ================================================================

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    images = generate_reference_set(
        config
    )

    training_images = images[:30]
    calibration_images = images[30:40]
    test_images = images[40:]

    # ================================================================
    # 2. Baseline
    # ================================================================

    training_signatures, X_training = (
        build_feature_matrix(
            training_images
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

    # ================================================================
    # 3. Temporal calibration
    # ================================================================

    calibration = calibrate_drift_signal(
        baseline=baseline,
        images=calibration_images,
    )

    drift_engine = DriftEngine(
        calibration=calibration,
        config=DriftMonitorConfig(
            ewma_lambda=0.20,
            ewma_limit=3.0,
            cusum_k=0.5,
            cusum_h=5.0,
            persistence=3,
        ),
    )

    trend_engine = TemporalPredictionEngine(
        analyzer=TrendAnalyzer(
            stable_slope_threshold=0.05
        )
    )

    # ================================================================
    # 4. Central analyzer
    # ================================================================

    history = DigitalTwinHistory()

    analyzer = DigitalTwinAnalyzer(
        baseline=baseline,
        history=history,
        drift_engine=drift_engine,
        trend_engine=trend_engine,
    )

    # ================================================================
    # 5. Sequential normal acquisitions
    # ================================================================

    print()
    print("NORMAL ACQUISITIONS")
    print("-" * 120)

    normal_results = []

    for index, image in enumerate(
        test_images[:3]
    ):

        result = analyzer.analyze(
            image=image,
            acquisition_id=f"normal_{index:03d}",
            source="simulated",
            timestamp=(
                f"2026-10-03T11:00:{index:02d}+00:00"
            ),
        )

        normal_results.append(result)

        print(
            f"{result.state.acquisition_id:15s} | "
            f"D²={result.detection.d2:10.4f} | "
            f"State={result.state.state:18s} | "
            f"Quality={result.state.quality_score:8.3f}"
        )

        if result.detection.state != "NOMINAL":
            raise AssertionError(
                "Une acquisition normale a été détectée "
                "comme anormale."
            )

    # ================================================================
    # 6. Controlled anomaly
    # ================================================================

    degraded_image = apply_degradation(
        image=test_images[0],
        degradation_type=DegradationType.INTENSITY_SHIFT,
        severity=0.8,
        seed=42,
    ).image

    print()
    print("CONTROLLED ANOMALY")
    print("-" * 120)

    anomaly_result = analyzer.analyze(
        image=degraded_image,
        acquisition_id="anomaly_001",
        source="simulated",
        timestamp="2026-10-03T11:01:00+00:00",
        explain_top_k=5,
    )

    print(
        f"{anomaly_result.state.acquisition_id:15s} | "
        f"D²={anomaly_result.detection.d2:10.4f} | "
        f"State={anomaly_result.state.state:18s} | "
        f"Quality={anomaly_result.state.quality_score:8.3f}"
    )

    print()
    print("TOP EXPLANATION")
    print("-" * 120)

    for item in anomaly_result.explanation.top_features:
        print(
            f"{item.rank:2d}. "
            f"{item.feature:30s} | "
            f"{item.contribution:.4f}"
        )

    if anomaly_result.detection.state == "NOMINAL":
        raise AssertionError(
            "L'anomalie contrôlée n'a pas été détectée."
        )

    # ================================================================
    # 7. History
    # ================================================================

    print()
    print("HISTORY")
    print("-" * 120)
    print(
        f"States count : {len(history)}"
    )
    print(
        f"Latest state : {history.latest.state}"
    )
    print(
        f"D² series    : "
        f"{history.mahalanobis_squared_series}"
    )

    if len(history) != 4:
        raise AssertionError(
            "L'historique devrait contenir 4 états."
        )

    # ================================================================
    # 8. Temporal analysis
    # ================================================================

    print()
    print("DRIFT ANALYSIS")
    print("-" * 120)

    if anomaly_result.drift is None:
        raise AssertionError(
            "L'analyse de drift devrait être disponible."
        )

    for observation in anomaly_result.drift.observations:
        print(
            f"idx={observation.index:02d} | "
            f"D²={observation.signal:10.4f} | "
            f"EWMA={observation.ewma:9.4f} | "
            f"CUSUM+={observation.cusum_positive:9.4f} | "
            f"{observation.status}"
        )

    print()
    print("TREND ANALYSIS")
    print("-" * 120)

    if anomaly_result.trend is None:
        raise AssertionError(
            "L'analyse de tendance devrait être disponible."
        )

    print(
        f"D² direction      : "
        f"{anomaly_result.trend.d2_direction}"
    )

    print(
        f"D² slope          : "
        f"{anomaly_result.trend.mahalanobis_squared.slope:.6f}"
    )

    print(
        f"Quality direction : "
        f"{anomaly_result.trend.quality_direction}"
    )

    print(
        f"Quality slope     : "
        f"{anomaly_result.trend.quality_score.slope:.6f}"
    )

    # ================================================================
    # 9. Global consistency
    # ================================================================

    if anomaly_result.state.state != (
        anomaly_result.detection.state
    ):
        raise AssertionError(
            "State incohérent avec DetectionResult."
        )

    if not np.isclose(
        anomaly_result.state.mahalanobis_squared,
        anomaly_result.detection.d2,
    ):
        raise AssertionError(
            "D² incohérent entre detection et state."
        )

    if not anomaly_result.explanation.top_features:
        raise AssertionError(
            "Aucune explication disponible."
        )

    if len(anomaly_result.drift.observations) != len(
        history
    ):
        raise AssertionError(
            "Drift et History doivent avoir la même longueur."
        )

    print()
    print("=" * 120)
    print(
        "DIGITAL TWIN ANALYZER FULL INTEGRATION V2 OK"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()