from __future__ import annotations

import numpy as np

from src.digital_twin.history import DigitalTwinHistory
from src.digital_twin.state import build_twin_state
from src.prediction.engine import TemporalPredictionEngine
from src.prediction.trend import TrendAnalyzer


def build_history(values_d2, values_quality):
    """
    Construit un historique minimal cohérent pour tester
    le moteur temporel.
    """

    history = DigitalTwinHistory()

    for index, (d2, quality) in enumerate(
        zip(values_d2, values_quality)
    ):

        d2 = float(d2)

        distance = float(
            np.sqrt(
                max(d2, 0.0)
            )
        )

        state = (
            "NOMINAL"
            if d2 <= 22.3620
            else "EARLY_DRIFT"
        )

        twin_state = build_twin_state(
            acquisition_id=f"prediction_{index:03d}",
            source="simulated",
            state=state,
            mahalanobis_distance=distance,
            mahalanobis_squared=d2,
            quality_score=float(quality),
            dimension_scores={
                "contrast": 1.0,
                "noise_texture": 1.0,
                "sharpness": 1.0,
                "structure": 1.0,
                "uniformity": 1.0,
            },
            feature_contributions=[
                ("test_feature", 1.0)
            ],
            timestamp=(
                f"2026-10-03T10:00:{index:02d}+00:00"
            ),
        )

        history.add(twin_state)

    return history


def main():

    print()
    print("=" * 110)
    print("TEMPORAL PREDICTION ENGINE V2")
    print("=" * 110)

    analyzer = TrendAnalyzer(
        stable_slope_threshold=0.05
    )

    engine = TemporalPredictionEngine(
        analyzer=analyzer
    )

    # ================================================================
    # 1. Stable history
    # ================================================================

    stable_d2 = [
        8.0,
        8.1,
        7.9,
        8.2,
        8.0,
        8.1,
        7.95,
        8.05,
    ]

    stable_quality = [
        80.0,
        79.5,
        80.2,
        79.8,
        80.1,
        79.7,
        80.0,
        79.9,
    ]

    stable_history = build_history(
        stable_d2,
        stable_quality,
    )

    stable_result = engine.analyze_history(
        stable_history
    )

    print()
    print("STABLE HISTORY")
    print("-" * 110)

    print(
        f"D² direction      : "
        f"{stable_result.mahalanobis_squared.direction}"
    )

    print(
        f"D² slope          : "
        f"{stable_result.mahalanobis_squared.slope:.6f}"
    )

    print(
        f"D² R²             : "
        f"{stable_result.mahalanobis_squared.r_squared:.6f}"
    )

    print(
        f"D² next estimate  : "
        f"{stable_result.mahalanobis_squared.predicted_next:.6f}"
    )

    print(
        f"Quality direction : "
        f"{stable_result.quality_score.direction}"
    )

    print(
        f"Quality slope     : "
        f"{stable_result.quality_score.slope:.6f}"
    )

    if (
        stable_result.d2_direction
        != "STABLE"
    ):
        raise AssertionError(
            "La série D² stable devrait être STABLE."
        )

    # ================================================================
    # 2. Increasing D² / decreasing quality
    # ================================================================

    degrading_d2 = [
        5.0,
        6.0,
        7.2,
        8.5,
        10.0,
        11.8,
        13.7,
        15.5,
    ]

    degrading_quality = [
        92.0,
        89.0,
        86.0,
        82.0,
        78.0,
        74.0,
        70.0,
        66.0,
    ]

    degrading_history = build_history(
        degrading_d2,
        degrading_quality,
    )

    degrading_result = engine.analyze_history(
        degrading_history
    )

    print()
    print("DEGRADING HISTORY")
    print("-" * 110)

    print(
        f"D² direction      : "
        f"{degrading_result.mahalanobis_squared.direction}"
    )

    print(
        f"D² slope          : "
        f"{degrading_result.mahalanobis_squared.slope:.6f}"
    )

    print(
        f"D² R²             : "
        f"{degrading_result.mahalanobis_squared.r_squared:.6f}"
    )

    print(
        f"D² next estimate  : "
        f"{degrading_result.mahalanobis_squared.predicted_next:.6f}"
    )

    print(
        f"Quality direction : "
        f"{degrading_result.quality_score.direction}"
    )

    print(
        f"Quality slope     : "
        f"{degrading_result.quality_score.slope:.6f}"
    )

    print(
        f"Quality next est. : "
        f"{degrading_result.quality_score.predicted_next:.6f}"
    )

    if (
        degrading_result.d2_direction
        != "INCREASING"
    ):
        raise AssertionError(
            "La série D² dégradante devrait être INCREASING."
        )

    if (
        degrading_result.quality_direction
        != "DECREASING"
    ):
        raise AssertionError(
            "La qualité devrait suivre une tendance DECREASING."
        )

    if (
        degrading_result.mahalanobis_squared.slope
        <= 0.0
    ):
        raise AssertionError(
            "La pente D² doit être positive."
        )

    if (
        degrading_result.quality_score.slope
        >= 0.0
    ):
        raise AssertionError(
            "La pente qualité doit être négative."
        )

    if not (
        0.0
        <= degrading_result.mahalanobis_squared.r_squared
        <= 1.0
    ):
        raise AssertionError(
            "R² D² invalide."
        )

    if not (
        0.0
        <= degrading_result.quality_score.r_squared
        <= 1.0
    ):
        raise AssertionError(
            "R² qualité invalide."
        )

    # ================================================================
    # 3. History integrity
    # ================================================================

    if len(stable_result.mahalanobis_squared.to_dict()) == 0:
        raise AssertionError(
            "Résultat D² vide."
        )

    if len(degrading_history) != len(
        degrading_d2
    ):
        raise AssertionError(
            "Historique dégradant incohérent."
        )

    print()
    print("=" * 110)
    print(
        "HISTORY → TREND → TEMPORAL PREDICTION V2 OK"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()