from __future__ import annotations

import numpy as np

from src.prediction.trend import TrendAnalyzer


def main():

    print()
    print("=" * 100)
    print("TEMPORAL TREND ANALYZER V2")
    print("=" * 100)

    analyzer = TrendAnalyzer(
        stable_slope_threshold=0.05
    )

    # ================================================================
    # 1. Stable signal
    # ================================================================

    stable_signal = [
        7.8,
        8.1,
        7.9,
        8.0,
        8.2,
        7.9,
        8.1,
        8.0,
    ]

    stable_result = analyzer.analyze(
        stable_signal
    )

    print()
    print("STABLE SIGNAL")
    print("-" * 100)
    print(
        f"Slope          : "
        f"{stable_result.slope:.6f}"
    )
    print(
        f"R²             : "
        f"{stable_result.r_squared:.6f}"
    )
    print(
        f"Direction      : "
        f"{stable_result.direction}"
    )
    print(
        f"Strength       : "
        f"{stable_result.strength:.6f}"
    )
    print(
        f"First value    : "
        f"{stable_result.first_value:.6f}"
    )
    print(
        f"Last value     : "
        f"{stable_result.last_value:.6f}"
    )
    print(
        f"Predicted next : "
        f"{stable_result.predicted_next:.6f}"
    )

    if stable_result.direction != "STABLE":
        raise AssertionError(
            "Le signal stable devrait être classé STABLE."
        )

    # ================================================================
    # 2. Increasing signal
    # ================================================================

    increasing_signal = [
        5.0,
        5.8,
        6.5,
        7.2,
        8.0,
        8.7,
        9.5,
        10.3,
    ]

    increasing_result = analyzer.analyze(
        increasing_signal
    )

    print()
    print("INCREASING SIGNAL")
    print("-" * 100)
    print(
        f"Slope          : "
        f"{increasing_result.slope:.6f}"
    )
    print(
        f"R²             : "
        f"{increasing_result.r_squared:.6f}"
    )
    print(
        f"Direction      : "
        f"{increasing_result.direction}"
    )
    print(
        f"Strength       : "
        f"{increasing_result.strength:.6f}"
    )
    print(
        f"First value    : "
        f"{increasing_result.first_value:.6f}"
    )
    print(
        f"Last value     : "
        f"{increasing_result.last_value:.6f}"
    )
    print(
        f"Predicted next : "
        f"{increasing_result.predicted_next:.6f}"
    )

    if increasing_result.direction != "INCREASING":
        raise AssertionError(
            "Le signal croissant devrait être classé INCREASING."
        )

    if increasing_result.slope <= 0.0:
        raise AssertionError(
            "La pente du signal croissant doit être positive."
        )

    if not 0.0 <= increasing_result.r_squared <= 1.0:
        raise AssertionError(
            "R² invalide."
        )

    # ================================================================
    # 3. Numerical consistency
    # ================================================================

    expected_next = (
        increasing_result.slope
        * len(increasing_signal)
        + increasing_result.intercept
    )

    if not np.isclose(
        increasing_result.predicted_next,
        expected_next,
    ):
        raise AssertionError(
            "predicted_next est incohérent avec la régression."
        )

    print()
    print("=" * 100)
    print("TEMPORAL TREND ANALYZER V2 OK")
    print("=" * 100)


if __name__ == "__main__":
    main()