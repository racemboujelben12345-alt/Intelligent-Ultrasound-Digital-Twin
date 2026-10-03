from __future__ import annotations



import numpy as np

from src.simulation.scenarios import build_scenario
from src.signature.baseline import StatisticalBaseline
from src.validation.experiments import (
    evaluate_scenario,
    evaluate_normal_holdout,
)
from src.validation.reference_generator import (
    ReferenceGenerationConfig,
    generate_reference_set,
)
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)


def build_feature_matrix(images):
    """
    Convert normal reference images into Digital Signature matrix.
    """

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


def print_scenario_result(result):
    """
    Affiche les résultats complets d'un scénario expérimental.
    """

    print()
    print("=" * 120)
    print(f"SCENARIO : {result.scenario_name}")
    print("=" * 120)

    print(
        f"{'severity':>8} | "
        f"{'D² mean':>12} | "
        f"{'D² std':>10} | "
        f"{'D mean':>10} | "
        f"{'TP':>4} | "
        f"{'FN':>4} | "
        f"{'Detection':>10} | "
        f"{'Early':>7} | "
        f"{'Significant':>12} | "
        f"{'High':>7}"
    )

    print("-" * 120)

    for level in result.levels:

        metrics = level.metrics

        print(
            f"{level.severity:8.1f} | "
            f"{level.d2_mean:12.4f} | "
            f"{level.d2_std:10.4f} | "
            f"{level.distance_mean:10.4f} | "
            f"{metrics.true_positive:4d} | "
            f"{metrics.false_negative:4d} | "
            f"{metrics.detection_rate:10.2%} | "
            f"{metrics.early_drift_count:7d} | "
            f"{metrics.significant_drift_count:12d} | "
            f"{metrics.high_deviation_count:7d}"
        )

def main():

    print()
    print("=" * 120)
    print(
        "MULTI-ACQUISITION ANOMALY VALIDATION V2"
    )
    print("=" * 120)

    # ================================================================
    # 1. Generate controlled reference population
    # ================================================================

    config = ReferenceGenerationConfig(
        size=128,
        n_samples=50,
        seed=42,
    )

    images = generate_reference_set(config)

    reference_images = images[:40]
    holdout_images = images[40:]

    # ================================================================
    # 2. Build baseline only from reference acquisitions
    # ================================================================

    reference_signatures, X_reference = (
        build_feature_matrix(
            reference_images
        )
    )

    baseline = StatisticalBaseline(
        X_reference=X_reference,
        feature_names=(
            reference_signatures[0]
            .numeric_values()
            .keys()
        ),
    )

    print()
    print("BASELINE")
    print("-" * 120)

    print(
        f"Reference acquisitions : "
        f"{len(reference_images)}"
    )

    print(
        f"Holdout acquisitions   : "
        f"{len(holdout_images)}"
    )

    print(
        f"Features               : "
        f"{baseline.n_features}"
    )

    print(
        f"Shrinkage              : "
        f"{baseline.shrinkage:.4f}"
    )

    print(
        f"Early threshold        : "
        f"{baseline.thresholds['early']:.4f}"
    )

    print(
        f"Significant threshold  : "
        f"{baseline.thresholds['significant']:.4f}"
    )

    print(
        f"Critical threshold     : "
        f"{baseline.thresholds['critical']:.4f}"
    )

    # ================================================================
    # 3. Normal holdout
    # ================================================================

    normal_metrics = evaluate_normal_holdout(
        baseline=baseline,
        images=holdout_images,
        source="simulated",
    )

    print()
    print("NORMAL HOLDOUT VALIDATION")
    print("-" * 120)

    print(
        f"Distance mean ± std : "
        f"{normal_metrics.mean_distance:.4f} ± "
        f"{normal_metrics.std_distance:.4f}"
    )

    print(
        f"TN={normal_metrics.true_negative} | "
        f"FP={normal_metrics.false_positive} | "
        f"FN={normal_metrics.false_negative} | "
        f"TP={normal_metrics.true_positive}"
    )

    print(
        f"False-positive rate : "
        f"{normal_metrics.false_positive_rate:.2%}"
    )

    print(
        f"Specificity         : "
        f"{normal_metrics.specificity:.2%}"
    )

    # ================================================================
    # 4. Controlled experiments
    # ================================================================

    scenario_names = [
        "speckle_progression",
        "noise_progression",
        "blur_progression",
        "contrast_progression",
        "intensity_progression",
    ]

    experiment_results = []

    for scenario_name in scenario_names:

        scenario = build_scenario(
            scenario_name
        )

        result = evaluate_scenario(
            baseline=baseline,
            images=holdout_images,
            scenario=scenario,
            seed=42,
            source="simulated",
        )

        experiment_results.append(result)

        print_scenario_result(result)

    # ================================================================
    # 5. Global summary
    # ================================================================

    print()
    print("=" * 120)
    print("GLOBAL VALIDATION SUMMARY")
    print("=" * 120)

    print(
        f"Normal holdout false-positive rate : "
        f"{normal_metrics.false_positive_rate:.2%}"
    )

    print()

    for result in experiment_results:

        anomalous_levels = [
            level
            for level in result.levels
            if level.severity > 0.0
        ]

        detection_rates = [
            level.metrics.detection_rate
            for level in anomalous_levels
        ]

        mean_detection = (
            float(np.mean(detection_rates))
            if detection_rates
            else 0.0
        )

        print(
            f"{result.scenario_name:35s} | "
            f"mean detection = {mean_detection:.2%}"
        )

    print()
    print("=" * 120)
    print(
        "MULTI-ACQUISITION VALIDATION V2 OK"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()