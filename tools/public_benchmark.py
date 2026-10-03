"""
Benchmark du pipeline SCAN A Digital Twin V2 sur des données publiques.

IMPORTANT
---------
Ce benchmark est destiné au développement et à la validation méthodologique.

Les données publiques :
- ne sont PAS des données SCAN A ;
- ne constituent PAS la baseline réelle du SCAN A ;
- servent uniquement à tester le comportement du pipeline
  de signature, dégradation et détection.

Usage
-----
python tools/public_benchmark.py chemin/vers/dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ------------------------------------------------------------------
# Initialisation du chemin racine du projet
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------
# Imports V2 uniquement
# ------------------------------------------------------------------

import config as C

from src.acquisition import load_public
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.signature.baseline import StatisticalBaseline
from src.simulation.runner import run_scenario
from src.simulation.scenarios import (
    ScenarioType,
)


# ------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------

def build_signature_matrix(
    images: list[np.ndarray],
    source: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    Transforme une collection d'images en matrice de Digital Signatures.
    """

    signatures = tuple(
        build_digital_signature_from_image(
            image,
            source=source,
        )
        for image in images
    )

    if not signatures:
        raise ValueError(
            "Aucune signature disponible."
        )

    first_values = signatures[0].numeric_values()

    feature_names = tuple(
        first_values.keys()
    )

    X = np.asarray(
        [
            [
                float(
                    signature.numeric_values()[feature]
                )
                for feature in feature_names
            ]
            for signature in signatures
        ],
        dtype=float,
    )

    return X, feature_names


def evaluate_monotonicity(
    baseline: StatisticalBaseline,
    images: list[np.ndarray],
) -> pd.DataFrame:
    """
    Teste la monotonie de la distance de Mahalanobis
    en fonction du niveau de dégradation.

    Pour chaque scénario :
        niveau de dégradation
            ↓
        image dégradée
            ↓
        Digital Signature
            ↓
        D² / D
    """

    rows: list[dict] = []

    scenarios = (
        ScenarioType.SPECKLE_PROGRESSION,
        ScenarioType.NOISE_PROGRESSION,
        ScenarioType.BLUR_PROGRESSION,
        ScenarioType.CONTRAST_PROGRESSION,
        ScenarioType.INTENSITY_PROGRESSION,
    )

    for scenario in scenarios:

        scenario_result = run_scenario(
            images[0],
            scenario,
            seed=C.RANDOM_SEED,
        )

        for level_index, degraded_image in enumerate(
            scenario_result
        ):

            signature = build_digital_signature_from_image(
                degraded_image.image,
                source="simulated",
            )

            vector = np.asarray(
                [
                    float(
                        signature.numeric_values()[feature]
                    )
                    for feature in baseline.feature_names
                ],
                dtype=float,
            )

            d2 = float(
                baseline.mahalanobis_squared(
                    vector
                )
            )

            distance = float(
                np.sqrt(d2)
            )

            severity = float(
                degraded_image.simulation_severity
                if degraded_image.simulation_severity is not None
                else level_index
            )

            rows.append(
                {
                    "scenario": scenario.value,
                    "level": severity,
                    "mean_D": distance,
                    "D2": d2,
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------------

def main() -> None:

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage : python tools/public_benchmark.py "
            "chemin/vers/dataset"
        )

    dataset_path = Path(sys.argv[1])

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {dataset_path}"
        )

    # --------------------------------------------------------------
    # Chargement des données publiques
    # --------------------------------------------------------------

    acquisitions = load_public(
        dataset_path
    )

    n = len(acquisitions)

    if n < 20:
        raise RuntimeError(
            f"Seulement {n} images exploitables : minimum 20 requis."
        )

    images = [
        acquisition.image
        for acquisition in acquisitions
    ]

    source = acquisitions[0].source

    if source != "public":
        raise RuntimeError(
            f"Source inattendue : {source}. "
            "Le benchmark doit utiliser uniquement des données publiques."
        )

    # --------------------------------------------------------------
    # Séparation développement / évaluation
    # --------------------------------------------------------------

    split = n // 2

    baseline_images = images[:split]
    evaluation_images = images[split:]

    X, feature_names = build_signature_matrix(
        baseline_images,
        source="public",
    )

    baseline = StatisticalBaseline(
        X_reference=X,
        feature_names=feature_names,
    )

    # --------------------------------------------------------------
    # Statistiques des features publiques
    # --------------------------------------------------------------

    feature_stats = pd.DataFrame(
        X,
        columns=feature_names,
    ).describe().T

    # --------------------------------------------------------------
    # Test de monotonie
    # --------------------------------------------------------------

    benchmark = evaluate_monotonicity(
        baseline,
        evaluation_images,
    )

    monotonicity: dict[str, float] = {}

    for scenario_name, group in benchmark.groupby(
        "scenario"
    ):

        if len(group) < 2:
            monotonicity[scenario_name] = float("nan")
            continue

        correlation, _ = spearmanr(
            group["level"],
            group["mean_D"],
        )

        monotonicity[scenario_name] = float(
            correlation
        )

    # --------------------------------------------------------------
    # Sortie
    # --------------------------------------------------------------

    output_dir = C.OUTPUT_DIR
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_stats.to_csv(
        output_dir / "public_feature_stats.csv"
    )

    benchmark.to_csv(
        output_dir / "public_monotonicity.csv",
        index=False,
    )

    summary = {
        "dataset": str(dataset_path),
        "source": source,
        "n_images": n,
        "development_images": len(baseline_images),
        "evaluation_images": len(evaluation_images),
        "n_features": len(feature_names),
        "signature_version": "2.0",
        "monotonicity_spearman": monotonicity,
    }

    pd.Series(summary).to_json(
        output_dir / "public_benchmark_summary.json",
        force_ascii=False,
        indent=2,
    )

    # --------------------------------------------------------------
    # Affichage
    # --------------------------------------------------------------

    print("=" * 70)
    print("SCAN A DIGITAL TWIN V2 — PUBLIC BENCHMARK")
    print("=" * 70)

    print(
        f"DATASET          : {dataset_path}"
    )

    print(
        f"IMAGES           : {n}"
    )

    print(
        f"DEVELOPMENT      : {len(baseline_images)}"
    )

    print(
        f"EVALUATION       : {len(evaluation_images)}"
    )

    print(
        f"FEATURES         : {len(feature_names)}"
    )

    print("-" * 70)

    print(
        "MONOTONICITY — Spearman(level, D)"
    )

    for scenario_name, value in monotonicity.items():
        if np.isnan(value):
            print(
                f"{scenario_name:28s}: n/a"
            )
        else:
            print(
                f"{scenario_name:28s}: {value:.3f}"
            )

    print("-" * 70)

    print(
        f"OUTPUT DIRECTORY : "
        f"{output_dir.resolve()}"
    )

    print(
        "PUBLIC BENCHMARK V2 OK"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()