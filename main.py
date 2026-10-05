from __future__ import annotations

from pathlib import Path

import numpy as np

import config as C

from src.acquisition import get_acquisitions
from src.digital_twin.analyzer import DigitalTwinAnalyzer
from src.digital_twin.history import DigitalTwinHistory
from src.drift.calibration import calibrate_drift_signal
from src.drift.control_charts import DriftMonitorConfig
from src.drift.engine import DriftEngine
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.prediction.engine import TemporalPredictionEngine
from src.prediction.trend import TrendAnalyzer
from src.reporting import (
    build_summary,
    ensure_output_dir,
    save_csv,
    save_json,
    save_text,
)
from src.signature.baseline import StatisticalBaseline
from src.validation.audit import partition_by_lineage


PROJECT_NAME = C.PROJECT_NAME


def build_signature_matrix(
    images: list[np.ndarray],
    source: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    Construit la matrice des Digital Signatures à partir des images.
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
            "Aucune image disponible pour construire les signatures."
        )

    # numeric_values est une MÉTHODE.
    first_values = signatures[0].numeric_values()

    feature_names = tuple(first_values.keys())

    X = np.asarray(
        [
            [
                float(signature.numeric_values()[name])
                for name in feature_names
            ]
            for signature in signatures
        ],
        dtype=float,
    )

    return X, feature_names


def run_pipeline() -> None:
    """
    Pipeline principal de l'Intelligent Ultrasound Digital Twin.

    Architecture :

    Acquisition
        ↓
    Digital Signature
        ↓
    Baseline statistique
        ↓
    Calibration
        ↓
    Détection d'anomalies
        ↓
    Digital Twin State
        ↓
    Drift temporel
        ↓
    Analyse de tendance
        ↓
    Reporting
    """

    print("=" * 70)
    print(PROJECT_NAME)
    print("=" * 70)

    # ==============================================================
    # 1. DOSSIER DE SORTIE
    # ==============================================================

    output_dir = ensure_output_dir(
        C.OUTPUT_DIR
    )

    # ==============================================================
    # 2. ACQUISITION DES DONNÉES
    # ==============================================================

    acquisitions = get_acquisitions(
        prefer_experimental=True,
        minimum_experimental=C.MIN_BASELINE_ACQUISITIONS,
        demo_size=(
            C.MIN_TOTAL_ACQUISITIONS
            + 10
        ),
        seed=C.RANDOM_SEED,
    )

    minimum_required = C.MIN_TOTAL_ACQUISITIONS

    if len(acquisitions) < minimum_required:
        raise RuntimeError(
            "Nombre insuffisant d'acquisitions pour "
            "exécuter le pipeline du Digital Twin."
        )

    source = acquisitions[0].source

    images = [
        acquisition.image
        for acquisition in acquisitions
    ]

    acquisition_ids = [
        acquisition.id
        for acquisition in acquisitions
    ]

    print(f"SOURCE           : {source}")
    print(f"ACQUISITIONS     : {len(acquisitions)}")

    # ==============================================================
    # 3. PARTITION DES DONNÉES
    # ==============================================================

    training_acquisitions, calibration_acquisitions, test_acquisitions = (
        partition_by_lineage(
            acquisitions,
            baseline_size=C.BASELINE_SIZE,
            holdout_size=C.HOLDOUT_SIZE,
        )
    )

    training_images = [acquisition.image for acquisition in training_acquisitions]
    calibration_images = [acquisition.image for acquisition in calibration_acquisitions]
    test_images = [acquisition.image for acquisition in test_acquisitions]
    test_ids = [acquisition.id for acquisition in test_acquisitions]

    if len(training_images) < C.BASELINE_SIZE:
        raise RuntimeError(
            "Partition baseline insuffisante."
        )

    if len(calibration_images) < C.HOLDOUT_SIZE:
        raise RuntimeError(
            "Partition calibration insuffisante."
        )

    if not test_images:
        raise RuntimeError(
            "Partition test vide."
        )

    print(
        f"BASELINE         : "
        f"{len(training_images)}"
    )

    print(
        f"CALIBRATION      : "
        f"{len(calibration_images)}"
    )

    print(
        f"TEST             : "
        f"{len(test_images)}"
    )

    # ==============================================================
    # 4. DIGITAL SIGNATURE
    # ==============================================================

    X_training, feature_names = build_signature_matrix(
        training_images,
        source,
    )

    print(
        f"FEATURES         : "
        f"{len(feature_names)}"
    )

    # ==============================================================
    # 5. BASELINE STATISTIQUE
    # ==============================================================

    baseline = StatisticalBaseline(
        X_reference=X_training,
        feature_names=feature_names,
    )

    # ==============================================================
    # 6. CALIBRATION DU DRIFT
    # ==============================================================

    calibration = calibrate_drift_signal(
        baseline,
        calibration_images,
        source=source,
    )

    drift_config = DriftMonitorConfig(
        ewma_lambda=C.EWMA_LAMBDA,
        ewma_limit=C.EWMA_LIMIT,
        cusum_k=C.CUSUM_K,
        cusum_h=C.CUSUM_H,
        persistence=C.PERSISTENCE,
    )

    drift_engine = DriftEngine(
        calibration=calibration,
        config=drift_config,
    )

    # ==============================================================
    # 7. MOTEUR DE TENDANCE
    # ==============================================================

    trend_analyzer = TrendAnalyzer(
        stable_slope_threshold=0.05,
    )

    prediction_engine = TemporalPredictionEngine(
        analyzer=trend_analyzer,
    )

    # ==============================================================
    # 8. HISTORIQUE DU DIGITAL TWIN
    # ==============================================================

    history = DigitalTwinHistory()

    # ==============================================================
    # 9. ANALYSEUR CENTRAL
    # ==============================================================

    analyzer = DigitalTwinAnalyzer(
        baseline=baseline,
        history=history,
        drift_engine=drift_engine,
        trend_engine=prediction_engine,
        ai_training_source=f"{source}:baseline_reference",
    )

    # ==============================================================
    # 10. ANALYSE DES ACQUISITIONS DE TEST
    # ==============================================================

    results = []

    for acquisition_id, image in zip(
        test_ids,
        test_images,
    ):
        result = analyzer.analyze(
            image=image,
            acquisition_id=acquisition_id,
            source=source,
            update_history=True,
            explain_top_k=5,
        )

        results.append(result)

    results = tuple(results)

    if not results:
        raise RuntimeError(
            "Aucun résultat n'a été généré."
        )

    # ==============================================================
    # 11. EXPORT DES ÉTATS DU DIGITAL TWIN
    # ==============================================================

    state_rows = []

    for result in results:

        drift_status = None

        if (
            result.drift is not None
            and result.drift.observations
        ):
            drift_status = (
                result.drift.observations[-1].status
            )

        state_rows.append(
            {
                "acquisition_id":
                    result.state.acquisition_id,

                "source":
                    result.state.source,

                "state":
                    result.state.state,

                "mahalanobis_distance":
                    result.state.mahalanobis_distance,

                "mahalanobis_squared":
                    result.state.mahalanobis_squared,

                "quality_score":
                    result.state.quality_score,

                "drift_status":
                    drift_status,

                "ai_anomaly_score":
                    result.ai.anomaly_score,

                "ai_anomaly_vote_rate":
                    result.ai.anomaly_vote_rate,

                "ai_confidence":
                    result.ai.confidence,

                "ai_ensemble_agreement":
                    result.ai.ensemble_agreement,

                "intelligence_fused_score":
                    result.fusion.fused_score,

                "intelligence_confidence":
                    result.fusion.confidence,

                "intelligence_state":
                    result.fusion.state,

                "twin_state":
                    result.twin_state.state,

                "twin_state_confidence":
                    result.twin_state.confidence,

                "twin_state_rationale":
                    result.twin_state.rationale,

                "ai_top_feature":
                    result.ai_feature_contributions[0][0]
                    if result.ai_feature_contributions else None,
            }
        )

    save_csv(
        state_rows,
        output_dir / "twin_states.csv",
    )

    # ==============================================================
    # 12. EXPORT DES DIGITAL SIGNATURES DE TEST
    # ==============================================================

    feature_rows = []

    for acquisition_id, image in zip(
        test_ids,
        test_images,
    ):
        signature = build_digital_signature_from_image(
            image,
            source=source,
        )

        feature_rows.append(
            {
                "acquisition_id": acquisition_id,
                "source": source,
                **signature.numeric_values(),
            }
        )

    save_csv(
        feature_rows,
        output_dir / "features.csv",
    )

    # ==============================================================
    # 13. EXPORT DE LA BASELINE
    # ==============================================================

    baseline_summary = baseline.summary()

    save_json(
        {
            "project": PROJECT_NAME,

            "source": source,

            "signature_version": C.SIGNATURE_VERSION,
            "pipeline_schema_version": C.PIPELINE_SCHEMA_VERSION,
            "project_version": C.PROJECT_VERSION,

            "feature_names":
                list(feature_names),

            "baseline_size":
                len(training_images),

            "calibration_size":
                len(calibration_images),

            "test_size":
                len(test_images),

            "baseline":
                baseline_summary,

            "calibration":
                {
                    "n_samples":
                        calibration.n_samples,

                    "mean_d2":
                        calibration.mean_d2,

                    "std_d2":
                        calibration.std_d2,

                    "median_d2":
                        calibration.median_d2,

                    "mad_d2":
                        calibration.mad_d2,

                    "min_d2":
                        calibration.min_d2,

                    "max_d2":
                        calibration.max_d2,

                    "percentiles":
                        calibration.percentiles,
                },
        },
        output_dir / "baseline.json",
    )

    # ==============================================================
    # 14. RÉSUMÉ GLOBAL
    # ==============================================================

    latest = results[-1]

    summary = build_summary(
        project=PROJECT_NAME,
        n_acquisitions=len(acquisitions),
        source=source,
        baseline_size=len(training_images),
        calibration_size=len(calibration_images),
        test_size=len(test_images),
        latest_state=latest.state.state,
        latest_distance=(
            latest.state.mahalanobis_distance
        ),
        latest_quality=(
            latest.state.quality_score
        ),
    )

    save_json(
        summary,
        output_dir / "summary.json",
    )

    # ==============================================================
    # 15. RAPPORT MARKDOWN
    # ==============================================================

    report_lines = [
        f"# {PROJECT_NAME}",
        "",
        "## Architecture exécutée",
        "",
        (
            "Acquisition → Digital Signature → Quality → "
            "Statistical Baseline → AI Ensemble → Evidence Fusion → "
            "Digital Twin State → Drift → Prediction → V&V → Reporting"
        ),
        "",
        "## Données",
        "",
        f"- Source : {source}",
        (
            "- Acquisitions totales : "
            f"{len(acquisitions)}"
        ),
        (
            "- Baseline : "
            f"{len(training_images)}"
        ),
        (
            "- Calibration : "
            f"{len(calibration_images)}"
        ),
        (
            "- Test : "
            f"{len(test_images)}"
        ),
        (
            "- Variables de signature : "
            f"{len(feature_names)}"
        ),
        "",
        "## AI intelligence",
        "",
        f"- AI anomaly evidence : {latest.ai.anomaly_score:.4f}",
        f"- AI ensemble agreement : {latest.ai.ensemble_agreement:.4f}",
        f"- AI confidence : {latest.ai.confidence:.4f}",
        f"- Fused intelligence score : {latest.fusion.fused_score:.4f}",
        f"- Fused confidence : {latest.fusion.confidence:.4f}",
        f"- Intelligence state : {latest.fusion.state}",
        f"- Unified Twin state : {latest.twin_state.state}",
        f"- Twin state confidence : {latest.twin_state.confidence:.4f}",
        f"- Twin state rationale : {latest.twin_state.rationale}",
        f"- Dominant AI feature : {latest.ai_feature_contributions[0][0] if latest.ai_feature_contributions else 'N/A'}",
        "",
        "## Dernière acquisition analysée",
        "",
        (
            "- Identifiant : "
            f"{latest.state.acquisition_id}"
        ),
        (
            "- État : "
            f"{latest.state.state}"
        ),
        (
            "- Distance de Mahalanobis : "
            f"{latest.state.mahalanobis_distance:.4f}"
        ),
        (
            "- Distance quadratique D² : "
            f"{latest.state.mahalanobis_squared:.4f}"
        ),
        (
            "- Score de qualité expérimental : "
            f"{latest.state.quality_score:.2f}"
        ),
        "",
        "## Calibration du drift",
        "",
        (
            "- N : "
            f"{calibration.n_samples}"
        ),
        (
            "- Moyenne D² : "
            f"{calibration.mean_d2:.4f}"
        ),
        (
            "- Écart-type D² : "
            f"{calibration.std_d2:.4f}"
        ),
        (
            "- Médiane D² : "
            f"{calibration.median_d2:.4f}"
        ),
        (
            "- MAD D² : "
            f"{calibration.mad_d2:.4f}"
        ),
        "",
        "## Limites",
        "",
        (
            "Le score de qualité est un indicateur "
            "statistique expérimental. Il ne constitue "
            "ni un score clinique, ni une probabilité "
            "de panne, ni une mesure physique absolue "
            "de l'état de l'équipement."
        ),
        "",
        (
            "Les dégradations numériques utilisées pour "
            "la validation sont des simulations contrôlées "
            "et ne représentent pas des pannes réelles "
            "d'un équipement d'échographie."
        ),
        "",
        (
            "Les éventuelles données publiques sont "
            "utilisées comme données de référence et "
            "ne doivent pas être confondues avec les "
            "acquisitions expérimentales d'un équipement."
        ),
    ]

    save_text(
        "\n".join(report_lines),
        output_dir / "report.md",
    )

    # ==============================================================
    # 16. AFFICHAGE FINAL
    # ==============================================================

    print("-" * 70)

    print(
        f"LATEST ID        : "
        f"{latest.state.acquisition_id}"
    )

    print(
        f"LATEST STATE     : "
        f"{latest.state.state}"
    )

    print(
        f"MAHALANOBIS D    : "
        f"{latest.state.mahalanobis_distance:.4f}"
    )

    print(
        f"MAHALANOBIS D²   : "
        f"{latest.state.mahalanobis_squared:.4f}"
    )

    print(
        f"QUALITY SCORE    : "
        f"{latest.state.quality_score:.2f}"
    )

    print(
        f"OUTPUT DIRECTORY : "
        f"{Path(output_dir).resolve()}"
    )

    print("-" * 70)
    print("DIGITAL TWIN ULTRASOUND DIGITAL TWIN PIPELINE OK")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()