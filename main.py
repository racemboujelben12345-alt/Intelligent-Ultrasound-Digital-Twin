from __future__ import annotations

from pathlib import Path
import argparse
import logging
import time

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
LOGGER = logging.getLogger("ultrasound_digital_twin")


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


def run_pipeline(*, source_mode: str | None = None, output_dir: Path | None = None, seed: int | None = None) -> dict:
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

    effective_output_dir = ensure_output_dir(
        output_dir if output_dir is not None else C.OUTPUT_DIR
    )

    # ==============================================================
    # 2. ACQUISITION DES DONNÉES
    # ==============================================================

    acquisitions = get_acquisitions(
        prefer_experimental=True,
        minimum_experimental=C.MIN_BASELINE_ACQUISITIONS,
        source_mode=(source_mode or C.ACQUISITION_SOURCE_MODE),
        demo_size=(C.MIN_TOTAL_ACQUISITIONS + 10),
        seed=(C.RANDOM_SEED if seed is None else seed),
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

    source_counts = {}
    for acquisition in acquisitions:
        source_counts[acquisition.source] = source_counts.get(acquisition.source, 0) + 1

    print(f"SOURCE POLICY    : {C.ACQUISITION_SOURCE_MODE}")
    print(f"SOURCE           : {source}")
    print(f"ACQUISITIONS     : {len(acquisitions)}")
    print(f"SOURCE COUNTS    : {source_counts}")

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

    for acquisition in test_acquisitions:
        result = analyzer.analyze(
            image=acquisition.image,
            acquisition_id=acquisition.id,
            source=acquisition.source,
            provenance=acquisition.provenance,
            timestamp=(
                acquisition.timestamp.isoformat()
                if hasattr(acquisition.timestamp, "isoformat")
                else acquisition.timestamp
            ),
            params=acquisition.params,
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

                "physical_evidence":
                    result.fusion.physical_evidence,

                "physical_consistency_score":
                    result.signature.physical_consistency_score,

                "physics_metadata_completeness":
                    result.state.physics_metadata_completeness,

                "evidence_fusion_score":
                    result.state.evidence_fusion_score,

                "evidence_disagreement":
                    result.state.evidence_disagreement,

                "causal_top_mechanism":
                    result.state.causal_top_mechanism,

                "causal_agreement_score":
                    result.state.causal_agreement_score,

                "wavelength_mm":
                    result.signature.wavelength_mm,

                "axial_resolution_mm":
                    result.signature.axial_resolution_mm,

                "attenuation_proxy_db_cm_mhz":
                    result.signature.attenuation_proxy_db_cm_mhz,

                "depth_uniformity":
                    result.signature.depth_uniformity,

                "intelligence_state":
                    result.fusion.state,

                "twin_state":
                    result.twin_state.state,

                "twin_state_confidence":
                    result.twin_state.confidence,

                "twin_state_rationale":
                    result.twin_state.rationale,

                "health_index":
                    result.health.health_index,

                "health_confidence":
                    result.health.confidence_score,

                "health_quality_component":
                    result.health.quality_component,

                "health_anomaly_component":
                    result.health.anomaly_component,

                "health_state":
                    result.health.health_state,

                "health_dominant_evidence":
                    ", ".join(result.health.dominant_evidence),

                "ai_top_feature":
                    result.ai_feature_contributions[0][0]
                    if result.ai_feature_contributions else None,
            }
        )

    save_csv(
        state_rows,
        effective_output_dir / "twin_states.csv",
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
        effective_output_dir / "features.csv",
    )

    # ==============================================================
    # 13. EXPORT DE LA BASELINE
    # ==============================================================

    baseline_summary = baseline.summary()

    save_json(
        {
            "project": PROJECT_NAME,

            "source": source,
            "source_policy": C.ACQUISITION_SOURCE_MODE,
            "source_counts": source_counts,

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
        effective_output_dir / "baseline.json",
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
        effective_output_dir / "summary.json",
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
        effective_output_dir / "report.md",
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
        f"{Path(effective_output_dir).resolve()}"
    )

    print("-" * 70)
    print("DIGITAL TWIN ULTRASOUND DIGITAL TWIN PIPELINE OK")
    print("=" * 70)

    
    return {
        "project": PROJECT_NAME,
        "source": source,
        "acquisitions": len(acquisitions),
        "baseline": len(training_images),
        "calibration": len(calibration_images),
        "test": len(test_images),
        "latest_state": latest.state.state,
        "latest_quality": latest.state.quality_score,
        "output_dir": str(Path(effective_output_dir).resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Intelligent Ultrasound Digital Twin pipeline."
    )
    parser.add_argument("--source", choices=("AUTO", "EXPERIMENTAL", "PUBLIC", "SIMULATED"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-size", type=int, default=None)
    parser.add_argument("--holdout-size", type=int, default=None)
    parser.add_argument("--version", action="version", version=f"%(prog)s {C.PROJECT_VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.seed is not None:
        C.RANDOM_SEED = args.seed
    if args.baseline_size is not None:
        if args.baseline_size < 2:
            parser.error("--baseline-size must be >= 2")
        C.BASELINE_SIZE = args.baseline_size
    if args.holdout_size is not None:
        if args.holdout_size < 1:
            parser.error("--holdout-size must be >= 1")
        C.HOLDOUT_SIZE = args.holdout_size

    C.MIN_BASELINE_ACQUISITIONS = C.BASELINE_SIZE
    C.MIN_HOLDOUT_ACQUISITIONS = C.HOLDOUT_SIZE
    C.MIN_TOTAL_ACQUISITIONS = C.BASELINE_SIZE + C.HOLDOUT_SIZE + 1

    started = time.perf_counter()
    LOGGER.info("Starting %s v%s", PROJECT_NAME, C.PROJECT_VERSION)
    LOGGER.info(
        "source=%s seed=%s baseline=%d holdout=%d",
        args.source or C.ACQUISITION_SOURCE_MODE,
        C.RANDOM_SEED,
        C.BASELINE_SIZE,
        C.HOLDOUT_SIZE,
    )

    try:
        result = run_pipeline(
            source_mode=args.source,
            output_dir=args.output,
            seed=C.RANDOM_SEED,
        )
    except KeyboardInterrupt:
        LOGGER.error("Execution interrupted by user.")
        return 130
    except Exception as exc:
        elapsed = time.perf_counter() - started
        LOGGER.error(
            "Pipeline FAILED after %.2fs: %s: %s",
            elapsed,
            type(exc).__name__,
            exc,
        )
        LOGGER.error(
            "Check acquisition source, dataset availability, "
            "baseline/holdout sizes and dependencies."
        )
        return 1

    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Pipeline completed successfully in %.2fs | state=%s | "
        "quality=%.2f | output=%s",
        elapsed,
        result["latest_state"],
        result["latest_quality"],
        result["output_dir"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
