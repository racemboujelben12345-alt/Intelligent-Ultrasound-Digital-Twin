"""
Intelligent Ultrasound Digital Twin V2
======================

Digital Twin Analyzer
---------------------

Rôle
----
Orchestrer l'analyse complète d'une acquisition.

Pipeline
--------
Image
  ↓
Digital Signature
  ↓
Statistical Anomaly Detector
  ↓
Digital Twin State
  ↓
History
  ↓
Temporal Drift
  ↓
Temporal Trend
  ↓
Explainability

Important
---------
Ce module coordonne les composants existants.
Il ne remplace pas leurs responsabilités.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ai.intelligence import (
    AIAnomalyAssessment,
    UltrasoundAIEngine,
)
from src.ai.fusion import IntelligenceFusion, fuse_intelligence
from src.acquisition.provenance import DataProvenance, provenance_evidence_score
from src.anomaly.statistical import (
    AnomalyDetectionResult,
    StatisticalAnomalyDetector,
)
from src.digital_twin.factory import (
    build_twin_state_from_detection,
)
from src.digital_twin.health import (
    TwinHealthAssessment,
    assess_twin_health,
)
from src.digital_twin.history import (
    DigitalTwinHistory,
)
from src.digital_twin.state import (
    DigitalTwinState,
)
from src.digital_twin.state_engine import TwinStateDecision, decide_twin_state
from src.drift.engine import (
    DriftAnalysis,
    DriftEngine,
)
from src.explainability.attribution import (
    Explanation,
    explain_detection,
)
from src.image_analysis.digital_signature import (
    DigitalSignature,
    build_digital_signature_from_image,
)
from src.prediction.engine import (
    TemporalPredictionEngine,
    TemporalPredictionResult,
)
from src.signature.baseline import (
    StatisticalBaseline,
)


@dataclass(frozen=True)
class TwinAnalysisResult:
    """
    Résultat complet d'analyse du Digital Twin.

    Les résultats temporels peuvent être absents lorsque
    l'historique contient moins de deux observations.
    """

    signature: DigitalSignature

    detection: AnomalyDetectionResult

    state: DigitalTwinState

    explanation: Explanation

    drift: DriftAnalysis | None

    trend: TemporalPredictionResult | None

    health: TwinHealthAssessment

    ai: AIAnomalyAssessment
    fusion: IntelligenceFusion
    ai_feature_contributions: tuple[tuple[str, float], ...]
    twin_state: TwinStateDecision

    def validate(self) -> None:
        self.detection.validate()
        self.state.validate()
        self.health.validate()
        self.ai.validate()
        self.fusion.validate()
        for name, value in self.ai_feature_contributions:
            if not name or not np.isfinite(value) or value < 0.0:
                raise ValueError("Invalid AI feature contribution.")
        self.explanation.validate()
        self.twin_state.validate()

        if self.drift is not None:
            self.drift.validate()

        if self.trend is not None:
            self.trend.validate()

        if self.state.state != self.detection.state:
            raise ValueError(
                "Incohérence entre detection.state et state.state."
            )

        if self.health.health_state not in {
            "NOMINAL", "WATCH", "EARLY_DRIFT", "HIGH_DEVIATION"
        }:
            raise ValueError("Invalid Twin Health state.")

        if not np.isclose(
            self.state.mahalanobis_squared,
            self.detection.d2,
        ):
            raise ValueError(
                "Incohérence entre D² et DigitalTwinState."
            )

        if not np.isclose(
            self.state.mahalanobis_distance,
            self.detection.distance,
        ):
            raise ValueError(
                "Incohérence entre D et DigitalTwinState."
            )


class DigitalTwinAnalyzer:
    """
    Orchestrateur principal du Digital Twin.

    Responsabilités
    ---------------
    - construire une Digital Signature ;
    - exécuter le détecteur statistique ;
    - construire le DigitalTwinState ;
    - enrichir l'historique ;
    - déclencher l'analyse temporelle ;
    - produire l'explication.

    Il ne contient pas les algorithmes internes
    de chaque sous-module.
    """

    def __init__(
        self,
        baseline: StatisticalBaseline,
        history: DigitalTwinHistory | None = None,
        drift_engine: DriftEngine | None = None,
        trend_engine: TemporalPredictionEngine | None = None,
        ai_training_source: str = "baseline_reference",
    ) -> None:

        if not isinstance(
            baseline,
            StatisticalBaseline,
        ):
            raise TypeError(
                "baseline doit être une instance de StatisticalBaseline."
            )

        self.baseline = baseline

        if history is None:
            history = DigitalTwinHistory()

        self.history = history

        self.detector = StatisticalAnomalyDetector(
            baseline=baseline
        )

        self.ai_engine = UltrasoundAIEngine()
        self.ai_engine.fit_reference(
            baseline.X_reference,
            feature_names=baseline.feature_names,
            training_source=ai_training_source,
        )

        self.drift_engine = drift_engine

        self.trend_engine = trend_engine

    def analyze(
        self,
        *,
        image: np.ndarray,
        acquisition_id: str,
        source: str = "simulated",
        provenance: DataProvenance | None = None,
        timestamp: str | None = None,
        update_history: bool = True,
        explain_top_k: int = 5,
    ) -> TwinAnalysisResult:
        """
        Analyse une acquisition complète.

        Parameters
        ----------
        image:
            Image 2D normalisée [0,1].

        acquisition_id:
            Identifiant unique de l'acquisition.

        source:
            Provenance logique de l'acquisition.

        timestamp:
            Timestamp optionnel.

        update_history:
            Si True, l'état produit est ajouté à l'historique.

        explain_top_k:
            Nombre maximal de contributions explicatives.
        """

        if not acquisition_id:
            raise ValueError(
                "acquisition_id ne peut pas être vide."
            )

        if provenance is not None:
            provenance.validate()
            expected_category = (
                "experimental" if source == "scan_a" else source
            )
            if provenance.source_category != expected_category:
                raise ValueError(
                    "Incohérence entre source d'analyse et provenance."
                )

        if explain_top_k < 1:
            raise ValueError(
                "explain_top_k doit être >= 1."
            )

        # ============================================================
        # 1. Digital Signature
        # ============================================================

        signature = build_digital_signature_from_image(
            image,
            source=source,
        )

        vector = signature.to_vector()

        # ============================================================
        # 2. Anomaly Detection
        # ============================================================

        detection = self.detector.detect(
            vector
        )

        # ============================================================
        # 3. Digital Twin State
        # ============================================================

        state = build_twin_state_from_detection(
            acquisition_id=acquisition_id,
            source=source,
            vector=vector,
            detection=detection,
            baseline=self.baseline,
            timestamp=timestamp,
        )

        # ============================================================
        # 4. AI Ensemble Intelligence
        # ============================================================

        ai = self.ai_engine.assess(vector)
        ai_feature_contributions = tuple(
            self.ai_engine.anomaly_feature_contributions(vector)[:explain_top_k]
        )
        fusion = fuse_intelligence(
            mahalanobis_squared=detection.d2,
            critical_threshold=self.baseline.thresholds["critical"],
            quality_score=state.quality_score,
            ai=ai,
        )

        # ============================================================
        # 5. Twin Health + Evidence Confidence
        # ============================================================

        provenance_score = provenance_evidence_score(provenance)
        if provenance is None:
            # Legacy callers that only provide a source string remain supported.
            provenance_score = {
                "experimental": 100.0,
                "public": 80.0,
                "simulated": 60.0,
                "scan_a": 100.0,
            }.get(str(source).lower(), 50.0)

        health = assess_twin_health(
            quality_score=state.quality_score,
            mahalanobis_distance=state.mahalanobis_distance,
            baseline_observations=self.baseline.n_samples,
            feature_count=self.baseline.n_features,
            expected_feature_count=self.baseline.n_features,
            provenance_score=provenance_score,
        )

        # ============================================================
        # 6. Explainability
        # ============================================================

        explanation = explain_detection(
            detection,
            top_k=explain_top_k,
        )

        # ============================================================
        # 7. History update
        # ============================================================

        if update_history:
            self.history.add(
                state
            )

        # ============================================================
        # 8. Temporal drift
        # ============================================================

        drift = None

        if (
            self.drift_engine is not None
            and len(self.history) >= 1
        ):
            drift = self.drift_engine.analyze_history(
                self.history
            )

        # ============================================================
        # 9. Temporal trend
        # ============================================================

        trend = None

        if (
            self.trend_engine is not None
            and len(self.history) >= 2
        ):
            trend = self.trend_engine.analyze_history(
                self.history
            )

        # ============================================================
        # 10. Complete result
        # ============================================================

        twin_state = decide_twin_state(
            statistical_state=state.state,
            fusion=fusion,
            health=health,
            drift=drift,
        )

        result = TwinAnalysisResult(
            signature=signature,
            detection=detection,
            state=state,
            explanation=explanation,
            drift=drift,
            trend=trend,
            health=health,
            ai=ai,
            fusion=fusion,
            ai_feature_contributions=ai_feature_contributions,
            twin_state=twin_state,
        )

        result.validate()

        return result

    def analyze_many(
        self,
        acquisitions: list[
            tuple[str, np.ndarray]
        ]
        | tuple[
            tuple[str, np.ndarray], ...
        ],
        *,
        source: str = "simulated",
        update_history: bool = True,
        explain_top_k: int = 5,
    ) -> tuple[TwinAnalysisResult, ...]:
        """
        Analyse plusieurs acquisitions dans l'ordre fourni.
        """

        if not acquisitions:
            raise ValueError(
                "Au moins une acquisition est requise."
            )

        results = []

        for acquisition_id, image in acquisitions:

            result = self.analyze(
                image=image,
                acquisition_id=acquisition_id,
                source=source,
                update_history=update_history,
                explain_top_k=explain_top_k,
            )

            results.append(
                result
            )

        return tuple(results)