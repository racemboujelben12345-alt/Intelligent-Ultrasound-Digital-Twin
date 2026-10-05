"""
Intelligent Ultrasound Digital Twin
======================

Temporal Prediction Engine
--------------------------

Rôle
----
Analyser la trajectoire temporelle des signaux du Digital Twin.

Le moteur utilise l'historique du Digital Twin pour estimer
des tendances statistiques sur :

- Mahalanobis² ;
- Mahalanobis ;
- score de conformité.

Important
---------
Cette analyse décrit une trajectoire statistique.

Elle ne prédit PAS :
- une panne physique ;
- une date de panne ;
- une durée de vie réelle ;
- un événement clinique.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.digital_twin.history import DigitalTwinHistory
from src.prediction.trend import (
    TrendAnalyzer,
    TrendResult,
)


@dataclass(frozen=True)
class TemporalPredictionResult:
    """
    Résultats des analyses de tendance temporelle.
    """

    mahalanobis_squared: TrendResult
    mahalanobis_distance: TrendResult
    quality_score: TrendResult

    def validate(self) -> None:
        self.mahalanobis_squared.validate()
        self.mahalanobis_distance.validate()
        self.quality_score.validate()

    @property
    def d2_direction(self) -> str:
        return self.mahalanobis_squared.direction

    @property
    def quality_direction(self) -> str:
        return self.quality_score.direction


class TemporalPredictionEngine:
    """
    Moteur d'analyse temporelle du Digital Twin.
    """

    def __init__(
        self,
        analyzer: TrendAnalyzer | None = None,
    ) -> None:

        if analyzer is None:
            analyzer = TrendAnalyzer(
                stable_slope_threshold=0.05
            )

        self.analyzer = analyzer

    def analyze_history(
        self,
        history: DigitalTwinHistory,
    ) -> TemporalPredictionResult:
        """
        Analyse la trajectoire temporelle des principaux
        signaux enregistrés dans l'historique.
        """

        history.validate()

        if len(history) < 2:
            raise ValueError(
                "Au moins deux états sont nécessaires "
                "pour analyser une tendance."
            )

        d2_series = history.mahalanobis_squared_series
        distance_series = history.mahalanobis_distance_series
        quality_series = history.quality_score_series

        d2_trend = self.analyzer.analyze(
            d2_series
        )

        distance_trend = self.analyzer.analyze(
            distance_series
        )

        quality_trend = self.analyzer.analyze(
            quality_series
        )

        result = TemporalPredictionResult(
            mahalanobis_squared=d2_trend,
            mahalanobis_distance=distance_trend,
            quality_score=quality_trend,
        )

        result.validate()

        return result