"""
SCAN A Digital Twin V2
======================

Temporal Drift Engine
---------------------

Rôle
----
Orchestrer le monitoring temporel d'un Digital Twin.

Architecture
------------
DigitalTwinHistory
        ↓
Mahalanobis² series
        ↓
DriftCalibration
        ↓
DriftMonitor
        ↓
Temporal drift observations

Ce module ne :
- construit pas la baseline ;
- ne calcule pas les Digital Signatures ;
- ne détecte pas les anomalies instantanées ;
- ne modifie pas l'historique.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.digital_twin.history import DigitalTwinHistory
from src.drift.calibration import DriftCalibration
from src.drift.control_charts import (
    DriftMonitor,
    DriftMonitorConfig,
    DriftObservation,
)


@dataclass(frozen=True)
class DriftAnalysis:
    """
    Résultat complet d'une analyse temporelle.
    """

    observations: tuple[DriftObservation, ...]

    def validate(self) -> None:
        if not self.observations:
            raise ValueError(
                "Aucune observation de drift."
            )

        for observation in self.observations:
            observation.validate()

    @property
    def latest(self) -> DriftObservation:
        """
        Retourne la dernière observation temporelle.
        """

        if not self.observations:
            raise ValueError(
                "Aucune observation disponible."
            )

        return self.observations[-1]

    @property
    def persistent_drift_detected(self) -> bool:
        """
        Indique si un drift persistant a été observé.
        """

        return any(
            observation.status == "PERSISTENT_DRIFT"
            for observation in self.observations
        )


class DriftEngine:
    """
    Moteur officiel de monitoring temporel.
    """

    def __init__(
        self,
        calibration: DriftCalibration,
        config: DriftMonitorConfig | None = None,
    ) -> None:

        calibration.validate()

        self.calibration = calibration

        if config is None:
            config = DriftMonitorConfig()

        config.validate()

        self.config = config

    def create_monitor(self) -> DriftMonitor:
        """
        Construit un DriftMonitor à partir de la calibration.
        """

        return DriftMonitor(
            reference_mean=self.calibration.mean_d2,
            reference_std=self.calibration.std_d2,
            config=self.config,
        )

    def analyze_history(
        self,
        history: DigitalTwinHistory,
    ) -> DriftAnalysis:
        """
        Analyse la série temporelle D² d'un historique.
        """

        history.validate()

        d2_series = history.mahalanobis_squared_series

        if not d2_series:
            raise ValueError(
                "L'historique ne contient aucune valeur D²."
            )

        monitor = self.create_monitor()

        observations = monitor.update_many(
            list(d2_series)
        )

        result = DriftAnalysis(
            observations=observations
        )

        result.validate()

        return result