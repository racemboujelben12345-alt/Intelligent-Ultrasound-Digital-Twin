from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    CUSUM_H,
    CUSUM_K,
    EWMA_LAMBDA,
    EWMA_LIMIT,
    PERSISTENCE,
)


# ============================================================
# DRIFT STATUS
# ============================================================

VALID_DRIFT_STATUSES = {
    "STABLE",
    "DRIFT_SIGNAL",
    "PERSISTENT_DRIFT",
}


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DriftMonitorConfig:
    """
    Configuration du monitoring temporel.

    Le signal surveillé doit être une mesure scalaire cohérente
    entre acquisitions, par exemple D² de Mahalanobis.
    """

    ewma_lambda: float = EWMA_LAMBDA
    ewma_limit: float = EWMA_LIMIT

    cusum_k: float = CUSUM_K
    cusum_h: float = CUSUM_H

    persistence: int = PERSISTENCE

    def validate(self) -> None:
        if not 0.0 < self.ewma_lambda <= 1.0:
            raise ValueError(
                "ewma_lambda doit être dans ]0, 1]."
            )

        if self.ewma_limit <= 0.0:
            raise ValueError(
                "ewma_limit doit être strictement positif."
            )

        if self.cusum_k < 0.0:
            raise ValueError(
                "cusum_k doit être positif ou nul."
            )

        if self.cusum_h <= 0.0:
            raise ValueError(
                "cusum_h doit être strictement positif."
            )

        if self.persistence < 1:
            raise ValueError(
                "persistence doit être >= 1."
            )


# ============================================================
# OBSERVATION
# ============================================================

@dataclass(frozen=True)
class DriftObservation:
    """
    Résultat d'une observation temporelle.
    """

    index: int
    signal: float
    z_score: float

    ewma: float

    cusum_positive: float
    cusum_negative: float

    point_alarm: bool
    ewma_alarm: bool
    cusum_alarm: bool

    consecutive_alarms: int
    persistent_alarm: bool

    status: str

    def validate(self) -> None:
        numeric_values = (
            self.signal,
            self.z_score,
            self.ewma,
            self.cusum_positive,
            self.cusum_negative,
        )

        if not all(
            np.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "Les valeurs du drift monitor doivent être finies."
            )

        if self.index < 0:
            raise ValueError(
                "index doit être positif ou nul."
            )

        if self.consecutive_alarms < 0:
            raise ValueError(
                "consecutive_alarms invalide."
            )

        if self.status not in VALID_DRIFT_STATUSES:
            raise ValueError(
                f"Statut de drift inconnu : {self.status}"
            )


# ============================================================
# DRIFT MONITOR
# ============================================================

class DriftMonitor:
    """
    Moniteur temporel EWMA + CUSUM.

    Architecture
    ------------
        scalar surveillance signal
                    ↓
             standardisation
                    ↓
             EWMA + CUSUM
                    ↓
             persistence logic
                    ↓
              drift status

    Le module ne modifie pas la baseline.
    Il observe uniquement l'évolution temporelle.
    """

    def __init__(
        self,
        reference_mean: float,
        reference_std: float,
        config: DriftMonitorConfig | None = None,
    ) -> None:

        self.reference_mean = float(
            reference_mean
        )

        self.reference_std = float(
            reference_std
        )

        if not np.isfinite(
            self.reference_mean
        ):
            raise ValueError(
                "reference_mean doit être fini."
            )

        if (
            not np.isfinite(self.reference_std)
            or self.reference_std <= 0.0
        ):
            raise ValueError(
                "reference_std doit être strictement positif."
            )

        if config is None:
            config = DriftMonitorConfig()

        config.validate()

        self.config = config

        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Réinitialise l'état temporel du monitor.
        """

        self._index = 0
        self._ewma = 0.0

        self._cusum_positive = 0.0
        self._cusum_negative = 0.0

        self._consecutive_alarms = 0

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        signal: float,
    ) -> DriftObservation:
        """
        Ajoute une nouvelle observation temporelle.

        Parameters
        ----------
        signal:
            Signal numérique à surveiller.
            Exemple : D² de Mahalanobis.
        """

        signal = float(signal)

        if not np.isfinite(signal):
            raise ValueError(
                "signal doit être fini."
            )

        # ----------------------------------------------------
        # Standardisation
        # ----------------------------------------------------

        z = (
            signal - self.reference_mean
        ) / self.reference_std

        # ----------------------------------------------------
        # EWMA
        # ----------------------------------------------------

        self._ewma = (
            self.config.ewma_lambda * z
            + (
                1.0
                - self.config.ewma_lambda
            )
            * self._ewma
        )

        ewma_std_factor = np.sqrt(
            self.config.ewma_lambda
            / (
                2.0
                - self.config.ewma_lambda
            )
        )

        ewma_limit = (
            self.config.ewma_limit
            * ewma_std_factor
        )

        ewma_alarm = (
            abs(self._ewma)
            >= ewma_limit
        )

        # ----------------------------------------------------
        # CUSUM
        # ----------------------------------------------------

        self._cusum_positive = max(
            0.0,
            self._cusum_positive
            + z
            - self.config.cusum_k,
        )

        self._cusum_negative = min(
            0.0,
            self._cusum_negative
            + z
            + self.config.cusum_k,
        )

        cusum_alarm = (
            self._cusum_positive
            >= self.config.cusum_h
            or
            abs(self._cusum_negative)
            >= self.config.cusum_h
        )

        # ----------------------------------------------------
        # Point alarm
        # ----------------------------------------------------

        point_alarm = (
            abs(z)
            >= self.config.ewma_limit
        )

        # ----------------------------------------------------
        # Combined temporal signal
        # ----------------------------------------------------

        current_alarm = (
            point_alarm
            or ewma_alarm
            or cusum_alarm
        )

        if current_alarm:
            self._consecutive_alarms += 1
        else:
            self._consecutive_alarms = 0

        persistent_alarm = (
            self._consecutive_alarms
            >= self.config.persistence
        )

        if persistent_alarm:
            status = "PERSISTENT_DRIFT"
        elif current_alarm:
            status = "DRIFT_SIGNAL"
        else:
            status = "STABLE"

        result = DriftObservation(
            index=self._index,
            signal=signal,
            z_score=float(z),
            ewma=float(self._ewma),
            cusum_positive=float(
                self._cusum_positive
            ),
            cusum_negative=float(
                self._cusum_negative
            ),
            point_alarm=bool(
                point_alarm
            ),
            ewma_alarm=bool(
                ewma_alarm
            ),
            cusum_alarm=bool(
                cusum_alarm
            ),
            consecutive_alarms=(
                self._consecutive_alarms
            ),
            persistent_alarm=bool(
                persistent_alarm
            ),
            status=status,
        )

        result.validate()

        self._index += 1

        return result

    # ========================================================
    # STREAM
    # ========================================================

    def update_many(
        self,
        signals: list[float] | tuple[float, ...],
    ) -> tuple[DriftObservation, ...]:
        """
        Traite une séquence temporelle complète.
        """

        return tuple(
            self.update(signal)
            for signal in signals
        )