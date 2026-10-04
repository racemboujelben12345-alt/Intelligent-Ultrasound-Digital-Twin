"""
Intelligent Ultrasound Digital Twin V2
======================

Digital Twin State.

Rôle
----
Représenter l'état courant du Digital Twin après analyse
d'une acquisition.

Cette couche ne réalise pas :
    - l'extraction des features ;
    - la construction de la baseline ;
    - la détection statistique elle-même ;
    - la simulation ;
    - la prédiction.

Elle rassemble les résultats produits par ces couches
afin de former un état cohérent et exploitable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping

import numpy as np


# ============================================================
# ÉTATS AUTORISÉS
# ============================================================

VALID_STATES = (
    "NOMINAL",
    "EARLY_DRIFT",
    "SIGNIFICANT_DRIFT",
    "HIGH_DEVIATION",
)


# ============================================================
# DIGITAL TWIN STATE
# ============================================================

@dataclass(frozen=True)
class DigitalTwinState:
    """
    État courant du Digital Twin.

    Il représente une observation à un instant donné.

    Exemple conceptuel
    ------------------
        acquisition
             ↓
        analyse statistique
             ↓
        DigitalTwinState
    """

    # --------------------------------------------------------
    # IDENTIFICATION
    # --------------------------------------------------------

    acquisition_id: str
    source: str

    # --------------------------------------------------------
    # TEMPS
    # --------------------------------------------------------

    timestamp: str

    # --------------------------------------------------------
    # ÉTAT GLOBAL
    # --------------------------------------------------------

    state: str

    # --------------------------------------------------------
    # INDICATEURS STATISTIQUES
    # --------------------------------------------------------

    mahalanobis_distance: float
    mahalanobis_squared: float

    quality_score: float

    # --------------------------------------------------------
    # EXPLICATIONS PAR DIMENSION
    # --------------------------------------------------------

    dimension_scores: dict[str, float]

    # --------------------------------------------------------
    # CONTRIBUTIONS DES FEATURES
    # --------------------------------------------------------

    feature_contributions: tuple[
        tuple[str, float],
        ...
    ]

    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    state_version: str = "2.0"

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> None:
        """
        Vérifie la cohérence de l'état.
        """

        if not self.acquisition_id:
            raise ValueError(
                "acquisition_id ne peut pas être vide."
            )

        if not self.source:
            raise ValueError(
                "source ne peut pas être vide."
            )

        if self.state not in VALID_STATES:
            raise ValueError(
                f"État invalide : {self.state}. "
                f"États autorisés : {VALID_STATES}"
            )

        if not np.isfinite(
            self.mahalanobis_distance
        ):
            raise ValueError(
                "mahalanobis_distance doit être finie."
            )

        if not np.isfinite(
            self.mahalanobis_squared
        ):
            raise ValueError(
                "mahalanobis_squared doit être fini."
            )

        if self.mahalanobis_distance < 0:
            raise ValueError(
                "La distance de Mahalanobis ne peut "
                "pas être négative."
            )

        if self.mahalanobis_squared < 0:
            raise ValueError(
                "La distance de Mahalanobis² ne peut "
                "pas être négative."
            )

        if not np.isfinite(
            self.quality_score
        ):
            raise ValueError(
                "quality_score doit être fini."
            )

        if not (
            0.0 <= self.quality_score <= 100.0
        ):
            raise ValueError(
                "quality_score doit être compris "
                "entre 0 et 100."
            )

        for name, value in (
            self.dimension_scores.items()
        ):
            if not np.isfinite(value):
                raise ValueError(
                    f"Dimension non finie : "
                    f"{name}={value}"
                )

            if value < 0:
                raise ValueError(
                    f"Dimension négative : "
                    f"{name}={value}"
                )

        for feature, contribution in (
            self.feature_contributions
        ):
            if not feature:
                raise ValueError(
                    "Le nom d'une feature "
                    "ne peut pas être vide."
                )

            if not np.isfinite(
                contribution
            ):
                raise ValueError(
                    f"Contribution non finie : "
                    f"{feature}={contribution}"
                )

            if contribution < 0:
                raise ValueError(
                    f"Contribution négative : "
                    f"{feature}={contribution}"
                )

        if not self.state_version:
            raise ValueError(
                "state_version est obligatoire."
            )

    # ========================================================
    # CONVERSION DICTIONNAIRE
    # ========================================================

    def to_dict(self) -> dict:
        """
        Convertit l'état en dictionnaire sérialisable.
        """

        self.validate()

        return asdict(self)

    # ========================================================
    # RÉSUMÉ HUMAIN
    # ========================================================

    def summary(self) -> str:
        """
        Retourne un résumé court de l'état.
        """

        return (
            f"Acquisition={self.acquisition_id} | "
            f"Source={self.source} | "
            f"État={self.state} | "
            f"Mahalanobis={self.mahalanobis_distance:.3f} | "
            f"Quality={self.quality_score:.2f}"
        )


# ============================================================
# CONSTRUCTION D'UN ÉTAT
# ============================================================

def build_twin_state(
    acquisition_id: str,
    source: str,
    state: str,
    mahalanobis_distance: float,
    mahalanobis_squared: float,
    quality_score: float,
    dimension_scores: Mapping[str, float] | None = None,
    feature_contributions: list[
        tuple[str, float]
    ] | tuple[
        tuple[str, float],
        ...
    ] | None = None,
    timestamp: str | None = None,
) -> DigitalTwinState:
    """
    Construit un état du Digital Twin.

    Parameters
    ----------
    acquisition_id:
        Identifiant de l'acquisition.

    source:
        Source de l'acquisition.

    state:
        État statistique calculé.

    mahalanobis_distance:
        Distance de Mahalanobis.

    mahalanobis_squared:
        Distance de Mahalanobis au carré.

    quality_score:
        Score qualité relatif.

    dimension_scores:
        Indicateurs par dimension.

    feature_contributions:
        Contributions des features.

    timestamp:
        Horodatage ISO 8601.
        Si absent, l'heure UTC actuelle est utilisée.
    """

    if timestamp is None:
        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

    if dimension_scores is None:
        dimension_scores = {}

    if feature_contributions is None:
        feature_contributions = []

    twin_state = DigitalTwinState(
        acquisition_id=acquisition_id,
        source=source,
        timestamp=timestamp,
        state=state,
        mahalanobis_distance=float(
            mahalanobis_distance
        ),
        mahalanobis_squared=float(
            mahalanobis_squared
        ),
        quality_score=float(
            quality_score
        ),
        dimension_scores={
            key: float(value)
            for key, value in dimension_scores.items()
        },
        feature_contributions=tuple(
            (
                str(feature),
                float(contribution),
            )
            for feature, contribution
            in feature_contributions
        ),
    )

    twin_state.validate()

    return twin_state