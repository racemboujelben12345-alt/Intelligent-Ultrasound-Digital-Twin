from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.acquisition.provenance import DataProvenance


VALID_SOURCES = {
    "scan_a",
    "public",
    "simulated",
}


@dataclass
class Acquisition:
    """
    Modèle canonique d'une acquisition du Digital Twin.

    Une acquisition contient :
    - identité ;
    - provenance ;
    - image ;
    - paramètres disponibles ;
    - session ;
    - timestamp ;
    - informations de simulation éventuelles.

    Important
    ---------
    Les paramètres ne sont jamais inventés.
    Seules les valeurs réellement disponibles sont stockées.
    """

    id: str
    source: str

    t: int

    image: np.ndarray

    params: dict = field(
        default_factory=dict
    )

    provenance: Optional[DataProvenance] = None

    session_id: Optional[str] = None

    timestamp: Optional[datetime] = None

    truth_level: float = 0.0

    simulation_scenario: Optional[str] = None

    simulation_severity: Optional[float] = None

    def validate(self) -> None:
        """
        Valide la cohérence de l'acquisition.
        """

        if not self.id:
            raise ValueError(
                "L'identifiant de l'acquisition ne peut pas être vide."
            )

        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"Source invalide : {self.source}. "
                f"Sources autorisées : {sorted(VALID_SOURCES)}"
            )

        if not isinstance(
            self.t,
            int,
        ):
            raise TypeError(
                "t doit être un entier."
            )

        if self.t < 0:
            raise ValueError(
                "t doit être positif ou nul."
            )

        if self.image is None:
            raise ValueError(
                "L'image ne peut pas être None."
            )

        if not isinstance(
            self.image,
            np.ndarray,
        ):
            raise TypeError(
                "image doit être un numpy.ndarray."
            )

        if self.image.ndim != 2:
            raise ValueError(
                "L'image doit être une matrice 2D en niveaux de gris."
            )

        if not np.all(
            np.isfinite(self.image)
        ):
            raise ValueError(
                "L'image contient NaN ou Inf."
            )

        if self.source == "simulated":
            if self.simulation_scenario is None:
                raise ValueError(
                    "Une acquisition simulée doit déclarer "
                    "son scénario de simulation."
                )

            if self.simulation_severity is None:
                raise ValueError(
                    "Une acquisition simulée doit déclarer "
                    "sa sévérité."
                )

            if not 0.0 <= self.simulation_severity <= 1.0:
                raise ValueError(
                    "simulation_severity doit être dans [0,1]."
                )

        else:
            if self.simulation_scenario is not None:
                raise ValueError(
                    "Une acquisition non simulée ne peut pas "
                    "avoir de simulation_scenario."
                )

            if self.simulation_severity is not None:
                raise ValueError(
                    "Une acquisition non simulée ne peut pas "
                    "avoir de simulation_severity."
                )

        if not 0.0 <= self.truth_level <= 1.0:
            raise ValueError(
                "truth_level doit être dans [0,1]."
            )

        if self.provenance is not None:
            self.provenance.validate()

            expected_source = (
                self.provenance.source.value
            )

            if self.source != expected_source:
                raise ValueError(
                    "Incohérence entre source et provenance : "
                    f"{self.source} != {expected_source}"
                )

    @property
    def is_real_scan_a(self) -> bool:
        return self.source == "scan_a"

    @property
    def is_public_reference(self) -> bool:
        return self.source == "public"

    @property
    def is_simulated(self) -> bool:
        return self.source == "simulated"

    @property
    def image_shape(self) -> tuple[int, int]:
        return (
            int(self.image.shape[0]),
            int(self.image.shape[1]),
        )