"""
SCAN A Digital Twin V2
======================

Digital Twin History
--------------------

Rôle
----
Conserver l'évolution temporelle des états du Digital Twin.

Responsabilités
---------------
- stockage chronologique des états ;
- validation des états enregistrés ;
- accès aux séries temporelles ;
- statistiques simples sur l'historique.

Ce module ne :
- calcule pas les features ;
- ne construit pas le baseline ;
- ne détecte pas directement les anomalies ;
- ne simule pas de dégradation ;
- ne réalise pas le monitoring EWMA/CUSUM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .state import DigitalTwinState


@dataclass
class DigitalTwinHistory:
    """
    Historique chronologique des états du Digital Twin.
    """

    states: list[DigitalTwinState] = field(
        default_factory=list
    )

    # ========================================================
    # AJOUT
    # ========================================================

    def add(
        self,
        state: DigitalTwinState,
    ) -> None:
        """
        Ajoute un nouvel état après validation.
        """

        state.validate()
        self.states.append(state)

    def extend(
        self,
        states: Iterable[DigitalTwinState],
    ) -> None:
        """
        Ajoute plusieurs états dans l'ordre fourni.
        """

        for state in states:
            self.add(state)

    # ========================================================
    # TAILLE / ACCÈS
    # ========================================================

    def __len__(self) -> int:
        return len(self.states)

    @property
    def latest(
        self,
    ) -> DigitalTwinState | None:
        """
        Retourne le dernier état enregistré.
        """

        if not self.states:
            return None

        return self.states[-1]

    def get_states(
        self,
    ) -> tuple[DigitalTwinState, ...]:
        """
        Retourne les états sous forme immuable.
        """

        return tuple(self.states)

    # ========================================================
    # SÉRIES TEMPORELLES
    # ========================================================

    @property
    def mahalanobis_squared_series(
        self,
    ) -> tuple[float, ...]:
        """
        Retourne la série temporelle de D².

        D² constitue le signal principal utilisé
        ultérieurement par le monitoring temporel.
        """

        return tuple(
            float(
                state.mahalanobis_squared
            )
            for state in self.states
        )

    @property
    def mahalanobis_distance_series(
        self,
    ) -> tuple[float, ...]:
        """
        Retourne la série temporelle de D.
        """

        return tuple(
            float(
                state.mahalanobis_distance
            )
            for state in self.states
        )

    @property
    def quality_score_series(
        self,
    ) -> tuple[float, ...]:
        """
        Retourne la série temporelle du score
        de conformité statistique.
        """

        return tuple(
            float(
                state.quality_score
            )
            for state in self.states
        )

    @property
    def state_series(
        self,
    ) -> tuple[str, ...]:
        """
        Retourne la séquence temporelle des états.
        """

        return tuple(
            state.state
            for state in self.states
        )

    @property
    def timestamp_series(
        self,
    ) -> tuple[str, ...]:
        """
        Retourne la séquence temporelle des timestamps.
        """

        return tuple(
            state.timestamp
            for state in self.states
        )

    # ========================================================
    # STATISTIQUES SIMPLES
    # ========================================================

    def count_by_state(
        self,
    ) -> dict[str, int]:
        """
        Compte le nombre d'occurrences de chaque état.
        """

        counts: dict[str, int] = {}

        for state in self.states:
            counts[state.state] = (
                counts.get(
                    state.state,
                    0,
                )
                + 1
            )

        return counts

    def sources(
        self,
    ) -> set[str]:
        """
        Retourne les différentes sources présentes
        dans l'historique.
        """

        return {
            state.source
            for state in self.states
        }

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
    ) -> None:
        """
        Valide l'ensemble de l'historique.
        """

        for state in self.states:
            state.validate()

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    def summary(
        self,
    ) -> str:
        """
        Résumé lisible de l'historique.
        """

        latest_state = self.latest

        if latest_state is None:
            return (
                "Digital Twin History | "
                "0 état"
            )

        return (
            f"Digital Twin History | "
            f"{len(self.states)} états | "
            f"Dernier={latest_state.state} | "
            f"Acquisition="
            f"{latest_state.acquisition_id}"
        )