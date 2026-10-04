"""
Intelligent Ultrasound Digital Twin V2
======================

Controlled Simulation Scenarios
--------------------------------

Rôle
----
Définir des scénarios expérimentaux reproductibles.

Ce module décrit :
- quelle dégradation appliquer ;
- quelles sévérités tester ;
- dans quel ordre effectuer l'expérience.

Il ne réalise pas lui-même la dégradation.
Cette responsabilité appartient à degradation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .degradation import DegradationType


class ScenarioType(str, Enum):
    """Scénarios expérimentaux disponibles."""

    SPECKLE_PROGRESSION = "speckle_progression"
    NOISE_PROGRESSION = "noise_progression"
    BLUR_PROGRESSION = "blur_progression"
    CONTRAST_PROGRESSION = "contrast_progression"
    INTENSITY_PROGRESSION = "intensity_progression"


@dataclass(frozen=True)
class SimulationScenario:
    """
    Décrit une expérience de simulation.

    Un scénario associe un type de dégradation
    à une série de niveaux de sévérité.
    """

    name: str
    degradation_type: DegradationType
    severities: tuple[float, ...]

    def validate(self) -> None:
        """Vérifie la cohérence du scénario."""

        if not self.name:
            raise ValueError("Le nom du scénario ne peut pas être vide.")

        if not self.severities:
            raise ValueError(
                "Un scénario doit contenir au moins une sévérité."
            )

        for severity in self.severities:
            if not 0.0 <= severity <= 1.0:
                raise ValueError(
                    f"Sévérité invalide : {severity}. "
                    "Elle doit être comprise entre 0 et 1."
                )

        if tuple(sorted(self.severities)) != self.severities:
            raise ValueError(
                "Les niveaux de sévérité doivent être "
                "dans un ordre croissant."
            )

    @property
    def number_of_levels(self) -> int:
        """Nombre de niveaux de dégradation."""

        return len(self.severities)


DEFAULT_SEVERITIES = (
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
)


def build_scenario(
    scenario_type: ScenarioType | str,
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
) -> SimulationScenario:
    """
    Construit un scénario expérimental.

    Parameters
    ----------
    scenario_type:
        Type d'expérience.

    severities:
        Niveaux de dégradation à tester.
    """

    if isinstance(scenario_type, ScenarioType):
        scenario = scenario_type
    else:
        try:
            scenario = ScenarioType(str(scenario_type))
        except ValueError as exc:
            valid = [item.value for item in ScenarioType]

            raise ValueError(
                f"Scénario inconnu : {scenario_type}. "
                f"Scénarios disponibles : {valid}"
            ) from exc

    mapping = {
        ScenarioType.SPECKLE_PROGRESSION: (
            "Speckle progression",
            DegradationType.SPECKLE,
        ),
        ScenarioType.NOISE_PROGRESSION: (
            "Gaussian noise progression",
            DegradationType.GAUSSIAN_NOISE,
        ),
        ScenarioType.BLUR_PROGRESSION: (
            "Blur progression",
            DegradationType.BLUR,
        ),
        ScenarioType.CONTRAST_PROGRESSION: (
            "Contrast reduction progression",
            DegradationType.CONTRAST_REDUCTION,
        ),
        ScenarioType.INTENSITY_PROGRESSION: (
            "Intensity shift progression",
            DegradationType.INTENSITY_SHIFT,
        ),
    }

    name, degradation_type = mapping[scenario]

    result = SimulationScenario(
        name=name,
        degradation_type=degradation_type,
        severities=tuple(float(value) for value in severities),
    )

    result.validate()

    return result
