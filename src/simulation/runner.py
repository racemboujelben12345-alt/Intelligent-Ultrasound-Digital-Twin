"""
SCAN A Digital Twin V2
======================

Simulation Scenario Runner
--------------------------

Rôle
----
Exécuter un scénario expérimental sur une image.

Ce module orchestre :
    scénario → niveaux de sévérité → dégradation

Il ne réalise pas lui-même les opérations mathématiques
de dégradation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .degradation import DegradationResult, apply_degradation
from .scenarios import SimulationScenario


@dataclass(frozen=True)
class SimulationStep:
    """Résultat d'un niveau de simulation."""

    level: int
    severity: float
    result: DegradationResult

    def validate(self) -> None:
        if self.level < 0:
            raise ValueError("level doit être positif ou nul.")

        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity doit être comprise entre 0 et 1.")

        self.result.validate()


def run_scenario(
    image: np.ndarray,
    scenario: SimulationScenario,
    seed: int = 42,
) -> tuple[SimulationStep, ...]:
    """
    Exécute tous les niveaux d'un scénario.

    Parameters
    ----------
    image:
        Image 2D normalisée dans [0, 1].

    scenario:
        Scénario expérimental à exécuter.

    seed:
        Graine de reproductibilité.
    """

    scenario.validate()

    steps: list[SimulationStep] = []

    for level, severity in enumerate(scenario.severities):

        result = apply_degradation(
            image=image,
            degradation_type=scenario.degradation_type,
            severity=severity,
            seed=seed + level,
        )

        step = SimulationStep(
            level=level,
            severity=severity,
            result=result,
        )

        step.validate()
        steps.append(step)

    return tuple(steps)