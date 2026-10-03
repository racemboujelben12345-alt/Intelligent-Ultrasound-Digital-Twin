"""
SCAN A Digital Twin V2
======================

Temporal Trend Analysis
-----------------------

Rôle
----
Estimer la tendance d'un signal temporel du Digital Twin.

Ce module permet d'analyser :
- D² de Mahalanobis ;
- distance de Mahalanobis ;
- score de conformité ;
- toute autre série numérique temporelle.

Important
---------
Ce module réalise une analyse de tendance.

Il ne prédit PAS :
- une panne physique ;
- une date de panne ;
- une défaillance clinique ;
- une durée de vie réelle de l'équipement.

Les résultats doivent être interprétés comme des indicateurs
statistiques de trajectoire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VALID_DIRECTIONS = {
    "DECREASING",
    "STABLE",
    "INCREASING",
}


@dataclass(frozen=True)
class TrendResult:
    """
    Résultat d'une estimation de tendance linéaire.
    """

    n_samples: int

    slope: float
    intercept: float
    r_squared: float

    direction: str
    strength: float

    first_value: float
    last_value: float

    predicted_next: float

    def validate(self) -> None:

        numeric_values = (
            self.slope,
            self.intercept,
            self.r_squared,
            self.strength,
            self.first_value,
            self.last_value,
            self.predicted_next,
        )

        if not all(
            np.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "Le résultat de tendance contient "
                "des valeurs non finies."
            )

        if self.n_samples < 2:
            raise ValueError(
                "Au moins deux observations sont nécessaires."
            )

        if not 0.0 <= self.r_squared <= 1.0:
            raise ValueError(
                "R² doit être compris entre 0 et 1."
            )

        if self.strength < 0.0:
            raise ValueError(
                "La force de tendance doit être positive ou nulle."
            )

        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"Direction inconnue : {self.direction}"
            )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "n_samples": self.n_samples,
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "direction": self.direction,
            "strength": self.strength,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "predicted_next": self.predicted_next,
        }


class TrendAnalyzer:
    """
    Analyseur de tendance temporelle.

    La tendance est estimée par régression linéaire
    sur l'index temporel des observations.
    """

    def __init__(
        self,
        stable_slope_threshold: float = 0.0,
    ) -> None:

        if not np.isfinite(
            stable_slope_threshold
        ):
            raise ValueError(
                "stable_slope_threshold doit être fini."
            )

        if stable_slope_threshold < 0.0:
            raise ValueError(
                "stable_slope_threshold doit être positif ou nul."
            )

        self.stable_slope_threshold = float(
            stable_slope_threshold
        )

    def analyze(
        self,
        values: list[float]
        | tuple[float, ...]
        | np.ndarray,
    ) -> TrendResult:
        """
        Estime la tendance d'une série temporelle.

        Parameters
        ----------
        values:
            Série chronologique ordonnée.
        """

        y = np.asarray(
            values,
            dtype=np.float64,
        )

        if y.ndim != 1:
            raise ValueError(
                "values doit être un vecteur 1D."
            )

        if len(y) < 2:
            raise ValueError(
                "Au moins deux observations sont nécessaires."
            )

        if not np.all(
            np.isfinite(y)
        ):
            raise ValueError(
                "values contient NaN ou Inf."
            )

        x = np.arange(
            len(y),
            dtype=np.float64,
        )

        # --------------------------------------------------------
        # Régression linéaire
        # --------------------------------------------------------

        coefficients = np.polyfit(
            x,
            y,
            deg=1,
        )

        slope = float(
            coefficients[0]
        )

        intercept = float(
            coefficients[1]
        )

        y_hat = (
            slope * x
            + intercept
        )

        residuals = (
            y - y_hat
        )

        ss_res = float(
            np.sum(
                residuals ** 2
            )
        )

        y_mean = float(
            np.mean(y)
        )

        ss_tot = float(
            np.sum(
                (y - y_mean) ** 2
            )
        )

        if ss_tot <= 1e-15:
            r_squared = 1.0
        else:
            r_squared = max(
                0.0,
                min(
                    1.0,
                    1.0 - ss_res / ss_tot,
                ),
            )

        # --------------------------------------------------------
        # Direction
        # --------------------------------------------------------

        threshold = (
            self.stable_slope_threshold
        )

        if slope > threshold:
            direction = "INCREASING"
        elif slope < -threshold:
            direction = "DECREASING"
        else:
            direction = "STABLE"

        # --------------------------------------------------------
        # Strength
        # --------------------------------------------------------

        strength = float(
            abs(slope)
            * np.sqrt(r_squared)
        )

        # --------------------------------------------------------
        # Next-step trajectory
        # --------------------------------------------------------

        next_index = float(
            len(y)
        )

        predicted_next = float(
            slope * next_index
            + intercept
        )

        result = TrendResult(
            n_samples=len(y),
            slope=slope,
            intercept=intercept,
            r_squared=float(
                r_squared
            ),
            direction=direction,
            strength=strength,
            first_value=float(
                y[0]
            ),
            last_value=float(
                y[-1]
            ),
            predicted_next=predicted_next,
        )

        result.validate()

        return result