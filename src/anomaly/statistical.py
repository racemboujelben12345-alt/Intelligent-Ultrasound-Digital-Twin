from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.signature.baseline import StatisticalBaseline


@dataclass(frozen=True)
class AnomalyDetectionResult:
    """
    Résultat structuré de détection statistique.

    d2 :
        Distance de Mahalanobis au carré.

    distance :
        Distance de Mahalanobis.

    state :
        État opérationnel retourné par le baseline.

    z_scores :
        Contributions standardisées de chaque feature.

    feature_contributions :
        Contribution relative descriptive de chaque feature.
    """

    d2: float
    distance: float
    state: str
    z_scores: np.ndarray
    feature_contributions: dict[str, float]

    def validate(self) -> None:
        if not np.isfinite(self.d2):
            raise ValueError(
                "d2 doit être fini."
            )

        if self.d2 < 0.0:
            raise ValueError(
                "d2 ne peut pas être négatif."
            )

        if not np.isfinite(self.distance):
            raise ValueError(
                "distance doit être finie."
            )

        if self.distance < 0.0:
            raise ValueError(
                "distance ne peut pas être négative."
            )

        if not self.state:
            raise ValueError(
                "state ne peut pas être vide."
            )

        if self.z_scores.ndim != 1:
            raise ValueError(
                "z_scores doit être un vecteur 1D."
            )


class StatisticalAnomalyDetector:
    """
    Détecteur d'anomalies basé sur une baseline statistique.

    Architecture
    ------------
    Digital Signature
            ↓
    Statistical Baseline
            ↓
    Mahalanobis
            ↓
    Health State
            ↓
    Explainability

    Le détecteur ne construit pas la baseline.
    """

    def __init__(
        self,
        baseline: StatisticalBaseline,
    ) -> None:

        if not isinstance(
            baseline,
            StatisticalBaseline,
        ):
            raise TypeError(
                "baseline doit être une instance "
                "de StatisticalBaseline."
            )

        self.baseline = baseline

    def detect(
        self,
        vector: np.ndarray,
    ) -> AnomalyDetectionResult:
        """
        Détecte une anomalie à partir d'une Digital Signature.

        Parameters
        ----------
        vector:
            Vecteur de Digital Signature.
        """

        x = np.asarray(
            vector,
            dtype=np.float64,
        )

        if x.ndim == 1:
            x_for_model = x
        elif x.ndim == 2 and x.shape[0] == 1:
            x_for_model = x
        else:
            raise ValueError(
                "vector doit être un vecteur 1D "
                "ou une matrice contenant une observation."
            )

        if not np.all(
            np.isfinite(x_for_model)
        ):
            raise ValueError(
                "vector contient NaN ou Inf."
            )

        d2_array = (
            self.baseline
            .mahalanobis_squared(
                x_for_model
            )
        )

        d2 = float(
            d2_array[0]
        )

        distance = float(
            np.sqrt(
                max(d2, 0.0)
            )
        )

        state = str(
            self.baseline.classify_distance(
                d2
            )
        )

        z_scores = np.asarray(
            self.baseline.z_score(
                x_for_model
            )[0],
            dtype=np.float64,
        )

        feature_contributions = (
            self._compute_feature_contributions(
                x_for_model
            )
        )

        result = AnomalyDetectionResult(
            d2=d2,
            distance=distance,
            state=state,
            z_scores=z_scores,
            feature_contributions=(
                feature_contributions
            ),
        )

        result.validate()

        return result

    def _compute_feature_contributions(
        self,
        vector: np.ndarray,
    ) -> dict[str, float]:
        """
        Calcule une contribution descriptive basée sur z².

        Attention :
        il s'agit d'une attribution statistique descriptive,
        pas d'une preuve causale.
        """

        x = np.asarray(
            vector,
            dtype=np.float64,
        )

        if x.ndim == 1:
            x = x.reshape(1, -1)

        z = self.baseline.z_score(x)[0]

        squared = np.square(z)

        total = float(
            np.sum(squared)
        )

        if total <= 1e-12:
            contributions = np.zeros_like(
                squared
            )
        else:
            contributions = (
                squared / total
            )

        return {
            feature_name: float(value)
            for feature_name, value in zip(
                self.baseline.feature_names,
                contributions,
            )
        }

    def is_anomaly(
        self,
        vector: np.ndarray,
    ) -> bool:
        """
        Retourne True pour tout état non-NOMINAL.
        """

        result = self.detect(
            vector
        )

        return result.state != "NOMINAL"