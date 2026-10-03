from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VALID_STATES = {
    "NOMINAL",
    "EARLY_DRIFT",
    "SIGNIFICANT_DRIFT",
    "HIGH_DEVIATION",
}


@dataclass(frozen=True)
class DetectionMetrics:
    """
    Métriques quantitatives de validation du détecteur d'anomalies.

    La vérité terrain est définie explicitement par expected_anomaly.
    """

    n_samples: int

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    nominal_count: int
    early_drift_count: int
    significant_drift_count: int
    high_deviation_count: int

    mean_distance: float
    std_distance: float

    @property
    def nominal_rate(self) -> float:
        return self.nominal_count / self.n_samples

    @property
    def early_drift_rate(self) -> float:
        return self.early_drift_count / self.n_samples

    @property
    def significant_drift_rate(self) -> float:
        return self.significant_drift_count / self.n_samples

    @property
    def high_deviation_rate(self) -> float:
        return self.high_deviation_count / self.n_samples

    @property
    def detection_rate(self) -> float:
        """
        Taux de vrais positifs parmi les observations réellement anormales.
        """

        denominator = (
            self.true_positive
            + self.false_negative
        )

        if denominator == 0:
            return 0.0

        return (
            self.true_positive
            / denominator
        )

    @property
    def false_positive_rate(self) -> float:
        """
        Taux de faux positifs parmi les observations normales.
        """

        denominator = (
            self.false_positive
            + self.true_negative
        )

        if denominator == 0:
            return 0.0

        return (
            self.false_positive
            / denominator
        )

    @property
    def specificity(self) -> float:
        """
        Spécificité du détecteur.
        """

        return 1.0 - self.false_positive_rate

    @property
    def sensitivity(self) -> float:
        """
        Sensibilité = true positive rate.
        """

        return self.detection_rate

    def to_dict(self) -> dict[str, float | int]:
        """
        Conversion vers une structure sérialisable.
        """

        return {
            "n_samples": self.n_samples,

            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,

            "nominal_count": self.nominal_count,
            "early_drift_count": self.early_drift_count,
            "significant_drift_count": (
                self.significant_drift_count
            ),
            "high_deviation_count": (
                self.high_deviation_count
            ),

            "nominal_rate": self.nominal_rate,
            "early_drift_rate": self.early_drift_rate,
            "significant_drift_rate": (
                self.significant_drift_rate
            ),
            "high_deviation_rate": (
                self.high_deviation_rate
            ),

            "mean_distance": self.mean_distance,
            "std_distance": self.std_distance,

            "detection_rate": self.detection_rate,
            "sensitivity": self.sensitivity,
            "false_positive_rate": (
                self.false_positive_rate
            ),
            "specificity": self.specificity,
        }


def compute_detection_metrics(
    distances: np.ndarray | list[float],
    states: list[str] | tuple[str, ...],
    expected_anomaly: bool | list[bool] | tuple[bool, ...],
) -> DetectionMetrics:
    """
    Calcule les métriques quantitatives du détecteur.

    Parameters
    ----------
    distances:
        Distances de Mahalanobis.

    states:
        États produits par le détecteur.

    expected_anomaly:
        Vérité terrain :
            False -> observation normale attendue
            True  -> anomalie/dégradation attendue
    """

    distances_array = np.asarray(
        distances,
        dtype=np.float64,
    )

    if distances_array.ndim != 1:
        raise ValueError(
            "distances doit être un vecteur 1D."
        )

    if len(distances_array) == 0:
        raise ValueError(
            "Au moins une observation est requise."
        )

    if not np.all(
        np.isfinite(distances_array)
    ):
        raise ValueError(
            "distances contient NaN ou Inf."
        )

    states_list = list(states)

    if len(states_list) != len(
        distances_array
    ):
        raise ValueError(
            "distances et states doivent "
            "avoir la même longueur."
        )

    unknown_states = (
        set(states_list) - VALID_STATES
    )

    if unknown_states:
        raise ValueError(
            f"États inconnus : "
            f"{sorted(unknown_states)}"
        )

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    if isinstance(
        expected_anomaly,
        bool,
    ):
        ground_truth = np.full(
            len(states_list),
            expected_anomaly,
            dtype=bool,
        )
    else:
        ground_truth = np.asarray(
            expected_anomaly,
            dtype=bool,
        )

        if ground_truth.ndim != 1:
            raise ValueError(
                "expected_anomaly doit être "
                "un vecteur 1D."
            )

        if len(ground_truth) != len(
            states_list
        ):
            raise ValueError(
                "expected_anomaly doit avoir "
                "la même longueur que states."
            )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    predicted_anomaly = np.asarray(
        [
            state != "NOMINAL"
            for state in states_list
        ],
        dtype=bool,
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    true_positive = int(
        np.sum(
            ground_truth
            & predicted_anomaly
        )
    )

    false_positive = int(
        np.sum(
            ~ground_truth
            & predicted_anomaly
        )
    )

    true_negative = int(
        np.sum(
            ~ground_truth
            & ~predicted_anomaly
        )
    )

    false_negative = int(
        np.sum(
            ground_truth
            & ~predicted_anomaly
        )
    )

    # ------------------------------------------------------------
    # State distribution
    # ------------------------------------------------------------

    nominal_count = states_list.count(
        "NOMINAL"
    )

    early_drift_count = states_list.count(
        "EARLY_DRIFT"
    )

    significant_drift_count = (
        states_list.count(
            "SIGNIFICANT_DRIFT"
        )
    )

    high_deviation_count = (
        states_list.count(
            "HIGH_DEVIATION"
        )
    )

    # ------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------

    result = DetectionMetrics(
        n_samples=len(states_list),

        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,

        nominal_count=nominal_count,
        early_drift_count=early_drift_count,
        significant_drift_count=(
            significant_drift_count
        ),
        high_deviation_count=(
            high_deviation_count
        ),

        mean_distance=float(
            np.mean(distances_array)
        ),
        std_distance=float(
            np.std(distances_array)
        ),
    )

    return result