from __future__ import annotations

import numpy as np

from src.anomaly.statistical import AnomalyDetectionResult
from src.signature.baseline import StatisticalBaseline
from src.digital_twin.state import DigitalTwinState, build_twin_state


def build_twin_state_from_detection(
    *,
    acquisition_id: str,
    source: str,
    vector: np.ndarray,
    detection: AnomalyDetectionResult,
    baseline: StatisticalBaseline,
    timestamp: str | None = None,
) -> DigitalTwinState:
    """
    Construit un DigitalTwinState à partir d'un résultat de détection.

    Architecture
    ------------
    Digital Signature
          ↓
    StatisticalAnomalyDetector
          ↓
    AnomalyDetectionResult
          ↓
    DigitalTwinState
    """

    x = np.asarray(
        vector,
        dtype=np.float64,
    )

    if x.ndim == 1:
        x_model = x
    elif x.ndim == 2 and x.shape[0] == 1:
        x_model = x
    else:
        raise ValueError(
            "vector doit être un vecteur 1D "
            "ou une matrice contenant une seule observation."
        )

    if not np.all(np.isfinite(x_model)):
        raise ValueError(
            "vector contient NaN ou Inf."
        )

    if x_model.shape[-1] != baseline.n_features:
        raise ValueError(
            "Le nombre de features du vector ne correspond "
            "pas à celui du baseline."
        )

    # ------------------------------------------------------------
    # Dimension scores
    # ------------------------------------------------------------

    raw_dimensions = baseline.health_dimensions(
        x_model
    )

    dimension_scores: dict[str, float] = {}

    for dimension, values in raw_dimensions.items():

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.size == 0:
            raise ValueError(
                f"Dimension vide : {dimension}"
            )

        value = float(
            array.reshape(-1)[0]
        )

        if not np.isfinite(value):
            raise ValueError(
                f"Score non fini pour la dimension : {dimension}"
            )

        dimension_scores[dimension] = value

    # ------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------

    quality_values = baseline.quality_score(
        x_model
    )

    quality_array = np.asarray(
        quality_values,
        dtype=np.float64,
    )

    if quality_array.size == 0:
        raise ValueError(
            "quality_score n'a retourné aucune valeur."
        )

    quality_score = float(
        quality_array.reshape(-1)[0]
    )

    if not np.isfinite(quality_score):
        raise ValueError(
            "quality_score doit être fini."
        )

    # ------------------------------------------------------------
    # Feature contributions
    # ------------------------------------------------------------

    feature_contributions = sorted(
        detection.feature_contributions.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    feature_contributions = [
        (
            str(feature),
            float(contribution),
        )
        for feature, contribution
        in feature_contributions
    ]

    # ------------------------------------------------------------
    # Build state
    # ------------------------------------------------------------

    state = build_twin_state(
        acquisition_id=acquisition_id,
        source=source,
        state=detection.state,
        mahalanobis_distance=detection.distance,
        mahalanobis_squared=detection.d2,
        quality_score=quality_score,
        dimension_scores=dimension_scores,
        feature_contributions=feature_contributions,
        timestamp=timestamp,
    )

    state.validate()

    return state