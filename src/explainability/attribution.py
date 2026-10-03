from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.anomaly.statistical import AnomalyDetectionResult


@dataclass(frozen=True)
class ExplanationItem:
    """
    Élément explicatif d'une déviation.
    """

    feature: str
    contribution: float
    rank: int

    def validate(self) -> None:
        if not self.feature:
            raise ValueError(
                "feature ne peut pas être vide."
            )

        if not np.isfinite(self.contribution):
            raise ValueError(
                "contribution doit être finie."
            )

        if self.contribution < 0.0:
            raise ValueError(
                "contribution ne peut pas être négative."
            )

        if self.rank < 1:
            raise ValueError(
                "rank doit être >= 1."
            )


@dataclass(frozen=True)
class Explanation:
    """
    Explication structurée d'un résultat de détection.
    """

    state: str
    top_features: tuple[ExplanationItem, ...]
    dominant_feature: str | None
    dominant_contribution: float

    def validate(self) -> None:
        if not self.state:
            raise ValueError(
                "state ne peut pas être vide."
            )

        if self.dominant_feature is not None:
            if not self.dominant_feature:
                raise ValueError(
                    "dominant_feature invalide."
                )

        if not np.isfinite(
            self.dominant_contribution
        ):
            raise ValueError(
                "dominant_contribution doit être finie."
            )


def explain_detection(
    detection: AnomalyDetectionResult,
    *,
    top_k: int = 5,
) -> Explanation:
    """
    Transforme les contributions statistiques en explication
    structurée et lisible.

    Important
    ---------
    Les contributions sont descriptives.
    Elles ne démontrent pas la cause physique de l'anomalie.
    """

    detection.validate()

    if top_k < 1:
        raise ValueError(
            "top_k doit être >= 1."
        )

    ranked = sorted(
        detection.feature_contributions.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    selected = ranked[:top_k]

    items = tuple(
        ExplanationItem(
            feature=feature,
            contribution=float(contribution),
            rank=index,
        )
        for index, (feature, contribution)
        in enumerate(selected, start=1)
    )

    for item in items:
        item.validate()

    if items:
        dominant_feature = items[0].feature
        dominant_contribution = (
            items[0].contribution
        )
    else:
        dominant_feature = None
        dominant_contribution = 0.0

    result = Explanation(
        state=detection.state,
        top_features=items,
        dominant_feature=dominant_feature,
        dominant_contribution=dominant_contribution,
    )

    result.validate()

    return result


def explanation_text(
    explanation: Explanation,
) -> str:
    """
    Produit une représentation textuelle courte
    destinée au reporting ou au dashboard.
    """

    explanation.validate()

    if not explanation.top_features:
        return (
            f"State={explanation.state} | "
            "Aucune contribution disponible."
        )

    lines = [
        f"State={explanation.state}",
        "Principales contributions statistiques :",
    ]

    for item in explanation.top_features:
        lines.append(
            f"- {item.feature}: "
            f"{item.contribution:.2%}"
        )

    return "\n".join(lines)