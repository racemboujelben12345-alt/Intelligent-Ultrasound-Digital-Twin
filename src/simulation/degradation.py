"""
SCAN A Digital Twin V2
======================

Controlled Image Degradation Engine
------------------------------------

Rôle
----
Générer des dégradations numériques contrôlées sur des images
ultrasonores afin de tester la sensibilité du Digital Twin.

Important
---------
Ces dégradations sont purement numériques.
Elles ne représentent pas automatiquement une panne physique
réelle du SCAN A.

Le module ne :
- détecte pas les anomalies ;
- ne calcule pas le baseline ;
- ne construit pas la signature numérique ;
- ne modifie jamais les données originales.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.ndimage import gaussian_filter


class DegradationType(str, Enum):
    """Types de dégradations numériques supportées."""

    SPECKLE = "speckle"
    GAUSSIAN_NOISE = "gaussian_noise"
    BLUR = "blur"
    CONTRAST_REDUCTION = "contrast_reduction"
    INTENSITY_SHIFT = "intensity_shift"


@dataclass(frozen=True)
class DegradationResult:
    """
    Résultat d'une simulation de dégradation.
    """

    image: np.ndarray
    degradation_type: str
    severity: float
    seed: int | None = None
    simulation_version: str = "1.0"

    def validate(self) -> None:
        """Vérifie la cohérence du résultat."""

        if not isinstance(self.image, np.ndarray):
            raise TypeError("image doit être un numpy.ndarray.")

        if self.image.ndim != 2:
            raise ValueError("L'image doit être une matrice 2D.")

        if not np.all(np.isfinite(self.image)):
            raise ValueError("L'image contient des valeurs non finies.")

        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(
                "La sévérité doit être comprise entre 0 et 1."
            )

        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("seed doit être un entier ou None.")

        if not self.simulation_version:
            raise ValueError("simulation_version ne peut pas être vide.")

        if not self.degradation_type:
            raise ValueError(
                "Le type de dégradation ne peut pas être vide."
            )


def _validate_image(image: np.ndarray) -> np.ndarray:
    """
    Valide et convertit une image en float32 dans [0, 1].
    """

    array = np.asarray(image, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError("L'image doit être une matrice 2D.")

    if not np.all(np.isfinite(array)):
        raise ValueError("L'image contient des valeurs non finies.")

    minimum = float(array.min())
    maximum = float(array.max())

    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            "L'image doit être normalisée dans l'intervalle [0, 1]."
        )

    return array


def _clip(image: np.ndarray) -> np.ndarray:
    """Limite les intensités dans [0, 1]."""

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def add_speckle(
    image: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ajoute un bruit multiplicatif de type speckle.

    severity :
        0 = aucune dégradation
        1 = dégradation maximale définie par le modèle
    """

    image = _validate_image(image)

    sigma = 0.05 * float(severity)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    ).astype(np.float32)

    degraded = image + image * noise

    return _clip(degraded)


def add_gaussian_noise(
    image: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ajoute un bruit additif gaussien.
    """

    image = _validate_image(image)

    sigma = 0.08 * float(severity)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    ).astype(np.float32)

    degraded = image + noise

    return _clip(degraded)


def add_blur(
    image: np.ndarray,
    severity: float,
) -> np.ndarray:
    """
    Applique un flou gaussien contrôlé.
    """

    image = _validate_image(image)

    sigma = 0.15 + 2.5 * float(severity)

    degraded = gaussian_filter(
        image,
        sigma=sigma,
    )

    return _clip(degraded)


def reduce_contrast(
    image: np.ndarray,
    severity: float,
) -> np.ndarray:
    """
    Réduit progressivement le contraste autour de la moyenne.
    """

    image = _validate_image(image)

    factor = 1.0 - 0.75 * float(severity)

    mean = float(np.mean(image))

    degraded = mean + factor * (image - mean)

    return _clip(degraded)


def shift_intensity(
    image: np.ndarray,
    severity: float,
) -> np.ndarray:
    """
    Introduit un décalage global contrôlé de l'intensité.
    """

    image = _validate_image(image)

    shift = 0.20 * float(severity)

    degraded = image + shift

    return _clip(degraded)


def apply_degradation(
    image: np.ndarray,
    degradation_type: DegradationType | str,
    severity: float,
    seed: int | None = None,
) -> DegradationResult:
    """
    Applique une dégradation numérique contrôlée.

    Parameters
    ----------
    image:
        Image 2D normalisée dans [0, 1].

    degradation_type:
        Type de dégradation.

    severity:
        Intensité normalisée dans [0, 1].

    seed:
        Graine optionnelle pour rendre la simulation reproductible.
    """

    image = _validate_image(image)

    severity = float(severity)

    if not 0.0 <= severity <= 1.0:
        raise ValueError(
            "severity doit être comprise entre 0 et 1."
        )

    if isinstance(degradation_type, DegradationType):
        degradation = degradation_type
    else:
        try:
            degradation = DegradationType(str(degradation_type))
        except ValueError as exc:
            valid = [item.value for item in DegradationType]

            raise ValueError(
                f"Type de dégradation inconnu : {degradation_type}. "
                f"Types disponibles : {valid}"
            ) from exc

    rng = np.random.default_rng(seed)

    if degradation == DegradationType.SPECKLE:
        degraded = add_speckle(
            image,
            severity,
            rng,
        )

    elif degradation == DegradationType.GAUSSIAN_NOISE:
        degraded = add_gaussian_noise(
            image,
            severity,
            rng,
        )

    elif degradation == DegradationType.BLUR:
        degraded = add_blur(
            image,
            severity,
        )

    elif degradation == DegradationType.CONTRAST_REDUCTION:
        degraded = reduce_contrast(
            image,
            severity,
        )

    elif degradation == DegradationType.INTENSITY_SHIFT:
        degraded = shift_intensity(
            image,
            severity,
        )

    else:
        raise RuntimeError(
            f"Dégradation non implémentée : {degradation}"
        )

    result = DegradationResult(
        image=degraded,
        degradation_type=degradation.value,
        severity=severity,
        seed=seed,
        simulation_version="1.0",
    )

    result.validate()

    return result
