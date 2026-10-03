"""
SCAN A Digital Twin V2
========================================

Statistical Baseline Model.

Rôle
----
Construire une représentation statistique du comportement
normal d'un ensemble d'acquisitions de référence.

La baseline constitue la référence contre laquelle une
nouvelle acquisition peut être comparée.

Méthodes principales
--------------------
- centrage / réduction
- covariance régularisée Ledoit-Wolf
- distance de Mahalanobis
- seuils statistiques basés sur Chi²
- contribution des caractéristiques
- score de conformité relatif
- dimensions de surveillance
- état de santé

Important
---------
Cette classe ne sait PAS si les données proviennent de SCAN A,
d'un dataset public ou d'une simulation.

La provenance est gérée par la couche acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf


# ============================================================
# CONFIGURATION DES CARACTÉRISTIQUES
# ============================================================

DEFAULT_FEATURES = (
    "mean_intensity",
    "std_intensity",
    "dynamic_range",
    "coefficient_variation",
    "rms_contrast",
    "entropy",
    "sharpness_laplacian",
    "speckle_proxy",
    "edge_density",
    "gradient_mean",
    "gradient_std",
    "gradient_max",
    "uniformity",
)


# ============================================================
# DIMENSIONS DE SURVEILLANCE
# ============================================================

HEALTH_DIMENSIONS = {
    "contrast": (
        "std_intensity",
        "rms_contrast",
        "dynamic_range",
    ),
    "noise_texture": (
        "speckle_proxy",
        "entropy",
    ),
    "sharpness": (
        "sharpness_laplacian",
        "gradient_mean",
        "gradient_std",
    ),
    "structure": (
        "edge_density",
        "gradient_mean",
        "gradient_max",
    ),
    "uniformity": (
        "coefficient_variation",
        "uniformity",
    ),
}


# ============================================================
# SEUILS STATISTIQUES
# ============================================================

DEFAULT_ALPHAS = {
    "early": 0.95,
    "significant": 0.99,
    "critical": 0.999,
}


# ============================================================
# UTILITAIRE : CONVERSION EN MATRICE
# ============================================================

def _as_matrix(
    X: Iterable,
    *,
    min_observations: int = 1,
) -> np.ndarray:
    """
    Convertit une entrée en matrice numérique 2D.

    Parameters
    ----------
    X:
        Données numériques.

    min_observations:
        Nombre minimal d'observations accepté.

        1 :
            utilisé lorsqu'on analyse une nouvelle acquisition.

        2 ou plus :
            utilisé lorsqu'on construit une baseline.

    Returns
    -------
    np.ndarray
        Matrice de forme (n_observations, n_features).
    """

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError(
            "Les données doivent être une matrice 2D."
        )

    if X.shape[0] < min_observations:
        raise ValueError(
            f"Au moins {min_observations} observation(s) "
            "sont nécessaires."
        )

    if X.shape[1] < 2:
        raise ValueError(
            "Au moins deux caractéristiques sont nécessaires."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "Les données contiennent des valeurs non finies."
        )

    return X


# ============================================================
# RÉSUMÉ DE LA BASELINE
# ============================================================

@dataclass
class BaselineStatistics:
    """
    Résumé statistique explicite de la baseline.
    """

    n_samples: int
    n_features: int
    feature_names: tuple[str, ...]

    mean: np.ndarray
    std: np.ndarray

    covariance: np.ndarray
    precision: np.ndarray

    shrinkage: float

    thresholds: dict[str, float]


# ============================================================
# MODÈLE STATISTIQUE
# ============================================================

class StatisticalBaseline:
    """
    Modèle statistique de référence du Digital Twin.

    Pipeline
    --------
        acquisitions normales
                 ↓
              features
                 ↓
          μ + σ + covariance
                 ↓
        Ledoit-Wolf shrinkage
                 ↓
       distance Mahalanobis²
                 ↓
       seuils statistiques
                 ↓
          état de santé
    """

    def __init__(
        self,
        X_reference: Iterable,
        feature_names: Iterable[str] | None = None,
        alphas: dict[str, float] | None = None,
    ) -> None:

        # ----------------------------------------------------
        # Une baseline nécessite plusieurs acquisitions.
        # ----------------------------------------------------

        X = _as_matrix(
            X_reference,
            min_observations=2,
        )

        # ----------------------------------------------------
        # Noms des caractéristiques
        # ----------------------------------------------------

        if feature_names is None:
            feature_names = tuple(
                f"feature_{i}"
                for i in range(
                    X.shape[1]
                )
            )
        else:
            feature_names = tuple(
                str(name)
                for name in feature_names
            )

        if len(feature_names) != X.shape[1]:
            raise ValueError(
                "Le nombre de noms de features doit "
                "correspondre au nombre de colonnes."
            )

        if len(set(feature_names)) != len(
            feature_names
        ):
            raise ValueError(
                "Les noms de features doivent être uniques."
            )

        # ----------------------------------------------------
        # Validation des seuils
        # ----------------------------------------------------

        if alphas is None:
            alphas = dict(
                DEFAULT_ALPHAS
            )
        else:
            alphas = dict(alphas)

        self._validate_alphas(
            alphas
        )

        # ----------------------------------------------------
        # Données de référence
        # ----------------------------------------------------

        self.X_reference = X.copy()

        self.feature_names = feature_names

        self.n_samples = X.shape[0]

        self.n_features = X.shape[1]

        # ----------------------------------------------------
        # Moyenne μ
        # ----------------------------------------------------

        self.mu = np.mean(
            X,
            axis=0,
        )

        # ----------------------------------------------------
        # Écart-type σ
        # ----------------------------------------------------

        self.sigma = (
            np.std(
                X,
                axis=0,
                ddof=1,
            )
            + 1e-12
        )

        # ----------------------------------------------------
        # Normalisation
        # ----------------------------------------------------

        Z = (
            X - self.mu
        ) / self.sigma

        # ----------------------------------------------------
        # Covariance régularisée
        # ----------------------------------------------------

        self.ledoit_wolf = (
            LedoitWolf().fit(Z)
        )

        self.covariance = (
            self.ledoit_wolf.covariance_
        )

        self.precision = (
            self.ledoit_wolf.precision_
        )

        self.shrinkage = float(
            self.ledoit_wolf.shrinkage_
        )

        # ----------------------------------------------------
        # Seuils statistiques
        # ----------------------------------------------------

        self.alphas = alphas

        self.thresholds = {
            name: float(
                chi2.ppf(
                    alpha,
                    self.n_features,
                )
            )
            for name, alpha in alphas.items()
        }

        # ----------------------------------------------------
        # Référence du score qualité
        # ----------------------------------------------------

        self._build_quality_reference()

    # ========================================================
    # VALIDATION DES ALPHAS
    # ========================================================

    @staticmethod
    def _validate_alphas(
        alphas: dict[str, float],
    ) -> None:

        required = {
            "early",
            "significant",
            "critical",
        }

        missing = (
            required
            - set(alphas.keys())
        )

        if missing:
            raise ValueError(
                "Seuils manquants : "
                f"{sorted(missing)}"
            )

        for name, alpha in alphas.items():

            if not (
                0.0 < alpha < 1.0
            ):
                raise ValueError(
                    f"Alpha invalide pour "
                    f"'{name}' : {alpha}"
                )

        if not (
            alphas["early"]
            < alphas["significant"]
            < alphas["critical"]
        ):
            raise ValueError(
                "Les seuils doivent respecter : "
                "early < significant < critical."
            )

    # ========================================================
    # Z-SCORE
    # ========================================================

    def z_score(
        self,
        X: Iterable,
    ) -> np.ndarray:
        """
        Calcule les z-scores par rapport à la baseline.

        Une ou plusieurs observations sont acceptées.
        """

        X = _as_matrix(
            X,
            min_observations=1,
        )

        if X.shape[1] != self.n_features:
            raise ValueError(
                "Nombre de features incompatible "
                "avec la baseline."
            )

        return (
            X - self.mu
        ) / self.sigma

    # ========================================================
    # DISTANCE DE MAHALANOBIS AU CARRÉ
    # ========================================================

    def mahalanobis_squared(
        self,
        X: Iterable,
    ) -> np.ndarray:
        """
        Calcule la distance de Mahalanobis au carré.

        Elle prend en compte les corrélations entre
        caractéristiques.
        """

        Z = self.z_score(
            X
        )

        d2 = np.einsum(
            "ij,jk,ik->i",
            Z,
            self.precision,
            Z,
        )

        return np.maximum(
            d2,
            0.0,
        )

    # ========================================================
    # DISTANCE DE MAHALANOBIS
    # ========================================================

    def mahalanobis(
        self,
        X: Iterable,
    ) -> np.ndarray:
        """
        Calcule la distance de Mahalanobis.
        """

        d2 = self.mahalanobis_squared(
            X
        )

        return np.sqrt(
            np.maximum(
                d2,
                0.0,
            )
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def classify(
        self,
        X: Iterable,
    ) -> np.ndarray:
        """
        Classe une ou plusieurs acquisitions.

        États
        -----
        NOMINAL
        EARLY_DRIFT
        SIGNIFICANT_DRIFT
        HIGH_DEVIATION
        """

        d2 = self.mahalanobis_squared(
            X
        )

        return self.classify_distance(
            d2
        )

    # ========================================================
    # CLASSIFICATION À PARTIR DE D²
    # ========================================================

    def classify_distance(
        self,
        d2: Iterable,
    ) -> np.ndarray:
        """
        Classe directement des distances de Mahalanobis².
        """

        d2 = np.asarray(
            d2,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(d2)
        ):
            raise ValueError(
                "d2 contient NaN ou Inf."
            )

        if np.any(
            d2 < 0.0
        ):
            raise ValueError(
                "d2 ne peut pas être négatif."
            )

        return np.where(
            d2 <= self.thresholds["early"],
            "NOMINAL",
            np.where(
                d2 <= self.thresholds["significant"],
                "EARLY_DRIFT",
                np.where(
                    d2 <= self.thresholds["critical"],
                    "SIGNIFICANT_DRIFT",
                    "HIGH_DEVIATION",
                ),
            ),
        )

    # ========================================================
    # CONTRIBUTIONS DES FEATURES
    # ========================================================

    def feature_contributions(
        self,
        x: Iterable,
    ) -> list[tuple[str, float]]:
        """
        Estime la contribution relative de chaque
        caractéristique à l'écart observé.

        Méthode
        -------
        contribution_i = z_i² / Σ z_j²

        Attention
        ---------
        Cette attribution est descriptive.
        Elle ne constitue pas une preuve de causalité
        physique ou médicale.
        """

        x = np.asarray(
            x,
            dtype=np.float64,
        ).reshape(1, -1)

        if x.shape[1] != self.n_features:
            raise ValueError(
                "Nombre de features incompatible."
            )

        z = self.z_score(
            x
        )[0]

        squared = np.square(
            z
        )

        total = float(
            np.sum(squared)
        )

        if total <= 1e-15:
            contributions = np.zeros_like(
                squared
            )
        else:
            contributions = (
                squared
                / total
            )

        result = list(
            zip(
                self.feature_names,
                contributions,
            )
        )

        return sorted(
            result,
            key=lambda item: item[1],
            reverse=True,
        )

    # ========================================================
    # DIMENSIONS DE SANTÉ
    # ========================================================

    def health_dimensions(
        self,
        X: Iterable,
    ) -> dict[str, np.ndarray]:
        """
        Calcule des indicateurs relatifs par dimension.

        Les dimensions mesurent l'amplitude moyenne des
        écarts standardisés dans chaque famille de features.
        """

        Z = np.abs(
            self.z_score(
                X
            )
        )

        index = {
            name: i
            for i, name in enumerate(
                self.feature_names
            )
        }

        result: dict[str, np.ndarray] = {}

        for dimension, features in (
            HEALTH_DIMENSIONS.items()
        ):

            available = [
                index[name]
                for name in features
                if name in index
            ]

            if not available:
                continue

            result[dimension] = np.mean(
                Z[:, available],
                axis=1,
            )

        return result

    # ========================================================
    # SCORE DE CONFORMITÉ RELATIF
    # ========================================================

    def quality_score(
        self,
        X: Iterable,
    ) -> np.ndarray:
        """
        Calcule un score de conformité à la baseline statistique.

        Interprétation
        --------------
        100 :
            observation au centre de la baseline.

        0 :
            observation atteignant ou dépassant la limite critique.

        Le score est borné dans [0, 100].

        Important
        ---------
        Ce score mesure la conformité statistique à la référence.

        Ce n'est PAS :
            - un score clinique ;
            - une probabilité de panne ;
            - une mesure physique absolue de qualité d'image.
        """

        d2 = self.mahalanobis_squared(
            X
        )

        critical_threshold = float(
            self.thresholds["critical"]
        )

        if critical_threshold <= 0.0:
            raise ValueError(
                "Le seuil critique doit être strictement positif."
            )

        score = (
            100.0
            * (
                1.0
                - d2
                / critical_threshold
            )
        )

        return np.clip(
            score,
            0.0,
            100.0,
        )

    # ========================================================
    # RÉFÉRENCE DU SCORE QUALITÉ
    # ========================================================

    def _build_quality_reference(
        self,
    ) -> None:
        """
        Calcule les statistiques de référence des distances
        sur les acquisitions ayant servi à construire la baseline.

        Ces statistiques sont descriptives et permettent
        de caractériser la dispersion interne de la référence.
        """

        d2 = self.mahalanobis_squared(
            self.X_reference
        )

        self.reference_distance_mean = float(
            np.mean(
                d2
            )
        )

        self.reference_distance_std = float(
            np.std(
                d2,
                ddof=1,
            )
        )

    # ========================================================
    # RÉSUMÉ
    # ========================================================

    def summary(
        self,
    ) -> BaselineStatistics:
        """
        Retourne un résumé statistique indépendant du modèle.
        """

        return BaselineStatistics(
            n_samples=self.n_samples,
            n_features=self.n_features,
            feature_names=self.feature_names,
            mean=self.mu.copy(),
            std=self.sigma.copy(),
            covariance=self.covariance.copy(),
            precision=self.precision.copy(),
            shrinkage=self.shrinkage,
            thresholds=dict(
                self.thresholds
            ),
        )