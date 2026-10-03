"""
SCAN A Digital Twin V3
======================

Temporal Drift Calibration
---------------------------

Rôle
----
Construire une référence statistique indépendante pour le
monitoring temporel du signal D² de Mahalanobis.

Architecture
------------
Baseline training set
        ↓
StatisticalBaseline
        ↓
Calibration acquisitions
        ↓
Digital Signature
        ↓
D² de Mahalanobis
        ↓
mean / std / distribution
        ↓
DriftMonitor

Important
---------
La calibration temporelle est distincte de la construction
de la baseline.

Les données utilisées pour calibrer le monitoring temporel
ne doivent pas être les mêmes que celles utilisées pour
entraîner la baseline dans une validation expérimentale
indépendante.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.anomaly.statistical import StatisticalAnomalyDetector
from src.signature.baseline import StatisticalBaseline
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)


# ============================================================
# CALIBRATION STATISTICS
# ============================================================

@dataclass(frozen=True)
class DriftCalibration:
    """
    Statistiques de référence du signal temporel D².

    Le signal calibré correspond à la distance de Mahalanobis²
    obtenue pour des acquisitions supposées normales.
    """

    n_samples: int

    mean_d2: float
    std_d2: float

    median_d2: float
    mad_d2: float

    min_d2: float
    max_d2: float

    percentiles: dict[str, float]

    def validate(self) -> None:
        """
        Vérifie la cohérence des statistiques.
        """

        if self.n_samples < 2:
            raise ValueError(
                "Au moins deux observations sont nécessaires "
                "pour calibrer le monitoring temporel."
            )

        numeric_values = (
            self.mean_d2,
            self.std_d2,
            self.median_d2,
            self.mad_d2,
            self.min_d2,
            self.max_d2,
            *self.percentiles.values(),
        )

        if not all(
            np.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError(
                "La calibration contient des valeurs non finies."
            )

        if self.std_d2 <= 0.0:
            raise ValueError(
                "L'écart-type de calibration doit être "
                "strictement positif."
            )

        if self.min_d2 < 0.0:
            raise ValueError(
                "D² ne peut pas être négatif."
            )

        if self.max_d2 < self.min_d2:
            raise ValueError(
                "max_d2 doit être supérieur ou égal à min_d2."
            )

    def to_dict(self) -> dict:
        """
        Convertit la calibration en dictionnaire sérialisable.
        """

        return {
            "n_samples": self.n_samples,
            "mean_d2": self.mean_d2,
            "std_d2": self.std_d2,
            "median_d2": self.median_d2,
            "mad_d2": self.mad_d2,
            "min_d2": self.min_d2,
            "max_d2": self.max_d2,
            "percentiles": dict(
                self.percentiles
            ),
        }


# ============================================================
# D² EXTRACTION
# ============================================================

def compute_calibration_d2(
    baseline: StatisticalBaseline,
    images: list[np.ndarray]
    | tuple[np.ndarray, ...],
    source: str = "simulated",
) -> np.ndarray:
    """
    Calcule les D² de Mahalanobis pour une population
    de calibration supposée normale.

    Parameters
    ----------
    baseline:
        Baseline statistique déjà construite.

    images:
        Acquisitions indépendantes destinées à la calibration
        temporelle.
    """

    if not images:
        raise ValueError(
            "Au moins une acquisition de calibration est requise."
        )

    detector = StatisticalAnomalyDetector(
        baseline=baseline
    )

    d2_values = []

    for image in images:

        signature = build_digital_signature_from_image(
            image,
            source=source,
        )

        detection = detector.detect(
            signature.to_vector()
        )

        d2_values.append(
            detection.d2
        )

    values = np.asarray(
        d2_values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "La série D² doit être un vecteur 1D."
        )

    if len(values) < 2:
        raise ValueError(
            "Au moins deux valeurs D² sont nécessaires."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "La série D² contient NaN ou Inf."
        )

    if np.any(
        values < 0.0
    ):
        raise ValueError(
            "La série D² contient des valeurs négatives."
        )

    return values


# ============================================================
# ROBUST MAD
# ============================================================

def _compute_mad(
    values: np.ndarray,
) -> float:
    """
    Calcule la Median Absolute Deviation (MAD).

    MAD = median(|x - median(x)|)

    Cette statistique est conservée comme indicateur robuste
    de dispersion, notamment pour diagnostiquer la présence
    d'observations extrêmes dans la calibration.
    """

    median = float(
        np.median(values)
    )

    return float(
        np.median(
            np.abs(
                values - median
            )
        )
    )


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_drift_signal(
    baseline: StatisticalBaseline,
    images: list[np.ndarray]
    | tuple[np.ndarray, ...],
    source: str = "simulated",
) -> DriftCalibration:
    """
    Construit la calibration temporelle du signal D².

    Les acquisitions fournies doivent être considérées
    comme normales pour la période de calibration.
    """

    d2_values = compute_calibration_d2(
        baseline=baseline,
        images=images,
        source=source,
    )

    percentiles_array = np.percentile(
        d2_values,
        [1, 5, 25, 50, 75, 95, 99],
    )

    result = DriftCalibration(
        n_samples=len(d2_values),

        mean_d2=float(
            np.mean(d2_values)
        ),

        std_d2=float(
            np.std(
                d2_values,
                ddof=1,
            )
        ),

        median_d2=float(
            np.median(d2_values)
        ),

        mad_d2=_compute_mad(
            d2_values
        ),

        min_d2=float(
            np.min(d2_values)
        ),

        max_d2=float(
            np.max(d2_values)
        ),

        percentiles={
            "p01": float(
                percentiles_array[0]
            ),
            "p05": float(
                percentiles_array[1]
            ),
            "p25": float(
                percentiles_array[2]
            ),
            "p50": float(
                percentiles_array[3]
            ),
            "p75": float(
                percentiles_array[4]
            ),
            "p95": float(
                percentiles_array[5]
            ),
            "p99": float(
                percentiles_array[6]
            ),
        },
    )

    result.validate()

    return result