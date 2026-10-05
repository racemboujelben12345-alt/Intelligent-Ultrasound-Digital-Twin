"""
SCAN A Digital Twin V2
======================

Digital Signature of an Ultrasound Acquisition.

Rôle
----
Transformer les métriques extraites d'une image échographique
en une signature numérique structurée et traçable.

Architecture :

    Image
      ↓
    basic_metrics
      ↓
    DigitalSignature
      ↓
    vector de features
      ↓
    StatisticalBaseline
      ↓
    anomaly / drift / health state
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from src.physics.ultrasound import compute_ultrasound_physics

from .basic_metrics import (
    analyze_image,
    compute_basic_metrics,
)


# ============================================================
# VERSION
# ============================================================

SIGNATURE_VERSION = "2.1"


# ============================================================
# ORDRE CANONIQUE DES FEATURES
# ============================================================

FEATURE_ORDER = (
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
    "wavelength_mm",
    "axial_resolution_mm",
    "attenuation_proxy_db_cm_mhz",
    "axial_intensity_slope",
    "depth_uniformity",
    "near_field_energy_ratio",
    "physical_consistency_score",
    "theoretical_max_prf_hz",
    "prf_depth_margin",
    "metadata_complete",
)


# ============================================================
# DIGITAL SIGNATURE
# ============================================================

@dataclass(frozen=True)
class DigitalSignature:
    """
    Signature numérique d'une acquisition échographique.

    Cette classe représente l'état image observé à un instant
    donné.

    Elle ne réalise PAS :
        - la détection d'anomalie
        - la classification
        - la prédiction
        - la simulation

    Ces responsabilités appartiennent à d'autres modules.
    """

    # --------------------------------------------------------
    # Identité
    # --------------------------------------------------------

    source: str
    image_path: str

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    width: int
    height: int

    # --------------------------------------------------------
    # Intensité
    # --------------------------------------------------------

    mean_intensity: float
    std_intensity: float
    median_intensity: float

    min_intensity: float
    max_intensity: float

    dynamic_range: float

    percentile_1: float
    percentile_5: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    percentile_99: float

    # --------------------------------------------------------
    # Contraste / distribution
    # --------------------------------------------------------

    rms_contrast: float
    coefficient_variation: float
    entropy: float
    uniformity: float

    # --------------------------------------------------------
    # Texture / speckle
    # --------------------------------------------------------

    speckle_proxy: float

    # --------------------------------------------------------
    # Structure / contours
    # --------------------------------------------------------

    edge_density: float

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradient_mean: float
    gradient_std: float
    gradient_max: float

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    sharpness_laplacian: float

    # --------------------------------------------------------
    # Version
    # --------------------------------------------------------

    signature_version: str = SIGNATURE_VERSION
    wavelength_mm: float = 0.154
    axial_resolution_mm: float = 0.077
    attenuation_proxy_db_cm_mhz: float = 0.0
    axial_intensity_slope: float = 0.0
    depth_uniformity: float = 1.0
    near_field_energy_ratio: float = 0.5
    physical_consistency_score: float = 1.0
    theoretical_max_prf_hz: float = 0.0
    prf_depth_margin: float = 0.0
    metadata_complete: float = 0.0

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> None:
        """Vérifie la cohérence interne de la signature."""

        if not self.source:
            raise ValueError(
                "La source de la signature ne peut pas être vide."
            )

        if self.width <= 0:
            raise ValueError(
                "La largeur de l'image doit être positive."
            )

        if self.height <= 0:
            raise ValueError(
                "La hauteur de l'image doit être positive."
            )

        if not self.signature_version:
            raise ValueError(
                "La version de signature est obligatoire."
            )

        values = self.numeric_values()

        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"Feature non finie : {name}={value}"
                )

        if self.min_intensity > self.max_intensity:
            raise ValueError(
                "min_intensity ne peut pas être supérieure "
                "à max_intensity."
            )

        if self.dynamic_range < 0:
            raise ValueError(
                "dynamic_range ne peut pas être négatif."
            )

        if self.std_intensity < 0:
            raise ValueError(
                "std_intensity ne peut pas être négatif."
            )

        if self.rms_contrast < 0:
            raise ValueError(
                "rms_contrast ne peut pas être négatif."
            )

        if self.sharpness_laplacian < 0:
            raise ValueError(
                "sharpness_laplacian ne peut pas être négatif."
            )

    # ========================================================
    # FEATURES NUMÉRIQUES
    # ========================================================

    def numeric_values(self) -> dict[str, float]:
        """Retourne les caractéristiques utilisées par le modèle."""

        return {
            "mean_intensity": float(self.mean_intensity),
            "std_intensity": float(self.std_intensity),
            "dynamic_range": float(self.dynamic_range),
            "coefficient_variation": float(
                self.coefficient_variation
            ),
            "rms_contrast": float(self.rms_contrast),
            "entropy": float(self.entropy),
            "sharpness_laplacian": float(
                self.sharpness_laplacian
            ),
            "speckle_proxy": float(self.speckle_proxy),
            "edge_density": float(self.edge_density),
            "gradient_mean": float(self.gradient_mean),
            "gradient_std": float(self.gradient_std),
            "gradient_max": float(self.gradient_max),
            "uniformity": float(self.uniformity),
            "wavelength_mm": float(self.wavelength_mm),
            "axial_resolution_mm": float(self.axial_resolution_mm),
            "attenuation_proxy_db_cm_mhz": float(self.attenuation_proxy_db_cm_mhz),
            "axial_intensity_slope": float(self.axial_intensity_slope),
            "depth_uniformity": float(self.depth_uniformity),
            "near_field_energy_ratio": float(self.near_field_energy_ratio),
            "physical_consistency_score": float(self.physical_consistency_score),
            "theoretical_max_prf_hz": float(self.theoretical_max_prf_hz),
            "prf_depth_margin": float(self.prf_depth_margin),
            "metadata_complete": float(self.metadata_complete),
        }

    # ========================================================
    # VECTORISATION
    # ========================================================

    def to_vector(
        self,
        feature_order: Iterable[str] = FEATURE_ORDER,
    ) -> np.ndarray:
        """
        Convertit la signature en vecteur numérique.

        L'ordre des features est explicitement contrôlé afin
        d'éviter une incompatibilité avec la baseline.
        """

        values = self.numeric_values()

        feature_order = tuple(feature_order)

        missing = [
            name
            for name in feature_order
            if name not in values
        ]

        if missing:
            raise ValueError(
                "Features absentes de la signature : "
                f"{missing}"
            )

        vector = np.asarray(
            [
                values[name]
                for name in feature_order
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Le vecteur de signature contient "
                "des valeurs non finies."
            )

        return vector

    # ========================================================
    # DICTIONNAIRE
    # ========================================================

    def to_dict(self) -> dict:
        """Sérialise la signature sous forme de dictionnaire."""

        return {
            "signature_version": self.signature_version,
            "source": self.source,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            **self.numeric_values(),
        }


# ============================================================
# CONSTRUCTION À PARTIR DES MÉTRIQUES
# ============================================================

def _build_signature_from_metrics(
    metrics: dict,
    source: str,
    image_path: str | None,
    physics: object | None = None,
) -> DigitalSignature:
    """
    Fonction interne commune pour construire une DigitalSignature
    à partir d'un dictionnaire de métriques.
    """

    signature = DigitalSignature(
        source=source,
        image_path=image_path or "",

        width=int(metrics["width"]),
        height=int(metrics["height"]),

        mean_intensity=float(
            metrics["mean_intensity"]
        ),

        std_intensity=float(
            metrics["std_intensity"]
        ),

        median_intensity=float(
            metrics["median_intensity"]
        ),

        min_intensity=float(
            metrics["min_intensity"]
        ),

        max_intensity=float(
            metrics["max_intensity"]
        ),

        dynamic_range=float(
            metrics["dynamic_range"]
        ),

        percentile_1=float(
            metrics["percentile_1"]
        ),

        percentile_5=float(
            metrics["percentile_5"]
        ),

        percentile_25=float(
            metrics["percentile_25"]
        ),

        percentile_50=float(
            metrics["percentile_50"]
        ),

        percentile_75=float(
            metrics["percentile_75"]
        ),

        percentile_95=float(
            metrics["percentile_95"]
        ),

        percentile_99=float(
            metrics["percentile_99"]
        ),

        rms_contrast=float(
            metrics["rms_contrast"]
        ),

        coefficient_variation=float(
            metrics["coefficient_variation"]
        ),

        entropy=float(
            metrics["entropy"]
        ),

        uniformity=float(
            metrics["uniformity"]
        ),

        speckle_proxy=float(
            metrics["speckle_proxy"]
        ),

        edge_density=float(
            metrics["edge_density"]
        ),

        gradient_mean=float(
            metrics["gradient_mean"]
        ),

        gradient_std=float(
            metrics["gradient_std"]
        ),

        gradient_max=float(
            metrics["gradient_max"]
        ),

        sharpness_laplacian=float(
            metrics["sharpness_laplacian"]
        ),

        signature_version=SIGNATURE_VERSION,
        wavelength_mm=float(getattr(physics, "wavelength_mm", 0.154)),
        axial_resolution_mm=float(getattr(physics, "axial_resolution_mm", 0.077)),
        attenuation_proxy_db_cm_mhz=float(getattr(physics, "attenuation_proxy_db_cm_mhz", 0.0)),
        axial_intensity_slope=float(getattr(physics, "axial_intensity_slope", 0.0)),
        depth_uniformity=float(getattr(physics, "depth_uniformity", 1.0)),
        near_field_energy_ratio=float(getattr(physics, "near_field_energy_ratio", 0.5)),
        physical_consistency_score=float(getattr(physics, "physical_consistency_score", 1.0)),
        theoretical_max_prf_hz=float(getattr(physics, "theoretical_max_prf_hz", 0.0) or 0.0),
        prf_depth_margin=float(getattr(physics, "prf_depth_margin", 0.0) or 0.0),
        metadata_complete=float(bool(getattr(physics, "metadata_complete", False))),
    )

    signature.validate()

    return signature


# ============================================================
# CONSTRUCTION À PARTIR D'UN FICHIER
# ============================================================

def build_digital_signature(
    image_path: str | Path,
    source: str = "UNKNOWN",
    params: dict | None = None,
) -> DigitalSignature:
    """
    Construit une DigitalSignature à partir d'une image
    enregistrée sur le disque.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    metrics = analyze_image(image_path)
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64) / 255.0
    physics = compute_ultrasound_physics(image, params=params)

    return _build_signature_from_metrics(
        metrics=metrics,
        source=source,
        image_path=str(image_path),
        physics=physics,
    )


# ============================================================
# CONSTRUCTION À PARTIR D'UNE IMAGE EN MÉMOIRE
# ============================================================

def build_digital_signature_from_image(
    image: np.ndarray,
    source: str = "UNKNOWN",
    image_path: str | Path | None = None,
    params: dict | None = None,
) -> DigitalSignature:
    """
    Construit une DigitalSignature directement à partir
    d'une image numpy.ndarray en mémoire.

    Cette fonction est notamment utilisée par le moteur
    de simulation :

        image originale
              ↓
        dégradation simulée
              ↓
        image numpy
              ↓
        DigitalSignature

    Parameters
    ----------
    image:
        Image 2D en niveaux de gris.

    source:
        Origine de l'image :
            scan_a
            public
            simulated

    image_path:
        Chemin optionnel permettant de conserver une trace
        de l'image si elle existe sur disque.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "image doit être un numpy.ndarray."
        )

    if image.ndim != 2:
        raise ValueError(
            "L'image doit être une image 2D en niveaux de gris."
        )

    if not np.all(np.isfinite(image)):
        raise ValueError(
            "L'image contient des valeurs non finies."
        )

    metrics = compute_basic_metrics(image)
    physics = compute_ultrasound_physics(image, params=params)

    return _build_signature_from_metrics(
        metrics=metrics,
        source=source,
        image_path=(
            str(image_path)
            if image_path is not None
            else ""
        ),
        physics=physics,
    )


# ============================================================
# COMPATIBILITÉ / UTILITAIRE
# ============================================================

def signature_to_dict(
    signature: DigitalSignature,
) -> dict:
    """
    Convertit explicitement une signature en dictionnaire.
    """

    if not isinstance(
        signature,
        DigitalSignature,
    ):
        raise TypeError(
            "signature doit être une instance "
            "de DigitalSignature."
        )

    signature.validate()

    return signature.to_dict()