"""
SCAN A Digital Twin V2
--------------------------------
Moteur d'analyse quantitative des images échographiques.

Objectif :
    image -> métriques quantitatives -> Digital Signature

Ce module ne fait PAS de diagnostic médical.
Il caractérise l'image pour :
    - surveillance de qualité
    - comparaison à une référence
    - détection de dérive
    - simulation de dégradation
    - validation du Digital Twin
"""

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. CHARGEMENT
# ============================================================

def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """
    Charge une image et la convertit en niveaux de gris.

    Returns
    -------
    np.ndarray
        Image uint8 2D.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Impossible de lire l'image : {image_path}"
        )

    return image


# ============================================================
# 2. NORMALISATION
# ============================================================

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convertit l'image vers float32 dans une échelle cohérente.

    Politique de normalisation V2
    -----------------------------
    - uint8  -> division par 255
    - uint16 -> division par 65535
    - float  -> conservé tel quel si déjà dans [0, 1]

    Important
    ---------
    Aucune normalisation min-max n'est effectuée image par image.

    Cela permet de préserver les variations relatives de :
        - contraste
        - dynamique d'intensité
        - niveau moyen
        - bruit
        - texture

    Cette politique est nécessaire pour la surveillance
    quantitative d'un système d'imagerie.
    """

    array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(
            "L'image doit être 2D en niveaux de gris."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "L'image contient des valeurs non finies."
        )

    # --------------------------------------------------------
    # Entiers : conversion physique de l'échelle numérique
    # --------------------------------------------------------

    if np.issubdtype(array.dtype, np.uint8):
        return (
            array.astype(np.float32) / 255.0
        )

    if np.issubdtype(array.dtype, np.uint16):
        return (
            array.astype(np.float32) / 65535.0
        )

    # --------------------------------------------------------
    # Flottants : l'image doit déjà être dans [0, 1]
    # --------------------------------------------------------

    image_float = array.astype(np.float32)

    minimum = float(np.min(image_float))
    maximum = float(np.max(image_float))

    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            "Une image flottante doit être normalisée dans [0, 1]. "
            "Aucune normalisation min-max automatique n'est appliquée."
        )

    return image_float

# ============================================================
# 3. STATISTIQUES D'INTENSITÉ
# ============================================================

def compute_intensity_statistics(
    image: np.ndarray,
) -> dict:
    """
    Statistiques fondamentales de l'intensité.
    """
    image = normalize_image(image)

    percentiles = np.percentile(
        image,
        [1, 5, 25, 50, 75, 95, 99],
    )

    mean = float(np.mean(image))
    std = float(np.std(image))

    return {
        "min_intensity": float(np.min(image)),
        "max_intensity": float(np.max(image)),
        "mean_intensity": mean,
        "median_intensity": float(np.median(image)),
        "std_intensity": std,
        "dynamic_range": float(
            np.max(image) - np.min(image)
        ),
        "percentile_1": float(percentiles[0]),
        "percentile_5": float(percentiles[1]),
        "percentile_25": float(percentiles[2]),
        "percentile_50": float(percentiles[3]),
        "percentile_75": float(percentiles[4]),
        "percentile_95": float(percentiles[5]),
        "percentile_99": float(percentiles[6]),
        "coefficient_variation": float(
            std / (mean + 1e-8)
        ),
    }


# ============================================================
# 4. CONTRASTE
# ============================================================

def compute_contrast(
    image: np.ndarray,
) -> dict:
    """
    Mesures liées au contraste global.
    """
    image = normalize_image(image)

    mean = float(np.mean(image))
    std = float(np.std(image))

    return {
        "contrast_std": std,
        "rms_contrast": float(
            std / (mean + 1e-8)
        ),
    }


# ============================================================
# 5. ENTROPIE
# ============================================================

def compute_entropy(
    image: np.ndarray,
    bins: int = 256,
) -> float:
    """
    Entropie Shannon de l'image.

    Une variation de l'entropie peut refléter une modification
    de la distribution des intensités et de la texture.
    """
    image = normalize_image(image)

    histogram, _ = np.histogram(
        image,
        bins=bins,
        range=(0.0, 1.0),
    )

    probability = (
        histogram.astype(np.float64)
        / max(np.sum(histogram), 1)
    )

    probability = probability[
        probability > 0
    ]

    entropy = -np.sum(
        probability * np.log2(probability)
    )

    return float(entropy)


# ============================================================
# 6. GRADIENTS
# ============================================================

def compute_gradient_statistics(
    image: np.ndarray,
) -> dict:
    """
    Analyse de la structure des gradients.
    """
    image = normalize_image(image)

    gradient_x = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gradient_x,
        gradient_y,
    )

    return {
        "gradient_mean": float(
            np.mean(magnitude)
        ),
        "gradient_std": float(
            np.std(magnitude)
        ),
        "gradient_max": float(
            np.max(magnitude)
        ),
    }


# ============================================================
# 7. DENSITÉ DE CONTOURS
# ============================================================

def compute_edge_density(
    image: np.ndarray,
) -> float:
    """
    Proportion de pixels appartenant aux contours détectés.
    """
    image = normalize_image(image)

    image_uint8 = (
        image * 255
    ).astype(np.uint8)

    edges = cv2.Canny(
        image_uint8,
        threshold1=50,
        threshold2=150,
    )

    total_pixels = edges.size

    if total_pixels == 0:
        return 0.0

    return float(
        np.count_nonzero(edges)
        / total_pixels
    )


# ============================================================
# 8. NETTETÉ
# ============================================================

def compute_sharpness(
    image: np.ndarray,
) -> float:
    """
    Variance du Laplacien.

    Cette mesure sert de proxy de netteté.
    """
    image = normalize_image(image)

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_32F,
    )

    return float(
        np.var(laplacian)
    )


# ============================================================
# 9. SPECKLE / BRUIT
# ============================================================

def compute_speckle_proxy(
    image: np.ndarray,
) -> float:
    """
    Estimation simple du contenu haute fréquence local.

    Ce n'est pas une estimation physique absolue du speckle.
    Il s'agit d'un indicateur relatif utile pour comparer
    différentes acquisitions dans un même protocole.
    """
    image = normalize_image(image)

    smooth = cv2.GaussianBlur(
        image,
        (5, 5),
        0,
    )

    residual = image - smooth

    residual_std = float(
        np.std(residual)
    )

    mean_intensity = float(
        np.mean(image)
    )

    return float(
        residual_std
        / (mean_intensity + 1e-8)
    )


# ============================================================
# 10. UNIFORMITÉ
# ============================================================

def compute_uniformity(
    image: np.ndarray,
) -> float:
    """
    Indicateur simple d'homogénéité globale.
    """
    image = normalize_image(image)

    mean = float(np.mean(image))
    std = float(np.std(image))

    return float(
        1.0
        / (
            1.0
            + std / (mean + 1e-8)
        )
    )


# ============================================================
# 11. COEFFICIENT DE VARIATION
# ============================================================

def compute_coefficient_variation(
    image: np.ndarray,
) -> float:
    """
    CV = sigma / moyenne.
    """
    image = normalize_image(image)

    mean = float(np.mean(image))
    std = float(np.std(image))

    return float(
        std / (mean + 1e-8)
    )


# ============================================================
# 12. ANALYSE GLOBALE
# ============================================================

def compute_basic_metrics(
    image: np.ndarray,
) -> dict:
    """
    Pipeline principal d'analyse.

    Toutes les métriques sont calculées à partir
    de la même image normalisée conceptuellement.
    """
    if image.ndim != 2:
        raise ValueError(
            "L'image doit être en niveaux de gris 2D."
        )

    image = np.asarray(image)

    metrics = {}

    # Dimensions
    metrics["height"] = int(
        image.shape[0]
    )

    metrics["width"] = int(
        image.shape[1]
    )

    # Intensité
    metrics.update(
        compute_intensity_statistics(image)
    )

    # Contraste
    metrics.update(
        compute_contrast(image)
    )

    # Texture / information
    metrics["entropy"] = compute_entropy(
        image
    )

    # Netteté
    metrics["sharpness_laplacian"] = (
        compute_sharpness(image)
    )

    # Speckle / bruit
    metrics["speckle_proxy"] = (
        compute_speckle_proxy(image)
    )

    # Contours
    metrics["edge_density"] = (
        compute_edge_density(image)
    )

    # Gradients
    metrics.update(
        compute_gradient_statistics(image)
    )

    # Uniformité
    metrics["uniformity"] = (
        compute_uniformity(image)
    )

    return metrics


# ============================================================
# 13. ANALYSE D'UN FICHIER
# ============================================================

def analyze_image(
    image_path: str | Path,
) -> dict:
    """
    Charge et analyse une image.
    """
    image_path = Path(image_path)

    image = load_grayscale_image(
        image_path
    )

    metrics = compute_basic_metrics(
        image
    )

    metrics["image_path"] = str(
        image_path
    )

    return metrics