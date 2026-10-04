"""Robust alternatives for the edge-density candidate feature.

The existing baseline edge_density implementation uses Canny thresholds 50/150.
This module does not replace it automatically. It lets the research pipeline
compare thresholding strategies before a final engineering decision.
"""

from __future__ import annotations

import cv2
import numpy as np


def fixed_canny_edge_density(
    image: np.ndarray,
    threshold1: float = 50.0,
    threshold2: float = 150.0,
) -> float:
    """Current fixed-threshold Canny definition used by the prototype."""
    image_u8 = _to_uint8(image)
    edges = cv2.Canny(image_u8, threshold1, threshold2)
    return float(np.count_nonzero(edges) / edges.size)


def adaptive_median_canny_edge_density(image: np.ndarray) -> float:
    """Adaptive Canny with thresholds derived from the image median."""
    image_u8 = _to_uint8(image)
    median = float(np.median(image_u8))
    lower = max(0.0, 0.66 * median)
    upper = min(255.0, 1.33 * median)

    if upper <= lower:
        upper = min(255.0, lower + 1.0)

    edges = cv2.Canny(image_u8, lower, upper)
    return float(np.count_nonzero(edges) / edges.size)


def gradient_percentile_density(
    image: np.ndarray,
    percentile: float = 90.0,
) -> float:
    """Density of pixels whose gradient magnitude exceeds a robust percentile."""
    image_f = _to_float01(image)

    gx = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    threshold = float(np.percentile(magnitude, percentile))
    return float(np.mean(magnitude >= threshold))


def compare_edge_definitions(image: np.ndarray) -> dict[str, float]:
    """Return all candidate definitions on the same image."""
    return {
        "edge_density_fixed_canny": fixed_canny_edge_density(image),
        "edge_density_adaptive_median_canny": adaptive_median_canny_edge_density(image),
        "edge_density_gradient_p90": gradient_percentile_density(
            image, percentile=90.0
        ),
    }


def _to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError("Image must be a 2D grayscale array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Image contains non-finite values.")

    if np.issubdtype(array.dtype, np.uint8):
        return array.astype(np.float32) / 255.0
    if np.issubdtype(array.dtype, np.uint16):
        return array.astype(np.float32) / 65535.0

    array = array.astype(np.float32)
    if np.min(array) < 0.0 or np.max(array) > 1.0:
        raise ValueError("Float images must be in [0, 1].")
    return array


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(_to_float01(image) * 255.0, 0, 255).astype(np.uint8)


__all__ = [
    "fixed_canny_edge_density",
    "adaptive_median_canny_edge_density",
    "gradient_percentile_density",
    "compare_edge_definitions",
]
