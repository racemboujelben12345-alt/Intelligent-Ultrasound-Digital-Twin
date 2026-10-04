from pathlib import Path
import csv
import json

import cv2
import numpy as np

from src.image_analysis.edge_quality import (
    adaptive_median_canny_edge_density,
    fixed_canny_edge_density,
)
# ============================================================
# EDGE DEFINITIONS
# ============================================================

def edge_p90_gradient(image):
    """Historical percentile definition retained for comparison."""
    grad = gradient_magnitude(image)
    threshold = np.percentile(grad, 90)
    return float(np.mean(grad > threshold))


def edge_fixed_gradient(image, threshold=0.05):
    """Historical fixed-gradient definition retained for comparison."""
    grad = gradient_magnitude(image)
    return float(np.mean(grad > threshold))


def edge_canny_fixed(image):
    """Canonical fixed Canny definition from the image-analysis module."""
    return fixed_canny_edge_density(image)


def edge_adaptive_canny(image):
    """Canonical adaptive-median Canny definition from the image-analysis module."""
    return adaptive_median_canny_edge_density(image)

