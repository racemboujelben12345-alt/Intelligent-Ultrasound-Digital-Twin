"""Compare edge-density definitions under controlled digital perturbations.

This is a software/feature robustness audit, not a physical scanner QA test.
Each transformed image must be generated and documented by the caller.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.image_analysis.edge_quality import compare_edge_definitions


def evaluate_edge_metric_sensitivity(
    reference_image: np.ndarray,
    perturbed_images: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compare edge-density metrics on a reference and named image variants.

    Returns JSON-friendly baseline values and per-variant absolute/relative
    changes. Relative change is None when the baseline metric is zero.
    This function does not label any definition as universally superior.
    """
    if not isinstance(perturbed_images, Mapping):
        raise TypeError("perturbed_images must be a mapping of names to images.")
    if not perturbed_images:
        raise ValueError("At least one named perturbed image is required.")

    baseline = compare_edge_definitions(_validate_image(reference_image, "reference_image"))
    variants: list[dict[str, Any]] = []

    for name, image in perturbed_images.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Perturbation names must be non-empty strings.")
        metrics = compare_edge_definitions(_validate_image(image, name))
        changes = {}
        for metric_name, base_value in baseline.items():
            value = metrics[metric_name]
            delta = float(value - base_value)
            changes[metric_name] = {
                "reference": float(base_value),
                "perturbed": float(value),
                "absolute_change": delta,
                "relative_change": (
                    float(delta / abs(base_value)) if base_value != 0 else None
                ),
            }
        variants.append({"perturbation": name, "metrics": changes})

    return {
        "analysis_type": "digital_edge_metric_sensitivity",
        "evidence_class": "simulated_or_digitally_transformed",
        "reference_metrics": {key: float(value) for key, value in baseline.items()},
        "variants": variants,
        "interpretation_limit": (
            "Measures sensitivity to the supplied digital image transformations; "
            "does not measure physical repeatability or validate scanner QA."
        ),
    }


def _validate_image(image: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty 2D grayscale image.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values.")
    return array


__all__ = ["evaluate_edge_metric_sensitivity"]
