"""Physics-parameter counterfactual experiments.

This module perturbs acquisition metadata, recomputes model-derived physics,
and compares the resulting physics profile. It does not claim that metadata
perturbation reproduces a real hardware fault.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
import math

from src.physics.ultrasound import compute_ultrasound_physics


@dataclass(frozen=True)
class PhysicsCounterfactual:
    parameter: str
    baseline_value: float
    counterfactual_value: float
    wavelength_ratio: float
    axial_resolution_ratio: float
    prf_margin_ratio: float | None
    physical_consistency_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


def perturb_physics(
    image,
    params: dict,
    parameter: str,
    factor: float,
) -> PhysicsCounterfactual:
    """Apply a multiplicative metadata perturbation and compare physics."""
    if factor <= 0 or not math.isfinite(factor):
        raise ValueError("factor must be finite and > 0.")
    if parameter not in params:
        raise KeyError(f"Missing physics parameter: {parameter}")

    baseline_params = deepcopy(params)
    counter_params = deepcopy(params)

    baseline_value = float(baseline_params[parameter])
    counter_value = baseline_value * float(factor)
    counter_params[parameter] = counter_value

    baseline = compute_ultrasound_physics(image, params=baseline_params)
    counter = compute_ultrasound_physics(image, params=counter_params)

    margin_ratio = None
    if baseline.prf_depth_margin is not None and baseline.prf_depth_margin != 0:
        if counter.prf_depth_margin is not None:
            margin_ratio = float(
                counter.prf_depth_margin / baseline.prf_depth_margin
            )

    return PhysicsCounterfactual(
        parameter=parameter,
        baseline_value=baseline_value,
        counterfactual_value=counter_value,
        wavelength_ratio=float(counter.wavelength_mm / baseline.wavelength_mm),
        axial_resolution_ratio=float(
            counter.axial_resolution_mm / baseline.axial_resolution_mm
        ),
        prf_margin_ratio=margin_ratio,
        physical_consistency_delta=float(
            counter.physical_consistency_score
            - baseline.physical_consistency_score
        ),
    )
