"""Physics-aware ultrasound feature engine.

The module provides engineering proxies derived from B-mode images and,
when available, acquisition metadata. It deliberately distinguishes:
- quantities with a direct physical model (wavelength, nominal axial resolution);
- image-derived proxies (relative attenuation, depth uniformity);
- metadata consistency checks.

It is not a substitute for phantom QA measurements or calibrated scanner data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class UltrasoundPhysicsProfile:
    sound_speed_m_s: float
    frequency_mhz: float
    depth_mm: float | None
    wavelength_mm: float
    axial_resolution_mm: float
    attenuation_proxy_db_cm_mhz: float
    axial_intensity_slope: float
    depth_uniformity: float
    near_field_energy_ratio: float
    physical_consistency_score: float
    metadata_complete: bool

    def validate(self) -> None:
        vals = (
            self.sound_speed_m_s, self.frequency_mhz, self.wavelength_mm,
            self.axial_resolution_mm, self.attenuation_proxy_db_cm_mhz,
            self.axial_intensity_slope, self.depth_uniformity,
            self.near_field_energy_ratio, self.physical_consistency_score,
        )
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("Physics profile contains non-finite values.")
        if self.sound_speed_m_s <= 0 or self.frequency_mhz <= 0:
            raise ValueError("Sound speed and frequency must be positive.")
        if self.depth_mm is not None and self.depth_mm <= 0:
            raise ValueError("depth_mm must be positive when supplied.")
        if not 0.0 <= self.depth_uniformity <= 1.0:
            raise ValueError("depth_uniformity must be in [0,1].")
        if not 0.0 <= self.near_field_energy_ratio <= 1.0:
            raise ValueError("near_field_energy_ratio must be in [0,1].")
        if not 0.0 <= self.physical_consistency_score <= 1.0:
            raise ValueError("physical_consistency_score must be in [0,1].")


def _meta_number(params: dict | None, *keys: str) -> float | None:
    if not params:
        return None
    for key in keys:
        value = params.get(key)
        if value is None or value == "":
            continue
        try:
            number = float(value)
            if np.isfinite(number):
                return number
        except (TypeError, ValueError):
            continue
    return None


def compute_ultrasound_physics(
    image: np.ndarray,
    *,
    params: dict | None = None,
) -> UltrasoundPhysicsProfile:
    """Compute physics-aware descriptors from an image and optional metadata.

    Defaults follow the conventional soft-tissue reference speed of 1540 m/s.
    Frequency and depth should come from the scanner metadata whenever possible.
    Image-derived attenuation is explicitly a relative proxy because display
    gray levels are not calibrated acoustic pressure measurements.
    """
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("image must be a finite 2D array.")
    x = np.clip(x, 0.0, 1.0)

    c = _meta_number(params, "sound_speed_m_s", "speed_of_sound_m_s")
    frequency = _meta_number(params, "frequency_mhz", "frequency")
    depth_mm = _meta_number(params, "depth_mm", "depth")

    c = 1540.0 if c is None else c
    frequency = 5.0 if frequency is None else frequency
    metadata_complete = depth_mm is not None and _meta_number(
        params, "frequency_mhz", "frequency"
    ) is not None

    wavelength_mm = (c / (frequency * 1e6)) * 1e3
    axial_resolution_mm = wavelength_mm / 2.0

    # Robust depth profile. A log transform gives a stable relative attenuation
    # descriptor, but is not an absolute dB/cm/MHz measurement.
    profile = np.median(x, axis=1)
    depth_axis = (
        np.linspace(0.0, depth_mm, len(profile))
        if depth_mm is not None
        else np.linspace(0.0, 1.0, len(profile))
    )
    eps = 1e-4
    log_profile = np.log(np.clip(profile, eps, None))
    slope = float(np.polyfit(depth_axis, log_profile, 1)[0])
    relative_db_per_cm = abs(slope) * 10.0 / math.log(10.0) if depth_mm else abs(slope)
    attenuation_proxy = float(relative_db_per_cm / max(frequency, 1e-9))

    q = max(1, len(profile) // 4)
    near = float(np.mean(profile[:q]))
    far = float(np.mean(profile[-q:]))
    near_field_ratio = float(np.clip(near / (near + far + eps), 0.0, 1.0))
    depth_cv = float(np.std(profile) / (np.mean(profile) + eps))
    depth_uniformity = float(np.clip(1.0 / (1.0 + depth_cv), 0.0, 1.0))

    score = 1.0
    # Metadata plausibility, not equipment-failure probability.
    if not 1.0 <= frequency <= 20.0:
        score -= 0.35
    if not 1450.0 <= c <= 1650.0:
        score -= 0.35
    if depth_mm is not None and not 1.0 <= depth_mm <= 300.0:
        score -= 0.20
    if not np.isfinite(slope):
        score -= 0.20
    score = float(np.clip(score, 0.0, 1.0))

    result = UltrasoundPhysicsProfile(
        sound_speed_m_s=float(c),
        frequency_mhz=float(frequency),
        depth_mm=None if depth_mm is None else float(depth_mm),
        wavelength_mm=float(wavelength_mm),
        axial_resolution_mm=float(axial_resolution_mm),
        attenuation_proxy_db_cm_mhz=float(attenuation_proxy),
        axial_intensity_slope=float(slope),
        depth_uniformity=depth_uniformity,
        near_field_energy_ratio=near_field_ratio,
        physical_consistency_score=score,
        metadata_complete=bool(metadata_complete),
    )
    result.validate()
    return result
