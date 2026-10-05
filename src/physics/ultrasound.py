"""Physics-informed ultrasound engineering layer.

The module converts acquisition metadata and B-mode image observables into
traceable physics descriptors. It deliberately distinguishes:

* model-derived quantities: wavelength, pulse length and nominal axial
  resolution when pulse duration is available;
* consistency checks: depth/PRF timing and parameter plausibility;
* image-derived proxies: relative attenuation and depth uniformity.

The layer is an engineering model, not a calibrated acoustic measurement
system. Absolute attenuation, acoustic pressure, MI/TI and beam profiles
require calibrated equipment/phantom measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_SOUND_SPEED_M_S = 1540.0


@dataclass(frozen=True)
class UltrasoundPhysicsProfile:
    sound_speed_m_s: float
    frequency_mhz: float
    depth_mm: float | None
    wavelength_mm: float
    pulse_duration_us: float | None
    spatial_pulse_length_mm: float | None
    axial_resolution_mm: float
    theoretical_max_prf_hz: float | None
    prf_hz: float | None
    prf_depth_margin: float | None
    attenuation_proxy_db_cm_mhz: float
    axial_intensity_slope: float
    depth_uniformity: float
    near_field_energy_ratio: float
    physical_consistency_score: float
    metadata_complete: bool

    def validate(self) -> None:
        vals = (
            self.sound_speed_m_s,
            self.frequency_mhz,
            self.wavelength_mm,
            self.axial_resolution_mm,
            self.attenuation_proxy_db_cm_mhz,
            self.axial_intensity_slope,
            self.depth_uniformity,
            self.near_field_energy_ratio,
            self.physical_consistency_score,
        )
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("Physics profile contains non-finite values.")

        if self.sound_speed_m_s <= 0 or self.frequency_mhz <= 0:
            raise ValueError("Sound speed and frequency must be positive.")

        if self.depth_mm is not None and self.depth_mm <= 0:
            raise ValueError("depth_mm must be positive when supplied.")

        for name, value in (
            ("pulse_duration_us", self.pulse_duration_us),
            ("spatial_pulse_length_mm", self.spatial_pulse_length_mm),
            ("theoretical_max_prf_hz", self.theoretical_max_prf_hz),
            ("prf_hz", self.prf_hz),
            ("prf_depth_margin", self.prf_depth_margin),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when supplied.")

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
    """Compute physics descriptors from an image and optional metadata.

    Physics relationships used here:

        lambda = c / f
        SPL    = c * PD
        R_axial ~= SPL / 2
        PRF_max ~= c / (2D)

    The PRF relation is a timing upper bound for conventional pulse-echo
    acquisition; real scanners may use additional architecture such as
    parallel receive, interleaving or mode-specific constraints.

    Image gray levels are display-domain quantities. Therefore attenuation is
    explicitly reported as a relative proxy, never as calibrated acoustic
    attenuation.
    """
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("image must be a finite 2D array.")
    x = np.clip(x, 0.0, 1.0)

    c = _meta_number(params, "sound_speed_m_s", "speed_of_sound_m_s")
    frequency = _meta_number(params, "frequency_mhz", "frequency")
    depth_mm = _meta_number(params, "depth_mm", "depth")
    pulse_duration_us = _meta_number(
        params, "pulse_duration_us", "pulse_duration", "pd_us", "pd"
    )
    prf_hz = _meta_number(params, "prf_hz", "prf")

    c = DEFAULT_SOUND_SPEED_M_S if c is None else c
    frequency = 5.0 if frequency is None else frequency

    raw_frequency = _meta_number(params, "frequency_mhz", "frequency")
    metadata_complete = depth_mm is not None and raw_frequency is not None

    wavelength_mm = (c / (frequency * 1e6)) * 1e3

    if pulse_duration_us is not None and pulse_duration_us > 0:
        spatial_pulse_length_mm = c * pulse_duration_us * 1e-3
        axial_resolution_mm = spatial_pulse_length_mm / 2.0
    else:
        spatial_pulse_length_mm = None
        # This is explicitly a wavelength-scale proxy, not a scanner QA result.
        axial_resolution_mm = wavelength_mm / 2.0

    theoretical_max_prf_hz = None
    prf_depth_margin = None
    if depth_mm is not None:
        depth_m = depth_mm * 1e-3
        theoretical_max_prf_hz = c / (2.0 * depth_m)
        if prf_hz is not None and theoretical_max_prf_hz > 0:
            prf_depth_margin = theoretical_max_prf_hz / prf_hz

    profile = np.median(x, axis=1)
    depth_axis = (
        np.linspace(0.0, depth_mm, len(profile))
        if depth_mm is not None
        else np.linspace(0.0, 1.0, len(profile))
    )

    eps = 1e-4
    log_profile = np.log(np.clip(profile, eps, None))
    slope = float(np.polyfit(depth_axis, log_profile, 1)[0])

    relative_db_per_cm = (
        abs(slope) * 10.0 / math.log(10.0) if depth_mm else abs(slope)
    )
    attenuation_proxy = float(
        relative_db_per_cm / max(frequency, 1e-9)
    )

    q = max(1, len(profile) // 4)
    near = float(np.mean(profile[:q]))
    far = float(np.mean(profile[-q:]))
    near_field_ratio = float(
        np.clip(near / (near + far + eps), 0.0, 1.0)
    )

    depth_cv = float(np.std(profile) / (np.mean(profile) + eps))
    depth_uniformity = float(
        np.clip(1.0 / (1.0 + depth_cv), 0.0, 1.0)
    )

    score = 1.0

    if not 1.0 <= frequency <= 20.0:
        score -= 0.30
    if not 1450.0 <= c <= 1650.0:
        score -= 0.30
    if depth_mm is not None and not 1.0 <= depth_mm <= 300.0:
        score -= 0.15
    if pulse_duration_us is not None and not 0.01 <= pulse_duration_us <= 20.0:
        score -= 0.10
    if prf_hz is not None and prf_hz <= 0:
        score -= 0.15

    # A reported PRF above the simple pulse-echo timing bound is flagged as
    # a metadata/physics inconsistency, not as proof of hardware failure.
    if (
        prf_depth_margin is not None
        and prf_depth_margin < 1.0
    ):
        score -= 0.20

    score = float(np.clip(score, 0.0, 1.0))

    result = UltrasoundPhysicsProfile(
        sound_speed_m_s=float(c),
        frequency_mhz=float(frequency),
        depth_mm=None if depth_mm is None else float(depth_mm),
        wavelength_mm=float(wavelength_mm),
        pulse_duration_us=(
            None if pulse_duration_us is None else float(pulse_duration_us)
        ),
        spatial_pulse_length_mm=(
            None
            if spatial_pulse_length_mm is None
            else float(spatial_pulse_length_mm)
        ),
        axial_resolution_mm=float(axial_resolution_mm),
        theoretical_max_prf_hz=(
            None
            if theoretical_max_prf_hz is None
            else float(theoretical_max_prf_hz)
        ),
        prf_hz=None if prf_hz is None else float(prf_hz),
        prf_depth_margin=(
            None if prf_depth_margin is None else float(prf_depth_margin)
        ),
        attenuation_proxy_db_cm_mhz=float(attenuation_proxy),
        axial_intensity_slope=float(slope),
        depth_uniformity=depth_uniformity,
        near_field_energy_ratio=near_field_ratio,
        physical_consistency_score=score,
        metadata_complete=bool(metadata_complete),
    )
    result.validate()
    return result
