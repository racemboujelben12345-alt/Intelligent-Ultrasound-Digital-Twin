"""Validation utilities for controlled Digital Twin evidence experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from src.digital_twin.evidence_fusion import fuse_evidence
from src.physics.causal_evidence import rank_causal_evidence
from src.image_analysis.digital_signature import build_digital_signature_from_image
from src.simulation.degradation import DegradationType, apply_degradation


@dataclass(frozen=True)
class EvidenceVVRecord:
    scenario: str
    severity: float
    signature_delta_l2: float
    top_mechanism: str
    causal_agreement: float
    fusion_score: float
    disagreement: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_controlled_degradation(
    image: np.ndarray,
    *,
    scenario: DegradationType | str,
    severity: float,
    params: dict | None = None,
    seed: int = 42,
) -> EvidenceVVRecord:
    """Compare baseline and a controlled digital degradation.

    This is software/simulation V&V evidence only; it is not evidence of a
    physical SCAN A failure.
    """
    baseline = build_digital_signature_from_image(
        image, source="simulated", params=params
    )
    degraded = apply_degradation(image, scenario, severity, seed=seed)
    observed = build_digital_signature_from_image(
        degraded.image, source="simulated", params=params
    )

    delta = observed.to_vector() - baseline.to_vector()
    l2 = float(np.linalg.norm(delta))

    names = tuple(baseline.feature_names)
    delta_map = {name: float(value) for name, value in zip(names, delta)}
    ranked = rank_causal_evidence(delta_map)
    top = ranked[0] if ranked else None

    causal = float(top.agreement_score) if top else 0.0
    mechanism = top.mechanism if top else ""

    # Both statistical and AI channels are intentionally represented here by
    # normalized perturbation magnitude proxies. The production analyzer uses
    # its calibrated detector/AI outputs instead.
    perturbation = float(np.clip(l2 / (1.0 + l2), 0.0, 1.0))
    fusion = fuse_evidence(
        statistical=perturbation,
        ai=perturbation,
        physics=perturbation,
        causal=causal,
        counterfactual=None,
    )

    return EvidenceVVRecord(
        scenario=str(scenario.value if isinstance(scenario, DegradationType) else scenario),
        severity=float(severity),
        signature_delta_l2=l2,
        top_mechanism=mechanism,
        causal_agreement=causal,
        fusion_score=fusion.fusion_score,
        disagreement=fusion.disagreement,
    )
