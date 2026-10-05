"""Deterministic evidence-to-state engine for the Intelligent Ultrasound Digital Twin.

This module is the single decision layer that translates independent evidence
streams into one engineering state. It does not estimate failure probability
and does not diagnose hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ai.fusion import IntelligenceFusion
from src.digital_twin.health import TwinHealthAssessment
from src.drift.engine import DriftAnalysis


VALID_TWIN_STATES = ("NOMINAL", "WATCH", "EARLY_DRIFT", "HIGH_DEVIATION")


@dataclass(frozen=True)
class TwinStateDecision:
    state: str
    confidence: float
    evidence: tuple[str, ...]
    rationale: str
    version: str = "1.0"

    def validate(self) -> None:
        if self.state not in VALID_TWIN_STATES:
            raise ValueError(f"Invalid Twin state: {self.state}")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1].")
        if not self.evidence:
            raise ValueError("At least one evidence item is required.")
        if not self.rationale:
            raise ValueError("rationale is required.")
        if not self.version:
            raise ValueError("version is required.")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "state": self.state,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "rationale": self.rationale,
            "version": self.version,
        }


def decide_twin_state(
    *,
    statistical_state: str,
    fusion: IntelligenceFusion,
    health: TwinHealthAssessment,
    drift: DriftAnalysis | None,
) -> TwinStateDecision:
    """Produce one deterministic state from all available evidence.

    Precedence is deliberately conservative:
    HIGH_DEVIATION > EARLY_DRIFT > WATCH > NOMINAL.
    A persistent temporal drift is sufficient for EARLY_DRIFT, while a single
    drift signal only supports WATCH.
    """
    fusion.validate()
    health.validate()

    valid_statistical = {"NOMINAL", "EARLY_DRIFT", "SIGNIFICANT_DRIFT", "HIGH_DEVIATION"}
    if statistical_state not in valid_statistical:
        raise ValueError(f"Invalid statistical state: {statistical_state}")

    evidence: list[str] = [f"statistical:{statistical_state}"]
    rationale: list[str] = []

    if fusion.state == "HIGH_EVIDENCE":
        evidence.append("fusion:HIGH_EVIDENCE")
        rationale.append("fused evidence exceeds the high-evidence threshold")
    elif fusion.state == "WATCH":
        evidence.append("fusion:WATCH")
        rationale.append("fused evidence indicates a watch condition")

    if health.health_state == "HIGH_DEVIATION":
        evidence.append("health:HIGH_DEVIATION")
        rationale.append("engineering health index is in high-deviation range")
    elif health.health_state == "EARLY_DRIFT":
        evidence.append("health:EARLY_DRIFT")
        rationale.append("engineering health index indicates early drift")
    elif health.health_state == "WATCH":
        evidence.append("health:WATCH")
        rationale.append("engineering health index indicates watch")

    drift_status = None
    if drift is not None:
        drift.validate()
        drift_status = drift.latest.status
        evidence.append(f"drift:{drift_status}")
        if drift.persistent_drift_detected:
            rationale.append("persistent temporal drift detected")
        elif drift_status == "DRIFT_SIGNAL":
            rationale.append("a temporal drift signal is present")

    high = (
        statistical_state == "HIGH_DEVIATION"
        or fusion.state == "HIGH_EVIDENCE"
        or health.health_state == "HIGH_DEVIATION"
    )
    early = (
        statistical_state in {"EARLY_DRIFT", "SIGNIFICANT_DRIFT"}
        or health.health_state == "EARLY_DRIFT"
        or (drift is not None and drift.persistent_drift_detected)
    )
    watch = (
        statistical_state not in {"NOMINAL"}
        or fusion.state == "WATCH"
        or health.health_state == "WATCH"
        or drift_status == "DRIFT_SIGNAL"
    )

    if high:
        state = "HIGH_DEVIATION"
    elif early:
        state = "EARLY_DRIFT"
    elif watch:
        state = "WATCH"
    else:
        state = "NOMINAL"

    confidence_terms = [fusion.confidence, health.confidence_score / 100.0]
    if drift is not None:
        confidence_terms.append(
            (
                1.0
                if drift.persistent_drift_detected
                else 0.75
                if drift.latest.status == "DRIFT_SIGNAL"
                else 0.5
            )
        )
    confidence = float(np.clip(np.mean(confidence_terms), 0.0, 1.0))

    if not rationale:
        rationale.append("all available evidence remains within nominal bounds")

    result = TwinStateDecision(
        state=state,
        confidence=confidence,
        evidence=tuple(evidence),
        rationale="; ".join(rationale),
    )
    result.validate()
    return result
