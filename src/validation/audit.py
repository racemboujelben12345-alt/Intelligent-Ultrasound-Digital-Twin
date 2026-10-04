"""Automated verification and validation helpers for the Intelligent Ultrasound Digital Twin.

The audit is intentionally source-aware. It checks structural integrity, provenance,
simulation reproducibility and parent-lineage constraints without pretending that
synthetic perturbations are physical failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.acquisition.models import Acquisition
from src.simulation.degradation import DegradationType, apply_degradation


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class AuditReport:
    checks: tuple[AuditCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[AuditCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(check.passed for check in self.checks)
        return f"V&V audit: {passed}/{total} checks passed"


def audit_acquisitions(acquisitions: Iterable[Acquisition]) -> AuditReport:
    """Run deterministic structural/provenance checks over acquisitions."""
    items = tuple(acquisitions)
    checks: list[AuditCheck] = []

    try:
        for acquisition in items:
            acquisition.validate()
        checks.append(AuditCheck(
            "acquisition_contract",
            True,
            f"{len(items)} acquisitions satisfy the canonical contract.",
        ))
    except Exception as exc:
        checks.append(AuditCheck("acquisition_contract", False, str(exc)))

    try:
        ids = [item.id for item in items]
        unique = len(ids) == len(set(ids))
        checks.append(AuditCheck(
            "unique_acquisition_ids",
            unique,
            "IDs are unique." if unique else "Duplicate acquisition IDs detected.",
        ))
    except Exception as exc:
        checks.append(AuditCheck("unique_acquisition_ids", False, str(exc)))

    try:
        simulated = [item for item in items if item.is_simulated]
        valid_lineage = all(
            item.parent_acquisition_id
            and item.simulation_seed is not None
            and item.simulation_version
            for item in simulated
        )
        checks.append(AuditCheck(
            "simulation_lineage",
            valid_lineage,
            f"{len(simulated)} simulated acquisitions checked.",
        ))
    except Exception as exc:
        checks.append(AuditCheck("simulation_lineage", False, str(exc)))

    return AuditReport(tuple(checks))


def verify_reproducibility(
    image: np.ndarray,
    degradation_type: DegradationType | str,
    severity: float,
    seed: int,
) -> AuditCheck:
    """Verify that a seeded degradation is exactly reproducible."""
    try:
        first = apply_degradation(image, degradation_type, severity, seed)
        second = apply_degradation(image, degradation_type, severity, seed)
        passed = np.array_equal(first.image, second.image)
        details = (
            "Identical outputs for identical input/seed."
            if passed else
            "Outputs differ despite identical input/seed."
        )
        return AuditCheck("simulation_reproducibility", passed, details)
    except Exception as exc:
        return AuditCheck("simulation_reproducibility", False, str(exc))


def verify_severity_response(
    image: np.ndarray,
    degradation_type: DegradationType | str,
    severities: Iterable[float],
    seed: int = 42,
) -> AuditCheck:
    """Verify that increasing severity produces a non-trivial image response.

    This is a sensitivity check, not a claim that every metric must be
    monotonically increasing for every degradation family.
    """
    try:
        levels = tuple(float(x) for x in severities)
        outputs = [
            apply_degradation(image, degradation_type, level, seed).image
            for level in levels
        ]
        deltas = [
            float(np.mean(np.abs(outputs[i] - outputs[0])))
            for i in range(len(outputs))
        ]
        non_trivial = all(delta >= -1e-12 for delta in deltas)
        progressive = any(delta > 1e-5 for delta in deltas[1:])
        passed = non_trivial and progressive
        return AuditCheck(
            "severity_sensitivity",
            passed,
            f"severity levels={len(levels)}, max image delta={max(deltas):.6f}",
        )
    except Exception as exc:
        return AuditCheck("severity_sensitivity", False, str(exc))
