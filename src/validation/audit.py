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
        # Generic degradations are not required to be monotonic in every
        # image metric. V&V checks measurable sensitivity instead.
        measurable = [delta for delta in deltas[1:] if delta > 1e-5]
        passed = bool(measurable)
        return AuditCheck(
            "severity_sensitivity",
            passed,
            f"severity levels={len(levels)}, max image delta={max(deltas):.6f}",
        )
    except Exception as exc:
        return AuditCheck("severity_sensitivity", False, str(exc))


def verify_no_parent_leakage(
    acquisitions: Iterable[Acquisition],
    train_ids: set[str],
    holdout_ids: set[str],
) -> AuditCheck:
    """Detect parent/derived-family overlap between train and holdout.

    A synthetic derivative belongs to the same information family as its
    parent. Therefore splitting only by image ID can leak information.
    """
    try:
        items = tuple(acquisitions)

        def root_id(item: Acquisition) -> str:
            return item.parent_acquisition_id or item.id

        train_roots = {
            root_id(item) for item in items if item.id in train_ids
        }
        holdout_roots = {
            root_id(item) for item in items if item.id in holdout_ids
        }

        overlap = train_roots & holdout_roots
        passed = not overlap

        return AuditCheck(
            "parent_group_leakage",
            passed,
            "No parent-family overlap detected."
            if passed
            else f"Leakage detected in parent families: {sorted(overlap)}",
        )
    except Exception as exc:
        return AuditCheck("parent_group_leakage", False, str(exc))

def lineage_root_id(acquisition: Acquisition) -> str:
    """Return the stable information-family identifier for an acquisition."""
    return acquisition.parent_acquisition_id or acquisition.id


def partition_by_lineage(
    acquisitions: Iterable[Acquisition],
    *,
    baseline_size: int,
    holdout_size: int,
) -> tuple[tuple[Acquisition, ...], tuple[Acquisition, ...], tuple[Acquisition, ...]]:
    """Split acquisitions into baseline/holdout/test without parent-family leakage.

    Whole lineage families are kept in one partition. If the requested partition
    cannot be formed without splitting a family, the function fails explicitly
    instead of silently introducing leakage.
    """
    items = tuple(acquisitions)
    if baseline_size < 1 or holdout_size < 1:
        raise ValueError("baseline_size and holdout_size must be >= 1.")

    groups: dict[str, list[Acquisition]] = {}
    for item in items:
        item.validate()
        groups.setdefault(lineage_root_id(item), []).append(item)

    ordered_groups = [tuple(groups[key]) for key in sorted(groups)]
    baseline: list[Acquisition] = []
    holdout: list[Acquisition] = []
    test: list[Acquisition] = []

    for group in ordered_groups:
        if len(baseline) < baseline_size:
            baseline.extend(group)
        elif len(holdout) < holdout_size:
            holdout.extend(group)
        else:
            test.extend(group)

    if len(baseline) < baseline_size:
        raise ValueError(
            "Insufficient independent lineage groups for the requested baseline."
        )
    if len(holdout) < holdout_size:
        raise ValueError(
            "Insufficient independent lineage groups for the requested holdout."
        )
    if not test:
        raise ValueError(
            "No independent lineage group remains for the test partition."
        )

    # Enforce disjoint roots as a final invariant.
    partitions = (baseline, holdout, test)
    roots = [{lineage_root_id(item) for item in part} for part in partitions]
    if roots[0] & roots[1] or roots[0] & roots[2] or roots[1] & roots[2]:
        raise RuntimeError("Lineage leakage detected during partitioning.")

    return tuple(baseline), tuple(holdout), tuple(test)
