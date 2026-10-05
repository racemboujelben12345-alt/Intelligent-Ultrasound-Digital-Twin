"""Automated verification and validation helpers for the Intelligent Ultrasound Digital Twin.

The audit is source-aware and lineage-aware. Synthetic derivatives are treated as
members of the same information family as their ultimate parent.
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


def lineage_root_id(
    acquisition: Acquisition,
    by_id: dict[str, Acquisition] | None = None,
) -> str:
    """Resolve the ultimate information-family root.

    A self-referencing parent ID represents a root acquisition and is not
    considered a lineage cycle.

    Missing parents are represented by their parent ID.
    Genuine multi-acquisition cycles are rejected.
    """
    if by_id is None:
        parent_id = acquisition.parent_acquisition_id

        if not parent_id or parent_id == acquisition.id:
            return acquisition.id

        return parent_id

    current = acquisition
    seen: set[str] = set()

    while current.parent_acquisition_id:
        parent_id = current.parent_acquisition_id

        # A self-reference identifies a root acquisition.
        if parent_id == current.id:
            return current.id

        # Detect genuine lineage cycles.
        if current.id in seen:
            raise ValueError(
                f"Simulation lineage cycle detected at {current.id!r}."
            )

        seen.add(current.id)

        parent = by_id.get(parent_id)

        # Parent is not present in the current collection.
        if parent is None:
            return parent_id

        current = parent

    return current.id


def audit_acquisitions(
    acquisitions: Iterable[Acquisition],
) -> AuditReport:
    """Run deterministic structural, provenance and lineage checks."""
    items = tuple(acquisitions)
    checks: list[AuditCheck] = []

    try:
        for acquisition in items:
            acquisition.validate()

        checks.append(
            AuditCheck(
                "acquisition_contract",
                True,
                f"{len(items)} acquisitions satisfy the canonical contract.",
            )
        )
    except Exception as exc:
        checks.append(
            AuditCheck(
                "acquisition_contract",
                False,
                str(exc),
            )
        )

    try:
        ids = [item.id for item in items]
        unique = len(ids) == len(set(ids))

        checks.append(
            AuditCheck(
                "unique_acquisition_ids",
                unique,
                (
                    "IDs are unique."
                    if unique
                    else "Duplicate acquisition IDs detected."
                ),
            )
        )
    except Exception as exc:
        checks.append(
            AuditCheck(
                "unique_acquisition_ids",
                False,
                str(exc),
            )
        )

    try:
        simulated = [item for item in items if item.is_simulated]
        by_id = {item.id: item for item in items}

        for item in simulated:
            root = lineage_root_id(item, by_id)

            if not item.parent_acquisition_id or not item.simulation_version:
                raise ValueError(
                    f"Incomplete lineage metadata for {item.id!r}."
                )

            if root == item.id:
                raise ValueError(
                    f"Simulation {item.id!r} resolves to itself."
                )

        checks.append(
            AuditCheck(
                "simulation_lineage",
                True,
                f"{len(simulated)} simulated acquisitions have valid lineage.",
            )
        )

    except Exception as exc:
        checks.append(
            AuditCheck(
                "simulation_lineage",
                False,
                str(exc),
            )
        )

    return AuditReport(tuple(checks))


def verify_reproducibility(
    image: np.ndarray,
    degradation_type: DegradationType | str,
    severity: float,
    seed: int,
) -> AuditCheck:
    """Verify exact reproducibility for a seeded degradation."""
    try:
        first = apply_degradation(
            image,
            degradation_type,
            severity,
            seed,
        )

        second = apply_degradation(
            image,
            degradation_type,
            severity,
            seed,
        )

        passed = np.array_equal(
            first.image,
            second.image,
        )

        details = (
            "Identical outputs for identical input/seed."
            if passed
            else "Outputs differ despite identical input/seed."
        )

        return AuditCheck(
            "simulation_reproducibility",
            passed,
            details,
        )

    except Exception as exc:
        return AuditCheck(
            "simulation_reproducibility",
            False,
            str(exc),
        )


def verify_severity_response(
    image: np.ndarray,
    degradation_type: DegradationType | str,
    severities: Iterable[float],
    seed: int = 42,
) -> AuditCheck:
    """Verify measurable sensitivity; monotonicity is not assumed."""
    try:
        levels = tuple(float(x) for x in severities)

        if len(levels) < 2:
            raise ValueError(
                "At least two severity levels are required."
            )

        outputs = [
            apply_degradation(
                image,
                degradation_type,
                level,
                seed,
            ).image
            for level in levels
        ]

        deltas = [
            float(
                np.mean(
                    np.abs(outputs[i] - outputs[0])
                )
            )
            for i in range(len(outputs))
        ]

        measurable = any(
            delta > 1e-5
            for delta in deltas[1:]
        )

        return AuditCheck(
            "severity_sensitivity",
            measurable,
            (
                f"severity levels={len(levels)}, "
                f"max image delta={max(deltas):.6f}"
            ),
        )

    except Exception as exc:
        return AuditCheck(
            "severity_sensitivity",
            False,
            str(exc),
        )


def verify_no_parent_leakage(
    acquisitions: Iterable[Acquisition],
    train_ids: set[str],
    holdout_ids: set[str],
) -> AuditCheck:
    """Detect overlap of ultimate information families between partitions."""
    try:
        items = tuple(acquisitions)
        by_id = {item.id: item for item in items}

        train_roots = {
            lineage_root_id(item, by_id)
            for item in items
            if item.id in train_ids
        }

        holdout_roots = {
            lineage_root_id(item, by_id)
            for item in items
            if item.id in holdout_ids
        }

        overlap = train_roots & holdout_roots
        passed = not overlap

        return AuditCheck(
            "parent_group_leakage",
            passed,
            (
                "No parent-family overlap detected."
                if passed
                else (
                    "Leakage detected in parent families: "
                    f"{sorted(overlap)}"
                )
            ),
        )

    except Exception as exc:
        return AuditCheck(
            "parent_group_leakage",
            False,
            str(exc),
        )


def partition_by_lineage(
    acquisitions: Iterable[Acquisition],
    *,
    baseline_size: int,
    holdout_size: int,
) -> tuple[
    tuple[Acquisition, ...],
    tuple[Acquisition, ...],
    tuple[Acquisition, ...],
]:
    """Partition complete information families without cross-partition leakage."""
    items = tuple(acquisitions)

    if baseline_size < 1 or holdout_size < 1:
        raise ValueError(
            "baseline_size and holdout_size must be >= 1."
        )

    by_id = {item.id: item for item in items}

    groups: dict[str, list[Acquisition]] = {}

    for item in items:
        item.validate()

        root_id = lineage_root_id(
            item,
            by_id,
        )

        groups.setdefault(
            root_id,
            [],
        ).append(item)

    ordered_groups = [
        tuple(groups[key])
        for key in sorted(groups)
    ]

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
            "Insufficient independent lineage groups for baseline."
        )

    if len(holdout) < holdout_size:
        raise ValueError(
            "Insufficient independent lineage groups for holdout."
        )

    if not test:
        raise ValueError(
            "No independent lineage group remains for test."
        )

    roots = [
        {
            lineage_root_id(item, by_id)
            for item in part
        }
        for part in (baseline, holdout, test)
    ]

    if (
        roots[0] & roots[1]
        or roots[0] & roots[2]
        or roots[1] & roots[2]
    ):
        raise RuntimeError(
            "Lineage leakage detected during partitioning."
        )

    return (
        tuple(baseline),
        tuple(holdout),
        tuple(test),
    )