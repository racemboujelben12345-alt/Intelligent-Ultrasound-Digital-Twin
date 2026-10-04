"""Controlled, reproducible degradation registry for the ultrasound Digital Twin.

The registry makes every virtual perturbation auditable:
parent acquisition -> scenario -> severity -> seed -> simulator version -> output hash.

Simulation records are engineering evidence, not proof of physical device failure.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Iterable

import numpy as np

from .degradation import DegradationResult, DegradationType, apply_degradation
from .scenarios import SimulationScenario


@dataclass(frozen=True)
class SimulationRecord:
    """Immutable lineage record for one virtual degradation."""

    simulation_id: str
    parent_acquisition_id: str
    scenario_name: str
    degradation_type: str
    severity: float
    seed: int | None
    simulation_version: str
    output_sha256: str

    def validate(self) -> None:
        if not self.simulation_id:
            raise ValueError("simulation_id is required.")
        if not self.parent_acquisition_id:
            raise ValueError("parent_acquisition_id is required.")
        if not self.scenario_name:
            raise ValueError("scenario_name is required.")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1].")
        if not self.simulation_version:
            raise ValueError("simulation_version is required.")
        if len(self.output_sha256) != 64:
            raise ValueError("output_sha256 must be a SHA-256 hex digest.")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


class DegradationRegistry:
    """Auditable registry of controlled virtual perturbations."""

    def __init__(self) -> None:
        self.records: list[SimulationRecord] = []

    @staticmethod
    def image_hash(image: np.ndarray) -> str:
        array = np.ascontiguousarray(
            np.asarray(image, dtype=np.float32)
        )
        return sha256(array.tobytes()).hexdigest()

    @staticmethod
    def simulation_id(
        parent_acquisition_id: str,
        scenario_name: str,
        severity: float,
        seed: int | None,
        simulation_version: str,
    ) -> str:
        payload = (
            f"{parent_acquisition_id}|{scenario_name}|"
            f"{severity:.8f}|{seed}|{simulation_version}"
        ).encode("utf-8")
        return sha256(payload).hexdigest()[:16]

    def apply(
        self,
        *,
        image: np.ndarray,
        parent_acquisition_id: str,
        scenario: SimulationScenario,
        severity: float,
        seed: int | None,
    ) -> tuple[DegradationResult, SimulationRecord]:
        if not parent_acquisition_id:
            raise ValueError("parent_acquisition_id is required.")
        scenario.validate()
        if severity not in scenario.severities:
            raise ValueError(
                f"Severity {severity} is not registered in scenario "
                f"{scenario.name!r}."
            )

        result = apply_degradation(
            image=image,
            degradation_type=scenario.degradation_type,
            severity=severity,
            seed=seed,
        )

        record = SimulationRecord(
            simulation_id=self.simulation_id(
                parent_acquisition_id,
                scenario.name,
                severity,
                seed,
                result.simulation_version,
            ),
            parent_acquisition_id=parent_acquisition_id,
            scenario_name=scenario.name,
            degradation_type=result.degradation_type,
            severity=result.severity,
            seed=result.seed,
            simulation_version=result.simulation_version,
            output_sha256=self.image_hash(result.image),
        )
        record.validate()
        self.records.append(record)
        return result, record

    def validate(self) -> None:
        for record in self.records:
            record.validate()

        ids = [record.simulation_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate simulation_id detected.")

    def to_dicts(self) -> list[dict]:
        self.validate()
        return [record.to_dict() for record in self.records]

    def records_for_parent(
        self,
        parent_acquisition_id: str,
    ) -> tuple[SimulationRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.parent_acquisition_id == parent_acquisition_id
        )
