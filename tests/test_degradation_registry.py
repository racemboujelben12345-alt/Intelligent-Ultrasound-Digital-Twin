import numpy as np
import pytest

from src.simulation.degradation import DegradationType, apply_degradation
from src.simulation.registry import DegradationRegistry
from src.simulation.scenarios import build_scenario, ScenarioType


def test_registry_preserves_parent_lineage_and_hash():
    image = np.full((32, 32), 0.5, dtype=np.float32)
    scenario = build_scenario(
        ScenarioType.BLUR_PROGRESSION,
        severities=(0.0, 0.5, 1.0),
    )
    registry = DegradationRegistry()

    result, record = registry.apply(
        image=image,
        parent_acquisition_id="public-001",
        scenario=scenario,
        severity=0.5,
        seed=42,
    )

    assert record.parent_acquisition_id == "public-001"
    assert record.degradation_type == DegradationType.BLUR.value
    assert record.severity == 0.5
    assert len(record.output_sha256) == 64
    assert registry.records_for_parent("public-001") == (record,)
    result.validate()


def test_same_lineage_and_seed_are_reproducible():
    image = np.random.default_rng(7).random((32, 32), dtype=np.float32)
    scenario = build_scenario(
        ScenarioType.NOISE_PROGRESSION,
        severities=(0.4,),
    )

    registry = DegradationRegistry()
    a, ra = registry.apply(
        image=image,
        parent_acquisition_id="public-001",
        scenario=scenario,
        severity=0.4,
        seed=123,
    )
    b = apply_degradation(
        image=image,
        degradation_type=scenario.degradation_type,
        severity=0.4,
        seed=123,
    )

    assert np.array_equal(a.image, b.image)
    assert ra.output_sha256 == registry.image_hash(b.image)


def test_registry_rejects_unregistered_severity():
    image = np.zeros((16, 16), dtype=np.float32)
    scenario = build_scenario(
        ScenarioType.SPECKLE_PROGRESSION,
        severities=(0.0, 0.5),
    )

    with pytest.raises(ValueError, match="not registered"):
        DegradationRegistry().apply(
            image=image,
            parent_acquisition_id="x",
            scenario=scenario,
            severity=0.7,
            seed=1,
        )
