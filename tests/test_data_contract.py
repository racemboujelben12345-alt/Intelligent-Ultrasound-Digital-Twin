import numpy as np

from src.acquisition.models import Acquisition
from src.acquisition.provenance import DataProvenance, DataSource
from src.simulation.degradation import DegradationType, apply_degradation


def test_public_provenance_is_source_separated():
    provenance = DataProvenance(
        source=DataSource.PUBLIC,
        dataset="kaggle_ultrasound",
        relative_path="sample.png",
        is_public_reference=True,
    )
    acquisition = Acquisition(
        id="public_00001",
        source="public",
        t=0,
        image=np.zeros((16, 16), dtype=np.float32),
        provenance=provenance,
    )
    acquisition.validate()
    assert acquisition.is_public_reference


def test_simulation_is_reproducible_and_versioned():
    image = np.full((32, 32), 0.5, dtype=np.float32)

    first = apply_degradation(
        image,
        DegradationType.GAUSSIAN_NOISE,
        severity=0.5,
        seed=123,
    )
    second = apply_degradation(
        image,
        DegradationType.GAUSSIAN_NOISE,
        severity=0.5,
        seed=123,
    )

    assert np.array_equal(first.image, second.image)
    assert first.seed == 123
    assert first.simulation_version


def test_simulation_lineage_is_required():
    image = np.zeros((16, 16), dtype=np.float32)
    provenance = DataProvenance(
        source=DataSource.SIMULATED,
        dataset="simulation",
        relative_path="synthetic.png",
        is_simulation=True,
    )

    acquisition = Acquisition(
        id="synthetic_00001",
        source="simulated",
        t=0,
        image=image,
        provenance=provenance,
        simulation_scenario="blur",
        simulation_severity=0.5,
        parent_acquisition_id="public_00001",
        simulation_seed=42,
        simulation_version="1.0",
    )
    acquisition.validate()
