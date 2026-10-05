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


def test_lineage_aware_partition_has_disjoint_families():
    from src.validation.audit import partition_by_lineage

    def acq(identifier, parent=None):
        return Acquisition(
            id=identifier,
            source="simulated",
            t=0,
            image=np.zeros((8, 8), dtype=np.float32),
            simulation_scenario="test",
            simulation_severity=0.0,
            parent_acquisition_id=parent or identifier,
            simulation_seed=1,
            simulation_version="1.0",
        )

    items = tuple(acq(f"a{i}") for i in range(12))
    baseline, holdout, test = partition_by_lineage(
        items, baseline_size=5, holdout_size=3
    )
    roots = [
        {item.parent_acquisition_id for item in part}
        for part in (baseline, holdout, test)
    ]
    assert len(baseline) >= 5
    assert len(holdout) >= 3
    assert test
    assert not roots[0] & roots[1]
    assert not roots[0] & roots[2]
    assert not roots[1] & roots[2]


def test_lineage_aware_partition_rejects_unsplittable_family():
    from src.validation.audit import partition_by_lineage

    items = tuple(
        Acquisition(
            id=f"derived_{i}",
            source="simulated",
            t=i,
            image=np.zeros((8, 8), dtype=np.float32),
            simulation_scenario="blur",
            simulation_severity=0.5,
            parent_acquisition_id="same_parent",
            simulation_seed=i,
            simulation_version="1.0",
        )
        for i in range(8)
    )
    with pytest.raises(ValueError, match="independent lineage groups"):
        partition_by_lineage(items, baseline_size=4, holdout_size=2)
