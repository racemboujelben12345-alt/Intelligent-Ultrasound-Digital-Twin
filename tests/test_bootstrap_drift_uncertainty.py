import pytest

from src.validation.bootstrap_drift_uncertainty import bootstrap_mean_interval


def test_bootstrap_interval_is_reproducible_and_contains_estimate():
    values = (0.0, 1.0, 2.0, 3.0, 4.0)
    first = bootstrap_mean_interval(values, seed=17, n_resamples=500)
    second = bootstrap_mean_interval(values, seed=17, n_resamples=500)
    assert first == second
    assert first.estimate == pytest.approx(2.0)
    assert first.lower <= first.estimate <= first.upper
    assert first.sample_size == 5
    assert first.to_dict()["seed"] == 17


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": (1.0,)},
        {"values": (1.0, float("nan"))},
        {"values": (1.0, 2.0), "confidence_level": 1.0},
        {"values": (1.0, 2.0), "n_resamples": 99},
        {"values": (1.0, 2.0), "seed": True},
    ],
)
def test_bootstrap_interval_rejects_invalid_inputs(kwargs):
    values = kwargs.pop("values")
    with pytest.raises(ValueError):
        bootstrap_mean_interval(values, **kwargs)


def test_constant_values_have_degenerate_interval():
    result = bootstrap_mean_interval((3.0, 3.0, 3.0), n_resamples=200)
    assert result.estimate == result.lower == result.upper == 3.0
