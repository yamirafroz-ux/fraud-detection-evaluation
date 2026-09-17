import numpy as np
import pytest

from fluxfraud.config import Config
from fluxfraud.processes import gibbs_probabilities, hawkes_interval, ou_step


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(branching_ratio=1),
        dict(temperature=0),
        dict(base_rate=float("nan")),
        dict(seed=-1),
        dict(days=1),
        dict(customers=3),
        dict(days=12.5),
        dict(label_delay_hours=1000),
        dict(seed=True),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_ou_exact_moments():
    rng = np.random.default_rng(11)
    x = ou_step(np.full(100000, 2.0), 1, 0.5, 0.4, 3, rng)
    assert abs(x.mean() - (1 + np.exp(-1.5))) < 0.004
    assert abs(x.var() - 0.16 * (1 - np.exp(-3))) < 0.003


def test_poisson_limit_and_hawkes_mean():
    rng = np.random.default_rng(19)
    poisson = np.array([len(hawkes_interval(2, 0, 0, 1, 10, rng)[0]) for _ in range(1500)])
    assert abs(poisson.mean() - 20) < 0.5
    assert abs(poisson.var() - 20) < 2
    events, _ = hawkes_interval(2, 0, 0.4, 2, 12000, rng)
    assert abs(len(events) / 12000 - 2 / (1 - 0.4)) < 0.13


def test_hawkes_residual_and_safety():
    events, residual = hawkes_interval(0, 0.00001, 0, 2, 1, np.random.default_rng(2))
    assert len(events) == 0
    assert residual == pytest.approx(0.00001 * np.exp(-2))
    with pytest.raises(RuntimeError):
        hawkes_interval(100, 0, 0.4, 2, 10, np.random.default_rng(1), max_events=2)


def test_gibbs_temperature_limits():
    assert np.allclose(gibbs_probabilities([0, 1, 2], 1e9), np.full(3, 1 / 3))
    assert gibbs_probabilities([0, 1, 2], 0.001)[0] == pytest.approx(1)
    assert np.isfinite(gibbs_probabilities([1e8, 1e8 + 1], 0.1)).all()


@pytest.mark.parametrize("x", [float("nan"), float("inf")])
def test_ou_rejects_nonfinite(x):
    with pytest.raises(ValueError):
        ou_step(x, 0, 1, 1, 1, np.random.default_rng(1))


def test_gibbs_rejects_empty():
    with pytest.raises(ValueError):
        gibbs_probabilities([], 1)
