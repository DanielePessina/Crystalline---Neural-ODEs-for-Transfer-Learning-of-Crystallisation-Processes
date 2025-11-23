"""Tests for :func:`simulateCrystallisation`."""

import random

import pytest
from sklearn.preprocessing import MinMaxScaler

from Crystalline.data_functions import simulateCrystallisation

jax = pytest.importorskip("jax")
pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("diffrax")
pytest.importorskip("equinox")
pytest.importorskip("optax")

## This test checks the shapes of the output from `simulateCrystallisation`


@pytest.mark.parametrize("n_experiments, ntimesteps", [(2, 50), (1, 14)])
def test_simulate_crystallisation_shapes(n_experiments, ntimesteps):
    key = jax.random.PRNGKey(0)
    concentrations = [random.uniform(13, 20) for _ in range(n_experiments)]
    scaler = None  # if n_experiments == 2 else MinMaxScaler().fit([[0.0], [1.0]])

    ts, ys, returned_scaler = simulateCrystallisation(
        list_initialconcentrations=concentrations,
        ntimesteps=ntimesteps,
        scaler=scaler,
        key=key,
        nucl_params=[35.0, 1.0],
        growth_params=[0.4, 2.5],
        save_idxs=[0, 3],
        noise=False,
        masked=False,
    )

    assert ts.shape == (ntimesteps,)
    assert ys.shape == (n_experiments, ntimesteps, 2)
    if scaler is None:
        assert isinstance(returned_scaler, MinMaxScaler)
    else:
        assert returned_scaler is scaler
