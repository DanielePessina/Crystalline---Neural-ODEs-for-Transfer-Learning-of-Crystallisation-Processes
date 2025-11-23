import importlib.util
from pathlib import Path

import jax.random as jr
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "calculations", Path(__file__).resolve().parents[1] / "metrics" / "calculations.py"
)
calculations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calculations)

calculate_all_metrics = calculations.calculate_all_metrics
calculate_all_metrics_latent = calculations.calculate_all_metrics_latent


def _build_model(ys):
    def model(ts, y0):
        if np.allclose(y0, ys[0, 0]):
            return ys[0]
        return ys[1]

    return model


@pytest.mark.parametrize("mask_d43", [False, True])
def test_calculate_all_metrics_identity(mask_d43):
    ts = np.array([0.0, 1.0])
    ys = np.array(
        [
            [[1.0, 2.0], [2.0, 3.0]],
            [[3.0, 4.0], [4.0, 5.0]],
        ]
    )
    model = _build_model(ys)
    metrics = calculate_all_metrics(
        ts,
        ys,
        model,
        None,
        [1.0, 3.0],
        "Test",
        ntimesteps=2,
        noise_level=0.0,
        mask_d43=mask_d43,
    )
    assert len(metrics) == 2
    for m in metrics:
        assert m["MAE"] == 0.0
        assert m["Concentration_MAE"] == 0.0
        if not mask_d43:
            assert m["D43_MAE"] == 0.0


jax = pytest.importorskip("jax")


def test_calculate_all_metrics_latent_identity():
    ts = np.array([0.0, 1.0])
    ys = np.array(
        [
            [[1.0, 2.0], [2.0, 3.0]],
            [[3.0, 4.0], [4.0, 5.0]],
        ]
    )

    def latent_model(ts, y, key):
        return y, None

    metrics = calculate_all_metrics_latent(
        ts,
        ys,
        latent_model,
        jr.PRNGKey(0),
        None,
        [1.0, 3.0],
        "Test",
        ntimesteps=2,
        noise_level=0.0,
    )
    assert len(metrics) == 2
    for m in metrics:
        assert m["MAE"] == 0.0
        assert m["Concentration_MAE"] == 0.0
        assert m["D43_MAE"] == 0.0
