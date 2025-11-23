"""Simulation runner utilities for the Dash app.

This module contains functions to parse user inputs from the Dash interface
and convert them into TrainConfig objects, as well as utilities to build
graphs and tables from training results.
"""

import re

import jax.nn as jnn
import jax.numpy as jnp
from dash import dcc
from postproc import summarize_model
from training_wrapper import TLConfig, TrainConfig

# Activation function mapping
_ACT_MAP = {
    "swish": jnn.swish,
    "relu": jnn.relu,
    "tanh": jnp.tanh,
    "sigmoid": jnn.sigmoid,
    "elu": jnn.elu,
    "gelu": jnn.gelu,
}


class ProgressAggregator:
    """Aggregates progress messages and throttles UI updates."""

    def __init__(self, push_fn, min_interval=0.5):
        """Initialize progress aggregator.

        Parameters
        ----------
        push_fn : callable
            Function to call with progress updates.
        min_interval : float
            Minimum time interval between updates in seconds.
        """
        self.buf = []
        self.push_fn = push_fn
        self._last = 0.0
        self._dt = min_interval

    def __call__(self, msg: str):
        """Add a message to the buffer and potentially push update."""
        import time

        self.buf.append(msg)
        t = time.time()
        if t - self._last >= self._dt:
            self.push_fn(("".join(self.buf),))
            self._last = t


def _floats(s: str) -> list[float]:
    """Parse comma/space-separated string into list of floats."""
    if not s or not s.strip():
        return []
    return [float(x) for x in re.split(r"[,\s]+", s.strip()) if x]


def parse_inputs_to_cfg(
    nucl_A,
    nucl_b,
    growth_k,
    growth_g,
    base_inits,
    test_inits,
    noise_level,
    width_size,
    depth,
    ntimesteps,
    seed,
    activation,
    lr_low,
    lr_high,
    steps1,
    steps2,
    len1,
    len2,
    constraints,
    include_time,
) -> TrainConfig:
    """Parse Dash input state into TrainConfig object."""
    # Parse constraints
    cons_map = {}
    if constraints:
        if "neg0" in constraints:
            cons_map[0] = "neg"
        if "pos1" in constraints:
            cons_map[1] = "pos"

    # Parse activation function
    act_func = _ACT_MAP.get(activation, jnn.swish)

    return TrainConfig(
        base_inits=_floats(base_inits or ""),
        test_inits=_floats(test_inits or ""),
        nucl_params=(float(nucl_A or 39.81), float(nucl_b or 0.675)),
        growth_params=(float(growth_k or 0.345), float(growth_g or 3.344)),
        lr_strategy=(float(lr_low or 4e-3), float(lr_high or 10e-3)),
        steps_strategy=(int(steps1 or 600), int(steps2 or 600)),
        length_strategy=(float(len1 or 0.33), float(len2 or 1.0)),
        width_size=int(width_size or 100),
        depth=int(depth or 4),
        activation=act_func,
        ntimesteps=int(ntimesteps or 25),
        seed=int(seed or 467),
        noise=True,
        noise_level=float(noise_level or 0.1),
        output_constraints=cons_map,
        include_time=bool(include_time),
        batch_size="all",
        augment_dim=2,
        splitplot=False,
        make_plots=True,
    )


def build_graphs_and_table(res) -> tuple[list, list, list]:
    """Build graphs and table from training results.

    Parameters
    ----------
    res : TrainResult
        Training results object.

    Returns
    -------
    tuple
        (graphs, columns, data) where:
        - graphs: list of dcc.Graph components
        - columns: list of column definitions for DataTable
        - data: list of data records for DataTable
    """
    # Build graphs from figures
    graphs = [dcc.Graph(figure=f) for f in (res.figures or [])]

    # Build table from model summary
    df = summarize_model(res.model)
    columns = [{"name": c, "id": c} for c in df.columns]
    data = df.to_dict("records")

    return graphs, columns, data


def validate_inputs(base_inits: str, test_inits: str) -> tuple[bool, str]:
    """Validate user inputs before training.

    Parameters
    ----------
    base_inits, test_inits : str
        Comma-separated strings of initial concentrations.

    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    try:
        base_vals = _floats(base_inits)
        test_vals = _floats(test_inits)

        if not base_vals:
            return False, "Base initial concentrations cannot be empty"

        if not test_vals:
            return False, "Test initial concentrations cannot be empty"

        if any(x <= 0 for x in base_vals):
            return False, "Base initial concentrations must be positive"

        if any(x <= 0 for x in test_vals):
            return False, "Test initial concentrations must be positive"

        return True, ""

    except ValueError as e:
        return False, f"Invalid number format: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_default_values() -> dict:
    """Get default values for all input fields.

    Returns
    -------
    dict
        Dictionary mapping field IDs to default values.
    """
    return {
        "nucl-A": 39.81,
        "nucl-b": 0.675,
        "growth-k": 0.345,
        "growth-g": 3.344,
        "base-inits": "13.0, 15.5, 16.5, 19.0",
        "test-inits": "12.0, 14.3, 17.7, 20.0",
        "noise-level": 0.05,
        "width-size": 64,
        "depth": 4,
        "ntimesteps": 25,
        "seed": 467,
        "activation": "swish",
        "lr-low": 4e-4,
        "lr-high": 6e-4,
        "steps1": 600,
        "steps2": 600,
        "len1": 0.33,
        "len2": 1.0,
        "constraints": None,
        "include_time": True,
    }


def get_default_tl_values() -> dict:
    """Default values for the TL tab based on augmented.py __main__ examples.

    Uses TL-friendly learning rates and strategies and defaults to refitting the scaler,
    freezing the last layer with both weights and biases.
    """
    return {
        "nucl-A": 39.81,
        "nucl-b": 0.675,
        "growth-k": 0.345,
        "growth-g": 3.344,
        "base-inits": "16.5, 17.0",
        "test-inits": "14.3, 17.7, 20.0",
        "noise-level": 0.05,
        "ntimesteps": 25,
        "seed": 4700,
        "lr-low": 6e-4,
        "lr-high": 6e-4,
        "steps1": 600,
        "steps2": 600,
        "len1": 0.33,
        "len2": 1.0,
        "scale-strategy": "refit_scaler",
        "idx-frozen": "last",
        "freeze-mode": "both",
        "penalise": False,
        "penalty-lambda": 1.0,
        "penalty-strategy": "all",
    }


def parse_tl_inputs_to_cfg(
    nucl_A,
    nucl_b,
    growth_k,
    growth_g,
    base_inits,
    test_inits,
    noise_level,
    ntimesteps,
    seed,
    lr_low,
    lr_high,
    steps1,
    steps2,
    len1,
    len2,
    scale_strategy,
    idx_frozen,
    freeze_mode,
    penalise,
    penalty_lambda,
    penalty_strategy,
) -> TLConfig:
    """Parse Dash TL inputs into a TLConfig object."""

    return TLConfig(
        base_inits=_floats(base_inits or ""),
        test_inits=_floats(test_inits or ""),
        nucl_params=(float(nucl_A or 39.81), float(nucl_b or 0.675)),
        growth_params=(float(growth_k or 0.345), float(growth_g or 3.344)),
        lr_strategy=(float(lr_low or 6e-4), float(lr_high or 6e-4)),
        steps_strategy=(int(steps1 or 600), int(steps2 or 600)),
        length_strategy=(float(len1 or 0.33), float(len2 or 1.0)),
        ntimesteps=int(ntimesteps or 25),
        seed=int(seed or 4700),
        noise=True,
        noise_level=float(noise_level or 0.1),
        scale_strategy=str(scale_strategy or "refit_scaler"),
        idx_frozen=idx_frozen or "last",
        freeze_mode=freeze_mode or "both",
        penalise_deviations=bool(penalise),
        penalty_lambda=float(penalty_lambda or 1.0),
        penalty_strategy=penalty_strategy or "all",
    )
