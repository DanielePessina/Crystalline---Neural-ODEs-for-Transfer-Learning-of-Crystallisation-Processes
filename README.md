# Crystalline

Research code for training **Augmented Neural ODE (AugNODE)** models on crystallisation kinetics using JAX, Equinox, and Diffrax. The package includes training utilities for fresh and transfer‑learned AugNODEs plus a Dash dashboard to launch runs and inspect plots interactively.

## Repository layout
- `Crystalline/`: core library (`augmented.py` holds the AugNODE trainers).
- `dashapp/`: Dash dashboard (`app.py`) plus helpers for parsing inputs and rendering results.
- `uv.lock`, `pyproject.toml`: dependency and environment lockfiles (Python 3.12+).

## Installation
The project is managed with [uv](https://github.com/astral-sh/uv).
1) Install uv if needed: `pip install --user uv`
2) From the repo root:
```bash
uv sync
```
This creates the virtual environment and installs all pinned dependencies.

## Run the Dash dashboard
Launch the local dashboard for training and visualisation:
```bash
uv run python dashapp/app.py
```
Open http://127.0.0.1:8050 to:
- configure nucleation/growth parameters and AugNODE hyperparameters,
- run fresh or transfer‑learning training via the UI,
- view plots (train/test trajectories, metrics) and model summaries.

## Python usage (programmatic)
The main training APIs live in `Crystalline/augmented.py`. The constants `C_TRAIN_SOURCE`, `C_TRAIN_TARGET`, and `C_TEST` mirror the examples in the module’s `__main__` block.

```python
import jax.nn as jnn
from Crystalline.augmented import (
    C_TEST,
    C_TRAIN_SOURCE,
    C_TRAIN_TARGET,
    train_AugNODE_fresh,
    train_AugNODE_TL,
)

# 1) Train a fresh AugNODE on simulated crystallisation trajectories
ts, ys_train, ys_test, base_model, scaler, metrics, (aug_min, aug_max) = train_AugNODE_fresh(
    C_TRAIN_SOURCE,
    C_TEST,
    lr_strategy=(2e-4, 4e-4),
    steps_strategy=(600, 600),
    length_strategy=(0.33, 1.0),
    width_size=32,
    depth=4,
    activation=jnn.swish,
    ntimesteps=5,
    seed=4700,
    augment_dim=2,
    batch_size="all",
    noise=True,
    noise_level=0.05,
    splitplot=True,   # optional: matplotlib plots
    saveplot=False,
)

# 2) Transfer‑learn on a new concentration range, keeping late layers fixed
_, _, _, tl_model, _, tl_metrics = train_AugNODE_TL(
    C_TRAIN_TARGET,
    C_TEST,
    base_model,
    scaler,
    idx_frozen="last",     # freeze the final layer
    freeze_mode="both",    # freeze weights and biases of the frozen layers
    lr_strategy=(4e-4, 6e-4),
    steps_strategy=(400, 400),
    length_strategy=(0.5, 1.0),
    ntimesteps=5,
    batch_size="all",
)

print("Fresh RMSE:", [m["RMSE"] for m in metrics if m["Experiment_Tag"] == "Train"])
print("TL RMSE:", [m["RMSE"] for m in tl_metrics if m["Experiment_Tag"] == "Train"])
```

Notes
- Both training functions optionally generate plots (`splitplot=True`) and learning curves (`lossplot=True`).
- `train_AugNODE_TL` accepts additional variants (`idx_frozen`/`freeze_mode`) and scale strategies (`scale_strategy="refit_scaler"` or `None` to reuse the original scaler).
- The helpers in `Crystalline.data_functions` simulate crystallisation trajectories; replace them with your own data loader via `dataloader` if you have experimental data.
