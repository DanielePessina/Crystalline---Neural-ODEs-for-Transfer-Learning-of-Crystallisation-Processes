"""Dash-specific training functions with real-time progress reporting.

This module provides modified versions of the standard training functions
that integrate properly with Dash progress callbacks instead of Rich progress bars.
"""

import copy
import time
from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from jax import Array
from sklearn.preprocessing import MinMaxScaler

from Crystalline.augmented import AugmentedNeuralODE
from Crystalline.data_functions import dataloader, simulateCrystallisation
from Crystalline.metrics.calculations import calculate_all_metrics
from Crystalline.metrics.error import make_failure_metrics


def train_AugNODE_dash_realtime(
    training_initialconcentrations: list[float],
    testing_initialconcentrations: list[float],
    *,
    ntimesteps: int = 85,
    nucl_params: list[float] = [39.81, 0.675],
    growth_params: list[float] = [0.345, 3.344],
    noise: bool = False,
    noise_level: float = 0.1,
    lr_strategy: tuple[float, float] = (4e-3, 10e-3),
    steps_strategy: tuple[int, int] = (600, 600),
    length_strategy: tuple[float, float] = (0.33, 1),
    batch_size: int | str = 1,
    scale_strategy: str | None = None,
    width_size: int = 100,
    depth: int = 4,
    augment_dim: int = 1,
    include_time: bool = True,
    activation: Callable = jnn.swish,
    solver: diffrax.AbstractSolver = diffrax.Tsit5(),
    seed: int = 467,
    print_every: int = 100,
    splitplot: bool = False,
    plotly_plots: bool = False,
    saveplot: bool = False,
    lossplot: bool = False,
    extratitlestring: str = "",
    save_idxs: list[int] = [0, 3],
    output_constraints: dict | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress_updater: Callable | None = None,
) -> tuple[Array, Array, Array, AugmentedNeuralODE, MinMaxScaler | None, list[dict]]:
    """Train an augmented NODE with real-time progress reporting for Dash.

    This function is specifically designed for Dash applications and provides
    real-time progress updates through a callback function instead of Rich progress bars.

    Parameters
    ----------
    training_initialconcentrations, testing_initialconcentrations : list[float]
        Initial solute concentrations for training and testing experiments.
    ntimesteps : int, optional
        Number of discretisation points for the simulated trajectories.
    nucl_params, growth_params : list[float]
        Kinetic parameters for the moment model.
    noise : bool, optional
        If ``True`` Gaussian noise with magnitude ``noise_level`` is added to
        the synthetic data before scaling.
    lr_strategy, steps_strategy, length_strategy : tuple
        Hyperparameter schedules for the two training phases.
    batch_size : int | str
        Batch size for the data loader or ``"all"`` to use full batch.
    scale_strategy : str | None
        ``None`` to fit a new scaler, ``"no_scale"`` to disable scaling.
    width_size, depth, augment_dim : int
        Neural network architecture parameters.
    activation : Callable
        Activation function for the MLP.
    solver : diffrax.AbstractSolver
        ODE solver used during training.
    seed : int
        Random seed for data generation and model initialisation.
    print_every : int
        How often to report progress (in steps).
    progress_callback : Callable[[str], None] | None
        Callback function to receive progress updates for Dash terminal.

    Returns
    -------
    tuple
        ``(ts, ys_train, ys_test, model, scaler, metrics_list)``
    """

    def log(msg: str):
        """Log progress message to callback if available."""
        if progress_callback:
            timestamp = time.strftime("%H:%M:%S")
            progress_callback(f"[{timestamp}] {msg}")

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, split_key = jr.split(key, 4)

    if batch_size == "all":
        batch_size = len(training_initialconcentrations)

    log("Initializing data generation...")

    if scale_strategy is None:
        # Generate training and testing data separately
        ts, ys_train, scaler = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )

        ts_test, ys_test, _ = simulateCrystallisation(
            testing_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )

    elif scale_strategy == "no_scale":
        ts, ys_train, scaler = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            scaler,
            scale_strategy,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )

        ts_test, ys_test, _ = simulateCrystallisation(
            testing_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )

    # Get dimensions from the data
    _, length_size, data_size = ys_train.shape

    log(f"Creating AugNODE model (width={width_size}, depth={depth}, augment_dim={augment_dim})...")

    model = AugmentedNeuralODE(
        data_size,
        augment_dim,
        width_size,
        depth,
        key=model_key,
        activation=activation,
        solver=solver,
        include_time=include_time,
        output_constraints=output_constraints,
    )

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    try:
        train_losses = []
        test_losses = []

        # Calculate total steps for progress tracking
        total_steps = sum(steps_strategy)
        current_step = 0

        log("Starting training phases...")

        for phase_idx, (lr, steps, length) in enumerate(zip(lr_strategy, steps_strategy, length_strategy, strict=True)):
            phase_name = f"Phase {phase_idx + 1}/2"
            log(f"{phase_name}: lr={lr:.2e}, steps={steps}, length={length:.2f}")

            if progress_updater:
                progress_updater(current_step, total_steps, phase_name, None, f"{phase_name} starting...")

            optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

            _ts = ts[: int(length_size * length)]
            _ys_train = ys_train[:, : int(length_size * length)]
            train_loader = dataloader((_ys_train,), batch_size, key=loader_key)

            for step in range(steps):
                (yi,) = next(train_loader)
                loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                train_losses.append(loss)
                current_step += 1

                if (step % print_every) == 0 or step == steps - 1:
                    _ys_test = ys_test[:, : int(length_size * length)]
                    test_loss, _ = grad_loss(model, _ts, _ys_test)
                    test_losses.append(test_loss)

                    # Calculate progress percentage
                    progress_pct = (current_step / total_steps) * 100

                    # Send detailed progress update
                    log(
                        f"Step {current_step}/{total_steps} ({progress_pct:.1f}%) - "
                        f"Train Loss: {loss:.3e}, Test Loss: {test_loss:.3e}"
                    )

                    # Update progress bar if updater provided
                    if progress_updater:
                        progress_updater(
                            current_step,
                            total_steps,
                            phase_name,
                            float(loss),
                            f"{phase_name} - Step {step + 1}/{steps}",
                        )

        log("Training completed successfully!")

        # Calculate final test loss
        final_test_loss, _ = grad_loss(model, ts_test, ys_test)

        log("Computing metrics...")

        # Use new metrics function
        metrics_train = calculate_all_metrics(
            ts,
            ys_train,
            model,
            scaler,
            training_initialconcentrations,
            "Train",
            ntimesteps,
            noise_level if noise else 0.0,
        )
        metrics_test = calculate_all_metrics(
            ts,
            ys_test,
            model,
            scaler,
            testing_initialconcentrations,
            "Test",
            ntimesteps,
            noise_level if noise else 0.0,
        )

        metrics_list = [
            dict(
                **m,
                **{
                    "Training_Experiments": len(training_initialconcentrations),
                    "Training_Timepoints": ntimesteps,
                    "Final_Train_Loss": train_losses[-1] if train_losses else 0.0,
                    "Final_Test_Loss": final_test_loss,
                },
            )
            for m in (metrics_train + metrics_test)
        ]

        # Calculate average RMSEs for logging
        train_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Train") / len(metrics_train)
        test_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Test") / len(metrics_test)

        log(f"Final Results - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        return ts, ys_train, ys_test, model, scaler, metrics_list

    except (eqx.EquinoxRuntimeError, jax._src.linear_util.StoreException) as e:
        log(f"Training failed with error: {str(e)}")

        error_metrics_list = make_failure_metrics(
            training_initialconcentrations,
            "Train",
            ntimesteps,
            noise_level if noise else 0.0,
        ) + make_failure_metrics(
            testing_initialconcentrations,
            "Test",
            ntimesteps,
            noise_level if noise else 0.0,
        )

        log("Setting error metrics (999.0) for all experiments due to training failure.")
        return (None, None, None, None, None, error_metrics_list)


def train_AugNODE_TL_dash_realtime(
    training_initialconcentrations: list[float],
    testing_initialconcentrations: list[float],
    *,
    model: AugmentedNeuralODE,
    scaler: MinMaxScaler | None,
    idx_frozen: int | tuple[int, int] | str = "last",
    freeze_mode: str = "both",
    ntimesteps: int = 85,
    nucl_params: list[float] = [39.81, 0.675],
    growth_params: list[float] = [0.345, 3.344],
    noise: bool = False,
    noise_level: float = 0.1,
    lr_strategy: tuple[float, float] = (6e-4, 6e-4),
    steps_strategy: tuple[int, int] = (600, 600),
    length_strategy: tuple[float, float] = (0.33, 1.0),
    batch_size: int | str = 1,
    scale_strategy: str | None = "refit_scaler",
    seed: int = 4700,
    print_every: int = 100,
    save_idxs: list[int] = [0, 3],
    progress_callback: Callable[[str], None] | None = None,
    progress_updater: Callable | None = None,
):
    """Transfer learning with layer freezing, adapted for Dash realtime progress.

    Mirrors train_AugNODE_TL but streams progress via callbacks.
    """

    def log(msg: str):
        if progress_callback:
            timestamp = time.strftime("%H:%M:%S")
            progress_callback(f"[{timestamp}] {msg}")

    key = jr.PRNGKey(seed)
    data_key, loader_key = jr.split(key, 2)

    if batch_size == "all":
        batch_size = len(training_initialconcentrations)

    log("Initializing data for TL…")
    if scale_strategy is None or scale_strategy == "keep_scaler":
        ts, ys_train, _ = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )
        ts_test, ys_test, _ = simulateCrystallisation(
            testing_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )
    elif scale_strategy == "refit_scaler":
        ts, ys_train, scaler = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )
        ts_test, ys_test, _ = simulateCrystallisation(
            testing_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )

    # Determine layers to freeze
    num_layers = len(model.func.mlp.layers)
    if idx_frozen == "none":
        frozen_indices = set()
    elif idx_frozen == "last":
        frozen_indices = {num_layers - 1}
    elif idx_frozen == "last_two":
        frozen_indices = {num_layers - 1, num_layers - 2}
    elif idx_frozen == "first":
        frozen_indices = {0}
    elif idx_frozen == "first_two":
        frozen_indices = {0, 1}
    elif idx_frozen == "first_three":
        frozen_indices = {0, 1, 2}
    elif idx_frozen == "all":
        frozen_indices = set(range(num_layers))
    elif isinstance(idx_frozen, int):
        frozen_indices = {idx_frozen}
    elif isinstance(idx_frozen, tuple):
        frozen_slice = slice(*idx_frozen)
        frozen_indices = set(range(num_layers)[frozen_slice])
    else:
        raise TypeError("Invalid idx_frozen spec")

    if freeze_mode not in ["weights", "biases", "both"]:
        raise ValueError("freeze_mode must be 'weights', 'biases', or 'both'")

    filter_spec = jtu.tree_map(lambda _: True, model)
    for idx in frozen_indices:
        if 0 <= idx < num_layers:
            if freeze_mode in ["weights", "both"]:
                filter_spec = eqx.tree_at(lambda t: t.func.mlp.layers[idx].weight, filter_spec, replace=False)
            if freeze_mode in ["biases", "both"]:
                filter_spec = eqx.tree_at(lambda t: t.func.mlp.layers[idx].bias, filter_spec, replace=False)

    # Loss fns
    def calculate_loss(m, ti, yi):
        pred = jax.vmap(m, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, m, opt_state):
        diff_m, static_m = eqx.partition(m, filter_spec)

        @eqx.filter_value_and_grad
        def loss_fn(dm):
            fm = eqx.combine(dm, static_m)
            return calculate_loss(fm, ti, yi)

        loss_value, grads = loss_fn(diff_m)
        updates, opt_state = optim.update(grads, opt_state, params=diff_m)
        diff_m = eqx.apply_updates(diff_m, updates)
        m = eqx.combine(diff_m, static_m)
        return loss_value, m, opt_state

    try:
        train_losses = []
        test_losses = []
        total_steps = sum(steps_strategy)
        current_step = 0
        log("Starting TL training…")

        _, length_size, _ = ys_train.shape
        for phase_idx, (lr, steps, length) in enumerate(zip(lr_strategy, steps_strategy, length_strategy, strict=True)):
            phase_name = f"Phase {phase_idx + 1}/2"
            if progress_updater:
                progress_updater(current_step, total_steps, phase_name, None, f"{phase_name} starting…")

            optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))
            diff_m, static_m = eqx.partition(model, filter_spec)
            opt_state = optim.init(eqx.filter(diff_m, eqx.is_inexact_array))

            _ts = ts[: int(length_size * length)]
            _ys_train = ys_train[:, : int(length_size * length)]
            train_loader = dataloader((_ys_train,), batch_size, key=loader_key)

            for step in range(steps):
                (yi,) = next(train_loader)
                loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                train_losses.append(loss)
                current_step += 1

                if (step % print_every) == 0 or step == steps - 1:
                    _ys_test = ys_test[:, : int(length_size * length)]
                    test_loss = calculate_loss(model, _ts, _ys_test)
                    test_losses.append(test_loss)
                    if progress_updater:
                        progress_updater(
                            current_step, total_steps, phase_name, float(loss), f"{phase_name} step {step + 1}/{steps}"
                        )

        final_test_loss = calculate_loss(model, ts, ys_test)
        log("Computing TL metrics…")

        metrics_train = calculate_all_metrics(
            ts,
            ys_train,
            model,
            scaler,
            training_initialconcentrations,
            "Train",
            ntimesteps,
            noise_level if noise else 0.0,
        )
        metrics_test = calculate_all_metrics(
            ts, ys_test, model, scaler, testing_initialconcentrations, "Test", ntimesteps, noise_level if noise else 0.0
        )

        metrics_list = [
            dict(
                **m,
                **{"Final_Train_Loss": train_losses[-1] if train_losses else 0.0, "Final_Test_Loss": final_test_loss},
            )
            for m in (metrics_train + metrics_test)
        ]

        return ts, ys_train, ys_test, model, scaler, metrics_list

    except (eqx.EquinoxRuntimeError, jax._src.linear_util.StoreException):
        error_metrics_list = make_failure_metrics(
            training_initialconcentrations, "Train", ntimesteps, noise_level if noise else 0.0
        ) + make_failure_metrics(testing_initialconcentrations, "Test", ntimesteps, noise_level if noise else 0.0)
        return (None, None, None, None, None, error_metrics_list)


def train_AugNODE_TL_penalty_dash_realtime(
    training_initialconcentrations: list[float],
    testing_initialconcentrations: list[float],
    *,
    model: AugmentedNeuralODE,
    scaler: MinMaxScaler | None,
    penalty_lambda: float = 1.0,
    penalty_strategy: int | tuple[int, int] | str = "all",
    ntimesteps: int = 85,
    nucl_params: list[float] = [39.81, 0.675],
    growth_params: list[float] = [0.345, 3.344],
    noise: bool = False,
    noise_level: float = 0.1,
    lr_strategy: tuple[float, float] = (6e-4, 6e-4),
    steps_strategy: tuple[int, int] = (600, 600),
    length_strategy: tuple[float, float] = (0.33, 1.0),
    batch_size: int | str = 1,
    scale_strategy: str | None = "refit_scaler",
    seed: int = 4700,
    print_every: int = 100,
    save_idxs: list[int] = [0, 3],
    progress_callback: Callable[[str], None] | None = None,
    progress_updater: Callable | None = None,
):
    """Transfer learning with penalty against deviation from the base model, Dash realtime."""

    def log(msg: str):
        if progress_callback:
            timestamp = time.strftime("%H:%M:%S")
            progress_callback(f"[{timestamp}] {msg}")

    key = jr.PRNGKey(seed)
    data_key, loader_key = jr.split(key, 2)

    if batch_size == "all":
        batch_size = len(training_initialconcentrations)

    log("Initializing data for TL-penalty…")
    if scale_strategy is None or scale_strategy == "keep_scaler":
        ts, ys_train, _ = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )
        ts_test, ys_test, _ = simulateCrystallisation(
            testing_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )
    elif scale_strategy == "refit_scaler":
        ts, ys_train, scaler = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )
        ts_test, ys_test, _ = simulateCrystallisation(
            testing_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )

    original_model = copy.deepcopy(model)

    # Determine penalty indices
    num_layers = len(model.func.mlp.layers)
    if penalty_strategy == "none":
        penalty_indices = set()
    elif penalty_strategy == "all":
        penalty_indices = set(range(num_layers))
    elif penalty_strategy == "last":
        penalty_indices = {num_layers - 1}
    elif penalty_strategy == "last_two":
        penalty_indices = {num_layers - 1, num_layers - 2}
    elif penalty_strategy == "first":
        penalty_indices = {0}
    elif penalty_strategy == "first_two":
        penalty_indices = {0, 1}
    elif penalty_strategy == "first_three":
        penalty_indices = {0, 1, 2}
    elif isinstance(penalty_strategy, int):
        penalty_indices = {penalty_strategy}
    elif isinstance(penalty_strategy, tuple):
        penalty_slice = slice(*penalty_strategy)
        penalty_indices = set(range(num_layers)[penalty_slice])
    else:
        raise TypeError("Invalid penalty_strategy spec")

    def calculate_loss_with_penalty(m, ti, yi):
        pred = jax.vmap(m, in_axes=(None, 0))(ti, yi[:, 0])
        mse_loss = jnp.mean((yi - pred) ** 2)
        penalty = 0.0
        for idx, (l1, l2) in enumerate(zip(m.func.mlp.layers, original_model.func.mlp.layers, strict=False)):
            if idx in penalty_indices:
                penalty += jnp.sum((l1.weight - l2.weight) ** 2)
                penalty += jnp.sum((l1.bias - l2.bias) ** 2)
        return mse_loss + penalty_lambda * penalty

    @eqx.filter_jit
    def make_step(ti, yi, m, opt_state):
        @eqx.filter_value_and_grad
        def loss_fn(m):
            return calculate_loss_with_penalty(m, ti, yi)

        loss, grads = loss_fn(m)
        updates, opt_state = optim.update(grads, opt_state, params=m)
        m = eqx.apply_updates(m, updates)
        return loss, m, opt_state

    try:
        train_losses = []
        test_losses = []
        total_steps = sum(steps_strategy)
        current_step = 0
        log("Starting TL-penalty training…")

        _, length_size, _ = ys_train.shape
        for phase_idx, (lr, steps, length) in enumerate(zip(lr_strategy, steps_strategy, length_strategy, strict=True)):
            phase_name = f"Phase {phase_idx + 1}/2"
            if progress_updater:
                progress_updater(current_step, total_steps, phase_name, None, f"{phase_name} starting…")

            optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

            _ts = ts[: int(length_size * length)]
            _ys_train = ys_train[:, : int(length_size * length)]
            train_loader = dataloader((_ys_train,), batch_size, key=loader_key)

            for step in range(steps):
                (yi,) = next(train_loader)
                loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                train_losses.append(loss)
                current_step += 1

                if (step % print_every) == 0 or step == steps - 1:
                    _ys_test = ys_test[:, : int(length_size * length)]
                    pred = jax.vmap(model, in_axes=(None, 0))(_ts, _ys_test[:, 0])
                    test_loss = jnp.mean((_ys_test - pred) ** 2)
                    test_losses.append(test_loss)
                    if progress_updater:
                        progress_updater(
                            current_step, total_steps, phase_name, float(loss), f"{phase_name} step {step + 1}/{steps}"
                        )

        # Final test loss (MSE only)
        pred = jax.vmap(model, in_axes=(None, 0))(ts, ys_test[:, 0])
        final_test_loss = jnp.mean((ys_test - pred) ** 2)

        log("Computing TL-penalty metrics…")
        metrics_train = calculate_all_metrics(
            ts,
            ys_train,
            model,
            scaler,
            training_initialconcentrations,
            "Train",
            ntimesteps,
            noise_level if noise else 0.0,
        )
        metrics_test = calculate_all_metrics(
            ts, ys_test, model, scaler, testing_initialconcentrations, "Test", ntimesteps, noise_level if noise else 0.0
        )
        metrics_list = [
            dict(
                **m,
                **{"Final_Train_Loss": train_losses[-1] if train_losses else 0.0, "Final_Test_Loss": final_test_loss},
            )
            for m in (metrics_train + metrics_test)
        ]

        return ts, ys_train, ys_test, model, scaler, metrics_list

    except (eqx.EquinoxRuntimeError, jax._src.linear_util.StoreException):
        error_metrics_list = make_failure_metrics(
            training_initialconcentrations, "Train", ntimesteps, noise_level if noise else 0.0
        ) + make_failure_metrics(testing_initialconcentrations, "Test", ntimesteps, noise_level if noise else 0.0)
        return (None, None, None, None, None, error_metrics_list)
