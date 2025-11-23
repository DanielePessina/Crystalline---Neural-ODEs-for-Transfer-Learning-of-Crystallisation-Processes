"""Augmented NODE training with mass‑conservation constraints.

Provides wrappers around the standard training loops that include
additional penalty terms to enforce monotonic concentration decrease and
particle size growth, encouraging physically consistent solutions.
"""

import time
from collections.abc import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax  # https://github.com/deepmind/optax
from jax import Array
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler

from Crystalline.augmented import AugmentedNeuralODE
from Crystalline.data_functions import dataloader, simulateCrystallisation
from Crystalline.metrics.calculations import calculate_all_metrics
from Crystalline.metrics.error import make_failure_metrics
from Crystalline.plotting import (
    plot_loss_curves,
    splitplot_model_vs_data,
)


def train_AugNODEconstrained_fresh(
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
    activation: Callable = jnn.swish,
    solver: diffrax.AbstractSolver = diffrax.Tsit5(),
    seed: int = 467,
    verbose: bool = True,
    print_every: int = 301,
    splitplot: bool = False,
    saveplot: bool = False,
    lossplot: bool = False,
    extratitlestring: str = "",
    save_idxs: list[int] = [0, 3],
    penalty_weight: float = 1e-5,
) -> tuple[Array, Array, Array, AugmentedNeuralODE, MinMaxScaler | None, list[dict]]:
    """Train an Augmented NODE under simple physical constraints.

    This routine mirrors :func:`train_AugNODE_fresh` but adds penalties on the
    time derivatives to enforce monotonic concentration decrease and particle
    size growth.  The strength of the constraint is controlled by
    ``penalty_weight``.

    Returns
    -------
    tuple
        ``(ts, ys_train, ys_test, model, scaler, metrics_list)`` as described
        in :func:`train_AugNODE_fresh`.
    """
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, split_key = jr.split(key, 4)

    if batch_size == "all":
        batch_size = len(training_initialconcentrations)

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

        ts_test, ys_test, _ = simulateCrystallisation(  # Use same scaler from training
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

        ts_test, ys_test, _ = simulateCrystallisation(  # Use same scaler from training
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

    model = AugmentedNeuralODE(
        data_size,
        augment_dim,
        width_size,
        depth,
        key=model_key,
        activation=activation,
        solver=solver,
    )

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])

        # Base reconstruction loss
        reconstruction_loss = jnp.mean(abs(yi - y_pred) ** 2)

        # Constraint penalties
        constraint_penalty = 0.0
        # penalty_weight = 1e-5  # Adjust this weight as needed

        # Apply constraints for each trajectory in the batch
        # Apply constraints for each trajectory in the batch
        for i in range(yi.shape[0]):
            y_init = yi[i, 0]

            # Augment the initial condition for the AugmentedNeuralODE
            y_init_aug = jnp.concatenate([y_init, jnp.zeros(augment_dim)])

            # Solve ODE once to get the full trajectory
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(model.func),
                model.solver,
                ti[0],
                ti[-1],
                dt0=None,
                y0=y_init_aug,
                saveat=diffrax.SaveAt(ts=ti),
                stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4),
            )

            # Get derivatives at each time point by evaluating the vector field
            # at the solution states
            def get_derivatives_at_solution_points():
                derivatives = []
                for j, t in enumerate(ti):
                    y_at_t = sol.ys[j]
                    dy_dt = model.func(t, y_at_t, args=None)
                    derivatives.append(dy_dt)
                return jnp.array(derivatives)

            derivatives_at_ti = get_derivatives_at_solution_points()

            # Extract only the original data dimensions for constraints (not augmented dims)
            # Penalize positive derivatives of feature 0 (concentration should decrease)
            positive_derivs_feat0 = jnp.maximum(0.0, derivatives_at_ti[:, 0])
            constraint_penalty += penalty_weight * jnp.mean(positive_derivs_feat0**2)

            # Penalize negative derivatives of feature 1 (particle size should increase)
            negative_derivs_feat1 = jnp.maximum(0.0, -derivatives_at_ti[:, 1])
            constraint_penalty += penalty_weight * jnp.mean(negative_derivs_feat1**2)

        return reconstruction_loss + constraint_penalty

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    try:
        train_losses = []
        test_losses = []

        for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy, strict=True):
            # optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
            optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))
            # optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adabelief(lr))
            # optim = optax.adabelief(lr)
            # optim = optax.adamw(lr, weight_decay=1e-4)

            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            _ts = ts[: int(length_size * length)]
            _ys_train = ys_train[:, : int(length_size * length)]
            train_loader = dataloader((_ys_train,), batch_size, key=loader_key)
            for step in range(steps):
                (yi,) = next(train_loader)
                start = time.time()
                loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                train_losses.append(loss)
                end = time.time()
                if verbose and ((step % print_every) == 0 or step == steps - 1):
                    _ys_test = ys_test[:, : int(length_size * length)]
                    test_loss, _ = grad_loss(model, _ts, _ys_test)
                    test_losses.append(test_loss)
                    print(
                        f"Step: {step}, Train Loss: {loss:.4e}, Test Loss: {test_loss:.4e}, Computation time: {end - start:.4e}"
                    )

        final_test_loss, _ = grad_loss(model, ts_test, ys_test)
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
                    "Final_Train_Loss": loss,
                    "Final_Test_Loss": final_test_loss,
                },
            )
            for m in (metrics_train + metrics_test)
        ]
        # Calculate average RMSEs for title
        train_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Train")
        test_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Test")

        rmse_str = f"RMSE - Train: {train_rmse:.4f} Test: {test_rmse:.4f}"

        extratitlestring = f"{extratitlestring} (Fresh)\n{rmse_str}"

    except (eqx.EquinoxRuntimeError, jax._src.linear_util.StoreException):
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

        # Print warning about failed training
        print("Warning: NODE training failed. Metrics will be set to 999.0 (error state) for all experiments.")

        return (None, None, None, None, None, error_metrics_list)

    if splitplot:
        # Compose a short string with key hyperparameters for filename
        filename_prefix = (
            f"AugNODE_w{width_size}_d{depth}_aug{augment_dim}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(l) for l in length_strategy)}_predictions"
        )
        splitplot_model_vs_data(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            extratitlestring,
            saveplot,
            filename_prefix=f"{filename_prefix}.png",
        )

    if lossplot:
        filename_prefix = (
            f"AugNODE_w{width_size}_d{depth}_aug{augment_dim}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(l) for l in length_strategy)}_loss"
        )
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"Loss Curves - {extratitlestring}",
            saveplot=saveplot,
            filename=f"plots/{filename_prefix}.png",
        )

    return ts, ys_train, ys_test, model, scaler, metrics_list


def train_AugNODEconstrained_TL(
    training_initialconcentrations: list[float],
    testing_initialconcentrations: list[float],
    model: AugmentedNeuralODE,
    scaler: MinMaxScaler | None,
    idx_frozen: int | tuple[int, int] | str,
    freeze_mode: str = "both",
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
    seed: int = 467,
    verbose: bool = True,
    print_every: int = 301,
    splitplot: bool = False,
    saveplot: bool = False,
    lossplot: bool = False,
    extratitlestring: str = "",
    save_idxs: list[int] = [0, 3],
    penalty_weight: float = 1e-5,
) -> tuple[Array, Array, Array, AugmentedNeuralODE, MinMaxScaler | None, list[dict]]:
    """Fine‑tune a constrained Augmented NODE.

    The arguments mirror :func:`train_AugNODEconstrained_fresh` with the extra
    ``idx_frozen`` and ``freeze_mode`` parameters controlling which parts of the
    network remain fixed during optimisation.

    Returns
    -------
    tuple
        ``(ts, ys_train, ys_test, model, scaler, metrics_list)`` analogous to
        :func:`train_AugNODEconstrained_fresh`.
    """
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, split_key, init_key = jr.split(key, 5)

    if batch_size == "all":
        batch_size = len(training_initialconcentrations)

    ### if scale_strategy is None: reuse the same scaler from the first NODE:
    if scale_strategy is None or scale_strategy == "keep_scaler":
        # Generate training and testing data separately
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

        ts_test, ys_test, _ = simulateCrystallisation(  # Use same scaler from training
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
        # Generate training and testing data separately, refitting the scaler
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

        ts_test, ys_test, _ = simulateCrystallisation(  # Use same scaler from training
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

    # Derive model properties instead of passing them if possible
    # Assuming model.func.mlp gives access to the MLP
    model_depth = len(model.func.mlp.layers)
    # Assuming the first layer's weight shape gives width_size (output dim)
    model_width = model.func.mlp.layers[0].weight.shape[0]

    augment_dim = model.augment_size
    # Activation and solver are part of the passed model object
    # Get dimensions from the data
    _, length_size, data_size = ys_train.shape

    # Determine the indices of layers to freeze
    num_layers = len(model.func.mlp.layers)
    if idx_frozen == "none":
        frozen_indices = set()  # Empty set means no layers are frozen
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
        raise TypeError(
            "idx_frozen must be an integer, a tuple representing a slice, the string 'last', or the string 'none'"
        )

    if freeze_mode not in ["weights", "biases", "both"]:
        raise ValueError("freeze_mode must be 'weights', 'biases', or 'both'")

    # Freeze specified layers/parameters - keep others trainable
    filter_spec = jtu.tree_map(lambda _: True, model)  # Set all params trainable by default
    for idx in frozen_indices:
        if 0 <= idx < num_layers:
            freeze_weights = freeze_mode in ["weights", "both"]
            freeze_biases = freeze_mode in ["biases", "both"]
            # Use eqx.tree_at multiple times or structure it carefully

            if freeze_weights:
                filter_spec = eqx.tree_at(
                    lambda tree: tree.func.mlp.layers[idx].weight,
                    filter_spec,
                    replace=False,
                )
            if freeze_biases:
                filter_spec = eqx.tree_at(
                    lambda tree: tree.func.mlp.layers[idx].bias,
                    filter_spec,
                    replace=False,
                )

        else:
            print(f"Warning: Index {idx} out of bounds for model with {num_layers} layers. Skipping.")

    @eqx.filter_value_and_grad
    def calculate_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])

        # Base reconstruction loss
        reconstruction_loss = jnp.mean(abs(yi - y_pred) ** 2)

        # Constraint penalties
        constraint_penalty = 0.0

        # Extract the function for constraint calculation
        vector_field = model.func  # This should be a callable

        # Apply constraints for each trajectory in the batch
        for i in range(yi.shape[0]):
            y_init = yi[i, 0]

            # Augment the initial condition for the AugmentedNeuralODE
            y_init_aug = jnp.concatenate([y_init, jnp.zeros(augment_dim)])

            # Solve ODE once to get the full trajectory
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(vector_field),
                model.solver,
                ti[0],
                ti[-1],
                dt0=None,
                y0=y_init_aug,
                saveat=diffrax.SaveAt(ts=ti),
                stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4),
            )

            # Get derivatives at each time point by evaluating the vector field
            # at the solution states
            def get_derivatives_at_solution_points():
                derivatives = []
                for j, t in enumerate(ti):
                    y_at_t = sol.ys[j]
                    # Use the vector field directly, not model.func
                    dy_dt = vector_field(t, y_at_t, args=None)
                    derivatives.append(dy_dt)
                return jnp.array(derivatives)

            derivatives_at_ti = get_derivatives_at_solution_points()

            # Extract only the original data dimensions for constraints (not augmented dims)
            # Penalize positive derivatives of feature 0 (concentration should decrease)
            positive_derivs_feat0 = jnp.maximum(0.0, derivatives_at_ti[:, 0])
            constraint_penalty += penalty_weight * jnp.mean(positive_derivs_feat0**2)

            # Penalize negative derivatives of feature 1 (particle size should increase)
            negative_derivs_feat1 = jnp.maximum(0.0, -derivatives_at_ti[:, 1])
            constraint_penalty += penalty_weight * jnp.mean(negative_derivs_feat1**2)

        return reconstruction_loss + constraint_penalty

    # Remove the decorator from calculate_loss and create a separate function for loss_fn
    def loss_only(model, ti, yi):
        # This function only returns the loss value, no gradients
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])

        # Base reconstruction loss
        reconstruction_loss = jnp.mean(abs(yi - y_pred) ** 2)

        # Constraint penalties
        constraint_penalty = 0.0

        # Extract the function for constraint calculation
        vector_field = model.func

        # Apply constraints for each trajectory in the batch
        for i in range(yi.shape[0]):
            y_init = yi[i, 0]
            y_init_aug = jnp.concatenate([y_init, jnp.zeros(augment_dim)])

            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(vector_field),
                model.solver,
                ti[0],
                ti[-1],
                dt0=None,
                y0=y_init_aug,
                saveat=diffrax.SaveAt(ts=ti),
                stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4),
            )

            def get_derivatives_at_solution_points():
                derivatives = []
                for j, t in enumerate(ti):
                    y_at_t = sol.ys[j]
                    dy_dt = vector_field(t, y_at_t, args=None)
                    derivatives.append(dy_dt)
                return jnp.array(derivatives)

            derivatives_at_ti = get_derivatives_at_solution_points()

            positive_derivs_feat0 = jnp.maximum(0.0, derivatives_at_ti[:, 0])
            constraint_penalty += penalty_weight * jnp.mean(positive_derivs_feat0**2)

            negative_derivs_feat1 = jnp.maximum(0.0, -derivatives_at_ti[:, 1])
            constraint_penalty += penalty_weight * jnp.mean(negative_derivs_feat1**2)

        return reconstruction_loss + constraint_penalty

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)

        @eqx.filter_value_and_grad
        def loss_fn(diff_model):
            # Recombine the model for forward pass
            full_model = eqx.combine(diff_model, static_model)
            return loss_only(full_model, ti, yi)  # Use loss_only here

        loss_value, grads = loss_fn(diff_model)
        updates, opt_state = optim.update(grads, opt_state, params=diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
        return loss_value, model, opt_state

    try:
        train_losses = []
        test_losses = []

        for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy, strict=True):
            optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))

            # Initialize optimizer state with only trainable parameters
            diff_model, static_model = eqx.partition(model, filter_spec)
            opt_state = optim.init(eqx.filter(diff_model, eqx.is_inexact_array))

            _ts = ts[: int(length_size * length)]
            _ys_train = ys_train[:, : int(length_size * length)]
            for step, (yi,) in zip(range(steps), dataloader((_ys_train,), batch_size, key=loader_key), strict=False):
                start = time.time()
                loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                end = time.time()
                train_losses.append(loss)

                if verbose and ((step % print_every) == 0 or step == steps - 1):
                    # Calculate test loss directly
                    _ys_test = ys_test[:, : int(length_size * length)]
                    test_loss, _ = calculate_loss(model, _ts, _ys_test)
                    test_losses.append(test_loss)
                    print(
                        f"Step: {step}, Train Loss: {loss:.4e}, Test Loss: {test_loss:.4e}, Computation time: {end - start:.4e}"
                    )
        # Calculate final test loss
        final_test_loss, _ = calculate_loss(model, ts, ys_test)

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
                    "Network_Width": model_width,
                    "Network_Depth": model_depth,
                    "Learning_Rate_1": lr_strategy[0] if len(lr_strategy) > 0 else None,
                    "Learning_Rate_2": lr_strategy[1] if len(lr_strategy) > 1 else None,
                    "Epochs_1": steps_strategy[0] if len(steps_strategy) > 0 else None,
                    "Epochs_2": steps_strategy[1] if len(steps_strategy) > 1 else None,
                    "Final_Train_Loss": loss,
                    "Final_Test_Loss": final_test_loss,
                },
            )
            for m in (metrics_train + metrics_test)
        ]

        train_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Train")
        test_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Test")

        rmse_str = f"RMSE - Train: {train_rmse:.4f} Test: {test_rmse:.4f}"

        extratitlestring = f"{extratitlestring} (TL)\n{rmse_str}"

    except (eqx.EquinoxRuntimeError, jax._src.linear_util.StoreException):
        error_metrics_list = []
        for metrics_dict in make_failure_metrics(
            training_initialconcentrations, "Train", ntimesteps, noise_level if noise else 0.0
        ):
            metrics_dict.update(
                {
                    "Network_Width": model_width,
                    "Network_Depth": model_depth,
                    "Learning_Rate_1": lr_strategy[0] if len(lr_strategy) > 0 else None,
                    "Learning_Rate_2": lr_strategy[1] if len(lr_strategy) > 1 else None,
                    "Epochs_1": steps_strategy[0] if len(steps_strategy) > 0 else None,
                    "Epochs_2": steps_strategy[1] if len(steps_strategy) > 1 else None,
                }
            )
            error_metrics_list.append(metrics_dict)

        for metrics_dict in make_failure_metrics(
            testing_initialconcentrations, "Test", ntimesteps, noise_level if noise else 0.0
        ):
            metrics_dict.update(
                {
                    "Network_Width": model_width,
                    "Network_Depth": model_depth,
                    "Learning_Rate_1": lr_strategy[0] if len(lr_strategy) > 0 else None,
                    "Learning_Rate_2": lr_strategy[1] if len(lr_strategy) > 1 else None,
                    "Epochs_1": steps_strategy[0] if len(steps_strategy) > 0 else None,
                    "Epochs_2": steps_strategy[1] if len(steps_strategy) > 1 else None,
                }
            )
            error_metrics_list.append(metrics_dict)

        # Print warning about failed training
        print("Warning: AugNODE TL training failed. Metrics will be set to 999.0 (error state) for all experiments.")

        return (None, None, None, None, None, error_metrics_list)

    if splitplot:
        filename_prefix = (
            f"AugNODE_TL_w{model_width}_d{model_depth}_aug{augment_dim}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(l) for l in length_strategy)}_predictions"
        )
        splitplot_model_vs_data(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            extratitlestring,
            saveplot,
            filename_prefix=f"{filename_prefix}.png",
        )
    if lossplot:
        filename_prefix = (
            f"AugNODE_TL_w{model_width}_d{model_depth}_aug{augment_dim}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(l) for l in length_strategy)}_loss"
        )
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"Loss Curves - {extratitlestring}",
            saveplot=saveplot,
            filename=f"plots/{filename_prefix}.png",
        )
    return ts, ys_train, ys_test, model, scaler, metrics_list


# %%
if __name__ == "__main__":
    base_system_kineticparams = [
        [39.8, 0.675],  # nucl_params
        [0.37, 3.3],
    ]

    hydrox = [
        [29.9, 0.49],  # nucl_params
        [0.37, 3.3],  # growth_params
        "Hydroxyl",
    ]

    carboxyl = [
        [22.7, 0.14],  # nucl_params
        [0.37, 3.3],  # growth_params
        "Carboxyl",
    ]
    butyl = [
        [27.4, 0.32],  # nucl_params
        [0.37, 3.3],  # growth_params
        "Butyl",
    ]

    ts, ys_train, ys_test, model, scaler, metrics = train_AugNODEconstrained_fresh(
        # [13.0, 15.5, 16.5, 19.0],
        [
            15.5,
            16.5,
        ],
        [12, 14.3, 17.7, 20],
        lr_strategy=(2e-4, 4e-4),
        steps_strategy=(600, 600),
        length_strategy=(0.33, 1),
        width_size=32,
        depth=4,
        activation=jnn.swish,
        ntimesteps=14 * 2,
        # solver=diffrax.Kvaerno3(),
        seed=4700,
        splitplot=True,
        lossplot=True,
        save_idxs=[0, 3],
        noise=False,
        noise_level=0.05,
        print_every=100,
        batch_size="all",
        augment_dim=1,
        #
        penalty_weight=1e-2,
        #     # scale_strategy = "none",
        #     ## Homogeneous crystallization
        nucl_params=base_system_kineticparams[0],
        growth_params=base_system_kineticparams[1],
        #     ## Hydroxyl template
        # nucl_params=[28.2, 0.44],
        # growth_params=[0.186, 3.02],
        #     ## Carboxyl template
        # nucl_params=[32.94, 0.53348],
        # growth_params=[0.1, 3.3125],
    )

    ts_tl, ys_train_tl, ys_test_tl, model_tl, scaler_tl, metrics_tl = train_AugNODEconstrained_TL(
        [
            15.5,
            16.5,
        ],
        [12, 14.3, 17.7, 20],
        model,
        scaler,
        idx_frozen="last",
        freeze_mode="both",
        lr_strategy=(2e-4, 4e-4),
        steps_strategy=(600, 600),
        length_strategy=(0.33, 1),
        ntimesteps=14 * 2,
        # solver=diffrax.Kvaerno3(),
        seed=4700,
        splitplot=True,
        lossplot=True,
        save_idxs=[0, 3],
        noise=False,
        noise_level=0.05,
        print_every=100,
        batch_size="all",
        penalty_weight=1e-2,
    )

# %%
