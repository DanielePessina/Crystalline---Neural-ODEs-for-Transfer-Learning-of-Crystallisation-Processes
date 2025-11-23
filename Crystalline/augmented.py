"""Training utilities for Augmented Neural ODE models.

This module defines the :class:`AugmentedNeuralODE` architecture along
with helper classes and functions to train it either from scratch or
via transfer learning.  It implements the standard Augmented NODE
approach where the state is extended with additional dimensions to
capture complex dynamics.
"""

from collections.abc import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax  # https://github.com/deepmind/optax
from jax import Array
from rich.console import Console

# Progress bar and pretty‑printing utilities
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.preprocessing import MinMaxScaler
from traitlets import Bool

from Crystalline.data_functions import dataloader, simulateCrystallisation
from Crystalline.metrics.calculations import calculate_all_metrics
from Crystalline.metrics.error import make_failure_metrics
from Crystalline.plotting import (
    interactive_splitplot_model_vs_data_1d,
    plot_loss_curves,
    splitplot_model_vs_data,
    splitplot_model_vs_data_1d,
)


def compute_model_checksum(model: eqx.Module) -> float:
    """Compute a checksum (sum of squares of numeric leaves) for an Equinox model.

    Args:
        model: Equinox module or PyTree of parameters.

    Returns:
        Floating checksum useful for equality checks across copies.
    """

    leaves = jtu.tree_leaves(model)
    total = 0.0
    for leaf in leaves:
        try:
            arr = np.asarray(leaf)
            if arr.dtype.kind in ("f", "i", "u"):
                total += float(np.sum(arr * arr))
        except Exception:
            continue
    return total


def compute_scaler_signature(scaler) -> tuple:
    """Return a lightweight signature of a sklearn MinMaxScaler instance.

    Includes sums and shapes of key learned attributes to detect mutation.
    """

    attrs = []
    for name in ("data_min_", "data_max_", "min_", "scale_"):
        if hasattr(scaler, name):
            val = getattr(scaler, name)
            arr = np.asarray(val)
            attrs.append((name, arr.shape, float(np.sum(arr))))
    return tuple(attrs)


# Harmonized concentration lists (used in __main__ examples)
C_TRAIN_SOURCE = [15.5, 16.5, 19]
C_TRAIN_TARGET = [16.5, 17.0]
C_TEST = [14.3, 17.7, 20]


class AugmentedFunc(eqx.Module):
    """
    Vector field for an Augmented Neural ODE.

    Defines dynamics over an augmented space [x; a] ∈ ℝ^{d + d_aug}, allowing for more expressive flows.

    Args:
        data_size: Dimensionality of the original input space.
        augment_size: Dimensionality of the augmented variables.
        width_size: Width of hidden layers in the MLP.
        depth: Number of hidden layers in the MLP.
        activation: Activation function used in the MLP.
        key: PRNGKey for initializing parameters.
        output_constraints: Dictionary mapping output indices to constraints ('pos', 'neg', 'none').
    """

    mlp: eqx.nn.MLP
    include_time: bool = eqx.static_field()
    output_constraints: dict | None = eqx.static_field()

    def __init__(
        self,
        data_size: int,
        augment_size: int,
        width_size: int,
        depth: int,
        include_time: bool = True,
        *,
        key,
        activation: Callable = jnn.softplus,
        output_constraints: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        total_dim = data_size + augment_size
        self.include_time = include_time
        self.output_constraints = output_constraints
        self.mlp = eqx.nn.MLP(
            in_size=total_dim + (1 if include_time else 0),
            out_size=total_dim,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )

    def __call__(self, t: Array, y_aug: Array, args: None = None) -> Array:
        if self.include_time:
            t_feat = jnp.atleast_1d(t)
            mlp_in = jnp.concatenate([y_aug, t_feat], axis=-1)
        else:
            mlp_in = y_aug
        output = self.mlp(mlp_in)

        if self.output_constraints is None:
            return output

        # Apply constraints based on the provided dictionary
        pos_indices = [k for k, v in self.output_constraints.items() if v == "pos"]
        neg_indices = [k for k, v in self.output_constraints.items() if v == "neg"]

        # Use softplus to enforce positivity
        softplus_values = jnn.softplus(output)

        # Update positive-constrained outputs
        if pos_indices:
            output = output.at[jnp.array(pos_indices)].set(softplus_values[jnp.array(pos_indices)])

        # Update negative-constrained outputs
        if neg_indices:
            output = output.at[jnp.array(neg_indices)].set(-softplus_values[jnp.array(neg_indices)])

        return output


class AugmentedNeuralODE(eqx.Module):
    """
    Augmented Neural ODE model that integrates dynamics in a higher-dimensional space.

    Args:
        data_size: Original data dimension (d).
        augment_size: Additional dimensions (d_aug).
        width_size: Width of hidden layers.
        depth: Number of hidden layers.
        solver: Diffrax ODE solver (defaults to Tsit5).
        key: PRNGKey for initialization.
        activation: Activation function for MLP.
        output_constraints: Dictionary mapping output indices to constraints ('pos', 'neg', 'none').
    """

    func: AugmentedFunc
    solver: diffrax.AbstractSolver = eqx.static_field()
    augment_size: int = eqx.static_field()
    include_time: bool = eqx.static_field()

    def __init__(
        self,
        data_size: int,
        augment_size: int,
        width_size: int,
        depth: int,
        include_time: bool = True,
        *,
        key,
        activation: Callable = jnn.softplus,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        output_constraints: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.augment_size = augment_size
        self.include_time = include_time
        self.func = AugmentedFunc(
            data_size,
            augment_size,
            width_size,
            depth,
            include_time=include_time,
            key=key,
            activation=activation,
            output_constraints=output_constraints,
        )
        self.solver = solver

    def __call__(self, ts: Array, y0: Array, *, return_augmented: Bool = False) -> Array:
        """
        Args:
            ts: Time points of shape (T,)
            y0: Initial condition in ℝ^d (original space), shape (d,)

        Returns:
            Trajectory in original space ℝ^d, shape (T, d)
        """
        # Augment initial condition with zeros
        y0_aug = jnp.concatenate([y0, jnp.zeros(self.augment_size, dtype=y0.dtype)], axis=-1)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=1e-3,
            y0=y0_aug,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        )

        if return_augmented:
            return solution.ys  # shape (T, d+d_aug)

        return solution.ys[..., : y0.shape[-1]]  # original behaviour
        # Return only the original state portion

    def get_augmented_trajectory(self, ts: Array, y0: Array) -> Array:
        """
        Get the full augmented trajectory including both original and augmented dimensions.

        Args:
            ts: Time points of shape (T,)
            y0: Initial condition in ℝ^d (original space), shape (d,)

        Returns:
            Full augmented trajectory in ℝ^{d+d_aug}, shape (T, d+d_aug)
        """
        return self(ts, y0, return_augmented=True)


def collect_augmented_bounds(
    model: AugmentedNeuralODE,
    ts: Array,
    training_data: Array,
) -> tuple[Array, Array]:
    """Collect min/max bounds for augmented dimensions from training data.

    Args:
        model: Trained AugmentedNeuralODE model.
        ts: Time points array of shape (T,).
        training_data: Training trajectories array of shape (N, T, data_size)
                      where N is number of training examples.

    Returns:
        Tuple of (augmented_min, augmented_max) arrays, each of shape (augment_size,)
        containing the minimum and maximum values observed for each augmented dimension
        across all training examples and time points.
    """
    if len(training_data.shape) != 3:
        raise ValueError(f"Expected training_data shape (N, T, data_size), got {training_data.shape}")

    n_examples, n_timesteps, data_size = training_data.shape

    # Get model dimensions
    total_dim = model.func.mlp.layers[-1].weight.shape[0]
    augment_size = model.augment_size

    if total_dim != data_size + augment_size:
        raise ValueError(
            f"Model dimension mismatch: total_dim={total_dim}, data_size={data_size}, augment_size={augment_size}"
        )

    # Initialize arrays to collect augmented statistics
    augmented_min = jnp.full((augment_size,), jnp.inf)
    augmented_max = jnp.full((augment_size,), -jnp.inf)

    # Process each training example
    for i in range(n_examples):
        # Get initial condition (first time step)
        y0 = training_data[i, 0, :]  # shape (data_size,)

        # Get full augmented trajectory
        aug_trajectory = model.get_augmented_trajectory(ts, y0)  # shape (T, data_size + augment_size)

        # Extract augmented dimensions (last augment_size columns)
        aug_dims = aug_trajectory[:, data_size:]  # shape (T, augment_size)

        # Update min/max bounds
        augmented_min = jnp.minimum(augmented_min, jnp.min(aug_dims, axis=0))
        augmented_max = jnp.maximum(augmented_max, jnp.max(aug_dims, axis=0))

    return augmented_min, augmented_max


def train_AugNODE_fresh(
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
    verbose: bool = True,
    print_every: int = 301,
    splitplot: bool = False,
    plotly_plots: bool = False,
    saveplot: bool = False,
    lossplot: bool = False,
    extratitlestring: str = "",
    save_idxs: list[int] = [0, 3],
    output_constraints: dict | None = None,
) -> tuple[Array, Array, Array, AugmentedNeuralODE, MinMaxScaler | None, list[dict]]:
    """Train an augmented NODE from scratch using synthetic crystallisation data.

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

    Returns
    -------
    tuple
        ``(ts, ys_train, ys_test, model, scaler, metrics_list, train_losses, test_losses)``
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
        include_time=include_time,
        output_constraints=output_constraints,
    )

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean(abs(yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    try:
        train_losses = []
        test_losses = []
        # Force a predictable width so bars don’t scroll off‑screen in VS Code notebooks
        console = Console(width=100)

        # Prepare a single progress bar for the whole training procedure
        total_steps = sum(steps_strategy) - 2
        training_progress = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=20, complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.description}", justify="right"),
            console=console,
            auto_refresh=True,
            refresh_per_second=5,
            transient=False,
        )

        progress_task = training_progress.add_task("[cyan]Training", total=total_steps)

        with training_progress:
            for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy, strict=True):
                optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))
                opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
                _ts = ts[: int(length_size * length)]
                _ys_train = ys_train[:, : int(length_size * length)]
                train_loader = dataloader((_ys_train,), batch_size, key=loader_key)
                for step in range(steps):
                    (yi,) = next(train_loader)
                    # start = time.time()
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                    train_losses.append(loss)

                    if verbose and ((step % print_every) == 0 or step == steps - 1):
                        _ys_test = ys_test[:, : int(length_size * length)]
                        test_loss, _ = grad_loss(model, _ts, _ys_test)
                        test_losses.append(test_loss)
                        # Update progress bar with current metrics
                        training_progress.update(
                            progress_task,
                            advance=1,
                            description=f"loss={loss:.3e} test={test_loss:.3e}",
                        )
                    else:
                        training_progress.advance(progress_task)
                        # keep description stable to avoid width creep
                        # training_progress.update(progress_task, description="")

        training_progress.stop()

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
            f"len{'-'.join(str(length) for length in length_strategy)}_predictions"
        )
        splitplot_model_vs_data_1d(
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

    if plotly_plots:
        filename_prefix = (
            f"AugNODE_w{width_size}_d{depth}_aug{augment_dim}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(length) for length in length_strategy)}_bokeh"
        )

        interactive_splitplot_model_vs_data_1d(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            extratitlestring,
            saveplot=saveplot,
            filename_prefix=filename_prefix,
            output_mode="notebook",
        )

    if lossplot:
        filename_prefix = (
            f"AugNODE_w{width_size}_d{depth}_aug{augment_dim}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(length) for length in length_strategy)}_loss"
        )
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"Loss Curves - {extratitlestring}",
            saveplot=saveplot,
            filename=f"plots/{filename_prefix}.png",
        )

    # Collect augmented dimension bounds from training data
    aug_min, aug_max = collect_augmented_bounds(model, ts, ys_train)

    return ts, ys_train, ys_test, model, scaler, metrics_list, (aug_min, aug_max)  # , train_losses, test_losses


def train_AugNODE_TL(
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
) -> tuple[Array, Array, Array, AugmentedNeuralODE, MinMaxScaler | None, list[dict]]:
    """Fine‑tune an existing Augmented NODE on new data.

    Parameters
    ----------
    model : AugmentedNeuralODE
        Pre‑trained model to start from.
    idx_frozen : int | tuple[int, int] | str
        Which layers of the MLP to freeze during optimisation.  ``"last"`` and
        ``"first"`` are convenient shorthands.
    freeze_mode : str, optional
        Choose ``"weights"``, ``"biases"`` or ``"both"`` to control which
        parameters of the selected layers remain fixed.

    Other parameters are identical to :func:`train_AugNODE_fresh`.

    Returns
    -------
    tuple
        ``(ts, ys_train, ys_test, model, scaler, metrics_list)`` analogous to
        :func:`train_AugNODE_fresh` but using the fine‑tuned ``model``.
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

    # Define loss calculation function
    def calculate_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)

        @eqx.filter_value_and_grad
        def loss_fn(diff_model):
            # Recombine the model for forward pass
            full_model = eqx.combine(diff_model, static_model)
            return calculate_loss(full_model, ti, yi)

        loss_value, grads = loss_fn(diff_model)
        updates, opt_state = optim.update(grads, opt_state, params=diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
        return loss_value, model, opt_state

    try:
        train_losses = []
        test_losses = []
        # Force a predictable width so bars don’t scroll off‑screen in VSCode notebooks
        console = Console(width=100)

        # Prepare a single progress bar for the whole training procedure
        total_steps = sum(steps_strategy) - 2
        training_progress = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=20, complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.description}", justify="right"),
            console=console,
            auto_refresh=True,
            refresh_per_second=5,
            transient=False,
        )

        progress_task = training_progress.add_task("[cyan]Fine-tuning", total=total_steps)

        with training_progress:
            for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy, strict=True):
                optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))

                # Initialize optimizer state with only trainable parameters
                diff_model, static_model = eqx.partition(model, filter_spec)
                opt_state = optim.init(eqx.filter(diff_model, eqx.is_inexact_array))

                _ts = ts[: int(length_size * length)]
                _ys_train = ys_train[:, : int(length_size * length)]
                train_loader = dataloader((_ys_train,), batch_size, key=loader_key)
                for step in range(steps):
                    (yi,) = next(train_loader)
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                    train_losses.append(loss)

                    if verbose and ((step % print_every) == 0 or step == steps - 1):
                        _ys_test = ys_test[:, : int(length_size * length)]
                        test_loss = calculate_loss(model, _ts, _ys_test)
                        test_losses.append(test_loss)
                        # Update progress bar with current metrics
                        training_progress.update(
                            progress_task,
                            advance=1,
                            description=f"loss={loss:.3e} test={test_loss:.3e}",
                        )
                    else:
                        training_progress.advance(progress_task)

        training_progress.stop()
        # Calculate final test loss
        final_test_loss = calculate_loss(model, ts, ys_test)

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
            f"len{'-'.join(str(length) for length in length_strategy)}_predictions"
        )
        splitplot_model_vs_data_1d(
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
            f"len{'-'.join(str(length) for length in length_strategy)}_loss"
        )
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"Loss Curves - {extratitlestring}",
            saveplot=saveplot,
            filename=f"plots/{filename_prefix}.png",
        )

    # Collect augmented dimension bounds from training data
    aug_min, aug_max = collect_augmented_bounds(model, ts, ys_train)

    return ts, ys_train, ys_test, model, scaler, metrics_list, (aug_min, aug_max)


def train_AugNODE_TL_penalise_deviation(
    training_initialconcentrations: list[float],
    testing_initialconcentrations: list[float],
    model: AugmentedNeuralODE,
    scaler: MinMaxScaler | None,
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
    penalty_lambda: float = 1.0,
    penalty_strategy: int | tuple[int, int] | str = "all",  # NEW ARGUMENT
) -> tuple[Array, Array, Array, AugmentedNeuralODE, MinMaxScaler | None, list[dict]]:
    """Fine‑tune an Augmented NODE with a deviation penalty.

    Parameters
    ----------
    model : AugmentedNeuralODE
        Base model whose parameters act as the reference weights.
    penalty_lambda : float
        Scaling applied to the L2 penalty that discourages the new parameters
        from deviating from their initial values.
    penalty_strategy : int | tuple[int, int] | str
        Selects which layers contribute to the penalty (e.g. ``"last"`` or
        ``"all"``).

    Returns
    -------
    tuple
        ``(ts, ys_train, ys_test, model, scaler, metrics_list)`` similar to
        :func:`train_AugNODE_fresh` but including the penalty in the loss.
    """
    import copy

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, split_key, init_key = jr.split(key, 5)

    if batch_size == "all":
        batch_size = len(training_initialconcentrations)

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

    # Save a copy of the original model for penalty computation

    original_model = copy.deepcopy(model)

    model_depth = len(model.func.mlp.layers)
    model_width = model.func.mlp.layers[0].weight.shape[0]
    augment_dim = model.augment_size
    _, length_size, data_size = ys_train.shape

    # --- Penalty strategy logic ---
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
        raise TypeError(
            "penalty_strategy must be an integer, a tuple representing a slice, the string 'last', or the string 'all'"
        )
    # --- End penalty strategy logic ---

    def calculate_loss_with_penalty(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        mse_loss = jnp.mean((yi - y_pred) ** 2)
        # Penalty: sum of squared differences for selected weights and biases
        penalty = 0.0
        for idx, (l1, l2) in enumerate(zip(model.func.mlp.layers, original_model.func.mlp.layers, strict=False)):
            if idx in penalty_indices:
                penalty += jnp.sum((l1.weight - l2.weight) ** 2)
                penalty += jnp.sum((l1.bias - l2.bias) ** 2)
        return mse_loss + penalty_lambda * penalty

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        @eqx.filter_value_and_grad
        def loss_fn(model):
            return calculate_loss_with_penalty(model, ti, yi)

        loss, grads = loss_fn(model)
        updates, opt_state = optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    try:
        train_losses = []
        test_losses = []
        # Force a predictable width so bars don’t scroll off‑screen in VSCode notebooks
        console = Console(width=100)

        # Prepare a single progress bar for the whole training procedure
        total_steps = sum(steps_strategy) - 2
        training_progress = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=20, complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.description}", justify="right"),
            console=console,
            auto_refresh=True,
            refresh_per_second=5,
            transient=False,
        )

        progress_task = training_progress.add_task("[cyan]Fine-tuning (Penalty)", total=total_steps)

        with training_progress:
            for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy, strict=True):
                optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))
                opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
                _ts = ts[: int(length_size * length)]
                _ys_train = ys_train[:, : int(length_size * length)]
                train_loader = dataloader((_ys_train,), batch_size, key=loader_key)
                for step in range(steps):
                    (yi,) = next(train_loader)
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                    train_losses.append(loss)
                    if verbose and ((step % print_every) == 0 or step == steps - 1):
                        _ys_test = ys_test[:, : int(length_size * length)]
                        # Use only the MSE part for reporting test loss
                        y_pred = jax.vmap(model, in_axes=(None, 0))(_ts, _ys_test[:, 0])
                        test_loss = jnp.mean((_ys_test - y_pred) ** 2)
                        test_losses.append(test_loss)
                        training_progress.update(
                            progress_task,
                            advance=1,
                            description=f"loss={loss:.3e} test={test_loss:.3e}",
                        )
                    else:
                        training_progress.advance(progress_task)

        training_progress.stop()

        # Final test loss (MSE only)
        y_pred = jax.vmap(model, in_axes=(None, 0))(ts, ys_test[:, 0])
        final_test_loss = jnp.mean((ys_test - y_pred) ** 2)

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

        extratitlestring = f"{extratitlestring} (TL-penalty)\n{rmse_str}"

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
        print(
            "Warning: AugNODE TL-penalty training failed. Metrics will be set to 999.0 (error state) for all experiments."
        )

        return (None, None, None, None, None, error_metrics_list)

    if splitplot:
        filename_prefix = (
            f"AugNODE_TLpenalty_w{model_width}_d{model_depth}_aug{augment_dim}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(length) for length in length_strategy)}_predictions"
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
            f"AugNODE_TLpenalty_w{model_width}_d{model_depth}_aug{augment_dim}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(length) for length in length_strategy)}_loss"
        )
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"Loss Curves - {extratitlestring}",
            saveplot=saveplot,
            filename=f"plots/{filename_prefix}.png",
        )

    # Collect augmented dimension bounds from training data
    aug_min, aug_max = collect_augmented_bounds(model, ts, ys_train)

    return ts, ys_train, ys_test, model, scaler, metrics_list, (aug_min, aug_max)


def train_AugNODE_TL_append(
    training_initialconcentrations: list[float],
    testing_initialconcentrations: list[float],
    model: AugmentedNeuralODE,
    scaler: MinMaxScaler | None,
    append_layers: int = 1,
    append_width: int | None = None,
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
) -> tuple[Array, Array, Array, AugmentedNeuralODE, MinMaxScaler | None, list[dict]]:
    """Perform transfer learning by extending the network architecture.

    Parameters
    ----------
    model : AugmentedNeuralODE
        Base model providing the frozen core.
    append_layers : int
        How many new layers to prepend and append to the existing network.
    append_width : int | None
        Width of the new layers; defaults to the original model width if ``None``.

    Other parameters mirror :func:`train_AugNODE_fresh`.

    Returns
    -------
    tuple
        ``(ts, ys_train, ys_test, new_model, scaler, metrics_list)`` where
        ``new_model`` is the augmented network containing the frozen core.
    """
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, split_key, init_key = jr.split(key, 5)

    if batch_size == "all":
        batch_size = len(training_initialconcentrations)

    # Generate data
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

    # Get model properties
    original_layers = model.func.mlp.layers
    num_layers = len(original_layers)
    total_dim = original_layers[0].weight.shape[1]  # Input dimension
    original_width = original_layers[0].weight.shape[0] if num_layers > 0 else total_dim

    if append_width is None:
        append_width = original_width

    _, length_size, data_size = ys_train.shape

    # Create new AugmentedFunc with appended layers
    class AppendedAugmentedFunc(eqx.Module):
        pre_layers: list
        frozen_layers: list
        post_layers: list
        include_time: bool = eqx.static_field()
        output_constraints: dict | None = eqx.static_field()

        def __init__(self, original_model, append_layers, append_width, key):
            pre_key, post_key = jr.split(key, 2)

            # Frozen core (original model)
            self.frozen_layers = original_model.func.mlp.layers
            self.include_time = getattr(original_model, "include_time", True)
            self.output_constraints = original_model.func.output_constraints

            # Pre-layers (trainable)
            self.pre_layers = []
            if append_layers > 0:
                pre_keys = jr.split(pre_key, append_layers)
                current_dim = total_dim
                for i in range(append_layers):
                    layer = eqx.nn.Linear(current_dim, append_width, key=pre_keys[i])
                    self.pre_layers.append(layer)
                    current_dim = append_width

            # Post-layers (trainable)
            self.post_layers = []
            if append_layers > 0:
                post_keys = jr.split(post_key, append_layers)
                current_dim = original_layers[-1].weight.shape[0] if num_layers > 0 else append_width
                for i in range(append_layers - 1):
                    layer = eqx.nn.Linear(current_dim, append_width, key=post_keys[i])
                    self.post_layers.append(layer)
                    current_dim = append_width
                # Final layer to match output dimension
                final_layer = eqx.nn.Linear(current_dim, total_dim, key=post_keys[-1])
                self.post_layers.append(final_layer)

        def __call__(self, t: Array, y_aug: Array) -> Array:
            if self.include_time:
                x = jnp.concatenate([y_aug, jnp.atleast_1d(t)], axis=-1)
            else:
                x = y_aug

            # Forward through pre-layers
            for layer in self.pre_layers:
                x = jnn.swish(layer(x))

            # Forward through frozen layers
            for layer in self.frozen_layers:
                x = jnn.swish(layer(x))

            # Forward through post-layers
            for i, layer in enumerate(self.post_layers):
                if i < len(self.post_layers) - 1:
                    x = jnn.swish(layer(x))
                else:
                    x = layer(x)  # No activation on final layer

            if self.output_constraints is None:
                return x

            # Apply constraints based on the provided dictionary
            pos_indices = [k for k, v in self.output_constraints.items() if v == "pos"]
            neg_indices = [k for k, v in self.output_constraints.items() if v == "neg"]

            # Use softplus to enforce positivity
            softplus_values = jnn.softplus(x)

            # Update positive-constrained outputs
            if pos_indices:
                x = x.at[jnp.array(pos_indices)].set(softplus_values[jnp.array(pos_indices)])

            # Update negative-constrained outputs
            if neg_indices:
                x = x.at[jnp.array(neg_indices)].set(-softplus_values[jnp.array(neg_indices)])

            return x

    # Create new model with appended architecture
    class AppendedAugmentedNeuralODE(eqx.Module):
        func: AppendedAugmentedFunc
        solver: diffrax.AbstractSolver = eqx.static_field()
        augment_size: int = eqx.static_field()

        def __init__(self, original_model, append_layers, append_width, key):
            self.augment_size = original_model.augment_size
            self.func = AppendedAugmentedFunc(original_model, append_layers, append_width, key)
            self.solver = original_model.solver

        def __call__(self, ts: Array, y0: Array, *, return_augmented: bool = False) -> Array:
            y0_aug = jnp.concatenate([y0, jnp.zeros(self.augment_size, dtype=y0.dtype)], axis=-1)

            def vector_field(t, y, args):
                return self.func(t, y)

            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(vector_field),
                solver=self.solver,
                t0=ts[0],
                t1=ts[-1],
                dt0=0.01 * (ts[1] - ts[0]),
                y0=y0_aug,
                saveat=diffrax.SaveAt(ts=ts),
                stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4),
            )

            if return_augmented:
                return solution.ys

            return solution.ys[..., : y0.shape[-1]]

    # Create the new model
    new_model = AppendedAugmentedNeuralODE(model, append_layers, append_width, init_key)

    # Set up training with frozen core
    def is_trainable(x):
        # Only pre_layers and post_layers are trainable
        return True

    def is_frozen_core(path, x):
        # Freeze the original layers
        return "frozen_layers" in str(path)

    filter_spec = jtu.tree_map_with_path(
        lambda path, x: not is_frozen_core(path, x) if eqx.is_inexact_array(x) else True, new_model
    )

    def calculate_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)

        @eqx.filter_value_and_grad
        def loss_fn(diff_model):
            full_model = eqx.combine(diff_model, static_model)
            return calculate_loss(full_model, ti, yi)

        loss_value, grads = loss_fn(diff_model)
        updates, opt_state = optim.update(grads, opt_state, params=diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
        return loss_value, model, opt_state

    try:
        train_losses = []
        test_losses = []
        # Force a predictable width so bars don’t scroll off‑screen in VSCode notebooks
        console = Console(width=100)

        # Prepare a single progress bar for the whole training procedure
        total_steps = sum(steps_strategy) - 2
        training_progress = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=20, complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.description}", justify="right"),
            console=console,
            auto_refresh=True,
            refresh_per_second=5,
            transient=False,
        )

        progress_task = training_progress.add_task("[cyan]Fine-tuning (Append)", total=total_steps)

        with training_progress:
            for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy, strict=True):
                optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-4))

                diff_model, static_model = eqx.partition(new_model, filter_spec)
                opt_state = optim.init(eqx.filter(diff_model, eqx.is_inexact_array))

                _ts = ts[: int(length_size * length)]
                _ys_train = ys_train[:, : int(length_size * length)]
                train_loader = dataloader((_ys_train,), batch_size, key=loader_key)
                for step in range(steps):
                    (yi,) = next(train_loader)
                    # start = time.time()
                    loss, new_model, opt_state = make_step(_ts, yi, new_model, opt_state)
                    train_losses.append(loss)

                    if verbose and ((step % print_every) == 0 or step == steps - 1):
                        _ys_test = ys_test[:, : int(length_size * length)]
                        test_loss = calculate_loss(new_model, _ts, _ys_test)
                        test_losses.append(test_loss)
                        training_progress.update(
                            progress_task,
                            advance=1,
                            description=f"loss={loss:.3e} test={test_loss:.3e}",
                        )
                    else:
                        training_progress.advance(progress_task)

        training_progress.stop()

        final_test_loss = calculate_loss(new_model, ts, ys_test)

        # Calculate metrics
        metrics_train = calculate_all_metrics(
            ts,
            ys_train,
            new_model,
            scaler,
            training_initialconcentrations,
            "Train",
            ntimesteps,
            noise_level if noise else 0.0,
        )
        metrics_test = calculate_all_metrics(
            ts,
            ys_test,
            new_model,
            scaler,
            testing_initialconcentrations,
            "Test",
            ntimesteps,
            noise_level if noise else 0.0,
        )

        total_layers = num_layers + 2 * append_layers
        metrics_list = [
            dict(
                **m,
                **{
                    "Training_Experiments": len(training_initialconcentrations),
                    "Training_Timepoints": ntimesteps,
                    "Network_Width": append_width,
                    "Network_Depth": total_layers,
                    "Appended_Layers": append_layers,
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
        extratitlestring = f"{extratitlestring} (TL-append)\n{rmse_str}"

    except (eqx.EquinoxRuntimeError, jax._src.linear_util.StoreException):
        # Error handling similar to other functions
        error_metrics_list = []
        for metrics_dict in make_failure_metrics(
            training_initialconcentrations, "Train", ntimesteps, noise_level if noise else 0.0
        ):
            metrics_dict.update(
                {
                    "Network_Width": append_width,
                    "Network_Depth": num_layers + 2 * append_layers,
                    "Appended_Layers": append_layers,
                }
            )
            error_metrics_list.append(metrics_dict)

        for metrics_dict in make_failure_metrics(
            testing_initialconcentrations, "Test", ntimesteps, noise_level if noise else 0.0
        ):
            metrics_dict.update(
                {
                    "Network_Width": append_width,
                    "Network_Depth": num_layers + 2 * append_layers,
                    "Appended_Layers": append_layers,
                }
            )
            error_metrics_list.append(metrics_dict)

        print(
            "Warning: AugNODE TL-append training failed. Metrics will be set to 999.0 (error state) for all experiments."
        )
        return (None, None, None, None, None, error_metrics_list)

    if splitplot:
        filename_prefix = (
            f"AugNODE_TLappend_w{append_width}_d{total_layers}_append{append_layers}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(length) for length in length_strategy)}_predictions"
        )
        splitplot_model_vs_data(
            ts,
            ys_train,
            ys_test,
            new_model,
            scaler,
            length_strategy,
            extratitlestring,
            saveplot,
            filename_prefix=f"{filename_prefix}.png",
        )

    if lossplot:
        filename_prefix = (
            f"AugNODE_TLappend_w{append_width}_d{total_layers}_append{append_layers}_"
            f"lr{'-'.join(str(lr) for lr in lr_strategy)}_"
            f"steps{'-'.join(str(s) for s in steps_strategy)}_"
            f"len{'-'.join(str(length) for length in length_strategy)}_loss"
        )
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"Loss Curves - {extratitlestring}",
            saveplot=saveplot,
            filename=f"plots/{filename_prefix}.png",
        )

    # Collect augmented dimension bounds from training data
    aug_min, aug_max = collect_augmented_bounds(new_model, ts, ys_train)

    return ts, ys_train, ys_test, new_model, scaler, metrics_list, (aug_min, aug_max)


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

    ts, ys_train, ys_test, model, scaler, metrics, (aug_min, aug_max) = train_AugNODE_fresh(
        C_TRAIN_SOURCE,
        C_TEST,
        lr_strategy=(2e-4, 4e-4),
        steps_strategy=(600, 600),
        length_strategy=(0.33, 1),
        width_size=32,
        depth=4,
        activation=jnn.swish,
        ntimesteps=5,
        # solver=diffrax.Kvaerno3(),
        seed=4700,
        splitplot=True,
        plotly_plots=False,
        save_idxs=[0, 3],
        noise=True,
        noise_level=0.05,
        print_every=100,
        batch_size="all",
        augment_dim=2,
        #
        #     # scale_strategy = "none",
        #     ## Homogeneous crystallization
        nucl_params=base_system_kineticparams[0],
        growth_params=base_system_kineticparams[1],
        # nucl_params=hydrox[0],
        # growth_params=hydrox[1],
        output_constraints={
            0: "neg",  # Concentration of species 1 must be non-negative
            1: "pos",  # Concentration of species 2 must be non-negative
        },
        # output_constraints={
        #     0: "none",  # Concentration of species 1 must be non-negative
        #     1: "pos",  # Concentration of species 2 must be non-negative
        # },
    )

    # Print augmented dimension bounds
    print("\nAugmented dimension bounds from training data:")
    print(f"Min bounds: {aug_min}")
    print(f"Max bounds: {aug_max}")
    print(f"Augmented dimension range: {aug_max - aug_min}")


# %%


def _latin_hypercube_sample(n_samples: int, dim: int, *, key: Array) -> Array:
    """Generate Latin Hypercube samples in [0, 1]^dim using JAX.

    Args:
        n_samples: Number of samples to draw (rows of the design).
        dim: Dimension of the design space.
        key: JAX PRNG key for reproducibility.

    Returns:
        Array of shape ``(n_samples, dim)`` with entries in ``[0, 1]``.
    """
    if n_samples <= 0:
        return jnp.empty((0, dim))
    if dim <= 0:
        return jnp.empty((n_samples, 0))

    # For each dimension: stratify [0,1] into n intervals; shuffle the strata; add uniform jitter.
    keys = jr.split(key, dim)
    strata = jnp.linspace(0.0, 1.0, n_samples + 1)
    # Use midpoints with jitter for robustness
    base = (strata[:-1] + strata[1:]) * 0.5  # (n_samples,)

    def _sample_one_dim(k: Array) -> Array:
        # Shuffle strata order
        idx = jr.permutation(k, jnp.arange(n_samples))
        # Add small jitter within each stratum; keep inside [0,1]
        eps = jr.uniform(k, (n_samples,), minval=-0.5 / n_samples, maxval=0.5 / n_samples)
        return jnp.clip(base + eps, 0.0, 1.0)[idx]

    samples = jax.vmap(_sample_one_dim)(jnp.stack(keys))  # (dim, n_samples)
    return samples.T  # (n_samples, dim)


import pandas as pd


def sample_augmented_vector_field_lhs(
    model: AugmentedNeuralODE,
    n_samples: int,
    *,
    state_bounds: tuple[float, float] = (-1.0, 1.0),
    aug_bounds: tuple[float, float] = (-1.0, 1.0),
    time_bounds: tuple[float, float] = (0.0, 1.0),
    seed: int = 0,
    state_names: list[str] | None = None,
    aug_names: list[str] | None = None,
) -> "pd.DataFrame":
    """Evaluate the model's vector field (MLP) on Latin Hypercube-sampled inputs.

    This samples the inputs to the underlying MLP used by the augmented NODE's
    vector field: the augmented state ``y_aug = concat([x, a])`` and, if the
    model was built with ``include_time=True``, the time ``t``. It then returns
    a DataFrame of inputs and corresponding vector-field outputs.

    Args:
        model: Trained base :class:`AugmentedNeuralODE` model.
        n_samples: Number of LHS samples to generate.
        state_bounds: Tuple ``(low, high)`` for original state dimensions (scaled space).
        aug_bounds: Tuple ``(low, high)`` for augmented dimensions.
        time_bounds: Tuple ``(low, high)`` for time ``t`` (typically ``(0, 1)``).
        seed: Integer random seed for reproducibility.
        state_names: Optional names for the original state dimensions, length ``data_size``.
        aug_names: Optional names for the augmented dimensions, length ``augment_size``.

    Returns:
        pandas.DataFrame with columns for inputs and outputs. Input columns are
        ordered as ``[x_*, a_*, (t)]``. Output columns are the vector field
        components ordered as ``[dx_*/dt, da_*/dt]``.

    Example:
        >>> df = sample_augmented_vector_field_lhs(model, n_samples=512, seed=123)
        >>> df.head()
    """
    # Local import to avoid making pandas a hard dependency at module import time
    import pandas as pd  # type: ignore

    if not isinstance(model, AugmentedNeuralODE):
        raise TypeError("Expected an AugmentedNeuralODE model.")
    if n_samples <= 0:
        # Return an empty but well-formed DataFrame
        return pd.DataFrame()

    total_dim = model.func.mlp.layers[-1].weight.shape[0]  # data_size + augment_size
    augment_size = model.augment_size
    data_size = total_dim - augment_size

    # Names
    if state_names is None:
        state_names = [f"x{i}" for i in range(data_size)]
    if aug_names is None:
        aug_names = [f"a{i}" for i in range(augment_size)]
    if len(state_names) != data_size:
        raise ValueError("state_names length must equal the model's data_size.")
    if len(aug_names) != augment_size:
        raise ValueError("aug_names length must equal the model's augment_size.")

    # LHS per block so we can put different bounds on each
    key = jr.PRNGKey(seed)
    key_x, key_a, key_t = jr.split(key, 3)

    # State block
    xs_unit = _latin_hypercube_sample(n_samples, data_size, key=key_x)
    x_low, x_high = state_bounds
    xs = x_low + xs_unit * (x_high - x_low)

    # Augmented block
    as_unit = _latin_hypercube_sample(n_samples, augment_size, key=key_a)
    a_low, a_high = aug_bounds
    if augment_size > 0:
        aas = a_low + as_unit * (a_high - a_low)
    else:
        aas = jnp.empty((n_samples, 0))

    # Time block (always generated; ignored if model.include_time=False)
    t_unit = _latin_hypercube_sample(n_samples, 1, key=key_t).squeeze(-1)
    t_low, t_high = time_bounds
    ts = t_low + t_unit * (t_high - t_low)

    # Build augmented state and evaluate vector field with vmap
    y_aug = jnp.concatenate([xs, aas], axis=-1)  # (n_samples, total_dim)

    @eqx.filter_jit
    def eval_field(ts_in: Array, y_in: Array) -> Array:
        return jax.vmap(lambda t, y: model.func(t, y))(ts_in, y_in)

    outs = eval_field(ts, y_aug)  # (n_samples, total_dim)

    # Assemble DataFrame
    input_cols = {name: np.asarray(xs[:, i]) for i, name in enumerate(state_names)}
    input_cols.update({name: np.asarray(aas[:, i]) for i, name in enumerate(aug_names)})
    if model.include_time:
        input_cols["t"] = np.asarray(ts)

    out_state_names = [f"d{name}/dt" for name in state_names]
    out_aug_names = [f"d{name}/dt" for name in aug_names]

    output_cols = {name: np.asarray(outs[:, i]) for i, name in enumerate(out_state_names)}
    # Augmented derivative columns follow
    for j, name in enumerate(out_aug_names, start=data_size):
        output_cols[name] = np.asarray(outs[:, j])

    return pd.DataFrame({**input_cols, **output_cols})


def lhs_sample_model_inputs(
    model: AugmentedNeuralODE,
    n_samples: int,
    *,
    input_bounds: dict[str, tuple[float, float]] | Array | np.ndarray | None = None,
    seed: int = 0,
    state_names: list[str] | None = None,
    aug_names: list[str] | None = None,
) -> "pd.DataFrame":
    """LHS-sample per-input ranges and evaluate the trained model's vector field.

    This is a higher-level convenience over ``sample_augmented_vector_field_lhs``
    that supports per-dimension ranges. It samples the inputs to the MLP driving
    the augmented NODE (original state, augmented state, and optionally time),
    evaluates ``model.func(t, y_aug)``, and returns a tidy DataFrame.

    Args:
        model: Trained :class:`AugmentedNeuralODE` model to probe.
        n_samples: Number of Latin Hypercube samples to draw.
        input_bounds: Optional bounds specification for each input dimension. One of:
            - dict mapping names to ``(low, high)``; names from ``state_names``,
              ``aug_names``, and optionally ``"t"`` when ``model.include_time``.
            - array of shape ``(K, 2)`` with ``K = data_size + augment_size + include_time``
              (in order ``[x_*, a_*, t?]``).
            - ``None`` uses defaults ``[-1, 1]`` for state+aug, and ``[0, 1]`` for time.
        seed: PRNG seed for reproducibility.
        state_names: Optional names for original state dimensions; defaults ``x0, x1, ...``.
        aug_names: Optional names for augmented dimensions; defaults ``a0, a1, ...``.

    Returns:
        pandas.DataFrame with input and output columns. Input columns are
        ordered as ``[x_*, a_*, (t)]`` and outputs as ``[d x_*/dt, d a_*/dt]``.

    Example:
        >>> bounds = {"x0": (-1.0, 1.0), "x1": (-0.5, 0.5), "t": (0.0, 1.0)}
        >>> df = lhs_sample_model_inputs(model, 1000, input_bounds=bounds, seed=123)
    """
    # Local imports to avoid global hard dependency at import time
    import pandas as pd  # type: ignore

    if not isinstance(model, AugmentedNeuralODE):
        raise TypeError("Expected an AugmentedNeuralODE model.")
    if n_samples <= 0:
        return pd.DataFrame()

    total_dim = model.func.mlp.layers[-1].weight.shape[0]  # data_size + augment_size
    augment_size = model.augment_size
    data_size = total_dim - augment_size

    # Names
    if state_names is None:
        state_names = [f"x{i}" for i in range(data_size)]
    if aug_names is None:
        aug_names = [f"a{i}" for i in range(augment_size)]
    if len(state_names) != data_size:
        raise ValueError("state_names length must equal the model's data_size.")
    if len(aug_names) != augment_size:
        raise ValueError("aug_names length must equal the model's augment_size.")

    include_time = bool(model.include_time)
    input_names = state_names + aug_names + (["t"] if include_time else [])
    K = len(input_names)

    # Resolve per-dimension bounds to arrays of shape (K,)
    if input_bounds is None:
        lows = np.array([-1.0] * (data_size + augment_size) + ([0.0] if include_time else []), dtype=float)
        highs = np.array([1.0] * (data_size + augment_size) + ([1.0] if include_time else []), dtype=float)
    elif isinstance(input_bounds, dict):
        lows_list: list[float] = []
        highs_list: list[float] = []
        for name in input_names:
            if name in input_bounds:
                lo, hi = input_bounds[name]
            else:
                if name == "t":
                    lo, hi = 0.0, 1.0
                else:
                    lo, hi = -1.0, 1.0
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"Bounds for {name} must be finite; got {(lo, hi)}")
            if hi < lo:
                raise ValueError(f"Upper bound < lower bound for {name}: {(lo, hi)}")
            lows_list.append(float(lo))
            highs_list.append(float(hi))
        lows = np.array(lows_list, dtype=float)
        highs = np.array(highs_list, dtype=float)
    else:
        arr = np.asarray(input_bounds)
        if arr.shape != (K, 2):
            raise ValueError(f"input_bounds array must have shape {(K, 2)}; got {arr.shape}")
        lows = np.asarray(arr[:, 0], dtype=float)
        highs = np.asarray(arr[:, 1], dtype=float)
        if np.any(~np.isfinite(lows)) or np.any(~np.isfinite(highs)):
            raise ValueError("input_bounds must be finite.")
        if np.any(highs < lows):
            raise ValueError("Each high must be >= low in input_bounds.")

    # Generate LHS in unit hypercube then rescale per-dimension
    key = jr.PRNGKey(seed)
    if K > 0:
        unit = _latin_hypercube_sample(n_samples, K, key=key)  # (n, K)
        bounds_span = jnp.asarray(highs - lows)
        lows_j = jnp.asarray(lows)
        samples = lows_j + unit * bounds_span  # (n, K)
    else:
        samples = jnp.empty((n_samples, 0))

    # Partition into components
    xs = samples[:, :data_size] if data_size > 0 else jnp.empty((n_samples, 0))
    aas = samples[:, data_size : data_size + augment_size] if augment_size > 0 else jnp.empty((n_samples, 0))
    if include_time:
        ts = samples[:, -1]
    else:
        ts = jnp.zeros((n_samples,))

    y_aug = jnp.concatenate([xs, aas], axis=-1) if (data_size + augment_size) > 0 else jnp.zeros((n_samples, 0))

    @eqx.filter_jit
    def eval_field(ts_in: Array, y_in: Array) -> Array:
        return jax.vmap(lambda t, y: model.func(t, y))(ts_in, y_in)

    outs = eval_field(ts, y_aug)  # (n, total_dim)

    # Build DataFrame
    input_cols = {name: np.asarray(xs[:, i]) for i, name in enumerate(state_names)}
    input_cols.update({name: np.asarray(aas[:, i]) for i, name in enumerate(aug_names)})
    if include_time:
        input_cols["t"] = np.asarray(ts)

    out_state_names = [f"d{name}/dt" for name in state_names]
    out_aug_names = [f"d{name}/dt" for name in aug_names]

    output_cols = {name: np.asarray(outs[:, i]) for i, name in enumerate(out_state_names)}
    for j, name in enumerate(out_aug_names, start=data_size):
        output_cols[name] = np.asarray(outs[:, j])

    return pd.DataFrame({**input_cols, **output_cols})


def create_y_dataset_sheet(
    model: AugmentedNeuralODE,
    ys_train: Array,
    ts: Array,
    state_names: list[str] | None = None,
    aug_names: list[str] | None = None,
) -> "pd.DataFrame":
    """Create a DataFrame with the original y-data (concatenated experiments).

    This function takes the original training data and formats it as a flat
    DataFrame with columns: exp_id, t, x0, x1, a0, a1, ... where the experiments
    are concatenated vertically.

    Args:
        model: Trained AugmentedNeuralODE.
        ys_train: Original training data of shape (num_experiments, num_timepoints, data_size).
        ts: Time array of shape (num_timepoints,).
        state_names: Optional state dimension names.
        aug_names: Optional augmented dimension names.

    Returns:
        DataFrame with concatenated training data across experiments.
    """
    import pandas as pd

    # Validate model dims
    total_dim = model.func.mlp.layers[-1].weight.shape[0]
    augment_size = model.augment_size
    data_size = total_dim - augment_size

    if state_names is None:
        state_names = [f"x{i}" for i in range(data_size)]
    if aug_names is None:
        aug_names = [f"a{i}" for i in range(augment_size)]

    # Get augmented trajectories for each experiment individually
    # to avoid shape mismatch issues with vmap
    augmented_trajectories = []
    for exp_idx in range(ys_train.shape[0]):
        y0_exp = ys_train[exp_idx, 0, :]  # Get initial condition for this experiment
        aug_traj = model.get_augmented_trajectory(ts, y0_exp)  # Shape: (num_timepoints, total_dim)
        augmented_trajectories.append(aug_traj)

    # Stack to get (num_experiments, num_timepoints, total_dim)
    augmented_trajectories = jnp.stack(augmented_trajectories, axis=0)

    # Build DataFrame with concatenated experiments
    all_data = []
    num_experiments = ys_train.shape[0]

    for exp_idx in range(num_experiments):
        for t_idx, t in enumerate(ts):
            record = {"exp_id": exp_idx, "t": float(t)}

            # Add state dimensions
            for i, name in enumerate(state_names):
                record[name] = float(ys_train[exp_idx, t_idx, i])

            # Add augmented dimensions
            for i, name in enumerate(aug_names):
                record[name] = float(augmented_trajectories[exp_idx, t_idx, data_size + i])

            all_data.append(record)

    return pd.DataFrame(all_data)


def lhs_sample_model_inputs_using_scaler(
    model: AugmentedNeuralODE,
    scaler: MinMaxScaler | object,
    n_samples: int,
    *,
    physical_bounds: dict[str, tuple[float, float]] | np.ndarray | None = None,
    aug_bounds_scaled: tuple[float, float] | np.ndarray | None = (-1.0, 1.0),
    time_physical_bounds: tuple[float, float] | None = None,
    time_span_physical: tuple[float, float] | None = (0.0, 300.0),
    excel_path: str | None = None,
    seed: int = 0,
    state_names: list[str] | None = None,
    aug_names: list[str] | None = None,
    rescale_gradients: bool = True,
    save_unscaled: bool = False,
) -> "pd.DataFrame":
    """LHS-sample with physical bounds mapped via scaler, and optionally save to Excel.

    This routine accepts bounds in the unscaled (physical) space for the original
    state dimensions, converts them to the model's scaled space using the trained
    sklearn scaler, performs Latin Hypercube Sampling across state, augmented, and
    time dimensions, evaluates the model's vector field, and returns the results
    as a DataFrame. Optionally writes an Excel file with sheets: ``samples``,
    ``bounds``, and optionally ``unscaled_samples`` and ``y_dataset``.

    For MinMaxScaler, gradients are properly rescaled back to physical units using
    the chain rule: dy/dt = (1 / scaler.scale_[i]) / time_scale_max × (dy_std/dt_std),
    where time_scale_max is derived from time_span_physical.

    Args:
        model: Trained :class:`AugmentedNeuralODE`.
        scaler: Fitted sklearn scaler used during training (e.g., MinMaxScaler or StandardScaler).
        n_samples: Number of LHS samples.
        physical_bounds: Per-state-dimension bounds in physical space. One of:
            - dict mapping state names (``x*``) to ``(low, high)``.
            - array of shape ``(data_size, 2)`` in order ``[x0..]``.
            - ``None`` defaults to scaler-derived bounds: MinMax->(data_min_, data_max_),
              StandardScaler->(mean_±3*scale_).
        aug_bounds_scaled: Bounds for augmented dims in scaled space. Either a single
            ``(low, high)`` tuple applied to all augmented dims or an array of shape
            ``(augment_size, 2)``.
        time_physical_bounds: Optional physical time bounds ``(low, high)`` [s]. If provided
            and ``model.include_time`` is True, these are mapped to scaled time using
            ``time_span_physical``. If ``None``, default scaled time bounds are used.
        time_span_physical: Physical time span ``(t0, t1)`` [s] used to map physical→scaled time
            via ``t_scaled = (t - t0)/(t1 - t0)``. Defaults to (0, 300) seconds.
        excel_path: If provided, writes sheets: 'samples', 'bounds', and optionally 'unscaled_samples' and 'y_dataset'.
        seed: PRNG seed.
        state_names: Optional state dimension names.
        aug_names: Optional augmented dimension names.
        rescale_gradients: If True, rescale gradients back to physical units using chain rule.
        save_unscaled: If True, saves unscaled version of samples and optionally y_dataset (if training data provided).

    Returns:
        DataFrame with input columns ``[x_*, a_*, (t)]`` and output columns
        ``[d x_*/dt, d a_*/dt]``. If rescale_gradients=True, inputs are in physical units
        and gradients are rescaled to physical units; otherwise, all values are in scaled space.
    """
    import pandas as pd  # type: ignore

    # Validate model dims
    total_dim = model.func.mlp.layers[-1].weight.shape[0]
    augment_size = model.augment_size
    data_size = total_dim - augment_size

    if state_names is None:
        state_names = [f"x{i}" for i in range(data_size)]
    if aug_names is None:
        aug_names = [f"a{i}" for i in range(augment_size)]
    include_time = bool(model.include_time)

    # Derive default physical bounds from scaler if not provided
    def _default_phys_bounds_from_scaler() -> tuple[np.ndarray, np.ndarray, str]:
        # Returns (low_phys[data_size], high_phys[data_size], note)
        if (
            hasattr(scaler, "data_min_")
            and hasattr(scaler, "data_max_")
            and hasattr(scaler, "scale_")
            and hasattr(scaler, "min_")
        ):
            note = "MinMaxScaler: using data_min_/data_max_"
            return np.asarray(scaler.data_min_, float), np.asarray(scaler.data_max_, float), note
        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            note = "StandardScaler: using mean±3*scale"
            mean = np.asarray(scaler.mean_, float)
            scale = np.asarray(scaler.scale_, float)
            return mean - 3 * scale, mean + 3 * scale, note
        note = "Unknown scaler: default [-1,1]"
        return np.full((data_size,), -1.0), np.full((data_size,), 1.0), note

    # Resolve physical bounds array
    if physical_bounds is None:
        low_phys, high_phys, pb_note = _default_phys_bounds_from_scaler()
    elif isinstance(physical_bounds, dict):
        low_list: list[float] = []
        high_list: list[float] = []
        for nm in state_names:
            if nm in physical_bounds:
                lo, hi = physical_bounds[nm]
            else:
                # fallback per-dim from scaler
                low_fallback, high_fallback, _ = _default_phys_bounds_from_scaler()
                idx = state_names.index(nm)
                lo, hi = float(low_fallback[idx]), float(high_fallback[idx])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
                raise ValueError(f"Invalid physical bounds for {nm}: {(lo, hi)}")
            low_list.append(float(lo))
            high_list.append(float(hi))
        low_phys = np.asarray(low_list, float)
        high_phys = np.asarray(high_list, float)
        pb_note = "User-provided dict"
    else:
        arr = np.asarray(physical_bounds)
        if arr.shape != (data_size, 2):
            raise ValueError(f"physical_bounds must have shape {(data_size, 2)}; got {arr.shape}")
        low_phys = np.asarray(arr[:, 0], float)
        high_phys = np.asarray(arr[:, 1], float)
        if np.any(~np.isfinite(low_phys)) or np.any(~np.isfinite(high_phys)) or np.any(high_phys < low_phys):
            raise ValueError("physical_bounds must be finite with high >= low for all dims.")
        pb_note = "User-provided array"

    # Map physical bounds → scaled bounds for state dims
    if (
        hasattr(scaler, "data_min_")
        and hasattr(scaler, "data_max_")
        and hasattr(scaler, "scale_")
        and hasattr(scaler, "min_")
    ):
        # MinMaxScaler: scaled = x * scale_ + min_
        scale = np.asarray(scaler.scale_, float)
        min_shift = np.asarray(scaler.min_, float)
        low_scaled_state = low_phys * scale + min_shift
        high_scaled_state = high_phys * scale + min_shift
        scaler_type = "MinMaxScaler"
        fr_low, fr_high = getattr(scaler, "feature_range", (-1.0, 1.0))
        scaler_note = f"feature_range={fr_low, fr_high}"
    elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        # StandardScaler: scaled = (x - mean_) / scale_
        mean = np.asarray(scaler.mean_, float)
        scale = np.asarray(scaler.scale_, float)
        # Avoid division by zero if any scale is zero
        if np.any(scale == 0.0):
            raise ValueError("StandardScaler has zero scale for at least one feature; cannot scale bounds.")
        low_scaled_state = (low_phys - mean) / scale
        high_scaled_state = (high_phys - mean) / scale
        scaler_type = "StandardScaler"
        scaler_note = ""
    else:
        raise TypeError("Unsupported scaler type. Provide a fitted MinMaxScaler or StandardScaler.")

    # Augmented dims scaled bounds
    if augment_size > 0:
        if isinstance(aug_bounds_scaled, tuple):
            lo_a, hi_a = aug_bounds_scaled
            low_scaled_aug = np.full((augment_size,), float(lo_a))
            high_scaled_aug = np.full((augment_size,), float(hi_a))
        else:
            arr = np.asarray(aug_bounds_scaled)
            if arr.shape != (augment_size, 2):
                raise ValueError(f"aug_bounds_scaled must have shape {(augment_size, 2)}; got {arr.shape}")
            low_scaled_aug = np.asarray(arr[:, 0], float)
            high_scaled_aug = np.asarray(arr[:, 1], float)
    else:
        low_scaled_aug = np.empty((0,), float)
        high_scaled_aug = np.empty((0,), float)

    # Time bounds: map physical → scaled using time_span_physical if provided; else default [0,1]
    if include_time:
        if time_physical_bounds is not None and time_span_physical is not None:
            t0, t1 = map(float, time_span_physical)
            if t1 <= t0:
                raise ValueError("time_span_physical must satisfy t1 > t0")
            t_lo_phys, t_hi_phys = map(float, time_physical_bounds)
            t_lo_scaled = (t_lo_phys - t0) / (t1 - t0)
            t_hi_scaled = (t_hi_phys - t0) / (t1 - t0)
            # Clamp to [0,1] to be safe
            t_lo_scaled = float(np.clip(t_lo_scaled, 0.0, 1.0))
            t_hi_scaled = float(np.clip(t_hi_scaled, 0.0, 1.0))
        else:
            t_lo_scaled, t_hi_scaled = 0.0, 1.0
    else:
        t_lo_scaled = t_hi_scaled = None

    # Build combined scaled input bounds in order [x_*, a_*, (t)]
    low_scaled = np.concatenate([low_scaled_state, low_scaled_aug])
    high_scaled = np.concatenate([high_scaled_state, high_scaled_aug])
    input_names = state_names + aug_names
    if include_time:
        low_scaled = np.concatenate([low_scaled, np.array([t_lo_scaled], float)])
        high_scaled = np.concatenate([high_scaled, np.array([t_hi_scaled], float)])
        input_names = input_names + ["t"]

    # Sample and evaluate using scaled bounds
    input_bounds_scaled = np.stack([low_scaled, high_scaled], axis=1)  # (K,2)
    df = lhs_sample_model_inputs(
        model,
        n_samples,
        input_bounds=input_bounds_scaled,
        seed=seed,
        state_names=state_names,
        aug_names=aug_names,
    )

    # Determine scaler type for metadata
    if hasattr(scaler, "scale_") and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        scaler_type = "MinMaxScaler"
    elif hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
        scaler_type = "StandardScaler"
    else:
        scaler_type = "Unknown"

    # Apply gradient rescaling if requested
    if rescale_gradients:
        # Calculate time scale factor for chain rule
        if time_span_physical is not None:
            t0_phys, t1_phys = time_span_physical
            time_scale_max = float(t1_phys - t0_phys)
        else:
            time_scale_max = 1.0  # Default if no time span provided

        # Rescale state gradients back to physical units using chain rule
        if scaler_type == "MinMaxScaler":
            # MinMaxScaler case: dy/dt = (1 / scaler.scale_[i]) / time_scale_max × (dy_std/dt_std)
            scale_factors = np.asarray(scaler.scale_, float)

            for i, state_name in enumerate(state_names):
                grad_col = f"d{state_name}/dt"
                if grad_col in df.columns and i < len(scale_factors):
                    # Apply chain rule: multiply by (1/scale) and divide by time_scale_max
                    df[grad_col] = df[grad_col] * (1.0 / scale_factors[i]) / time_scale_max

        elif scaler_type == "StandardScaler":
            # StandardScaler case: dy/dt = scale_[i] × (dy_std/dt_std)
            scale_factors = np.asarray(scaler.scale_, float)

            for i, state_name in enumerate(state_names):
                grad_col = f"d{state_name}/dt"
                if grad_col in df.columns and i < len(scale_factors):
                    # Apply chain rule: multiply by scale factor
                    df[grad_col] = df[grad_col] * scale_factors[i]

        # Rescale input values back to physical units using inverse scaler transformation
        if scaler_type == "MinMaxScaler":
            # For MinMaxScaler: physical = (scaled - min_) / scale_
            scale_factors = np.asarray(scaler.scale_, float)
            min_shifts = np.asarray(scaler.min_, float)

            for i, state_name in enumerate(state_names):
                if state_name in df.columns and i < len(scale_factors):
                    # Apply inverse MinMax transformation
                    df[state_name] = (df[state_name] - min_shifts[i]) / scale_factors[i]

        elif scaler_type == "StandardScaler":
            # For StandardScaler: physical = scaled * scale_ + mean_
            scale_factors = np.asarray(scaler.scale_, float)
            means = np.asarray(scaler.mean_, float)

            for i, state_name in enumerate(state_names):
                if state_name in df.columns and i < len(scale_factors):
                    # Apply inverse StandardScaler transformation
                    df[state_name] = df[state_name] * scale_factors[i] + means[i]

        # Rescale time back to physical units if time is included
        if include_time and "t" in df.columns and time_physical_bounds is not None:
            t_lo_phys, t_hi_phys = time_physical_bounds
            df["t"] = t_lo_phys + df["t"] * (t_hi_phys - t_lo_phys)

    # Prepare bounds report
    rows = []
    for i, nm in enumerate(state_names):
        rows.append(
            {
                "name": nm,
                "kind": "state",
                "physical_low": float(low_phys[i]),
                "physical_high": float(high_phys[i]),
                "scaled_low": float(low_scaled_state[i]),
                "scaled_high": float(high_scaled_state[i]),
            }
        )
    for j, nm in enumerate(aug_names):
        if augment_size > 0:
            rows.append(
                {
                    "name": nm,
                    "kind": "aug",
                    "physical_low": np.nan,
                    "physical_high": np.nan,
                    "scaled_low": float(low_scaled_aug[j]),
                    "scaled_high": float(high_scaled_aug[j]),
                }
            )
    if include_time:
        rows.append(
            {
                "name": "t",
                "kind": "time",
                "physical_low": float(time_physical_bounds[0]) if time_physical_bounds is not None else np.nan,
                "physical_high": float(time_physical_bounds[1]) if time_physical_bounds is not None else np.nan,
                "scaled_low": float(t_lo_scaled),
                "scaled_high": float(t_hi_scaled),
            }
        )

    bounds_df = pd.DataFrame(rows)

    # Add scaler details to meta for reproducibility
    scaler_params = {}
    if scaler_type == "MinMaxScaler":
        scaler_params = {
            "min_": [float(x) for x in list(getattr(scaler, "min_", []))],
            "scale_": [float(x) for x in list(getattr(scaler, "scale_", []))],
            "data_min_": [float(x) for x in list(getattr(scaler, "data_min_", []))],
            "data_max_": [float(x) for x in list(getattr(scaler, "data_max_", []))],
        }
    elif scaler_type == "StandardScaler":
        scaler_params = {
            "mean_": [float(x) for x in list(getattr(scaler, "mean_", []))],
            "scale_": [float(x) for x in list(getattr(scaler, "scale_", []))],
            "var_": [float(x) for x in list(getattr(scaler, "var_", []))],
        }

    bounds_meta = {
        "scaler_type": scaler_type,
        "scaler_params": scaler_params,
        "scaler_note": scaler_note,
        "physical_bounds_note": pb_note,
        "gradients_rescaled": rescale_gradients,
        "time_scale_max": float(time_span_physical[1] - time_span_physical[0]) if time_span_physical else None,
        "feature_names": state_names + aug_names,
        "data_shape": (n_samples, len(state_names + aug_names)),
    }

    if excel_path is not None:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="samples", index=False)
            # Write bounds and metadata
            bounds_df.to_excel(writer, sheet_name="bounds", index=False)
            # Add a small metadata sheet for traceability
            meta_df = pd.DataFrame([bounds_meta])
            meta_df.to_excel(writer, sheet_name="meta", index=False)

            # Save unscaled version if requested
            if save_unscaled:
                # Create unscaled version by copying the existing scaled data
                # but without the gradient rescaling
                df_unscaled = df.copy()

                # For unscaled version, we want the data in the model's scaled space
                # The existing 'df' already has the properly sampled points, we just need
                # to remove the physical unit conversion if it was applied
                if rescale_gradients and scaler_type == "MinMaxScaler":
                    # Reverse the gradient rescaling to get back to model space
                    for i, name in enumerate(state_names):
                        grad_col = f"d{name}/dt"
                        if grad_col in df_unscaled.columns:
                            scale_factors = np.asarray(scaler.scale_, float)
                            time_scale_max = (
                                time_span_physical[1] - time_span_physical[0] if time_span_physical else 1.0
                            )
                            # Reverse: dy/dt_scaled = dy/dt_physical * scale[i] * time_scale_max
                            df_unscaled[grad_col] = df_unscaled[grad_col] * scale_factors[i] * time_scale_max

                df_unscaled.to_excel(writer, sheet_name="unscaled_samples", index=False)

    return df


if __name__ == "__main__":
    # --- Enhanced dataset generation with timestamped subfolder ---
    import json
    import os
    from datetime import datetime

    # Create timestamped subfolder in sr_datasets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_path = f"sr_datasets/{timestamp}"
    os.makedirs(subfolder_path, exist_ok=True)
    print(f"Created dataset subfolder: {subfolder_path}")

    # --- Save training and model configs as dictionaries ---
    training_config = {
        "C_TRAIN_SOURCE": C_TRAIN_SOURCE,
        "C_TEST": C_TEST,
        "lr_strategy": (2e-4, 4e-4),
        "steps_strategy": (600, 600),
        "length_strategy": (0.33, 1),
        "width_size": 32,
        "depth": 4,
        "activation": "swish",
        "ntimesteps": 5,
        "seed": 4700,
        "save_idxs": [0, 3],
        "noise": True,
        "noise_level": 0.00,
        "batch_size": "all",
        "augment_dim": 2,
        "nucl_params": base_system_kineticparams[0],
        "growth_params": base_system_kineticparams[1],
        "num_tsave_points": 50,  # Number of timepoints to save in dataset
    }

    model_config = {
        "data_size": model.func.mlp.layers[-1].weight.shape[0] - model.augment_size,
        "augment_size": model.augment_size,
        "width_size": model.func.mlp.layers[0].weight.shape[0],
        "depth": len(model.func.mlp.layers),
        "include_time": model.include_time,
        "solver": str(type(model.solver).__name__),
        "activation": "swish",  # from training config
    }

    # Save configs as JSON
    with open(f"{subfolder_path}/training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    with open(f"{subfolder_path}/model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print("Saved training and model configurations")

    # --- Save input/output bounds ---
    # Derive names and build default bounds from the trained model
    total_dim = model.func.mlp.layers[-1].weight.shape[0]
    augment_size = model.augment_size
    data_size = total_dim - augment_size
    state_names = [f"x{i}" for i in range(data_size)]
    aug_names = [f"a{i}" for i in range(augment_size)]

    # Input bounds (physical)
    try:
        if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
            print("Using MinMaxScaler data_min_/data_max_ for input bounds")
            input_bounds = {
                name: (float(lo), float(hi)) for name, lo, hi in zip(state_names, scaler.data_min_, scaler.data_max_)
            }
        elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            input_bounds = {
                name: (float(m - 3 * s), float(m + 3 * s))
                for name, m, s in zip(state_names, scaler.mean_, scaler.scale_)
            }
        else:
            input_bounds = dict.fromkeys(state_names, (-1.0, 1.0))
    except Exception:
        input_bounds = dict.fromkeys(state_names, (-1.0, 1.0))

    # Augmented bounds (scaled)
    aug_bounds_dict = {name: (float(aug_min[i]), float(aug_max[i])) for i, name in enumerate(aug_names)}

    # Output bounds (for completeness - can be estimated from data)
    output_bounds = {}
    for i, name in enumerate(state_names):
        all_data = np.concatenate([ys_train[:, :, i].flatten(), ys_test[:, :, i].flatten()])
        output_bounds[name] = (float(np.min(all_data)), float(np.max(all_data)))

    bounds_config = {
        "input_bounds_physical": input_bounds,
        "input_bounds_scaled_augmented": aug_bounds_dict,
        "output_bounds_physical": output_bounds,
        "time_bounds_physical": (0.0, 300.0),
        "time_bounds_scaled": (0.0, 1.0),
    }

    with open(f"{subfolder_path}/bounds_config.json", "w") as f:
        json.dump(bounds_config, f, indent=2)
    print("Saved input/output bounds")

    # --- Create LHS sampling file ---
    # Choose number of samples N and per-input ranges (unscaled physical space for state vars)
    N = 2**10

    # Time in physical units [s]; training used 0..300 s mapped to 0..1
    time_physical_bounds = (0.0, 300.0)
    time_span_physical = (0.0, 300.0)

    # Augmented dims have no scaler; provide scaled bounds directly using collected bounds
    aug_bounds_scaled = (float(np.min(aug_min)), float(np.max(aug_max)))

    excel_path_lhs = f"{subfolder_path}/lhs_samples.xlsx"

    df_lhs = lhs_sample_model_inputs_using_scaler(
        model,
        scaler,
        N,
        physical_bounds=input_bounds,
        aug_bounds_scaled=aug_bounds_scaled,
        time_physical_bounds=time_physical_bounds,
        time_span_physical=time_span_physical,
        excel_path=excel_path_lhs,
        seed=4701,
        state_names=state_names,
        aug_names=aug_names,
        rescale_gradients=True,
        save_unscaled=True,
    )

    print(f"LHS samples saved to {excel_path_lhs}: shape={df_lhs.shape}")

    # Add y_dataset sheet to the LHS Excel file
    y_dataset_df = create_y_dataset_sheet(model, ys_train, ts, state_names, aug_names)

    # Append y_dataset sheet to the existing Excel file
    with pd.ExcelWriter(excel_path_lhs, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        y_dataset_df.to_excel(writer, sheet_name="y_dataset", index=False)

    print(f"Added y_dataset sheet with {y_dataset_df.shape[0]} rows from {ys_train.shape[0]} experiments")

    # --- Create file with densely sampled training points from the trained model's trajectory ---
    # Get the number of timepoints for denser sampling from config
    num_tsave_points = training_config.get("num_tsave_points", 50)  # Default to 50 points

    # Create a denser time array for sampling the trained model's continuous trajectory
    # This spans the same time range as the original training data but provides much
    # better temporal resolution for symbolic regression analysis
    ts_dense = jnp.linspace(ts[0], ts[-1], num_tsave_points)

    # Print sampling information
    print(f"Original training timepoints: {len(ts)}")
    print(f"Dense sampling timepoints: {len(ts_dense)}")
    print(f"Time range: [{ts_dense[0]:.3f}, {ts_dense[-1]:.3f}] (scaled)")
    print(f"Physical time range: [{ts_dense[0] * 300:.1f}, {ts_dense[-1] * 300:.1f}] seconds")

    # Collect training data points at dense timepoints using the trained model
    training_points_data = []

    # Process all experiments together with proper vmap batching
    y0_batch = ys_train[:, 0, :]  # Shape: (num_experiments, data_size)

    # Get all trajectories at dense timepoints using the trained model
    dense_trajectories = jax.vmap(model, in_axes=(None, 0))(
        ts_dense, y0_batch
    )  # Shape: (num_experiments, num_tsave_points, data_size)

    # Get all augmented trajectories at dense timepoints
    dense_aug_trajectories = jax.vmap(model.get_augmented_trajectory, in_axes=(None, 0))(
        ts_dense, y0_batch
    )  # Shape: (num_experiments, num_tsave_points, data_size + augment_size)

    # Determine scaler type for gradient rescaling
    if hasattr(scaler, "scale_") and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        scaler_type = "MinMaxScaler"
    elif hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
        scaler_type = "StandardScaler"
    else:
        scaler_type = "Unknown"

    # Time scale factor for chain rule (physical time span is 300 seconds)
    time_scale_max = 300.0

    for exp_idx in range(ys_train.shape[0]):  # For each training experiment
        for t_idx in range(len(ts_dense)):  # For each dense timepoint
            # Get the state at this dense timepoint from model prediction
            state_at_tsave = dense_trajectories[exp_idx, t_idx, :]  # Shape: (data_size,)

            # Get augmented dimensions at this dense timepoint
            aug_at_tsave = dense_aug_trajectories[exp_idx, t_idx, data_size:]  # Augmented dimensions at tsave

            # Create record
            record = {
                "experiment_id": exp_idx,
                "tsave_idx": int(t_idx),
                "t": float(ts_dense[t_idx] * 300.0),  # Convert scaled to physical
                "time_scaled": float(ts_dense[t_idx]),
            }

            # Add state values (rescale back to physical units for consistency with LHS)
            if scaler_type == "MinMaxScaler":
                # For MinMaxScaler: physical = (scaled - min_) / scale_
                scale_factors = np.asarray(scaler.scale_, float)
                min_shifts = np.asarray(scaler.min_, float)
                for i, name in enumerate(state_names):
                    if i < len(scale_factors):
                        record[name] = float((state_at_tsave[i] - min_shifts[i]) / scale_factors[i])
                    else:
                        record[name] = float(state_at_tsave[i])
            elif scaler_type == "StandardScaler":
                # For StandardScaler: physical = scaled * scale_ + mean_
                scale_factors = np.asarray(scaler.scale_, float)
                means = np.asarray(scaler.mean_, float)
                for i, name in enumerate(state_names):
                    if i < len(scale_factors):
                        record[name] = float(state_at_tsave[i] * scale_factors[i] + means[i])
                    else:
                        record[name] = float(state_at_tsave[i])
            else:
                # No rescaling for unknown scaler types
                for i, name in enumerate(state_names):
                    record[name] = float(state_at_tsave[i])

            # Add augmented values (remain in scaled space as they have no physical meaning)
            for i, name in enumerate(aug_names):
                record[name] = float(aug_at_tsave[i])

            # Evaluate vector field at this point (using scaled inputs)
            y_aug_input = np.concatenate([state_at_tsave, aug_at_tsave])
            vector_field_output = model.func(ts_dense[t_idx], y_aug_input)

            # Add vector field outputs with gradient rescaling to match LHS samples
            for i, name in enumerate(state_names):
                grad_raw = float(vector_field_output[i])
                if scaler_type == "MinMaxScaler" and i < len(scale_factors):
                    # Apply chain rule: dy/dt = (1 / scale) / time_scale_max × (dy_std/dt_std)
                    record[f"d{name}/dt"] = grad_raw * (1.0 / scale_factors[i]) / time_scale_max
                elif scaler_type == "StandardScaler" and i < len(scale_factors):
                    # Apply chain rule: dy/dt = scale × (dy_std/dt_std)
                    record[f"d{name}/dt"] = grad_raw * scale_factors[i]
                else:
                    record[f"d{name}/dt"] = grad_raw

            # Add augmented gradients (no rescaling as augmented dims have no physical meaning)
            for i, name in enumerate(aug_names):
                record[f"d{name}/dt"] = float(vector_field_output[data_size + i])

            training_points_data.append(record)

    # Convert to DataFrame and save
    df_training_points = pd.DataFrame(training_points_data)
    excel_path_training = f"{subfolder_path}/training_points_tsave.xlsx"

    with pd.ExcelWriter(excel_path_training, engine="xlsxwriter") as writer:
        df_training_points.to_excel(writer, sheet_name="training_points", index=False)

        # Also save summary info with scaler details
        # Add scaler details to summary for reproducibility
        scaler_params = {}
        if scaler_type == "MinMaxScaler":
            scaler_params = {
                "min_": [float(x) for x in list(getattr(scaler, "min_", []))],
                "scale_": [float(x) for x in list(getattr(scaler, "scale_", []))],
                "data_min_": [float(x) for x in list(getattr(scaler, "data_min_", []))],
                "data_max_": [float(x) for x in list(getattr(scaler, "data_max_", []))],
            }
        elif scaler_type == "StandardScaler":
            scaler_params = {
                "mean_": [float(x) for x in list(getattr(scaler, "mean_", []))],
                "scale_": [float(x) for x in list(getattr(scaler, "scale_", []))],
                "var_": [float(x) for x in list(getattr(scaler, "var_", []))],
            }

        summary_data = {
            "num_experiments": ys_train.shape[0],
            "original_training_timepoints": len(ts),
            "dense_sampling_timepoints": len(ts_dense),
            "total_training_points": len(training_points_data),
            "tsave_times_physical": [float(t * 300.0) for t in ts_dense],
            "tsave_times_scaled": [float(t) for t in ts_dense],
            "scaler_type": scaler_type,
            "scaler_params": scaler_params,
            "time_scale_max": 300.0,  # Fixed time scale for training data
            "feature_names": state_names + aug_names,
        }
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_excel(writer, sheet_name="summary", index=False)

        # Save unscaled training points (without gradient rescaling)
        training_points_data_unscaled = []
        for record in training_points_data:
            unscaled_record = record.copy()

            # Reverse gradient rescaling for state dimensions
            if scaler_type == "MinMaxScaler":
                for i, name in enumerate(state_names):
                    grad_col = f"d{name}/dt"
                    if grad_col in unscaled_record and i < len(scale_factors):
                        # Reverse: dy/dt_scaled = dy/dt_physical * scale[i] * time_scale_max
                        unscaled_record[grad_col] = record[grad_col] * scale_factors[i] * time_scale_max
            elif scaler_type == "StandardScaler":
                for i, name in enumerate(state_names):
                    grad_col = f"d{name}/dt"
                    if grad_col in unscaled_record and i < len(scale_factors):
                        # Reverse: dy/dt_scaled = dy/dt_physical / scale[i]
                        unscaled_record[grad_col] = record[grad_col] / scale_factors[i]

            training_points_data_unscaled.append(unscaled_record)

        df_training_points_unscaled = pd.DataFrame(training_points_data_unscaled)
        df_training_points_unscaled.to_excel(writer, sheet_name="unscaled_training_points", index=False)

        # Add y_dataset sheet
        y_dataset_df = create_y_dataset_sheet(model, ys_train, ts, state_names, aug_names)
        y_dataset_df.to_excel(writer, sheet_name="y_dataset", index=False)

    print(f"Training points at dense timepoints saved to {excel_path_training}: shape={df_training_points.shape}")
    print("  - training_points (scaled with gradient rescaling)")
    print("  - unscaled_training_points (model space without rescaling)")
    print(f"  - y_dataset (concatenated original training data: {y_dataset_df.shape[0]} rows)")
    print("  - summary (metadata)")
    print(
        f"Summary: {len(training_points_data)} total points from {ys_train.shape[0]} experiments at {len(ts_dense)} dense timepoints"
    )
    print(
        f"Model was trained on {len(ts)} original timepoints, sampled densely to {len(ts_dense)} points for symbolic regression"
    )

    print(f"\nAll datasets saved to: {subfolder_path}")
    print("Files created:")
    print("  - training_config.json")
    print("  - model_config.json")
    print("  - bounds_config.json")
    print(f"  - lhs_samples.xlsx ({df_lhs.shape[0]} samples)")
    print("    sheets: samples, unscaled_samples, bounds, meta, y_dataset")
    print(f"  - training_points_tsave.xlsx ({df_training_points.shape[0]} points)")
    print("    sheets: training_points, unscaled_training_points, y_dataset, summary")

# %%
