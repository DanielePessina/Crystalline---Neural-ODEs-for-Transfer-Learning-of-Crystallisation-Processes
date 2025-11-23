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

