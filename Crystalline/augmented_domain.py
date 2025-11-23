"""System‑specific Augmented Neural ODEs with fixed embeddings.

This module extends the basic Augmented NODE by concatenating a
one‑hot (or otherwise provided) embedding representing the
crystallisation system.  The embeddings are treated as constants during
integration, enabling a single model to operate across multiple
systems.
"""

from collections.abc import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax  # https://github.com/deepmind/optax
from jax import Array
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from traitlets import Bool

from Crystalline.data_functions import dataloader, simulateCrystallisation
from Crystalline.plotting import (
    plot_loss_curves,
    splitplot_system_augmented_model_vs_data,
)


class SystemAugmentedFunc(eqx.Module):
    """
    Vector field for a System-Augmented Neural ODE.

    Defines dynamics over an augmented space [x; a; s] ∈ ℝ^{d + d_aug + d_sys},
    where s is the system embedding with zero time derivatives.

    Args:
        data_size: Dimensionality of the original input space.
        augment_size: Dimensionality of the augmented variables.
        system_embed_size: Dimensionality of the system embedding.
        width_size: Width of hidden layers in the MLP.
        depth: Number of hidden layers in the MLP.
        activation: Activation function used in the MLP.
        key: PRNGKey for initializing parameters.
    """

    mlp: eqx.nn.MLP
    system_embed_size: int
    include_time: bool = eqx.static_field()
    output_constraints: dict | None = eqx.static_field()

    def __init__(
        self,
        data_size: int,
        augment_size: int,
        system_embed_size: int,
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
        total_dim = data_size + augment_size + system_embed_size
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
        self.system_embed_size = system_embed_size

    def __call__(self, t: Array, y_aug_sys: Array, args: Array | None = None) -> Array:
        """
        Args:
            t: Time point
            y_aug_sys: Concatenated state [data; augment; system_embedding]

        Returns:
            Time derivatives [dy/dt; da/dt; ds/dt] where ds/dt = 0
        """
        if self.include_time:
            t_feat = jnp.atleast_1d(t)
            mlp_in = jnp.concatenate([y_aug_sys, t_feat], axis=-1)
        else:
            mlp_in = y_aug_sys
        # Get derivatives from MLP
        output = self.mlp(mlp_in)

        if self.output_constraints is not None:
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

        # Set system embedding derivatives to zero (constant system context)

        return output.at[-self.system_embed_size :].set(0.0)


class SystemAugmentedNeuralODE(eqx.Module):
    """
    System-Augmented Neural ODE model that integrates dynamics with system context.

    Args:
        data_size: Original data dimension (d).
        augment_size: Additional dimensions (d_aug).
        system_embed_size: System embedding dimension (d_sys).
        width_size: Width of hidden layers.
        depth: Number of hidden layers.
        num_systems: Number of unique systems (for embedding).
        solver: Diffrax ODE solver (defaults to Tsit5).
        key: PRNGKey for initialization.
        activation: Activation function for MLP.
    """

    func: SystemAugmentedFunc
    embedding_layer: eqx.nn.Embedding
    solver: diffrax.AbstractSolver = eqx.static_field()
    augment_size: int = eqx.static_field()
    system_embed_size: int = eqx.static_field()
    include_time: bool = eqx.static_field()

    def __init__(
        self,
        data_size: int,
        augment_size: int,
        system_embed_size: int,
        width_size: int,
        depth: int,
        num_systems: int,
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
        self.system_embed_size = system_embed_size
        self.include_time = include_time
        self.func = SystemAugmentedFunc(
            data_size,
            augment_size,
            system_embed_size,
            width_size,
            depth,
            include_time=include_time,
            key=key,
            activation=activation,
            output_constraints=output_constraints,
        )
        # Split key for embedding
        key, embed_key = jr.split(key)
        self.embedding_layer = eqx.nn.Embedding(
            num_embeddings=num_systems, embedding_size=system_embed_size, key=embed_key
        )
        self.solver = solver

    def __call__(self, ts: Array, y0: Array, *, system_id: int, return_augmented: Bool = False) -> Array:
        """
        Args:
            ts: Time points of shape (T,)
            y0: Initial condition in ℝ^d (original space), shape (d,)
            system_id: Integer system ID
            return_augmented: Whether to return full augmented trajectory

        Returns:
            Trajectory in original space ℝ^d, shape (T, d) or full augmented space
        """
        # Get embedding vector from embedding layer
        system_embedding = self.embedding_layer(jnp.array(system_id))
        y0_aug = jnp.concatenate([y0, jnp.zeros(self.augment_size, dtype=y0.dtype), system_embedding], axis=-1)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=0.01 * (ts[1] - ts[0]),
            y0=y0_aug,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4),
        )

        if return_augmented:
            return solution.ys  # shape (T, d+d_aug+d_sys)

        return solution.ys[..., : y0.shape[-1]]  # Return only original state portion


def create_system_embedding(system_params: dict, normalize: bool = True, verbose: bool = False) -> dict:
    """Build simple one‑hot embeddings for each crystallisation system.

    Parameters
    ----------
    system_params : dict[int, dict]
        Mapping from system identifier to its kinetic parameter dictionary.
    normalize : bool, optional
        If ``True`` the embeddings are standardised (zero mean, unit variance).
    verbose : bool, optional
        If ``True`` print summary information about the embeddings.

    Returns
    -------
    dict[int, Array]
        Dictionary mapping each ``system_id`` to its embedding vector.
    """
    embeddings = {}

    for system_id, params in system_params.items():
        # Concatenate nucleation and growth parameters
        # embedding = jnp.array(params["nucl_params"] + params["growth_params"])
        # Use one-hot encoding for system embeddings
        embedding = jnp.zeros(len(system_params))
        embedding = embedding.at[system_id].set(1.0)
        embeddings[system_id] = embedding

    if normalize:
        # Stack all embeddings to compute global normalization
        all_embeddings = jnp.stack([embeddings[i] for i in sorted(embeddings.keys())])
        mean = jnp.mean(all_embeddings, axis=0)
        std = jnp.std(all_embeddings, axis=0) + 1e-8  # Add small epsilon

        # Normalize each embedding
        for system_id in embeddings:
            embeddings[system_id] = (embeddings[system_id] - mean) / std

    if verbose:
        print("System embeddings created with normalization:", normalize)
        print("Embeddings shape:", {k: v.shape for k, v in embeddings.items()})
        print("Embeddings keys:", embeddings.keys())
        print("Embeddings example (system 0):", embeddings[0])
    return embeddings


def train_SystemAugNODE_fresh(
    training_data: list[tuple[list[float], int]],
    testing_data: list[tuple[list[float], int]],
    model=None,
    scalers: dict[int, MinMaxScaler | StandardScaler] | None = None,
    *,
    ntimesteps: int = 85,
    system_params: dict[int, dict] = None,
    noise: bool = False,
    noise_level: float = 0.1,
    lr_strategy: tuple[float, float] = (4e-3, 10e-3),
    steps_strategy: tuple[int, int] = (600, 600),
    length_strategy: tuple[float, float] = (0.33, 1),
    batch_size: int | str = 1,
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
    saveplot: bool = False,
    lossplot: bool = False,
    extratitlestring: str = "",
    save_idxs: list[int] = [0, 3],
    output_constraints: dict | None = None,
    system_embeddings: dict[int, Array] | None = None,  # Not used anymore
) -> tuple[Array, Array, Array, SystemAugmentedNeuralODE, dict[int, MinMaxScaler | StandardScaler] | None, list[dict]]:
    """Train a system‑aware Augmented NODE on multiple crystallisation systems.

    Parameters
    ----------
    training_data, testing_data : list[tuple[list[float], int]]
        Each entry contains a list of initial concentrations and the integer
        ``system_id``.  The ``system_id`` indexes ``system_params`` and the
        embedding table.
    scalers : dict[int, MinMaxScaler | StandardScaler] | None, optional
        Pre‑fitted scalers for each system.  If ``None`` new scalers are created
        from the first experiment of each system.
    system_params : dict[int, dict]
        Mapping of ``system_id`` to nucleation and growth parameters used for
        data simulation.
    system_embeddings : dict[int, Array] | None
        Optional table of precomputed system embeddings.

    Returns
    -------
    tuple
        ``(ts, ys_train, ys_test, model, scalers, metrics)`` where ``model`` is
        the trained :class:`SystemAugmentedNeuralODE` and ``metrics`` contains
        per‑system evaluation summaries.
    """
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    if batch_size == "all":
        batch_size = len(training_data)

    if system_params is None:
        raise ValueError("system_params must be provided as a dictionary mapping system_id to parameters.")

    num_systems = len(system_params)
    system_embed_size = 4  # Set as needed, or make it a parameter

    if scalers is None:
        scalers = {}

    # First pass: fit scalers for each system using first experiment from that system
    for initial_concs_list, system_id in training_data:
        if system_id not in scalers:
            params = system_params[system_id]
            # Use the first experiment from this system to fit the scaler
            _, ys_for_scaler, new_scaler = simulateCrystallisation(
                initial_concs_list,
                ntimesteps,
                scaler=None,
                key=data_key,
                nucl_params=params["nucl_params"],
                growth_params=params["growth_params"],
                save_idxs=save_idxs,
                noise=False,  # Fit scaler on noise-free data
                noise_level=0.0,
            )
            scalers[system_id] = new_scaler

    # Second pass: generate all training data using fitted scalers
    all_train_data = []
    all_train_system_ids = []

    for initial_concs_list, system_id in training_data:
        params = system_params[system_id]
        current_scaler = scalers[system_id]

        ts, ys, _ = simulateCrystallisation(
            initial_concs_list,
            ntimesteps,
            scaler=current_scaler,
            key=data_key,
            nucl_params=params["nucl_params"],
            growth_params=params["growth_params"],
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )
        all_train_data.append(ys)
        all_train_system_ids.extend([system_id] * len(initial_concs_list))

    ys_train = jnp.concatenate(all_train_data, axis=0)
    train_system_ids = jnp.array(all_train_system_ids)

    # Generate test data
    all_test_data = []
    all_test_system_ids = []

    for initial_concs_list, system_id in testing_data:
        params = system_params[system_id]
        current_scaler = scalers.get(system_id)
        if current_scaler is None:
            raise ValueError(
                f"Scaler for system_id {system_id} not found. "
                "Ensure all test systems have corresponding training data for scaler fitting."
            )

        ts_test, ys_test_sys, _ = simulateCrystallisation(
            initial_concs_list,
            ntimesteps,
            scaler=current_scaler,
            key=data_key,
            nucl_params=params["nucl_params"],
            growth_params=params["growth_params"],
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
        )
        all_test_data.append(ys_test_sys)
        all_test_system_ids.extend([system_id] * len(initial_concs_list))

    ys_test = jnp.concatenate(all_test_data, axis=0)
    test_system_ids = jnp.array(all_test_system_ids)

    _, length_size, data_size = ys_train.shape

    if model is None:
        model = SystemAugmentedNeuralODE(
            data_size,
            augment_dim,
            system_embed_size,
            width_size,
            depth,
            num_systems,
            include_time=include_time,
            key=model_key,
            activation=activation,
            solver=solver,
            output_constraints=output_constraints,
        )
    else:
        augment_dim = model.augment_size
        system_embed_size = model.system_embed_size

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi, system_ids_batch):
        def predict_single(y0, system_id_single):
            return model(ti, y0, system_id=system_id_single)

        y_pred = jax.vmap(predict_single)(yi[:, 0], system_ids_batch)
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, system_ids_batch, model, opt_state):
        loss, grads = grad_loss(model, ti, yi, system_ids_batch)
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
                optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adamw(lr, weight_decay=1e-3))
                opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

                _ts = ts[: int(length_size * length)]
                _ys_train_slice = ys_train[:, : int(length_size * length)]

                train_loader = dataloader((_ys_train_slice, train_system_ids), batch_size, key=loader_key)

                for step in range(steps):
                    yi_batch, batch_system_ids_loader = next(train_loader)
                    loss, model, opt_state = make_step(_ts, yi_batch, batch_system_ids_loader, model, opt_state)
                    train_losses.append(loss)

                    if verbose and ((step % print_every) == 0 or step == steps - 1):
                        _ys_test_slice = ys_test[:, : int(length_size * length)]
                        test_loss_val, _ = grad_loss(model, _ts, _ys_test_slice, test_system_ids)
                        test_losses.append(test_loss_val)
                        training_progress.update(
                            progress_task,
                            advance=1,
                            description=f"loss={loss:.3e} test={test_loss_val:.3e}",
                        )
                    else:
                        training_progress.advance(progress_task)

        training_progress.stop()

        def calculate_system_metrics(ys_data_metrics, system_ids_metrics, tag):
            metrics_list = []
            for i in range(len(ys_data_metrics)):
                system_id = int(system_ids_metrics[i])
                current_scaler = scalers.get(system_id)

                # No need to fetch embedding; just pass system_id
                y_pred_single = model(ts, ys_data_metrics[i, 0], system_id=system_id)

                y_pred_np = jnp.array(y_pred_single)
                y_true_np = jnp.array(ys_data_metrics[i])

                if current_scaler is not None:
                    y_pred_unscaled = current_scaler.inverse_transform(y_pred_np.reshape(-1, data_size)).reshape(
                        -1, data_size
                    )
                    y_true_unscaled = current_scaler.inverse_transform(y_true_np.reshape(-1, data_size)).reshape(
                        -1, data_size
                    )
                else:
                    print(
                        f"Warning: Scaler for system_id {system_id} not found during metrics for {tag} trajectory {i}. Using scaled values."
                    )
                    y_pred_unscaled = y_pred_np
                    y_true_unscaled = y_true_np

                rmse_overall = float(jnp.sqrt(jnp.mean((y_true_unscaled - y_pred_unscaled) ** 2)))
                rmse_concentration = float(jnp.sqrt(jnp.mean((y_true_unscaled[:, 0] - y_pred_unscaled[:, 0]) ** 2)))
                d43_idx = data_size - 1
                if d43_idx < 0:
                    d43_idx = 0
                if data_size > 1:
                    rmse_d43 = float(
                        jnp.sqrt(jnp.mean((y_true_unscaled[:, d43_idx] - y_pred_unscaled[:, d43_idx]) ** 2))
                    )
                else:
                    rmse_d43 = 0.0

                metrics_list.append(
                    {
                        "experiment_type": tag,
                        "system_id": system_id,
                        "trajectory_idx": i,
                        "rmse_overall": rmse_overall,
                        "rmse_concentration": rmse_concentration,
                        "rmse_d43": rmse_d43,
                        "width_size": width_size,
                        "depth": depth,
                        "augment_dim": augment_dim,
                        "system_embed_size": system_embed_size,
                    }
                )
            return metrics_list

        train_metrics = calculate_system_metrics(ys_train, train_system_ids, "Train")
        test_metrics = calculate_system_metrics(ys_test, test_system_ids, "Test")
        all_metrics = train_metrics + test_metrics

        if verbose:
            print("\n=== Metrics Summary ===")
            for exp_type in ["Train", "Test"]:
                exp_metrics = [m for m in all_metrics if m["experiment_type"] == exp_type]
                if exp_metrics:
                    avg_rmse = jnp.mean(jnp.array([m["rmse_overall"] for m in exp_metrics]))
                    print(f"{exp_type.upper()} - Average RMSE: {avg_rmse:.4f}")
                    for sys_id_unique in sorted({m["system_id"] for m in exp_metrics}):
                        sys_metrics_filtered = [m for m in exp_metrics if m["system_id"] == sys_id_unique]
                        sys_rmse = jnp.mean(jnp.array([m["rmse_overall"] for m in sys_metrics_filtered]))
                        print(f"  System {sys_id_unique}: RMSE = {sys_rmse:.4f} (n={len(sys_metrics_filtered)})")

    except (eqx.EquinoxRuntimeError, jax._src.linear_util.StoreException, ValueError) as e:
        print(f"Warning: Training failed due to {type(e).__name__}: {e}. Returning error metrics.")
        all_metrics = [{"error": True, "rmse_overall": 999.0, "error_message": str(e)}]
        return (None, None, None, None, scalers, all_metrics)

    if lossplot:
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"System-Augmented Neural ODE Training - {extratitlestring}",
            saveplot=saveplot,
            filename="plots/system_augmented_loss_curve.png",
        )

    if splitplot:
        splitplot_system_augmented_model_vs_data(
            ts,
            ys_train,
            ys_test,
            model,
            scalers,
            train_system_ids,
            test_system_ids,
            system_embeddings,
            length_strategy,
            f"System-Augmented Neural ODE - {extratitlestring}",
            saveplot,
            filename_prefix="system_augmented_predictions",
        )

    return ts, ys_train, ys_test, model, scalers, all_metrics


# %%

if __name__ == "__main__":
    # Define your training data with system IDs

    training_data = [
        ([15.5, 16.5, 19], 0),  # System 0 (base system with lots of data)
        ([16.5, 17], 1),  # System 1 (limited data)
        ([16.5, 17], 2),  # System 2 (limited data)
        ([16.5, 17], 3),  # System 3 (limited data)
    ]

    testing_data = [
        ([14, 17.7, 20], 0),
        ([14, 17.7, 20], 1),
        ([14, 17.7, 20], 2),
        ([14, 17.7, 20], 3),
    ]

    ### ACS System Params

    ### ACS Parameters
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

    system_params = {
        0: {"nucl_params": [39.81, 0.675], "growth_params": [0.345, 3.344]},
        1: {"nucl_params": [29.9, 0.49], "growth_params": [0.37, 3.3]},
        2: {"nucl_params": [22.7, 0.14], "growth_params": [0.37, 3.3]},
        3: {"nucl_params": [27.4, 0.32], "growth_params": [0.37, 3.3]},
    }

    system_embeddings = create_system_embedding(system_params, normalize=True)

    print("System embeddings created:")
    for sys_id, embedding in system_embeddings.items():
        print(f"System {sys_id}: Embedding = {embedding}")
    # [I 2025-05-27 21:40:35,424] Trial 20
    #
    #  finished with value: 0.20716053729230913 and parameters: {'lr3': 0.0046294866733420205, 'lr4': 0.004814324632931157, 'epochs3': 912, 'epochs4': 876}. Best is trial 20 with value: 0.20716053729230913.
    # Train the model

    # Trial 358 finished with value: 0.030931512997258926 and
    #
    #
    # parameters: {'lr1': 0.008722025235804278, 'lr2': 0.0005310886539385006, 'epochs1': 1200, 'epochs2': 1200, 'width_size': 56, 'depth': 5, 'augment_dim': 2}. Best is trial 358 with value: 0.030931512997258926.
    # [I 2025-05-28 05:20:08,724] Trial 873 finished with value: 0.0278025394444086 and
    # parameters: {'lr1': 0.008999958796627959, 'lr2': 0.00038813799135356227, 'epochs1': 1100, 'epochs2': 1200, 'width_size': 64, 'depth': 5, 'augment_dim': 4}. Best is trial 873 with value: 0.0278025394444086.

    # Trial 78 finished with value: 0.021140020379767404 and
    # parameters: {'lr1': 0.006225181986804846, 'lr2': 0.00045524431834495494,
    # 'epochs1': 1700, 'epochs2': 1300, 'width_size': 96, 'depth': 6, 'augment_dim': 4}. Best is trial 78 with value: 0.021140020379767404.
    results = train_SystemAugNODE_fresh(
        # [training_data[0]],
        training_data,
        testing_data,
        augment_dim=4,
        width_size=32,
        depth=3,
        ntimesteps=28,
        batch_size="all",
        noise_level=0.0,
        lr_strategy=(
            1e-3,
            0.00045524431834495494,
        ),
        steps_strategy=(600, 600),
        length_strategy=(0.33, 1),
        splitplot=True,
        lossplot=True,
        extratitlestring="Fresh Training",
        system_params=system_params,
    )

    # Print metrics summary for each system
    if results[-1] is not None:
        all_metrics = results[-1]
        print("\n=== Per-System Metrics Summary ===")
        for sys_id in {m["system_id"] for m in all_metrics if "system_id" in m}:
            sys_metrics = [m for m in all_metrics if m.get("system_id") == sys_id]
            if sys_metrics:
                avg_rmse = jnp.mean(jnp.array([m["rmse_overall"] for m in sys_metrics]))
                print(f"System {sys_id}: Average RMSE = {avg_rmse:.4f} (n={len(sys_metrics)})")
                avg_rmse_conc = jnp.mean(jnp.array([m["rmse_concentration"] for m in sys_metrics]))
                avg_rmse_d43 = jnp.mean(jnp.array([m["rmse_d43"] for m in sys_metrics]))
                print(f"    Average Concentration RMSE = {avg_rmse_conc:.4f}")
                print(f"    Average d43 RMSE = {avg_rmse_d43:.4f}")
                # for m in sys_metrics:
                #     print(f"  [{m['experiment_type']}] Trajectory {m['trajectory_idx']}: RMSE = {m['rmse_overall']:.4f}")

    # results = train_SystemAugNODE_TL(
    #     training_data[1:],
    #     testing_data[1:],
    #     results[3],
    #     results[4],
    #     idx_frozen="last",
    #     ntimesteps=28,
    #     batch_size="all",
    #     noise_level=0.0,
    #     lr_strategy=(4.6e-3, 4.8e-4,),
    #     steps_strategy=(900, 900),
    #     splitplot=True,
    #     lossplot=True,
    #     extratitlestring="Transferred Training",
    #     system_params=system_params,
    # )

# %%
