"""
data_functions.py

This module provides core data utilities for Neural ODE modeling of crystallization dynamics, including:

- dataloader: A JAX-based infinite generator for batching aligned datasets, suitable for mini-batch training.
- simulateCrystallisation: A function to simulate crystallization experiments using the moment method ODE,
    with optional data scaling and noise injection. This serves as the main data generation ("getData") function.

These utilities support model training, evaluation, and experimentation workflows for Neural ODEs in crystallization research.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

# --- My own imports ---
from Crystalline.ode_mom import solve_MoM_ODE_icarr_outputidx

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX operations


def dataloader(arrays, batch_size, *, key):
    """
    Constructs a JAX-based infinite generator for mini-batch training from one or more aligned datasets.

    Args:
        arrays: Tuple of input arrays, each with shape (N, ...).
        batch_size: Number of samples per batch.
        key: PRNGKey for sampling permutations.

    Returns:
        A Python generator yielding batches of the same shape as input arrays.
    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)

    def data_generator(key):
        current_key = key
        while True:
            current_key, subkey = jr.split(current_key)
            perm = jr.permutation(subkey, indices)

            start = 0
            while start < dataset_size:  # Changed condition to ensure we always process data
                end = min(start + batch_size, dataset_size)
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end

    return data_generator(key)


def simulateCrystallisation(
    list_initialconcentrations: list[float],
    ntimesteps: float,
    scaler: str | BaseEstimator | None = None,
    *,
    key: jax.Array,
    nucl_params: list[float] | tuple[float, float],
    growth_params: list[float] | tuple[float, float],
    save_idxs: list[int] = [0, 3],
    noise: bool = False,
    noise_level: float = 0.1,
    masked: bool = False,
    length_strategy: tuple[float, float] = (0.33, 1),
) -> tuple[jnp.ndarray, jnp.ndarray, BaseEstimator | None]:
    """
    Simulates crystallization dynamics using the moment method (MoM) ODE with optional data scaling.

    Args:
        list_initialconcentrations: List of initial concentrations (float per experiment).
        ntimesteps: Number of temporal discretization points.
        scaler: Optional scaler. If None, creates and fits new MinMaxScaler. If "no_scale", disables scaling.
                If sklearn scaler, uses it for transformation.
        key: JAX PRNG key for reproducibility.
        nucl_params: Parameters controlling nucleation (list or tuple of floats).
        growth_params: Parameters controlling growth (list or tuple of floats).
        save_idxs: Indices of variables to retain from MoM output.
        noise: Whether to add Gaussian noise to the data.
        noise_level: Scaling factor for noise magnitude.
        masked: If True, zeroes the particle-size feature during the initial
            portion of the trajectory defined by ``length_strategy``.
        length_strategy: Tuple specifying the fraction of the trajectory used
            for the first and second training phases when ``masked`` is True.

    Returns:
        ts: Normalized time points (0 to 1), shape (ntimesteps,)
        ys: Scaled or unscaled simulation outputs, shape (n_experiments, ntimesteps, n_outputs)
        scaler: Fitted MinMaxScaler instance, or provided scaler, or None if no scaling
    """
    concentration_values = list_initialconcentrations
    simulation_timespace = jnp.linspace(0, 300, int(ntimesteps) + 1)

    ys_unscaled = solve_MoM_ODE_icarr_outputidx(
        concentration_values,
        nucl_params,
        growth_params,
        simulation_timespace,
        save_idxs,
    )
    # Add Gaussian noise if requested BEFORE scaling
    # Add Gaussian noise if requested BEFORE scaling
    if noise:
        noise_key, _ = jr.split(key)
        # Generate independent noise for each experiment
        n_experiments = ys_unscaled.shape[0]
        noise_keys = jr.split(noise_key, n_experiments)

        # Apply noise independently to each experiment
        noise_values = jnp.zeros_like(ys_unscaled)
        for i in range(n_experiments):
            # Compute pointwise noise: noise_level% of each value's magnitude (avoid zero by adding small epsilon)
            epsilon = 1e-8
            experiment_data = ys_unscaled[i]
            noise_scale = jnp.abs(experiment_data) * noise_level + epsilon
            experiment_noise = jr.normal(noise_keys[i], experiment_data.shape) * noise_scale
            noise_values = noise_values.at[i].set(experiment_noise)

        # Ensure no noise is added to the initial condition
        if ys_unscaled.shape[1] > 0:
            noise_values = noise_values.at[:, 0, :].set(0.0)

        ys_unscaled = ys_unscaled + noise_values
        ys_unscaled = jnp.maximum(ys_unscaled, 0)

    if masked:
        T1 = int(length_strategy[0] * ntimesteps)
        # Handle masking for both jax and numpy arrays
        if isinstance(ys_unscaled, jnp.ndarray):
            ys_unscaled = ys_unscaled.at[:, :T1, 1].set(0.0)  # JAX array update
        else:
            ys_unscaled[:, :T1, 1] = 0.0  # NumPy array update

    # Reshape for scaling
    reshaped = ys_unscaled.reshape(-1, ys_unscaled.shape[-1])

    # Convert to NumPy array for sklearn compatibility
    reshaped_np = np.array(reshaped)

    # Scale data if scaler is provided or create new scaler
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(reshaped_np)
    elif isinstance(scaler, str) and scaler == "no_scale":
        scaled = reshaped_np
        scaler = None
    elif isinstance(scaler, MinMaxScaler | StandardScaler):
        scaled = scaler.transform(reshaped_np)
    else:
        raise ValueError("scaler must be None, 'no_scale', or a fitted sklearn TransformerMixin.")

    # Reshape back to original shape and convert to JAX array
    ys = jnp.asarray(scaled.reshape(ys_unscaled.shape))
    ts = jnp.linspace(0, 1, ys.shape[1])

    return ts, ys, scaler


# def train_NODE_TL(
#     training_initialconcentrations,
#     testing_initialconcentrations,
#     model,
#     base_scaler,
#     idx_frozen,  # Can be an integer, a tuple representing a slice, "last", or "none"
#     freeze_mode="both",  # New parameter: 'weights', 'biases', or 'both'
#     *,
#     lr_strategy=(4e-3, 10e-3),
#     steps_strategy=(600, 600),
#     length_strategy=(0.33, 1),
#     seed=467,
#     verbose=True,
#     print_every=301,
#     ntimesteps=85,
#     nucl_params=[39.81, 0.675],
#     growth_params=[0.345, 3.344],
#     plot=False,
#     splitplot=False,
#     saveplot=False,
#     extratitlestring="",
#     save_idxs=[0, 3],
#     noise=False,
#     noise_level=0.1,
# ):
#     """
#     Performs transfer learning on a pre-trained NeuralODE, with optional layer freezing.

#     Args:
#         model: A pre-trained NeuralODE model.
#         idx_frozen: Layer(s) to freeze ("last", "none", int, or tuple slice).
#         freeze_mode: Whether to freeze "weights", "biases", or "both".

#     Returns:
#         Same format as NODE_train_from_scratch.
#     """
#     key = jr.PRNGKey(seed)
#     data_key, model_key, loader_key, split_key, init_key = jr.split(key, 5)

#     batch_size = 1
#     dataset_size = len(training_initialconcentrations)

#     # Derive model properties instead of passing them if possible
#     # Assuming model.func.mlp gives access to the MLP
#     model_depth = len(model.func.mlp.layers)
#     # Assuming the first layer's weight shape gives width_size (output dim)
#     model_width = model.func.mlp.layers[0].weight.shape[0]
#     # Activation and solver are part of the passed model object

#     # Generate training and testing data separately
#     ts, ys_train, scaler = simulateCrystallisation(
#         training_initialconcentrations,
#         ntimesteps,
#         base_scaler,
#         key=data_key,
#         nucl_params=nucl_params,
#         growth_params=growth_params,
#         save_idxs=save_idxs,
#         noise=noise,
#         noise_level=noise_level,
#     )

#     ts_test, ys_test, _ = simulateCrystallisation(  # Use same scaler from training
#         testing_initialconcentrations,
#         ntimesteps,
#         base_scaler,
#         key=data_key,
#         nucl_params=nucl_params,
#         growth_params=growth_params,
#         save_idxs=save_idxs,
#         noise=noise,
#         noise_level=noise_level,
#     )

#     # Get dimensions from the data
#     _, length_size, data_size = ys_train.shape

#     # Determine the indices of layers to freeze
#     num_layers = len(model.func.mlp.layers)
#     if idx_frozen == "none":
#         frozen_indices = set()  # Empty set means no layers are frozen
#     elif idx_frozen == "last":
#         frozen_indices = {num_layers - 1}
#     elif idx_frozen == "last_two":
#         frozen_indices = {num_layers - 1, num_layers - 2}
#     elif idx_frozen == "all":
#         frozen_indices = set(range(num_layers))
#     elif isinstance(idx_frozen, int):
#         frozen_indices = {idx_frozen}
#     elif isinstance(idx_frozen, tuple):
#         frozen_slice = slice(*idx_frozen)
#         frozen_indices = set(range(num_layers)[frozen_slice])
#     else:
#         raise TypeError(
#             "idx_frozen must be an integer, a tuple representing a slice, the string 'last', or the string 'none'"
#         )

#     if freeze_mode not in ["weights", "biases", "both"]:
#         raise ValueError("freeze_mode must be 'weights', 'biases', or 'both'")

#     # Freeze specified layers/parameters - keep others trainable
#     filter_spec = jtu.tree_map(lambda _: True, model)  # Set all params trainable by default
#     for idx in frozen_indices:
#         if 0 <= idx < num_layers:
#             freeze_weights = freeze_mode in ["weights", "both"]
#             freeze_biases = freeze_mode in ["biases", "both"]
#             # Use eqx.tree_at multiple times or structure it carefully

#             if freeze_weights:
#                 filter_spec = eqx.tree_at(
#                     lambda tree: tree.func.mlp.layers[idx].weight,
#                     filter_spec,
#                     replace=False,
#                 )
#             if freeze_biases:
#                 filter_spec = eqx.tree_at(
#                     lambda tree: tree.func.mlp.layers[idx].bias,
#                     filter_spec,
#                     replace=False,
#                 )

#         else:
#             print(f"Warning: Index {idx} out of bounds for model with {num_layers} layers. Skipping.")

#     # Define loss calculation function
#     def calculate_loss(model, ti, yi):
#         y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
#         return jnp.mean((yi - y_pred) ** 2)

#     @eqx.filter_jit
#     def make_step(ti, yi, model, opt_state):
#         diff_model, static_model = eqx.partition(model, filter_spec)

#         @eqx.filter_value_and_grad
#         def loss_fn(diff_model):
#             # Recombine the model for forward pass
#             full_model = eqx.combine(diff_model, static_model)
#             return calculate_loss(full_model, ti, yi)

#         loss_value, grads = loss_fn(diff_model)
#         updates, opt_state = optim.update(grads, opt_state, params=diff_model)
#         diff_model = eqx.apply_updates(diff_model, updates)
#         model = eqx.combine(diff_model, static_model)
#         return loss_value, model, opt_state

#     for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
#         optim = optax.chain(optax.clip_by_global_norm(1e-1), optax.adamw(lr, weight_decay=1e-2))

#         # Initialize optimizer state with only trainable parameters
#         diff_model, static_model = eqx.partition(model, filter_spec)
#         opt_state = optim.init(eqx.filter(diff_model, eqx.is_inexact_array))

#         _ts = ts[: int(length_size * length)]
#         _ys_train = ys_train[:, : int(length_size * length)]
#         for step, (yi,) in zip(range(steps), dataloader((_ys_train,), batch_size, key=loader_key)):
#             start = time.time()
#             loss, model, opt_state = make_step(_ts, yi, model, opt_state)
#             end = time.time()

#             if verbose and ((step % print_every) == 0 or step == steps - 1):
#                 # Calculate test loss directly
#                 _ys_test = ys_test[:, : int(length_size * length)]
#                 test_loss = calculate_loss(model, _ts, _ys_test)
#                 print(
#                     f"Step: {step}, Train Loss: {loss:.4e}, Test Loss: {test_loss:.4e}, Computation time: {end - start:.4e}"
#                 )
#     # Calculate final test loss
#     final_test_loss = calculate_loss(model, ts, ys_test)

#     # Use new metrics function
#     metrics_train = calculate_all_metrics(
#         ts,
#         ys_train,
#         model,
#         scaler,
#         training_initialconcentrations,
#         "Train",
#         ntimesteps,
#         noise_level if noise else 0.0,
#     )
#     metrics_test = calculate_all_metrics(
#         ts,
#         ys_test,
#         model,
#         scaler,
#         testing_initialconcentrations,
#         "Test",
#         ntimesteps,
#         noise_level if noise else 0.0,
#     )
#     metrics_list = [
#         dict(
#             **m,
#             **{
#                 "Training_Experiments": len(training_initialconcentrations),
#                 "Training_Timepoints": ntimesteps,
#                 "Network_Width": model_width,
#                 "Network_Depth": model_depth,
#                 "Learning_Rate_1": lr_strategy[0] if len(lr_strategy) > 0 else None,
#                 "Learning_Rate_2": lr_strategy[1] if len(lr_strategy) > 1 else None,
#                 "Epochs_1": steps_strategy[0] if len(steps_strategy) > 0 else None,
#                 "Epochs_2": steps_strategy[1] if len(steps_strategy) > 1 else None,
#                 "Final_Train_Loss": loss,
#                 "Final_Test_Loss": final_test_loss,
#             },
#         )
#         for m in (metrics_train + metrics_test)
#     ]

#     train_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Train")
#     test_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Test")

#     rmse_str = f"RMSE - Train: {train_rmse:.4f} Test: {test_rmse:.4f}"

#     extratitlestring = f"{extratitlestring} (Fresh)\n{rmse_str}"

#     if plot and splitplot:
#         splitplot_model_vs_data(
#             ts,
#             ys_train,
#             ys_test,
#             model,
#             scaler,
#             length_strategy,
#             extratitlestring,
#             saveplot,
#         )
#     return ts, ys_train, ys_test, model, scaler, metrics_list


# def train_NODE_fresh_ensemble(
#     num_models: int,
#     base_seed: int,
#     training_initialconcentrations,
#     testing_initialconcentrations,
#     *,
#     lr_strategy=(4e-3, 10e-3),
#     steps_strategy=(600, 600),
#     length_strategy=(0.33, 1),
#     width_size=100,
#     depth=4,
#     activation=jnn.swish,
#     solver=diffrax.Tsit5(),
#     verbose=False,  # Set verbose=False for ensemble training to avoid excessive output
#     print_every=10000,  # Avoid printing during ensemble runs unless debugging
#     ntimesteps=85,
#     nucl_params=[39.81, 0.675],
#     growth_params=[0.345, 3.344],
#     save_idxs=[0, 3],
#     noise=False,
#     noise_level=0.1,
#     ensembleplot=False,
#     ensemblesaveplot=False,
#     **kwargs,  # Allow passing other args to NODE_train_from_scratch if needed
# ):
#     """
#     Trains an ensemble of NeuralODEs from scratch using different seeds.

#     Useful for uncertainty quantification and robustness evaluation.
#     """
#     results_list = []
#     print(f"Starting ensemble training for {num_models} models...")
#     for i in tqdm(range(num_models)):
#         current_seed = base_seed + i
#         # print(f"Training model {i + 1}/{num_models} with seed {current_seed}...")
#         try:
#             res_tuple = train_NODE_fresh(
#                 training_initialconcentrations=training_initialconcentrations,
#                 testing_initialconcentrations=testing_initialconcentrations,
#                 lr_strategy=lr_strategy,
#                 steps_strategy=steps_strategy,
#                 length_strategy=length_strategy,
#                 width_size=width_size,
#                 depth=depth,
#                 seed=current_seed,  # Use the incremented seed
#                 activation=activation,
#                 solver=solver,
#                 verbose=verbose,
#                 print_every=print_every,
#                 ntimesteps=ntimesteps,
#                 nucl_params=nucl_params,
#                 growth_params=growth_params,
#                 plot=False,  # Disable plotting within the loop
#                 save_idxs=save_idxs,
#                 noise=noise,
#                 noise_level=noise_level,
#                 **kwargs,
#             )
#             # Check if training failed (indicated by None values)
#             if res_tuple[0] is not None:
#                 results_list.append(res_tuple)
#             else:
#                 print(f"Warning: Model {i + 1} (seed {current_seed}) failed training. Skipping.")

#         except Exception as e:
#             print(f"Error training model {i + 1} (seed {current_seed}): {e}. Skipping.")

#     print(f"Ensemble training finished. Successfully trained {len(results_list)} models.")
#     # Plot ensemble predictions if requested
#     if len(results_list) > 0 and ensembleplot:
#         ts, ys_train, ys_test, _, scaler, _ = results_list[0]
#         models = [r[3] for r in results_list]
#         plot_ensemble_predictions(
#             ts,
#             ys_train,
#             models,
#             scaler,
#             "Ensemble Prediction (Train)",
#             ensemblesaveplot,
#             "ensemble_train",
#         )
#         plot_ensemble_predictions(
#             ts,
#             ys_test,
#             models,
#             scaler,
#             "Ensemble Prediction (Test)",
#             ensemblesaveplot,
#             "ensemble_test",
#         )
#     return results_list


# def train_NODE_TL_ensemble(
#     num_models: int,
#     base_seed: int,
#     training_initialconcentrations,
#     testing_initialconcentrations,
#     base_model,  # The pre-trained model to start TL from
#     base_scaler,  # The scaler associated with the base_model
#     idx_frozen,
#     freeze_mode="both",
#     *,
#     lr_strategy=(4e-3, 10e-3),
#     steps_strategy=(600, 600),
#     length_strategy=(0.33, 1),
#     verbose=False,  # Set verbose=False for ensemble training
#     print_every=10000,  # Avoid printing during ensemble runs
#     ntimesteps=85,
#     nucl_params=[39.81, 0.675],  # Target domain parameters
#     growth_params=[0.345, 3.344],  # Target domain parameters
#     save_idxs=[0, 3],
#     noise=False,
#     noise_level=0.1,
#     ensembleplot=False,
#     ensemblesaveplot=False,
#     **kwargs,  # Allow passing other args to NODE_TL_IDXFrozen if needed
# ):
#     """
#     Trains an ensemble of transfer-learned NeuralODEs with specified layer freezing.

#     Starts from a common base model and evaluates adaptation across seeds and experiments.
#     """
#     results_list = []
#     print(f"Starting ensemble TL training for {num_models} models...")
#     print(f"Base model: {type(base_model)}, Frozen layers: {idx_frozen}, Freeze mode: {freeze_mode}")

#     for i in tqdm(range(num_models)):
#         current_seed = base_seed + i
#         # print(f"Training TL model {i + 1}/{num_models} with seed {current_seed}...")

#         # Important: Create a deep copy of the base model for each ensemble member
#         # to avoid modifying the original or interfering between members.
#         # Equinox models are pytrees, so jax.tree_util.tree_map can copy.
#         current_model = jtu.tree_map(lambda x: x, base_model)

#         try:
#             res_tuple = train_NODE_TL(
#                 training_initialconcentrations=training_initialconcentrations,
#                 testing_initialconcentrations=testing_initialconcentrations,
#                 model=current_model,  # Use the copied model
#                 base_scaler=base_scaler,
#                 idx_frozen=idx_frozen,
#                 freeze_mode=freeze_mode,
#                 lr_strategy=lr_strategy,
#                 steps_strategy=steps_strategy,
#                 length_strategy=length_strategy,
#                 seed=current_seed,  # Use the incremented seed
#                 verbose=verbose,
#                 print_every=print_every,
#                 ntimesteps=ntimesteps,
#                 nucl_params=nucl_params,
#                 growth_params=growth_params,
#                 plot=False,  # Disable plotting within the loop
#                 save_idxs=save_idxs,
#                 noise=noise,
#                 noise_level=noise_level,
#                 **kwargs,
#             )
#             # Check if training failed (indicated by None values in NODE_train_from_scratch error handling)
#             # NODE_TL_IDXFrozen doesn't currently have the same explicit error return,
#             # but we can check if the first element (ts) is None if added later, or just assume success if no exception.
#             # For now, assume success if no exception.
#             results_list.append(res_tuple)

#         except Exception as e:
#             print(f"Error training TL model {i + 1} (seed {current_seed}): {e}. Skipping.")

#     print(f"Ensemble TL training finished. Successfully trained {len(results_list)} models.")
#     # Plot ensemble predictions if requested
#     if len(results_list) > 0 and ensembleplot:
#         ts, ys_train, ys_test, _, scaler, _ = results_list[0]
#         models = [r[3] for r in results_list]
#         plot_ensemble_predictions(
#             ts,
#             ys_train,
#             models,
#             scaler,
#             "Ensemble Prediction (Train)",
#             ensemblesaveplot,
#             "ensemble_train",
#         )
#         plot_ensemble_predictions(
#             ts,
#             ys_test,
#             models,
#             scaler,
#             "Ensemble Prediction (Test)",
#             ensemblesaveplot,
#             "ensemble_test",
#         )
#     return results_list


# class CustomMLP(eqx.Module):
#     layers: list
#     activation: callable = eqx.static_field()

#     def __init__(self, in_size, hidden_layers, out_size, activation, *, key):
#         keys = jr.split(key, len(hidden_layers) + 1)
#         # Add 1 to in_size for time variable t
#         layer_sizes = [in_size + 1] + hidden_layers + [out_size]
#         self.layers = [
#             eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1], key=keys[i]) for i in range(len(layer_sizes) - 1)
#         ]
#         self.activation = activation

#     def __call__(self, t, x):
#         # Concatenate time variable t (as array) to input x
#         x = jnp.concatenate([jnp.atleast_1d(t), x])
#         for lyr in self.layers[:-1]:
#             x = self.activation(lyr(x))
#         x = self.layers[-1](x)
#         return x


# class FuncCustom(eqx.Module):
#     mlp: CustomMLP

#     def __init__(self, data_size, hidden_layers, *, key, activation=jnn.softplus, **kwargs):
#         super().__init__(**kwargs)
#         self.mlp = CustomMLP(
#             in_size=data_size,
#             hidden_layers=hidden_layers,
#             out_size=data_size,
#             activation=activation,
#             key=key,
#         )

#     def __call__(self, t, y, args):
#         return self.mlp(t, y)


# class NeuralODECustom(eqx.Module):
#     func: FuncCustom
#     solver: diffrax.AbstractSolver = eqx.static_field()

#     def __init__(
#         self,
#         data_size,
#         hidden_layers,
#         *,
#         key,
#         activation=jnn.swish,
#         solver=diffrax.Tsit5(),
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         # data_size is the size of y, but CustomMLP expects in_size + 1 (for t) inside FuncCustom
#         self.func = FuncCustom(data_size, hidden_layers, key=key, activation=activation)
#         self.solver = solver

#     def __call__(self, ts, y0):
#         solution = diffrax.diffeqsolve(
#             diffrax.ODETerm(self.func),
#             self.solver,
#             t0=ts[0],
#             t1=ts[-1],
#             dt0=0.1 * (ts[1] - ts[0]),
#             y0=y0,
#             stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-5),
#             saveat=diffrax.SaveAt(ts=ts),
#         )
#         return solution.ys


# def train_NODE_fresh_custom_layers(
#     training_initialconcentrations,
#     testing_initialconcentrations,
#     *,
#     hidden_layers=[100, 100, 100, 100],  # Example: 4 hidden layers of 100 neurons each
#     seed=467,
#     activation=jnn.swish,
#     solver=diffrax.Tsit5(),
#     verbose=True,
#     print_every=301,
#     ntimesteps=85,
#     nucl_params=[39.81, 0.675],
#     growth_params=[0.345, 3.344],
#     plot=False,
#     splitplot=False,
#     saveplot=False,
#     lossplot=False,
#     extratitlestring="",
#     save_idxs=[0, 3],
#     noise=False,
#     noise_level=0.1,
#     lr_strategy=(4e-3, 10e-3),
#     steps_strategy=(600, 600),
#     length_strategy=(0.33, 1),
# ):
#     key = jr.PRNGKey(seed)
#     data_key, model_key, loader_key, split_key = jr.split(key, 4)

#     batch_size = 1

#     ts, ys_train, scaler = simulateCrystallisation(
#         training_initialconcentrations,
#         ntimesteps,
#         key=data_key,
#         nucl_params=nucl_params,
#         growth_params=growth_params,
#         save_idxs=save_idxs,
#         noise=noise,
#         noise_level=noise_level,
#     )

#     ts_test, ys_test, _ = simulateCrystallisation(
#         testing_initialconcentrations,
#         ntimesteps,
#         scaler,
#         key=data_key,
#         nucl_params=nucl_params,
#         growth_params=growth_params,
#         save_idxs=save_idxs,
#         noise=noise,
#         noise_level=noise_level,
#     )

#     _, length_size, data_size = ys_train.shape

#     model = NeuralODECustom(
#         data_size,
#         hidden_layers,
#         key=model_key,
#         activation=activation,
#         solver=solver,
#     )

#     @eqx.filter_value_and_grad
#     def grad_loss(model, ti, yi):
#         y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
#         return jnp.mean((yi - y_pred) ** 2)

#     @eqx.filter_jit
#     def make_step(ti, yi, model, opt_state):
#         loss, grads = grad_loss(model, ti, yi)
#         updates, opt_state = optim.update(grads, opt_state, params=model)
#         model = eqx.apply_updates(model, updates)
#         return loss, model, opt_state

#     try:
#         train_losses = []
#         test_losses = []

#         for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy, strict=True):
#             optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
#             opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
#             _ts = ts[: int(length_size * length)]
#             _ys_train = ys_train[:, : int(length_size * length)]
#             for step, (yi,) in zip(range(steps), dataloader((_ys_train,), batch_size, key=loader_key)):
#                 start = time.time()
#                 loss, model, opt_state = make_step(_ts, yi, model, opt_state)
#                 train_losses.append(loss)
#                 end = time.time()
#                 if verbose and ((step % print_every) == 0 or step == steps - 1):
#                     _ys_test = ys_test[:, : int(length_size * length)]
#                     test_loss, _ = grad_loss(model, _ts, _ys_test)
#                     test_losses.append(test_loss)
#                     print(
#                         f"Step: {step}, Train Loss: {loss:.4e}, Test Loss: {test_loss:.4e}, Computation time: {end - start:.4e}"
#                     )

#         final_test_loss, _ = grad_loss(model, ts_test, ys_test)
#         metrics_train = calculate_all_metrics(
#             ts,
#             ys_train,
#             model,
#             scaler,
#             training_initialconcentrations,
#             "Train",
#             ntimesteps,
#             noise_level if noise else 0.0,
#         )
#         metrics_test = calculate_all_metrics(
#             ts,
#             ys_test,
#             model,
#             scaler,
#             testing_initialconcentrations,
#             "Test",
#             ntimesteps,
#             noise_level if noise else 0.0,
#         )
#         metrics_list = [
#             dict(
#                 **m,
#                 **{
#                     "Training_Experiments": len(training_initialconcentrations),
#                     "Training_Timepoints": ntimesteps,
#                     "Final_Train_Loss": loss,
#                     "Final_Test_Loss": final_test_loss,
#                 },
#             )
#             for m in (metrics_train + metrics_test)
#         ]
#         train_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Train")
#         test_rmse = sum(m["RMSE"] for m in metrics_list if m["Experiment_Tag"] == "Test")

#         rmse_str = f"RMSE - Train: {train_rmse:.4f} Test: {test_rmse:.4f}"

#         extratitlestring = f"{extratitlestring} (Fresh)\n{rmse_str}"

#     except eqx.EquinoxRuntimeError:
#         error_metrics_list = []
#         for initial_conc in training_initialconcentrations:
#             metrics_dict = {
#                 "Training_Experiments": len(training_initialconcentrations),
#                 "Training_Timepoints": ntimesteps,
#                 "Experiment_Tag": "Train",
#                 "Initial_Concentration": initial_conc,
#                 "Measurement_Noise": noise_level if noise else 0.0,
#                 "MAE_Total": 999.0,
#                 "MSE_Total": 999.0,
#                 "RMSE_Total": 999.0,
#                 "MAPE_Total": 999.0,
#                 "MAE_Concentration": 999.0,
#                 "MSE_Concentration": 999.0,
#                 "RMSE_Concentration": 999.0,
#                 "MAPE_Concentration": 999.0,
#                 "MAE_D43": 999.0,
#                 "MSE_D43": 999.0,
#                 "RMSE_D43": 999.0,
#                 "MAPE_D43": 999.0,
#                 "Final_Train_Loss": 999.0,
#                 "Final_Test_Loss": 999.0,
#             }
#             error_metrics_list.append(metrics_dict)
#         for initial_conc in testing_initialconcentrations:
#             metrics_dict = {
#                 "Training_Experiments": len(training_initialconcentrations),
#                 "Training_Timepoints": ntimesteps,
#                 "Experiment_Tag": "Test",
#                 "Initial_Concentration": initial_conc,
#                 "Measurement_Noise": noise_level if noise else 0.0,
#                 "MAE_Total": 999.0,
#                 "MSE_Total": 999.0,
#                 "RMSE_Total": 999.0,
#                 "MAPE_Total": 999.0,
#                 "MAE_Concentration": 999.0,
#                 "MSE_Concentration": 999.0,
#                 "RMSE_Concentration": 999.0,
#                 "MAPE_Concentration": 999.0,
#                 "MAE_D43": 999.0,
#                 "MSE_D43": 999.0,
#                 "RMSE_D43": 999.0,
#                 "MAPE_D43": 999.0,
#                 "Final_Train_Loss": 999.0,
#                 "Final_Test_Loss": 999.0,
#             }
#             error_metrics_list.append(metrics_dict)
#         return (None, None, None, None, None, error_metrics_list)

#     if plot and splitplot:
#         splitplot_model_vs_data(
#             ts,
#             ys_train,
#             ys_test,
#             model,
#             scaler,
#             length_strategy,
#             extratitlestring,
#             saveplot,
#         )

#     if lossplot:
#         plot_loss_curves(
#             train_losses,
#             test_losses,
#             title=f"Loss Curves - {extratitlestring}",
#             saveplot=saveplot,
#             filename=f"plots/loss_curve_{extratitlestring}.png",
#         )

#     return ts, ys_train, ys_test, model, scaler, metrics_list


# (
#     ts_tl_last,
#     ys_train_tl_last,
#     ys_test_tl_last,
#     model_tl_last,
#     scaler_tl_last,
#     metrics_tl_last,
# ) = train_NODE_TL(
#     [19.0],  # Training data
#     [14.0, 18.0],  # Testing data
#     model,  # Pre-trained model
#     scaler,  # Original scaler
#     idx_frozen="last_two",  # Freeze the last layer
#     lr_strategy=(1e-3, 1e-3),
#     steps_strategy=(100, 100),
#     length_strategy=(0.5, 1),
#     ntimesteps=14,
#     seed=470,  # New seed
#     plot=True,
#     extratitlestring="TL_Last",
#     save_idxs=[0, 3],
#     noise=True,
#     noise_level=0.05,
#     nucl_params=[28.2, 0.44],
#     growth_params=[0.186, 3.02],
#     verbose=True,
# )

# ts, ys_train, ys_test, model, scaler, metrics = train_NODE_fresh_custom_layers(
#     [13.0, 15.5, 16.5, 19.0],
#     [12, 14.3, 17.7, 20],
#     lr_strategy=(1e-3, 1e-3),
#     steps_strategy=(700, 700),
#     length_strategy=(0.33, 1),
#     hidden_layers=[32, 128, 64,],  # Example: 4 hidden layers of 100 neurons each
#     activation=jnn.swish,
#     ntimesteps=14,
#     solver=diffrax.Tsit5(),
#     seed=46,
#     plot=True,
#     splitplot=True,
#     lossplot=True,
#     save_idxs=[0, 3],
#     noise=True,
#     noise_level=0.1,
#     print_every=100,

#     ## Homogeneous crystallization
#     # nucl_params=[27.039, 0.49148],
#     # growth_params=[0.24133, 3.3502],

#     ## Hydroxyl template
#     # nucl_params=[28.2, 0.44],
#     # growth_params=[0.186, 3.02],

#     ## Carboxyl template
#     nucl_params=[32.94, 0.53348],
#     growth_params=[0.1, 3.3125],
# )

# %%
