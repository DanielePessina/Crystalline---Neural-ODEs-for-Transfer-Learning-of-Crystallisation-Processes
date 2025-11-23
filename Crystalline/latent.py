"""Latent Neural ODE model for crystallisation trajectories."""

import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import Array

from Crystalline.data_functions import dataloader, simulateCrystallisation
from Crystalline.metrics.calculations import calculate_all_metrics_latent
from Crystalline.metrics.error import make_failure_metrics
from Crystalline.plotting import (
    plot_loss_curves,
    splitplot_model_vs_data_latent,
)


class LatentODEFunc(eqx.Module):
    """
    Defines the vector field for the ODE in the latent space.
    """

    mlp: eqx.nn.MLP
    scale: Array

    def __init__(self, latent_size: int, width_size: int, depth: int, key: jr.PRNGKey):
        mlp_key, scale_key = jr.split(key)
        self.mlp = eqx.nn.MLP(
            in_size=latent_size,
            out_size=latent_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.scale = jnp.ones(())  # Scaling factor for the vector field

    def __call__(self, t: float, y: Array, args: None | None = None) -> Array:
        return self.scale * self.mlp(y)


class LatentNeuralODE(eqx.Module):
    """
    Implements the Latent Neural ODE model.
    """

    func: LatentODEFunc
    rnn_cell: eqx.nn.GRUCell
    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear
    hidden_size: int
    latent_size: int

    def __init__(
        self, data_size: int, hidden_size: int, latent_size: int, width_size: int, depth: int, key: jr.PRNGKey
    ):
        func_key, rnn_key, hl_key, lh_key, hd_key = jr.split(key, 5)
        self.func = LatentODEFunc(latent_size, width_size, depth, func_key)
        self.rnn_cell = eqx.nn.GRUCell(input_size=data_size + 1, hidden_size=hidden_size, key=rnn_key)
        self.hidden_to_latent = eqx.nn.Linear(in_features=hidden_size, out_features=2 * latent_size, key=hl_key)
        self.latent_to_hidden = eqx.nn.MLP(
            in_size=latent_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.softplus,
            key=lh_key,
        )
        self.hidden_to_data = eqx.nn.Linear(in_features=hidden_size, out_features=data_size, key=hd_key)
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    def encode(self, ts: Array, ys: Array) -> tuple[Array, Array, Array]:
        """
        Encodes the observed data into a latent representation.
        """
        # Concatenate time and observations so the RNN receives explicit time information
        time_steps, _ = ys.shape  # ys has shape (T, D)
        ts_broadcasted = ts[:, None]  # (T, 1)
        data = jnp.concatenate([ts_broadcasted, ys], axis=-1)  # (T, D + 1)

        # Run a single GRU backwards through time
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in data[::-1]:  # iterate from last to first time‑step
            hidden = self.rnn_cell(data_i, hidden)

        context = self.hidden_to_latent(hidden)  # (2 * latent_size,)
        mean, log_std = jnp.split(context, 2)  # each has shape (latent_size,)
        std = jnp.exp(log_std)
        return mean, std, hidden

    def reparameterize(self, mean: Array, std: Array, key: jr.PRNGKey) -> Array:
        """
        Samples from the latent distribution using the reparameterization trick.
        """
        eps = jr.normal(key, shape=mean.shape)
        return mean + eps * std

    def decode(self, ts: Array, latent: Array) -> Array:
        """
        Decodes the latent representation into the data space by solving the ODE.
        """
        y0 = self.latent_to_hidden(latent)
        term = diffrax.ODETerm(self.func)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=0.1, y0=y0, saveat=saveat)
        return jax.vmap(self.hidden_to_data)(sol.ys)

    def __call__(self, ts: Array, ys: Array, key: jr.PRNGKey) -> tuple[Array, float]:
        """
        Performs a forward pass through the model and computes the loss.
        """
        mean, std, _ = self.encode(ts, ys)
        latent = self.reparameterize(mean, std, key)
        pred_ys = self.decode(ts, latent)
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        kl_divergence = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        loss = reconstruction_loss + kl_divergence
        return pred_ys, loss


def train_LatentNODE_fresh(
    training_initialconcentrations,
    testing_initialconcentrations,
    *,
    lr_strategy=(4e-3, 10e-3),
    steps_strategy=(600, 600),
    length_strategy=(0.33, 1),
    hidden_size=16,  # hidden size
    width_size=16,  # hidden size
    latent_size=16,
    depth=2,
    seed=467,
    activation=jnn.swish,
    solver=diffrax.Tsit5(),
    verbose=True,
    print_every=301,
    ntimesteps=85,
    nucl_params=[39.81, 0.675],
    growth_params=[0.345, 3.344],
    splitplot=False,
    saveplot=False,
    lossplot=False,
    extratitlestring="",
    save_idxs=[0, 3],
    noise=False,
    noise_level=0.1,
    batch_size=1,
    scale_strategy=None,
    masked=False,
):
    """
    Trains a Neural ODE model from scratch on simulated crystallization data.

    Supports two-stage training with varying learning rates, trajectory lengths, and training steps.

    Returns:
        ts: Time points.
        ys_train: Scaled training trajectories.
        ys_test: Scaled test trajectories.
        model: Trained NeuralODE instance.
        scaler: MinMaxScaler used for preprocessing.
        metrics_list: List of dictionaries containing evaluation metrics.
    """
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, metrics_key = jr.split(key, 4)

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
            masked=masked,
            length_strategy=length_strategy,
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
            masked=masked,
            length_strategy=length_strategy,
        )

    elif scale_strategy == "none":
        ts, ys_train, scaler = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            scale_strategy,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
            masked=masked,
            length_strategy=length_strategy,
        )

        ts_test, ys_test, _ = simulateCrystallisation(  # Use same scaler from training
            testing_initialconcentrations,
            ntimesteps,
            scale_strategy,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            save_idxs=save_idxs,
            noise=noise,
            noise_level=noise_level,
            masked=masked,
            length_strategy=length_strategy,
        )
    # Get dimensions from the data
    _, length_size, data_size = ys_train.shape

    model = LatentNeuralODE(
        data_size,
        hidden_size,
        latent_size,
        width_size,
        depth,
        model_key,
    )

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi, key):
        batch_size = yi.shape[0]
        keys = jr.split(key, batch_size)
        y_pred, _ = jax.vmap(model, in_axes=(None, 0, 0))(ti, yi, keys)  # (B, T, D)
        return jnp.mean((yi - y_pred) ** 2)

    # Masked version: mask out the second output feature (index 1) after the first time step (t>0)
    @eqx.filter_value_and_grad
    def grad_loss_masked(model, ti, yi, key):
        batch_size = yi.shape[0]
        keys = jr.split(key, batch_size)
        y_pred, _ = jax.vmap(model, in_axes=(None, 0, 0))(ti, yi, keys)
        # yi and y_pred are (B, T, D)
        B, T, D = yi.shape
        # Create mask: second feature visible only at t=0, hidden afterwards
        jnp.array([1.0, 1.0])  # shape (D,)
        second_feature_mask = jnp.concatenate([jnp.ones((1, 1)), jnp.zeros((T - 1, 1))], axis=0)  # shape (T, 1)
        mask = jnp.concatenate([jnp.ones((T, 1)), second_feature_mask], axis=1)  # shape (T, D)
        mask = jnp.broadcast_to(mask, (B, T, D))  # shape (B, T, D)
        yi_masked = yi * mask
        y_pred_masked = y_pred * mask
        return jnp.mean((yi_masked - y_pred_masked) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state, key, masked: bool):
        loss_fn = grad_loss_masked if masked else grad_loss
        loss, grads = loss_fn(model, ti, yi, key)
        updates, opt_state = optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    try:
        train_losses = []
        test_losses = []

        for i, (lr, steps, length) in enumerate(zip(lr_strategy, steps_strategy, length_strategy, strict=True)):
            optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adabelief(lr))
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            _ts = ts[: int(length_size * length)]
            _ys_train = ys_train[:, : int(length_size * length)]
            train_loader = dataloader((_ys_train,), batch_size, key=loader_key)
            for step in range(steps):
                (yi,) = next(train_loader)
                step_key, loader_key = jr.split(loader_key)
                start = time.time()
                loss, model, opt_state = make_step(_ts, yi, model, opt_state, step_key, masked=(masked and i == 0))
                train_losses.append(loss)
                end = time.time()
                if verbose and ((step % print_every) == 0 or step == steps - 1):
                    _ys_test = ys_test[:, : int(length_size * length)]
                    test_key, loader_key = jr.split(loader_key)
                    test_loss, _ = grad_loss(model, _ts, _ys_test, test_key)
                    test_losses.append(test_loss)
                    print(
                        f"Step: {step}, Train Loss: {loss:.4e}, Test Loss: {test_loss:.4e}, Computation time: {end - start:.4e}"
                    )

        final_test_key, _ = jr.split(loader_key)
        final_test_loss, _ = grad_loss(model, ts_test, ys_test, final_test_key)
        # Use new metrics function
        metrics_train = calculate_all_metrics_latent(
            ts,
            ys_train,
            model,
            metrics_key,
            scaler,
            training_initialconcentrations,
            "Train",
            ntimesteps,
            noise_level if noise else 0.0,
        )
        metrics_test = calculate_all_metrics_latent(
            ts,
            ys_test,
            model,
            metrics_key,
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
        splitplot_model_vs_data_latent(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            extratitlestring,
            saveplot,
        )

    if lossplot:
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"Loss Curves - {extratitlestring}",
            saveplot=saveplot,
            filename=f"plots/loss_curve_{extratitlestring}.png",
        )

    return ts, ys_train, ys_test, model, scaler, metrics_list


# Transfer learning for LatentNODE
import jax.tree_util as jtu


def train_LatentNODE_TL(
    training_initialconcentrations,
    testing_initialconcentrations,
    model: LatentNeuralODE,
    scaler,
    idx_frozen: int | tuple[int, int] | str,
    freeze_mode: str = "both",
    *,
    lr_strategy=(4e-3, 10e-3),
    steps_strategy=(600, 600),
    length_strategy=(0.33, 1),
    ntimesteps=85,
    nucl_params=[39.81, 0.675],
    growth_params=[0.345, 3.344],
    noise=False,
    noise_level=0.1,
    batch_size=1,
    scale_strategy=None,
    print_every=301,
    verbose=True,
    seed=567,
    saveplot=False,
    lossplot=False,
    splitplot=False,
    extratitlestring="",
    masked=False,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    if batch_size == "all":
        batch_size = len(training_initialconcentrations)

    if scale_strategy is None:
        ts, ys_train, scaler = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            noise=noise,
            noise_level=noise_level,
            masked=masked,
            length_strategy=length_strategy,
        )
        ts_test, ys_test, _ = simulateCrystallisation(
            testing_initialconcentrations,
            ntimesteps,
            scaler,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            noise=noise,
            noise_level=noise_level,
            masked=masked,
            length_strategy=length_strategy,
        )
    elif scale_strategy == "none":
        ts, ys_train, scaler = simulateCrystallisation(
            training_initialconcentrations,
            ntimesteps,
            scale_strategy,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            noise=noise,
            noise_level=noise_level,
            masked=masked,
            length_strategy=length_strategy,
        )
        ts_test, ys_test, _ = simulateCrystallisation(
            testing_initialconcentrations,
            ntimesteps,
            scale_strategy,
            key=data_key,
            nucl_params=nucl_params,
            growth_params=growth_params,
            noise=noise,
            noise_level=noise_level,
            masked=masked,
            length_strategy=length_strategy,
        )

    num_layers = len(model.latent_to_hidden.layers)
    if idx_frozen == "none":
        frozen_indices = set()
    elif idx_frozen == "last":
        frozen_indices = {num_layers - 1}
    elif idx_frozen == "last_two":
        frozen_indices = {num_layers - 1, num_layers - 2}
    elif idx_frozen == "all":
        frozen_indices = set(range(num_layers))
    elif isinstance(idx_frozen, int):
        frozen_indices = {idx_frozen}
    elif isinstance(idx_frozen, tuple):
        frozen_indices = set(range(*idx_frozen))
    else:
        raise TypeError("idx_frozen must be an integer, tuple, 'last', or 'none'")

    if freeze_mode not in ["weights", "biases", "both"]:
        raise ValueError("freeze_mode must be 'weights', 'biases', or 'both'")

    filter_spec = jtu.tree_map(lambda _: True, model)
    for idx in frozen_indices:
        if 0 <= idx < num_layers:
            if freeze_mode in ["weights", "both"]:
                filter_spec = eqx.tree_at(lambda m: m.latent_to_hidden.layers[idx].weight, filter_spec, replace=False)
            if freeze_mode in ["biases", "both"]:
                filter_spec = eqx.tree_at(lambda m: m.latent_to_hidden.layers[idx].bias, filter_spec, replace=False)

    def loss_fn(model, ti, yi, key):
        batch_size = yi.shape[0]
        keys = jr.split(key, batch_size)
        y_pred, _ = jax.vmap(model, in_axes=(None, 0, 0))(ti, yi, keys)
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state, key):
        diff_model, static_model = eqx.partition(model, filter_spec)

        @eqx.filter_value_and_grad
        def inner_loss(diff_model):
            full_model = eqx.combine(diff_model, static_model)
            return loss_fn(full_model, ti, yi, key)

        loss_value, grads = inner_loss(diff_model)
        updates, opt_state = optim.update(grads, opt_state, params=diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
        return loss_value, model, opt_state

    train_losses = []
    test_losses = []

    _, length_size, _ = ys_train.shape

    for i, (lr, steps, length) in enumerate(zip(lr_strategy, steps_strategy, length_strategy, strict=True)):
        optim = optax.chain(optax.clip_by_global_norm(1e-2), optax.adabelief(lr))
        diff_model, _ = eqx.partition(model, filter_spec)
        opt_state = optim.init(eqx.filter(diff_model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys_train = ys_train[:, : int(length_size * length)]
        train_loader = dataloader((_ys_train,), batch_size, key=loader_key)

        for step in range(steps):
            (yi,) = next(train_loader)
            step_key, loader_key = jr.split(loader_key)
            loss, model, opt_state = make_step(_ts, yi, model, opt_state, step_key)
            train_losses.append(loss)
            if verbose and (step % print_every == 0 or step == steps - 1):
                _ys_test = ys_test[:, : int(length_size * length)]
                test_key, loader_key = jr.split(loader_key)
                test_loss = loss_fn(model, _ts, _ys_test, test_key)
                test_losses.append(test_loss)
                print(f"Step: {step}, Train Loss: {loss:.4e}, Test Loss: {test_loss:.4e}")

    if splitplot:
        splitplot_model_vs_data_latent(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            extratitlestring,
            saveplot,
        )

    if lossplot:
        plot_loss_curves(
            train_losses,
            test_losses,
            title=f"Loss Curves - {extratitlestring}",
            saveplot=saveplot,
            filename=f"plots/loss_curve_{extratitlestring}.png",
        )

    return ts, ys_train, ys_test, model, scaler, (train_losses, test_losses)


if __name__ == "__main__":
    ts, ys_train, ys_test, model, scaler, metrics = train_LatentNODE_fresh(
        # [13.0, 15.5, 16.5, 19.0],
        [15.5, 16.5, 19],
        [12, 14.3, 17.7, 20],
        lr_strategy=(1e-3, 1e-3),
        steps_strategy=(2200, 3600),
        length_strategy=(0.3, 1),
        ntimesteps=14 * 3,
        # solver=diffrax.Kvaerno3(),
        seed=4700,
        splitplot=True,
        lossplot=True,
        save_idxs=[0, 3],
        noise=False,
        noise_level=0.1,
        print_every=100,
        batch_size="all",
        # scale_strategy = "none",
        ## Homogeneous crystallization
        nucl_params=[27.039, 0.49148],
        growth_params=[0.24133, 3.3502],
        ## Hydroxyl template
        # nucl_params=[28.2, 0.44],
        # growth_params=[0.186, 3.02],
        ## Carboxyl template
        # nucl_params=[32.94, 0.53348],
        # growth_params=[0.1, 3.3125],
        # masked=True,
    )

    ts_tl, ys_train_tl, ys_test_tl, model_tl, scaler_tl, metrics_tl = train_LatentNODE_TL(
        [16.5, 17],
        [12, 14.3, 17.7, 20],
        model,
        scaler,
        idx_frozen="last_two",
        freeze_mode="both",
        lr_strategy=(1e-3, 1e-3),
        steps_strategy=(2200, 3600),
        length_strategy=(0.3, 1),
        ntimesteps=14 * 3,
        # solver=diffrax.Kvaerno3(),
        seed=4700,
        splitplot=True,
        lossplot=True,
        noise=False,
        noise_level=0.1,
        print_every=100,
        batch_size="all",
        ## Homogeneous crystallization
        # nucl_params=[27.039, 0.49148],
        # growth_params=[0.24133, 3.3502],
        ## Hydroxyl template
        nucl_params=[28.2, 0.44],
        growth_params=[0.186, 3.02],
        ## Carboxyl template
        # nucl_params=[32.94, 0.53348],
        # growth_params=[0.1, 3.3125],
        # masked=True,
    )


# %%
