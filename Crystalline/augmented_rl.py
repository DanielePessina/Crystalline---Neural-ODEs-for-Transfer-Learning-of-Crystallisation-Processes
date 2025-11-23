"""RL-driven transfer learning utilities for Augmented NODEs (JAX).

This module provides a jittable Gymnax-style environment that executes
short fine-tuning segments under different layer-freezing and penalty
strategies, plus lightweight PPO scaffolding built with Equinox.

Design goals
- JIT safety: fixed shapes; no Python control flow inside JIT regions.
- Clean separation: params (arrays) in state; static model structure separate.
- Reproducibility: explicit PRNG threading and deterministic sampling.

Notes
- This is a first-pass scaffold to “start implementing the plan”. It focuses
  on mask construction, a micro-training segment, and a basic env skeleton.
- Integration with the existing data pipeline mirrors augmented.py conventions
  but may require iteration on shapes once wired to your exact datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from jax import Array

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# Mask specification utilities
# -----------------------------------------------------------------------------

FreezeStrategy = Literal[
    "none",
    "last",
    "last_two",
    "first",
    "first_two",
    "first_three",
    "all",
]


def _layer_bool_tree(model: eqx.Module, default: bool) -> eqx.Module:
    """Return a boolean tree with same structure as model arrays.

    Non-array leaves remain as-is; array leaves become a scalar boolean.
    """

    def to_mask_leaf(x):
        # Only arrays participate in masks; represent mask as scalar bool array
        return jnp.asarray(bool(default), dtype=jnp.bool_) if eqx.is_array(x) else x

    return jtu.tree_map(to_mask_leaf, model)


def _set_layer_slice(mask_tree: eqx.Module, layer_indices: Sequence[int], value: bool) -> eqx.Module:
    """Set mask booleans for specified `eqx.nn.MLP.layers` indices.

    The mask is expected to mirror the structure of the model. This helper
    targets `.layers[i].weight` and `.layers[i].bias` for each index in
    `layer_indices`.
    """

    def where_weights(tree):
        return tuple(tree.func.mlp.layers[i].weight for i in layer_indices)

    def where_biases(tree):
        return tuple(tree.func.mlp.layers[i].bias for i in layer_indices)

    # Update weights
    mask_tree = eqx.tree_at(where_weights, mask_tree, replace=tuple(bool(value) for _ in layer_indices))
    # Update biases
    return eqx.tree_at(where_biases, mask_tree, replace=tuple(bool(value) for _ in layer_indices))


def build_mask_specs(model: eqx.Module) -> tuple[list[eqx.Module], list[eqx.Module]]:
    """Build freeze and penalty mask specs for 7 common strategies.

    Args:
        model: Equinox model (expected to contain `func.mlp.layers`).

    Returns:
        - freeze_specs: list of boolean PyTrees (True=trainable, False=frozen).
        - penalty_specs: list of boolean PyTrees (True=include in L2 penalty).

    Strategies are ordered as:
        ["none", "last", "last_two", "first", "first_two", "first_three", "all"].
    """

    num_layers = len(model.func.mlp.layers)
    idx_last = [num_layers - 1]
    idx_last_two = [num_layers - 2, num_layers - 1] if num_layers >= 2 else idx_last
    idx_first = [0]
    idx_first_two = [0, 1] if num_layers >= 2 else idx_first
    idx_first_three = [0, 1, 2] if num_layers >= 3 else list(range(num_layers))
    list(range(num_layers))

    # Freeze masks: True=trainable; False=frozen
    freeze_specs: list[eqx.Module] = []
    # Penalty masks: True=include in penalty; False=ignore
    penalty_specs: list[eqx.Module] = []

    # Base masks
    all_trainable = _layer_bool_tree(model, True)
    all_frozen = _layer_bool_tree(model, False)
    none_penalty = _layer_bool_tree(model, False)
    all_penalty = _layer_bool_tree(model, True)

    # 0) none → no freezing; no penalty by default (user chooses index for penalty)
    freeze_none = all_trainable
    penalty_none = none_penalty

    # 1) last
    freeze_last = _set_layer_slice(_layer_bool_tree(model, True), idx_last, False)
    penalty_last = _set_layer_slice(_layer_bool_tree(model, False), idx_last, True)

    # 2) last_two
    freeze_last_two = _set_layer_slice(_layer_bool_tree(model, True), idx_last_two, False)
    penalty_last_two = _set_layer_slice(_layer_bool_tree(model, False), idx_last_two, True)

    # 3) first
    freeze_first = _set_layer_slice(_layer_bool_tree(model, True), idx_first, False)
    penalty_first = _set_layer_slice(_layer_bool_tree(model, False), idx_first, True)

    # 4) first_two
    freeze_first_two = _set_layer_slice(_layer_bool_tree(model, True), idx_first_two, False)
    penalty_first_two = _set_layer_slice(_layer_bool_tree(model, False), idx_first_two, True)

    # 5) first_three
    freeze_first_three = _set_layer_slice(_layer_bool_tree(model, True), idx_first_three, False)
    penalty_first_three = _set_layer_slice(_layer_bool_tree(model, False), idx_first_three, True)

    # 6) all
    freeze_all = all_frozen
    penalty_all = all_penalty

    freeze_specs.extend(
        [
            freeze_none,
            freeze_last,
            freeze_last_two,
            freeze_first,
            freeze_first_two,
            freeze_first_three,
            freeze_all,
        ]
    )
    penalty_specs.extend(
        [
            penalty_none,
            penalty_last,
            penalty_last_two,
            penalty_first,
            penalty_first_two,
            penalty_first_three,
            penalty_all,
        ]
    )

    # Reduce masks to arrays-only trees to match params/updates structure
    def arrays_only(tree):
        arrays, _ = eqx.partition(tree, eqx.is_array)
        return arrays

    freeze_specs = [arrays_only(t) for t in freeze_specs]
    penalty_specs = [arrays_only(t) for t in penalty_specs]

    return freeze_specs, penalty_specs


def stack_mask_specs(mask_specs: list[eqx.Module]) -> eqx.Module:
    """Stack a list of boolean PyTrees along a new leading axis.

    This allows JIT-time selection of a mask by integer index without Python
    control flow. For each array leaf, returns an array of shape (K, *leaf.shape)
    where K=len(mask_specs).
    """

    def stack_leaves(*leaves):
        # All leaves are booleans or non-arrays; skip non-arrays
        if not eqx.is_array(leaves[0]):
            return leaves[0]
        arrs = [jnp.asarray(l, dtype=jnp.bool_) for l in leaves]
        return jnp.stack(arrs, axis=0)

    return jtu.tree_map(stack_leaves, *mask_specs)


def l2_penalty_masked(params: eqx.Module, ref_params: eqx.Module, mask: eqx.Module) -> Array:
    """Compute L2(θ − θ₀) masked over boolean PyTree `mask`.

    Args:
        params: Current parameter PyTree (arrays only or mixed).
        ref_params: Reference parameter PyTree with same structure.
        mask: Boolean PyTree; True → include leaf in penalty, False → ignore.

    Returns:
        Scalar penalty value.
    """

    def leaf_pen(p, r, m):
        if not eqx.is_array(p):
            return 0.0
        m = jnp.asarray(m, dtype=jnp.bool_)
        diff = jnp.where(m, p - r, 0.0)
        return jnp.sum(diff * diff)

    leaf_vals = jtu.tree_map(leaf_pen, params, ref_params, mask)
    return jnp.sum(jnp.asarray(jtu.tree_leaves(leaf_vals)))


# -----------------------------------------------------------------------------
# Micro-training segment
# -----------------------------------------------------------------------------


def make_micro_segment(
    static_model: eqx.Module,
    ts: Array,
    ys_train: Array,
    ys_val: Array,
    *,
    steps_per_decision: int,
    batch_size: int,
    base_optimizer: optax.GradientTransformation,
    penalty_lambda: float,
    freeze_masks_stacked: eqx.Module,
    penalty_masks_stacked: eqx.Module,
) -> Callable[
    [eqx.Module, optax.OptState, Array, int, int, eqx.Module], tuple[eqx.Module, optax.OptState, Array, Array]
]:
    """Create a jittable micro training segment.

    The returned function performs `steps_per_decision` SGD steps with fixed
    shape, masking parameter updates according to `freeze_idx`, and adding a
    deviation penalty according to `penalty_idx`.

    Args:
        static_model: Non-array part of model (from `eqx.partition`).
        ts: Time grid [T].
        ys_train: Training trajectories [N, T, D].
        ys_val: Validation trajectories [M, T, D].
        steps_per_decision: Number of SGD steps per environment step.
        batch_size: Batch size for SGD.
        base_optimizer: Optax optimizer (static, used inside JIT).
        penalty_lambda: Weight for deviation penalty.
        freeze_masks_stacked: Boolean PyTree with leading dim K_f (strategies).
        penalty_masks_stacked: Boolean PyTree with leading dim K_p (strategies).

    Returns:
        A function `(params, opt_state, key, freeze_idx, penalty_idx, ref_params)
        -> (new_params, new_opt_state, last_train_loss, val_mse_after)`.
    """

    @eqx.filter_value_and_grad
    def train_loss_fn(params, batch_y, penalty_mask, ref_params):
        model = eqx.combine(params, static_model)
        # y0 for each trajectory is the first timepoint
        y0_batch = batch_y[:, 0]
        y_pred = jax.vmap(model, in_axes=(None, 0))(ts, y0_batch)
        mse = jnp.mean((batch_y - y_pred) ** 2)
        penalty = l2_penalty_masked(params, ref_params, penalty_mask)
        return mse + penalty_lambda * penalty

    def val_mse(params):
        model = eqx.combine(params, static_model)
        y0_val = ys_val[:, 0]
        y_pred = jax.vmap(model, in_axes=(None, 0))(ts, y0_val)
        return jnp.mean((ys_val - y_pred) ** 2)

    @eqx.filter_jit
    def segment(params, opt_state, key, freeze_idx: int, penalty_idx: int, ref_params):
        # Select masks by integer indices from stacked trees
        freeze_mask = jtu.tree_map(lambda x: x[freeze_idx] if eqx.is_array(x) else x, freeze_masks_stacked)
        penalty_mask = jtu.tree_map(lambda x: x[penalty_idx] if eqx.is_array(x) else x, penalty_masks_stacked)

        def sgd_step(carry, _t):
            params, opt_state, key = carry
            key, key_idx = jr.split(key, 2)
            # Sample a batch via uniform indices with replacement
            n_train = ys_train.shape[0]
            idx = jr.randint(key_idx, (batch_size,), minval=0, maxval=n_train)
            batch_y = ys_train[idx]

            loss, grads = train_loss_fn(params, batch_y, penalty_mask, ref_params)
            updates, opt_state = base_optimizer.update(grads, opt_state, params=params)
            # Mask the updates to enforce freezing strictly
            masked_updates = jtu.tree_map(
                lambda upd, m: jnp.where(m, upd, jnp.zeros_like(upd)) if eqx.is_array(upd) else upd,
                updates,
                freeze_mask,
            )
            params = eqx.apply_updates(params, masked_updates)
            return (params, opt_state, key), loss

        # Unroll SGD steps
        (params, opt_state, _), losses = jax.lax.scan(
            sgd_step, init=(params, opt_state, key), xs=jnp.arange(steps_per_decision)
        )
        return params, opt_state, losses[-1], val_mse(params)

    return segment


# -----------------------------------------------------------------------------
# Gymnax-style Environment
# -----------------------------------------------------------------------------


class EnvParams(eqx.Module):
    """Environment parameters (mostly static data and configuration).

    Attributes
    - static_model: Non-array part of Equinox model.
    - ts: Time grid [T].
    - ys_train: Training trajectories [N, T, D].
    - ys_val: Validation trajectories [M, T, D].
    - ref_params: Reference parameters (for penalty baseline).
    - base_optimizer: Optax optimizer for micro-segments.
    - steps_per_decision: SGD steps per env step (static int).
    - decisions_per_episode: Number of environment decisions per episode (static int).
    - batch_size: Batch size for micro training (static int).
    - penalty_lambda: Penalty weight (float).
    - freeze_masks_stacked: Boolean PyTree with leading K_f.
    - penalty_masks_stacked: Boolean PyTree with leading K_p.
    - n_freeze: Number of freeze strategies.
    - n_penalty: Number of penalty strategies.
    """

    static_model: eqx.Module = eqx.static_field()
    ts: Array
    ys_train: Array
    ys_val: Array
    ref_params: eqx.Module
    base_optimizer: optax.GradientTransformation = eqx.static_field()
    steps_per_decision: int = eqx.static_field()
    decisions_per_episode: int = eqx.static_field()
    batch_size: int = eqx.static_field()
    penalty_lambda: float = eqx.static_field()
    freeze_masks_stacked: eqx.Module = eqx.static_field()
    penalty_masks_stacked: eqx.Module = eqx.static_field()
    n_freeze: int = eqx.static_field()
    n_penalty: int = eqx.static_field()


class EnvState(eqx.Module):
    """Environment state (all array-like to be JAX-traceable).

    Attributes
    - params: Current trainable parameter PyTree (arrays-only part).
    - opt_state: Optimizer state PyTree.
    - key: PRNGKey for sampling.
    - t: Current decision step index.
    - last_val_mse: Validation MSE after last micro-segment.
    """

    params: eqx.Module
    opt_state: optax.OptState
    key: Array
    t: Array
    last_val_mse: Array


class RLTransferEnv(eqx.Module):
    """Custom Gymnax-style environment for RL-driven transfer learning.

    Methods follow the Gymnax signatures:
    - reset(key, params) -> obs, state
    - step(key, state, action, params) -> obs, state, reward, done, info
    """

    obs_dim: int = eqx.static_field()

    def __init__(self, obs_dim: int = 3):
        self.obs_dim = obs_dim

    def action_space(self, env_params: EnvParams):
        """Return a discrete action space consistent with freeze×penalty choices.

        We import lazily to avoid import-time dependency if Gymnax is absent.
        """

        try:
            from gymnax.environments import spaces  # type: ignore

            return spaces.Discrete(env_params.n_freeze * env_params.n_penalty)
        except Exception:
            # Lightweight fallback stub mimicking Gymnax's Discrete for sampling.
            class _Discrete:
                def __init__(self, n: int):
                    self.n = int(n)

                def sample(self, key: Array) -> Array:
                    return jr.randint(key, (), 0, self.n)

            return _Discrete(env_params.n_freeze * env_params.n_penalty)

    @eqx.filter_jit
    def reset(self, key: Array, env_params: EnvParams) -> tuple[Array, EnvState]:
        params = env_params.ref_params
        opt_state = env_params.base_optimizer.init(params)

        # Compute initial validation MSE without updating parameters
        model = eqx.combine(params, env_params.static_model)
        y0_val = env_params.ys_val[:, 0]
        y_pred = jax.vmap(model, in_axes=(None, 0))(env_params.ts, y0_val)
        val_mse = jnp.mean((env_params.ys_val - y_pred) ** 2)
        state = EnvState(
            params=params,
            opt_state=opt_state,
            key=key,
            t=jnp.array(0, dtype=jnp.int32),
            last_val_mse=val_mse,
        )
        obs = jnp.array([0.0, val_mse, 0.0])[: self.obs_dim]
        return obs, state

    @eqx.filter_jit
    def step(
        self,
        key: Array,
        state: EnvState,
        action: Array,
        env_params: EnvParams,
    ) -> tuple[Array, EnvState, Array, Array, dict]:
        # Decode action into (freeze_idx, penalty_idx)
        n_pen = env_params.n_penalty
        freeze_idx = (action // n_pen).astype(jnp.int32)
        penalty_idx = (action % n_pen).astype(jnp.int32)

        segment = make_micro_segment(
            env_params.static_model,
            env_params.ts,
            env_params.ys_train,
            env_params.ys_val,
            steps_per_decision=env_params.steps_per_decision,
            batch_size=env_params.batch_size,
            base_optimizer=env_params.base_optimizer,
            penalty_lambda=env_params.penalty_lambda,
            freeze_masks_stacked=env_params.freeze_masks_stacked,
            penalty_masks_stacked=env_params.penalty_masks_stacked,
        )

        new_params, new_opt_state, train_loss, val_mse_after = segment(
            state.params,
            state.opt_state,
            key,
            freeze_idx,
            penalty_idx,
            env_params.ref_params,
        )

        improvement = state.last_val_mse - val_mse_after
        reward = improvement  # compute-cost term can be added later if needed

        new_t = state.t + 1
        done = new_t >= jnp.array(env_params.decisions_per_episode, dtype=jnp.int32)
        obs = jnp.array([new_t.astype(jnp.float32), val_mse_after, train_loss])[: self.obs_dim]

        new_state = EnvState(
            params=new_params,
            opt_state=new_opt_state,
            key=key,
            t=new_t,
            last_val_mse=val_mse_after,
        )
        info = {}
        return obs, new_state, reward, done, info


# -----------------------------------------------------------------------------
# PPO scaffolding (lightweight)
# -----------------------------------------------------------------------------


class PPOAgent(eqx.Module):
    """Lightweight PPO agent with Equinox MLP policy and value nets.

    This is a scaffold; it exposes shapes and functions but omits a full
    training loop until the environment is fully validated.
    """

    policy: eqx.nn.MLP
    value: eqx.nn.MLP

    def __init__(self, obs_dim: int, n_actions: int, width: int = 64, depth: int = 2, *, key):
        k1, k2 = jr.split(key)
        self.policy = eqx.nn.MLP(obs_dim, n_actions, width, depth, key=k1)
        self.value = eqx.nn.MLP(obs_dim, 1, width, depth, key=k2)

    def logits(self, obs: Array) -> Array:
        return self.policy(obs)

    def act(self, key: Array, obs: Array) -> tuple[Array, Array]:
        logits = self.logits(obs)
        a = jax.random.categorical(key, logits)
        logp = jax.nn.log_softmax(logits)[a]
        return a, logp

    def v(self, obs: Array) -> Array:
        return jnp.squeeze(self.value(obs), axis=-1)


# -----------------------------------------------------------------------------
# Public entry point (skeleton)
# -----------------------------------------------------------------------------


def train_AugNODE_TL_RL(
    *,
    model: eqx.Module,
    ts: Array,
    ys_train: Array,
    ys_val: Array,
    steps_per_decision: int = 200,
    decisions_per_episode: int = 3,
    batch_size: int = 8,
    penalty_lambda: float = 1.0,
    micro_lr: float = 3e-3,
    seed: int = 42,
):
    """Run a minimal RL-driven transfer learning session (skeleton).

    Args:
        model: Pre-trained Augmented NODE model (Equinox module).
        ts: Time grid [T].
        ys_train: Training trajectories [N, T, D].
        ys_val: Validation trajectories [M, T, D].
        steps_per_decision: SGD steps inside each env step.
        decisions_per_episode: Number of decisions per episode (horizon).
        batch_size: Training batch size.
        penalty_lambda: L2(θ−θ₀) penalty weight.
        micro_lr: Learning rate for micro-segments.
        seed: PRNG seed.

    Returns:
        Tuple of `(env, env_params, init_obs, init_state)` for iteration/testing.
    """

    key = jr.PRNGKey(seed)
    params, static_model = eqx.partition(model, eqx.is_inexact_array)
    freeze_specs, penalty_specs = build_mask_specs(model)
    freeze_masks_stacked = stack_mask_specs(freeze_specs)
    penalty_masks_stacked = stack_mask_specs(penalty_specs)

    base_optimizer = optax.chain(optax.clip_by_global_norm(1e-2), optax.adam(micro_lr))

    env_params = EnvParams(
        static_model=static_model,
        ts=ts,
        ys_train=ys_train,
        ys_val=ys_val,
        ref_params=params,
        base_optimizer=base_optimizer,
        steps_per_decision=steps_per_decision,
        decisions_per_episode=decisions_per_episode,
        batch_size=batch_size,
        penalty_lambda=penalty_lambda,
        freeze_masks_stacked=freeze_masks_stacked,
        penalty_masks_stacked=penalty_masks_stacked,
        n_freeze=len(freeze_specs),
        n_penalty=len(penalty_specs),
    )

    env = RLTransferEnv(obs_dim=3)
    obs0, state0 = env.reset(key, env_params)
    return env, env_params, obs0, state0


def plot_rl_model_vs_data(
    ts: Array,
    ys_train: Array,
    ys_test: Array,
    model: eqx.Module,
    scaler: MinMaxScaler | None,
    *,
    length_strategy: tuple[float, float] = (0.33, 1.0),
    title: str = "AugNODE RL Predictions",
    saveplot: bool = False,
    filename_prefix: str = "AugNODE_RL_predictions.png",
) -> None:
    """Plot model predictions versus data for train/test concentrations.

    Produces 2-panel plots similar to the utilities used in `augmented.py`.
    If the data dimension is 1, a simplified 1D plotter is used.

    Args:
        ts: Time grid of shape [T]. Units: [min].
        ys_train: Training trajectories [N_train, T, D] in scaled space.
        ys_test: Test trajectories [N_test, T, D] in scaled space.
        model: Equinox model with signature `model(ts, y0)` → [T, D] in scaled space.
        scaler: Optional `MinMaxScaler` used to invert scale for plotting.
        length_strategy: Fractions of the time grid used during staged training.
        title: Title string for the figure.
        saveplot: If True, saves to `plots/{filename_prefix}`.
        filename_prefix: Output filename (with extension) under `plots/`.

    Returns:
        None. Displays the plot and optionally saves it.
    """
    from Crystalline.plotting import splitplot_model_vs_data, splitplot_model_vs_data_1d

    data_size = int(ys_train.shape[-1])
    if data_size == 1:
        splitplot_model_vs_data_1d(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            title,
            saveplot,
            filename_prefix=filename_prefix,
        )
    else:
        splitplot_model_vs_data(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            title,
            saveplot,
            filename_prefix=filename_prefix,
        )


if __name__ == "__main__":
    # Minimal demonstration that mirrors augmented.py style: train a base model
    # and then invoke the RL transfer environment once.
    from Crystalline.augmented import C_TEST, C_TRAIN_SOURCE, train_AugNODE_fresh

    # Fresh base model (keep very small for a quick demo)
    ts, ys_train, ys_test, model, scaler, metrics = train_AugNODE_fresh(
        C_TRAIN_SOURCE,
        C_TEST,
        lr_strategy=(2e-4, 4e-4),
        steps_strategy=(100, 100),
        length_strategy=(0.5, 1.0),
        width_size=16,
        depth=3,
        ntimesteps=5,
        seed=1234,
        splitplot=False,
        plotly_plots=False,
        save_idxs=[0, 3],
        noise=False,
        print_every=50,
        batch_size="all",
        augment_dim=1,
    )

    # Build RL env from the base model and data
    env, env_params, obs0, state0 = train_AugNODE_TL_RL(
        model=model,
        ts=ts,
        ys_train=ys_train,
        ys_val=ys_test,
        steps_per_decision=20,
        decisions_per_episode=2,
        batch_size=int(min(ys_train.shape[0], 4)),
        penalty_lambda=1.0,
        micro_lr=3e-3,
        seed=4321,
    )

    print("[RL Env] init obs:", obs0)
    try:
        space = env.action_space(env_params)
        n_actions = getattr(space, "n", None)
        print(f"[RL Env] action_space.n = {n_actions}")
        a0 = space.sample(state0.key)
    except Exception:
        # Fallback: choose action 0
        a0 = jnp.array(0, dtype=jnp.int32)

    obs1, state1, r1, done1, info1 = env.step(state0.key, state0, a0, env_params)
    print("[RL Env] step -> obs, reward, done:", obs1, float(r1), bool(done1))

    # Plot predictions of the updated model after one RL decision
    updated_model = eqx.combine(state1.params, env_params.static_model)
    plot_rl_model_vs_data(
        ts,
        ys_train,
        ys_test,
        updated_model,
        scaler,
        length_strategy=(0.5, 1.0),
        title="AugNODE RL: After 1 decision",
        saveplot=False,
        filename_prefix="AugNODE_RL_predictions.png",
    )
