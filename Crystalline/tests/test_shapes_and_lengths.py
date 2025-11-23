"""Tests for `Crystalline` package."""

# tests/test_shapes_and_lengths.py
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jr = jax.random

Crystalline_data = pytest.importorskip("Crystalline.data_functions")
simulateCrystallisation = Crystalline_data.simulateCrystallisation
dataloader = Crystalline_data.dataloader

aug = pytest.importorskip("Crystalline.augmented_domain_learnable")
SystemAugmentedNeuralODE = aug.SystemAugmentedNeuralODE
create_system_embedding = aug.create_system_embedding


pytestmark = pytest.mark.filterwarnings("ignore:shape requires ndarray or scalar arguments:DeprecationWarning")


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def key():
    return jr.PRNGKey(0)


@pytest.fixture(scope="module")
def nt():
    # from __main__ usage in your training script
    return 28


@pytest.fixture(scope="module")
def save_idxs():
    # from simulateCrystallisation defaults in your examples
    return [0, 3]


@pytest.fixture(scope="module")
def system_params():
    # copied from your __main__ block
    return {
        0: {"nucl_params": [39.81, 0.675], "growth_params": [0.345, 3.344]},
        1: {"nucl_params": [29.9, 0.49], "growth_params": [0.37, 3.3]},
        2: {"nucl_params": [22.7, 0.14], "growth_params": [0.37, 3.3]},
        3: {"nucl_params": [27.4, 0.32], "growth_params": [0.37, 3.3]},
    }


@pytest.fixture(scope="module")
def training_data():
    return [
        ([15.5, 16.5, 19.0], 0),
        ([16.5, 17.0], 1),
        ([16.5, 17.0], 2),
        ([16.5, 17.0], 3),
    ]


@pytest.fixture(scope="module")
def testing_data():
    return [
        ([14.0, 17.7, 20.0], 0),
        ([14.0, 17.7, 20.0], 1),
        ([14.0, 17.7, 20.0], 2),
        ([14.0, 17.7, 20.0], 3),
    ]


@pytest.fixture(scope="module")
def model_hparams():
    # from your __main__ settings
    return {
        "augment_dim": 1,
        "width": 64,
        "depth": 7,
        "embed_proj_size": 64,
        "include_time": True,
    }


# ---------------------------------------------------------------------
# simulateCrystallisation: shapes only
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "dataset_idx, scaler, noise, masked, length_strategy, sys_idx",
    [
        # fit scaler path
        (0, None, False, False, None, 0),
        # no_scale + noise + masked path (with provided length_strategy)
        (0, "no_scale", True, True, (0.33, 1), 1),
    ],
)
def test_simulateCrystallisation_shapes(
    training_data,
    testing_data,
    system_params,
    nt,
    save_idxs,
    key,
    dataset_idx,
    scaler,
    noise,
    masked,
    length_strategy,
    sys_idx,
):
    init_concs = training_data[dataset_idx][0] if scaler is None else testing_data[dataset_idx][0]
    kwargs = {
        "scaler": scaler,
        "key": key,
        "nucl_params": system_params[sys_idx]["nucl_params"],
        "growth_params": system_params[sys_idx]["growth_params"],
        "save_idxs": save_idxs,
        "noise": noise,
        "masked": masked,
    }
    if length_strategy is not None:
        kwargs["length_strategy"] = length_strategy

    ts, ys, out_scaler = simulateCrystallisation(init_concs, nt, **kwargs)

    n_exp = len(init_concs)
    n_out = len(save_idxs)

    assert ts.shape == (nt,)
    assert ys.shape == (n_exp, nt, n_out)
    if scaler is None:
        assert out_scaler is not None
    elif scaler == "no_scale":
        assert out_scaler is None


# ---------------------------------------------------------------------
# dataloader: just batch dimension integrity
# ---------------------------------------------------------------------
def test_dataloader_batching_shapes(training_data, system_params, nt, save_idxs, key):
    init_concs = training_data[0][0]
    _, ys, _ = simulateCrystallisation(
        init_concs,
        nt,
        scaler=None,
        key=key,
        nucl_params=system_params[0]["nucl_params"],
        growth_params=system_params[0]["growth_params"],
        save_idxs=save_idxs,
    )
    N = ys.shape[0]
    system_ids = jnp.array([training_data[0][1]] * N)
    batch_size = 8  # matches your __main__

    loader = dataloader((ys, system_ids), batch_size, key=key)

    seen = 0
    trailing = ys.shape[1:]
    while seen < N:
        yb, sb = next(loader)
        assert yb.shape[1:] == trailing
        assert sb.ndim == 1 and sb.shape[0] == yb.shape[0]
        seen += yb.shape[0]

    assert seen == N


# ---------------------------------------------------------------------
# SystemAugmentedNeuralODE: shapes of returned trajectories only
# ---------------------------------------------------------------------
def test_system_augmented_ode_output_shapes(model_hparams, nt, save_idxs, key):
    data_size = len(save_idxs)
    ts = jnp.linspace(0.0, 1.0, nt)
    y0 = jnp.zeros((data_size,))
    system_embedding = jnp.zeros((model_hparams["embed_proj_size"],))

    ode = SystemAugmentedNeuralODE(
        data_size=data_size,
        augment_size=model_hparams["augment_dim"],
        system_embed_size=model_hparams["embed_proj_size"],
        width_size=model_hparams["width"],
        depth=model_hparams["depth"],
        include_time=model_hparams["include_time"],
        key=key,
    )

    ys = ode(ts, y0, system_embedding=system_embedding, return_augmented=False)
    ys_full = ode(ts, y0, system_embedding=system_embedding, return_augmented=True)

    assert ys.shape == (nt, data_size)
    assert ys_full.shape == (
        nt,
        data_size + model_hparams["augment_dim"] + model_hparams["embed_proj_size"],
    )


# ---------------------------------------------------------------------
# Learnable path: embedding creation + projection call contract
# (We avoid heavy deps by using a minimal linear projector and just
# check that the ODE call with a projected embedding returns (T, d))
# ---------------------------------------------------------------------
def test_embedding_lookup_and_projection_shapes(system_params, model_hparams, nt, save_idxs, key):
    # one-hot / normalized embeddings from your helper
    sys_embeds = create_system_embedding(system_params, normalize=True)
    raw_embed_size = len(next(iter(sys_embeds.values())))
    initial_embeddings = jnp.stack([sys_embeds[i] for i in range(len(sys_embeds))])

    data_size = len(save_idxs)
    ode = SystemAugmentedNeuralODE(
        data_size=data_size,
        augment_size=model_hparams["augment_dim"],
        system_embed_size=model_hparams["embed_proj_size"],
        width_size=model_hparams["width"],
        depth=model_hparams["depth"],
        include_time=model_hparams["include_time"],
        key=key,
    )

    # lightweight linear projector raw -> rich (no external deps)
    proj_key = jr.PRNGKey(123)
    W = jr.normal(proj_key, (raw_embed_size, model_hparams["embed_proj_size"]))

    def project(x):
        return x @ W

    ts = jnp.linspace(0.0, 1.0, nt)
    y0 = jnp.zeros((data_size,))
    sid = 0
    rich_embed = project(initial_embeddings[sid])

    out = ode(ts, y0, system_embedding=rich_embed, return_augmented=False)
    assert out.shape == (nt, data_size)
