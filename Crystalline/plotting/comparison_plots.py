"""Plotting utilities for training and evaluation results."""

import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pyfonts import load_google_font, set_default_font

mpl.rcParams["figure.dpi"] = 600

# set_default_font(load_google_font("Roboto"))  # Sans-serif

# set_default_font(load_google_font("Roboto Mono")) # Similar to courier

set_default_font(load_google_font("Roboto Slab"))  # Serif

# set_default_font(load_google_font("Merriweather")) # Serif

# set_default_font(load_google_font("Open Sans")) # Sans-serif

# set_default_font(load_google_font("Libertinus Serif"))  # Serif

# Set Libertinus font for all text
# mpl.rcParams["font.family"] = "serif"
# mpl.rcParams["font.serif"] = ["Libertinus Serif", "Times", "DejaVu Serif"]
# # Use 'custom' for mathtext.fontset to allow custom fonts
# mpl.rcParams["mathtext.fontset"] = "custom"
# mpl.rcParams["mathtext.rm"] = "Libertinus Serif"
# mpl.rcParams["mathtext.it"] = "Libertinus Serif:italic"
# mpl.rcParams["mathtext.bf"] = "Libertinus Serif:bold"

# Set rcParams for minor gridlines
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["grid.alpha"] = 0.4
mpl.rcParams["grid.linewidth"] = 0.6
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
# mpl.rcParams["grid.which"] = "both"


def splitplot_model_vs_data(
    ts,
    ys_train,
    ys_test,
    model,
    scaler,
    length_strategy,
    extratitlestring,
    saveplot,
    filename_prefix="neural_ode_plot",
):
    """Plot model vs data for training and test sets."""
    n_train = len(ys_train)
    n_test = len(ys_test)
    data_size = ys_train.shape[-1]
    length_size = ys_train.shape[1]
    train_colors = [plt.cm.Set2(i % plt.cm.Set2.N) for i in range(n_train)]
    test_colors = [plt.cm.Dark2(i % plt.cm.Dark2.N) for i in range(n_test)]

    ts_pred = np.linspace(0, 300, length_size)  # Adjusted to match the original time scale

    # Create 2x2 subplot layout
    fig, ((ax1_train, ax1_test), (ax2_train, ax2_test)) = plt.subplots(2, 2, figsize=(8, 6), sharey="row", sharex=True)

    # Plot training data and predictions (left column)
    for i, color in enumerate(train_colors):
        y_unscaled = scaler.inverse_transform(np.array(ys_train[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )
        # y_unscaled = ys_train[i]
        ax1_train.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=22, alpha=0.7)
        ax2_train.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=22, alpha=0.7)

        y_pred = np.array(model(ts, ys_train[i, 0])).reshape(-1, data_size)

        y_unscaled = scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        # y_unscaled = y_pred

        ax1_train.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_train.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Plot test data and predictions (right column)
    for i, color in enumerate(test_colors):
        y_unscaled = scaler.inverse_transform(np.array(ys_test[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )
        # y_unscaled = ys_test[i]
        ax1_test.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=30, alpha=0.7)
        ax2_test.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=30, alpha=0.7)

        y_pred = np.array(model(ts, ys_test[i, 0])).reshape(-1, data_size)

        y_unscaled = scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        # y_unscaled = y_pred  # scaler.inverse_transform(y_pred).reshape(length_size, data_size)

        ax1_test.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_test.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Add vertical lines for splits
    for length in length_strategy:
        split_idx = int(length_size * length)
        if split_idx < len(ts):
            for ax in [ax1_train, ax1_test, ax2_train, ax2_test]:
                ax.axvline(x=ts_pred[split_idx], color="gray", linestyle=":", alpha=0.5)

    # Set labels
    ax2_train.set_xlabel("Time [min]")
    ax2_test.set_xlabel("Time [min]")
    ax1_train.set_ylabel("Concentration [mg/ml]")
    ax2_train.set_ylabel("d43 [µm]")

    # Set titles
    ax1_train.set_title("Training Set", fontsize=10)
    ax1_test.set_title("Test Set", fontsize=10)
    fig.suptitle(f"{extratitlestring}", fontsize=12)

    # Add legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Real Train Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Real Test Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=1.5,
            linestyle="-",
            alpha=0.7,
            label="Model Prediction",
        ),
    ]
    ax2_test.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if saveplot:
        plt.savefig(f"plots/{filename_prefix}")
    plt.show()


def splitplot_model_vs_data_1d(
    ts,
    ys_train,
    ys_test,
    model,
    scaler,
    length_strategy,
    extratitlestring,
    saveplot,
    filename_prefix="neural_ode_plot",
):
    """Plot model vs data for 1D data."""
    n_train = len(ys_train)
    n_test = len(ys_test)
    data_size = ys_train.shape[-1]
    length_size = ys_train.shape[1]
    train_colors = [plt.cm.Set2(i % plt.cm.Set2.N) for i in range(n_train)]
    test_colors = [plt.cm.Dark2(i % plt.cm.Dark2.N) for i in range(n_test)]

    ts_pred = np.linspace(0, 300, length_size)  # Adjusted to match the original time scale

    if data_size != 1:
        # Call the original function for data_size > 1
        splitplot_model_vs_data(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            extratitlestring,
            saveplot,
            filename_prefix=filename_prefix,
        )
        return

    # Only two subplots needed: train and test
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)

    # Plot training data and predictions (left)
    for i, color in enumerate(train_colors):
        y_unscaled = scaler.inverse_transform(np.array(ys_train[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )
        ax_train.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=22, alpha=0.7)
        y_pred = np.array(model(ts, ys_train[i, 0])).reshape(-1, data_size)
        y_unscaled_pred = scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        ax_train.plot(ts_pred, y_unscaled_pred[:, 0], color=color, linestyle="-", alpha=0.7)

    # Plot test data and predictions (right)
    for i, color in enumerate(test_colors):
        y_unscaled = scaler.inverse_transform(np.array(ys_test[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )
        ax_test.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=30, alpha=0.7)
        y_pred = np.array(model(ts, ys_test[i, 0])).reshape(-1, data_size)
        y_unscaled_pred = scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        ax_test.plot(ts_pred, y_unscaled_pred[:, 0], color=color, linestyle="-", alpha=0.7)

    # Add vertical lines for splits
    for length in length_strategy:
        split_idx = int(length_size * length)
        if split_idx < len(ts):
            for ax in [ax_train, ax_test]:
                ax.axvline(x=ts_pred[split_idx], color="gray", linestyle=":", alpha=0.5)

    # Set labels and titles
    ax_train.set_xlabel("Time [min]")
    ax_test.set_xlabel("Time [min]")
    ax_train.set_ylabel("Value")
    ax_train.set_title("Training Set", fontsize=10)
    ax_test.set_title("Test Set", fontsize=10)
    fig.suptitle(f"{extratitlestring}", fontsize=12)

    # Add legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Real Train Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Real Test Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=1.5,
            linestyle="-",
            alpha=0.7,
            label="Model Prediction",
        ),
    ]
    ax_test.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if saveplot:
        plt.savefig(f"plots/{filename_prefix}")
    plt.show()


def plot_ensemble_predictions(
    ts,
    ys_data,
    models,
    scaler,
    title,
    saveplot=False,
    filename_prefix="ensemble_plot",
    length_strategy=None,
):
    """Plot ensemble predictions for a given dataset (train or test).

    Args:
        ts: Time array.
        ys_data: Ground truth data, shape (N, T, D).
        models: List of trained NeuralODE models.
        scaler: Fitted scaler to invert scaling.
        title: Title string for the plot.
        saveplot: Whether to save the plot.
        filename_prefix: Filename prefix for saving.
    """
    n_samples = ys_data.shape[0]
    length_size = ys_data.shape[1]
    data_size = ys_data.shape[2]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True)

    # Assign colors to ensemble members: first half Set2, second half Dark2 using model index
    n_models = len(models)
    n_half = n_models // 2
    set2_colors = plt.cm.Set2(np.linspace(0, 1, n_half))
    dark2_colors = plt.cm.Dark2(np.linspace(0, 1, n_models - n_half))
    model_colors = np.concatenate([set2_colors, dark2_colors], axis=0)

    all_preds = []
    # Plot ensemble member predictions
    for model_idx, (model, color) in enumerate(zip(models, model_colors, strict=False)):
        preds = []
        for i in range(n_samples):
            pred = np.array(model(ts, ys_data[i, 0]))
            preds.append(pred)
            unscaled = scaler.inverse_transform(pred).reshape(length_size, data_size)
            ax1.plot(ts, unscaled[:, 0], color=color, alpha=0.3)
            ax2.plot(ts, unscaled[:, -1], color=color, alpha=0.3)
        all_preds.append(np.stack(preds))  # shape (N, T, D)

    # Compute mean prediction across ensemble
    mean_preds = np.mean(np.stack(all_preds), axis=0)  # shape (N, T, D)
    for i in range(n_samples):
        unscaled = scaler.inverse_transform(mean_preds[i]).reshape(length_size, data_size)
        ax1.plot(ts, unscaled[:, 0], color="black", linewidth=2)
        ax2.plot(ts, unscaled[:, -1], color="black", linewidth=2)

    # Add vertical lines for split indices if length_strategy is provided
    if length_strategy:
        for length in length_strategy:
            split_idx = int(length_size * length)
            if split_idx < len(ts):
                ax1.axvline(x=ts[split_idx], color="gray", linestyle=":", alpha=0.5)
                ax2.axvline(x=ts[split_idx], color="gray", linestyle=":", alpha=0.5)

    ax1.set_ylabel("Concentration")
    ax2.set_ylabel("D43")
    ax2.set_xlabel("Time")
    ax1.set_title(f"{title}", fontsize=10)

    # Updated legend using Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Mean Prediction"),
        Line2D([0], [0], color="gray", alpha=0.3, lw=2, label="Ensemble Member"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()
    if saveplot:
        plt.savefig(f"plots/{filename_prefix}.png")
    plt.show()


def splitplot_model_vs_data_latent(
    ts,
    ys_train,
    ys_test,
    model,
    scaler,
    length_strategy,
    extratitlestring,
    saveplot,
    filename_prefix="neural_ode_plot",
):
    """Plot latent model vs data."""
    n_train = len(ys_train)
    n_test = len(ys_test)
    data_size = ys_train.shape[-1]
    length_size = ys_train.shape[1]
    train_colors = [plt.cm.Set2(i % plt.cm.Set2.N) for i in range(n_train)]
    test_colors = [plt.cm.Dark2(i % plt.cm.Dark2.N) for i in range(n_test)]

    ts_pred = np.linspace(0, 300, length_size)  # Adjusted to match the original time scale
    # Keys for stochastic predictions
    master_key = jr.PRNGKey(0)
    keys_all = jr.split(master_key, n_train + n_test)
    keys_train = keys_all[:n_train]
    keys_test = keys_all[n_train:]

    # Create 2x2 subplot layout
    fig, ((ax1_train, ax1_test), (ax2_train, ax2_test)) = plt.subplots(2, 2, figsize=(12, 8), sharey="row", sharex=True)

    # Plot training data and predictions (left column)
    for i, color in enumerate(train_colors):
        # y_unscaled = scaler.inverse_transform(np.array(ys_train[i]).reshape(-1, data_size)).reshape(
        #     length_size, data_size
        # )
        y_unscaled = ys_train[i]
        ax1_train.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=22, alpha=0.7)
        ax2_train.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=22, alpha=0.7)

        pred, _ = model(ts, ys_train[i], keys_train[i])
        y_pred = np.array(pred).reshape(-1, data_size)
        y_unscaled = y_pred  # scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        ax1_train.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_train.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Plot test data and predictions (right column)
    for i, color in enumerate(test_colors):
        # y_unscaled = scaler.inverse_transform(np.array(ys_test[i]).reshape(-1, data_size)).reshape(
        #     length_size, data_size
        # )
        y_unscaled = ys_test[i]
        ax1_test.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=30, alpha=0.7)
        ax2_test.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=30, alpha=0.7)

        pred, _ = model(ts, ys_test[i], keys_test[i])
        y_pred = np.array(pred).reshape(-1, data_size)
        y_unscaled = y_pred  # scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        ax1_test.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_test.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Add vertical lines for splits
    for length in length_strategy:
        split_idx = int(length_size * length)
        if split_idx < len(ts):
            for ax in [ax1_train, ax1_test, ax2_train, ax2_test]:
                ax.axvline(x=ts_pred[split_idx], color="gray", linestyle=":", alpha=0.5)

    # Set labels
    ax2_train.set_xlabel("Time [min]")
    ax2_test.set_xlabel("Time [min]")
    ax1_train.set_ylabel("Concentration [mg/ml]")
    ax2_train.set_ylabel("D43 [µm]")

    # Set titles
    ax1_train.set_title("Training Set", fontsize=10)
    ax1_test.set_title("Test Set", fontsize=10)
    fig.suptitle(f"{extratitlestring}", fontsize=12)

    # Add legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Real Train Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Real Test Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=1.5,
            linestyle="-",
            alpha=0.7,
            label="Model Prediction",
        ),
    ]
    ax2_test.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if saveplot:
        lengths_str = "_".join([f"{length:.2f}" for length in length_strategy[:-1]])
        plt.savefig(f"plots/{filename_prefix}_{lengths_str}.png")
    plt.show()


def splitplot_system_encoded_model_vs_data(
    ts,
    ys_train,
    ys_test,
    model,
    scaler,
    train_system_ids,
    test_system_ids,
    length_strategy,
    extratitlestring,
    saveplot,
    filename_prefix="system_encoded_neural_ode_plot",
):
    """Plot system-encoded model predictions vs data with system-specific colors.

    Args:
        ts: Time array
        ys_train: Training data array
        ys_test: Test data array
        model: Trained SystemEncodedNeuralODE model
        scaler: Data scaler for inverse transformation
        train_system_ids: Array of system IDs for training data
        test_system_ids: Array of system IDs for test data
        length_strategy: Training length strategy for vertical lines
        extratitlestring: Additional title string
        saveplot: Whether to save the plot
        filename_prefix: Prefix for saved filename
    """
    data_size = ys_train.shape[-1]
    length_size = ys_train.shape[1]

    # Create system-specific colors
    unique_systems = jnp.unique(jnp.concatenate([train_system_ids, test_system_ids]))
    n_systems = len(unique_systems)
    system_colors = plt.cm.tab10(np.linspace(0, 1, n_systems))

    # Convert JAX arrays to Python integers for dictionary keys
    unique_systems_int = [int(sys_id) for sys_id in unique_systems]
    color_map = {sys_id: system_colors[i] for i, sys_id in enumerate(unique_systems_int)}

    # Convert system IDs to integers for color mapping
    train_colors = [color_map[int(sys_id)] for sys_id in train_system_ids]
    test_colors = [color_map[int(sys_id)] for sys_id in test_system_ids]

    ts_pred = np.linspace(0, 300, length_size)  # Adjusted to match the original time scale

    # Create 2x2 subplot layout
    fig, ((ax1_train, ax1_test), (ax2_train, ax2_test)) = plt.subplots(2, 2, figsize=(12, 8), sharey="row", sharex=True)

    # Plot training data and predictions (left column)
    for i, (color, system_id) in enumerate(zip(train_colors, train_system_ids, strict=False)):
        y_unscaled = scaler.inverse_transform(np.array(ys_train[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )
        ax1_train.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=22, alpha=0.7)
        ax2_train.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=22, alpha=0.7)

        y_pred = np.array(model(ts, ys_train[i, 0], int(system_id))).reshape(-1, data_size)
        y_unscaled = scaler.inverse_transform(y_pred).reshape(length_size, data_size)

        ax1_train.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_train.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Plot test data and predictions (right column)
    for i, (color, system_id) in enumerate(zip(test_colors, test_system_ids, strict=False)):
        y_unscaled = scaler.inverse_transform(np.array(ys_test[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )
        ax1_test.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=30, alpha=0.7)
        ax2_test.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=30, alpha=0.7)

        y_pred = np.array(model(ts, ys_test[i, 0], int(system_id))).reshape(-1, data_size)
        y_unscaled = scaler.inverse_transform(y_pred).reshape(length_size, data_size)

        ax1_test.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_test.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Add vertical lines for splits
    for length in length_strategy:
        split_idx = int(length_size * length)
        if split_idx < len(ts):
            for ax in [ax1_train, ax1_test, ax2_train, ax2_test]:
                ax.axvline(x=ts_pred[split_idx], color="gray", linestyle=":", alpha=0.5)

    # Set labels
    ax2_train.set_xlabel("Time [min]")
    ax2_test.set_xlabel("Time [min]")
    ax1_train.set_ylabel("Concentration [mg/ml]")
    ax2_train.set_ylabel("D43 [µm]")

    # Set titles
    ax1_train.set_title("Training Set", fontsize=10)
    ax1_test.set_title("Test Set", fontsize=10)
    fig.suptitle(f"{extratitlestring}", fontsize=12)

    # Add legend with system information
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Real Train Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Real Test Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=1.5,
            linestyle="-",
            alpha=0.7,
            label="Model Prediction",
        ),
    ]

    # Add system-specific legend entries
    for sys_id in unique_systems_int:
        legend_elements.append(Line2D([0], [0], color=color_map[sys_id], lw=2, label=f"System {sys_id}"))

    ax2_test.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if saveplot:
        lengths_str = "_".join([f"{length:.2f}" for length in length_strategy[:-1]])
        plt.savefig(f"plots/{filename_prefix}_{lengths_str}.png")
    plt.show()


def splitplot_system_augmented_model_vs_data(
    ts,
    ys_train,
    ys_test,
    model,
    scalers: dict,
    train_system_ids,
    test_system_ids,
    system_embeddings,
    length_strategy,
    extratitlestring,
    saveplot,
    filename_prefix="system_augmented_neural_ode_plot",
):
    """Plot system-augmented model predictions vs data with system-specific colors.

    Args:
        ts: Time array
        ys_train: Training data array
        ys_test: Test data array
        model: Trained SystemAugmentedNeuralODE model
        scalers: Dictionary mapping system IDs to data scalers for inverse transformation
        train_system_ids: Array of system IDs for training data
        test_system_ids: Array of system IDs for test data
        system_embeddings: Dictionary mapping system IDs to embedding arrays (required for fixed‑embedding models).
        length_strategy: Training length strategy for vertical lines
        extratitlestring: Additional title string
        saveplot: Whether to save the plot
        filename_prefix: Prefix for saved filename
    """
    data_size = ys_train.shape[-1]
    length_size = ys_train.shape[1]

    # Create system-specific colors
    unique_systems = jnp.unique(jnp.concatenate([train_system_ids, test_system_ids]))
    n_systems = len(unique_systems)
    system_colors = plt.cm.tab10(np.linspace(0, 1, n_systems))

    # Convert JAX arrays to Python integers for dictionary keys
    unique_systems_int = [int(sys_id) for sys_id in unique_systems]
    color_map = {sys_id: system_colors[i] for i, sys_id in enumerate(unique_systems_int)}

    # Convert system IDs to integers for color mapping
    train_colors = [color_map[int(sys_id)] for sys_id in train_system_ids]
    test_colors = [color_map[int(sys_id)] for sys_id in test_system_ids]

    ts_pred = np.linspace(0, 300, length_size)  # Adjusted to match the original time scale

    # Create 2x2 subplot layout
    fig, ((ax1_train, ax1_test), (ax2_train, ax2_test)) = plt.subplots(2, 2, figsize=(12, 8), sharey="row", sharex=True)

    # Plot training data and predictions (left column)
    for i, (color, system_id) in enumerate(zip(train_colors, train_system_ids, strict=False)):
        system_id_int = int(system_id)
        current_scaler = scalers.get(system_id_int)

        if current_scaler is not None:
            y_unscaled = current_scaler.inverse_transform(np.array(ys_train[i]).reshape(-1, data_size)).reshape(
                length_size, data_size
            )
        else:
            print(f"Warning: No scaler found for system {system_id_int}, using raw data")
            y_unscaled = np.array(ys_train[i])

        ax1_train.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=22, alpha=0.7)
        ax2_train.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=22, alpha=0.7)

        # Get system embedding and make prediction (fixed‑embedding models)
        # system_embedding = system_embeddings[system_id_int]
        y_pred = np.array(model(ts, ys_train[i, 0], system_id=system_id)).reshape(-1, data_size)

        if current_scaler is not None:
            y_unscaled = current_scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        else:
            y_unscaled = y_pred

        ax1_train.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_train.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Plot test data and predictions (right column)
    for i, (color, system_id) in enumerate(zip(test_colors, test_system_ids, strict=False)):
        system_id_int = int(system_id)
        current_scaler = scalers.get(system_id_int)

        if current_scaler is not None:
            y_unscaled = current_scaler.inverse_transform(np.array(ys_test[i]).reshape(-1, data_size)).reshape(
                length_size, data_size
            )
        else:
            print(f"Warning: No scaler found for system {system_id_int}, using raw data")
            y_unscaled = np.array(ys_test[i])

        ax1_test.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=30, alpha=0.7)
        ax2_test.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=30, alpha=0.7)

        # Get system embedding and make prediction (fixed‑embedding models)
        y_pred = np.array(model(ts, ys_test[i, 0], system_id=system_id)).reshape(-1, data_size)

        if current_scaler is not None:
            y_unscaled = current_scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        else:
            y_unscaled = y_pred

        ax1_test.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_test.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Add vertical lines for splits
    for length in length_strategy:
        split_idx = int(length_size * length)
        if split_idx < len(ts):
            for ax in [ax1_train, ax1_test, ax2_train, ax2_test]:
                ax.axvline(x=ts_pred[split_idx], color="gray", linestyle=":", alpha=0.5)

    # Set labels
    ax2_train.set_xlabel("Time [min]")
    ax2_test.set_xlabel("Time [min]")
    ax1_train.set_ylabel("Concentration [mg/ml]")
    ax2_train.set_ylabel("D43 [µm]")

    # Set titles
    ax1_train.set_title("Training Set", fontsize=10)
    ax1_test.set_title("Test Set", fontsize=10)
    fig.suptitle(f"{extratitlestring}", fontsize=12)

    # Add legend with system information
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Real Train Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Real Test Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=1.5,
            linestyle="-",
            alpha=0.7,
            label="Model Prediction",
        ),
    ]

    # Add system-specific legend entries
    for sys_id in unique_systems_int:
        legend_elements.append(Line2D([0], [0], color=color_map[sys_id], lw=2, label=f"System {sys_id}"))

    ax2_test.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if saveplot:
        lengths_str = "_".join([f"{length:.2f}" for length in length_strategy[:-1]])
        plt.savefig(f"plots/{filename_prefix}_{lengths_str}.png")
    plt.show()


# -----------------------------------------------------------------------------
# Learnable embedding version (system_id interface)
# -----------------------------------------------------------------------------


def splitplot_system_augmented_model_vs_data_learnable(
    ts,
    ys_train,
    ys_test,
    model,
    scalers: dict,
    train_system_ids,
    test_system_ids,
    length_strategy,
    extratitlestring,
    saveplot,
    filename_prefix="system_augmented_neural_ode_plot_learnable",
):
    """Same visualisation as `splitplot_system_augmented_model_vs_data`, but for *learnable*-embedding models that expect `system_id=` instead of a manual embedding vector."""
    data_size = ys_train.shape[-1]
    length_size = ys_train.shape[1]

    # Create system-specific colors
    unique_systems = jnp.unique(jnp.concatenate([train_system_ids, test_system_ids]))
    n_systems = len(unique_systems)
    system_colors = plt.cm.tab10(np.linspace(0, 1, n_systems))

    # Convert JAX arrays to Python integers for dictionary keys
    unique_systems_int = [int(sys_id) for sys_id in unique_systems]
    color_map = {sys_id: system_colors[i] for i, sys_id in enumerate(unique_systems_int)}

    # Convert system IDs to integers for color mapping
    train_colors = [color_map[int(sys_id)] for sys_id in train_system_ids]
    test_colors = [color_map[int(sys_id)] for sys_id in test_system_ids]

    ts_pred = np.linspace(0, 300, length_size)  # Adjusted to match the original time scale

    # Create 2x2 subplot layout
    fig, ((ax1_train, ax1_test), (ax2_train, ax2_test)) = plt.subplots(2, 2, figsize=(12, 8), sharey="row", sharex=True)

    # Plot training data and predictions (left column)
    for i, (color, system_id) in enumerate(zip(train_colors, train_system_ids, strict=False)):
        system_id_int = int(system_id)
        current_scaler = scalers.get(system_id_int)

        if current_scaler is not None:
            y_unscaled = current_scaler.inverse_transform(np.array(ys_train[i]).reshape(-1, data_size)).reshape(
                length_size, data_size
            )
        else:
            print(f"Warning: No scaler found for system {system_id_int}, using raw data")
            y_unscaled = np.array(ys_train[i])

        ax1_train.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=22, alpha=0.7)
        ax2_train.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=22, alpha=0.7)

        # Make prediction using system_id
        y_pred = np.array(model(ts, ys_train[i, 0], system_id=system_id_int)).reshape(-1, data_size)

        if current_scaler is not None:
            y_unscaled = current_scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        else:
            y_unscaled = y_pred

        ax1_train.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_train.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Plot test data and predictions (right column)
    for i, (color, system_id) in enumerate(zip(test_colors, test_system_ids, strict=False)):
        system_id_int = int(system_id)
        current_scaler = scalers.get(system_id_int)

        if current_scaler is not None:
            y_unscaled = current_scaler.inverse_transform(np.array(ys_test[i]).reshape(-1, data_size)).reshape(
                length_size, data_size
            )
        else:
            print(f"Warning: No scaler found for system {system_id_int}, using raw data")
            y_unscaled = np.array(ys_test[i])

        ax1_test.scatter(ts_pred, y_unscaled[:, 0], color=color, marker="o", s=30, alpha=0.7)
        ax2_test.scatter(ts_pred, y_unscaled[:, -1], color=color, marker="o", s=30, alpha=0.7)

        # Make prediction using system_id
        y_pred = np.array(model(ts, ys_test[i, 0], system_id=system_id_int)).reshape(-1, data_size)

        if current_scaler is not None:
            y_unscaled = current_scaler.inverse_transform(y_pred).reshape(length_size, data_size)
        else:
            y_unscaled = y_pred

        ax1_test.plot(ts_pred, y_unscaled[:, 0], color=color, linestyle="-", alpha=0.7)
        ax2_test.plot(ts_pred, y_unscaled[:, -1], color=color, linestyle="-", alpha=0.7)

    # Add vertical lines for splits
    for length in length_strategy:
        split_idx = int(length_size * length)
        if split_idx < len(ts):
            for ax in [ax1_train, ax1_test, ax2_train, ax2_test]:
                ax.axvline(x=ts_pred[split_idx], color="gray", linestyle=":", alpha=0.5)

    # Set labels
    ax2_train.set_xlabel("Time [min]")
    ax2_test.set_xlabel("Time [min]")
    ax1_train.set_ylabel("Concentration [mg/ml]")
    ax2_train.set_ylabel("D43 [µm]")

    # Set titles
    ax1_train.set_title("Training Set", fontsize=10)
    ax1_test.set_title("Test Set", fontsize=10)
    fig.suptitle(f"{extratitlestring}", fontsize=12)

    # Add legend with system information
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Real Train Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Real Test Data",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=1.5,
            linestyle="-",
            alpha=0.7,
            label="Model Prediction",
        ),
    ]

    # Add system-specific legend entries
    for sys_id in unique_systems_int:
        legend_elements.append(Line2D([0], [0], color=color_map[sys_id], lw=2, label=f"System {sys_id}"))

    ax2_test.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if saveplot:
        lengths_str = "_".join([f"{length:.2f}" for length in length_strategy[:-1]])
        plt.savefig(f"plots/{filename_prefix}_{lengths_str}.png")
    plt.show()
