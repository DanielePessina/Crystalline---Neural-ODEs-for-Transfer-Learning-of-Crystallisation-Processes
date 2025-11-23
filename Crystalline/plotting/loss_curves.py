"""Loss curve visualisations."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["grid.alpha"] = 0.4
mpl.rcParams["grid.linewidth"] = 0.6
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True


def plot_loss_curves(train_losses, test_losses=None, title="Loss Curve", saveplot=False, filename="loss_curve.png"):
    """Plot training and optional test loss curves."""
    plt.figure(figsize=(7, 4))
    train_losses = np.array(train_losses)
    window_size = min(50, len(train_losses) // 10)
    kernel = np.ones(window_size) / window_size
    smooth_losses = np.convolve(train_losses, kernel, mode="valid")

    plt.plot(train_losses, color="lightblue", alpha=0.4)
    offset = (window_size - 1) // 2
    x_vals = list(range(offset, len(train_losses) - offset))[1:]
    min_len = min(len(x_vals), len(smooth_losses))
    plt.plot(x_vals[:min_len], smooth_losses[:min_len], label="Train Loss (Smoothed)", color="darkblue", linewidth=2)

    if test_losses is not None and len(test_losses) > 0:
        plt.plot(
            list(range(0, len(train_losses), max(1, len(train_losses) // len(test_losses))))[: len(test_losses)],
            test_losses,
            label="Test Loss",
            color="orange",
            marker="o",
            linestyle="--",
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.yscale("log")
    plt.ylim(top=1e1, bottom=1e-4)
    plt.legend()
    plt.tight_layout()
    if saveplot:
        plt.savefig(filename)
    plt.show()
