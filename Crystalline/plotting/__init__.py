"""Plotting helpers split into submodules."""

from .comparison_plots import (
    plot_ensemble_predictions,
    splitplot_model_vs_data,
    splitplot_model_vs_data_1d,
    splitplot_model_vs_data_latent,
    splitplot_system_augmented_model_vs_data,
    splitplot_system_augmented_model_vs_data_learnable,
    splitplot_system_encoded_model_vs_data,
)
from .loss_curves import plot_loss_curves
from .plotly_plots import (
    interactive_splitplot_model_vs_data,
    interactive_splitplot_model_vs_data_1d,
    setup_plotly_output,
)

__all__ = [
    "splitplot_model_vs_data",
    "splitplot_model_vs_data_1d",
    "plot_ensemble_predictions",
    "splitplot_model_vs_data_latent",
    "splitplot_system_encoded_model_vs_data",
    "splitplot_system_augmented_model_vs_data",
    "splitplot_system_augmented_model_vs_data_learnable",
    "plot_loss_curves",
    "interactive_splitplot_model_vs_data",
    "interactive_splitplot_model_vs_data_1d",
    "setup_bokeh_output",
]
