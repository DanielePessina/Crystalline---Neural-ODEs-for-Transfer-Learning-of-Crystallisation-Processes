"""Convenience exports for the :mod:`Crystalline` package.

This ``__init__`` file collects the most commonly used training
functions and utilities from the submodules so that they can be
imported directly from :mod:`Crystalline`.
"""

# from .augmented import train_AugNODE_fresh, train_AugNODE_TL, train_AugNODE_TL_penalise_deviation
from .augmented import train_AugNODE_fresh, train_AugNODE_TL, train_AugNODE_TL_penalise_deviation
from .augmented_domain_learnable import create_system_embedding, train_SystemAugNODElearnable_fresh
from .constrained import train_AugNODEconstrained_fresh, train_AugNODEconstrained_TL
from .data_functions import (
    simulateCrystallisation,
)
from .metrics.calculations import calculate_all_metrics
from .plotting import (
    plot_ensemble_predictions,
    plot_loss_curves,
    splitplot_model_vs_data,
    splitplot_model_vs_data_1d,
)

_all__ = [
    "train_AugNODE_fresh",
    "train_AugNODE_TL",
    "train_AugNODE_TL_penalise_deviation",
    "create_system_embedding",
    "train_SystemAugNODElearnable_fresh",
    "train_AugNODEconstrained_fresh",
    "train_AugNODEconstrained_TL",
    "simulateCrystallisation",
    "calculate_all_metrics",
    "plot_ensemble_predictions",
    "plot_loss_curves",
    "splitplot_model_vs_data",
    "splitplot_model_vs_data_1d",
]
