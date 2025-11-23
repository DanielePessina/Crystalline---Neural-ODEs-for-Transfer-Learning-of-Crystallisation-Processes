"""Dash-friendly training API wrapper for Crystalline.

This module provides a clean interface for training Augmented NODEs in a Dash
application context, with progress callbacks and structured return types.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import plotly.graph_objects as go

# Import the Dash-specific training function
from Crystalline.dash.training import (
    train_AugNODE_dash_realtime,
    train_AugNODE_TL_dash_realtime,
    train_AugNODE_TL_penalty_dash_realtime,
)

ProgressFn = Callable[[str], None]  # receives incremental log text


@dataclass
class TrainConfig:
    """Configuration for training an Augmented NODE."""

    base_inits: list[float]
    test_inits: list[float]
    nucl_params: tuple[float, float]
    growth_params: tuple[float, float]
    lr_strategy: tuple[float, float]
    steps_strategy: tuple[int, int]
    length_strategy: tuple[float, float]
    width_size: int
    depth: int
    activation: object
    ntimesteps: int
    seed: int
    noise: bool
    noise_level: float
    output_constraints: dict[int, str]
    include_time: bool = True
    batch_size: str = "all"
    augment_dim: int = 2
    splitplot: bool = False
    make_plots: bool = True


@dataclass
class TrainResult:
    """Results from training an Augmented NODE."""

    ts: object
    ys_train: object
    ys_test: object
    model: object
    scaler: object
    metrics: dict[str, float]
    figures: list[go.Figure]


@dataclass
class TLConfig:
    """Configuration for transfer learning of an Augmented NODE."""

    base_inits: list[float]
    test_inits: list[float]
    nucl_params: tuple[float, float]
    growth_params: tuple[float, float]
    lr_strategy: tuple[float, float]
    steps_strategy: tuple[int, int]
    length_strategy: tuple[float, float]
    ntimesteps: int
    seed: int
    noise: bool
    noise_level: float
    scale_strategy: str
    idx_frozen: int | tuple[int, int] | str
    freeze_mode: str
    penalise_deviations: bool = False
    penalty_lambda: float = 1.0
    penalty_strategy: int | tuple[int, int] | str = "all"


def train_AugNODE_dash(
    cfg: TrainConfig, progress: ProgressFn | None = None, progress_updater: Callable | None = None
) -> TrainResult:
    """Dash-oriented wrapper around the existing training pipeline.

    Parameters
    ----------
    cfg : TrainConfig
        Training configuration containing all hyperparameters.
    progress : ProgressFn, optional
        Callback function to emit progress messages.

    Returns
    -------
    TrainResult
        Structured results including model, scaler, metrics, and optional figures.
    """

    def log(msg: str):
        if progress:
            progress(msg if msg.endswith("\n") else msg + "\n")

    log("[setup] Seeding & preparing data…")

    # Call the original training function with explicit kwargs
    log("[fit] Starting training run…")
    try:
        ts, ys_train, ys_test, model, scaler, metrics_list = train_AugNODE_dash_realtime(
            cfg.base_inits,
            cfg.test_inits,
            lr_strategy=cfg.lr_strategy,
            steps_strategy=cfg.steps_strategy,
            length_strategy=cfg.length_strategy,
            width_size=cfg.width_size,
            depth=cfg.depth,
            activation=cfg.activation,
            ntimesteps=cfg.ntimesteps,
            seed=cfg.seed,
            splitplot=cfg.splitplot,
            plotly_plots=cfg.make_plots,
            noise=cfg.noise,
            noise_level=cfg.noise_level,
            batch_size=cfg.batch_size,
            augment_dim=cfg.augment_dim,
            nucl_params=list(cfg.nucl_params),
            growth_params=list(cfg.growth_params),
            output_constraints=cfg.output_constraints,
            include_time=cfg.include_time,
            print_every=100,
            progress_callback=log,  # Pass the Dash progress callback
            progress_updater=progress_updater,  # Pass the progress updater for real-time updates
        )
        log("[fit] Training complete.")

        # Extract scalar metrics from the metrics list
        metrics_dict = {}
        if metrics_list:
            # Aggregate metrics from the list
            for metric_dict in metrics_list:
                if isinstance(metric_dict, dict):
                    for key, value in metric_dict.items():
                        if isinstance(value, int | float | jnp.ndarray):
                            # Convert JAX arrays to Python scalars if needed
                            if hasattr(value, "item"):
                                metrics_dict[key] = value.item()
                            else:
                                metrics_dict[key] = value

        figures: list[go.Figure] = []
        if cfg.make_plots:
            log("[plot] Building figures…")
            figures = build_training_figures(
                ts, ys_train, ys_test, metrics_dict, model=model, scaler=scaler, length_strategy=cfg.length_strategy
            )
            log(f"[plot] Generated {len(figures)} figures.")

        log("[summary] Computing summary table…")

        return TrainResult(ts, ys_train, ys_test, model, scaler, metrics_dict, figures)

    except Exception as e:
        log(f"[error] Training failed: {str(e)}")
        raise


def train_AugNODE_TL_dash(
    cfg: TLConfig,
    base_model,
    base_scaler,
    progress: ProgressFn | None = None,
    progress_updater: Callable | None = None,
) -> TrainResult:
    """Dash wrapper for transfer learning training.

    Chooses between freezing-only TL and penalty TL based on cfg.penalise_deviations.
    """

    def log(msg: str):
        if progress:
            progress(msg if msg.endswith("\n") else msg + "\n")

    log("[setup] Preparing TL run…")

    try:
        if cfg.penalise_deviations:
            ts, ys_train, ys_test, model, scaler, metrics_list = train_AugNODE_TL_penalty_dash_realtime(
                cfg.base_inits,
                cfg.test_inits,
                model=base_model,
                scaler=base_scaler,
                penalty_lambda=cfg.penalty_lambda,
                penalty_strategy=cfg.penalty_strategy,
                ntimesteps=cfg.ntimesteps,
                lr_strategy=cfg.lr_strategy,
                steps_strategy=cfg.steps_strategy,
                length_strategy=cfg.length_strategy,
                seed=cfg.seed,
                noise=cfg.noise,
                noise_level=cfg.noise_level,
                scale_strategy=cfg.scale_strategy,
                nucl_params=list(cfg.nucl_params),
                growth_params=list(cfg.growth_params),
                progress_callback=log,
                progress_updater=progress_updater,
            )
        else:
            ts, ys_train, ys_test, model, scaler, metrics_list = train_AugNODE_TL_dash_realtime(
                cfg.base_inits,
                cfg.test_inits,
                model=base_model,
                scaler=base_scaler,
                idx_frozen=cfg.idx_frozen,
                freeze_mode=cfg.freeze_mode,
                ntimesteps=cfg.ntimesteps,
                lr_strategy=cfg.lr_strategy,
                steps_strategy=cfg.steps_strategy,
                length_strategy=cfg.length_strategy,
                seed=cfg.seed,
                noise=cfg.noise,
                noise_level=cfg.noise_level,
                scale_strategy=cfg.scale_strategy,
                nucl_params=list(cfg.nucl_params),
                growth_params=list(cfg.growth_params),
                progress_callback=log,
                progress_updater=progress_updater,
            )

        metrics_dict: dict[str, float] = {}
        if metrics_list:
            for metric_dict in metrics_list:
                if isinstance(metric_dict, dict):
                    for key, value in metric_dict.items():
                        if isinstance(value, int | float | jnp.ndarray):
                            if hasattr(value, "item"):
                                metrics_dict[key] = value.item()
                            else:
                                metrics_dict[key] = value

        figures: list[go.Figure] = []
        figures = build_training_figures(
            ts, ys_train, ys_test, metrics_dict, model=model, scaler=scaler, length_strategy=cfg.length_strategy
        )

        return TrainResult(ts, ys_train, ys_test, model, scaler, metrics_dict, figures)

    except Exception as e:
        log(f"[error] TL Training failed: {str(e)}")
        raise


def build_training_figures(
    ts, ys_train, ys_test, metrics, model=None, scaler=None, length_strategy=None
) -> list[go.Figure]:
    """Build Plotly figures from training results using the sophisticated plotly_plots functions.

    Parameters
    ----------
    ts : array-like
        Time points.
    ys_train, ys_test : array-like
        Training and testing trajectories.
    metrics : dict
        Metrics dictionary containing scalar values.
    model : object, optional
        Trained model for generating predictions.
    scaler : object, optional
        Fitted scaler for inverse transformation.
    length_strategy : list, optional
        List of training length fractions for vertical lines.

    Returns
    -------
    List[go.Figure]
        List of Plotly figures for display in Dash.
    """
    figs: list[go.Figure] = []

    try:
        # Import the plotting functions from Crystalline
        import numpy as np

        from Crystalline.plotting.plotly_plots import (
            interactive_splitplot_model_vs_data,
            interactive_splitplot_model_vs_data_1d,
        )

        # Convert JAX arrays to numpy if needed
        if hasattr(ts, "shape"):
            ts = np.array(ts)
        if hasattr(ys_train, "shape"):
            ys_train = np.array(ys_train)
        if hasattr(ys_test, "shape"):
            ys_test = np.array(ys_test)

        # Create a title string from metrics
        title_parts = []
        if "RMSE" in metrics:
            title_parts.append(f"RMSE: {metrics['RMSE']:.4f}")
        if "Concentration_RMSE" in metrics and "D43_RMSE" in metrics:
            title_parts.append(f"Train: {metrics['Concentration_RMSE']:.4f} Test: {metrics['D43_RMSE']:.4f}")

        extratitlestring = f" (Fresh)\n{' '.join(title_parts)}" if title_parts else " (Fresh)"

        # Use default length strategy if not provided
        if length_strategy is None:
            length_strategy = [0.33, 1.0]

        # Check if we have model and scaler for predictions
        if model is not None and scaler is not None:
            # Determine if we have 1D or 2D data
            data_size = ys_train.shape[-1] if ys_train.size > 0 else 1

            if data_size == 1:
                # Use 1D plotting function
                fig = interactive_splitplot_model_vs_data_1d(
                    ts=ts,
                    ys_train=ys_train,
                    ys_test=ys_test,
                    model=model,
                    scaler=scaler,
                    length_strategy=length_strategy,
                    extratitlestring=extratitlestring,
                    saveplot=False,
                    output_mode="notebook",
                )
            else:
                # Use 2D plotting function
                fig = interactive_splitplot_model_vs_data(
                    ts=ts,
                    ys_train=ys_train,
                    ys_test=ys_test,
                    model=model,
                    scaler=scaler,
                    length_strategy=length_strategy,
                    extratitlestring=extratitlestring,
                    saveplot=False,
                    output_mode="notebook",
                )

            figs.append(fig)

        # else:
        #     # Fallback to simple plotting if model/scaler not available
        #     # Create a basic comparison plot
        #     fig = go.Figure()

        #     # Add training data
        #     if ys_train is not None and ys_train.size > 0:
        #         for i, traj in enumerate(ys_train):
        #             fig.add_trace(
        #                 go.Scatter(
        #                     x=ts[:len(traj)] if len(ts) >= len(traj) else ts,
        #                     y=traj[:, 0] if len(traj.shape) > 1 else traj,
        #                     mode="markers",
        #                     name=f"Train {i+1}",
        #                     opacity=0.7
        #                 )
        #             )

        #     # Add test data
        #     if ys_test is not None and ys_test.size > 0:
        #         for i, traj in enumerate(ys_test):
        #             fig.add_trace(
        #                 go.Scatter(
        #                     x=ts[:len(traj)] if len(ts) >= len(traj) else ts,
        #                     y=traj[:, 0] if len(traj.shape) > 1 else traj,
        #                     mode="markers",
        #                     name=f"Test {i+1}",
        #                     opacity=0.7
        #                 )
        #             )

        #     fig.update_layout(
        #         title=f"Training Results{extratitlestring}",
        #         xaxis_title="Time",
        #         yaxis_title="Value",
        #         showlegend=True,
        #         width=1000,
        #         height=400
        #     )
        #     figs.append(fig)

        # # Add metrics summary figure
        # if metrics:
        #     # Filter out non-numeric metrics for display
        #     display_metrics = {}
        #     for key, value in metrics.items():
        #         if isinstance(value, int | float) and not np.isnan(value):
        #             # Shorten long metric names
        #             short_key = key.replace('Concentration_', 'Conc_').replace('_RMSE', ' RMSE').replace('_MAE', ' MAE')
        #             display_metrics[short_key] = value

        #     if display_metrics:
        #         metric_names = list(display_metrics.keys())
        #         metric_values = list(display_metrics.values())

        #         fig_metrics = go.Figure(data=[
        #             go.Bar(
        #                 x=metric_names,
        #                 y=metric_values,
        #                 text=[f"{v:.4f}" for v in metric_values],
        #                 textposition='auto',
        #             )
        #         ])
        #         fig_metrics.update_layout(
        #             title="Training Metrics Summary",
        #             xaxis_title="Metric",
        #             yaxis_title="Value",
        #             showlegend=False,
        #             width=1000,
        #             height=400
        #         )
        #         figs.append(fig_metrics)

    except Exception as e:
        # If plotting fails, return an error figure
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error generating plots: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        error_fig.update_layout(title="Plotting Error", width=1000, height=400)
        figs.append(error_fig)

    return figs
