from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.random as jr
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Compute standard regression metrics between predictions and targets."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    abs_errors = np.abs(y_true - y_pred)
    squared_errors = (y_true - y_pred) ** 2
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    percentage_errors = np.abs((y_true - y_pred) / y_true_safe) * 100
    return {
        f"{prefix}MAE": float(np.mean(abs_errors)),
        f"{prefix}MSE": float(np.mean(squared_errors)),
        f"{prefix}RMSE": float(np.sqrt(np.mean(squared_errors))),
        f"{prefix}MAPE": float(np.mean(percentage_errors[np.isfinite(percentage_errors)])),
    }


def calculate_all_metrics(
    ts: Iterable[float],
    ys: np.ndarray,
    model: Callable[[Iterable[float], np.ndarray], np.ndarray],
    scaler: Any,
    initial_concentrations: Iterable[float],
    tag: str,
    ntimesteps: int,
    noise_level: float,
    mask_d43: bool = False,
) -> list[dict[str, float | int | str]]:
    metrics_list = []
    for i, initial_conc in enumerate(initial_concentrations):
        pred = model(ts, ys[i, 0])
        metrics: dict[str, float] = {}
        metrics.update(compute_metrics_dict(ys[i].reshape(-1), pred.reshape(-1), prefix=""))
        metrics.update(
            compute_metrics_dict(
                ys[i, :, 0].reshape(-1),
                pred[:, 0].reshape(-1),
                prefix="Concentration_",
            )
        )
        if not mask_d43:
            metrics.update(
                compute_metrics_dict(
                    ys[i, :, 1].reshape(-1),
                    pred[:, 1].reshape(-1),
                    prefix="D43_",
                )
            )
        metrics_list.append(
            {
                "Experiment_Tag": tag,
                "Initial_Concentration": initial_conc,
                "Measurement_Noise": noise_level,
                **metrics,
            }
        )
    return metrics_list


def calculate_all_metrics_latent(
    ts: Iterable[float],
    ys: np.ndarray,
    model: Callable[[Iterable[float], np.ndarray, jr.PRNGKey], tuple[np.ndarray, Any]],
    key: jr.PRNGKey,
    scaler: Any,
    initial_concentrations: Iterable[float],
    tag: str,
    ntimesteps: int,
    noise_level: float,
    mask_d43: bool = False,
) -> list[dict[str, float | int | str]]:
    metrics_list = []
    keys = jr.split(key, len(initial_concentrations))
    for i, initial_conc in enumerate(initial_concentrations):
        pred, _ = model(ts, ys[i], keys[i])
        metrics: dict[str, float] = {}
        metrics.update(compute_metrics_dict(ys[i].reshape(-1), pred.reshape(-1), prefix=""))
        metrics.update(
            compute_metrics_dict(
                ys[i, :, 0].reshape(-1),
                pred[:, 0].reshape(-1),
                prefix="Concentration_",
            )
        )
        if not mask_d43:
            metrics.update(
                compute_metrics_dict(
                    ys[i, :, 1].reshape(-1),
                    pred[:, 1].reshape(-1),
                    prefix="D43_",
                )
            )
        metrics_list.append(
            {
                "Experiment_Tag": tag,
                "Initial_Concentration": initial_conc,
                "Measurement_Noise": noise_level,
                **metrics,
            }
        )
    return metrics_list
