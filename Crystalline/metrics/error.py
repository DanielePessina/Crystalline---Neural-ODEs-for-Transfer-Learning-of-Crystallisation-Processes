from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def make_failure_metrics(
    concentrations: Iterable[float],
    tag: str,
    ntimesteps: int,
    noise_level: float,
) -> list[dict[str, float | int | str]]:
    """Return placeholder metric dictionaries for failed training runs."""
    concs = list(concentrations)
    metrics_list: list[dict[str, float | int | str]] = []
    for conc in concs:
        metrics_list.append(
            {
                "Training_Experiments": len(concs),
                "Training_Timepoints": ntimesteps,
                "Experiment_Tag": tag,
                "Initial_Concentration": conc,
                "Measurement_Noise": noise_level,
                "MAE_Total": 999.0,
                "MSE_Total": 999.0,
                "RMSE_Total": 999.0,
                "MAPE_Total": 999.0,
                "MAE_Concentration": 999.0,
                "MSE_Concentration": 999.0,
                "RMSE_Concentration": 999.0,
                "MAPE_Concentration": 999.0,
                "MAE_D43": 999.0,
                "MSE_D43": 999.0,
                "RMSE_D43": 999.0,
                "MAPE_D43": 999.0,
                "Final_Train_Loss": 999.0,
                "Final_Test_Loss": 999.0,
            }
        )
    return metrics_list
