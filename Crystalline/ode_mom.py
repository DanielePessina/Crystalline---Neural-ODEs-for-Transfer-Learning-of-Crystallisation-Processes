"""Method of Moments ODE solvers used for data generation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.integrate as spi


def solve_MoM_ODE(
    initial_concentration: float,
    params_nucl: list[float] | tuple[float, float],
    params_growth: list[float] | tuple[float, float],
    t_save: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve the Method of Moments ODE for a single experiment.

    Parameters
    ----------
    initial_concentration : float
        Starting concentration of the solute.
    params_nucl : list | tuple
        Kinetic parameters for nucleation.
    params_growth : list | tuple
        Kinetic parameters for crystal growth.
    t_save : array_like
        Array of time points at which the solution is required.

    Returns
    -------
    np.ndarray
        Array with columns ``[concentration, d43]`` evaluated at ``t_save``.
    """

    ## Physical Parameters

    temperature = 273.15 + 20  # K
    rho_c = 1370.0
    conc_sat = 2.47
    kv = 0.81
    molecular_volume = 2.97e-26
    kb = 1.38064852e-23

    def CNTRate(S: float) -> float:
        if S < 1 + 1e-5:
            return 0

        return (
            60
            * np.exp(params_nucl[0])
            * S
            * np.exp(
                -16
                * np.pi
                * (params_nucl[1] * 1e-3) ** 3
                * molecular_volume**2
                / (3 * (kb * temperature) ** 3 * (np.log(S)) ** 2)
            )
        )

    def GrowthRate(S: float) -> float:
        if S < 1 + 1e-5:
            return 0
        return 1e-9 * params_growth[0] * (S - 1) ** params_growth[1]

    def ode_model(t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        moments = y[:-1]
        S = y[-1] / conc_sat

        growth = GrowthRate(S)

        nucleation = CNTRate(S)

        return np.array(
            [
                nucleation,
                growth * moments[0],
                2 * growth * moments[1],
                3 * growth * moments[2],
                4 * growth * moments[3],
                -3 * kv * rho_c * growth * moments[2],
            ]
        )

    t_0, t_end = t_save[0], t_save[-1]

    y_0 = np.concatenate([np.zeros(5), [initial_concentration]])

    solution = spi.solve_ivp(ode_model, (t_0, t_end), y_0, t_eval=t_save)

    concentration = solution.y[-1]

    d43 = (solution.y[4] + 1e-25) / (solution.y[3] + 1e-6) * 1e6

    return np.transpose(np.vstack([concentration, d43]))


def solve_MoM_ODE_icarr(
    initial_concentrations: list[float],
    params_nucl: list[float] | tuple[float, float],
    params_growth: list[float] | tuple[float, float],
    t_save: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Vectorised wrapper over :func:`solve_MoM_ODE` for many experiments.

    Parameters
    ----------
    initial_concentrations : Sequence[float]
        Initial concentrations for each experiment.
    params_nucl : list | tuple
        Nucleation kinetics parameters.
    params_growth : list | tuple
        Growth kinetics parameters.
    t_save : array_like
        Time points at which the solution is desired.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_exp, len(t_save), 2)`` containing concentration and
        d43 for each experiment.
    """

    Nc = len(initial_concentrations)
    Nt = len(t_save)
    result = np.zeros((Nc, Nt, 2))

    for i, conc in enumerate(initial_concentrations):
        result[i, :, :] = solve_MoM_ODE(conc, params_nucl, params_growth, t_save)

    return result


def solve_MoM_ODE_icarr_conc(
    initial_concentrations: list[float],
    params_nucl: list[float] | tuple[float, float],
    params_growth: list[float] | tuple[float, float],
    t_save: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute only concentration profiles for many experiments.

    Parameters
    ----------
    initial_concentrations : Sequence[float]
        Initial concentrations for each run.
    params_nucl : list | tuple
        Nucleation kinetics parameters.
    params_growth : list | tuple
        Growth kinetics parameters.
    t_save : array_like
        Time points at which the solution is desired.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_exp, len(t_save), 1)`` with concentration profiles.
    """

    Nc = len(initial_concentrations)
    Nt = len(t_save)
    result = np.zeros((Nc, Nt, 1))

    for i, conc in enumerate(initial_concentrations):
        result[i, :, 0] = solve_MoM_ODE(conc, params_nucl, params_growth, t_save)[:, 0]

    return result


def solve_MoM_ODE_retsol(
    initial_concentration: float,
    params_nucl: list[float] | tuple[float, float],
    params_growth: list[float] | tuple[float, float],
    t_save: npt.NDArray[np.float64],
) -> spi.OdeResult:
    """Return the full SciPy solution of the MoM system for a single run."""

    ## Physical Parameters

    temperature = 273.15 + 20  # K
    rho_c = 1370.0
    conc_sat = 2.47
    kv = 0.81
    molecular_volume = 2.97e-26
    kb = 1.38064852e-23

    def CNTRate(S: float) -> float:
        return (
            60
            * np.exp(params_nucl[0])
            * S
            * np.exp(
                -16
                * np.pi
                * (params_nucl[1] * 1e-3) ** 3
                * molecular_volume**2
                / (3 * (kb * temperature) ** 3 * (np.log(S)) ** 2)
            )
        )

    def GrowthRate(S: float) -> float:
        return 1e-9 * params_growth[0] * (S - 1) ** params_growth[1]

    def ode_model(t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        moments = y[:-1]
        S = y[-1] / conc_sat

        growth = GrowthRate(S)

        nucleation = CNTRate(S)

        return np.array(
            [
                nucleation,
                growth * moments[0],
                2 * growth * moments[1],
                3 * growth * moments[2],
                4 * growth * moments[3],
                -3 * kv * rho_c * growth * moments[2],
            ]
        )

    t_0, t_end = t_save[0], t_save[-1]

    y_0 = np.concatenate([np.zeros(5), [initial_concentration]])

    return spi.solve_ivp(ode_model, (t_0, t_end), y_0, t_eval=t_save)


def solve_MoM_ODE_icarr_outputidx(
    initial_concentrations: list[float],
    params_nucl: list[float] | tuple[float, float],
    params_growth: list[float] | tuple[float, float],
    t_save: npt.NDArray[np.float64],
    save_idxs: list[int],
) -> npt.NDArray[np.float64]:
    """
    Solve the Method of Moments ODE system for multiple initial concentrations and return specified outputs.

    Args:
        initial_concentrations: Array of initial concentrations
        params_nucl: Tuple of nucleation parameters
        params_growth: Tuple of growth parameters
        t_save: Array of time points to save
        save_idxs: List of indices of outputs to save (0:conc, 1:d10, 2:d32, 3:d43, 4:mu0)

    Returns:
        NDArray of shape (n_concentrations, n_timepoints, n_outputs) containing the solutions
    """
    Nc = len(initial_concentrations)
    Nt = t_save.size  # Changed from len() to .size
    n_outputs = len(save_idxs)
    result = np.zeros((Nc, Nt, n_outputs))

    for i, conc in enumerate(initial_concentrations):
        solution = solve_MoM_ODE_retsol(conc, params_nucl, params_growth, t_save)

        concentration = solution.y[-1]
        mu0 = solution.y[0]
        d43 = (solution.y[4] + 1e-25) / (solution.y[3] + 1e-6) * 1e6
        d10 = (solution.y[1] + 1e-18) / (solution.y[0] + 1e-8) * 1e6
        d32 = (solution.y[3] + 1e-18) / (solution.y[2] + 1e-8) * 1e6

        all_outputs = np.vstack([concentration, d10, d32, d43, mu0])
        result[i, :, :] = all_outputs[save_idxs, :].T

    return result
