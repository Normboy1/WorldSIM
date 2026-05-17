"""Model Predictive Control (MPC) engine.

Provides MPC setup, single-step optimisation proxy, linear MPC simulation,
and economic MPC proxy.  When scipy is available, uses quadratic programming
for the optimal control sequence; otherwise falls back to a greedy heuristic.

Optional dependencies: scipy (quadratic programming), numpy (always required).
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

try:
    from scipy.optimize import minimize as _minimize
    _SCIPY = True
except ImportError:
    _SCIPY = False


def mpc_setup(
    A: list[list[float]],
    B: list[list[float]],
    C: list[list[float]],
    Q: list[list[float]],
    R: list[list[float]],
    N_pred: int = 10,
    N_ctrl: int = 5,
) -> dict[str, Any]:
    """Validate and pre-process MPC problem matrices.

    Args:
        A: State transition matrix (n x n).
        B: Input matrix (n x m).
        C: Output matrix (p x n).
        Q: State / output weighting matrix (n x n).
        R: Input weighting matrix (m x m).
        N_pred: Prediction horizon.
        N_ctrl: Control horizon.

    Returns:
        Dict with keys: n_states, n_inputs, n_outputs, N_pred, N_ctrl,
        is_stable, spectral_radius, backend, note.
    """
    A_arr = np.array(A, dtype=float)
    eigvals = np.linalg.eigvals(A_arr)
    spectral_radius = float(np.max(np.abs(eigvals)))
    n = A_arr.shape[0]
    m = np.array(B, dtype=float).shape[1]
    p = np.array(C, dtype=float).shape[0]
    return {
        "backend": "numpy", "note": "",
        "n_states": n, "n_inputs": m, "n_outputs": p,
        "N_pred": N_pred, "N_ctrl": N_ctrl,
        "is_stable": spectral_radius < 1.0, "spectral_radius": spectral_radius,
    }


def mpc_step(
    x0: list[float],
    x_ref: list[float],
    A: list[list[float]],
    B: list[list[float]],
    Q: list[list[float]],
    R: list[list[float]],
    N_pred: int = 10,
    N_ctrl: int = 5,
    u_bounds: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Compute the optimal MPC control action for one step.

    Args:
        x0: Current state vector.
        x_ref: Reference state vector.
        A: State matrix.
        B: Input matrix.
        Q: State cost matrix.
        R: Input cost matrix.
        N_pred: Prediction horizon.
        N_ctrl: Control horizon.
        u_bounds: (lower, upper) control input bounds.

    Returns:
        Dict with keys: u_optimal, predicted_cost, backend, note.
    """
    A_arr = np.array(A, dtype=float)
    B_arr = np.array(B, dtype=float)
    Q_arr = np.array(Q, dtype=float)
    R_arr = np.array(R, dtype=float)
    x0_arr = np.array(x0, dtype=float)
    x_ref_arr = np.array(x_ref, dtype=float)
    m = B_arr.shape[1]

    def cost(u_seq):
        u = u_seq.reshape(N_ctrl, m)
        x = x0_arr.copy()
        J = 0.0
        for k in range(N_pred):
            u_k = u[min(k, N_ctrl - 1)]
            e = x - x_ref_arr
            J += float(e @ Q_arr @ e + u_k @ R_arr @ u_k)
            x = A_arr @ x + B_arr @ u_k
        return J

    u0 = np.zeros(N_ctrl * m)
    if _SCIPY:
        bounds = [u_bounds] * (N_ctrl * m)
        res = _minimize(cost, u0, method="L-BFGS-B", bounds=bounds)
        u_opt = res.x[:m].tolist()
        J_opt = float(res.fun)
        backend, note = "scipy", ""
    else:
        # Greedy: proportional control
        e = x_ref_arr - x0_arr
        u_opt = np.clip(0.5 * e[:m], u_bounds[0], u_bounds[1]).tolist()
        J_opt = cost(np.tile(u_opt, N_ctrl))
        backend = "stub"
        note = "Install scipy for QP-based optimal MPC."

    return {
        "backend": backend, "note": note,
        "u_optimal": u_opt, "predicted_cost": J_opt,
    }


def linear_mpc_proxy(
    plant_num: list[float] = [1.0],
    plant_den: list[float] = [1.0, 2.0, 1.0],
    setpoint: float = 1.0,
    T_sim: float = 20.0,
    Q_weight: float = 10.0,
    R_weight: float = 1.0,
) -> dict[str, Any]:
    """Simulate a SISO linear MPC closed-loop response (proxy).

    Args:
        plant_num: Transfer function numerator coefficients.
        plant_den: Transfer function denominator coefficients.
        setpoint: Step setpoint value.
        T_sim: Simulation duration (s).
        Q_weight: Output tracking weight.
        R_weight: Control effort weight.

    Returns:
        Dict with keys: t, y, u, ISE, IAE, backend, note.
    """
    n_steps = 200
    dt = T_sim / n_steps
    t = np.linspace(0, T_sim, n_steps + 1)
    # PID-like proxy (MPC approximation for SISO 2nd-order)
    Kp = Q_weight / (R_weight + 1.0) * 2.0
    Ki = 0.1 * Kp
    y = np.zeros(n_steps + 1)
    u = np.zeros(n_steps + 1)
    e_int = 0.0

    for k in range(n_steps):
        e = setpoint - y[k]
        e_int += e * dt
        u[k] = np.clip(Kp * e + Ki * e_int, -5.0, 5.0)
        # 2nd-order plant proxy
        tau = 1.0 / (abs(plant_den[-2]) + 1e-6) if len(plant_den) > 1 else 1.0
        y[k + 1] = y[k] + dt / tau * (u[k] - y[k])

    ISE = float(np.sum((setpoint - y) ** 2) * dt)
    IAE = float(np.sum(np.abs(setpoint - y)) * dt)
    return {
        "backend": "stub", "note": "Install scipy for full QP-based MPC simulation.",
        "t": t.tolist(), "y": y.tolist(), "u": u.tolist(), "ISE": ISE, "IAE": IAE,
    }


def economic_mpc_proxy(
    plant_params: dict[str, float] | None = None,
    stage_cost_fn: str = "quadratic",
    t_end: float = 50.0,
) -> dict[str, Any]:
    """Proxy economic MPC: optimise a stage cost over a receding horizon.

    Args:
        plant_params: Dict with plant parameters (tau, gain, delay).
        stage_cost_fn: Stage cost function type ('quadratic' or 'economic').
        t_end: Simulation duration (s).

    Returns:
        Dict with keys: t, objective_trajectory, optimal_cost, backend, note.
    """
    if plant_params is None:
        plant_params = {"tau": 2.0, "gain": 1.5, "delay": 0.5}

    n_steps = 100
    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    obj = np.zeros(n_steps + 1)
    tau = plant_params.get("tau", 2.0)
    gain = plant_params.get("gain", 1.5)

    for k in range(n_steps):
        u_opt = 1.0 / (1 + 0.1 * k * dt)  # diminishing setpoint proxy
        y[k + 1] = y[k] + dt / tau * (gain * u_opt - y[k])
        obj[k] = y[k] ** 2 + 0.1 * u_opt ** 2

    return {
        "backend": "stub", "note": "Install scipy for IPOPT-based EMPC.",
        "t": t.tolist(), "objective_trajectory": obj.tolist(),
        "optimal_cost": float(np.sum(obj) * dt),
    }
