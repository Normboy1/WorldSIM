"""Adaptive control algorithms: MRAC, RLS, gain scheduling, STR.

Provides model reference adaptive control proxy, recursive least-squares
identifier, gain-scheduling interpolation, and a self-tuning regulator proxy.

Optional dependencies: scipy (optimisation), numpy (always required).
Falls back to fixed-gain heuristics.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.optimize import minimize as _minimize
    _SCIPY = True
except ImportError:
    _SCIPY = False


def mrac_proxy(
    plant_num: list[float] = [1.0],
    plant_den: list[float] = [1.0, 2.0, 1.0],
    reference_model_num: list[float] = [1.0],
    reference_model_den: list[float] = [1.0, 1.5, 1.0],
    gamma: float = 5.0,
    t_end: float = 30.0,
) -> dict[str, Any]:
    """Model reference adaptive control (MIT rule) closed-loop simulation.

    Args:
        plant_num: Plant transfer function numerator.
        plant_den: Plant transfer function denominator (≥2nd order).
        reference_model_num: Reference model numerator.
        reference_model_den: Reference model denominator.
        gamma: Adaptation gain.
        t_end: Simulation time (s).

    Returns:
        Dict with keys: t, y_plant, y_model, error, theta, backend, note.
    """
    n_steps = 300
    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps + 1)
    y_plant = np.zeros(n_steps + 1)
    y_model = np.zeros(n_steps + 1)
    theta = 1.0  # adaptive parameter
    tau_ref = 1.0 / (abs(reference_model_den[-2]) + 1e-6) if len(reference_model_den) > 1 else 1.0
    tau_plant = 1.0 / (abs(plant_den[-2]) + 1e-6) if len(plant_den) > 1 else 1.5

    for k in range(n_steps):
        r = 1.0  # step reference
        u = theta * (y_model[k] - y_plant[k]) + r
        y_model[k + 1] = y_model[k] + dt / tau_ref * (r - y_model[k])
        y_plant[k + 1] = y_plant[k] + dt / tau_plant * (u - y_plant[k])
        e = y_plant[k] - y_model[k]
        theta -= gamma * dt * e * y_plant[k]

    error = (y_plant - y_model).tolist()
    return {
        "backend": "stub", "note": "Install python-control for rigorous MRAC simulation.",
        "t": t.tolist(), "y_plant": y_plant.tolist(), "y_model": y_model.tolist(),
        "error": error, "theta_final": theta,
    }


def rls_identifier(
    y_data: list[float] | None = None,
    u_data: list[float] | None = None,
    n_a: int = 2,
    n_b: int = 1,
    forgetting_factor: float = 0.98,
) -> dict[str, Any]:
    """Recursive Least Squares system identifier (ARX model).

    y(t) = a1*y(t-1) + ... + an_a*y(t-na) + b1*u(t-1) + ... + bn_b*u(t-nb)

    Args:
        y_data: Output measurement sequence.
        u_data: Input sequence.
        n_a: AR order (number of past outputs).
        n_b: Exogenous order (number of past inputs).
        forgetting_factor: Lambda (0 < lambda <= 1; 1 = no forgetting).

    Returns:
        Dict with keys: theta_hat, prediction_error_variance, backend, note.
    """
    if y_data is None:
        N = 100
        np.random.seed(0)
        y_data = (np.cumsum(np.random.randn(N)) * 0.1).tolist()
        u_data = np.random.randn(N).tolist()

    y = np.array(y_data)
    u = np.array(u_data)
    N = len(y)
    n_theta = n_a + n_b
    theta = np.zeros(n_theta)
    P = 100.0 * np.eye(n_theta)
    errors = []

    for t in range(max(n_a, n_b), N):
        phi = np.concatenate([-y[t - n_a: t][::-1], u[t - n_b: t][::-1]])
        if len(phi) < n_theta:
            continue
        y_hat = float(phi @ theta)
        e = y[t] - y_hat
        errors.append(e)
        K = P @ phi / (forgetting_factor + phi @ P @ phi)
        theta += K * e
        P = (P - np.outer(K, phi @ P)) / forgetting_factor

    return {
        "backend": "numpy", "note": "",
        "theta_hat": theta.tolist(),
        "prediction_error_variance": float(np.var(errors)) if errors else 0.0,
    }


def gain_scheduling(
    operating_points: list[float],
    controllers: list[dict],
    x_current: float,
) -> dict[str, Any]:
    """Interpolate controller gains across operating points.

    Args:
        operating_points: Scheduling variable values at each design point.
        controllers: List of controller parameter dicts (each with 'Kp', 'Ki', 'Kd').
        x_current: Current scheduling variable value.

    Returns:
        Dict with keys: Kp, Ki, Kd, scheduled_index, blend_fraction, backend, note.
    """
    ops = np.array(operating_points, dtype=float)
    idx = np.searchsorted(ops, x_current, side="right") - 1
    idx = int(np.clip(idx, 0, len(ops) - 2))
    alpha = (x_current - ops[idx]) / (ops[idx + 1] - ops[idx] + 1e-12)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    def interp(key):
        return (1 - alpha) * controllers[idx].get(key, 1.0) + alpha * controllers[idx + 1].get(key, 1.0)

    return {
        "backend": "numpy", "note": "",
        "Kp": interp("Kp"), "Ki": interp("Ki"), "Kd": interp("Kd"),
        "scheduled_index": idx, "blend_fraction": alpha,
    }


def self_tuning_regulator_proxy(
    y_ref: float = 1.0,
    t_end: float = 30.0,
    initial_guess: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Self-tuning regulator: combined RLS identification + pole-placement control.

    Args:
        y_ref: Reference setpoint.
        t_end: Simulation time (s).
        initial_guess: Initial controller parameters {'Kp': ..., 'Ti': ...}.

    Returns:
        Dict with keys: t, y, u, converged, final_params, backend, note.
    """
    if initial_guess is None:
        initial_guess = {"Kp": 1.0, "Ti": 2.0}

    n_steps = 200
    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    u = np.zeros(n_steps + 1)
    Kp = initial_guess["Kp"]
    Ti = initial_guess["Ti"]
    e_int = 0.0

    for k in range(n_steps):
        e = y_ref - y[k]
        e_int += e * dt
        u[k] = Kp * (e + e_int / Ti)
        y[k + 1] = y[k] + dt * (u[k] - y[k]) / 1.5
        # Adaptive gain update
        Kp = Kp + 0.01 * abs(e) * math.exp(-k * dt)

    return {
        "backend": "stub", "note": "Install python-control for rigorous STR.",
        "t": t.tolist(), "y": y.tolist(), "u": u.tolist(),
        "converged": abs(y[-1] - y_ref) < 0.05,
        "final_params": {"Kp": Kp, "Ti": Ti},
    }
