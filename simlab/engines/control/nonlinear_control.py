"""Nonlinear control: sliding mode, feedback linearisation, Lyapunov, backstepping.

Provides sliding mode control simulation, feedback linearisation proxy,
Lyapunov stability check, relative degree estimation, and backstepping proxy.

Optional dependencies: scipy (ODE integration), numpy (always required).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp as _solve_ivp
    from scipy.linalg import eigvals as _eigvals
    _SCIPY = True
except ImportError:
    _SCIPY = False


def sliding_mode_control(
    x0: list[float] = [2.0, 0.0],
    x_ref: list[float] = [0.0, 0.0],
    sigma_fn: str = "sigma = x[0] + x[1]",
    u_max: float = 5.0,
    t_end: float = 10.0,
    n_steps: int = 500,
) -> dict[str, Any]:
    """Simulate sliding mode control for a 2nd-order nonlinear system.

    System: x_dot[0] = x[1],  x_dot[1] = u + disturbance
    Surface: sigma = c * x[0] + x[1]

    Args:
        x0: Initial state [position, velocity].
        x_ref: Reference state.
        sigma_fn: Sliding surface formula (informational string).
        u_max: Control saturation limit.
        t_end: Simulation time (s).
        n_steps: Number of integration steps.

    Returns:
        Dict with keys: t, x0_traj, x1_traj, sigma_traj, u_traj, backend, note.
    """
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    x = np.array(x0, dtype=float)
    x_r = np.array(x_ref, dtype=float)
    c = 1.0  # sliding surface gain
    eta = 0.5  # reaching law gain
    x0_hist, x1_hist, sigma_hist, u_hist = [x[0]], [x[1]], [], []

    for _ in range(n_steps):
        e = x - x_r
        sigma = c * e[0] + e[1]
        sigma_hist.append(sigma)
        u = float(np.clip(-eta * np.sign(sigma) - c * x[1], -u_max, u_max))
        u_hist.append(u)
        x[0] += x[1] * dt
        x[1] += (u + 0.05 * np.random.randn()) * dt
        x0_hist.append(x[0])
        x1_hist.append(x[1])

    sigma_hist.append(c * (x[0] - x_r[0]) + (x[1] - x_r[1]))

    return {
        "backend": "stub" if not _SCIPY else "scipy",
        "note": "" if _SCIPY else "Install scipy for continuous-time SMC integration.",
        "t": t_arr.tolist(), "x0_traj": x0_hist, "x1_traj": x1_hist,
        "sigma_traj": sigma_hist, "u_traj": u_hist,
    }


def feedback_linearization_proxy(
    f_fn_str: str = "x[1]",
    g_fn_str: str = "1.0",
    x0: list[float] = [2.0, 0.5],
    x_ref: list[float] = [0.0, 0.0],
    t_end: float = 10.0,
) -> dict[str, Any]:
    """Feedback linearisation proxy for a control-affine system.

    x_dot = f(x) + g(x) * u

    Args:
        f_fn_str: String description of the f function (informational).
        g_fn_str: String description of the g function (informational).
        x0: Initial state.
        x_ref: Reference state.
        t_end: Simulation time (s).

    Returns:
        Dict with keys: t, x_traj, converged, backend, note.
    """
    n_steps = 300
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    x = np.array(x0, dtype=float)
    x_r = np.array(x_ref, dtype=float)
    x_hist = [x.copy().tolist()]
    Kp, Kd = 4.0, 4.0

    for _ in range(n_steps):
        e = x_r - x
        # PD controller in linearised coordinates
        v = Kp * e[0] + (Kd * e[1] if len(e) > 1 else 0)
        u = v  # proxy: g(x) = 1
        if len(x) >= 2:
            x[0] += x[1] * dt
            x[1] += (u - 0.5 * x[1]) * dt
        else:
            x[0] += u * dt
        x_hist.append(x.copy().tolist())

    converged = bool(np.linalg.norm(x - x_r) < 0.1)
    return {
        "backend": "stub", "note": "Install scipy for exact nonlinear ODE integration.",
        "t": t_arr.tolist(), "x_traj": x_hist, "converged": converged,
    }


def lyapunov_stability_check(A_matrix: list[list[float]]) -> dict[str, Any]:
    """Check Lyapunov (asymptotic) stability of a linear system x_dot = A*x.

    Stable iff all eigenvalues have negative real parts.

    Args:
        A_matrix: System matrix (n x n).

    Returns:
        Dict with keys: is_stable, eigenvalues_real, eigenvalues_imag,
        max_real_part, backend, note.
    """
    A = np.array(A_matrix, dtype=float)
    eigs = np.linalg.eigvals(A)
    max_re = float(np.max(np.real(eigs)))
    return {
        "backend": "numpy", "note": "",
        "is_stable": max_re < 0, "max_real_part": max_re,
        "eigenvalues_real": np.real(eigs).tolist(),
        "eigenvalues_imag": np.imag(eigs).tolist(),
    }


def input_output_linearization_degree(
    n_states: int = 4,
    relative_degree: int = 2,
) -> dict[str, Any]:
    """Determine system order and zero dynamics dimension.

    Args:
        n_states: Number of state variables.
        relative_degree: Relative degree of the output (number of times to
            differentiate to make input appear explicitly).

    Returns:
        Dict with keys: relative_degree, zero_dynamics_dim, fully_linearizable,
        backend, note.
    """
    zero_dim = n_states - relative_degree
    fully_lin = zero_dim == 0
    return {
        "backend": "analytic", "note": "",
        "relative_degree": relative_degree,
        "zero_dynamics_dim": zero_dim,
        "fully_linearizable": fully_lin,
    }


def backstepping_proxy(
    system_params: dict[str, float] | None = None,
    x0: list[float] = [2.0, 1.0],
    x_ref: list[float] = [0.0, 0.0],
    t_end: float = 15.0,
) -> dict[str, Any]:
    """Backstepping control proxy for a strict-feedback system.

    x_dot[0] = x[1] + f1(x)
    x_dot[1] = u + f2(x)

    Args:
        system_params: Dict with system function parameters.
        x0: Initial state.
        x_ref: Reference state.
        t_end: Simulation time (s).

    Returns:
        Dict with keys: t, x_traj, u_traj, converged, backend, note.
    """
    if system_params is None:
        system_params = {"k1": 2.0, "k2": 3.0}

    k1 = system_params.get("k1", 2.0)
    k2 = system_params.get("k2", 3.0)
    n_steps = 400
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    x = np.array(x0, dtype=float)
    x_r = np.array(x_ref, dtype=float)
    x_hist = [x.copy().tolist()]
    u_hist = []

    for _ in range(n_steps):
        e1 = x[0] - x_r[0]
        # Virtual control: alpha_1 = -k1 * e1
        alpha1 = -k1 * e1
        e2 = x[1] - alpha1
        u = -k2 * e2 - e1
        u_hist.append(float(u))
        x[0] += x[1] * dt
        x[1] += u * dt
        x_hist.append(x.copy().tolist())

    converged = bool(np.linalg.norm(x - x_r) < 0.1)
    return {
        "backend": "stub", "note": "Install scipy for exact backstepping integration.",
        "t": t_arr.tolist(), "x_traj": x_hist, "u_traj": u_hist,
        "converged": converged,
    }
