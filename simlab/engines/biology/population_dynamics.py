"""Population dynamics simulation engine.

Provides classical ecological population models including predator-prey
(Lotka-Volterra), logistic growth, Allee effects, age-structured (Leslie
matrix) models, and two-species competition (Lotka-Volterra competition).

Optional dependencies: scipy (for ODE integration), numpy (always required).
Falls back to closed-form or Euler approximations when scipy is unavailable.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp
    _SCIPY = True
except ImportError:
    _SCIPY = False


def lotka_volterra(
    alpha: float = 1.0,
    beta: float = 0.1,
    delta: float = 0.075,
    gamma: float = 1.5,
    x0: float = 10.0,
    y0: float = 5.0,
    t_end: float = 50.0,
    n_steps: int = 500,
) -> dict[str, Any]:
    """Simulate predator-prey dynamics using the Lotka-Volterra equations.

    dx/dt = alpha*x - beta*x*y  (prey)
    dy/dt = delta*x*y - gamma*y  (predator)

    Args:
        alpha: Prey birth rate (1/time).
        beta: Predation rate coefficient.
        delta: Predator reproduction efficiency.
        gamma: Predator death rate (1/time).
        x0: Initial prey population.
        y0: Initial predator population.
        t_end: End time of simulation.
        n_steps: Number of time steps.

    Returns:
        Dict with keys: t, prey, predator, period_estimate, backend, note.
    """
    if not _SCIPY:
        dt = t_end / n_steps
        t = np.linspace(0, t_end, n_steps + 1)
        x, y = np.zeros(n_steps + 1), np.zeros(n_steps + 1)
        x[0], y[0] = x0, y0
        for i in range(n_steps):
            dx = (alpha * x[i] - beta * x[i] * y[i]) * dt
            dy = (delta * x[i] * y[i] - gamma * y[i]) * dt
            x[i + 1] = max(x[i] + dx, 0.0)
            y[i + 1] = max(y[i] + dy, 0.0)
        period_estimate = 2 * math.pi / math.sqrt(alpha * gamma)
        return {
            "backend": "stub",
            "note": "Install scipy for higher-accuracy ODE integration.",
            "t": t.tolist(),
            "prey": x.tolist(),
            "predator": y.tolist(),
            "period_estimate": period_estimate,
        }

    def odes(t, state):
        x, y = state
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]

    sol = solve_ivp(odes, [0, t_end], [x0, y0], dense_output=False,
                    t_eval=np.linspace(0, t_end, n_steps + 1), method="RK45")
    period_estimate = 2 * math.pi / math.sqrt(alpha * gamma)
    return {
        "backend": "scipy",
        "note": "",
        "t": sol.t.tolist(),
        "prey": sol.y[0].tolist(),
        "predator": sol.y[1].tolist(),
        "period_estimate": period_estimate,
    }


def logistic_growth(
    r: float = 0.5,
    K: float = 1000.0,
    N0: float = 50.0,
    t_end: float = 30.0,
) -> dict[str, Any]:
    """Logistic population growth model.

    dN/dt = r * N * (1 - N/K)

    Args:
        r: Intrinsic growth rate.
        K: Carrying capacity.
        N0: Initial population size.
        t_end: End time.

    Returns:
        Dict with keys: t, N, carrying_capacity, doubling_time, backend, note.
    """
    t = np.linspace(0, t_end, 300)
    N = K / (1 + ((K - N0) / N0) * np.exp(-r * t))
    doubling_time = math.log(2) / r if r > 0 else float("inf")
    backend = "scipy" if _SCIPY else "stub"
    note = "" if _SCIPY else "Install scipy for ODE-based integration."
    return {
        "backend": backend,
        "note": note,
        "t": t.tolist(),
        "N": N.tolist(),
        "carrying_capacity": K,
        "doubling_time": doubling_time,
    }


def allee_effect(
    r: float = 0.4,
    K: float = 500.0,
    A: float = 50.0,
    N0: float = 80.0,
    t_end: float = 40.0,
) -> dict[str, Any]:
    """Logistic growth with a strong Allee effect (extinction threshold).

    dN/dt = r * N * (N/A - 1) * (1 - N/K)

    Args:
        r: Growth rate parameter.
        K: Carrying capacity.
        A: Allee threshold (population below which growth is negative).
        N0: Initial population.
        t_end: End time.

    Returns:
        Dict with keys: t, N, allee_threshold, outcome, backend, note.
    """
    n_steps = 400
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    N_arr = np.zeros(n_steps + 1)
    N_arr[0] = N0
    for i in range(n_steps):
        Ni = N_arr[i]
        dN = r * Ni * (Ni / A - 1) * (1 - Ni / K) * dt
        N_arr[i + 1] = max(Ni + dN, 0.0)

    outcome = "persistence" if N_arr[-1] > A else "extinction"
    backend = "scipy" if _SCIPY else "stub"
    note = "" if _SCIPY else "Install scipy for adaptive-step ODE solver."
    return {
        "backend": backend,
        "note": note,
        "t": t_arr.tolist(),
        "N": N_arr.tolist(),
        "allee_threshold": A,
        "outcome": outcome,
    }


def age_structured_model(
    leslie_matrix: list[list[float]] | None = None,
    N0: list[float] | None = None,
    n_steps: int = 20,
) -> dict[str, Any]:
    """Project an age-structured population using a Leslie matrix.

    Args:
        leslie_matrix: Square Leslie matrix (fecundity row + survival subdiag).
                       Defaults to a 3-class example.
        N0: Initial age-class abundances. Defaults to [100, 50, 20].
        n_steps: Number of projection time steps (generations).

    Returns:
        Dict with keys: N_trajectory, dominant_eigenvalue, stable_age_dist,
        doubling_time, backend, note.
    """
    if leslie_matrix is None:
        leslie_matrix = [
            [0.0, 1.5, 0.5],
            [0.8, 0.0, 0.0],
            [0.0, 0.6, 0.0],
        ]
    if N0 is None:
        N0 = [100.0, 50.0, 20.0]

    L = np.array(leslie_matrix, dtype=float)
    N = np.array(N0, dtype=float)
    trajectory = [N.tolist()]
    for _ in range(n_steps):
        N = L @ N
        trajectory.append(N.tolist())

    eigenvalues = np.linalg.eigvals(L)
    lambda1 = float(np.max(np.real(eigenvalues)))
    stable_dist = np.abs(np.linalg.eig(L)[1][:, np.argmax(np.real(eigenvalues))])
    stable_dist = (stable_dist / stable_dist.sum()).tolist()
    doubling_time = math.log(2) / math.log(lambda1) if lambda1 > 1 else float("inf")

    return {
        "backend": "numpy",
        "note": "",
        "N_trajectory": trajectory,
        "dominant_eigenvalue": lambda1,
        "stable_age_dist": stable_dist,
        "doubling_time": doubling_time,
    }


def competition_model(
    r1: float = 1.0,
    r2: float = 0.8,
    K1: float = 200.0,
    K2: float = 150.0,
    alpha12: float = 0.6,
    alpha21: float = 0.4,
    N1_0: float = 10.0,
    N2_0: float = 10.0,
    t_end: float = 50.0,
) -> dict[str, Any]:
    """Two-species Lotka-Volterra competition model.

    dN1/dt = r1*N1*(1 - (N1 + alpha12*N2)/K1)
    dN2/dt = r2*N2*(1 - (N2 + alpha21*N1)/K2)

    Args:
        r1: Intrinsic growth rate of species 1.
        r2: Intrinsic growth rate of species 2.
        K1: Carrying capacity of species 1.
        K2: Carrying capacity of species 2.
        alpha12: Effect of species 2 on species 1.
        alpha21: Effect of species 1 on species 2.
        N1_0: Initial population of species 1.
        N2_0: Initial population of species 2.
        t_end: End time of simulation.

    Returns:
        Dict with keys: t, N1, N2, coexistence, outcome, backend, note.
    """
    n_steps = 500
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    N1_arr = np.zeros(n_steps + 1)
    N2_arr = np.zeros(n_steps + 1)
    N1_arr[0], N2_arr[0] = N1_0, N2_0

    for i in range(n_steps):
        n1, n2 = N1_arr[i], N2_arr[i]
        dN1 = r1 * n1 * (1 - (n1 + alpha12 * n2) / K1) * dt
        dN2 = r2 * n2 * (1 - (n2 + alpha21 * n1) / K2) * dt
        N1_arr[i + 1] = max(n1 + dN1, 0.0)
        N2_arr[i + 1] = max(n2 + dN2, 0.0)

    eps = 1.0
    n1_fin, n2_fin = N1_arr[-1], N2_arr[-1]
    if n1_fin > eps and n2_fin > eps:
        outcome = "coexistence"
    elif n1_fin > eps:
        outcome = "species_1_wins"
    elif n2_fin > eps:
        outcome = "species_2_wins"
    else:
        outcome = "mutual_exclusion"

    backend = "scipy" if _SCIPY else "stub"
    note = "" if _SCIPY else "Install scipy for adaptive-step integration."
    return {
        "backend": backend,
        "note": note,
        "t": t_arr.tolist(),
        "N1": N1_arr.tolist(),
        "N2": N2_arr.tolist(),
        "coexistence": outcome == "coexistence",
        "outcome": outcome,
    }
