"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimisation.

Provides CMA-ES minimisation, a single CMA-ES update step, and BIPOP-CMA-ES
proxy for multimodal problems.

Optional dependencies: cma (Hansen's cma package), numpy (always required).
Falls back to a simplified mu/lambda ES.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import cma as _cma
    _CMA = True
except ImportError:
    _CMA = False


def cmaes_minimize(
    objective_fn_str: str = "sphere",
    x0: list[float] | None = None,
    sigma0: float = 0.3,
    max_iter: int = 500,
    bounds: list[list[float]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Minimise an objective using CMA-ES.

    Args:
        objective_fn_str: Name/description of objective ('sphere', 'rosenbrock',
            'rastrigin') or any string identifier.
        x0: Initial mean vector.  Defaults to random in [-2, 2]^2.
        sigma0: Initial step size.
        max_iter: Maximum number of generations.
        bounds: [[lb1, ub1], [lb2, ub2], ...] for each dimension.
        seed: Random seed.

    Returns:
        Dict with keys: x_best, f_best, n_evals, converged, backend, note.
    """
    np.random.seed(seed)
    if x0 is None:
        x0 = [1.5, -1.5]

    dim = len(x0)

    _FUNCS = {
        "sphere": lambda x: float(np.sum(np.array(x) ** 2)),
        "rosenbrock": lambda x: float(sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))),
        "rastrigin": lambda x: float(10 * len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)),
    }
    fn = _FUNCS.get(objective_fn_str, _FUNCS["sphere"])

    if _CMA:
        opts = {"maxiter": max_iter, "seed": seed, "verbose": -9}
        if bounds is not None:
            opts["bounds"] = [list(b) for b in zip(*bounds)]
        res = _cma.fmin(fn, x0, sigma0, opts)
        return {
            "backend": "cma", "note": "",
            "x_best": list(res[0]), "f_best": float(res[1]),
            "n_evals": int(res[4]), "converged": res[7] == 0,
        }

    # Simplified (mu, lambda) ES fallback
    mu = max(dim, 4)
    lam = 4 * mu
    mean = np.array(x0, dtype=float)
    sigma = sigma0
    f_best = fn(mean.tolist())
    x_best = mean.copy()
    n_evals = 0

    for _ in range(max_iter):
        samples = mean + sigma * np.random.randn(lam, dim)
        fits = [fn(s.tolist()) for s in samples]
        n_evals += lam
        idx = np.argsort(fits)[:mu]
        mean = samples[idx].mean(axis=0)
        sigma *= math.exp(0.2 * (np.std(fits) / (abs(np.mean(fits)) + 1e-12) - 0.3))
        sigma = max(sigma, 1e-8)
        if fits[idx[0]] < f_best:
            f_best = fits[idx[0]]
            x_best = samples[idx[0]].copy()

    return {
        "backend": "stub", "note": "Install cma for full CMA-ES with covariance adaptation.",
        "x_best": x_best.tolist(), "f_best": f_best,
        "n_evals": n_evals, "converged": f_best < 1e-6,
    }


def cmaes_step(
    mean: list[float],
    C: list[list[float]] | None = None,
    sigma: float = 0.3,
    p_sigma: list[float] | None = None,
    p_c: list[float] | None = None,
    weights: list[float] | None = None,
    mu_eff: float = 3.0,
) -> dict[str, Any]:
    """Perform a single CMA-ES update step (state update only, no evaluation).

    Args:
        mean: Current distribution mean.
        C: Covariance matrix (n x n).
        sigma: Current step size.
        p_sigma: Conjugate evolution path for sigma control.
        p_c: Conjugate evolution path for covariance.
        weights: Recombination weights.
        mu_eff: Effective mu (sum of squared weights inverse).

    Returns:
        Dict with keys: mean_new, sigma_new, C_new, backend, note.
    """
    n = len(mean)
    if C is None:
        C = np.eye(n).tolist()
    if p_sigma is None:
        p_sigma = [0.0] * n
    if p_c is None:
        p_c = [0.0] * n
    if weights is None:
        weights = [1.0 / max(int(mu_eff), 1)] * max(int(mu_eff), 1)

    C_arr = np.array(C, dtype=float)
    # Simplified Cholesky sample
    L = np.linalg.cholesky(C_arr + 1e-10 * np.eye(n))
    z = np.random.randn(n)
    sample = np.array(mean) + sigma * L @ z
    # Return updated mean as sample (proxy step)
    new_sigma = sigma * math.exp(0.1 * (np.linalg.norm(z) / math.sqrt(n) - 1.0))
    return {
        "backend": "numpy", "note": "Single-step proxy; install cma for full update.",
        "mean_new": sample.tolist(), "sigma_new": new_sigma, "C_new": C,
    }


def bipop_cmaes_proxy(
    objective_fn_str: str = "rastrigin",
    dim: int = 5,
    max_evals: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """BIPOP-CMA-ES for multimodal global optimisation (proxy).

    Two restart populations: one large (exploration) and one small (exploitation).

    Args:
        objective_fn_str: Objective function name ('sphere', 'rastrigin', etc.).
        dim: Problem dimensionality.
        max_evals: Maximum function evaluations budget.
        seed: Random seed.

    Returns:
        Dict with keys: x_best, f_best, n_restarts, backend, note.
    """
    n_restarts = max_evals // (100 * dim)
    best_results = []
    for restart in range(max(n_restarts, 2)):
        x0 = (np.random.randn(dim) * 2.0).tolist()
        budget = max_evals // max(n_restarts, 2)
        r = cmaes_minimize(objective_fn_str, x0=x0, sigma0=2.0 / (restart + 1),
                           max_iter=budget // (4 * dim), seed=seed + restart)
        best_results.append((r["f_best"], r["x_best"]))

    best_f, best_x = min(best_results, key=lambda t: t[0])
    return {
        "backend": "stub", "note": "Install cma for actual BIPOP-CMA-ES.",
        "x_best": best_x, "f_best": best_f, "n_restarts": max(n_restarts, 2),
    }
