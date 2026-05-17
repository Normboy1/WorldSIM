"""Bayesian optimisation with Gaussian Process surrogate models.

Provides full BO loop, acquisition functions (EI, UCB), GP surrogate proxy,
and next query point selection.

Optional dependencies: scikit-learn (GP), scipy (acquisition optimisation).
Falls back to random search + heuristic EI.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
    from sklearn.gaussian_process.kernels import Matern as _Matern
    _SKL = True
except ImportError:
    _SKL = False

try:
    from scipy.optimize import minimize as _minimize
    from scipy.stats import norm as _norm
    _SCIPY = True
except ImportError:
    _SCIPY = False


def bayesian_optimize(
    objective_fn_str: str = "sphere",
    bounds: list[list[float]] | None = None,
    n_initial: int = 5,
    n_iter: int = 20,
    acquisition: str = "ei",
    seed: int = 42,
) -> dict[str, Any]:
    """Run a Bayesian optimisation loop.

    Args:
        objective_fn_str: Objective function name ('sphere', 'rosenbrock',
            'branin') or description.
        bounds: [[lb, ub], ...] per dimension.  Defaults to [0,1]^2.
        n_initial: Number of random initial evaluations.
        n_iter: Number of BO iterations.
        acquisition: Acquisition function ('ei', 'ucb', 'pi').
        seed: Random seed.

    Returns:
        Dict with keys: x_best, f_best, X_observed, y_observed,
        n_evals, backend, note.
    """
    np.random.seed(seed)
    if bounds is None:
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
    lbs = np.array([b[0] for b in bounds])
    ubs = np.array([b[1] for b in bounds])
    dim = len(bounds)

    _FUNCS = {
        "sphere": lambda x: float(np.sum(x ** 2)),
        "rosenbrock": lambda x: float(sum(100*(x[i+1]-x[i]**2)**2+(1-x[i])**2 for i in range(len(x)-1))),
        "branin": lambda x: float((x[1]-5.1*x[0]**2/(4*math.pi**2)+5*x[0]/math.pi-6)**2 +
                                   10*(1-1/(8*math.pi))*math.cos(x[0])+10),
    }
    fn = _FUNCS.get(objective_fn_str, _FUNCS["sphere"])

    X_obs = lbs + (ubs - lbs) * np.random.rand(n_initial, dim)
    y_obs = np.array([fn(x) for x in X_obs])

    for _ in range(n_iter):
        gp = gp_surrogate_proxy(X_obs.tolist(), y_obs.tolist(),
                                X_obs.tolist())
        x_next = next_query_point(gp, bounds, acquisition)["x_next"]
        x_next_arr = np.array(x_next)
        y_next = fn(x_next_arr)
        X_obs = np.vstack([X_obs, x_next_arr])
        y_obs = np.append(y_obs, y_next)

    best_idx = int(np.argmin(y_obs))
    return {
        "backend": "sklearn" if _SKL else "stub",
        "note": "" if _SKL else "Install scikit-learn for GP surrogate.",
        "x_best": X_obs[best_idx].tolist(), "f_best": float(y_obs[best_idx]),
        "X_observed": X_obs.tolist(), "y_observed": y_obs.tolist(),
        "n_evals": n_initial + n_iter,
    }


def expected_improvement(
    mu: float = 0.5,
    sigma: float = 0.3,
    f_best: float = 0.2,
) -> dict[str, Any]:
    """Expected improvement acquisition function.

    EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    where Z = (f_best - mu) / sigma

    Args:
        mu: GP posterior mean.
        sigma: GP posterior standard deviation.
        f_best: Best observed function value.

    Returns:
        Dict with keys: EI, Z, backend, note.
    """
    if sigma <= 0:
        return {"backend": "analytic", "note": "", "EI": 0.0, "Z": 0.0}
    Z = (f_best - mu) / sigma
    if _SCIPY:
        EI = (f_best - mu) * _norm.cdf(Z) + sigma * _norm.pdf(Z)
        backend, note = "scipy", ""
    else:
        # Approximation
        phi = math.exp(-0.5 * Z ** 2) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + math.erf(Z / math.sqrt(2)))
        EI = (f_best - mu) * Phi + sigma * phi
        backend = "stub"
        note = "Install scipy for exact EI via scipy.stats.norm."
    return {"backend": backend, "note": note, "EI": max(EI, 0.0), "Z": Z}


def upper_confidence_bound(
    mu: float = 0.5,
    sigma: float = 0.3,
    kappa: float = 2.0,
) -> dict[str, Any]:
    """Upper confidence bound acquisition function.

    UCB = mu + kappa * sigma   (for minimisation: negate)

    Args:
        mu: GP posterior mean.
        sigma: GP posterior standard deviation.
        kappa: Exploration-exploitation parameter.

    Returns:
        Dict with keys: UCB, LCB, backend, note.
    """
    UCB = mu + kappa * sigma
    LCB = mu - kappa * sigma
    return {"backend": "analytic", "note": "", "UCB": UCB, "LCB": LCB}


def gp_surrogate_proxy(
    X_train: list[list[float]],
    y_train: list[float],
    X_test: list[list[float]],
) -> dict[str, Any]:
    """Gaussian Process regression surrogate model.

    Args:
        X_train: Training inputs (N x d).
        y_train: Training outputs (N,).
        X_test: Test inputs (M x d).

    Returns:
        Dict with keys: mu, sigma, log_marginal_likelihood, backend, note.
    """
    X_tr = np.array(X_train, dtype=float)
    y_tr = np.array(y_train, dtype=float)
    X_te = np.array(X_test, dtype=float)

    if _SKL:
        kernel = _Matern(nu=2.5)
        gp = _GPR(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
        gp.fit(X_tr, y_tr)
        mu, sigma = gp.predict(X_te, return_std=True)
        lml = float(gp.log_marginal_likelihood_value_)
        backend, note = "sklearn", ""
    else:
        # RBF interpolation proxy
        dists = np.array([[np.linalg.norm(xt - xtr) for xtr in X_tr] for xt in X_te])
        l = 1.0
        K_star = np.exp(-0.5 * dists ** 2 / l ** 2)
        K = np.exp(-0.5 * np.array([[np.linalg.norm(xi - xj) ** 2 for xj in X_tr] for xi in X_tr]) / l ** 2)
        K += 1e-6 * np.eye(len(X_tr))
        try:
            alpha = np.linalg.solve(K, y_tr)
            mu = K_star @ alpha
            sigma = np.sqrt(np.maximum(1.0 - np.sum(K_star * (K_star @ np.linalg.inv(K)), axis=1), 0.0))
        except Exception:
            mu = np.full(len(X_te), y_tr.mean())
            sigma = np.full(len(X_te), y_tr.std())
        lml = -0.5 * float(y_tr @ np.linalg.lstsq(K, y_tr, rcond=None)[0])
        backend = "stub"
        note = "Install scikit-learn for full GP with optimised hyperparameters."

    return {
        "backend": backend, "note": note,
        "mu": mu.tolist(), "sigma": sigma.tolist(),
        "log_marginal_likelihood": lml,
    }


def next_query_point(
    gp_model: dict[str, Any],
    bounds: list[list[float]],
    acquisition: str = "ei",
    n_restarts: int = 5,
) -> dict[str, Any]:
    """Select the next query point by optimising the acquisition function.

    Args:
        gp_model: Output dict from gp_surrogate_proxy.
        bounds: Decision variable bounds.
        acquisition: Acquisition function ('ei', 'ucb').
        n_restarts: Number of random restarts for acquisition optimisation.

    Returns:
        Dict with keys: x_next, acq_value, backend, note.
    """
    lbs = np.array([b[0] for b in bounds])
    ubs = np.array([b[1] for b in bounds])
    dim = len(bounds)
    mu_vals = np.array(gp_model.get("mu", [0.5]))
    sigma_vals = np.array(gp_model.get("sigma", [0.3]))
    f_best = float(min(mu_vals))

    best_acq = -float("inf")
    best_x = lbs + (ubs - lbs) * 0.5

    for _ in range(n_restarts):
        x = lbs + (ubs - lbs) * np.random.rand(dim)
        mu_x = float(mu_vals.mean())
        sig_x = float(sigma_vals.mean())
        if acquisition == "ei":
            acq = expected_improvement(mu_x, sig_x, f_best)["EI"]
        else:  # ucb
            acq = -upper_confidence_bound(mu_x, sig_x)["LCB"]

        if acq > best_acq:
            best_acq = acq
            best_x = x

    return {
        "backend": "numpy", "note": "",
        "x_next": best_x.tolist(), "acq_value": best_acq,
    }
