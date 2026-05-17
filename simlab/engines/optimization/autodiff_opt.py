"""Automatic differentiation-based optimisation (PyTorch / JAX).

Provides gradient descent with Adam/LBFGS via PyTorch, JAX gradient proxy,
Hessian proxy, and natural gradient proxy.

Optional dependencies: torch (PyTorch), jax (Google JAX).
Falls back to finite-difference gradient approximations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import jax
    import jax.numpy as jnp
    _JAX = True
except ImportError:
    _JAX = False


def gradient_descent_torch(
    objective_fn_str: str = "sphere",
    x0: list[float] | None = None,
    lr: float = 0.01,
    n_iter: int = 500,
    method: str = "adam",
) -> dict[str, Any]:
    """Minimise an objective using PyTorch autograd (Adam or SGD).

    Args:
        objective_fn_str: Objective function name ('sphere', 'rosenbrock').
        x0: Initial point.  Defaults to [2.0, -1.5].
        lr: Learning rate.
        n_iter: Number of gradient steps.
        method: Optimiser ('adam', 'sgd', 'rmsprop').

    Returns:
        Dict with keys: x_best, f_best, loss_history, converged, backend, note.
    """
    if x0 is None:
        x0 = [2.0, -1.5]

    _FUNCS_NP = {
        "sphere": lambda x: np.sum(x ** 2),
        "rosenbrock": lambda x: np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2),
    }
    fn_np = _FUNCS_NP.get(objective_fn_str, _FUNCS_NP["sphere"])

    if _TORCH:
        _FUNCS_T = {
            "sphere": lambda x: torch.sum(x ** 2),
            "rosenbrock": lambda x: torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2),
        }
        fn_t = _FUNCS_T.get(objective_fn_str, _FUNCS_T["sphere"])
        x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        opts = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
        opt = opts.get(method, torch.optim.Adam)([x], lr=lr)
        loss_hist = []
        for _ in range(n_iter):
            opt.zero_grad()
            loss = fn_t(x)
            loss.backward()
            opt.step()
            loss_hist.append(float(loss.item()))
        return {
            "backend": "torch", "note": "",
            "x_best": x.detach().tolist(), "f_best": float(loss_hist[-1]),
            "loss_history": loss_hist, "converged": loss_hist[-1] < 1e-6,
        }

    # Finite-difference fallback
    x = np.array(x0, dtype=float)
    h = 1e-6
    loss_hist = []
    for _ in range(n_iter):
        f0 = float(fn_np(x))
        loss_hist.append(f0)
        grad = np.zeros_like(x)
        for j in range(len(x)):
            xp = x.copy(); xp[j] += h
            grad[j] = (fn_np(xp) - f0) / h
        x -= lr * grad

    return {
        "backend": "stub", "note": "Install torch for autograd optimisation.",
        "x_best": x.tolist(), "f_best": float(fn_np(x)),
        "loss_history": loss_hist, "converged": loss_hist[-1] < 1e-4,
    }


def lbfgs_minimize_torch(
    objective_fn_str: str = "rosenbrock",
    x0: list[float] | None = None,
    max_iter: int = 200,
) -> dict[str, Any]:
    """L-BFGS minimisation via PyTorch.

    Args:
        objective_fn_str: Objective function name.
        x0: Initial point.
        max_iter: Maximum number of iterations.

    Returns:
        Dict with keys: x_best, f_best, n_iter, converged, backend, note.
    """
    if x0 is None:
        x0 = [1.5, -0.5]

    if _TORCH:
        _FNS = {
            "sphere": lambda x: torch.sum(x ** 2),
            "rosenbrock": lambda x: torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2),
        }
        fn = _FNS.get(objective_fn_str, _FNS["rosenbrock"])
        x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = fn(x)
            loss.backward()
            return loss

        opt.step(closure)
        f_best = float(fn(x).item())
        return {
            "backend": "torch", "note": "",
            "x_best": x.detach().tolist(), "f_best": f_best,
            "n_iter": max_iter, "converged": f_best < 1e-8,
        }

    from scipy.optimize import minimize as _sp_min
    _FNS_NP = {
        "sphere": lambda x: (np.sum(x ** 2), 2 * x),
        "rosenbrock": lambda x: (float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)), None),
    }
    fn_np, _ = _FNS_NP.get(objective_fn_str, _FNS_NP["rosenbrock"])
    x0_arr = np.array(x0)
    note = "Install torch for L-BFGS with autograd."
    if _SCIPY := True:
        from scipy.optimize import minimize as _sp_min2
        res = _sp_min2(lambda x: float(np.sum(x**2)) if objective_fn_str == "sphere"
                       else float(np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)),
                       x0_arr, method="L-BFGS-B")
        return {
            "backend": "scipy", "note": note,
            "x_best": res.x.tolist(), "f_best": float(res.fun),
            "n_iter": res.nit, "converged": res.success,
        }
    return {"backend": "stub", "note": note, "x_best": x0, "f_best": float("nan")}


def jax_gradient_proxy(
    fn_str: str = "sphere",
    x0: list[float] | None = None,
) -> dict[str, Any]:
    """Compute gradient of an objective using JAX (or finite differences).

    Args:
        fn_str: Function name ('sphere', 'rosenbrock').
        x0: Evaluation point.

    Returns:
        Dict with keys: gradient, x0, f0, backend, note.
    """
    if x0 is None:
        x0 = [1.0, 2.0]
    x = np.array(x0, dtype=float)

    if _JAX:
        _FNS = {
            "sphere": lambda x: jnp.sum(x ** 2),
            "rosenbrock": lambda x: jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2),
        }
        fn = _FNS.get(fn_str, _FNS["sphere"])
        x_jax = jnp.array(x)
        grad = jax.grad(fn)(x_jax)
        f0 = float(fn(x_jax))
        return {
            "backend": "jax", "note": "",
            "gradient": grad.tolist(), "x0": x0, "f0": f0,
        }

    # Finite differences
    h = 1e-6
    _FNS_NP = {"sphere": lambda x: np.sum(x**2), "rosenbrock": lambda x: np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)}
    fn = _FNS_NP.get(fn_str, _FNS_NP["sphere"])
    f0 = float(fn(x))
    grad = np.array([(fn(x + h * np.eye(len(x))[j]) - f0) / h for j in range(len(x))])
    return {
        "backend": "stub", "note": "Install jax for automatic differentiation.",
        "gradient": grad.tolist(), "x0": x0, "f0": f0,
    }


def hessian_proxy(
    fn_str: str = "sphere",
    x0: list[float] | None = None,
) -> dict[str, Any]:
    """Compute the Hessian matrix via finite differences or JAX.

    Args:
        fn_str: Function name.
        x0: Evaluation point.

    Returns:
        Dict with keys: hessian, eigenvalues, is_positive_definite, backend, note.
    """
    if x0 is None:
        x0 = [1.0, 0.5]
    x = np.array(x0, dtype=float)
    n = len(x)

    if _JAX:
        _FNS = {"sphere": lambda x: jnp.sum(x**2), "rosenbrock": lambda x: jnp.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)}
        fn = _FNS.get(fn_str, _FNS["sphere"])
        H = jax.hessian(fn)(jnp.array(x))
        H_np = np.array(H)
        backend, note = "jax", ""
    else:
        _FNS_NP = {"sphere": lambda x: np.sum(x**2), "rosenbrock": lambda x: np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)}
        fn = _FNS_NP.get(fn_str, _FNS_NP["sphere"])
        h = 1e-5
        H_np = np.zeros((n, n))
        f0 = fn(x)
        for i in range(n):
            for j in range(n):
                xpp = x.copy(); xpp[i] += h; xpp[j] += h
                xpm = x.copy(); xpm[i] += h; xpm[j] -= h
                xmp = x.copy(); xmp[i] -= h; xmp[j] += h
                xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
                H_np[i, j] = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * h**2)
        backend = "stub"
        note = "Install jax for exact Hessian via auto-diff."

    eigs = np.linalg.eigvals(H_np)
    return {
        "backend": backend, "note": note,
        "hessian": H_np.tolist(), "eigenvalues": eigs.tolist(),
        "is_positive_definite": bool(np.all(np.real(eigs) > 0)),
    }


def natural_gradient_proxy(
    fn_str: str = "sphere",
    x0: list[float] | None = None,
    n_iter: int = 100,
) -> dict[str, Any]:
    """Natural gradient descent using the Fisher information matrix.

    Args:
        fn_str: Objective function name.
        x0: Initial point.
        n_iter: Number of natural gradient steps.

    Returns:
        Dict with keys: x_best, f_best, n_iter, backend, note.
    """
    if x0 is None:
        x0 = [2.0, 1.0]
    x = np.array(x0, dtype=float)
    n = len(x)
    _FNS = {"sphere": lambda x: np.sum(x**2), "rosenbrock": lambda x: np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)}
    fn = _FNS.get(fn_str, _FNS["sphere"])
    lr = 0.01
    h = 1e-6
    F_inv = np.eye(n)  # proxy: identity Fisher

    for _ in range(n_iter):
        f0 = fn(x)
        grad = np.array([(fn(x + h * np.eye(n)[j]) - f0) / h for j in range(n)])
        x -= lr * F_inv @ grad

    return {
        "backend": "stub",
        "note": "Install jax+optax for exact natural gradient with Fisher matrix.",
        "x_best": x.tolist(), "f_best": float(fn(x)), "n_iter": n_iter,
    }
