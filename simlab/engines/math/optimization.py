"""Optimization engine using SciPy.

Provides function minimization, maximization, curve fitting, and interval search.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy import optimize as sci_optimize

from simlab.engines.math._sympy_parse import (
    safe_sympify,
    sympy_locals_for_symbols,
    sympy_locals_with_var,
)


def _build_callable_multi(f_str: str, variable: str = "x"):
    """Build a scalar callable from an expression string.

    Returns (callable, sympy_symbol).
    """
    local_dict = sympy_locals_with_var(variable)
    sym = local_dict[variable]
    expr = safe_sympify(f_str, local_dict)
    f = sp.lambdify(sym, expr, modules=["numpy"])
    return f, sym


class OptimizationEngine:
    """SciPy-backed function optimization and curve fitting."""

    def minimize_function(
        self, f_str: str, x0: float | list[float], method: str = "BFGS", variable: str = "x"
    ) -> dict:
        """Minimize a scalar function.

        Parameters
        ----------
        f_str:  expression string
        x0:     initial guess (scalar or list for multivariate)
        method: scipy.optimize.minimize method
        """
        f, sym = _build_callable_multi(f_str, variable)
        x0_arr = np.atleast_1d(np.array(x0, dtype=float))

        if x0_arr.ndim == 1 and len(x0_arr) == 1:
            # Scalar optimization
            result = sci_optimize.minimize(
                lambda x: float(f(x[0])), x0_arr, method=method
            )
        else:
            # Multivariate — we'd need a multivariate expression; fallback
            result = sci_optimize.minimize(
                lambda x: float(f(x[0])), x0_arr[:1], method=method
            )

        return {
            "x_opt": result.x.tolist(),
            "f_opt": float(result.fun),
            "success": bool(result.success),
            "message": result.message,
            "n_iterations": int(result.nit),
            "n_evaluations": int(result.nfev),
            "method": method,
        }

    def maximize_function(
        self, f_str: str, x0: float | list[float], method: str = "BFGS", variable: str = "x"
    ) -> dict:
        """Maximize f by minimizing -f."""
        f, sym = _build_callable_multi(f_str, variable)
        x0_arr = np.atleast_1d(np.array(x0, dtype=float))

        result = sci_optimize.minimize(
            lambda x: -float(f(x[0])), x0_arr[:1], method=method
        )
        return {
            "x_opt": result.x.tolist(),
            "f_opt": float(-result.fun),
            "success": bool(result.success),
            "message": result.message,
            "n_iterations": int(result.nit),
            "n_evaluations": int(result.nfev),
            "method": method,
        }

    def fit_curve(
        self,
        x_data: list[float],
        y_data: list[float],
        model_str: str,
        p0: list[float] | None = None,
    ) -> dict:
        """Fit a parametric model to data using scipy.optimize.curve_fit.

        Parameters
        ----------
        model_str:
            Expression using variable 'x' and parameters p0, p1, p2, ...
            e.g. 'p0 * exp(-p1 * x) + p2'
        p0:
            Initial parameter guesses. If None, defaults to ones.
        """
        x_arr = np.array(x_data, dtype=float)
        y_arr = np.array(y_data, dtype=float)
        if len(x_arr) != len(y_arr):
            raise ValueError("x_data and y_data must have the same length.")
        if len(x_arr) < 2:
            raise ValueError("At least two data points are required for curve_fit.")

        # Detect parameter names in model_str (p0, p1, ...)
        import re
        param_names = sorted(set(re.findall(r"p\d+", model_str)))
        n_params = len(param_names)

        if n_params == 0:
            raise ValueError(
                "model_str must contain parameters named p0, p1, ... e.g. 'p0 * x + p1'"
            )

        if p0 is None:
            p0 = [1.0] * n_params
        if len(p0) != n_params:
            raise ValueError(f"p0 must have length {n_params} (one guess per parameter).")

        local_dict = sympy_locals_for_symbols(["x"] + param_names)
        x_sym = local_dict["x"]
        p_syms = [local_dict[pn] for pn in param_names]
        expr = safe_sympify(model_str, local_dict)
        f_lam = sp.lambdify([x_sym] + p_syms, expr, modules=["numpy"])

        def model_func(x, *params):
            return f_lam(x, *params)

        try:
            popt, pcov = sci_optimize.curve_fit(model_func, x_arr, y_arr, p0=p0, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            y_pred = model_func(x_arr, *popt)
            ss_res = np.sum((y_arr - y_pred) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0

            return {
                "parameters": {pn: float(v) for pn, v in zip(param_names, popt)},
                "parameter_errors": {pn: float(e) for pn, e in zip(param_names, perr)},
                "r_squared": r_squared,
                "residual_sum_squares": float(ss_res),
                "covariance": pcov.tolist(),
                "success": True,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def find_minimum_on_interval(self, f_str: str, a: float, b: float, variable: str = "x") -> dict:
        """Find the minimum of f on the interval [a, b] using bounded scalar minimization."""
        if not (float(a) < float(b)):
            raise ValueError("Interval requires a < b.")
        f, sym = _build_callable_multi(f_str, variable)
        result = sci_optimize.minimize_scalar(lambda x: float(f(x)), bounds=(a, b), method="bounded")
        return {
            "x_opt": float(result.x),
            "f_opt": float(result.fun),
            "success": bool(result.success),
            "interval": [a, b],
            "n_evaluations": int(result.nfev),
        }
