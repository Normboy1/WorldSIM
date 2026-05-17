"""Numerical calculus engine using SymPy + SciPy.

Provides numerical derivatives, integrals, root finding, and gradient computation.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import sympy as sp
from scipy import integrate as sci_integrate
from scipy import optimize as sci_optimize

from simlab.engines.math._sympy_parse import safe_sympify, sympy_locals_for_symbols


def _build_callable(f_str: str, var_name: str = "x") -> Callable[[float], float]:
    """Compile an expression string into a Python callable."""
    local_dict = sympy_locals_for_symbols([var_name])
    sym = local_dict[var_name]
    expr = safe_sympify(f_str, local_dict)
    # Lambdify with numpy for vectorised evaluation
    f = sp.lambdify(sym, expr, modules=["numpy"])
    return f


class CalculusEngine:
    """Numerical calculus operations backed by SymPy + SciPy."""

    def numerical_derivative(
        self, f_str: str, x_val: float, h: float = 1e-7, var_name: str = "x"
    ) -> float:
        """Central-difference numerical derivative of f at x_val.

        Parameters
        ----------
        f_str:  expression string
        x_val:  point at which to evaluate the derivative
        h:      step size for finite differences
        """
        f = _build_callable(f_str, var_name)
        return (f(x_val + h) - f(x_val - h)) / (2 * h)

    def numerical_integral(
        self,
        f_str: str,
        a: float,
        b: float,
        method: str = "quad",
        var_name: str = "x",
    ) -> dict:
        """Numerically integrate f from a to b.

        Parameters
        ----------
        method: 'quad' (adaptive), 'romberg', 'simpson', or 'trapezoid'
        """
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Integration limits a and b must be finite.")
        if a == b:
            return {"value": 0.0, "method": method, "note": "a == b"}

        f = _build_callable(f_str, var_name)

        if method == "quad":
            value, error = sci_integrate.quad(f, a, b)
            return {"value": float(value), "error_estimate": float(error), "method": "quad"}
        elif method == "romberg":
            value = sci_integrate.romberg(f, a, b)
            return {"value": float(value), "method": "romberg"}
        elif method == "simpson":
            x_arr = np.linspace(a, b, 1001)
            y_arr = np.array([f(xi) for xi in x_arr])
            value = sci_integrate.simpson(y_arr, x=x_arr)
            return {"value": float(value), "method": "simpson"}
        elif method == "trapezoid":
            x_arr = np.linspace(a, b, 1001)
            y_arr = np.array([f(xi) for xi in x_arr])
            value = sci_integrate.trapezoid(y_arr, x_arr)
            return {"value": float(value), "method": "trapezoid"}
        else:
            raise ValueError(f"Unknown integration method: {method!r}")

    def find_roots(
        self, f_str: str, x0_list: list[float], var_name: str = "x"
    ) -> list[float]:
        """Find roots of f near each starting point in x0_list.

        Uses scipy.optimize.brentq when bracket can be established,
        otherwise falls back to fsolve.
        """
        f = _build_callable(f_str, var_name)
        roots: list[float] = []
        seen: set[float] = set()

        for x0 in x0_list:
            try:
                root = float(sci_optimize.fsolve(f, x0, full_output=False)[0])
                # Deduplicate (within tolerance)
                if not any(abs(root - r) < 1e-8 for r in seen):
                    roots.append(root)
                    seen.add(root)
            except Exception:
                pass

        return roots

    def compute_gradient(
        self,
        f_str: str,
        variables: list[str],
        point: list[float],
        h: float = 1e-7,
    ) -> list[float]:
        """Compute the gradient of a multivariate function at *point*.

        Uses central differences for each partial derivative.
        """
        if len(variables) != len(point):
            raise ValueError("variables and point must have the same length.")
        if not variables:
            raise ValueError("variables must be a non-empty list.")

        local_dict = sympy_locals_for_symbols(variables)
        expr = safe_sympify(f_str, local_dict)
        syms = [local_dict[v] for v in variables]
        f_lam = sp.lambdify(syms, expr, modules=["numpy"])

        gradient: list[float] = []
        for i in range(len(variables)):
            p_plus = list(point)
            p_minus = list(point)
            p_plus[i] += h
            p_minus[i] -= h
            dfdxi = (f_lam(*p_plus) - f_lam(*p_minus)) / (2 * h)
            gradient.append(float(dfdxi))

        return gradient
