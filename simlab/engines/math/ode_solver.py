"""ODE/PDE solver engine using SciPy.

Supports single ODEs, systems of ODEs, and second-order linear ODEs.
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from simlab.engines.math._sympy_parse import safe_sympify, sympy_locals_with_var


def _build_ode_func(f_str: str, var_t: str = "t", var_y: str = "y"):
    """Compile a single-ODE expression string into a callable f(t, y).

    The expression should be written in terms of `t` and `y`,
    representing dy/dt = f(t, y).
    """
    t_sym = sp.Symbol(var_t)
    y_sym = sp.Symbol(var_y)
    local_dict = sympy_locals_with_var(var_t)
    local_dict[var_y] = y_sym
    expr = safe_sympify(f_str, local_dict)
    f_lam = sp.lambdify((t_sym, y_sym), expr, modules=["numpy"])

    def ode_func(t, y_arr):
        return [f_lam(t, y_arr[0])]

    return ode_func


def _validate_t_span(t_span: list[float], t_eval_points: int) -> tuple[float, float]:
    if len(t_span) != 2:
        raise ValueError("t_span must be a two-element list [t_start, t_end].")
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not (math.isfinite(t0) and math.isfinite(t1)):
        raise ValueError("t_span endpoints must be finite numbers.")
    if t1 <= t0:
        raise ValueError("t_span must satisfy t_end > t_start.")
    if int(t_eval_points) < 2:
        raise ValueError("t_eval_points must be at least 2.")
    return t0, t1


class ODESolver:
    """SciPy-backed ODE solver."""

    def solve_ode(
        self,
        f_str: str,
        y0: float | list[float],
        t_span: list[float],
        t_eval_points: int = 100,
        method: str = "RK45",
    ) -> dict:
        """Solve a first-order ODE dy/dt = f(t, y).

        Parameters
        ----------
        f_str:
            Expression for dy/dt in terms of t and y.
        y0:
            Initial condition (scalar or list for single equation).
        t_span:
            [t_start, t_end]
        t_eval_points:
            Number of time points for output.
        method:
            SciPy integrator: RK45, RK23, DOP853, Radau, BDF, LSODA.
        """
        if isinstance(y0, (int, float)):
            y0 = [float(y0)]

        _validate_t_span(t_span, t_eval_points)

        ode_func = _build_ode_func(f_str)
        t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)

        sol = solve_ivp(
            ode_func,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            dense_output=False,
            rtol=1e-8,
            atol=1e-10,
        )

        return {
            "t": sol.t.tolist(),
            "y": sol.y[0].tolist(),
            "success": bool(sol.success),
            "message": sol.message,
            "method": method,
            "nfev": int(sol.nfev),
            "t_span": t_span,
            "y0": y0,
        }

    def solve_system(
        self,
        equations: list[str],
        y0: list[float],
        t_span: list[float],
        t_eval_points: int = 200,
        method: str = "RK45",
        var_names: list[str] | None = None,
    ) -> dict:
        """Solve a system of first-order ODEs dy_i/dt = f_i(t, y_0, ..., y_n).

        Parameters
        ----------
        equations:
            List of expression strings for each dy_i/dt.
            Variables: t, y0, y1, y2, ...  (or use var_names).
        y0:
            Initial conditions for each variable.
        var_names:
            Names for the y variables (default: y0, y1, ...).
        """
        n = len(equations)
        if n == 0:
            raise ValueError("equations list must not be empty.")
        _validate_t_span(t_span, t_eval_points)

        if var_names is None:
            var_names = [f"y{i}" for i in range(n)]

        if len(y0) != n:
            raise ValueError(f"y0 length ({len(y0)}) must match number of equations ({n}).")

        t_sym = sp.Symbol("t")
        y_syms = [sp.Symbol(v) for v in var_names]
        all_syms = [t_sym] + y_syms
        local_dict = sympy_locals_with_var("t")
        for v, s in zip(var_names, y_syms):
            local_dict[v] = s

        lambdas = []
        for eq_str in equations:
            expr = safe_sympify(eq_str, local_dict)
            lambdas.append(sp.lambdify(all_syms, expr, modules=["numpy"]))

        def system(t, y):
            args = [t] + list(y)
            return [lam(*args) for lam in lambdas]

        t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)
        sol = solve_ivp(system, t_span, y0, method=method, t_eval=t_eval, rtol=1e-8, atol=1e-10)

        result = {
            "t": sol.t.tolist(),
            "success": bool(sol.success),
            "message": sol.message,
            "method": method,
            "n_equations": n,
        }
        for i, name in enumerate(var_names):
            result[name] = sol.y[i].tolist()

        return result

    def solve_second_order(
        self,
        coeff_a: float,
        coeff_b: float,
        coeff_c: float,
        y0: float,
        yp0: float,
        t_span: list[float],
        t_eval_points: int = 200,
    ) -> dict:
        """Solve the second-order linear ODE: a*y'' + b*y' + c*y = 0.

        Converted to a first-order system:
            u0 = y,   u0' = u1
            u1 = y',  u1' = -(b*u1 + c*u0) / a

        Parameters
        ----------
        coeff_a, coeff_b, coeff_c: coefficients of y'', y', y
        y0: initial displacement
        yp0: initial velocity
        """
        if abs(coeff_a) < 1e-15:
            raise ValueError("coeff_a must be non-zero for a second-order ODE.")

        _validate_t_span(t_span, t_eval_points)

        def system(t, u):
            return [u[1], -(coeff_b * u[1] + coeff_c * u[0]) / coeff_a]

        t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)
        sol = solve_ivp(
            system, t_span, [y0, yp0], t_eval=t_eval, rtol=1e-8, atol=1e-10
        )

        # Compute discriminant for analytical classification
        discriminant = coeff_b**2 - 4 * coeff_a * coeff_c
        if discriminant > 0:
            regime = "overdamped"
        elif abs(discriminant) < 1e-10:
            regime = "critically_damped"
        else:
            regime = "underdamped"

        return {
            "t": sol.t.tolist(),
            "y": sol.y[0].tolist(),
            "yp": sol.y[1].tolist(),
            "success": bool(sol.success),
            "regime": regime,
            "discriminant": float(discriminant),
            "coefficients": {"a": coeff_a, "b": coeff_b, "c": coeff_c},
        }
