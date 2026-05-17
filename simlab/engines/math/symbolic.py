"""Symbolic math engine using SymPy.

Provides solve, simplify, differentiate, integrate, expand, factor,
critical points, and Taylor series expansion.
"""

from __future__ import annotations

import sympy as sp

from simlab.engines.math._sympy_parse import (
    safe_sympify,
    sympy_locals_with_var,
    try_split_binary_equals,
)


def _parse(expr_str: str, variable: str = "x") -> tuple[sp.Expr, sp.Symbol]:
    """Parse expression string and return (expr, symbol)."""
    sym = sp.Symbol(variable)
    local_dict = sympy_locals_with_var(variable)
    expr = safe_sympify(expr_str, local_dict)
    return expr, sym


def _try_numeric(expr: sp.Expr) -> float | None:
    """Attempt to evaluate expr to a float."""
    try:
        val = complex(expr.evalf())
        if val.imag == 0:
            return float(val.real)
        return None
    except Exception:
        return None


def _solutions_as_list(solutions: object) -> list:
    """Normalise sympy.solve output to a list of solution expressions."""
    if solutions is None:
        return []
    if isinstance(solutions, (list, tuple)):
        return list(solutions)
    if isinstance(solutions, dict):
        return list(solutions.values())
    return [solutions]


def _classify_critical_point(second_deriv: sp.Expr) -> str:
    """Classify a stationary point from d²y/dx² using numeric evaluation when possible."""
    try:
        val = complex(second_deriv.evalf())
    except (TypeError, ValueError):
        return "undetermined"
    if abs(val.imag) > 1e-8:
        return "undetermined"
    re = val.real
    if re > 1e-12:
        return "minimum"
    if re < -1e-12:
        return "maximum"
    return "inconclusive"


class SymbolicEngine:
    """SymPy-backed symbolic math engine."""

    def solve_equation(self, equation_str: str, variable: str = "x") -> dict:
        """Parse and solve an equation (set = 0 or explicit equality).

        Parameters
        ----------
        equation_str:
            Equation string, e.g. 'x**2 - 4' or 'x**2 - 4 = 0' or 'Eq(x**2, 4)'.
        variable:
            Variable to solve for.
        """
        sym = sp.Symbol(variable)
        local_dict = sympy_locals_with_var(variable)
        local_dict["Eq"] = sp.Eq

        pair = try_split_binary_equals(equation_str)
        if pair is not None:
            lhs_str, rhs_str = pair
            lhs = safe_sympify(lhs_str, local_dict)
            rhs = safe_sympify(rhs_str, local_dict)
            equation = sp.Eq(lhs, rhs)
        else:
            parsed = safe_sympify(equation_str.strip(), local_dict)
            if isinstance(parsed, sp.Eq):
                equation = parsed
            else:
                equation = parsed  # treat as expression = 0

        solutions_raw = sp.solve(equation, sym)
        solutions = _solutions_as_list(solutions_raw)
        solutions_str = [str(s) for s in solutions]
        solutions_latex = [sp.latex(s) for s in solutions]
        solutions_numeric = [_try_numeric(s) for s in solutions]

        return {
            "solutions": solutions_str,
            "latex": solutions_latex,
            "numeric": solutions_numeric,
            "count": len(solutions),
            "steps": f"Solved {equation_str!r} for {variable!r}",
        }

    def simplify_expression(self, expr_str: str, variable: str = "x") -> dict:
        """Simplify a mathematical expression."""
        expr, sym = _parse(expr_str, variable)
        simplified = sp.simplify(expr)
        return {
            "symbolic": str(simplified),
            "latex": sp.latex(simplified),
            "numeric": _try_numeric(simplified),
            "steps": f"Applied sympy.simplify to: {expr_str}",
        }

    def differentiate(
        self, expr_str: str, variable: str = "x", order: int = 1
    ) -> dict:
        """Differentiate an expression with respect to variable, order times."""
        if int(order) < 1:
            raise ValueError("order must be a positive integer.")
        expr, sym = _parse(expr_str, variable)
        result = sp.diff(expr, sym, order)
        simplified = sp.simplify(result)
        return {
            "symbolic": str(simplified),
            "latex": sp.latex(simplified),
            "numeric": _try_numeric(simplified),
            "order": order,
            "variable": variable,
            "steps": f"d^{order}/d{variable}^{order} of ({expr_str})",
        }

    def integrate(
        self,
        expr_str: str,
        variable: str = "x",
        limits: list[float] | None = None,
    ) -> dict:
        """Integrate an expression (definite if limits=[a,b], else indefinite)."""
        expr, sym = _parse(expr_str, variable)

        if limits is not None and len(limits) == 2:
            a, b = limits
            result = sp.integrate(expr, (sym, a, b))
            kind = "definite"
            steps = f"∫[{a},{b}] ({expr_str}) d{variable}"
        else:
            result = sp.integrate(expr, sym)
            kind = "indefinite"
            steps = f"∫ ({expr_str}) d{variable} + C"

        simplified = sp.simplify(result)
        return {
            "symbolic": str(simplified),
            "latex": sp.latex(simplified),
            "numeric": _try_numeric(simplified),
            "kind": kind,
            "limits": limits,
            "steps": steps,
        }

    def expand_expression(self, expr_str: str, variable: str = "x") -> dict:
        """Expand an algebraic expression."""
        expr, sym = _parse(expr_str, variable)
        result = sp.expand(expr)
        return {
            "symbolic": str(result),
            "latex": sp.latex(result),
            "numeric": _try_numeric(result),
            "steps": f"expand({expr_str})",
        }

    def factor_expression(self, expr_str: str, variable: str = "x") -> dict:
        """Factor a polynomial expression."""
        expr, sym = _parse(expr_str, variable)
        result = sp.factor(expr)
        return {
            "symbolic": str(result),
            "latex": sp.latex(result),
            "numeric": _try_numeric(result),
            "steps": f"factor({expr_str})",
        }

    def find_critical_points(self, expr_str: str, variable: str = "x") -> dict:
        """Find critical points by solving dy/dx = 0."""
        expr, sym = _parse(expr_str, variable)
        derivative = sp.diff(expr, sym)
        critical_raw = sp.solve(derivative, sym)
        critical = _solutions_as_list(critical_raw)

        points = []
        for cp in critical:
            y_val = expr.subs(sym, cp)
            second_deriv = sp.diff(derivative, sym).subs(sym, cp)
            kind = _classify_critical_point(second_deriv)
            points.append(
                {
                    "x": str(cp),
                    "x_numeric": _try_numeric(cp),
                    "y": str(y_val),
                    "y_numeric": _try_numeric(y_val),
                    "type": kind,
                }
            )

        return {
            "critical_points": points,
            "derivative": str(derivative),
            "derivative_latex": sp.latex(derivative),
            "steps": f"Solved d/d{variable}({expr_str}) = 0",
        }

    def taylor_series(
        self,
        expr_str: str,
        variable: str = "x",
        point: float = 0,
        order: int = 6,
    ) -> dict:
        """Compute Taylor series expansion around a point."""
        if int(order) < 1:
            raise ValueError("order must be at least 1.")
        expr, sym = _parse(expr_str, variable)
        series = sp.series(expr, sym, point, order)
        # Remove the O() term for cleaner output
        series_no_O = series.removeO()
        return {
            "symbolic": str(series_no_O),
            "full_series": str(series),
            "latex": sp.latex(series_no_O),
            "point": point,
            "order": order,
            "steps": f"Taylor series of ({expr_str}) around {variable}={point}, order {order}",
        }
