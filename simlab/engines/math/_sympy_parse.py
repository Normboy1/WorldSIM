"""Shared SymPy string parsing helpers (safe sympify, equation splitting)."""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# Built once at import time — dir(sp) is ~700 names, no need to recompute per call.
_SP_CALLABLE_NAMESPACE: dict = {
    name: obj
    for name in dir(sp)
    if callable(obj := getattr(sp, name))
    # Exclude dunder names and anything that could be used for code execution.
    and not name.startswith("_")
    and name not in {"import_module", "exec_", "eval_", "compile"}
}


def sympy_callable_namespace() -> dict:
    """Return a copy of the cached SymPy callable namespace."""
    return dict(_SP_CALLABLE_NAMESPACE)


def sympy_locals_with_var(variable: str) -> dict:
    """Namespace for parse_expr: variable symbol plus SymPy callables."""
    out = dict(_SP_CALLABLE_NAMESPACE)
    out[variable] = sp.Symbol(variable)
    return out


def sympy_locals_for_symbols(names: list[str]) -> dict:
    """Namespace for multivariate parse_expr."""
    out = dict(_SP_CALLABLE_NAMESPACE)
    for n in names:
        out[n] = sp.Symbol(n)
    return out


def safe_sympify(expr_str: str, local_dict: dict) -> sp.Expr:
    """Parse *expr_str* safely using parse_expr; raise ValueError on failure.

    Uses parse_expr (not sympify) to avoid arbitrary Python evaluation.
    """
    try:
        return parse_expr(expr_str, local_dict=local_dict, transformations=_TRANSFORMATIONS)
    except Exception as exc:
        raise ValueError(f"Could not parse expression: {expr_str!r}") from exc


# Do not split user input on '=' when these appear (avoids breaking <=, ==, etc.)
_RELATIONAL_TOKENS = ("==", "<=", ">=", "!=", "=>")


def try_split_binary_equals(equation_str: str) -> tuple[str, str] | None:
    """If *equation_str* is a simple ``lhs = rhs``, return (lhs, rhs); else None.

    Does not split when ``Eq(...)`` or multi-letter relations (``<=``, ``==``, …) appear.
    """
    s = equation_str.strip()
    if "Eq(" in s:
        return None
    if any(tok in s for tok in _RELATIONAL_TOKENS):
        return None
    if "=" not in s:
        return None
    parts = s.split("=", 1)
    if len(parts) != 2:
        return None
    lhs, rhs = parts[0].strip(), parts[1].strip()
    if not lhs or not rhs:
        return None
    return lhs, rhs
