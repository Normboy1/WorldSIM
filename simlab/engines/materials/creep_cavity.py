"""High-temperature creep and creep cavity nucleation/growth models.

Covers cavity nucleation rate, diffusional growth, constrained growth,
critical cavity radius, Monkman-Grant relation, and omega damage parameter.

Optional dependencies: scipy (ODE integration), numpy (always required).
Falls back to analytic expressions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp as _solve_ivp
    _SCIPY = True
except ImportError:
    _SCIPY = False

_KB = 1.380649e-23
_R  = 8.314


def cavity_nucleation_rate(
    sigma_MPa: float = 150.0,
    T_K: float = 900.0,
    Q_kJ_mol: float = 250.0,
    A: float = 1e10,
) -> dict[str, Any]:
    """Cavity nucleation rate at grain boundaries (stress-assisted).

    N_dot = A * sigma^n * exp(-Q / R*T)   (empirical power-law)

    Args:
        sigma_MPa: Applied stress (MPa).
        T_K: Temperature (K).
        Q_kJ_mol: Activation energy for cavity nucleation (kJ/mol).
        A: Pre-exponential factor.

    Returns:
        Dict with keys: nucleation_rate_m2_s, backend, note.
    """
    n = 5.0  # typical stress exponent
    rate = A * (sigma_MPa ** n) * math.exp(-Q_kJ_mol * 1e3 / (_R * T_K))
    return {
        "backend": "analytic", "note": "",
        "nucleation_rate_m2_s": rate,
        "sigma_MPa": sigma_MPa, "T_K": T_K,
    }


def cavity_growth_rate(
    r_m: float = 1e-7,
    sigma_MPa: float = 150.0,
    T_K: float = 900.0,
    D_gb: float = 1e-14,
    delta_gb: float = 5e-10,
    Omega: float = 1.2e-29,
) -> dict[str, Any]:
    """Hull-Rimmer diffusional cavity growth rate.

    dr/dt = D_gb * delta_gb * Omega * sigma / (kT * r^2 * ln(L/r))

    Args:
        r_m: Current cavity radius (m).
        sigma_MPa: Applied stress (MPa).
        T_K: Temperature (K).
        D_gb: Grain boundary diffusivity (m^2/s).
        delta_gb: Grain boundary width (m).
        Omega: Atomic volume (m^3).

    Returns:
        Dict with keys: dr_dt_m_s, backend, note.
    """
    sigma = sigma_MPa * 1e6
    L = 1e-6  # grain boundary spacing proxy
    ln_term = math.log(L / r_m) if r_m < L else 1.0
    dr_dt = D_gb * delta_gb * Omega * sigma / (_KB * T_K * r_m ** 2 * ln_term)
    return {
        "backend": "analytic", "note": "",
        "dr_dt_m_s": dr_dt, "r_m": r_m,
    }


def constrained_diffusional_growth(
    a_m: float = 1e-7,
    L_m: float = 5e-6,
    D_gb: float = 1e-14,
    delta_gb: float = 5e-10,
    sigma_MPa: float = 150.0,
    T_K: float = 900.0,
) -> dict[str, Any]:
    """Constrained diffusional creep cavity growth (Dyson 1976).

    Growth constrained by deformation rate of surrounding matrix.

    Args:
        a_m: Cavity radius (m).
        L_m: Cavity half-spacing (m).
        D_gb: Grain boundary diffusivity (m^2/s).
        delta_gb: GB width (m).
        sigma_MPa: Applied stress (MPa).
        T_K: Temperature (K).

    Returns:
        Dict with keys: da_dt_m_s, constraint_factor, backend, note.
    """
    sigma = sigma_MPa * 1e6
    Omega = 1.2e-29
    q = (a_m / L_m) ** 2
    constraint = 1.0 - q
    da_dt = D_gb * delta_gb * Omega * sigma / (_KB * T_K * a_m ** 2) * constraint
    return {
        "backend": "analytic", "note": "",
        "da_dt_m_s": da_dt, "constraint_factor": constraint,
    }


def critical_cavity_radius(
    sigma_MPa: float = 150.0,
    gamma_s: float = 2.0,
) -> dict[str, Any]:
    """Equilibrium (critical) cavity nucleus radius.

    r* = 2 * gamma_s / sigma

    Args:
        sigma_MPa: Applied stress (MPa).
        gamma_s: Surface energy (J/m^2).

    Returns:
        Dict with keys: r_critical_m, r_critical_nm, backend, note.
    """
    sigma = sigma_MPa * 1e6
    r_star = 2.0 * gamma_s / sigma
    return {
        "backend": "analytic", "note": "",
        "r_critical_m": r_star, "r_critical_nm": r_star * 1e9,
    }


def monkman_grant(
    t_r: float = 1000.0,
    epsilon_dot_min: float = 1e-7,
) -> dict[str, Any]:
    """Monkman-Grant relation: minimum creep rate times rupture life ≈ constant.

    C_MG = epsilon_dot_min * t_r^m  (m ~ 1)

    Args:
        t_r: Rupture life (h).
        epsilon_dot_min: Minimum (steady-state) creep rate (1/h).

    Returns:
        Dict with keys: C_monkman_grant, epsilon_dot_min, t_r, backend, note.
    """
    m_exp = 0.95  # slightly less than 1.0 for many alloys
    C = epsilon_dot_min * t_r ** m_exp
    return {
        "backend": "analytic", "note": "",
        "C_monkman_grant": C, "epsilon_dot_min": epsilon_dot_min, "t_r": t_r,
    }


def omega_damage(
    epsilon_c: float = 0.02,
    epsilon_f: float = 0.15,
    n_exponent: float = 5.0,
) -> dict[str, Any]:
    """Omega (omega) creep damage parameter (Cocks-Ashby).

    omega = 1 - (1 - epsilon_c / epsilon_f)^(1/(n+1))

    Args:
        epsilon_c: Accumulated creep strain.
        epsilon_f: Failure creep strain (ductility).
        n_exponent: Creep stress exponent.

    Returns:
        Dict with keys: omega, remaining_life_fraction, failed, backend, note.
    """
    frac = max(1.0 - epsilon_c / epsilon_f, 1e-6)
    omega = 1.0 - frac ** (1.0 / (n_exponent + 1.0))
    remaining = 1.0 - omega
    return {
        "backend": "analytic", "note": "",
        "omega": omega, "remaining_life_fraction": remaining, "failed": omega >= 1.0,
    }
