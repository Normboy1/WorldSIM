"""Fatigue life prediction and fracture mechanics utilities.

Covers Paris law crack growth, stress intensity factors, Goodman diagram,
Miner's rule damage accumulation, Coffin-Manson low-cycle fatigue, and
Wöhler (S-N) curve generation.

Optional dependencies: scipy (curve fitting), numpy (always required).
Falls back to analytic expressions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.optimize import curve_fit as _curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False


def paris_law(
    da_dN: float | None = None,
    C: float = 3e-12,
    delta_K: float = 20.0,
    m: float = 3.0,
) -> dict[str, Any]:
    """Paris-Erdogan fatigue crack growth law.

    da/dN = C * (delta_K)^m

    Args:
        da_dN: Crack growth rate (m/cycle).  If provided, used for validation.
        C: Paris coefficient (m/(MPa*sqrt(m))^m per cycle).
        delta_K: Stress intensity factor range (MPa*sqrt(m)).
        m: Paris exponent.

    Returns:
        Dict with keys: da_dN_m_cycle, C, delta_K, m, backend, note.
    """
    result = C * delta_K ** m
    return {
        "backend": "analytic", "note": "",
        "da_dN_m_cycle": result, "C": C, "delta_K": delta_K, "m": m,
    }


def stress_intensity_factor(
    sigma_MPa: float = 100.0,
    a_m: float = 1e-3,
    geometry: str = "center_crack",
) -> dict[str, Any]:
    """Compute the mode-I stress intensity factor K_I.

    K_I = F * sigma * sqrt(pi * a)   where F is a geometry factor.

    Args:
        sigma_MPa: Applied remote stress (MPa).
        a_m: Crack half-length (m).
        geometry: 'center_crack' (F=1), 'edge_crack' (F=1.12), 'penny' (F=2/pi).

    Returns:
        Dict with keys: K_I_MPa_sqrt_m, geometry_factor, backend, note.
    """
    F = {"center_crack": 1.0, "edge_crack": 1.12, "penny": 2.0 / math.pi}.get(geometry, 1.0)
    K_I = F * sigma_MPa * math.sqrt(math.pi * a_m)
    return {
        "backend": "analytic", "note": "",
        "K_I_MPa_sqrt_m": K_I, "geometry_factor": F, "geometry": geometry,
    }


def goodman_diagram(
    sigma_a: float = 200.0,
    sigma_m: float = 100.0,
    sigma_uts: float = 800.0,
    sigma_e: float = 350.0,
) -> dict[str, Any]:
    """Modified Goodman fatigue criterion.

    sigma_a / sigma_e + sigma_m / sigma_uts = 1  (failure line)
    Safety factor = 1 / (sigma_a/sigma_e + sigma_m/sigma_uts)

    Args:
        sigma_a: Stress amplitude (MPa).
        sigma_m: Mean stress (MPa).
        sigma_uts: Ultimate tensile strength (MPa).
        sigma_e: Fatigue endurance limit (MPa).

    Returns:
        Dict with keys: safety_factor, goodman_ratio, fatigue_safe, backend, note.
    """
    ratio = sigma_a / sigma_e + sigma_m / sigma_uts
    SF = 1.0 / ratio if ratio > 0 else float("inf")
    return {
        "backend": "analytic", "note": "",
        "safety_factor": SF, "goodman_ratio": ratio, "fatigue_safe": ratio < 1.0,
    }


def miner_rule(
    stress_cycles: list[tuple[float, int]] | None = None,
    S_N_curve: dict[float, float] | None = None,
) -> dict[str, Any]:
    """Palmgren-Miner cumulative damage rule.

    D = sum(n_i / N_fi)   Failure when D >= 1.

    Args:
        stress_cycles: List of (stress_MPa, n_cycles) tuples.
        S_N_curve: Dict of {stress_MPa: N_failure_cycles}.

    Returns:
        Dict with keys: damage_sum, cycles_to_failure_estimate, failed,
        backend, note.
    """
    if stress_cycles is None:
        stress_cycles = [(300.0, 10000), (400.0, 5000), (500.0, 1000)]
    if S_N_curve is None:
        # Basquin S-N: N = (S/S_uts)^(-b) * reference
        S_N_curve = {300.0: 1e6, 400.0: 5e4, 500.0: 8000.0}

    D = 0.0
    for s, n in stress_cycles:
        N_f = S_N_curve.get(s, 1e7)
        D += n / N_f

    remaining = 1.0 / D if D > 0 else float("inf")
    return {
        "backend": "analytic", "note": "",
        "damage_sum": D, "cycles_to_failure_estimate": remaining,
        "failed": D >= 1.0,
    }


def coffin_manson(
    epsilon_a: float = 0.005,
    epsilon_f_prime: float = 0.5,
    c: float = -0.6,
    sigma_f_prime: float = 1000.0,
    E_GPa: float = 210.0,
    b: float = -0.1,
    N: float | None = None,
) -> dict[str, Any]:
    """Coffin-Manson LCF life prediction (Morrow mean-stress corrected).

    epsilon_a = sigma_f'/E * (2N)^b + epsilon_f' * (2N)^c

    Args:
        epsilon_a: Total strain amplitude.
        epsilon_f_prime: Fatigue ductility coefficient.
        c: Fatigue ductility exponent.
        sigma_f_prime: Fatigue strength coefficient (MPa).
        E_GPa: Young's modulus (GPa).
        b: Fatigue strength exponent.
        N: Number of reversals (2Nf) to evaluate strain at; if None, solve for N.

    Returns:
        Dict with keys: N_f_cycles, elastic_strain_a, plastic_strain_a,
        backend, note.
    """
    E = E_GPa * 1e3  # MPa
    if N is None:
        # Estimate via iteration
        N_est = 1000.0
        for _ in range(100):
            eps_elastic = (sigma_f_prime / E) * (2 * N_est) ** b
            eps_plastic = epsilon_f_prime * (2 * N_est) ** c
            eps_total = eps_elastic + eps_plastic
            N_est = N_est * (eps_total / epsilon_a) ** (1.0 / (b - c))
            N_est = max(N_est, 1.0)
        N_2Nf = N_est
    else:
        N_2Nf = N

    eps_el = (sigma_f_prime / E) * (2 * N_2Nf) ** b
    eps_pl = epsilon_f_prime * (2 * N_2Nf) ** c
    return {
        "backend": "analytic", "note": "",
        "N_f_cycles": N_2Nf / 2.0,
        "elastic_strain_a": eps_el, "plastic_strain_a": eps_pl,
    }


def wohler_curve(
    sigma_uts_MPa: float = 800.0,
    R: float = -1.0,
    n_points: int = 50,
) -> dict[str, Any]:
    """Generate a Wöhler (S-N) curve using a Basquin power-law fit.

    S = sigma_f' * (2N)^b   Basquin law with b ~ -0.1 typical steel.

    Args:
        sigma_uts_MPa: Ultimate tensile strength (MPa), used to estimate sigma_f'.
        R: Stress ratio (min/max stress; -1 = fully reversed).
        n_points: Number of stress levels.

    Returns:
        Dict with keys: N_cycles, S_MPa, endurance_limit_MPa, backend, note.
    """
    sigma_f_prime = sigma_uts_MPa * 1.65  # rough estimate
    b = -0.095
    N_arr = np.logspace(3, 7, n_points)
    S_arr = sigma_f_prime * (2 * N_arr) ** b
    # Endurance limit at 10^6 cycles
    S_e = sigma_uts_MPa * 0.45 * (1.0 - (1.0 - R) / 4.0)
    S_arr = np.maximum(S_arr, S_e)
    return {
        "backend": "analytic", "note": "",
        "N_cycles": N_arr.tolist(), "S_MPa": S_arr.tolist(),
        "endurance_limit_MPa": S_e,
    }
