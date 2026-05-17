"""Grain boundary thermodynamics and kinetics.

Covers grain boundary energy (Read-Shockley), mobility, grain growth rate,
Zener pinning, and criteria for abnormal grain growth.

Optional dependencies: scipy (minimisation), numpy (always required).
Falls back to analytic expressions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import scipy.optimize as _opt
    _SCIPY = True
except ImportError:
    _SCIPY = False


def grain_boundary_energy(
    theta_deg: float = 15.0,
    E0: float = 0.7,
    theta0: float = 15.0,
) -> dict[str, Any]:
    """Read-Shockley grain boundary energy model.

    E(theta) = E0 * (theta/theta0) * [1 - ln(theta/theta0)]   for theta <= theta0
    E(theta) = E0                                               for theta > theta0

    Args:
        theta_deg: Misorientation angle (degrees).
        E0: Maximum grain boundary energy (J/m^2).
        theta0: Critical angle for high-angle grain boundary (degrees).

    Returns:
        Dict with keys: E_J_m2, theta_deg, gb_type, backend, note.
    """
    frac = theta_deg / theta0
    if theta_deg <= theta0:
        E = E0 * frac * (1.0 - math.log(frac)) if frac > 0 else 0.0
        gb_type = "low_angle"
    else:
        E = E0
        gb_type = "high_angle"
    return {
        "backend": "analytic", "note": "",
        "E_J_m2": E, "theta_deg": theta_deg, "gb_type": gb_type,
    }


def read_shockley_model(
    theta_deg: float = 10.0,
    b_m: float = 2.86e-10,
    nu: float = 0.33,
    G_Pa: float = 80.0,
) -> dict[str, Any]:
    """Read-Shockley energy from dislocation spacing in low-angle GBs.

    E = (G*b/(4*pi*(1-nu))) * theta * [A - ln(theta)]
    where A = 1 + ln(b/(2*pi*r0)), r0 ~ b.

    Args:
        theta_deg: Misorientation angle (degrees).
        b_m: Burgers vector (m).
        nu: Poisson's ratio.
        G_Pa: Shear modulus (GPa).

    Returns:
        Dict with keys: E_J_m2, dislocation_spacing_m, backend, note.
    """
    G = G_Pa * 1e9
    theta = math.radians(theta_deg)
    r0 = b_m
    A = 1.0 + math.log(b_m / (2.0 * math.pi * r0))
    if theta > 0:
        E = (G * b_m / (4.0 * math.pi * (1.0 - nu))) * theta * (A - math.log(theta))
        D = b_m / theta  # dislocation spacing
    else:
        E = 0.0
        D = float("inf")
    return {
        "backend": "analytic", "note": "",
        "E_J_m2": max(E, 0.0), "dislocation_spacing_m": D,
    }


def grain_boundary_mobility(
    Q_kJ_mol: float = 150.0,
    T_K: float = 1000.0,
    M0: float = 1e-7,
) -> dict[str, Any]:
    """Arrhenius grain boundary mobility.

    M = M0 * exp(-Q / (R*T))

    Args:
        Q_kJ_mol: Activation energy (kJ/mol).
        T_K: Temperature (K).
        M0: Pre-exponential mobility (m^4/J/s).

    Returns:
        Dict with keys: M_m4_J_s, Q_kJ_mol, T_K, backend, note.
    """
    R = 8.314
    M = M0 * math.exp(-Q_kJ_mol * 1e3 / (R * T_K))
    return {
        "backend": "analytic", "note": "",
        "M_m4_J_s": M, "Q_kJ_mol": Q_kJ_mol, "T_K": T_K,
    }


def grain_growth_rate(
    M: float = 1e-15,
    P_driving: float = 5e4,
    R_pinning: float = 1e4,
) -> dict[str, Any]:
    """Grain boundary migration rate from driving and pinning pressure balance.

    v = M * (P_driving - P_pinning)

    Args:
        M: Grain boundary mobility (m^4/J/s).
        P_driving: Curvature driving pressure (Pa).
        R_pinning: Total pinning pressure from particles (Pa).

    Returns:
        Dict with keys: velocity_m_s, net_pressure_Pa, growing, backend, note.
    """
    P_net = P_driving - R_pinning
    v = M * max(P_net, 0.0)
    return {
        "backend": "analytic", "note": "",
        "velocity_m_s": v, "net_pressure_Pa": P_net, "growing": P_net > 0,
    }


def zener_pinning(
    f_v: float = 0.01,
    r_m: float = 1e-7,
) -> dict[str, Any]:
    """Zener pinning pressure from second-phase particles.

    P_Z = 3 * gamma * f_v / (2 * r)

    Args:
        f_v: Volume fraction of pinning particles.
        r_m: Mean particle radius (m).

    Returns:
        Dict with keys: P_zener_Pa, limiting_grain_size_m, backend, note.
    """
    gamma_gb = 0.5  # J/m^2 typical
    P_Z = 3.0 * gamma_gb * f_v / (2.0 * r_m)
    R_lim = 4.0 * r_m / (3.0 * f_v)  # Zener limiting grain radius
    return {
        "backend": "analytic", "note": "",
        "P_zener_Pa": P_Z, "limiting_grain_size_m": 2 * R_lim,
    }


def abnormal_grain_growth_criterion(
    M_ratio: float = 3.0,
    P_ratio: float = 0.5,
) -> dict[str, Any]:
    """Check whether abnormal (secondary) grain growth can occur.

    A grain is abnormal if M_abn / M_matrix > P_zener / P_curvature ratio.

    Args:
        M_ratio: Ratio of abnormal grain mobility to matrix grain mobility.
        P_ratio: Ratio of Zener pinning to curvature driving pressure
            for the matrix.

    Returns:
        Dict with keys: abnormal_growth_possible, M_ratio, P_ratio, backend, note.
    """
    # Abnormal growth occurs if the mobility ratio exceeds the pressure ratio
    abnormal = M_ratio > 1.0 / (1.0 - P_ratio) if P_ratio < 1.0 else False
    return {
        "backend": "analytic", "note": "",
        "abnormal_growth_possible": abnormal,
        "M_ratio": M_ratio, "P_ratio": P_ratio,
    }
