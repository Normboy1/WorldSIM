"""Magnetohydrodynamics (MHD) dimensionless numbers and stability criteria.

Provides the Alfvén velocity, magnetic Reynolds number, MHD stability
criterion (safety factor and beta), Hartmann number, Lundquist number, and
fast/slow magnetosonic speeds.

Optional dependencies: scipy, numpy (always required).
All functions are analytic; scipy only enhances numerical helpers.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import scipy  # noqa: F401
    _SCIPY = True
except ImportError:
    _SCIPY = False

_MU0 = 1.2566370614e-6  # permeability of free space (H/m)


def alfven_velocity(
    B_T: float = 1.0,
    rho_kg_m3: float = 1e-7,
    mu0: float = _MU0,
) -> dict[str, Any]:
    """Compute the Alfvén wave velocity in a magnetised plasma.

    v_A = B / sqrt(mu0 * rho)

    Args:
        B_T: Magnetic field strength (Tesla).
        rho_kg_m3: Mass density (kg/m^3).
        mu0: Magnetic permeability (H/m).

    Returns:
        Dict with keys: v_alfven_m_s, v_alfven_km_s, backend, note.
    """
    v_A = B_T / math.sqrt(mu0 * rho_kg_m3)
    return {
        "backend": "analytic", "note": "",
        "v_alfven_m_s": v_A, "v_alfven_km_s": v_A / 1e3,
    }


def magnetic_reynolds_number(
    sigma: float = 1e6,
    u: float = 1e4,
    L: float = 1.0,
) -> dict[str, Any]:
    """Compute the magnetic Reynolds number.

    Rm = mu0 * sigma * u * L

    Args:
        sigma: Electrical conductivity (S/m).
        u: Characteristic velocity (m/s).
        L: Characteristic length scale (m).

    Returns:
        Dict with keys: Rm, interpretation, backend, note.
    """
    Rm = _MU0 * sigma * u * L
    interp = "advection-dominated" if Rm > 1 else "diffusion-dominated"
    return {
        "backend": "analytic", "note": "",
        "Rm": Rm, "interpretation": interp,
    }


def mhd_stability_criterion(
    q_factor: float = 1.5,
    beta_plasma: float = 0.05,
) -> dict[str, Any]:
    """Assess MHD stability via the Kruskal-Shafranov safety factor.

    Kink stable if q > 1.  Ballooning stable if beta < beta_crit.

    Args:
        q_factor: Safety factor q = r*B_toroidal / (R*B_poloidal).
        beta_plasma: Plasma beta = 2*mu0*p / B^2.

    Returns:
        Dict with keys: kink_stable, ballooning_margin, q_factor, beta_plasma,
        backend, note.
    """
    kink_stable = q_factor > 1.0
    beta_crit = 0.28 / (q_factor + 0.1)  # empirical Troyon limit proxy
    ballooning_margin = beta_crit - beta_plasma
    return {
        "backend": "analytic", "note": "",
        "kink_stable": kink_stable, "beta_crit": beta_crit,
        "ballooning_margin": ballooning_margin,
        "q_factor": q_factor, "beta_plasma": beta_plasma,
    }


def hartmann_number(
    B: float = 1.0,
    L: float = 0.01,
    sigma: float = 3.7e6,
    rho: float = 6361.0,
    nu: float = 2.5e-7,
) -> dict[str, Any]:
    """Compute the Hartmann number for MHD duct flow.

    Ha = B * L * sqrt(sigma / (rho * nu))

    Args:
        B: Magnetic field (T).
        L: Channel half-gap (m).
        sigma: Electrical conductivity (S/m).
        rho: Fluid density (kg/m^3).
        nu: Kinematic viscosity (m^2/s).

    Returns:
        Dict with keys: Ha, flow_regime, backend, note.
    """
    Ha = B * L * math.sqrt(sigma / (rho * nu))
    regime = "viscous_dominated" if Ha < 1 else ("transition" if Ha < 100 else "MHD_dominated")
    return {
        "backend": "analytic", "note": "",
        "Ha": Ha, "flow_regime": regime,
    }


def lundquist_number(
    B: float = 1.0,
    L: float = 1.0,
    sigma: float = 1e6,
    rho: float = 1e-7,
    mu0: float = _MU0,
) -> dict[str, Any]:
    """Compute the Lundquist number S (ratio of resistive diffusion to Alfvén time).

    S = mu0 * sigma * v_A * L  =  v_A * L / eta  where eta = 1/(mu0*sigma)

    Args:
        B: Magnetic field (T).
        L: System size (m).
        sigma: Electrical conductivity (S/m).
        rho: Mass density (kg/m^3).
        mu0: Magnetic permeability (H/m).

    Returns:
        Dict with keys: S, v_alfven_m_s, resistive_diffusion_time_s, backend, note.
    """
    v_A = B / math.sqrt(mu0 * rho)
    eta = 1.0 / (mu0 * sigma)
    S = v_A * L / eta
    t_diff = mu0 * sigma * L ** 2
    return {
        "backend": "analytic", "note": "",
        "S": S, "v_alfven_m_s": v_A,
        "resistive_diffusion_time_s": t_diff,
    }


def magnetosonic_speed(
    B: float = 1.0,
    rho: float = 1e-7,
    P: float = 1.0,
    gamma: float = 5.0 / 3.0,
    mu0: float = _MU0,
) -> dict[str, Any]:
    """Compute fast and slow magnetosonic speeds.

    c_f^2 = 0.5 * [(c_s^2 + v_A^2) + sqrt((c_s^2 + v_A^2)^2 - 4 c_s^2 v_A^2 cos^2 theta)]
    Evaluated at theta=90 deg (maximum fast mode).

    Args:
        B: Magnetic field (T).
        rho: Mass density (kg/m^3).
        P: Plasma pressure (Pa).
        gamma: Adiabatic index.
        mu0: Magnetic permeability (H/m).

    Returns:
        Dict with keys: c_fast_m_s, c_slow_m_s, c_sound_m_s, v_alfven_m_s,
        backend, note.
    """
    c_s2 = gamma * P / rho
    v_A2 = B ** 2 / (mu0 * rho)
    c_f = math.sqrt(c_s2 + v_A2)  # theta=90: c_f^2 = c_s^2 + v_A^2
    c_slow = math.sqrt(c_s2 * v_A2 / (c_s2 + v_A2))  # approx
    return {
        "backend": "analytic", "note": "",
        "c_fast_m_s": c_f, "c_slow_m_s": c_slow,
        "c_sound_m_s": math.sqrt(c_s2), "v_alfven_m_s": math.sqrt(v_A2),
    }
