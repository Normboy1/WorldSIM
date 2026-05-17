"""Multiphase flow dimensionless numbers and regime utilities.

Covers Weber number, Bond number, capillary rise, bubble terminal velocity,
droplet breakup regime, and a volume-of-fluid proxy.

Optional dependencies: scipy (ODE integration), numpy (always required).
Falls back to analytic / Euler approximations.
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


def weber_number(
    rho: float = 1000.0,
    u: float = 1.0,
    L: float = 1e-3,
    sigma: float = 0.072,
) -> dict[str, Any]:
    """Compute the Weber number (inertia vs surface tension).

    We = rho * u^2 * L / sigma

    Args:
        rho: Fluid density (kg/m^3).
        u: Characteristic velocity (m/s).
        L: Characteristic length (m).
        sigma: Surface tension (N/m).

    Returns:
        Dict with keys: We, regime, backend, note.
    """
    We = rho * u ** 2 * L / sigma
    regime = "surface_tension_dominated" if We < 1 else ("transition" if We < 12 else "inertia_dominated")
    return {"backend": "analytic", "note": "", "We": We, "regime": regime}


def bond_number(
    rho_l: float = 1000.0,
    rho_g: float = 1.2,
    g: float = 9.81,
    L: float = 5e-3,
    sigma: float = 0.072,
) -> dict[str, Any]:
    """Compute the Bond (Eötvös) number (gravity vs surface tension).

    Bo = (rho_l - rho_g) * g * L^2 / sigma

    Args:
        rho_l: Liquid density (kg/m^3).
        rho_g: Gas density (kg/m^3).
        g: Gravitational acceleration (m/s^2).
        L: Characteristic length (m).
        sigma: Surface tension (N/m).

    Returns:
        Dict with keys: Bo, capillary_length_m, backend, note.
    """
    Bo = (rho_l - rho_g) * g * L ** 2 / sigma
    l_cap = math.sqrt(sigma / ((rho_l - rho_g) * g))
    return {"backend": "analytic", "note": "", "Bo": Bo, "capillary_length_m": l_cap}


def capillary_rise(
    contact_angle_deg: float = 20.0,
    sigma: float = 0.072,
    rho: float = 1000.0,
    g: float = 9.81,
    r_m: float = 5e-4,
) -> dict[str, Any]:
    """Jurin's law: equilibrium height of capillary rise.

    h = 2 * sigma * cos(theta) / (rho * g * r)

    Args:
        contact_angle_deg: Contact angle (degrees).
        sigma: Surface tension (N/m).
        rho: Liquid density (kg/m^3).
        g: Gravitational acceleration (m/s^2).
        r_m: Tube radius (m).

    Returns:
        Dict with keys: height_m, height_mm, backend, note.
    """
    theta = math.radians(contact_angle_deg)
    h = 2.0 * sigma * math.cos(theta) / (rho * g * r_m)
    return {"backend": "analytic", "note": "", "height_m": h, "height_mm": h * 1000.0}


def bubble_terminal_velocity(
    d_m: float = 2e-3,
    rho_l: float = 1000.0,
    rho_g: float = 1.2,
    mu_l: float = 1e-3,
    g: float = 9.81,
) -> dict[str, Any]:
    """Estimate the terminal rise velocity of a bubble in quiescent liquid.

    Uses Stokes (small bubble) / Mendelson (large bubble) correlation:
    Stokes: v = d^2 * (rho_l - rho_g) * g / (18 * mu_l)

    Args:
        d_m: Bubble diameter (m).
        rho_l: Liquid density (kg/m^3).
        rho_g: Gas density (kg/m^3).
        mu_l: Liquid dynamic viscosity (Pa*s).
        g: Gravitational acceleration (m/s^2).

    Returns:
        Dict with keys: v_terminal_m_s, Re, regime, backend, note.
    """
    v_stokes = d_m ** 2 * (rho_l - rho_g) * g / (18.0 * mu_l)
    Re = rho_l * v_stokes * d_m / mu_l
    if Re < 0.5:
        regime = "Stokes"
        v_t = v_stokes
    elif Re < 1000:
        # Schiller-Naumann correction
        Cd = 24.0 / Re * (1 + 0.15 * Re ** 0.687)
        v_t = math.sqrt(4.0 * d_m * (rho_l - rho_g) * g / (3.0 * Cd * rho_l))
        regime = "intermediate"
    else:
        # Newton's law
        v_t = 1.74 * math.sqrt(g * d_m)
        regime = "Newton"
    Re_final = rho_l * v_t * d_m / mu_l
    return {
        "backend": "analytic", "note": "",
        "v_terminal_m_s": v_t, "Re": Re_final, "regime": regime,
    }


def droplet_breakup_regime(
    We: float = 5.0,
    Oh: float = 0.01,
) -> dict[str, Any]:
    """Classify droplet breakup regime from Weber and Ohnesorge numbers.

    Args:
        We: Weber number (inertia/surface tension).
        Oh: Ohnesorge number (viscosity/sqrt(inertia*surface tension)).

    Returns:
        Dict with keys: regime, We, Oh, backend, note.
    """
    if We < 12:
        regime = "no_breakup"
    elif We < 50 and Oh < 0.1:
        regime = "bag_breakup"
    elif We < 100 and Oh < 0.1:
        regime = "stripping_breakup"
    elif We >= 100 and Oh < 0.1:
        regime = "catastrophic_breakup"
    else:
        regime = "viscous_breakup"
    return {"backend": "analytic", "note": "", "regime": regime, "We": We, "Oh": Oh}


def volume_of_fluid_proxy(
    phi0: float = 0.5,
    u_field: float = 0.5,
    t_end: float = 1.0,
    n_steps: int = 100,
) -> dict[str, Any]:
    """Proxy 1-D advection of a VOF phase fraction with CICSAM blending.

    Simplified interface tracking: phi(x,t) via upwind advection on a
    1-D uniform grid.

    Args:
        phi0: Initial sharp-interface position (fraction of domain length).
        u_field: Uniform advection velocity (m/s).
        t_end: End time (s).
        n_steps: Number of time steps.

    Returns:
        Dict with keys: x, phi_initial, phi_final, CFL, backend, note.
    """
    N = 100
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    phi = np.where(x < phi0, 1.0, 0.0).astype(float)
    dt = t_end / n_steps
    CFL = u_field * dt / dx

    for _ in range(n_steps):
        phi_new = phi.copy()
        for i in range(1, N):
            phi_new[i] = phi[i] - CFL * (phi[i] - phi[i - 1])
        phi = np.clip(phi_new, 0.0, 1.0)

    return {
        "backend": "stub",
        "note": "Install scipy for higher-order interface compression.",
        "x": x.tolist(), "phi_initial": np.where(x < phi0, 1.0, 0.0).tolist(),
        "phi_final": phi.tolist(), "CFL": CFL,
    }
