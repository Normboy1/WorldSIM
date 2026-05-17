"""Mantle convection and geodynamics calculations.

Covers the Rayleigh number, convection regime classification, Stokes flow for
sinking slabs/plumes, thermal boundary layer thickness, and plume buoyancy flux.
These utilities support large-scale geodynamic modelling workflows.

Optional dependencies: scipy (numerical integration), numpy (always required).
Falls back to analytic approximations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import scipy.special as _sp
    _SCIPY = True
except ImportError:
    _SCIPY = False


def rayleigh_number(
    alpha: float = 3e-5,
    g: float = 9.81,
    delta_T: float = 1300.0,
    d: float = 2.9e6,
    kappa: float = 1e-6,
    nu: float = 1e17,
) -> dict[str, Any]:
    """Compute the thermal Rayleigh number for mantle convection.

    Ra = (alpha * g * delta_T * d^3) / (kappa * nu)

    Args:
        alpha: Thermal expansivity (1/K).
        g: Gravitational acceleration (m/s^2).
        delta_T: Temperature difference across the layer (K).
        d: Layer thickness (m).
        kappa: Thermal diffusivity (m^2/s).
        nu: Kinematic viscosity (m^2/s).

    Returns:
        Dict with keys: Ra, convection_regime, critical_Ra, backend, note.
    """
    Ra = (alpha * g * delta_T * d ** 3) / (kappa * nu)
    critical_Ra = 1100.0  # for free-slip boundary conditions
    if Ra < critical_Ra:
        regime = "conductive"
    elif Ra < 1e5:
        regime = "weakly_convective"
    elif Ra < 1e8:
        regime = "vigorous_convection"
    else:
        regime = "turbulent_convection"

    return {
        "backend": "analytic",
        "note": "",
        "Ra": Ra,
        "convection_regime": regime,
        "critical_Ra": critical_Ra,
    }


def convection_regime(Ra: float = 1e6) -> dict[str, Any]:
    """Classify the convection regime from the Rayleigh number.

    Args:
        Ra: Rayleigh number.

    Returns:
        Dict with keys: Ra, regime, nusselt_number_estimate, backend, note.
    """
    if Ra < 1100:
        regime = "conductive"
        Nu = 1.0
    elif Ra < 1e5:
        regime = "weakly_convective"
        Nu = 0.294 * Ra ** (1.0 / 3.0) / 100.0
    elif Ra < 1e9:
        regime = "vigorous_convection"
        Nu = 0.294 * (Ra / 1100.0) ** (1.0 / 3.0)
    else:
        regime = "hard_turbulence"
        Nu = 0.17 * Ra ** 0.278

    return {
        "backend": "analytic",
        "note": "",
        "Ra": Ra,
        "regime": regime,
        "nusselt_number_estimate": Nu,
    }


def stokes_flow_sphere(
    viscosity_Pa_s: float = 1e21,
    radius_m: float = 5e4,
    delta_rho: float = 80.0,
    g: float = 9.81,
) -> dict[str, Any]:
    """Stokes settling velocity of a sphere in a viscous fluid (slab/plume).

    v = (2 * delta_rho * g * r^2) / (9 * mu)

    Args:
        viscosity_Pa_s: Dynamic viscosity of the surrounding mantle (Pa*s).
        radius_m: Radius of the sinking/rising body (m).
        delta_rho: Density contrast (kg/m^3).
        g: Gravitational acceleration (m/s^2).

    Returns:
        Dict with keys: velocity_m_s, velocity_cm_yr, Re, backend, note.
    """
    v = (2.0 * abs(delta_rho) * g * radius_m ** 2) / (9.0 * viscosity_Pa_s)
    v_cm_yr = v * 3.1557e9  # seconds per Gyr -> cm/yr
    rho_mantle = 3300.0
    Re = rho_mantle * v * 2 * radius_m / viscosity_Pa_s
    return {
        "backend": "analytic",
        "note": "",
        "velocity_m_s": v,
        "velocity_cm_yr": v_cm_yr,
        "Re": Re,
    }


def thermal_boundary_layer(
    kappa: float = 1e-6,
    u_plate: float = 5e-10,
    L: float = 1e6,
) -> dict[str, Any]:
    """Estimate the thermal boundary layer thickness by the scaling argument.

    delta_T ~ sqrt(kappa * L / u)

    Args:
        kappa: Thermal diffusivity (m^2/s).
        u_plate: Plate velocity (m/s).
        L: Plate length (m).

    Returns:
        Dict with keys: delta_T_m, delta_T_km, age_at_tip_Ma, backend, note.
    """
    delta_T = math.sqrt(kappa * L / u_plate)
    age_s = L / u_plate if u_plate > 0 else float("inf")
    age_Ma = age_s / (3.1557e13)
    return {
        "backend": "analytic",
        "note": "",
        "delta_T_m": delta_T,
        "delta_T_km": delta_T / 1e3,
        "age_at_tip_Ma": age_Ma,
    }


def plume_buoyancy_flux(
    delta_T: float = 200.0,
    radius_m: float = 5e4,
    velocity_m_s: float = 1e-3,
    rho: float = 3300.0,
    cp: float = 1200.0,
) -> dict[str, Any]:
    """Estimate the buoyancy flux of a mantle plume.

    B = rho * cp * v * pi * r^2 * delta_T  (heat flux proxy)

    Args:
        delta_T: Temperature excess of the plume above ambient mantle (K).
        radius_m: Radius of the plume conduit (m).
        velocity_m_s: Upwelling velocity of the plume (m/s).
        rho: Mantle density (kg/m^3).
        cp: Specific heat capacity (J/kg/K).

    Returns:
        Dict with keys: heat_flux_W, buoyancy_flux_kg_s, backend, note.
    """
    area = math.pi * radius_m ** 2
    heat_flux = rho * cp * velocity_m_s * area * delta_T
    alpha = 3e-5
    g = 9.81
    buoyancy_flux = rho * alpha * g * delta_T * velocity_m_s * area
    return {
        "backend": "analytic",
        "note": "",
        "heat_flux_W": heat_flux,
        "buoyancy_flux_kg_s": buoyancy_flux,
    }
