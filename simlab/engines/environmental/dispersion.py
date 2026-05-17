"""Atmospheric dispersion and urban microclimate engine.

Covers the Gaussian plume model, Pasquill-Gifford stability classification,
sigma parameterisation, urban heat island effect, and noise propagation.

Optional dependencies: scipy (special functions), numpy (always required).
Falls back to analytic / look-up approximations.
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


def gaussian_plume(
    Q: float = 100.0,
    u: float = 5.0,
    sigma_y: float = 30.0,
    sigma_z: float = 15.0,
    x: float = 500.0,
    y: float = 0.0,
    z: float = 0.0,
    H: float = 50.0,
) -> dict[str, Any]:
    """Compute downwind ground-level concentration via the Gaussian plume model.

    C(x,y,z) = Q / (2*pi*u*sigma_y*sigma_z) *
                exp(-y^2/(2*sigma_y^2)) *
                [exp(-(z-H)^2/(2*sigma_z^2)) + exp(-(z+H)^2/(2*sigma_z^2))]

    Args:
        Q: Source emission rate (g/s).
        u: Mean wind speed (m/s).
        sigma_y: Lateral dispersion coefficient (m) at distance x.
        sigma_z: Vertical dispersion coefficient (m) at distance x.
        x: Downwind distance (m).
        y: Crosswind distance (m).
        z: Height of receptor (m).
        H: Effective stack height (m).

    Returns:
        Dict with keys: concentration_g_m3, x, y, z, backend, note.
    """
    if sigma_y <= 0 or sigma_z <= 0 or u <= 0:
        return {"backend": "analytic", "note": "Invalid parameters.", "concentration_g_m3": 0.0}

    C = (Q / (2.0 * math.pi * u * sigma_y * sigma_z)) * \
        math.exp(-y ** 2 / (2.0 * sigma_y ** 2)) * \
        (math.exp(-(z - H) ** 2 / (2.0 * sigma_z ** 2)) +
         math.exp(-(z + H) ** 2 / (2.0 * sigma_z ** 2)))
    return {
        "backend": "analytic", "note": "",
        "concentration_g_m3": C, "x": x, "y": y, "z": z,
    }


def pasquill_stability_class(
    wind_speed: float = 4.0,
    insolation: str = "moderate",
) -> dict[str, Any]:
    """Classify atmospheric stability using Pasquill-Gifford scheme.

    Args:
        wind_speed: Mean wind speed at 10 m height (m/s).
        insolation: Solar insolation level: 'strong', 'moderate', 'slight',
            'overcast', 'night_thin_cloud', 'night_clear'.

    Returns:
        Dict with keys: stability_class, description, wind_speed, insolation,
        backend, note.
    """
    table = {
        ("strong",           0.5): "A", ("strong",      1.0): "A",
        ("strong",           2.0): "B", ("moderate",    1.0): "B",
        ("moderate",         2.0): "B", ("moderate",    4.0): "C",
        ("slight",           2.0): "C", ("slight",      4.0): "D",
        ("overcast",         4.0): "D",
        ("night_thin_cloud", 4.0): "E",
        ("night_clear",      2.0): "F", ("night_clear", 3.0): "F",
    }
    descriptions = {
        "A": "very_unstable", "B": "moderately_unstable",
        "C": "slightly_unstable", "D": "neutral",
        "E": "slightly_stable", "F": "moderately_stable",
    }
    # Simplified classification
    if insolation in ("strong",):
        cls = "A" if wind_speed < 2 else ("B" if wind_speed < 5 else "C")
    elif insolation in ("moderate",):
        cls = "B" if wind_speed < 3 else ("C" if wind_speed < 5 else "D")
    elif insolation in ("slight",):
        cls = "C" if wind_speed < 4 else "D"
    elif insolation == "overcast":
        cls = "D"
    elif insolation == "night_thin_cloud":
        cls = "E" if wind_speed < 4 else "D"
    else:  # night_clear
        cls = "F" if wind_speed < 3 else "E"

    return {
        "backend": "analytic", "note": "",
        "stability_class": cls,
        "description": descriptions[cls],
        "wind_speed": wind_speed, "insolation": insolation,
    }


def sigma_from_stability(
    stability_class: str = "C",
    x_m: float = 1000.0,
) -> dict[str, Any]:
    """Compute sigma_y and sigma_z from Pasquill-Gifford stability class.

    Uses Briggs (1973) rural open-country equations.

    Args:
        stability_class: Pasquill class A-F.
        x_m: Downwind distance (m).

    Returns:
        Dict with keys: sigma_y_m, sigma_z_m, stability_class, x_m, backend, note.
    """
    # Briggs coefficients {class: (a_y, b_y, a_z, b_z)}
    params = {
        "A": (0.22, 0.0001, 0.20, 0.0),
        "B": (0.16, 0.0001, 0.12, 0.0),
        "C": (0.11, 0.0001, 0.08, 0.0002),
        "D": (0.08, 0.0001, 0.06, 0.0015),
        "E": (0.06, 0.0001, 0.03, 0.0003),
        "F": (0.04, 0.0001, 0.016, 0.0003),
    }
    cls = stability_class.upper()
    if cls not in params:
        cls = "D"
    a_y, b_y, a_z, b_z = params[cls]
    sigma_y = a_y * x_m * (1 + b_y * x_m) ** (-0.5)
    sigma_z = a_z * x_m * (1 + b_z * x_m) ** (-0.5) + 1e-6
    return {
        "backend": "analytic", "note": "",
        "sigma_y_m": sigma_y, "sigma_z_m": sigma_z,
        "stability_class": cls, "x_m": x_m,
    }


def urban_heat_island(
    T_rural_C: float = 18.0,
    building_density: float = 0.4,
    albedo: float = 0.15,
    vegetation_fraction: float = 0.10,
) -> dict[str, Any]:
    """Estimate urban heat island intensity.

    UHI intensity ~ 7.2 * log10(population_density) but here parameterised
    through building density and surface properties.

    Args:
        T_rural_C: Rural reference temperature (°C).
        building_density: Fraction of urban area covered by buildings (0-1).
        albedo: Urban surface albedo.
        vegetation_fraction: Fraction of urban area with vegetation (0-1).

    Returns:
        Dict with keys: T_urban_C, UHI_intensity_K, backend, note.
    """
    # Empirical scaling
    UHI = (6.5 * building_density
           - 3.0 * vegetation_fraction
           + 2.0 * (0.30 - albedo)
           + 0.5)
    UHI = max(UHI, 0.0)
    T_urban = T_rural_C + UHI
    return {
        "backend": "stub",
        "note": "Install scipy for numerical urban energy-balance model.",
        "T_urban_C": T_urban, "UHI_intensity_K": UHI,
    }


def noise_propagation(
    Lw_dB: float = 90.0,
    r_m: float = 100.0,
    absorption_dB_m: float = 0.003,
) -> dict[str, Any]:
    """Propagate sound pressure level from a point source.

    Lp = Lw - 20*log10(r) - 11 - alpha*r   (free-field spherical spreading)

    Args:
        Lw_dB: Source sound power level (dB re 1 pW).
        r_m: Receiver distance (m).
        absorption_dB_m: Air absorption coefficient (dB/m).

    Returns:
        Dict with keys: Lp_dB, r_m, spreading_loss_dB, absorption_loss_dB,
        backend, note.
    """
    spreading_loss = 20.0 * math.log10(r_m) + 11.0
    absorption_loss = absorption_dB_m * r_m
    Lp = Lw_dB - spreading_loss - absorption_loss
    return {
        "backend": "analytic", "note": "",
        "Lp_dB": Lp, "r_m": r_m,
        "spreading_loss_dB": spreading_loss,
        "absorption_loss_dB": absorption_loss,
    }
