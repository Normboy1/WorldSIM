"""Physical oceanography simulation utilities.

Covers thermohaline circulation density driving, mixed-layer depth, wave
dispersion, tidal forcing proxy, Ekman transport, and coastal upwelling.

Optional dependencies: scipy (ODE / special functions), numpy (always required).
Falls back to analytic approximations.
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

_G = 9.81  # m/s^2
_RHO0 = 1025.0  # reference seawater density kg/m^3


def thermohaline_circulation(
    T_surface: float = 25.0,
    S_surface: float = 36.0,
    T_deep: float = 2.0,
    S_deep: float = 34.7,
) -> dict[str, Any]:
    """Estimate density difference driving thermohaline overturning circulation.

    Uses linearised equation of state:
    rho = rho0 * (1 - alpha*(T-T0) + beta*(S-S0))

    Args:
        T_surface: Surface temperature (°C).
        S_surface: Surface salinity (PSU).
        T_deep: Deep water temperature (°C).
        S_deep: Deep water salinity (PSU).

    Returns:
        Dict with keys: delta_rho_kg_m3, overturning_tendency, backend, note.
    """
    alpha_T = 2e-4  # thermal expansion coefficient (1/K)
    beta_S = 7.4e-4  # haline contraction coefficient (1/PSU)
    T0, S0 = 10.0, 35.0

    rho_surf = _RHO0 * (1 - alpha_T * (T_surface - T0) + beta_S * (S_surface - S0))
    rho_deep = _RHO0 * (1 - alpha_T * (T_deep - T0) + beta_S * (S_deep - S0))
    delta_rho = rho_deep - rho_surf

    if delta_rho > 0.5:
        tendency = "strong_overturning"
    elif delta_rho > 0.1:
        tendency = "moderate_overturning"
    elif delta_rho > 0:
        tendency = "weak_overturning"
    else:
        tendency = "stable_stratification_inhibits_overturning"

    return {
        "backend": "analytic", "note": "",
        "delta_rho_kg_m3": delta_rho,
        "rho_surface_kg_m3": rho_surf,
        "rho_deep_kg_m3": rho_deep,
        "overturning_tendency": tendency,
    }


def mixed_layer_depth(
    wind_stress: float = 0.1,
    buoyancy_flux: float = 1e-7,
    f: float = 1e-4,
) -> dict[str, Any]:
    """Estimate the ocean mixed layer depth from forcing.

    Uses the Monin-Obukhov scaling:
    h ~ u*^3 / (kappa * B)    or   h ~ u* / f  (Ekman depth)

    Args:
        wind_stress: Wind stress magnitude (N/m^2).
        buoyancy_flux: Surface buoyancy flux (m^2/s^3, positive = destabilising).
        f: Coriolis parameter (1/s).

    Returns:
        Dict with keys: MLD_m, u_star_m_s, Monin_Obukhov_L_m, backend, note.
    """
    u_star = math.sqrt(abs(wind_stress) / _RHO0)
    kappa = 0.41
    if buoyancy_flux > 1e-12:
        L = -u_star ** 3 / (kappa * buoyancy_flux)
        MLD = min(abs(L) * 0.4, 200.0)
    else:
        MLD = 0.4 * u_star / abs(f)

    return {
        "backend": "analytic", "note": "",
        "MLD_m": MLD, "u_star_m_s": u_star, "Monin_Obukhov_L_m": abs(L) if buoyancy_flux > 1e-12 else None,
    }


def wave_dispersion(
    k: float = 0.1,
    h: float = 50.0,
) -> dict[str, Any]:
    """Ocean wave dispersion relation (linear water waves).

    omega^2 = g * k * tanh(k * h)

    Args:
        k: Wavenumber (rad/m).
        h: Water depth (m).

    Returns:
        Dict with keys: omega_rad_s, frequency_Hz, period_s, phase_speed_m_s,
        group_speed_m_s, wavelength_m, backend, note.
    """
    omega = math.sqrt(_G * k * math.tanh(k * h))
    f_Hz = omega / (2 * math.pi)
    T = 1.0 / f_Hz if f_Hz > 0 else float("inf")
    c_p = omega / k
    # Group velocity: c_g = d(omega)/dk
    kh = k * h
    N = math.tanh(kh) + kh / math.cosh(kh) ** 2
    c_g = 0.5 * math.sqrt(_G / (k * math.tanh(kh))) * N
    return {
        "backend": "analytic", "note": "",
        "omega_rad_s": omega, "frequency_Hz": f_Hz, "period_s": T,
        "phase_speed_m_s": c_p, "group_speed_m_s": c_g,
        "wavelength_m": 2 * math.pi / k,
    }


def tidal_forcing(
    lat_deg: float = 45.0,
    lon_deg: float = -70.0,
    t_days: float = 0.0,
) -> dict[str, Any]:
    """Proxy tidal height at a given location and time using M2 + S2 components.

    Args:
        lat_deg: Latitude (degrees North).
        lon_deg: Longitude (degrees East).
        t_days: Time since reference epoch (days).

    Returns:
        Dict with keys: tidal_height_m, M2_amplitude_m, S2_amplitude_m,
        backend, note.
    """
    # Amplitude proxy from equilibrium tidal theory
    M2_amp = 0.31 * abs(math.cos(math.radians(lat_deg))) ** 2
    S2_amp = 0.14 * abs(math.cos(math.radians(lat_deg))) ** 2

    t_s = t_days * 86400.0
    omega_M2 = 2.0 * math.pi / 44712.0  # M2 period = 12.42 hr
    omega_S2 = 2.0 * math.pi / 43200.0  # S2 period = 12.00 hr
    phase_lon = math.radians(lon_deg) / math.pi

    h = (M2_amp * math.cos(omega_M2 * t_s + phase_lon) +
         S2_amp * math.cos(omega_S2 * t_s + phase_lon))

    return {
        "backend": "stub",
        "note": "Install scipy for full harmonic tidal constituents.",
        "tidal_height_m": h, "M2_amplitude_m": M2_amp, "S2_amplitude_m": S2_amp,
    }


def ekman_transport(
    wind_stress_x: float = 0.1,
    wind_stress_y: float = 0.0,
    f: float = 1e-4,
    rho: float = 1025.0,
) -> dict[str, Any]:
    """Compute the Ekman transport (volume flux per unit width).

    U_ek = -tau_y / (rho * f)
    V_ek =  tau_x / (rho * f)

    Args:
        wind_stress_x: Zonal wind stress (N/m^2).
        wind_stress_y: Meridional wind stress (N/m^2).
        f: Coriolis parameter (1/s).
        rho: Seawater density (kg/m^3).

    Returns:
        Dict with keys: U_ekman_m2_s, V_ekman_m2_s, transport_magnitude_m2_s,
        backend, note.
    """
    U_ek = -wind_stress_y / (rho * f)
    V_ek = wind_stress_x / (rho * f)
    mag = math.sqrt(U_ek ** 2 + V_ek ** 2)
    return {
        "backend": "analytic", "note": "",
        "U_ekman_m2_s": U_ek, "V_ekman_m2_s": V_ek,
        "transport_magnitude_m2_s": mag,
    }


def upwelling_velocity(
    curl_tau: float = 1e-7,
    f: float = 1e-4,
    rho: float = 1025.0,
) -> dict[str, Any]:
    """Estimate coastal/open-ocean upwelling velocity from wind stress curl.

    w_Ek = curl(tau) / (rho * f)   (Sverdrup balance)

    Args:
        curl_tau: Wind stress curl (N/m^3).
        f: Coriolis parameter (1/s).
        rho: Seawater density (kg/m^3).

    Returns:
        Dict with keys: w_upwelling_m_s, w_upwelling_m_day, backend, note.
    """
    w = curl_tau / (rho * f)
    return {
        "backend": "analytic", "note": "",
        "w_upwelling_m_s": w,
        "w_upwelling_m_day": w * 86400.0,
    }
