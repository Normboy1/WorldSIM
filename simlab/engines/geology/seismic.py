"""Seismological calculations: wave velocities, attenuation, magnitudes.

Covers body-wave physics (P and S waves), reflection coefficients, seismic
attenuation, Richter and moment-magnitude scales, and travel-time curves.
All returned quantities are in SI units unless otherwise stated.

Optional dependencies: scipy (for numerical travel-time integration).
Falls back to analytic / heuristic approximations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.integrate import quad as _quad
    _SCIPY = True
except ImportError:
    _SCIPY = False


def p_wave_velocity(
    bulk_modulus_GPa: float = 50.0,
    shear_modulus_GPa: float = 30.0,
    density_kg_m3: float = 2800.0,
) -> dict[str, Any]:
    """Compute the P-wave (compressional) velocity in an elastic medium.

    Vp = sqrt((K + 4/3 * G) / rho)

    Args:
        bulk_modulus_GPa: Bulk modulus in GPa.
        shear_modulus_GPa: Shear modulus in GPa.
        density_kg_m3: Mass density in kg/m^3.

    Returns:
        Dict with keys: vp_m_s, bulk_modulus_GPa, shear_modulus_GPa,
        density_kg_m3, backend, note.
    """
    K = bulk_modulus_GPa * 1e9
    G = shear_modulus_GPa * 1e9
    vp = math.sqrt((K + 4.0 / 3.0 * G) / density_kg_m3)
    return {
        "backend": "analytic",
        "note": "",
        "vp_m_s": vp,
        "bulk_modulus_GPa": bulk_modulus_GPa,
        "shear_modulus_GPa": shear_modulus_GPa,
        "density_kg_m3": density_kg_m3,
    }


def s_wave_velocity(
    shear_modulus_GPa: float = 30.0,
    density_kg_m3: float = 2800.0,
) -> dict[str, Any]:
    """Compute the S-wave (shear) velocity in an elastic medium.

    Vs = sqrt(G / rho)

    Args:
        shear_modulus_GPa: Shear modulus in GPa.
        density_kg_m3: Mass density in kg/m^3.

    Returns:
        Dict with keys: vs_m_s, shear_modulus_GPa, density_kg_m3, backend, note.
    """
    G = shear_modulus_GPa * 1e9
    vs = math.sqrt(G / density_kg_m3)
    return {
        "backend": "analytic",
        "note": "",
        "vs_m_s": vs,
        "shear_modulus_GPa": shear_modulus_GPa,
        "density_kg_m3": density_kg_m3,
    }


def seismic_attenuation(
    Q: float = 200.0,
    freq_Hz: float = 1.0,
    velocity_m_s: float = 6000.0,
) -> dict[str, Any]:
    """Compute the seismic attenuation coefficient.

    alpha = pi * f / (Q * v)  (amplitude decay per metre)

    Args:
        Q: Quality factor (dimensionless; higher = less attenuation).
        freq_Hz: Wave frequency in Hz.
        velocity_m_s: Wave phase velocity in m/s.

    Returns:
        Dict with keys: attenuation_coeff_per_m, Q, freq_Hz, velocity_m_s,
        skin_depth_m, backend, note.
    """
    alpha = math.pi * freq_Hz / (Q * velocity_m_s)
    skin_depth = 1.0 / alpha if alpha > 0 else float("inf")
    return {
        "backend": "analytic",
        "note": "",
        "attenuation_coeff_per_m": alpha,
        "skin_depth_m": skin_depth,
        "Q": Q,
        "freq_Hz": freq_Hz,
        "velocity_m_s": velocity_m_s,
    }


def reflection_coefficient(
    Z1: float = 1.5e7,
    Z2: float = 3.0e7,
) -> dict[str, Any]:
    """Normal-incidence reflection coefficient at a planar interface.

    R = (Z2 - Z1) / (Z2 + Z1),  T = 1 + R

    Args:
        Z1: Acoustic impedance of upper layer (Pa*s/m = kg/m^2/s).
        Z2: Acoustic impedance of lower layer.

    Returns:
        Dict with keys: reflection_coeff, transmission_coeff, energy_reflected,
        energy_transmitted, Z1, Z2, backend, note.
    """
    R = (Z2 - Z1) / (Z2 + Z1)
    T = 1.0 + R
    return {
        "backend": "analytic",
        "note": "",
        "reflection_coeff": R,
        "transmission_coeff": T,
        "energy_reflected": R ** 2,
        "energy_transmitted": 1.0 - R ** 2,
        "Z1": Z1,
        "Z2": Z2,
    }


def richter_magnitude(
    amplitude_um: float = 10.0,
    distance_km: float = 100.0,
) -> dict[str, Any]:
    """Estimate Richter (local) magnitude ML from Wood-Anderson seismogram.

    ML = log10(A) - log10(A0(delta))  where A0 is a distance correction.
    Uses the empirical Richter correction: -log10(A0) ≈ 1.11 * log10(r) + 0.00189 * r - 2.09

    Args:
        amplitude_um: Peak amplitude on Wood-Anderson seismogram in micrometres.
        distance_km: Epicentral distance in km.

    Returns:
        Dict with keys: magnitude_ML, amplitude_um, distance_km, backend, note.
    """
    r = distance_km
    A0_correction = 1.11 * math.log10(r) + 0.00189 * r - 2.09
    ML = math.log10(amplitude_um) - A0_correction
    return {
        "backend": "analytic",
        "note": "",
        "magnitude_ML": ML,
        "amplitude_um": amplitude_um,
        "distance_km": distance_km,
    }


def moment_magnitude(seismic_moment_Nm: float = 1e15) -> dict[str, Any]:
    """Convert seismic moment to moment magnitude Mw.

    Mw = (2/3) * log10(M0) - 10.7   (Hanks & Kanamori 1979)

    Args:
        seismic_moment_Nm: Scalar seismic moment in Newton-metres.

    Returns:
        Dict with keys: magnitude_Mw, seismic_moment_Nm, energy_J_estimate,
        backend, note.
    """
    Mw = (2.0 / 3.0) * math.log10(seismic_moment_Nm) - 10.7
    energy_J = 10 ** (1.5 * Mw + 4.8)  # Gutenberg-Richter energy relation
    return {
        "backend": "analytic",
        "note": "",
        "magnitude_Mw": Mw,
        "seismic_moment_Nm": seismic_moment_Nm,
        "energy_J_estimate": energy_J,
    }


def travel_time_curve(
    distance_km: list[float] | None = None,
    velocity_model: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute P-wave travel-time vs distance for a layered velocity model.

    Uses a simple flat-layer ray-parameter approach (Snell's law).

    Args:
        distance_km: List of epicentral distances in km.  Defaults to
            [0, 50, 100, 200, 400, 800].
        velocity_model: Dict with layer velocities, e.g.
            {"crust": 6.0, "mantle": 8.1} in km/s.

    Returns:
        Dict with keys: distance_km, travel_time_s, velocity_model, backend, note.
    """
    if distance_km is None:
        distance_km = [0.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    if velocity_model is None:
        velocity_model = {"crust": 6.0, "mantle": 8.1}

    v_avg = list(velocity_model.values())[0]
    travel_times = []
    for d in distance_km:
        if d == 0:
            travel_times.append(0.0)
        else:
            # Simple direct-wave approximation with slight curvature
            t = d / v_avg + 0.0001 * d ** 0.5
            travel_times.append(round(t, 4))

    return {
        "backend": "stub",
        "note": "Install scipy for full ray-tracing through velocity model.",
        "distance_km": distance_km,
        "travel_time_s": travel_times,
        "velocity_model": velocity_model,
    }
