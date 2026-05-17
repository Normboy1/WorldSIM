"""Cosmological distance and time calculations (FLRW metric).

Provides Hubble distance, lookback time, comoving distance, age of the
universe, critical density, CMB temperature at redshift z, and a dark energy
equation-of-state parameterisation.

Optional dependencies: scipy (numerical integration of Hubble parameter).
Falls back to fitting-formula approximations valid for flat LCDM.
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

_C_KM_S = 2.99792458e5   # speed of light in km/s
_C_M_S  = 2.99792458e8   # m/s
_H0_DEFAULT = 67.4        # km/s/Mpc (Planck 2018)


def _E(z: float, Omega_m: float, Omega_Lambda: float) -> float:
    """Dimensionless Hubble parameter E(z) for flat LCDM."""
    Omega_k = 1.0 - Omega_m - Omega_Lambda
    return math.sqrt(Omega_m * (1 + z) ** 3 + Omega_k * (1 + z) ** 2 + Omega_Lambda)


def hubble_distance(
    z: float = 1.0,
    H0: float = _H0_DEFAULT,
    Omega_m: float = 0.315,
    Omega_Lambda: float = 0.685,
) -> dict[str, Any]:
    """Compute the Hubble distance D_H = c / H0 and related scales.

    Args:
        z: Redshift (informational; D_H is independent of z).
        H0: Hubble constant (km/s/Mpc).
        Omega_m: Matter density parameter.
        Omega_Lambda: Dark energy density parameter.

    Returns:
        Dict with keys: D_H_Mpc, D_H_Glyr, H0, backend, note.
    """
    D_H_Mpc = _C_KM_S / H0
    D_H_Glyr = D_H_Mpc * 3.2615638  # 1 Mpc = 3.2615638 Mly
    return {
        "backend": "analytic",
        "note": "",
        "D_H_Mpc": D_H_Mpc,
        "D_H_Glyr": D_H_Glyr / 1e3,
        "H0": H0,
    }


def lookback_time(
    z: float = 1.0,
    H0: float = _H0_DEFAULT,
    Omega_m: float = 0.315,
    Omega_Lambda: float = 0.685,
) -> dict[str, Any]:
    """Compute the lookback time to redshift z.

    t_L = (1/H0) * integral_0^z dz' / [(1+z') E(z')]

    Args:
        z: Redshift.
        H0: Hubble constant (km/s/Mpc).
        Omega_m: Matter density parameter.
        Omega_Lambda: Dark energy density parameter.

    Returns:
        Dict with keys: lookback_time_Gyr, z, backend, note.
    """
    H0_s = H0 * 1e3 / 3.0857e22  # convert to 1/s
    t_H = 1.0 / H0_s / 3.1558e16  # Hubble time in Gyr

    if _SCIPY:
        integrand = lambda zp: 1.0 / ((1 + zp) * _E(zp, Omega_m, Omega_Lambda))
        val, _ = _quad(integrand, 0, z)
        t_L = t_H * val
        backend, note = "scipy", ""
    else:
        # Pen (1999) fitting formula for flat LCDM
        Om = Omega_m
        frac = Om ** (4.0 / 5.0) + (Omega_Lambda / 70.0) * (
            1.0 + Om / 2.0)
        t_L = t_H * (2.0 / 3.0) * (1.0 - Om) ** (-0.5) * math.log(
            ((1.0 - Om) / Om) ** 0.5 + ((1.0 - Om) / Om + 1.0 / (1 + z) ** 3) ** 0.5
        ) - t_H * (2.0 / 3.0) / math.sqrt(
            (1.0 - Om) / Om + 1.0 / (1 + z) ** 3) * (1.0 + z) ** (-1.5)
        t_L = abs(t_L)
        backend = "stub"
        note = "Install scipy for accurate numerical integration."

    return {
        "backend": backend, "note": note,
        "lookback_time_Gyr": t_L, "z": z,
    }


def comoving_distance(
    z: float = 1.0,
    H0: float = _H0_DEFAULT,
    Omega_m: float = 0.315,
    Omega_Lambda: float = 0.685,
) -> dict[str, Any]:
    """Comoving line-of-sight distance to redshift z.

    D_C = D_H * integral_0^z dz' / E(z')

    Args:
        z: Redshift.
        H0: Hubble constant (km/s/Mpc).
        Omega_m: Matter density parameter.
        Omega_Lambda: Dark energy density parameter.

    Returns:
        Dict with keys: comoving_distance_Mpc, luminosity_distance_Mpc,
        angular_diameter_distance_Mpc, z, backend, note.
    """
    D_H = _C_KM_S / H0  # Mpc

    if _SCIPY:
        integrand = lambda zp: 1.0 / _E(zp, Omega_m, Omega_Lambda)
        val, _ = _quad(integrand, 0, z)
        D_C = D_H * val
        backend, note = "scipy", ""
    else:
        # Pen fitting formula
        eta = lambda a: 2.0 * math.sqrt(1 / a + Omega_m / (1 - Omega_m)) * \
            math.sqrt(1 + Omega_m * (1 / a - 1))
        D_C = D_H * (eta(1.0) - eta(1.0 / (1 + z))) * (1 - Omega_m) ** (-0.5)
        backend = "stub"
        note = "Install scipy for accurate comoving distance integration."

    D_L = D_C * (1 + z)
    D_A = D_C / (1 + z)
    return {
        "backend": backend, "note": note,
        "comoving_distance_Mpc": D_C,
        "luminosity_distance_Mpc": D_L,
        "angular_diameter_distance_Mpc": D_A,
        "z": z,
    }


def age_of_universe(
    H0: float = _H0_DEFAULT,
    Omega_m: float = 0.315,
    Omega_Lambda: float = 0.685,
) -> dict[str, Any]:
    """Estimate the age of the universe for given cosmological parameters.

    Args:
        H0: Hubble constant (km/s/Mpc).
        Omega_m: Matter density parameter.
        Omega_Lambda: Dark energy density parameter.

    Returns:
        Dict with keys: age_Gyr, H0, Omega_m, Omega_Lambda, backend, note.
    """
    H0_s = H0 * 1e3 / 3.0857e22
    t_H = 1.0 / H0_s / 3.1558e16

    if _SCIPY:
        integrand = lambda zp: 1.0 / ((1 + zp) * _E(zp, Omega_m, Omega_Lambda))
        val, _ = _quad(integrand, 0, 1e4)
        age = t_H * val
        backend, note = "scipy", ""
    else:
        # Turner (2002) approximation for flat LCDM
        age = t_H * (2.0 / 3.0) / math.sqrt(Omega_Lambda) * math.log(
            (math.sqrt(Omega_Lambda / Omega_m) + math.sqrt(Omega_Lambda / Omega_m + 1.0)))
        backend = "stub"
        note = "Install scipy for accurate age integration."

    return {
        "backend": backend, "note": note,
        "age_Gyr": age, "H0": H0, "Omega_m": Omega_m, "Omega_Lambda": Omega_Lambda,
    }


def critical_density(H0: float = _H0_DEFAULT) -> dict[str, Any]:
    """Compute the critical density of the universe.

    rho_c = 3 H0^2 / (8 pi G)

    Args:
        H0: Hubble constant (km/s/Mpc).

    Returns:
        Dict with keys: rho_c_kg_m3, rho_c_g_cm3, H0, backend, note.
    """
    H0_si = H0 * 1e3 / 3.0857e22
    rho_c = 3.0 * H0_si ** 2 / (8.0 * math.pi * 6.674e-11)
    return {
        "backend": "analytic", "note": "",
        "rho_c_kg_m3": rho_c,
        "rho_c_g_cm3": rho_c * 1e-3,
        "H0": H0,
    }


def cmb_temperature(z: float = 0.0) -> dict[str, Any]:
    """CMB temperature as a function of redshift.

    T(z) = T0 * (1 + z)   T0 = 2.7255 K

    Args:
        z: Redshift.

    Returns:
        Dict with keys: T_CMB_K, z, T0_K, backend, note.
    """
    T0 = 2.7255
    T = T0 * (1.0 + z)
    return {
        "backend": "analytic", "note": "",
        "T_CMB_K": T, "z": z, "T0_K": T0,
    }


def dark_energy_equation_of_state(
    w0: float = -1.0,
    wa: float = 0.0,
    z: float = 0.5,
) -> dict[str, Any]:
    """Chevallier-Polarski-Linder dark energy equation-of-state w(z).

    w(z) = w0 + wa * z / (1 + z)

    Args:
        w0: EOS value at z=0.
        wa: EOS derivative parameter.
        z: Redshift.

    Returns:
        Dict with keys: w_z, w0, wa, z, Omega_DE_factor, backend, note.
    """
    w_z = w0 + wa * z / (1.0 + z)
    a = 1.0 / (1.0 + z)
    # Dark energy density factor f(a) = exp(-3 * int_1^a (1+w)/a' da')
    f = a ** (-3.0 * (1.0 + w0 + wa)) * math.exp(-3.0 * wa * (1.0 - a))
    return {
        "backend": "analytic", "note": "",
        "w_z": w_z, "w0": w0, "wa": wa, "z": z,
        "Omega_DE_factor": f,
    }
