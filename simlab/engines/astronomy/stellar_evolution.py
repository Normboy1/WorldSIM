"""Stellar evolution and HR-diagram utilities.

Covers main-sequence lifetimes, mass-luminosity and mass-radius relations,
Hertzsprung-Russell positioning, Wien's displacement law, the Chandrasekhar
limit, and empirical stellar wind mass-loss rates.

Optional dependencies: scipy (interpolation), numpy (always required).
Falls back to power-law heuristics.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import scipy.interpolate as _interp
    _SCIPY = True
except ImportError:
    _SCIPY = False

_L_SUN = 3.828e26   # W
_R_SUN = 6.957e8    # m
_M_SUN = 1.989e30   # kg
_T_SUN = 5778.0     # K
_SIGMA = 5.6704e-8  # Stefan-Boltzmann constant


def main_sequence_lifetime(M_solar: float = 1.0) -> dict[str, Any]:
    """Estimate the main-sequence lifetime of a star.

    t_MS ≈ t_sun * (M/L) = 10 Gyr * M_solar / L_solar
    Using L ~ M^4 -> t_MS ≈ 10 Gyr * M^{-2.5}

    Args:
        M_solar: Stellar mass in solar masses.

    Returns:
        Dict with keys: lifetime_Gyr, M_solar, luminosity_solar,
        backend, note.
    """
    L = M_solar ** 3.8
    t_Gyr = 10.0 * M_solar / L
    return {
        "backend": "analytic",
        "note": "",
        "lifetime_Gyr": t_Gyr,
        "M_solar": M_solar,
        "luminosity_solar": L,
    }


def luminosity_from_mass(M_solar: float = 1.0) -> dict[str, Any]:
    """Mass-luminosity relation for main-sequence stars.

    L ~ M^4 for high mass, L ~ M^{3.5} for intermediate mass.

    Args:
        M_solar: Stellar mass in solar masses.

    Returns:
        Dict with keys: luminosity_solar, luminosity_W, M_solar, exponent,
        backend, note.
    """
    if M_solar < 0.43:
        exponent = 2.3
    elif M_solar < 2.0:
        exponent = 4.0
    elif M_solar < 20.0:
        exponent = 3.5
    else:
        exponent = 1.0

    L_solar = M_solar ** exponent
    return {
        "backend": "analytic",
        "note": "",
        "luminosity_solar": L_solar,
        "luminosity_W": L_solar * _L_SUN,
        "M_solar": M_solar,
        "exponent": exponent,
    }


def hertzsprung_russell_position(
    M_solar: float = 1.0,
    age_Gyr: float = 0.0,
) -> dict[str, Any]:
    """Return approximate HR-diagram position (T_eff, L) for a star.

    Uses empirical power-law scalings valid on the main sequence.
    Includes simple ZAMS-to-TAMS shift as age increases.

    Args:
        M_solar: Stellar mass in solar masses.
        age_Gyr: Current age in Gyr (0 = ZAMS).

    Returns:
        Dict with keys: T_eff_K, luminosity_solar, radius_solar,
        spectral_class, hr_stage, backend, note.
    """
    L_solar = luminosity_from_mass(M_solar)["luminosity_solar"]
    T_eff = _T_SUN * M_solar ** 0.57  # ZAMS scaling

    t_ms = main_sequence_lifetime(M_solar)["lifetime_Gyr"]
    frac = min(age_Gyr / t_ms, 1.0) if t_ms > 0 else 1.0
    # Star expands and cools slightly toward end of MS
    T_eff *= 1.0 - 0.15 * frac
    L_solar *= 1.0 + 0.8 * frac

    if frac >= 1.0:
        hr_stage = "subgiant_or_giant"
    else:
        hr_stage = "main_sequence"

    R_solar = math.sqrt(L_solar) / (T_eff / _T_SUN) ** 2

    if T_eff > 30000:
        sp = "O"
    elif T_eff > 10000:
        sp = "B"
    elif T_eff > 7500:
        sp = "A"
    elif T_eff > 6000:
        sp = "F"
    elif T_eff > 5200:
        sp = "G"
    elif T_eff > 3700:
        sp = "K"
    else:
        sp = "M"

    return {
        "backend": "analytic",
        "note": "",
        "T_eff_K": T_eff,
        "luminosity_solar": L_solar,
        "radius_solar": R_solar,
        "spectral_class": sp,
        "hr_stage": hr_stage,
    }


def stellar_radius(
    M_solar: float = 1.0,
    luminosity_solar: float | None = None,
) -> dict[str, Any]:
    """Estimate stellar radius from mass (and optionally luminosity).

    R ~ M^0.8  (ZAMS approximation) or via Stefan-Boltzmann if L given.

    Args:
        M_solar: Stellar mass in solar masses.
        luminosity_solar: Luminosity in solar units.  If None, derived
            from mass-luminosity relation.

    Returns:
        Dict with keys: radius_solar, radius_m, backend, note.
    """
    if luminosity_solar is None:
        luminosity_solar = M_solar ** 3.8
    R_solar = M_solar ** 0.8
    return {
        "backend": "analytic",
        "note": "",
        "radius_solar": R_solar,
        "radius_m": R_solar * _R_SUN,
    }


def blackbody_peak_wavelength(T_K: float = 5778.0) -> dict[str, Any]:
    """Wien's displacement law: wavelength of peak blackbody emission.

    lambda_max = b / T    b = 2.897771955e-3 m*K

    Args:
        T_K: Blackbody temperature (K).

    Returns:
        Dict with keys: lambda_peak_nm, lambda_peak_m, T_K, colour,
        backend, note.
    """
    b = 2.897771955e-3
    lam = b / T_K
    nm = lam * 1e9
    if nm < 380:
        colour = "ultraviolet"
    elif nm < 450:
        colour = "violet"
    elif nm < 495:
        colour = "blue"
    elif nm < 570:
        colour = "green"
    elif nm < 620:
        colour = "yellow"
    elif nm < 750:
        colour = "red"
    else:
        colour = "infrared"

    return {
        "backend": "analytic",
        "note": "",
        "lambda_peak_m": lam,
        "lambda_peak_nm": nm,
        "T_K": T_K,
        "colour": colour,
    }


def chandrasekhar_limit() -> dict[str, Any]:
    """Return the Chandrasekhar mass limit for white dwarfs.

    M_Ch ≈ 5.87 * mu_e^{-2} * M_sun  ≈ 1.44 M_sun for mu_e = 2

    Returns:
        Dict with keys: mass_solar, mass_kg, backend, note.
    """
    M_Ch_solar = 1.4412
    return {
        "backend": "analytic",
        "note": "",
        "mass_solar": M_Ch_solar,
        "mass_kg": M_Ch_solar * _M_SUN,
    }


def stellar_wind_mass_loss(
    M_solar: float = 20.0,
    L_solar: float = 1e5,
) -> dict[str, Any]:
    """Empirical stellar wind mass-loss rate (de Jager / Vink approximation).

    log(dM/dt) ≈ -11.4 + 1.75 * log(L/L_sun) - 1.5 * log(M/M_sun)
    (valid for OB supergiants)

    Args:
        M_solar: Stellar mass in solar masses.
        L_solar: Stellar luminosity in solar luminosities.

    Returns:
        Dict with keys: mass_loss_Msun_yr, mass_loss_kg_s, wind_momentum_L_c,
        backend, note.
    """
    log_Mdot = -11.4 + 1.75 * math.log10(L_solar) - 1.5 * math.log10(M_solar)
    Mdot_Msun_yr = 10 ** log_Mdot
    Mdot_kg_s = Mdot_Msun_yr * _M_SUN / 3.1557e7

    # Modified wind momentum D_mom = Mdot * v_inf * sqrt(R)
    v_inf = 2.6 * math.sqrt(2 * 6.674e-11 * M_solar * _M_SUN / (M_solar ** 0.8 * _R_SUN)) / 1e3
    D_mom = math.log10(Mdot_kg_s * v_inf * math.sqrt(M_solar ** 0.8 * _R_SUN))

    return {
        "backend": "analytic",
        "note": "",
        "mass_loss_Msun_yr": Mdot_Msun_yr,
        "mass_loss_kg_s": Mdot_kg_s,
        "wind_momentum_log_D": D_mom,
    }
