"""Nuclear fusion plasma physics calculations.

Covers the Lawson criterion, DT fusion power density, plasma beta, Debye
length, plasma frequency, ignition temperature, and tokamak safety factor q.

Optional dependencies: scipy, numpy (always required).
All functions are analytic.
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

_KB  = 1.380649e-23    # Boltzmann constant J/K
_EV  = 1.602176634e-19  # J per eV
_MU0 = 1.2566370614e-6  # H/m
_ME  = 9.1093837015e-31  # kg
_EPS0 = 8.8541878e-12   # F/m


def lawson_criterion(
    n_m3: float = 1e20,
    T_keV: float = 10.0,
    tau_E_s: float = 3.0,
) -> dict[str, Any]:
    """Evaluate whether the Lawson criterion for DT fusion break-even is met.

    n * tau_E > Q_Lawson / (sigma_v * E_alpha)
    Simplified: n * T * tau_E > 3e21 keV*m^{-3}*s for DT

    Args:
        n_m3: Plasma density (m^{-3}).
        T_keV: Ion temperature (keV).
        tau_E_s: Energy confinement time (s).

    Returns:
        Dict with keys: n_tau_T, threshold_keV_m3_s, break_even, Q_estimate,
        backend, note.
    """
    nTtau = n_m3 * T_keV * tau_E_s  # keV m^{-3} s
    threshold = 3e21  # keV m^{-3} s for DT at T ~ 10-20 keV
    break_even = nTtau >= threshold
    Q_est = nTtau / threshold if threshold > 0 else 0.0
    return {
        "backend": "analytic", "note": "",
        "n_tau_T": nTtau, "threshold_keV_m3_s": threshold,
        "break_even": break_even, "Q_estimate": Q_est,
    }


def fusion_power_density(
    n_D: float = 5e19,
    n_T: float = 5e19,
    T_keV: float = 20.0,
    reaction: str = "DT",
) -> dict[str, Any]:
    """Compute volumetric fusion power density.

    P_fus = n_D * n_T * <sigma*v>(T) * E_fusion

    Args:
        n_D: Deuterium density (m^{-3}).
        n_T: Tritium density (m^{-3}).
        T_keV: Ion temperature (keV).
        reaction: Fusion reaction: 'DT', 'DD', 'DHe3'.

    Returns:
        Dict with keys: power_density_W_m3, sigma_v_m3_s, E_fusion_J, backend, note.
    """
    # Bosch-Hale parametric fit for sigma*v (m^3/s)
    if reaction == "DT":
        E_MeV = 17.59
        # Parametric approximation valid 1-100 keV
        T = T_keV
        sigma_v = 3.68e-18 / (T ** (2.0 / 3.0)) * math.exp(-19.94 / T ** (1.0 / 3.0))
    elif reaction == "DD":
        E_MeV = 3.65
        T = T_keV
        sigma_v = 5.5e-21 / (T ** (2.0 / 3.0)) * math.exp(-18.33 / T ** (1.0 / 3.0))
    else:  # DHe3
        E_MeV = 18.35
        T = T_keV
        sigma_v = 5.6e-21 * T ** 1.8 / (1 + T / 65.0) ** 3.2
        sigma_v = max(sigma_v, 1e-30)

    E_J = E_MeV * 1e6 * _EV
    P = n_D * n_T * sigma_v * E_J
    return {
        "backend": "analytic", "note": "",
        "power_density_W_m3": P, "sigma_v_m3_s": sigma_v,
        "E_fusion_J": E_J, "reaction": reaction,
    }


def plasma_beta(
    n_m3: float = 1e20,
    T_eV: float = 1e4,
    B_T: float = 5.0,
) -> dict[str, Any]:
    """Compute plasma beta: ratio of kinetic to magnetic pressure.

    beta = 2 * mu0 * n * k_B * T / B^2

    Args:
        n_m3: Plasma density (m^{-3}).
        T_eV: Electron temperature (eV).
        B_T: Magnetic field (T).

    Returns:
        Dict with keys: beta, p_kinetic_Pa, p_magnetic_Pa, backend, note.
    """
    p_kin = n_m3 * T_eV * _EV  # Pa
    p_mag = B_T ** 2 / (2.0 * _MU0)
    beta = p_kin / p_mag
    return {
        "backend": "analytic", "note": "",
        "beta": beta, "p_kinetic_Pa": p_kin, "p_magnetic_Pa": p_mag,
    }


def debye_length(
    n_m3: float = 1e20,
    T_eV: float = 1e4,
) -> dict[str, Any]:
    """Compute the electron Debye screening length.

    lambda_D = sqrt(eps0 * k_B * T / (n * e^2))

    Args:
        n_m3: Electron density (m^{-3}).
        T_eV: Electron temperature (eV).

    Returns:
        Dict with keys: lambda_D_m, lambda_D_mm, plasma_parameter, backend, note.
    """
    T_J = T_eV * _EV
    lD = math.sqrt(_EPS0 * T_J / (n_m3 * _EV ** 2))
    Lambda = n_m3 * (4.0 / 3.0) * math.pi * lD ** 3
    return {
        "backend": "analytic", "note": "",
        "lambda_D_m": lD, "lambda_D_mm": lD * 1e3,
        "plasma_parameter": Lambda,
    }


def plasma_frequency(
    n_m3: float = 1e20,
    m_kg: float = _ME,
) -> dict[str, Any]:
    """Compute the electron (or ion) plasma frequency.

    omega_pe = sqrt(n * e^2 / (eps0 * m))

    Args:
        n_m3: Particle density (m^{-3}).
        m_kg: Particle mass (kg).

    Returns:
        Dict with keys: omega_rad_s, freq_Hz, backend, note.
    """
    e = _EV
    omega = math.sqrt(n_m3 * e ** 2 / (_EPS0 * m_kg))
    return {
        "backend": "analytic", "note": "",
        "omega_rad_s": omega, "freq_Hz": omega / (2.0 * math.pi),
    }


def ignition_temperature(reaction: str = "DT") -> dict[str, Any]:
    """Return the approximate ignition temperature for a fusion reaction.

    Args:
        reaction: 'DT', 'DD', or 'DHe3'.

    Returns:
        Dict with keys: ignition_temperature_keV, reaction, backend, note.
    """
    temps = {"DT": 4.3, "DD": 35.0, "DHe3": 50.0}
    T_ign = temps.get(reaction, 10.0)
    return {
        "backend": "analytic", "note": "",
        "ignition_temperature_keV": T_ign, "reaction": reaction,
    }


def q_factor(
    R_m: float = 6.2,
    a_m: float = 2.0,
    B_T: float = 5.3,
    I_MA: float = 15.0,
) -> dict[str, Any]:
    """Compute the tokamak safety factor q.

    q ≈ 2 * pi * a^2 * B_T / (mu0 * R * I_p)

    Args:
        R_m: Major radius (m).
        a_m: Minor radius (m).
        B_T: Toroidal field (T).
        I_MA: Plasma current (MA).

    Returns:
        Dict with keys: q95_estimate, kink_stable, backend, note.
    """
    I_A = I_MA * 1e6
    q = 2.0 * math.pi * a_m ** 2 * B_T / (_MU0 * R_m * I_A)
    return {
        "backend": "analytic", "note": "",
        "q95_estimate": q, "kink_stable": q > 2.0,
        "R_m": R_m, "a_m": a_m,
    }
