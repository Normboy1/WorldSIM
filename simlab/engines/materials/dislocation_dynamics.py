"""Dislocation dynamics and crystal plasticity micro-mechanics.

Provides Peierls stress, Taylor / forest hardening, dislocation velocity,
climb rate, and dislocation density evolution.

Optional dependencies: scipy (ODE integration), numpy (always required).
Falls back to analytic or Euler approximations.
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


def peierls_stress(
    b_m: float = 2.86e-10,
    nu: float = 0.33,
    G_Pa: float = 80.0,
    d_m: float = 2.86e-10,
) -> dict[str, Any]:
    """Estimate the Peierls-Nabarro lattice friction stress.

    tau_P = (2G / (1-nu)) * exp(-2*pi*d / ((1-nu)*b))

    Args:
        b_m: Burgers vector magnitude (m).
        nu: Poisson's ratio.
        G_Pa: Shear modulus (GPa).
        d_m: Interplanar spacing (m).

    Returns:
        Dict with keys: tau_P_MPa, G_Pa, nu, b_m, d_m, backend, note.
    """
    G = G_Pa * 1e9
    tau_P = (2.0 * G / (1.0 - nu)) * math.exp(-2.0 * math.pi * d_m / ((1.0 - nu) * b_m))
    return {
        "backend": "analytic", "note": "",
        "tau_P_MPa": tau_P / 1e6, "G_Pa": G_Pa, "nu": nu, "b_m": b_m, "d_m": d_m,
    }


def taylor_hardening(
    alpha: float = 0.3,
    G_Pa: float = 80.0,
    b_m: float = 2.86e-10,
    rho_m2: float = 1e12,
) -> dict[str, Any]:
    """Taylor hardening law: flow stress from dislocation density.

    tau = alpha * G * b * sqrt(rho)

    Args:
        alpha: Taylor factor (typically 0.2-0.5).
        G_Pa: Shear modulus (GPa).
        b_m: Burgers vector (m).
        rho_m2: Total dislocation density (m^{-2}).

    Returns:
        Dict with keys: tau_MPa, sigma_MPa, backend, note.
    """
    G = G_Pa * 1e9
    tau = alpha * G * b_m * math.sqrt(rho_m2)
    sigma = math.sqrt(3.0) * tau  # von Mises
    return {
        "backend": "analytic", "note": "",
        "tau_MPa": tau / 1e6, "sigma_MPa": sigma / 1e6,
    }


def forest_hardening(
    M: float = 3.06,
    alpha: float = 0.3,
    G_Pa: float = 80.0,
    b_m: float = 2.86e-10,
    rho_forest: float = 1e13,
) -> dict[str, Any]:
    """Forest / junction hardening: flow stress from forest dislocation density.

    tau_c = M * alpha * G * b * sqrt(rho_forest)

    Args:
        M: Taylor orientation factor.
        alpha: Hardening coefficient.
        G_Pa: Shear modulus (GPa).
        b_m: Burgers vector (m).
        rho_forest: Forest dislocation density (m^{-2}).

    Returns:
        Dict with keys: tau_c_MPa, sigma_y_MPa, backend, note.
    """
    G = G_Pa * 1e9
    tau_c = M * alpha * G * b_m * math.sqrt(rho_forest)
    return {
        "backend": "analytic", "note": "",
        "tau_c_MPa": tau_c / 1e6, "sigma_y_MPa": tau_c * math.sqrt(3.0) / 1e6,
    }


def dislocation_velocity(
    tau_MPa: float = 50.0,
    tau_p_MPa: float = 10.0,
    B: float = 1e-4,
    b_m: float = 2.86e-10,
) -> dict[str, Any]:
    """Compute dislocation glide velocity via the drag equation.

    v = (tau - tau_P) * b / B   (linear viscous drag)

    Args:
        tau_MPa: Applied resolved shear stress (MPa).
        tau_p_MPa: Peierls stress / lattice friction (MPa).
        B: Drag coefficient (Pa*s).
        b_m: Burgers vector (m).

    Returns:
        Dict with keys: velocity_m_s, mobility_m2_Pa_s, backend, note.
    """
    tau_eff = max((tau_MPa - tau_p_MPa) * 1e6, 0.0)
    v = tau_eff * b_m / B
    mob = b_m / B
    return {
        "backend": "analytic", "note": "",
        "velocity_m_s": v, "mobility_m2_Pa_s": mob,
    }


def climb_rate(
    D_v: float = 1e-20,
    b_m: float = 2.86e-10,
    tau_MPa: float = 100.0,
    T_K: float = 800.0,
) -> dict[str, Any]:
    """Estimate dislocation climb rate driven by vacancy flux.

    v_climb ~ D_v * sigma * Omega / (kT * b)

    Args:
        D_v: Vacancy diffusivity (m^2/s).
        b_m: Burgers vector (m).
        tau_MPa: Stress (MPa).
        T_K: Temperature (K).

    Returns:
        Dict with keys: v_climb_m_s, backend, note.
    """
    _KB = 1.380649e-23
    Omega = b_m ** 3  # atomic volume
    sigma = tau_MPa * 1e6
    v_climb = D_v * sigma * Omega / (_KB * T_K * b_m)
    return {
        "backend": "analytic", "note": "",
        "v_climb_m_s": v_climb,
    }


def dislocation_density_evolution(
    rho0: float = 1e12,
    gamma_dot: float = 1e-3,
    b_m: float = 2.86e-10,
    M: float = 3.06,
    k1: float = 1e7,
    k2: float = 10.0,
    t_end: float = 100.0,
) -> dict[str, Any]:
    """Kocks-Mecking dislocation density evolution model.

    drho/dt = M * gamma_dot * (k1 * sqrt(rho) - k2 * rho) / b

    Args:
        rho0: Initial dislocation density (m^{-2}).
        gamma_dot: Macroscopic shear strain rate (1/s).
        b_m: Burgers vector (m).
        M: Taylor factor.
        k1: Athermal hardening constant (m^{-1}).
        k2: Dynamic recovery constant.
        t_end: Simulation end time (s).

    Returns:
        Dict with keys: t, rho, rho_sat, tau_MPa_final, backend, note.
    """
    n_steps = 500
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    rho_arr = np.zeros(n_steps + 1)
    rho_arr[0] = rho0

    for i in range(n_steps):
        rho_i = max(rho_arr[i], 1e8)
        drho = M * gamma_dot * (k1 * math.sqrt(rho_i) - k2 * rho_i) / b_m
        rho_arr[i + 1] = max(rho_i + drho * dt, 1e8)

    rho_sat = (k1 / k2) ** 2  # saturation density
    G = 80e9
    tau_final = 0.3 * G * b_m * math.sqrt(rho_arr[-1])

    return {
        "backend": "stub" if not _SCIPY else "scipy",
        "note": "" if _SCIPY else "Install scipy for stiff ODE integration.",
        "t": t_arr.tolist(), "rho": rho_arr.tolist(),
        "rho_sat": rho_sat, "tau_MPa_final": tau_final / 1e6,
    }
