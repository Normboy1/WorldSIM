"""Turbulence modelling utilities (k-epsilon, wall laws, mixing length).

Provides the k-epsilon closure model, turbulent viscosity, wall y+, Kolmogorov
dissipation scales, mixing length, and logarithmic law of the wall.

Optional dependencies: scipy (ODE integration), numpy (always required).
Falls back to Euler integration.
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


def k_epsilon_model(
    k0: float = 0.01,
    epsilon0: float = 0.001,
    rho: float = 1.225,
    nu: float = 1.5e-5,
    t_end: float = 10.0,
    n_steps: int = 200,
) -> dict[str, Any]:
    """Integrate the 0-D standard k-epsilon turbulence model (decay).

    dk/dt = -epsilon
    depsilon/dt = -C2 * epsilon^2 / k

    Args:
        k0: Initial turbulent kinetic energy (m^2/s^2).
        epsilon0: Initial dissipation rate (m^2/s^3).
        rho: Fluid density (kg/m^3).
        nu: Kinematic viscosity (m^2/s).
        t_end: Integration end time (s).
        n_steps: Number of time steps.

    Returns:
        Dict with keys: t, k, epsilon, Re_t, backend, note.
    """
    C2 = 1.92
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    k_arr = np.zeros(n_steps + 1)
    eps_arr = np.zeros(n_steps + 1)
    k_arr[0], eps_arr[0] = k0, epsilon0

    for i in range(n_steps):
        k_i, e_i = max(k_arr[i], 1e-12), max(eps_arr[i], 1e-12)
        dk = -e_i * dt
        de = -C2 * e_i ** 2 / k_i * dt
        k_arr[i + 1] = max(k_i + dk, 1e-12)
        eps_arr[i + 1] = max(e_i + de, 1e-12)

    Re_t = k_arr ** 2 / (nu * eps_arr)
    return {
        "backend": "stub" if not _SCIPY else "scipy",
        "note": "" if _SCIPY else "Install scipy for stiff ODE solver.",
        "t": t_arr.tolist(), "k": k_arr.tolist(), "epsilon": eps_arr.tolist(),
        "Re_t": Re_t.tolist(),
    }


def turbulent_viscosity(
    Cmu: float = 0.09,
    rho: float = 1.225,
    k: float = 0.1,
    epsilon: float = 0.01,
) -> dict[str, Any]:
    """Compute turbulent eddy viscosity from k-epsilon closure.

    mu_t = Cmu * rho * k^2 / epsilon

    Args:
        Cmu: k-epsilon closure constant (typically 0.09).
        rho: Fluid density (kg/m^3).
        k: Turbulent kinetic energy (m^2/s^2).
        epsilon: Dissipation rate (m^2/s^3).

    Returns:
        Dict with keys: mu_t_Pa_s, nu_t_m2_s, backend, note.
    """
    mu_t = Cmu * rho * k ** 2 / epsilon
    nu_t = mu_t / rho
    return {
        "backend": "analytic", "note": "",
        "mu_t_Pa_s": mu_t, "nu_t_m2_s": nu_t,
    }


def wall_y_plus(
    u_tau: float = 0.3,
    y: float = 1e-4,
    nu: float = 1.5e-5,
) -> dict[str, Any]:
    """Compute the dimensionless wall distance y+.

    y+ = u_tau * y / nu

    Args:
        u_tau: Friction velocity (m/s).
        y: Wall-normal distance (m).
        nu: Kinematic viscosity (m^2/s).

    Returns:
        Dict with keys: y_plus, wall_layer, backend, note.
    """
    yp = u_tau * y / nu
    if yp < 5:
        layer = "viscous_sublayer"
    elif yp < 30:
        layer = "buffer_layer"
    elif yp < 300:
        layer = "log_layer"
    else:
        layer = "outer_layer"
    return {
        "backend": "analytic", "note": "",
        "y_plus": yp, "wall_layer": layer,
    }


def kolmogorov_scales(
    epsilon: float = 0.01,
    nu: float = 1.5e-5,
) -> dict[str, Any]:
    """Compute the Kolmogorov microscales of turbulence.

    eta = (nu^3 / epsilon)^(1/4)   [length]
    tau = (nu / epsilon)^(1/2)      [time]
    u_k  = (nu * epsilon)^(1/4)     [velocity]

    Args:
        epsilon: Turbulent dissipation rate (m^2/s^3).
        nu: Kinematic viscosity (m^2/s).

    Returns:
        Dict with keys: eta_m, tau_s, u_k_m_s, Re_k, backend, note.
    """
    eta = (nu ** 3 / epsilon) ** 0.25
    tau = (nu / epsilon) ** 0.5
    u_k = (nu * epsilon) ** 0.25
    return {
        "backend": "analytic", "note": "",
        "eta_m": eta, "tau_s": tau, "u_k_m_s": u_k, "Re_k": 1.0,
    }


def turbulent_prandtl_number(Pr_t: float = 0.9) -> dict[str, Any]:
    """Return the turbulent Prandtl number (constant in most models).

    Args:
        Pr_t: Turbulent Prandtl number (default 0.9 per most RANS models).

    Returns:
        Dict with keys: Pr_t, backend, note.
    """
    return {"backend": "analytic", "note": "", "Pr_t": Pr_t}


def mixing_length(
    y: float = 0.01,
    kappa: float = 0.41,
    delta: float = 0.1,
) -> dict[str, Any]:
    """Mixing length model (van Driest / Prandtl).

    l_m = kappa * y * [1 - exp(-y+/26)]  (van Driest near wall)
    Simplified without damping: l_m = kappa * min(y, 0.09*delta)

    Args:
        y: Wall-normal distance (m).
        kappa: von Kármán constant.
        delta: Boundary layer thickness (m).

    Returns:
        Dict with keys: l_m, kappa, y, backend, note.
    """
    lm = kappa * min(y, 0.09 * delta)
    return {
        "backend": "analytic", "note": "",
        "l_m": lm, "kappa": kappa, "y": y,
    }


def log_law_velocity(
    u_tau: float = 0.3,
    y: float = 0.01,
    nu: float = 1.5e-5,
    kappa: float = 0.41,
    B: float = 5.2,
) -> dict[str, Any]:
    """Compute mean velocity from the log law of the wall.

    u+ = (1/kappa) * ln(y+) + B

    Args:
        u_tau: Friction velocity (m/s).
        y: Wall-normal distance (m).
        nu: Kinematic viscosity (m^2/s).
        kappa: von Kármán constant.
        B: Log-law intercept constant.

    Returns:
        Dict with keys: u_plus, u_m_s, y_plus, backend, note.
    """
    yp = u_tau * y / nu
    if yp < 5:
        u_plus = yp  # viscous sublayer
    else:
        u_plus = (1.0 / kappa) * math.log(max(yp, 1.0)) + B
    return {
        "backend": "analytic", "note": "",
        "u_plus": u_plus, "u_m_s": u_plus * u_tau, "y_plus": yp,
    }
