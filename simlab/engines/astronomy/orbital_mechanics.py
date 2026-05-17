"""Orbital mechanics engine for planetary and spacecraft trajectory analysis.

Implements Keplerian orbit propagation, two-body integration, Hohmann transfer
delta-V, vis-viva velocity, escape velocity, orbital period, and Lagrange
point estimation.

Optional dependencies: scipy (ODE integration), numpy (always required).
Falls back to analytic approximations.
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

_G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
_AU = 1.495978707e11  # metres per AU


def kepler_orbit(
    a_AU: float = 1.0,
    e: float = 0.017,
    i_deg: float = 0.0,
    omega_deg: float = 102.9,
    Omega_deg: float = 174.9,
    M0_deg: float = 357.5,
    t_days: float = 365.25,
) -> dict[str, Any]:
    """Propagate a Keplerian orbit using classical orbital elements.

    Args:
        a_AU: Semi-major axis (AU).
        e: Eccentricity (0 <= e < 1 for elliptic).
        i_deg: Inclination (degrees).
        omega_deg: Argument of periapsis (degrees).
        Omega_deg: Longitude of ascending node (degrees).
        M0_deg: Initial mean anomaly (degrees).
        t_days: Propagation time (days).

    Returns:
        Dict with keys: x_AU, y_AU, z_AU, r_AU, true_anomaly_deg, period_days,
        backend, note.
    """
    M_sun = 1.989e30
    a_m = a_AU * _AU
    T_s = 2.0 * math.pi * math.sqrt(a_m ** 3 / (_G * M_sun))
    T_days = T_s / 86400.0
    n = 2.0 * math.pi / T_s  # mean motion

    t_s = t_days * 86400.0
    M = math.radians(M0_deg) + n * t_s
    # Solve Kepler's equation via Newton-Raphson
    E = M
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break

    nu = 2.0 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2.0),
                          math.sqrt(1 - e) * math.cos(E / 2.0))
    r = a_m * (1.0 - e * math.cos(E))

    # Perifocal to inertial
    omega = math.radians(omega_deg)
    Omega = math.radians(Omega_deg)
    i = math.radians(i_deg)
    x_p = r * math.cos(nu)
    y_p = r * math.sin(nu)
    x = (math.cos(Omega) * math.cos(omega) - math.sin(Omega) * math.sin(omega) * math.cos(i)) * x_p + \
        (-math.cos(Omega) * math.sin(omega) - math.sin(Omega) * math.cos(omega) * math.cos(i)) * y_p
    y = (math.sin(Omega) * math.cos(omega) + math.cos(Omega) * math.sin(omega) * math.cos(i)) * x_p + \
        (-math.sin(Omega) * math.sin(omega) + math.cos(Omega) * math.cos(omega) * math.cos(i)) * y_p
    z = math.sin(omega) * math.sin(i) * x_p + math.cos(omega) * math.sin(i) * y_p

    return {
        "backend": "analytic",
        "note": "",
        "x_AU": x / _AU,
        "y_AU": y / _AU,
        "z_AU": z / _AU,
        "r_AU": r / _AU,
        "true_anomaly_deg": math.degrees(nu),
        "period_days": T_days,
    }


def two_body_problem(
    m1_kg: float = 1.989e30,
    m2_kg: float = 5.972e24,
    r0: list[float] | None = None,
    v0: list[float] | None = None,
    t_end_s: float = 3.156e7,
    n_steps: int = 365,
) -> dict[str, Any]:
    """Numerically integrate the gravitational two-body problem.

    Args:
        m1_kg: Mass of body 1 (kg).
        m2_kg: Mass of body 2 (kg).
        r0: Initial relative position [x, y, z] in metres.
        v0: Initial relative velocity [vx, vy, vz] in m/s.
        t_end_s: Integration end time (s).
        n_steps: Number of time steps.

    Returns:
        Dict with keys: t_s, r_x, r_y, r_z, speed_m_s, energy_J, backend, note.
    """
    if r0 is None:
        r0 = [_AU, 0.0, 0.0]
    if v0 is None:
        v0 = [0.0, 29780.0, 0.0]

    mu = _G * (m1_kg + m2_kg)
    state0 = r0 + v0

    def deriv(t, y):
        rx, ry, rz, vx, vy, vz = y
        r_mag = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        a = -mu / r_mag ** 3
        return [vx, vy, vz, a * rx, a * ry, a * rz]

    t_eval = np.linspace(0, t_end_s, n_steps + 1)

    if _SCIPY:
        sol = _solve_ivp(deriv, [0, t_end_s], state0, t_eval=t_eval, method="DOP853",
                         rtol=1e-10, atol=1e-12)
        t_arr = sol.t.tolist()
        rx, ry, rz = sol.y[0].tolist(), sol.y[1].tolist(), sol.y[2].tolist()
        spd = [math.sqrt(sol.y[3][k]**2 + sol.y[4][k]**2 + sol.y[5][k]**2) for k in range(len(t_arr))]
        backend, note = "scipy", ""
    else:
        # Symplectic Euler
        state = list(state0)
        t_arr, rx, ry, rz, spd = [], [], [], [], []
        dt = t_end_s / n_steps
        for k in range(n_steps + 1):
            r_mag = math.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
            t_arr.append(k * dt)
            rx.append(state[0]); ry.append(state[1]); rz.append(state[2])
            spd.append(math.sqrt(state[3]**2 + state[4]**2 + state[5]**2))
            a_fac = -mu / r_mag ** 3
            state[3] += a_fac * state[0] * dt
            state[4] += a_fac * state[1] * dt
            state[5] += a_fac * state[2] * dt
            state[0] += state[3] * dt
            state[1] += state[4] * dt
            state[2] += state[5] * dt
        backend, note = "stub", "Install scipy for DOP853 high-accuracy integrator."

    r_mag0 = math.sqrt(r0[0]**2 + r0[1]**2 + r0[2]**2)
    v_mag0 = math.sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
    energy = 0.5 * m2_kg * v_mag0 ** 2 - _G * m1_kg * m2_kg / r_mag0

    return {
        "backend": backend, "note": note,
        "t_s": t_arr, "r_x": rx, "r_y": ry, "r_z": rz,
        "speed_m_s": spd, "energy_J": energy,
    }


def escape_velocity(
    M_kg: float = 5.972e24,
    r_m: float = 6.371e6,
) -> dict[str, Any]:
    """Compute the escape velocity from a spherical body.

    v_esc = sqrt(2 * G * M / r)

    Args:
        M_kg: Mass of the central body (kg).
        r_m: Distance from centre (m).

    Returns:
        Dict with keys: escape_velocity_m_s, escape_velocity_km_s, backend, note.
    """
    v_esc = math.sqrt(2.0 * _G * M_kg / r_m)
    return {
        "backend": "analytic", "note": "",
        "escape_velocity_m_s": v_esc,
        "escape_velocity_km_s": v_esc / 1000.0,
    }


def hohmann_transfer(
    r1_m: float = 6.571e6,
    r2_m: float = 4.216e7,
    M_kg: float = 5.972e24,
) -> dict[str, Any]:
    """Compute delta-V for a Hohmann transfer between two circular orbits.

    Args:
        r1_m: Radius of initial circular orbit (m).
        r2_m: Radius of final circular orbit (m).
        M_kg: Mass of central body (kg).

    Returns:
        Dict with keys: delta_v1_m_s, delta_v2_m_s, total_delta_v_m_s,
        transfer_time_s, backend, note.
    """
    mu = _G * M_kg
    v1 = math.sqrt(mu / r1_m)
    v2 = math.sqrt(mu / r2_m)
    a_t = (r1_m + r2_m) / 2.0
    v_p = math.sqrt(mu * (2.0 / r1_m - 1.0 / a_t))
    v_a = math.sqrt(mu * (2.0 / r2_m - 1.0 / a_t))
    dv1 = abs(v_p - v1)
    dv2 = abs(v2 - v_a)
    t_transfer = math.pi * math.sqrt(a_t ** 3 / mu)
    return {
        "backend": "analytic", "note": "",
        "delta_v1_m_s": dv1, "delta_v2_m_s": dv2,
        "total_delta_v_m_s": dv1 + dv2,
        "transfer_time_s": t_transfer,
    }


def vis_viva(
    M_kg: float = 1.989e30,
    r_m: float = _AU,
    a_m: float = _AU,
) -> dict[str, Any]:
    """Compute the orbital speed using the vis-viva equation.

    v = sqrt(G*M * (2/r - 1/a))

    Args:
        M_kg: Central body mass (kg).
        r_m: Current distance from central body (m).
        a_m: Semi-major axis of the orbit (m).

    Returns:
        Dict with keys: speed_m_s, speed_km_s, backend, note.
    """
    mu = _G * M_kg
    v = math.sqrt(mu * (2.0 / r_m - 1.0 / a_m))
    return {
        "backend": "analytic", "note": "",
        "speed_m_s": v, "speed_km_s": v / 1000.0,
    }


def orbital_period(
    a_m: float = _AU,
    M_kg: float = 1.989e30,
) -> dict[str, Any]:
    """Compute the orbital period from Kepler's third law.

    T = 2*pi * sqrt(a^3 / (G*M))

    Args:
        a_m: Semi-major axis (m).
        M_kg: Central body mass (kg).

    Returns:
        Dict with keys: period_s, period_days, period_years, backend, note.
    """
    T = 2.0 * math.pi * math.sqrt(a_m ** 3 / (_G * M_kg))
    return {
        "backend": "analytic", "note": "",
        "period_s": T, "period_days": T / 86400.0, "period_years": T / 3.1557e7,
    }


def lagrange_points(
    m1_kg: float = 1.989e30,
    m2_kg: float = 5.972e24,
    a_m: float = _AU,
) -> dict[str, Any]:
    """Estimate positions of the five Lagrange points in the CR3BP.

    L1/L2/L3 positions are found via Hill-sphere approximation.
    L4/L5 are at 60 degrees from m2 on the orbit.

    Args:
        m1_kg: Mass of primary body (kg).
        m2_kg: Mass of secondary body (kg).
        a_m: Orbital separation (m).

    Returns:
        Dict with keys: L1_m, L2_m, L3_m, L4, L5, mass_ratio, backend, note.
    """
    mu_r = m2_kg / (m1_kg + m2_kg)  # mass ratio
    r_hill = a_m * (mu_r / 3.0) ** (1.0 / 3.0)
    L1 = a_m - r_hill
    L2 = a_m + r_hill
    L3 = -a_m * (1.0 + 5.0 * mu_r / 12.0)
    # L4/L5 at 60 degrees from m2
    L4 = {"x": a_m * (0.5 - mu_r), "y": a_m * math.sqrt(3.0) / 2.0}
    L5 = {"x": a_m * (0.5 - mu_r), "y": -a_m * math.sqrt(3.0) / 2.0}
    return {
        "backend": "analytic", "note": "",
        "L1_m": L1, "L2_m": L2, "L3_m": L3,
        "L4": L4, "L5": L5,
        "hill_radius_m": r_hill, "mass_ratio": mu_r,
    }
