"""Hydrogeology and groundwater simulation utilities.

Covers Darcy flow, Theis well-drawdown, steady-state water table profiles,
1-D contaminant transport, and basic aquifer test analysis.

Optional dependencies: scipy (ODE / special functions), numpy (always required).
Falls back to analytic or finite-difference approximations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.special import exp1 as _exp1
    _SCIPY = True
except ImportError:
    _SCIPY = False


def darcy_flow(
    K: float = 1e-4,
    dh_dl: float = 0.005,
    A: float = 100.0,
) -> dict[str, Any]:
    """Compute volumetric discharge using Darcy's law.

    Q = -K * (dh/dl) * A

    Args:
        K: Hydraulic conductivity (m/s).
        dh_dl: Hydraulic gradient (m/m, positive = downslope).
        A: Cross-sectional area perpendicular to flow (m^2).

    Returns:
        Dict with keys: Q_m3_s, darcy_velocity_m_s, K, dh_dl, A, backend, note.
    """
    q = -K * dh_dl  # Darcy velocity (m/s)
    Q = q * A
    return {
        "backend": "analytic",
        "note": "",
        "Q_m3_s": Q,
        "darcy_velocity_m_s": q,
        "K": K,
        "dh_dl": dh_dl,
        "A": A,
    }


def theis_drawdown(
    Q: float = 0.05,
    T: float = 1e-3,
    S: float = 1e-4,
    r: float = 50.0,
    t: float = 3600.0,
) -> dict[str, Any]:
    """Compute drawdown at radius r and time t using the Theis equation.

    s(r,t) = Q / (4 * pi * T) * W(u)   where u = r^2 * S / (4 * T * t)
    W(u) = E1(u) (well function = exponential integral)

    Args:
        Q: Pumping rate (m^3/s).
        T: Transmissivity (m^2/s).
        S: Storativity (dimensionless).
        r: Radial distance from well (m).
        t: Time since pumping began (s).

    Returns:
        Dict with keys: drawdown_m, u, well_function, backend, note.
    """
    u = r ** 2 * S / (4.0 * T * t)

    if _SCIPY:
        W_u = float(_exp1(u))
        backend = "scipy"
        note = ""
    else:
        # Cooper-Jacob approximation valid for u < 0.05
        if u < 0.05:
            W_u = -0.5772 - math.log(u)
            note = "Cooper-Jacob approximation (valid for small u)."
        else:
            # Abramowitz & Stegun series truncation
            W_u = math.exp(-u) * (1.0 / (u + 0.2677) + 0.05)
            note = "Approximate well function; install scipy for E1(u)."
        backend = "stub"

    s = Q / (4.0 * math.pi * T) * W_u
    return {
        "backend": backend,
        "note": note,
        "drawdown_m": s,
        "u": u,
        "well_function": W_u,
    }


def water_table_profile(
    recharge_rate: float = 2e-8,
    K: float = 1e-4,
    L: float = 1000.0,
    h1: float = 10.0,
    h2: float = 8.0,
    n_points: int = 50,
) -> dict[str, Any]:
    """Steady-state water-table profile (Dupuit-Forchheimer).

    h^2(x) = h1^2 + (h2^2 - h1^2) * x/L + (W/K) * x * (L - x)

    Args:
        recharge_rate: Uniform recharge rate (m/s).
        K: Hydraulic conductivity (m/s).
        L: Domain length (m).
        h1: Head at left boundary (m above datum).
        h2: Head at right boundary (m above datum).
        n_points: Number of spatial points.

    Returns:
        Dict with keys: x_m, head_m, peak_head_m, peak_x_m, backend, note.
    """
    x = np.linspace(0.0, L, n_points)
    h2_profile = (h1 ** 2 + (h2 ** 2 - h1 ** 2) * x / L +
                  (recharge_rate / K) * x * (L - x))
    h = np.sqrt(np.maximum(h2_profile, 0.0))
    peak_idx = int(np.argmax(h))
    return {
        "backend": "analytic",
        "note": "",
        "x_m": x.tolist(),
        "head_m": h.tolist(),
        "peak_head_m": float(h[peak_idx]),
        "peak_x_m": float(x[peak_idx]),
    }


def contaminant_transport(
    v: float = 1e-5,
    D: float = 5e-6,
    C0: float = 100.0,
    x_max: float = 200.0,
    t: float = 1e6,
    n_points: int = 100,
) -> dict[str, Any]:
    """1-D advection-dispersion contaminant transport (analytical).

    C(x,t) = (C0/2) * erfc[(x - v*t) / (2*sqrt(D*t))]
              + (C0/2) * exp(v*x/D) * erfc[(x + v*t) / (2*sqrt(D*t))]

    Args:
        v: Pore-water velocity (m/s).
        D: Hydrodynamic dispersion coefficient (m^2/s).
        C0: Inlet concentration (mg/L or any unit).
        x_max: Maximum distance from source (m).
        t: Time (s).
        n_points: Spatial resolution.

    Returns:
        Dict with keys: x_m, C, peclet_number, backend, note.
    """
    x = np.linspace(0.0, x_max, n_points)
    Pe = v * x_max / D
    sqrt_Dt = math.sqrt(D * t)
    arg1 = (x - v * t) / (2.0 * sqrt_Dt)
    arg2 = (x + v * t) / (2.0 * sqrt_Dt)

    if _SCIPY:
        from scipy.special import erfc
        C = 0.5 * C0 * erfc(arg1) + 0.5 * C0 * np.exp(np.clip(v * x / D, -700, 700)) * erfc(arg2)
        backend = "scipy"
        note = ""
    else:
        # Vectorised approximation of erfc via series
        def _erfc_approx(z):
            a = np.abs(z)
            t_ = 1.0 / (1.0 + 0.3275911 * a)
            poly = t_ * (0.254829592 + t_ * (-0.284496736 + t_ * (
                1.421413741 + t_ * (-1.453152027 + t_ * 1.061405429))))
            result = poly * np.exp(-a ** 2)
            return np.where(z < 0, 2.0 - result, result)

        C = 0.5 * C0 * _erfc_approx(arg1) + 0.5 * C0 * np.exp(
            np.clip(v * x / D, -700, 700)) * _erfc_approx(arg2)
        backend = "stub"
        note = "Install scipy for erfc; using polynomial approximation."

    return {
        "backend": backend,
        "note": note,
        "x_m": x.tolist(),
        "C": np.clip(C, 0.0, C0).tolist(),
        "peclet_number": Pe,
    }


def aquifer_test_analysis(
    drawdown_data: list[float] | None = None,
    time_data: list[float] | None = None,
    Q: float = 0.05,
) -> dict[str, Any]:
    """Estimate transmissivity and storativity from pumping test data.

    Uses the Cooper-Jacob straight-line method on a log-time plot.

    Args:
        drawdown_data: List of drawdown values (m) at one observation well.
            Defaults to a synthetic dataset.
        time_data: Corresponding time values (s).  Defaults to synthetic data.
        Q: Pumping rate (m^3/s).

    Returns:
        Dict with keys: transmissivity_m2_s, storativity, r_squared,
        backend, note.
    """
    if time_data is None:
        time_data = [60, 120, 300, 600, 1200, 3600, 7200, 14400]
    if drawdown_data is None:
        drawdown_data = [0.05, 0.12, 0.24, 0.35, 0.48, 0.70, 0.82, 0.95]

    log_t = np.log10(np.array(time_data, dtype=float))
    s = np.array(drawdown_data, dtype=float)

    # Linear regression on log_t vs s
    coeffs = np.polyfit(log_t, s, 1)
    delta_s = coeffs[0] * math.log10(10)  # drawdown per log cycle
    T = 2.303 * Q / (4.0 * math.pi * delta_s) if delta_s > 0 else float("nan")

    t0 = 10 ** (-coeffs[1] / coeffs[0]) if coeffs[0] != 0 else float("nan")
    r = 50.0  # assumed observation well distance
    S = 2.246 * T * t0 / r ** 2

    s_pred = np.polyval(coeffs, log_t)
    ss_res = np.sum((s - s_pred) ** 2)
    ss_tot = np.sum((s - s.mean()) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "backend": "numpy",
        "note": "",
        "transmissivity_m2_s": T,
        "storativity": S,
        "r_squared": r_sq,
    }
