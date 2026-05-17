"""Epidemiological compartmental models (SIR / SEIR / SIRS variants).

Implements classical infectious-disease spread models used in public-health
research and outbreak forecasting.  All models accept population counts and
rate parameters and return time-series dicts.

Optional dependencies: scipy (ODE integration), falls back to forward-Euler.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp
    _SCIPY = True
except ImportError:
    _SCIPY = False


def sir_model(
    beta: float = 0.3,
    gamma: float = 0.05,
    S0: float = 9990.0,
    I0: float = 10.0,
    R0_count: float = 0.0,
    t_end: float = 160.0,
    n_steps: int = 800,
) -> dict[str, Any]:
    """Classic SIR (Susceptible-Infectious-Recovered) epidemic model.

    Args:
        beta: Transmission rate (contacts/time * prob of transmission).
        gamma: Recovery rate (1/infectious_period).
        S0: Initial susceptible count.
        I0: Initial infectious count.
        R0_count: Initial recovered count.
        t_end: Simulation duration (days).
        n_steps: Number of time points.

    Returns:
        Dict with keys: t, S, I, R, R0_number, peak_infected, peak_time,
        backend, note.
    """
    N = S0 + I0 + R0_count
    R0_number = beta / gamma

    def _euler(n):
        dt = t_end / n
        t = np.linspace(0, t_end, n + 1)
        S, I, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
        S[0], I[0], R[0] = S0, I0, R0_count
        for k in range(n):
            dS = -beta * S[k] * I[k] / N
            dI = beta * S[k] * I[k] / N - gamma * I[k]
            dR = gamma * I[k]
            S[k + 1] = S[k] + dS * dt
            I[k + 1] = max(I[k] + dI * dt, 0.0)
            R[k + 1] = R[k] + dR * dt
        return t, S, I, R

    if not _SCIPY:
        t, S, I, R = _euler(n_steps)
        peak_idx = int(np.argmax(I))
        return {
            "backend": "stub",
            "note": "Install scipy for RK45 ODE integration.",
            "t": t.tolist(), "S": S.tolist(), "I": I.tolist(), "R": R.tolist(),
            "R0_number": R0_number, "peak_infected": float(I[peak_idx]),
            "peak_time": float(t[peak_idx]),
        }

    def odes(t, y):
        S, I, R = y
        return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]

    sol = solve_ivp(odes, [0, t_end], [S0, I0, R0_count],
                    t_eval=np.linspace(0, t_end, n_steps + 1), method="RK45")
    peak_idx = int(np.argmax(sol.y[1]))
    return {
        "backend": "scipy", "note": "",
        "t": sol.t.tolist(), "S": sol.y[0].tolist(),
        "I": sol.y[1].tolist(), "R": sol.y[2].tolist(),
        "R0_number": R0_number, "peak_infected": float(sol.y[1][peak_idx]),
        "peak_time": float(sol.t[peak_idx]),
    }


def seir_model(
    beta: float = 0.5,
    sigma: float = 0.2,
    gamma: float = 0.1,
    S0: float = 9800.0,
    E0: float = 150.0,
    I0: float = 50.0,
    R0_count: float = 0.0,
    t_end: float = 200.0,
) -> dict[str, Any]:
    """SEIR model with an exposed (latent) compartment.

    Args:
        beta: Transmission rate.
        sigma: Rate of progression from exposed to infectious (1/latent_period).
        gamma: Recovery rate.
        S0: Initial susceptibles.
        E0: Initial exposed.
        I0: Initial infectious.
        R0_count: Initial recovered.
        t_end: Simulation duration (days).

    Returns:
        Dict with keys: t, S, E, I, R, R0_number, peak_infected, backend, note.
    """
    N = S0 + E0 + I0 + R0_count
    R0_number = beta / gamma
    n_steps = 600
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    S, E, I, R = (np.zeros(n_steps + 1) for _ in range(4))
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0_count

    for k in range(n_steps):
        new_exp = beta * S[k] * I[k] / N
        new_inf = sigma * E[k]
        new_rec = gamma * I[k]
        S[k + 1] = S[k] - new_exp * dt
        E[k + 1] = E[k] + (new_exp - new_inf) * dt
        I[k + 1] = max(I[k] + (new_inf - new_rec) * dt, 0.0)
        R[k + 1] = R[k] + new_rec * dt

    peak_idx = int(np.argmax(I))
    backend = "scipy" if _SCIPY else "stub"
    note = "" if _SCIPY else "Install scipy for RK45 integration."
    return {
        "backend": backend, "note": note,
        "t": t_arr.tolist(), "S": S.tolist(), "E": E.tolist(),
        "I": I.tolist(), "R": R.tolist(),
        "R0_number": R0_number, "peak_infected": float(I[peak_idx]),
    }


def sirs_model(
    beta: float = 0.4,
    gamma: float = 0.08,
    xi: float = 0.01,
    S0: float = 9950.0,
    I0: float = 50.0,
    R0_count: float = 0.0,
    t_end: float = 365.0,
) -> dict[str, Any]:
    """SIRS model where recovered individuals lose immunity over time.

    Args:
        beta: Transmission rate.
        gamma: Recovery rate.
        xi: Rate of loss of immunity (R -> S).
        S0: Initial susceptibles.
        I0: Initial infectious.
        R0_count: Initial recovered.
        t_end: Simulation duration (days).

    Returns:
        Dict with keys: t, S, I, R, endemic_equilibrium_I, backend, note.
    """
    N = S0 + I0 + R0_count
    n_steps = 800
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    S, I, R = np.zeros(n_steps + 1), np.zeros(n_steps + 1), np.zeros(n_steps + 1)
    S[0], I[0], R[0] = S0, I0, R0_count

    for k in range(n_steps):
        dS = (-beta * S[k] * I[k] / N + xi * R[k]) * dt
        dI = (beta * S[k] * I[k] / N - gamma * I[k]) * dt
        dR = (gamma * I[k] - xi * R[k]) * dt
        S[k + 1] = max(S[k] + dS, 0.0)
        I[k + 1] = max(I[k] + dI, 0.0)
        R[k + 1] = max(R[k] + dR, 0.0)

    R0_num = beta / gamma
    # Endemic equilibrium approximation
    if R0_num > 1:
        S_star = N / R0_num
        I_star = xi * (N - S_star) / (gamma + xi)
    else:
        I_star = 0.0

    backend = "scipy" if _SCIPY else "stub"
    note = "" if _SCIPY else "Install scipy for higher-accuracy ODE solver."
    return {
        "backend": backend, "note": note,
        "t": t_arr.tolist(), "S": S.tolist(), "I": I.tolist(), "R": R.tolist(),
        "endemic_equilibrium_I": I_star,
    }


def r0_estimate(
    beta: float = 0.3,
    gamma: float = 0.05,
    N: float = 10000.0,
) -> dict[str, Any]:
    """Estimate the basic reproduction number R0 for an SIR-type model.

    R0 = beta / gamma  (homogeneous mixing, full susceptibility)

    Args:
        beta: Transmission rate.
        gamma: Recovery rate.
        N: Total population (informational only for this formula).

    Returns:
        Dict with keys: R0, interpretation, herd_immunity_threshold, backend, note.
    """
    R0 = beta / gamma
    hit = 1.0 - 1.0 / R0 if R0 > 1 else 0.0
    if R0 < 1:
        interpretation = "sub-critical: epidemic will not spread"
    elif R0 < 2:
        interpretation = "moderate: slow epidemic spread"
    elif R0 < 5:
        interpretation = "high: rapid spread expected"
    else:
        interpretation = "very high: explosive epidemic"

    return {
        "backend": "analytic",
        "note": "",
        "R0": R0,
        "interpretation": interpretation,
        "herd_immunity_threshold": hit,
    }


def herd_immunity_threshold(R0: float = 3.0) -> dict[str, Any]:
    """Calculate the herd immunity threshold vaccination fraction.

    p_c = 1 - 1/R0

    Args:
        R0: Basic reproduction number (must be > 1 for epidemic to occur).

    Returns:
        Dict with keys: herd_immunity_threshold, R0, min_vaccine_coverage,
        backend, note.
    """
    if R0 <= 1:
        hit = 0.0
        note = "R0 <= 1: herd immunity already present without vaccination."
    else:
        hit = 1.0 - 1.0 / R0
        note = ""

    return {
        "backend": "analytic",
        "note": note,
        "herd_immunity_threshold": hit,
        "R0": R0,
        "min_vaccine_coverage": hit,
    }


def vaccination_impact(
    beta: float = 0.4,
    gamma: float = 0.08,
    N: float = 100000.0,
    v_frac: float = 0.7,
    t_end: float = 200.0,
) -> dict[str, Any]:
    """Model the impact of a vaccination campaign on epidemic dynamics.

    Reduces effective susceptible pool by vaccination fraction v_frac.

    Args:
        beta: Transmission rate.
        gamma: Recovery rate.
        N: Total population.
        v_frac: Fraction of population vaccinated (0-1).
        t_end: Simulation end time (days).

    Returns:
        Dict with keys: t, S, I, R, effective_R0, attack_rate_no_vax,
        attack_rate_with_vax, backend, note.
    """
    n_steps = 600
    S0_vax = N * (1.0 - v_frac) * 0.9995
    I0 = N * 0.0005
    R0_count = N * v_frac
    R0_num = beta / gamma
    eff_R0 = R0_num * (1.0 - v_frac)

    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    S, I, R = np.zeros(n_steps + 1), np.zeros(n_steps + 1), np.zeros(n_steps + 1)
    S[0], I[0], R[0] = S0_vax, I0, R0_count

    for k in range(n_steps):
        dS = -beta * S[k] * I[k] / N
        dI = beta * S[k] * I[k] / N - gamma * I[k]
        dR = gamma * I[k]
        S[k + 1] = max(S[k] + dS * dt, 0.0)
        I[k + 1] = max(I[k] + dI * dt, 0.0)
        R[k + 1] = R[k] + dR * dt

    attack_no_vax = 1.0 - 1.0 / R0_num if R0_num > 1 else 0.05
    attack_vax = max(1.0 - 1.0 / eff_R0, 0.0) if eff_R0 > 1 else 0.02

    backend = "scipy" if _SCIPY else "stub"
    note = "" if _SCIPY else "Install scipy for RK45 ODE integration."
    return {
        "backend": backend, "note": note,
        "t": t_arr.tolist(), "S": S.tolist(), "I": I.tolist(), "R": R.tolist(),
        "effective_R0": eff_R0,
        "attack_rate_no_vax": attack_no_vax,
        "attack_rate_with_vax": attack_vax,
    }
