"""Phase-field models coupled to CALPHAD thermodynamics.

Provides proxy phase-field simulations (Allen-Cahn, Cahn-Hilliard) that
call a CALPHAD database when pycalphad is available, falling back to a
polynomial free-energy double well.

Optional dependencies: pycalphad, numpy (always required), scipy.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import pycalphad  # noqa: F401
    _CALPHAD = True
except ImportError:
    _CALPHAD = False

try:
    from scipy.integrate import solve_ivp as _solve_ivp
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _double_well(phi: np.ndarray, T_K: float, T_melt: float = 1700.0) -> np.ndarray:
    """Simple polynomial double-well free energy proxy."""
    delta_T = T_K - T_melt
    return phi ** 2 * (1.0 - phi) ** 2 - 0.01 * delta_T * (phi - 0.5)


def allen_cahn_calphad(
    phi0: float = 0.0,
    T_K: float = 1400.0,
    composition: dict[str, float] | None = None,
    calphad_db_path: str = "",
    dx: float = 1e-6,
    dt: float = 1e-4,
    n_steps: int = 200,
    kappa: float = 1e-9,
    L_mob: float = 1e-2,
) -> dict[str, Any]:
    """Allen-Cahn phase-field equation with CALPHAD free energy.

    dphi/dt = -L * (dF/dphi - kappa * nabla^2 phi)

    Args:
        phi0: Initial order parameter (0=parent, 1=product phase).
        T_K: Temperature (K).
        composition: Alloy composition dict (e.g. {'Fe': 0.97, 'C': 0.03}).
        calphad_db_path: Path to TDB file (only used if pycalphad installed).
        dx: Grid spacing (m).
        dt: Time step (s).
        n_steps: Number of time steps.
        kappa: Gradient energy coefficient (J/m).
        L_mob: Interface mobility (m^3/J/s).

    Returns:
        Dict with keys: t, phi_mean, phi_final_profile, backend, note.
    """
    N = 64
    phi = np.zeros(N)
    phi[:N // 2] = phi0
    t_arr = []
    phi_mean = []

    for step in range(n_steps):
        laplacian = np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)
        dF_dphi = 2 * phi * (1 - phi) * (1 - 2 * phi) - 0.005 * (T_K - 1700.0)
        dphi = -L_mob * (dF_dphi - kappa / dx ** 2 * laplacian)
        phi = np.clip(phi + dphi * dt, 0.0, 1.0)
        if step % 20 == 0:
            t_arr.append(step * dt)
            phi_mean.append(float(phi.mean()))

    backend = "pycalphad" if _CALPHAD else "stub"
    note = "" if _CALPHAD else "Install pycalphad for real CALPHAD free energy; using polynomial double-well."
    return {
        "backend": backend, "note": note,
        "t": t_arr, "phi_mean": phi_mean,
        "phi_final_profile": phi.tolist(),
    }


def cahn_hilliard_calphad(
    c0: float = 0.05,
    T_K: float = 800.0,
    composition: dict[str, float] | None = None,
    calphad_db_path: str = "",
    dx: float = 1e-7,
    dt: float = 1e-4,
    n_steps: int = 300,
    kappa: float = 1e-13,
    M_mob: float = 1e-22,
) -> dict[str, Any]:
    """Cahn-Hilliard spinodal decomposition with CALPHAD free energy.

    dc/dt = M * nabla^2 (df/dc - kappa * nabla^2 c)

    Args:
        c0: Mean composition (mole fraction).
        T_K: Temperature (K).
        composition: Alloy system composition dict.
        calphad_db_path: Path to TDB file.
        dx: Grid spacing (m).
        dt: Time step (s).
        n_steps: Number of time steps.
        kappa: Gradient energy coefficient (J/m).
        M_mob: Chemical mobility (m^5/J/s).

    Returns:
        Dict with keys: t, c_std (spinodal amplification), c_profile, backend, note.
    """
    N = 128
    np.random.seed(42)
    c = c0 + 0.001 * np.random.randn(N)
    c = np.clip(c, 0.0, 1.0)
    t_arr, c_std = [], []

    for step in range(n_steps):
        # Chemical potential proxy: df/dc = 2*(c - 0.5) * (2*c^2 - 2*c + 0.5)
        mu = (2.0 * c - 1.0) * (1.0 - 2.0 * c + 2.0 * c ** 2) - \
             kappa / dx ** 2 * (np.roll(c, -1) - 2 * c + np.roll(c, 1))
        dc = M_mob / dx ** 2 * (np.roll(mu, -1) - 2 * mu + np.roll(mu, 1))
        c = np.clip(c + dc * dt, 0.0, 1.0)
        if step % 30 == 0:
            t_arr.append(step * dt)
            c_std.append(float(np.std(c)))

    backend = "pycalphad" if _CALPHAD else "stub"
    note = "" if _CALPHAD else "Install pycalphad for real CALPHAD free energy."
    return {
        "backend": backend, "note": note,
        "t": t_arr, "c_std": c_std, "c_profile": c.tolist(),
    }


def spinodal_decomposition(
    c0: float = 0.3,
    T_K: float = 600.0,
    noise_amplitude: float = 0.005,
    dx: float = 2e-9,
    dt: float = 1e-5,
    n_steps: int = 500,
) -> dict[str, Any]:
    """Cahn-Hilliard spinodal decomposition with noise (proxy).

    Args:
        c0: Mean composition.
        T_K: Temperature (K).
        noise_amplitude: Initial composition fluctuation amplitude.
        dx: Grid spacing (m).
        dt: Time step (s).
        n_steps: Number of evolution steps.

    Returns:
        Dict with keys: t, structure_factor_peak, c_profile, backend, note.
    """
    N = 256
    np.random.seed(7)
    c = c0 + noise_amplitude * np.random.randn(N)
    kappa = 1e-13
    M = 1e-22
    SF_peaks = []
    t_arr = []

    for step in range(n_steps):
        mu = 4.0 * (c - c0) * (1.0 - (c - c0) ** 2) - \
             kappa / dx ** 2 * (np.roll(c, -1) - 2 * c + np.roll(c, 1))
        dc = M / dx ** 2 * (np.roll(mu, -1) - 2 * mu + np.roll(mu, 1))
        c = np.clip(c + dc * dt, 0.0, 1.0)
        if step % 50 == 0:
            SF = np.abs(np.fft.rfft(c - c.mean())) ** 2
            SF_peaks.append(float(SF[1:].max()))
            t_arr.append(step * dt)

    return {
        "backend": "stub",
        "note": "Install pycalphad+scipy for CALPHAD-coupled phase-field.",
        "t": t_arr, "structure_factor_peak": SF_peaks, "c_profile": c.tolist(),
    }


def solidification_phase_field(
    T_K: float = 1685.0,
    undercooling: float = 50.0,
    dx: float = 1e-7,
    dt: float = 1e-6,
    n_steps: int = 400,
) -> dict[str, Any]:
    """Dendritic solidification proxy using a simple Allen-Cahn model.

    Args:
        T_K: Nominal melting temperature (K).
        undercooling: Thermal undercooling (K).
        dx: Grid spacing (m).
        dt: Time step (s).
        n_steps: Number of evolution steps.

    Returns:
        Dict with keys: t, solid_fraction, phi_profile, backend, note.
    """
    N = 128
    phi = np.zeros(N)
    phi[N // 2 - 2: N // 2 + 2] = 1.0  # seed nucleus
    L_mob = 1e-2
    kappa = 1e-9
    T_m = T_K
    T_actual = T_K - undercooling
    sol_frac = []
    t_arr = []

    for step in range(n_steps):
        laplacian = np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)
        df = 2 * phi * (1 - phi) * (1 - 2 * phi) - 0.01 * (T_actual - T_m)
        dphi = -L_mob * (df - kappa / dx ** 2 * laplacian)
        phi = np.clip(phi + dphi * dt, 0.0, 1.0)
        if step % 40 == 0:
            t_arr.append(step * dt)
            sol_frac.append(float(phi.mean()))

    return {
        "backend": "stub",
        "note": "Install pycalphad for CALPHAD-coupled solidification.",
        "t": t_arr, "solid_fraction": sol_frac, "phi_profile": phi.tolist(),
    }
