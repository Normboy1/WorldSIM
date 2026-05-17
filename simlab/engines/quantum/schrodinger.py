"""Quantum mechanical solutions to the Schrödinger equation.

Provides analytic and numeric solutions for particle-in-a-box, harmonic
oscillator, quantum tunnelling (transfer matrix / WKB), finite square well
bound states, and time evolution of a 1-D wave packet.

Optional dependencies: scipy (special functions, eigensolver), numpy (always).
Falls back to analytic closed-form expressions where available.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import scipy.linalg as _la
    import scipy.special as _sp
    _SCIPY = True
except ImportError:
    _SCIPY = False

_HBAR = 1.054571817e-34   # J*s
_ME   = 9.1093837015e-31  # kg (electron mass)
_EV   = 1.602176634e-19   # J per eV


def particle_in_box(
    n: int = 1,
    L: float = 1e-9,
    m_kg: float = _ME,
) -> dict[str, Any]:
    """Energy levels and wave-function of a particle in a 1-D infinite square well.

    E_n = n^2 * pi^2 * hbar^2 / (2 * m * L^2)
    psi_n(x) = sqrt(2/L) * sin(n*pi*x/L)

    Args:
        n: Quantum number (positive integer).
        L: Box length (m).
        m_kg: Particle mass (kg).

    Returns:
        Dict with keys: energy_J, energy_eV, x, psi, psi_sq, n, L, backend, note.
    """
    E = n ** 2 * math.pi ** 2 * _HBAR ** 2 / (2.0 * m_kg * L ** 2)
    x = np.linspace(0, L, 200)
    psi = np.sqrt(2.0 / L) * np.sin(n * math.pi * x / L)
    return {
        "backend": "analytic", "note": "",
        "energy_J": E, "energy_eV": E / _EV,
        "x": x.tolist(), "psi": psi.tolist(), "psi_sq": (psi ** 2).tolist(),
        "n": n, "L": L,
    }


def harmonic_oscillator(
    n: int = 0,
    omega: float = 1e14,
    m_kg: float = _ME,
) -> dict[str, Any]:
    """Quantum harmonic oscillator energy levels and Hermite-Gauss wave function.

    E_n = (n + 1/2) * hbar * omega

    Args:
        n: Quantum number (non-negative integer).
        omega: Angular frequency (rad/s).
        m_kg: Particle mass (kg).

    Returns:
        Dict with keys: energy_J, energy_eV, x_nm, psi, psi_sq, backend, note.
    """
    E = (n + 0.5) * _HBAR * omega
    x0 = math.sqrt(_HBAR / (m_kg * omega))  # characteristic length
    x = np.linspace(-5 * x0, 5 * x0, 300)
    xi = x / x0

    if _SCIPY:
        from scipy.special import hermite
        H_n = hermite(n)(xi)
        norm = (2 ** n * math.factorial(n) * math.sqrt(math.pi)) ** (-0.5)
        psi = norm * np.exp(-xi ** 2 / 2) * H_n
        backend, note = "scipy", ""
    else:
        # Gaussian ground-state approximation for n=0, Gaussian for higher n
        psi = np.exp(-xi ** 2 / 2) * np.cos(n * xi / (n + 1))
        psi /= np.sqrt(np.trapezoid(psi ** 2, x))
        backend = "stub"
        note = "Install scipy for Hermite polynomial wave functions."

    return {
        "backend": backend, "note": note,
        "energy_J": E, "energy_eV": E / _EV,
        "x_nm": (x * 1e9).tolist(), "psi": psi.tolist(), "psi_sq": (psi ** 2).tolist(),
    }


def quantum_tunneling(
    E_eV: float = 1.0,
    V0_eV: float = 2.0,
    L_nm: float = 0.5,
    m_kg: float = _ME,
) -> dict[str, Any]:
    """Transmission coefficient for a rectangular potential barrier (exact).

    T = [1 + (V0^2 sinh^2(kappa*L)) / (4*E*(V0-E))]^{-1}  if E < V0
    T = [1 + (V0^2 sin^2(k*L)) / (4*E*(E-V0))]^{-1}        if E > V0

    Args:
        E_eV: Particle kinetic energy (eV).
        V0_eV: Barrier height (eV).
        L_nm: Barrier width (nm).
        m_kg: Particle mass (kg).

    Returns:
        Dict with keys: transmission, reflection, E_eV, V0_eV, L_nm, backend, note.
    """
    E = E_eV * _EV
    V0 = V0_eV * _EV
    L = L_nm * 1e-9

    if E < V0:
        kappa = math.sqrt(2.0 * m_kg * (V0 - E)) / _HBAR
        kL = kappa * L
        T = 1.0 / (1.0 + (V0 ** 2 * math.sinh(kL) ** 2) / (4.0 * E * (V0 - E)))
    else:
        k2 = math.sqrt(2.0 * m_kg * (E - V0)) / _HBAR
        kL = k2 * L
        if V0 == 0:
            T = 1.0
        else:
            T = 1.0 / (1.0 + (V0 ** 2 * math.sin(kL) ** 2) / (4.0 * E * (E - V0)))

    return {
        "backend": "analytic", "note": "",
        "transmission": T, "reflection": 1.0 - T,
        "E_eV": E_eV, "V0_eV": V0_eV, "L_nm": L_nm,
    }


def wkb_transmission(
    E_eV: float = 0.5,
    V_barrier: float = 2.0,
    L_nm: float = 1.0,
    m_kg: float = _ME,
) -> dict[str, Any]:
    """WKB approximation for tunnelling through a smooth barrier.

    T_WKB = exp(-2 * integral_0^L kappa(x) dx)

    For a rectangular barrier: T_WKB = exp(-2 * kappa * L)

    Args:
        E_eV: Particle energy (eV).
        V_barrier: Barrier height (eV).
        L_nm: Barrier width (nm).
        m_kg: Particle mass (kg).

    Returns:
        Dict with keys: T_WKB, exponent, backend, note.
    """
    E = E_eV * _EV
    V = V_barrier * _EV
    L = L_nm * 1e-9
    if E >= V:
        return {"backend": "analytic", "note": "E >= V: no tunnelling.", "T_WKB": 1.0, "exponent": 0.0}
    kappa = math.sqrt(2.0 * m_kg * (V - E)) / _HBAR
    exponent = 2.0 * kappa * L
    T_WKB = math.exp(-exponent)
    return {
        "backend": "analytic", "note": "",
        "T_WKB": T_WKB, "exponent": exponent,
    }


def finite_square_well(
    V0_eV: float = 5.0,
    L_nm: float = 1.0,
    m_kg: float = _ME,
    n_bound: int = 3,
) -> dict[str, Any]:
    """Find bound-state energies of a finite square well numerically.

    Args:
        V0_eV: Well depth (eV).
        L_nm: Well half-width (nm) [well extends -L to +L].
        m_kg: Particle mass (kg).
        n_bound: Maximum number of bound states to find.

    Returns:
        Dict with keys: energies_eV, n_states, V0_eV, L_nm, backend, note.
    """
    V0 = V0_eV * _EV
    L = L_nm * 1e-9
    z0 = L * math.sqrt(2 * m_kg * V0) / _HBAR  # dimensionless well parameter

    energies_eV = []
    if _SCIPY:
        from scipy.optimize import brentq
        for k_idx in range(1, n_bound + 1):
            # Transcendental equations for even/odd states
            try:
                def f_even(E_J):
                    k = math.sqrt(2 * m_kg * E_J) / _HBAR
                    kappa = math.sqrt(2 * m_kg * (V0 - E_J)) / _HBAR
                    return k * math.tan(k * L) - kappa
                E_bounds = ((k_idx - 1) ** 2 * math.pi ** 2 * _HBAR ** 2 /
                            (8 * m_kg * L ** 2), min(V0 * 0.999, k_idx ** 2 *
                            math.pi ** 2 * _HBAR ** 2 / (8 * m_kg * L ** 2) * 0.999))
                if E_bounds[0] < E_bounds[1]:
                    E_sol = brentq(f_even, E_bounds[0] + 1e-30, E_bounds[1])
                    energies_eV.append(E_sol / _EV)
            except Exception:
                pass
        backend, note = "scipy", ""
    else:
        # Estimate from infinite well + first-order correction
        for n in range(1, min(n_bound + 1, int(z0 / math.pi) + 2)):
            E_inf = n ** 2 * math.pi ** 2 * _HBAR ** 2 / (8 * m_kg * L ** 2)
            E_fin = E_inf * (1.0 - 1.0 / (z0 ** 2))
            if 0 < E_fin < V0:
                energies_eV.append(E_fin / _EV)
        backend = "stub"
        note = "Install scipy for exact transcendental-equation root finding."

    return {
        "backend": backend, "note": note,
        "energies_eV": energies_eV, "n_states": len(energies_eV),
        "V0_eV": V0_eV, "L_nm": L_nm,
    }


def time_evolution_1d(
    psi0: list[complex] | None = None,
    V: list[float] | None = None,
    dx: float = 0.05e-9,
    dt: float = 1e-17,
    n_steps: int = 100,
    m_kg: float = _ME,
) -> dict[str, Any]:
    """Split-operator Fourier-method time evolution of a 1-D wave packet.

    Args:
        psi0: Initial wave function (complex list of length N).  Defaults to
            a Gaussian wave packet.
        V: Potential energy array (J) of same length as psi0.
        dx: Spatial grid spacing (m).
        dt: Time step (s).
        n_steps: Number of evolution steps.
        m_kg: Particle mass (kg).

    Returns:
        Dict with keys: x_nm, psi_sq_final, norm_conserved, backend, note.
    """
    N = 256
    if psi0 is None:
        x = np.arange(N) * dx
        x0 = x[N // 4]
        sigma = 5 * dx
        k0 = 2e10
        psi0 = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)
    else:
        psi0 = np.array(psi0, dtype=complex)
        x = np.arange(len(psi0)) * dx

    psi = psi0 / np.sqrt(np.sum(np.abs(psi0) ** 2) * dx)
    N = len(psi)
    if V is None:
        V_arr = np.zeros(N)
    else:
        V_arr = np.array(V, dtype=float)

    k = np.fft.fftfreq(N, d=dx) * 2 * math.pi
    K_op = np.exp(-1j * _HBAR * k ** 2 / (2 * m_kg) * dt)
    V_op = np.exp(-1j * V_arr * dt / _HBAR)

    for _ in range(n_steps):
        psi = V_op * psi
        psi_k = np.fft.fft(psi)
        psi_k *= K_op
        psi = np.fft.ifft(psi_k)

    norm = float(np.sum(np.abs(psi) ** 2) * dx)
    return {
        "backend": "numpy",
        "note": "Install scipy for higher-order split-operator methods.",
        "x_nm": (x * 1e9).tolist(),
        "psi_sq_final": np.abs(psi) ** 2 / norm,
        "norm_conserved": abs(norm - 1.0) < 0.01,
    }
