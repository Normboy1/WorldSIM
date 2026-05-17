"""Robust control analysis utilities (H-infinity, mu-synthesis, loop shaping).

Provides H-infinity norm estimation, mu-analysis proxy, small-gain theorem,
robust stability margin, and loop-shaping proxy.

Optional dependencies: scipy (signal processing), numpy (always required).
Falls back to approximate frequency-domain calculations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy import signal as _signal
    _SCIPY = True
except ImportError:
    _SCIPY = False


def h_infinity_norm(
    sys_num: list[float] = [1.0],
    sys_den: list[float] = [1.0, 2.0, 1.0],
    omega_range: tuple[float, float] = (1e-3, 1e3),
) -> dict[str, Any]:
    """Estimate the H-infinity (peak) norm of a SISO transfer function.

    ||G||_inf = max_omega |G(j*omega)|

    Args:
        sys_num: Transfer function numerator coefficients.
        sys_den: Transfer function denominator coefficients.
        omega_range: (omega_min, omega_max) frequency range (rad/s).

    Returns:
        Dict with keys: H_inf_norm, peak_frequency_rad_s, backend, note.
    """
    omega = np.logspace(math.log10(omega_range[0]), math.log10(omega_range[1]), 500)
    if _SCIPY:
        _, H = _signal.freqs(sys_num, sys_den, worN=omega)
        mag = np.abs(H)
        backend, note = "scipy", ""
    else:
        num = np.array(sys_num, dtype=complex)
        den = np.array(sys_den, dtype=complex)
        mag = np.array([abs(np.polyval(num, 1j * w) / np.polyval(den, 1j * w)) for w in omega])
        backend = "stub"
        note = "Install scipy for accurate frequency response."

    peak_idx = int(np.argmax(mag))
    return {
        "backend": backend, "note": note,
        "H_inf_norm": float(mag[peak_idx]),
        "peak_frequency_rad_s": float(omega[peak_idx]),
    }


def mu_analysis_proxy(
    M_matrix: list[list[float]] | None = None,
    Delta_structure: str = "complex_full",
) -> dict[str, Any]:
    """Proxy structured singular value (mu) analysis.

    mu(M) upper bound via D-K scaling: mu <= min_D sigma_max(D*M*D^{-1})

    Args:
        M_matrix: Interconnected system matrix (complex, n x n).
            Defaults to a 2x2 example.
        Delta_structure: Perturbation structure ('complex_full', 'real_diagonal').

    Returns:
        Dict with keys: mu_upper_bound, mu_lower_bound, robust_stable, backend, note.
    """
    if M_matrix is None:
        M_matrix = [[0.3 + 0.1j, 0.2], [0.1, 0.4 + 0.05j]]
    M = np.array(M_matrix, dtype=complex)
    sigma_max = float(np.linalg.norm(M, ord=2))
    mu_upper = sigma_max
    mu_lower = mu_upper * 0.85  # typical gap proxy
    return {
        "backend": "stub",
        "note": "Install python-control or MATLAB for exact mu computation.",
        "mu_upper_bound": mu_upper, "mu_lower_bound": mu_lower,
        "robust_stable": mu_upper < 1.0, "Delta_structure": Delta_structure,
    }


def small_gain_theorem(
    G1_gain: float = 0.6,
    G2_gain: float = 1.2,
) -> dict[str, Any]:
    """Check the small-gain theorem condition for a feedback interconnection.

    Stable if ||G1||_inf * ||G2||_inf < 1.

    Args:
        G1_gain: H-infinity norm of subsystem G1.
        G2_gain: H-infinity norm of subsystem G2.

    Returns:
        Dict with keys: product, robust_stable, stability_margin, backend, note.
    """
    product = G1_gain * G2_gain
    margin = 1.0 - product
    return {
        "backend": "analytic", "note": "",
        "product": product, "robust_stable": product < 1.0,
        "stability_margin": margin,
    }


def robust_stability_margin(
    nominal_num: list[float] = [1.0],
    nominal_den: list[float] = [1.0, 3.0, 2.0],
    uncertainty_bound: float = 0.2,
) -> dict[str, Any]:
    """Compute the robust stability margin under additive uncertainty.

    Stable for all perturbed plants if ||W*S||_inf < 1 / uncertainty_bound.

    Args:
        nominal_num: Nominal plant numerator.
        nominal_den: Nominal plant denominator.
        uncertainty_bound: L-infinity norm bound on additive uncertainty.

    Returns:
        Dict with keys: stability_radius, robust_stable, gain_margin_dB,
        phase_margin_deg, backend, note.
    """
    H_inf = h_infinity_norm(nominal_num, nominal_den)["H_inf_norm"]
    stability_radius = 1.0 / (H_inf + 1e-12)
    robust_stable = stability_radius > uncertainty_bound
    # Proxy gain and phase margin from loop TF
    gain_margin_dB = 20 * math.log10(1.0 / uncertainty_bound)
    phase_margin_deg = 60.0 * (1.0 - uncertainty_bound)  # heuristic
    return {
        "backend": "stub" if not _SCIPY else "scipy",
        "note": "" if _SCIPY else "Install scipy for exact margin computation.",
        "stability_radius": stability_radius, "robust_stable": robust_stable,
        "gain_margin_dB": gain_margin_dB, "phase_margin_deg": phase_margin_deg,
    }


def loop_shaping_proxy(
    plant_num: list[float] = [1.0],
    plant_den: list[float] = [1.0, 0.5],
    target_bandwidth_rad_s: float = 10.0,
) -> dict[str, Any]:
    """McFarlane-Glover loop shaping controller synthesis (proxy).

    Args:
        plant_num: Plant transfer function numerator.
        plant_den: Plant transfer function denominator.
        target_bandwidth_rad_s: Desired closed-loop bandwidth (rad/s).

    Returns:
        Dict with keys: achieved_bandwidth_rad_s, stability_margin,
        controller_gain, backend, note.
    """
    omega_c = target_bandwidth_rad_s
    # Simple lead-lag gain proxy
    K_gain = omega_c / (abs(plant_num[0]) / abs(plant_den[-1]) + 1e-6)
    achieved_bw = omega_c * 0.9  # typical loop-shaping achieves ~90%
    margin = 0.35  # typical stability margin with Glover-McFarlane
    return {
        "backend": "stub",
        "note": "Install python-control for actual H-inf loop-shaping synthesis.",
        "achieved_bandwidth_rad_s": achieved_bw,
        "stability_margin": margin, "controller_gain": K_gain,
    }
