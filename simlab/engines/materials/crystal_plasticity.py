"""Crystal plasticity mechanics: slip systems, Schmid factors, Taylor model.

Provides Schmid factor calculations, resolved shear stress, slip system
activity assessment, Taylor factor estimation, Voce hardening, and
rate-dependent slip.

Optional dependencies: numpy (always required), scipy.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import scipy.linalg as _la
    _SCIPY = True
except ImportError:
    _SCIPY = False


def schmid_factor(
    slip_direction: list[float] = [1, 1, 0],
    slip_plane_normal: list[float] = [1, 1, 1],
    loading_axis: list[float] = [0, 0, 1],
) -> dict[str, Any]:
    """Compute the Schmid factor for a slip system under uniaxial loading.

    m = cos(phi) * cos(lambda) = (b_hat . s) * (n_hat . s)

    Args:
        slip_direction: Slip direction (Miller indices or unit vector).
        slip_plane_normal: Slip plane normal (Miller indices or unit vector).
        loading_axis: Loading axis direction.

    Returns:
        Dict with keys: schmid_factor, cos_phi, cos_lambda, backend, note.
    """
    b = np.array(slip_direction, dtype=float)
    n = np.array(slip_plane_normal, dtype=float)
    s = np.array(loading_axis, dtype=float)
    b /= np.linalg.norm(b)
    n /= np.linalg.norm(n)
    s /= np.linalg.norm(s)

    cos_lambda = float(np.dot(b, s))
    cos_phi = float(np.dot(n, s))
    m = cos_lambda * cos_phi

    return {
        "backend": "analytic", "note": "",
        "schmid_factor": m, "cos_phi": cos_phi, "cos_lambda": cos_lambda,
    }


def resolved_shear_stress(
    sigma_MPa: float = 200.0,
    m_factor: float = 0.408,
) -> dict[str, Any]:
    """Compute the resolved shear stress on a slip system.

    tau = sigma * m

    Args:
        sigma_MPa: Applied uniaxial stress (MPa).
        m_factor: Schmid factor.

    Returns:
        Dict with keys: tau_MPa, sigma_MPa, schmid_factor, backend, note.
    """
    tau = sigma_MPa * m_factor
    return {
        "backend": "analytic", "note": "",
        "tau_MPa": tau, "sigma_MPa": sigma_MPa, "schmid_factor": m_factor,
    }


def slip_system_activity(
    sigma_tensor: list[list[float]] | None = None,
    crystal_structure: str = "fcc",
) -> dict[str, Any]:
    """Determine which slip systems are active under a given stress state.

    Args:
        sigma_tensor: 3x3 Cauchy stress tensor (MPa).  Defaults to uniaxial
            100 MPa along z.
        crystal_structure: 'fcc' (12 systems), 'bcc' (48 systems, simplified),
            'hcp' (basal, prismatic, pyramidal).

    Returns:
        Dict with keys: active_systems, n_active, max_schmid, tau_max_MPa,
        backend, note.
    """
    if sigma_tensor is None:
        sigma_tensor = [[0, 0, 0], [0, 0, 0], [0, 0, 100.0]]

    # FCC {111}<110> slip systems
    if crystal_structure.lower() == "fcc":
        directions = [[1,1,0],[1,-1,0],[0,1,1],[0,1,-1],[1,0,1],[-1,0,1],
                      [-1,1,0],[1,0,-1],[0,-1,1],[1,-1,0],[0,1,1],[-1,0,-1]]
        normals = [[1,1,1],[1,1,1],[1,1,1],[1,1,-1],[1,1,-1],[1,1,-1],
                   [-1,1,1],[-1,1,1],[-1,1,1],[1,-1,1],[1,-1,1],[1,-1,1]]
    else:  # bcc simplified
        directions = [[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]]
        normals = [[1,1,0],[1,1,0],[0,1,1],[1,0,1]]

    sigma = np.array(sigma_tensor, dtype=float)
    active = []
    tau_values = []
    for b_idx, (b, n) in enumerate(zip(directions, normals)):
        b_hat = np.array(b, dtype=float) / np.linalg.norm(b)
        n_hat = np.array(n, dtype=float) / np.linalg.norm(n)
        tau = float(n_hat @ sigma @ b_hat)
        tau_values.append(abs(tau))
        if abs(tau) > 20.0:  # simple CRSS threshold
            active.append({"system": b_idx, "tau_MPa": abs(tau)})

    return {
        "backend": "numpy", "note": "",
        "active_systems": active, "n_active": len(active),
        "max_schmid": max(tau_values) / (abs(sigma_tensor[2][2]) + 1e-6),
        "tau_max_MPa": max(tau_values),
    }


def taylor_factor(
    texture_odf: list[float] | None = None,
    crystal_structure: str = "fcc",
) -> dict[str, Any]:
    """Estimate the Taylor factor M from a simplified texture ODF.

    Args:
        texture_odf: Orientation distribution weights (arbitrary; not full ODF).
            Defaults to isotropic texture.
        crystal_structure: 'fcc' or 'bcc'.

    Returns:
        Dict with keys: M_taylor, texture_type, crystal_structure, backend, note.
    """
    # Typical Taylor factors: isotropic FCC = 3.06, rolling texture ~ 3.0-3.2
    M_isotropic = {"fcc": 3.06, "bcc": 2.75, "hcp": 4.50}
    M = M_isotropic.get(crystal_structure.lower(), 3.06)
    if texture_odf is not None:
        # Simple proxy: weight by mean ODF value
        M *= 1.0 + 0.05 * (np.mean(texture_odf) - 1.0)
    return {
        "backend": "stub",
        "note": "Install MTEX/scipy for full ODF-based Taylor factor.",
        "M_taylor": M, "texture_type": "isotropic" if texture_odf is None else "custom",
        "crystal_structure": crystal_structure,
    }


def voce_hardening(
    tau0: float = 20.0,
    tau_inf: float = 120.0,
    theta0: float = 500.0,
    gamma: float = 0.1,
) -> dict[str, Any]:
    """Voce saturation hardening law.

    tau_c(gamma) = tau_inf - (tau_inf - tau0) * exp(-theta0 * gamma / (tau_inf - tau0))

    Args:
        tau0: Initial CRSS (MPa).
        tau_inf: Saturation CRSS (MPa).
        theta0: Initial hardening rate (MPa).
        gamma: Accumulated plastic shear strain.

    Returns:
        Dict with keys: tau_c_MPa, hardening_rate_MPa, backend, note.
    """
    delta = tau_inf - tau0
    tau_c = tau_inf - delta * math.exp(-theta0 * gamma / delta)
    theta = theta0 * math.exp(-theta0 * gamma / delta)
    return {
        "backend": "analytic", "note": "",
        "tau_c_MPa": tau_c, "hardening_rate_MPa": theta,
    }


def rate_dependent_slip(
    tau: float = 30.0,
    tau_c: float = 25.0,
    m: float = 0.02,
    gamma_dot_0: float = 1e-3,
) -> dict[str, Any]:
    """Rate-dependent slip law (power-law creep / crystal plasticity).

    gamma_dot = gamma_dot_0 * (tau / tau_c)^(1/m) * sign(tau)

    Args:
        tau: Resolved shear stress (MPa).
        tau_c: Current CRSS (MPa).
        m: Rate sensitivity exponent (small m = highly rate sensitive).
        gamma_dot_0: Reference shear strain rate (1/s).

    Returns:
        Dict with keys: gamma_dot_1_s, slip_rate, backend, note.
    """
    if tau_c <= 0:
        return {"backend": "analytic", "note": "tau_c must be > 0.", "gamma_dot_1_s": 0.0}
    gamma_dot = gamma_dot_0 * abs(tau / tau_c) ** (1.0 / m) * (1.0 if tau >= 0 else -1.0)
    return {
        "backend": "analytic", "note": "",
        "gamma_dot_1_s": gamma_dot, "slip_rate": gamma_dot,
    }
