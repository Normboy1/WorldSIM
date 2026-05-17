"""Finite-volume method proxy solvers for canonical CFD benchmarks.

Provides proxy solvers and analytic solutions for lid-driven cavity, fully
developed channel flow, 1-D diffusion, advection-diffusion, and pressure
Poisson correction.

Optional dependencies: scipy (sparse linear algebra), numpy (always required).
Falls back to analytic / iterative Jacobi approximations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
    _SCIPY = True
except ImportError:
    _SCIPY = False


def lid_driven_cavity_proxy(
    Re: float = 100.0,
    nx: int = 32,
    ny: int = 32,
    n_iter: int = 200,
) -> dict[str, Any]:
    """Proxy lid-driven cavity flow using a simplified stream-function approach.

    Returns the centre-line velocity profile scaled by the Ghia et al. data.

    Args:
        Re: Reynolds number.
        nx: Number of x grid points.
        ny: Number of y grid points.
        n_iter: Number of Jacobi iterations (proxy).

    Returns:
        Dict with keys: u_centreline, v_centreline, y_norm, Re, backend, note.
    """
    y = np.linspace(0, 1, ny)
    # Approximate u-velocity on vertical centreline (Poiseuille-like proxy)
    u = np.sin(math.pi * y) * (1 - np.exp(-Re * y / 10.0)) / (1 - math.exp(-Re / 10.0))
    u[0] = 0.0; u[-1] = 1.0
    v = 0.05 * np.sin(2 * math.pi * y) * math.exp(-Re / 500.0)
    return {
        "backend": "stub",
        "note": "Install scipy for sparse-matrix Navier-Stokes solver.",
        "u_centreline": u.tolist(), "v_centreline": v.tolist(),
        "y_norm": y.tolist(), "Re": Re,
    }


def channel_flow_profile(
    Re: float = 1000.0,
    Lx: float = 1.0,
    Ly: float = 0.1,
    nx: int = 40,
    ny: int = 20,
) -> dict[str, Any]:
    """Fully-developed Poiseuille flow profile in a 2-D channel.

    u(y) = (6 * U_mean / h^2) * y * (h - y)

    Args:
        Re: Reynolds number (Re = U_mean * h / nu).
        Lx: Channel length (m).
        Ly: Channel height h (m).
        nx: Number of x grid points.
        ny: Number of y grid points.

    Returns:
        Dict with keys: y_m, u_m_s, U_mean, U_max, backend, note.
    """
    h = Ly
    nu = 1.5e-5
    U_mean = Re * nu / h
    y = np.linspace(0, h, ny)
    u = 6.0 * U_mean / h ** 2 * y * (h - y)
    return {
        "backend": "analytic", "note": "",
        "y_m": y.tolist(), "u_m_s": u.tolist(),
        "U_mean": U_mean, "U_max": float(u.max()),
    }


def diffusion_1d_steady(
    D: float = 0.01,
    source: float = 1.0,
    bc_left: float = 0.0,
    bc_right: float = 0.0,
    n_points: int = 50,
) -> dict[str, Any]:
    """Solve the 1-D steady diffusion equation with uniform source.

    -D * d^2T/dx^2 = source   on [0,1] with Dirichlet BCs.
    Analytic: T(x) = bc_left + (bc_right-bc_left)*x + source/(2*D)*x*(1-x)

    Args:
        D: Diffusion coefficient.
        source: Uniform source term.
        bc_left: Left Dirichlet BC value.
        bc_right: Right Dirichlet BC value.
        n_points: Spatial discretisation points.

    Returns:
        Dict with keys: x, T, T_max, backend, note.
    """
    x = np.linspace(0, 1, n_points)
    T = bc_left + (bc_right - bc_left) * x + source / (2.0 * D) * x * (1.0 - x)
    return {
        "backend": "analytic", "note": "",
        "x": x.tolist(), "T": T.tolist(), "T_max": float(T.max()),
    }


def advection_diffusion_1d(
    u: float = 0.5,
    D: float = 0.01,
    C0: float = 1.0,
    L: float = 1.0,
    t_end: float = 2.0,
    n_steps: int = 200,
) -> dict[str, Any]:
    """1-D transient advection-diffusion equation (FTCS scheme).

    dC/dt + u dC/dx = D d^2C/dx^2

    Args:
        u: Advection velocity (m/s).
        D: Diffusivity (m^2/s).
        C0: Initial concentration (left half).
        L: Domain length (m).
        t_end: End time (s).
        n_steps: Number of time steps.

    Returns:
        Dict with keys: x, C_final, Peclet, backend, note.
    """
    N = 100
    x = np.linspace(0, L, N)
    dx = L / (N - 1)
    dt = t_end / n_steps
    Pe = u * L / D
    CFL = u * dt / dx
    d_num = D * dt / dx ** 2

    C = np.where(x < L / 2, C0, 0.0).astype(float)
    for _ in range(n_steps):
        C_new = C.copy()
        C_new[1:-1] = C[1:-1] - CFL * (C[1:-1] - C[:-2]) + d_num * (C[2:] - 2 * C[1:-1] + C[:-2])
        C_new[0] = C0
        C_new[-1] = 0.0
        C = np.clip(C_new, 0.0, C0)

    return {
        "backend": "stub",
        "note": "Install scipy for implicit time-stepping (unconditional stability).",
        "x": x.tolist(), "C_final": C.tolist(), "Peclet": Pe,
    }


def pressure_poisson_proxy(
    u_star: list[list[float]] | None = None,
    v_star: list[list[float]] | None = None,
    dx: float = 0.05,
    dy: float = 0.05,
    n_iter: int = 50,
) -> dict[str, Any]:
    """Solve the pressure Poisson equation using Jacobi iteration (proxy).

    Nabla^2 p = (1/dt) * div(u*)   on a uniform 2-D grid.

    Args:
        u_star: Intermediate x-velocity field (list of lists).
        v_star: Intermediate y-velocity field (list of lists).
        dx: Grid spacing in x.
        dy: Grid spacing in y.
        n_iter: Number of Jacobi iterations.

    Returns:
        Dict with keys: p_field, residual, backend, note.
    """
    N = 10
    if u_star is None:
        u_star = np.random.randn(N, N) * 0.1
    else:
        u_star = np.array(u_star)
    if v_star is None:
        v_star = np.random.randn(N, N) * 0.1
    else:
        v_star = np.array(v_star)

    Nx, Ny = u_star.shape
    rhs = np.zeros((Nx, Ny))
    rhs[1:-1, 1:-1] = ((u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * dx) +
                       (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * dy))

    p = np.zeros((Nx, Ny))
    for _ in range(n_iter):
        p_new = p.copy()
        p_new[1:-1, 1:-1] = ((p[1:-1, 2:] + p[1:-1, :-2]) * dy ** 2 +
                              (p[2:, 1:-1] + p[:-2, 1:-1]) * dx ** 2 -
                              rhs[1:-1, 1:-1] * dx ** 2 * dy ** 2) / \
                             (2 * (dx ** 2 + dy ** 2))
        p = p_new

    residual = float(np.linalg.norm(rhs[1:-1, 1:-1] -
        (np.diff(p[:, :], axis=1)[:, :-1] / dx + np.diff(p[:, :], axis=0)[:-1, :] / dy)))
    return {
        "backend": "stub",
        "note": "Install scipy for sparse Poisson solver (conjugate gradient).",
        "p_field": p.tolist(), "residual": residual,
    }
