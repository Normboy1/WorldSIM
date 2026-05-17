"""Ecosystem-level simulation utilities.

Covers food-web stability analysis, species-area relationships, biodiversity
indices, nutrient cycling, and trophic cascades.  Useful for community ecology
and conservation biology research.

Optional dependencies: scipy (eigenvalue methods), numpy (always required).
Falls back to heuristic approximations when scipy is unavailable.
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


def food_web_stability(
    adjacency_matrix: list[list[float]] | None = None,
    interaction_strengths: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Assess local stability of a food web via community matrix eigenanalysis.

    The community matrix M = A * S where A is the adjacency matrix and S
    contains interaction strengths.  Stable if max(Re(eigenvalues)) < 0.

    Args:
        adjacency_matrix: Binary adjacency matrix (n x n).  Defaults to a
            4-species example.
        interaction_strengths: Signed interaction strength matrix (n x n).
            Defaults to a plausible random example.

    Returns:
        Dict with keys: max_real_eigenvalue, is_stable, connectance, n_species,
        eigenvalues_real, backend, note.
    """
    if adjacency_matrix is None:
        adjacency_matrix = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ]
    if interaction_strengths is None:
        interaction_strengths = [
            [-0.5, 0.0,  0.0,  0.0],
            [ 0.3, -0.4, 0.0,  0.0],
            [ 0.2,  0.4, -0.6, 0.0],
            [ 0.0,  0.3,  0.5, -0.3],
        ]

    A = np.array(adjacency_matrix, dtype=float)
    S = np.array(interaction_strengths, dtype=float)
    M = A * S
    n = len(A)
    links = int(np.sum(A))
    connectance = links / (n * (n - 1)) if n > 1 else 0.0

    eigs = np.linalg.eigvals(M)
    max_re = float(np.max(np.real(eigs)))

    return {
        "backend": "numpy",
        "note": "",
        "max_real_eigenvalue": max_re,
        "is_stable": max_re < 0,
        "connectance": connectance,
        "n_species": n,
        "eigenvalues_real": np.real(eigs).tolist(),
    }


def species_area_curve(
    A: float = 100.0,
    c: float = 0.5,
    z: float = 0.25,
) -> dict[str, Any]:
    """Compute species richness from area using the species-area relationship.

    S = c * A^z  (Power-law / MacArthur-Wilson form)

    Args:
        A: Habitat area (km^2 or arbitrary units).
        c: Species richness constant (varies by taxon and region).
        z: Exponent (typically 0.2-0.35 for islands; ~0.12 for mainland).

    Returns:
        Dict with keys: species_richness, area, c, z, backend, note.
    """
    S = c * (A ** z)
    return {
        "backend": "analytic",
        "note": "",
        "species_richness": S,
        "area": A,
        "c": c,
        "z": z,
    }


def biodiversity_index(
    species_counts: list[int] | None = None,
    method: str = "shannon",
) -> dict[str, Any]:
    """Calculate a biodiversity index from species abundance data.

    Args:
        species_counts: List of individual counts per species.
            Defaults to a 10-species example.
        method: One of 'shannon', 'simpson', 'margalef', 'evenness'.

    Returns:
        Dict with keys: index, method, n_species, total_individuals, backend, note.
    """
    if species_counts is None:
        species_counts = [120, 85, 60, 40, 30, 20, 15, 10, 5, 3]

    counts = np.array(species_counts, dtype=float)
    N = counts.sum()
    n_sp = len(counts)
    p = counts / N

    if method == "shannon":
        idx = float(-np.sum(p * np.log(p + 1e-15)))
    elif method == "simpson":
        idx = float(1.0 - np.sum(p ** 2))
    elif method == "margalef":
        idx = (n_sp - 1) / math.log(N + 1) if N > 1 else 0.0
    elif method == "evenness":
        H = float(-np.sum(p * np.log(p + 1e-15)))
        idx = H / math.log(n_sp) if n_sp > 1 else 1.0
    else:
        idx = float(-np.sum(p * np.log(p + 1e-15)))

    return {
        "backend": "numpy",
        "note": "",
        "index": idx,
        "method": method,
        "n_species": n_sp,
        "total_individuals": int(N),
    }


def nutrient_cycling(
    N_pool: float = 500.0,
    uptake_rate: float = 0.05,
    mineralization_rate: float = 0.03,
    t_end: float = 100.0,
) -> dict[str, Any]:
    """Simple two-compartment nutrient cycling model (available / organic pool).

    dN_avail/dt = mineralization * N_org - uptake * N_avail
    dN_org/dt   = uptake * N_avail - mineralization * N_org

    Args:
        N_pool: Total nutrient pool (sum of available + organic, arbitrary units).
        uptake_rate: Rate at which available nutrients are taken up by biomass.
        mineralization_rate: Rate of decomposition returning nutrients to
            available pool.
        t_end: Simulation end time (days).

    Returns:
        Dict with keys: t, N_available, N_organic, equilibrium_available,
        backend, note.
    """
    n_steps = 400
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    N_av = np.zeros(n_steps + 1)
    N_or = np.zeros(n_steps + 1)
    N_av[0] = N_pool * 0.3
    N_or[0] = N_pool * 0.7

    for i in range(n_steps):
        dAv = (mineralization_rate * N_or[i] - uptake_rate * N_av[i]) * dt
        dOr = (uptake_rate * N_av[i] - mineralization_rate * N_or[i]) * dt
        N_av[i + 1] = max(N_av[i] + dAv, 0.0)
        N_or[i + 1] = max(N_or[i] + dOr, 0.0)

    eq_av = N_pool * mineralization_rate / (uptake_rate + mineralization_rate)

    return {
        "backend": "stub",
        "note": "Install scipy for stiff ODE integration.",
        "t": t_arr.tolist(),
        "N_available": N_av.tolist(),
        "N_organic": N_or.tolist(),
        "equilibrium_available": eq_av,
    }


def trophic_cascade(
    trophic_levels: int = 3,
    biomass_0: list[float] | None = None,
    coupling_strengths: list[float] | None = None,
    t_end: float = 80.0,
) -> dict[str, Any]:
    """Simulate a linear food chain trophic cascade.

    Each level i preys on level i-1.  dB_i/dt = e_i * g_i * B_{i-1} * B_i
    - d_i * B_i, simplified chain ODE.

    Args:
        trophic_levels: Number of trophic levels in the chain (2-5).
        biomass_0: Initial biomass for each level.  Defaults to
            [100, 20, 5] for 3 levels.
        coupling_strengths: Interaction coefficients between adjacent levels.
            Defaults to [0.05, 0.08] for 3 levels.
        t_end: Simulation duration (arbitrary time units).

    Returns:
        Dict with keys: t, biomass_trajectory, level_names, backend, note.
    """
    if biomass_0 is None:
        biomass_0 = [100.0 / (2 ** i) for i in range(trophic_levels)]
    if coupling_strengths is None:
        coupling_strengths = [0.05] * (trophic_levels - 1)

    n = trophic_levels
    n_steps = 500
    dt = t_end / n_steps
    t_arr = np.linspace(0, t_end, n_steps + 1)
    B = np.zeros((n_steps + 1, n))
    B[0] = biomass_0[:n]

    growth = [0.4 / (i + 1) for i in range(n)]
    death = [0.05 * (i + 1) for i in range(n)]

    for step in range(n_steps):
        b = B[step].copy()
        db = np.zeros(n)
        db[0] = growth[0] * b[0] - (coupling_strengths[0] * b[0] * b[1] if n > 1 else 0)
        for i in range(1, n):
            pred_gain = coupling_strengths[i - 1] * b[i - 1] * b[i]
            pred_loss = (coupling_strengths[i] * b[i] * b[i + 1] if i < n - 1 else 0)
            db[i] = 0.3 * pred_gain - death[i] * b[i] - pred_loss
        B[step + 1] = np.maximum(B[step] + db * dt, 0.0)

    level_names = ["primary_producer"] + \
        [f"consumer_L{i}" for i in range(1, n - 1)] + \
        (["apex_predator"] if n > 1 else [])

    return {
        "backend": "stub",
        "note": "Install scipy for stiff ODE integration.",
        "t": t_arr.tolist(),
        "biomass_trajectory": B.tolist(),
        "level_names": level_names,
    }
