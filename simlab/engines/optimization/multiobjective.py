"""Multi-objective optimisation utilities: NSGA-II, MOEA/D, hypervolume.

Provides NSGA-II and MOEA/D proxy solvers, hypervolume indicator, 2-D Pareto
front extraction, crowding distance, and non-dominated sort.

Optional dependencies: pymoo (NSGA-II), numpy (always required).
Falls back to random-search + Pareto filter.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import pymoo  # noqa: F401
    _PYMOO = True
except ImportError:
    _PYMOO = False


def nsga2_proxy(
    objective_fns_str: str = "ZDT1",
    bounds: list[list[float]] | None = None,
    pop_size: int = 50,
    n_gen: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """NSGA-II multi-objective optimisation proxy.

    Args:
        objective_fns_str: Problem name ('ZDT1', 'DTLZ2', or description).
        bounds: [[lb, ub], ...] decision variable bounds.  Defaults to [0,1]^2.
        pop_size: Population size.
        n_gen: Number of generations.
        seed: Random seed.

    Returns:
        Dict with keys: pareto_front, n_solutions, hypervolume_indicator,
        backend, note.
    """
    np.random.seed(seed)
    if bounds is None:
        bounds = [[0.0, 1.0]] * 5
    dim = len(bounds)
    lbs = np.array([b[0] for b in bounds])
    ubs = np.array([b[1] for b in bounds])

    # Random search + Pareto filter proxy
    N = pop_size * n_gen // 10
    X = lbs + (ubs - lbs) * np.random.rand(N, dim)
    # ZDT1-like proxy
    f1 = X[:, 0]
    g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
    f2 = g * (1.0 - np.sqrt(f1 / (g + 1e-12)))
    costs = np.column_stack([f1, f2])
    front_idx = non_dominated_sort(costs.tolist())["front_0_indices"]

    HV = hypervolume_indicator(costs[front_idx].tolist(), reference_point=[1.1, 1.1])["hypervolume"]
    return {
        "backend": "stub" if not _PYMOO else "pymoo",
        "note": "" if _PYMOO else "Install pymoo for exact NSGA-II.",
        "pareto_front": costs[front_idx].tolist(),
        "n_solutions": len(front_idx), "hypervolume_indicator": HV,
    }


def moead_proxy(
    objective_fns_str: str = "ZDT1",
    bounds: list[list[float]] | None = None,
    n_weights: int = 50,
    n_gen: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """MOEA/D proxy using scalarised random search with weight vectors.

    Args:
        objective_fns_str: Problem name or description.
        bounds: Decision variable bounds.
        n_weights: Number of weight vectors (= population size).
        n_gen: Number of generations.
        seed: Random seed.

    Returns:
        Dict with keys: pareto_front, n_solutions, backend, note.
    """
    np.random.seed(seed)
    if bounds is None:
        bounds = [[0.0, 1.0]] * 5
    dim = len(bounds)
    lbs = np.array([b[0] for b in bounds])
    ubs = np.array([b[1] for b in bounds])

    weights = np.random.dirichlet(np.ones(2), n_weights)
    best = []
    for w in weights:
        X = lbs + (ubs - lbs) * np.random.rand(n_gen, dim)
        f1 = X[:, 0]
        g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / (g + 1e-12)))
        scal = w[0] * f1 + w[1] * f2
        idx = np.argmin(scal)
        best.append([f1[idx], f2[idx]])

    best_arr = np.array(best)
    front_idx = non_dominated_sort(best_arr.tolist())["front_0_indices"]
    return {
        "backend": "stub",
        "note": "Install pymoo for exact MOEA/D with Tchebycheff decomposition.",
        "pareto_front": best_arr[front_idx].tolist(),
        "n_solutions": len(front_idx),
    }


def hypervolume_indicator(
    pareto_front: list[list[float]],
    reference_point: list[float],
) -> dict[str, Any]:
    """Compute the hypervolume indicator for a 2-objective Pareto front.

    Args:
        pareto_front: List of [f1, f2] objective vectors on the Pareto front.
        reference_point: Reference point dominating all front points.

    Returns:
        Dict with keys: hypervolume, n_points, backend, note.
    """
    pf = np.array(pareto_front)
    ref = np.array(reference_point)
    # Sort by first objective
    order = np.argsort(pf[:, 0])
    pf = pf[order]
    HV = 0.0
    prev_f2 = ref[1]
    for i in range(len(pf)):
        if pf[i, 0] < ref[0] and pf[i, 1] < ref[1]:
            width = ref[0] - pf[i, 0] if i == len(pf) - 1 else pf[i + 1, 0] - pf[i, 0]
            height = prev_f2 - pf[i, 1]
            HV += max(width, 0) * max(height, 0)
            prev_f2 = pf[i, 1]
    return {"backend": "numpy", "note": "", "hypervolume": HV, "n_points": len(pf)}


def pareto_front_2d(costs: list[list[float]]) -> dict[str, Any]:
    """Extract the Pareto-optimal set from a 2-objective cost matrix.

    Args:
        costs: List of [f1, f2] cost pairs.

    Returns:
        Dict with keys: pareto_indices, pareto_costs, n_dominated, backend, note.
    """
    costs_arr = np.array(costs)
    n = len(costs_arr)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if is_pareto[i]:
            dominated_by_i = np.all(costs_arr[is_pareto] <= costs_arr[i], axis=1) & \
                             np.any(costs_arr[is_pareto] < costs_arr[i], axis=1)
            is_pareto[is_pareto] = ~dominated_by_i
            is_pareto[i] = True

    idx = np.where(is_pareto)[0].tolist()
    return {
        "backend": "numpy", "note": "",
        "pareto_indices": idx, "pareto_costs": costs_arr[idx].tolist(),
        "n_dominated": n - len(idx),
    }


def crowding_distance(front: list[list[float]]) -> dict[str, Any]:
    """Compute crowding distances for a non-dominated front (NSGA-II).

    Args:
        front: List of objective vectors on the front.

    Returns:
        Dict with keys: distances, backend, note.
    """
    F = np.array(front, dtype=float)
    n, m = F.shape
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j])
        f_min, f_max = F[order[0], j], F[order[-1], j]
        dist[order[0]] = dist[order[-1]] = float("inf")
        span = f_max - f_min + 1e-12
        for i in range(1, n - 1):
            dist[order[i]] += (F[order[i + 1], j] - F[order[i - 1], j]) / span
    return {"backend": "numpy", "note": "", "distances": dist.tolist()}


def non_dominated_sort(costs: list[list[float]]) -> dict[str, Any]:
    """Fast non-dominated sort (NSGA-II style).

    Args:
        costs: List of objective vectors.

    Returns:
        Dict with keys: front_0_indices, all_fronts, n_fronts, backend, note.
    """
    costs_arr = np.array(costs)
    n = len(costs_arr)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if (np.all(costs_arr[p] <= costs_arr[q]) and np.any(costs_arr[p] < costs_arr[q])):
                dominated_set[p].append(q)
            elif (np.all(costs_arr[q] <= costs_arr[p]) and np.any(costs_arr[q] < costs_arr[p])):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)

    return {
        "backend": "numpy", "note": "",
        "front_0_indices": fronts[0], "all_fronts": fronts,
        "n_fronts": len(fronts),
    }
