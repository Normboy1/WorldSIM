"""Grammar-guided genetic programming over the parametric geometry DSL.

BNF grammar
-----------
  Program  ::= Primitive | BoolOp(Program, Program)
  Primitive ::= linear_extrusion(w,d,h,t)
              | revolution(r_out, r_in, h)
              | sweep(r, L, c)
              | fillet_box(w,d,h,f)
  BoolOp    ::= union | diff

Each ``GeomNode`` is either a leaf (primitive + continuous params) or an
internal node (boolean op + two child programs).  The tree structure is the
*program*, the continuous params are the *genotype*.

Search
------
``grammar_guided_search(constraint_targets, ...)`` runs a (μ+λ) evolutionary
strategy over the program trees.  At each generation:
  - Structural mutation: randomly add/remove/replace a node.
  - Parameter mutation: Gaussian perturbation of continuous params.
  - Crossover: swap sub-trees between two parents.
  - Fitness: negative total constraint loss of the composed geometry.

Reference
---------
Jones et al. (2020). "ShapeAssembly: Learning to Generate Programs for 3D
Shape Structure Synthesis." SIGGRAPH Asia 2020.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Default parameter ranges per primitive
# ---------------------------------------------------------------------------

_DEFAULT_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "linear_extrusion": {"width": (0.01, 0.5), "depth": (0.01, 0.5),
                         "height": (0.01, 0.8), "taper_angle_deg": (0.0, 10.0)},
    "revolution":       {"outer_radius": (0.01, 0.3), "inner_radius": (0.0, 0.25),
                         "height": (0.01, 0.6)},
    "sweep":            {"radius": (0.005, 0.1), "spine_length": (0.05, 1.0),
                         "spine_curve": (-2.0, 2.0)},
    "fillet_box":       {"width": (0.01, 0.5), "depth": (0.01, 0.5),
                         "height": (0.01, 0.8), "fillet_radius": (0.001, 0.05)},
}

_PRIMITIVES = list(_DEFAULT_BOUNDS.keys())
_BOOL_OPS = ["union", "diff"]


# ---------------------------------------------------------------------------
# Tree nodes
# ---------------------------------------------------------------------------

@dataclass
class GeomNode:
    """A node in a geometry program tree.

    Leaf:     ``node_type = "primitive"``, ``primitive`` set, ``params`` set.
    Internal: ``node_type in {"union","diff"}``, ``left`` and ``right`` set.
    """
    node_type: str                          # "primitive" | "union" | "diff"
    primitive: str | None = None            # only for leaf nodes
    params: dict[str, float] = field(default_factory=dict)
    left: "GeomNode | None" = None
    right: "GeomNode | None" = None

    def is_leaf(self) -> bool:
        return self.node_type == "primitive"

    def depth(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + max(
            self.left.depth() if self.left else 0,
            self.right.depth() if self.right else 0,
        )

    def n_nodes(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + (self.left.n_nodes() if self.left else 0) + \
                   (self.right.n_nodes() if self.right else 0)

    def to_dict(self) -> dict:
        d: dict = {"node_type": self.node_type}
        if self.is_leaf():
            d["primitive"] = self.primitive
            d["params"] = dict(self.params)
        else:
            d["left"] = self.left.to_dict() if self.left else None
            d["right"] = self.right.to_dict() if self.right else None
        return d

    def clone(self) -> "GeomNode":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Random program generation
# ---------------------------------------------------------------------------

def _random_leaf(rng: np.random.Generator,
                 custom_bounds: dict | None = None) -> GeomNode:
    prim = _PRIMITIVES[rng.integers(0, len(_PRIMITIVES))]
    bounds = dict(_DEFAULT_BOUNDS[prim])
    if custom_bounds and prim in custom_bounds:
        bounds.update(custom_bounds[prim])
    params = {p: float(rng.uniform(lo, hi)) for p, (lo, hi) in bounds.items()}
    return GeomNode(node_type="primitive", primitive=prim, params=params)


def _random_program(rng: np.random.Generator,
                    max_depth: int = 3,
                    current_depth: int = 1,
                    custom_bounds: dict | None = None) -> GeomNode:
    """Recursively generate a random program tree."""
    # Probability of creating an internal node decreases with depth
    p_internal = max(0.0, 0.6 - 0.2 * current_depth)
    if current_depth >= max_depth or rng.random() > p_internal:
        return _random_leaf(rng, custom_bounds)
    op = _BOOL_OPS[rng.integers(0, len(_BOOL_OPS))]
    left = _random_program(rng, max_depth, current_depth + 1, custom_bounds)
    right = _random_program(rng, max_depth, current_depth + 1, custom_bounds)
    return GeomNode(node_type=op, left=left, right=right)


# ---------------------------------------------------------------------------
# Program execution — evaluate a tree to a GeomDescriptor
# ---------------------------------------------------------------------------

def execute_program(node: GeomNode, n_points: int = 0):
    """Recursively evaluate a GeomNode tree into a GeomDescriptor."""
    from simlab.engines.qdgeometry.primitives import (
        linear_extrusion, revolution, sweep, fillet_box,
        boolean_union, boolean_diff,
    )
    _PRIM_FNS = {
        "linear_extrusion": linear_extrusion,
        "revolution": revolution,
        "sweep": sweep,
        "fillet_box": fillet_box,
    }

    if node.is_leaf():
        fn = _PRIM_FNS.get(node.primitive)
        if fn is None:
            raise ValueError(f"Unknown primitive {node.primitive!r}")
        return fn(**node.params, n_points=n_points)

    left_geom = execute_program(node.left, n_points=0)
    right_geom = execute_program(node.right, n_points=0)

    if node.node_type == "union":
        return boolean_union(left_geom, right_geom)
    elif node.node_type == "diff":
        return boolean_diff(left_geom, right_geom)
    else:
        raise ValueError(f"Unknown node type {node.node_type!r}")


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _collect_nodes(node: GeomNode,
                   path: tuple = ()) -> list[tuple[tuple, GeomNode]]:
    """Collect all (path, node) pairs in the tree (BFS)."""
    result = [(path, node)]
    if not node.is_leaf():
        result += _collect_nodes(node.left,  path + ("left",))
        result += _collect_nodes(node.right, path + ("right",))
    return result


def _set_at_path(root: GeomNode, path: tuple, new_node: GeomNode) -> GeomNode:
    """Return a deep copy of root with the node at ``path`` replaced."""
    root = root.clone()
    if not path:
        return new_node
    cur = root
    for step in path[:-1]:
        cur = getattr(cur, step)
    setattr(cur, path[-1], new_node)
    return root


def mutate_structure(node: GeomNode,
                     rng: np.random.Generator,
                     custom_bounds: dict | None = None,
                     max_depth: int = 3) -> GeomNode:
    """Structural mutation: replace a random sub-tree with a new random program."""
    nodes = _collect_nodes(node)
    path, _ = nodes[rng.integers(0, len(nodes))]
    new_subtree = _random_program(rng, max_depth=max(1, max_depth - len(path)),
                                  custom_bounds=custom_bounds)
    return _set_at_path(node, path, new_subtree)


def mutate_params(node: GeomNode,
                  rng: np.random.Generator,
                  sigma: float = 0.05,
                  custom_bounds: dict | None = None) -> GeomNode:
    """Gaussian perturbation of continuous parameters across all leaf nodes."""
    node = node.clone()
    leaves = [n for _, n in _collect_nodes(node) if n.is_leaf()]
    for leaf in leaves:
        bounds = dict(_DEFAULT_BOUNDS[leaf.primitive])
        if custom_bounds and leaf.primitive in custom_bounds:
            bounds.update(custom_bounds[leaf.primitive])
        for p in list(leaf.params.keys()):
            lo, hi = bounds[p]
            leaf.params[p] = float(np.clip(
                leaf.params[p] + rng.normal(0.0, sigma * (hi - lo)),
                lo, hi,
            ))
    return node


def crossover(a: GeomNode, b: GeomNode,
              rng: np.random.Generator) -> tuple[GeomNode, GeomNode]:
    """Sub-tree crossover: swap a random sub-tree from ``a`` into ``b`` and vice versa."""
    nodes_a = _collect_nodes(a)
    nodes_b = _collect_nodes(b)
    path_a, sub_a = nodes_a[rng.integers(0, len(nodes_a))]
    path_b, sub_b = nodes_b[rng.integers(0, len(nodes_b))]
    child_a = _set_at_path(a, path_a, sub_b.clone())
    child_b = _set_at_path(b, path_b, sub_a.clone())
    return child_a, child_b


# ---------------------------------------------------------------------------
# (μ+λ) evolutionary search
# ---------------------------------------------------------------------------

def grammar_guided_search(
    constraint_targets: dict[str, float] | None = None,
    param_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    mu: int = 20,           # population size
    lam: int = 40,          # offspring per generation
    n_generations: int = 30,
    p_structural: float = 0.3,  # prob of structural vs parameter mutation
    p_crossover: float = 0.2,
    param_sigma: float = 0.05,
    max_depth: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """(μ+λ) grammar-guided GP over parametric geometry programs.

    Evolves a population of ``GeomNode`` trees, evaluating fitness as
    negative total constraint loss of the composed geometry.

    Parameters
    ----------
    mu : int
        Survivors kept each generation.
    lam : int
        Offspring generated per generation.
    n_generations : int
        Number of generations to run.
    p_structural : float
        Probability of applying a structural mutation vs parameter mutation.
    p_crossover : float
        Probability of crossover (vs pure mutation).
    max_depth : int
        Maximum allowed program tree depth.

    Returns
    -------
    dict with best program, fitness history, and population summary.
    """
    from simlab.engines.qdgeometry.primitives import total_constraint_loss

    t0 = time.time()
    rng = np.random.default_rng(seed)
    ct = constraint_targets or {}

    def _fitness(node: GeomNode) -> float:
        try:
            geom = execute_program(node, n_points=0)
            if geom is None or geom.volume_m3 <= 0:
                return -1e9
            taper = None
            if geom.primitive == "linear_extrusion":
                taper = node.params.get("taper_angle_deg")
            loss = total_constraint_loss(
                geom,
                target_volume_m3=ct.get("target_volume_m3"),
                target_aspect_ratio=ct.get("target_aspect_ratio"),
                min_wall_m=ct.get("min_wall_m"),
                taper_angle_deg=taper,
                min_draft_deg=ct.get("min_draft_deg", 1.0),
            )
            return -loss["total"]
        except Exception:
            return -1e9

    # Initialise population
    population = [
        (_fitness(p := _random_program(rng, max_depth=max_depth,
                                       custom_bounds=param_bounds)), p)
        for _ in range(mu)
    ]
    population.sort(key=lambda x: -x[0])

    fitness_history: list[dict] = []

    for gen in range(n_generations):
        offspring: list[tuple[float, GeomNode]] = []

        for _ in range(lam):
            # Select parents
            idx_a = rng.integers(0, mu)
            parent_a = population[idx_a][1]

            if rng.random() < p_crossover and mu >= 2:
                idx_b = rng.integers(0, mu)
                while idx_b == idx_a:
                    idx_b = rng.integers(0, mu)
                parent_b = population[idx_b][1]
                child, _ = crossover(parent_a, parent_b, rng)
            else:
                child = parent_a.clone()

            if rng.random() < p_structural:
                child = mutate_structure(child, rng,
                                         custom_bounds=param_bounds,
                                         max_depth=max_depth)
            else:
                child = mutate_params(child, rng,
                                      sigma=param_sigma,
                                      custom_bounds=param_bounds)

            f = _fitness(child)
            offspring.append((f, child))

        # (μ+λ) selection
        combined = population + offspring
        combined.sort(key=lambda x: -x[0])
        population = combined[:mu]

        best_f = population[0][0]
        mean_f = float(np.mean([f for f, _ in population]))
        fitness_history.append({
            "generation": gen + 1,
            "best_fitness": round(best_f, 6),
            "mean_fitness": round(mean_f, 6),
        })

    elapsed = time.time() - t0
    best_fitness, best_node = population[0]

    # Reconstruct best geometry
    best_geom = None
    try:
        best_geom = execute_program(best_node, n_points=200)
    except Exception:
        pass

    return {
        "method": "Grammar-Guided GP (μ+λ)",
        "reference": "Jones et al. (2020) ShapeAssembly, SIGGRAPH Asia",
        "constraint_targets": ct,
        "mu": mu,
        "lam": lam,
        "n_generations": n_generations,
        "timing_s": round(elapsed, 3),
        "best_fitness": round(best_fitness, 6),
        "best_program": best_node.to_dict(),
        "best_program_depth": best_node.depth(),
        "best_program_n_nodes": best_node.n_nodes(),
        "best_geometry": best_geom.to_dict() if best_geom else None,
        "fitness_history": fitness_history,
        "population_summary": [
            {"rank": i + 1, "fitness": round(f, 6), "depth": node.depth(),
             "n_nodes": node.n_nodes()}
            for i, (f, node) in enumerate(population[:5])
        ],
        "note": (
            "Grammar-guided GP searches the space of program *structures* "
            "(which primitives, in what boolean composition), not just "
            "continuous parameters. This enables topologically diverse "
            "geometry families that parameter-only search cannot reach."
        ),
    }
