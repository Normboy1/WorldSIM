"""Surrogate-Assisted Illumination (SAIL) for parametric geometry and alloy search.

Implements SAIL (Gaier, Asteroth & Mouret 2017) which wraps MAP-Elites with a
cheap surrogate model (Gaussian Process or polynomial regression) to reduce the
number of expensive ``quality_fn`` evaluations while still filling the archive.

SAIL loop
---------
1. Seed archive with a random batch (expensive ``quality_fn``).
2. Fit surrogate on all evaluated (feature_vector, quality) pairs.
3. Run *acquisition MAP-Elites*: MAP-Elites using surrogate predictions as
   the quality function — cheap, many iterations.
4. Collect top-k candidates from the acquisition archive (UCB score).
5. Evaluate those candidates with the expensive ``quality_fn``.
6. Update the real archive; repeat from step 2.

Reference
---------
Gaier, A., Asteroth, A., & Mouret, J.-B. (2017). "Data-efficient exploration,
optimization, and modeling of diverse designs through surrogate-assisted
illumination." GECCO 2017.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from simlab.engines.qdgeometry.mapelites import MAPElitesArchive

# ---------------------------------------------------------------------------
# Optional surrogate backends
# ---------------------------------------------------------------------------

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    from scipy.interpolate import RBFInterpolator as _RBFInterpolator
    _SCIPY_RBF = True
except ImportError:
    _SCIPY_RBF = False


# ---------------------------------------------------------------------------
# Surrogate wrapper
# ---------------------------------------------------------------------------

class _PolynomialSurrogate:
    """Degree-2 polynomial surrogate via least-squares — zero external deps."""

    def __init__(self):
        self._W = None
        self._mu = 0.0
        self._sigma = 1.0

    def _phi(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        rows = [np.ones((n, 1)), X]
        for i in range(d):
            for j in range(i, d):
                rows.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(rows)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._mu = y.mean()
        self._sigma = max(y.std(), 1e-9)
        y_norm = (y - self._mu) / self._sigma
        Phi = self._phi(X)
        self._W, _, _, _ = np.linalg.lstsq(Phi, y_norm, rcond=None)

    def predict(self, X: np.ndarray):
        Phi = self._phi(X)
        mean = Phi @ self._W * self._sigma + self._mu
        std = np.full(len(X), self._sigma * 0.1)
        return mean, std


class SurrogateSurface:
    """Unified surrogate: GP (sklearn) → RBF (scipy) → polynomial (numpy)."""

    def __init__(self, backend: str = "auto"):
        self._backend = backend
        self._model = None
        self._chosen: str = "none"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X) < 3:
            self._chosen = "poly"
            self._model = _PolynomialSurrogate()
            self._model.fit(X, y)
            return

        if self._backend in ("auto", "gp") and _SKLEARN and len(X) >= 5:
            try:
                kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
                gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                               normalize_y=True)
                gpr.fit(X, y)
                self._model = gpr
                self._chosen = "gp"
                return
            except Exception:
                pass

        if self._backend in ("auto", "rbf") and _SCIPY_RBF and len(X) >= 4:
            try:
                self._model = _RBFInterpolator(X, y, kernel="thin_plate_spline",
                                               degree=1, smoothing=1e-4)
                self._chosen = "rbf"
                return
            except Exception:
                pass

        self._chosen = "poly"
        self._model = _PolynomialSurrogate()
        self._model.fit(X, y)

    def predict_ucb(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound: mean + beta * std."""
        if self._chosen == "gp":
            mean, std = self._model.predict(X, return_std=True)
        elif self._chosen == "rbf":
            mean = self._model(X)
            std = np.full(len(X), 0.05 * max(abs(mean).max(), 1e-9))
        else:
            mean, std = self._model.predict(X)
        return mean + beta * std

    @property
    def backend_name(self) -> str:
        return self._chosen


# ---------------------------------------------------------------------------
# SAIL for geometry search
# ---------------------------------------------------------------------------

def sail_geometry_search(
    primitive: str = "linear_extrusion",
    param_bounds: dict[str, tuple[float, float]] | None = None,
    constraint_targets: dict[str, float] | None = None,
    # SAIL hyper-parameters
    n_sail_iterations: int = 5,
    n_initial: int = 20,
    n_acquisition_steps: int = 200,
    n_candidates_per_iter: int = 10,
    # Archive dimensions
    dim1_bins: int = 8,
    dim2_bins: int = 8,
    # Surrogate
    surrogate_backend: str = "auto",
    ucb_beta: float = 2.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Surrogate-Assisted Illumination over parametric geometry programs.

    Each SAIL iteration:
    1. Fits a surrogate on the current real-evaluation history.
    2. Runs acquisition MAP-Elites (cheap — surrogate quality).
    3. Picks ``n_candidates_per_iter`` cells with highest UCB.
    4. Evaluates those with the true constraint-loss function.
    5. Updates the real archive.

    Parameters
    ----------
    primitive : str
        Geometry primitive to illuminate.
    n_sail_iterations : int
        Number of surrogate-fit → acquisition → evaluate cycles.
    n_initial : int
        Number of random evaluations before the first surrogate fit.
    n_acquisition_steps : int
        MAP-Elites steps per acquisition phase (cheap).
    n_candidates_per_iter : int
        Candidates to evaluate with the true function each SAIL iteration.
    surrogate_backend : str
        ``"auto"``, ``"gp"``, ``"rbf"``, or ``"poly"``.
    ucb_beta : float
        Exploration coefficient for Upper Confidence Bound.

    Returns
    -------
    dict with archive, surrogate backend used, total evaluations, timing.
    """
    from simlab.engines.qdgeometry.primitives import (
        linear_extrusion, revolution, sweep, fillet_box, total_constraint_loss,
    )

    t0 = time.time()
    rng = np.random.default_rng(seed)

    _defaults: dict[str, dict[str, tuple[float, float]]] = {
        "linear_extrusion": {"width": (0.01, 0.5), "depth": (0.01, 0.5),
                             "height": (0.01, 0.8), "taper_angle_deg": (0.0, 10.0)},
        "revolution":       {"outer_radius": (0.01, 0.3), "inner_radius": (0.0, 0.25),
                             "height": (0.01, 0.6)},
        "sweep":            {"radius": (0.005, 0.1), "spine_length": (0.05, 1.0),
                             "spine_curve": (-2.0, 2.0)},
        "fillet_box":       {"width": (0.01, 0.5), "depth": (0.01, 0.5),
                             "height": (0.01, 0.8), "fillet_radius": (0.001, 0.05)},
    }
    if primitive not in _defaults:
        raise ValueError(f"Unknown primitive {primitive!r}")

    pb = dict(_defaults[primitive])
    if param_bounds:
        pb.update(param_bounds)
    params_order = list(pb.keys())
    l_bounds = np.array([pb[p][0] for p in params_order])
    u_bounds = np.array([pb[p][1] for p in params_order])
    ct = constraint_targets or {}

    _prim_fn = {"linear_extrusion": linear_extrusion, "revolution": revolution,
                "sweep": sweep, "fillet_box": fillet_box}[primitive]

    def _make_geom(vec):
        kwargs = {p: float(v) for p, v in zip(params_order, vec)}
        kwargs["n_points"] = 0
        try:
            return _prim_fn(**kwargs)
        except Exception:
            return None

    def _true_quality(vec) -> float:
        geom = _make_geom(vec)
        if geom is None:
            return -1e9
        loss = total_constraint_loss(
            geom,
            target_volume_m3=ct.get("target_volume_m3"),
            target_aspect_ratio=ct.get("target_aspect_ratio"),
            min_wall_m=ct.get("min_wall_m"),
            taper_angle_deg=vec[params_order.index("taper_angle_deg")]
                            if "taper_angle_deg" in params_order else None,
            min_draft_deg=ct.get("min_draft_deg", 1.0),
        )
        return -loss["total"]

    def _features(vec):
        geom = _make_geom(vec)
        if geom is None:
            return 1.0, 1e-6
        return geom.aspect_ratio, geom.volume_m3

    ar_max = 20.0
    vol_max = max(float(u_bounds[0]) * float(u_bounds[1]) * float(u_bounds[2])
                  if len(u_bounds) >= 3 else 0.01, 0.01)

    # Real archive
    real_archive = MAPElitesArchive(dim1_bins, dim2_bins, (1.0, ar_max), (1e-6, vol_max))

    # Surrogate
    surrogate = SurrogateSurface(backend=surrogate_backend)

    # History for surrogate fitting: list of (feature_vec, quality)
    X_hist: list[np.ndarray] = []
    y_hist: list[float] = []
    n_evaluations = 0
    n_sail_evals = 0

    def _evaluate_and_record(vec):
        nonlocal n_evaluations
        q = _true_quality(vec)
        f1, f2 = _features(vec)
        real_archive.try_insert({"params": dict(zip(params_order, vec.tolist()))}, q, f1, f2)
        feat = np.append(vec, [f1, f2])
        X_hist.append(feat)
        y_hist.append(q)
        n_evaluations += 1
        return q, f1, f2

    # ── Phase 0: random seed evaluations ──────────────────────────────────
    for _ in range(n_initial):
        vec = rng.uniform(l_bounds, u_bounds)
        _evaluate_and_record(vec)

    sigma_mut = (u_bounds - l_bounds) * 0.05

    # ── SAIL iterations ───────────────────────────────────────────────────
    sail_log: list[dict] = []
    for sail_iter in range(n_sail_iterations):
        # Fit surrogate
        X_arr = np.array(X_hist)
        y_arr = np.array(y_hist)
        surrogate.fit(X_arr, y_arr)

        # Acquisition MAP-Elites (cheap surrogate quality)
        acq_archive = MAPElitesArchive(dim1_bins, dim2_bins, (1.0, ar_max), (1e-6, vol_max))

        def _surrogate_quality(vec):
            f1, f2 = _features(vec)
            feat = np.append(vec, [f1, f2]).reshape(1, -1)
            try:
                return float(surrogate.predict_ucb(feat, beta=ucb_beta)[0])
            except Exception:
                return 0.0

        # Seed acquisition archive from real archive
        for cell_data in real_archive.grid.values():
            pv = cell_data["solution"]["params"]
            v = np.array([pv[p] for p in params_order])
            f1, f2 = _features(v)
            q_surr = _surrogate_quality(v)
            acq_archive.try_insert({"params": pv}, q_surr, f1, f2)

        # Run acquisition MAP-Elites
        for _ in range(n_acquisition_steps):
            elite = acq_archive.sample_elite(rng)
            if elite is None:
                vec = rng.uniform(l_bounds, u_bounds)
            else:
                pv = elite["solution"]["params"]
                vec = np.array([pv[p] for p in params_order])
                vec = np.clip(vec + rng.normal(0, sigma_mut), l_bounds, u_bounds)
            f1, f2 = _features(vec)
            q_surr = _surrogate_quality(vec)
            acq_archive.try_insert({"params": dict(zip(params_order, vec.tolist()))},
                                   q_surr, f1, f2)

        # Select top-k candidates by UCB from acquisition archive
        candidates = []
        for cell_data in acq_archive.grid.values():
            pv = cell_data["solution"]["params"]
            vec = np.array([pv[p] for p in params_order])
            f1, f2 = _features(vec)
            feat = np.append(vec, [f1, f2]).reshape(1, -1)
            ucb = float(surrogate.predict_ucb(feat, beta=ucb_beta)[0])
            candidates.append((ucb, vec))

        candidates.sort(key=lambda x: -x[0])
        selected = candidates[:n_candidates_per_iter]

        iter_evals = 0
        for _, vec in selected:
            _evaluate_and_record(vec)
            iter_evals += 1
            n_sail_evals += 1

        sail_log.append({
            "sail_iter": sail_iter + 1,
            "surrogate_backend": surrogate.backend_name,
            "n_candidates_evaluated": iter_evals,
            "real_archive_coverage": round(real_archive.coverage(), 4),
            "real_archive_n_elites": len(real_archive.grid),
            "best_quality": round(real_archive.best()["quality"], 6) if real_archive.grid else None,
        })

    elapsed = time.time() - t0
    best = real_archive.best()

    # Reconstruct best geometry with full point cloud
    best_geom = None
    if best:
        pv = best["solution"]["params"]
        try:
            best_geom = _prim_fn(**{**pv, "n_points": 200})
        except Exception:
            pass

    return {
        "method": "SAIL",
        "reference": "Gaier, Asteroth & Mouret (2017) GECCO",
        "primitive": primitive,
        "constraint_targets": ct,
        "surrogate_backend": surrogate.backend_name,
        "n_sail_iterations": n_sail_iterations,
        "n_initial_evaluations": n_initial,
        "n_acquisition_steps_per_iter": n_acquisition_steps,
        "n_candidates_per_iter": n_candidates_per_iter,
        "total_evaluations": n_evaluations,
        "sail_evaluations": n_sail_evals,
        "timing_s": round(elapsed, 3),
        "sail_log": sail_log,
        "archive": real_archive.to_dict(),
        "best_params": best["solution"]["params"] if best else None,
        "best_quality": round(best["quality"], 6) if best else None,
        "best_geometry": best_geom.to_dict() if best_geom else None,
        "note": (
            "SAIL uses a surrogate model (UCB acquisition) to guide which "
            "candidates are worth evaluating with the expensive true objective, "
            "reducing evaluations by ~10× vs blind MAP-Elites while achieving "
            "comparable or better archive coverage."
        ),
    }


# ---------------------------------------------------------------------------
# SAIL for alloy composition
# ---------------------------------------------------------------------------

def sail_alloy_search(
    elements: list[str],
    bounds: dict[str, tuple[float, float]],
    quality_fn: Callable[[dict[str, float]], float],
    feature_fn: Callable[[dict[str, float]], tuple[float, float]],
    n_sail_iterations: int = 5,
    n_initial: int = 30,
    n_acquisition_steps: int = 300,
    n_candidates_per_iter: int = 10,
    dim1_bins: int = 10,
    dim2_bins: int = 10,
    dim1_range: tuple[float, float] = (6.0, 10.0),
    dim2_range: tuple[float, float] = (7.0, 12.0),
    surrogate_backend: str = "auto",
    ucb_beta: float = 2.0,
    mutation_sigma: float = 0.03,
    seed: int = 42,
) -> dict[str, Any]:
    """SAIL over alloy composition space.

    Same SAIL loop as ``sail_geometry_search`` but over continuous
    mole-fraction vectors with normalisation constraints.
    """
    from scipy.stats import qmc

    t0 = time.time()
    rng = np.random.default_rng(seed)
    l_bounds = np.array([bounds[e][0] for e in elements])
    u_bounds = np.array([bounds[e][1] for e in elements])

    real_archive = MAPElitesArchive(dim1_bins, dim2_bins, dim1_range, dim2_range)
    surrogate = SurrogateSurface(backend=surrogate_backend)

    X_hist: list[np.ndarray] = []
    y_hist: list[float] = []
    n_evaluations = 0

    def _normalise(vec):
        vec = np.clip(vec, l_bounds, u_bounds)
        total = vec.sum()
        if total <= 0:
            vec = l_bounds.copy(); total = vec.sum()
        return {el: float(v / total) for el, v in zip(elements, vec)}

    def _evaluate_and_record(comp):
        nonlocal n_evaluations
        q = quality_fn(comp)
        f1, f2 = feature_fn(comp)
        real_archive.try_insert({"composition": comp}, q, f1, f2)
        feat = np.array([comp[e] for e in elements] + [f1, f2])
        X_hist.append(feat)
        y_hist.append(q)
        n_evaluations += 1

    # Phase 0: LHS seed
    sampler = qmc.LatinHypercube(d=len(elements), seed=int(seed))
    raw = sampler.random(n=n_initial)
    scaled = qmc.scale(raw, l_bounds, u_bounds)
    for row in scaled:
        comp = _normalise(row)
        _evaluate_and_record(comp)

    sail_log: list[dict] = []
    for sail_iter in range(n_sail_iterations):
        X_arr = np.array(X_hist)
        y_arr = np.array(y_hist)
        surrogate.fit(X_arr, y_arr)

        acq_archive = MAPElitesArchive(dim1_bins, dim2_bins, dim1_range, dim2_range)

        def _surrogate_quality(comp):
            f1, f2 = feature_fn(comp)
            feat = np.array([comp[e] for e in elements] + [f1, f2]).reshape(1, -1)
            try:
                return float(surrogate.predict_ucb(feat, beta=ucb_beta)[0])
            except Exception:
                return 0.0

        for cell_data in real_archive.grid.values():
            comp = cell_data["solution"]["composition"]
            f1, f2 = feature_fn(comp)
            q_surr = _surrogate_quality(comp)
            acq_archive.try_insert({"composition": comp}, q_surr, f1, f2)

        for _ in range(n_acquisition_steps):
            elite = acq_archive.sample_elite(rng)
            if elite is None:
                vec = rng.uniform(l_bounds, u_bounds)
                comp = _normalise(vec)
            else:
                parent = elite["solution"]["composition"]
                vec = np.array([parent.get(el, bounds[el][0]) for el in elements])
                vec = vec + rng.normal(0, mutation_sigma, size=len(elements))
                comp = _normalise(vec)
            f1, f2 = feature_fn(comp)
            q_surr = _surrogate_quality(comp)
            acq_archive.try_insert({"composition": comp}, q_surr, f1, f2)

        candidates = []
        for cell_data in acq_archive.grid.values():
            comp = cell_data["solution"]["composition"]
            f1, f2 = feature_fn(comp)
            feat = np.array([comp[e] for e in elements] + [f1, f2]).reshape(1, -1)
            ucb = float(surrogate.predict_ucb(feat, beta=ucb_beta)[0])
            candidates.append((ucb, comp))
        candidates.sort(key=lambda x: -x[0])

        iter_evals = 0
        for _, comp in candidates[:n_candidates_per_iter]:
            _evaluate_and_record(comp)
            iter_evals += 1

        sail_log.append({
            "sail_iter": sail_iter + 1,
            "surrogate_backend": surrogate.backend_name,
            "n_evaluated": iter_evals,
            "real_archive_coverage": round(real_archive.coverage(), 4),
            "best_quality": round(real_archive.best()["quality"], 6) if real_archive.grid else None,
        })

    elapsed = time.time() - t0
    best = real_archive.best()

    return {
        "method": "SAIL",
        "reference": "Gaier, Asteroth & Mouret (2017) GECCO",
        "elements": elements,
        "surrogate_backend": surrogate.backend_name,
        "n_sail_iterations": n_sail_iterations,
        "total_evaluations": n_evaluations,
        "timing_s": round(elapsed, 3),
        "sail_log": sail_log,
        "archive": real_archive.to_dict(),
        "best_composition": best["solution"]["composition"] if best else None,
        "best_quality": best["quality"] if best else None,
        "best_features": list(best["features"]) if best else None,
    }
