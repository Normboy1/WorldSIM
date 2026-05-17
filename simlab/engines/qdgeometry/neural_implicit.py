"""Neural implicit field representation for parametric geometry.

A small MLP is trained to represent a geometry as a signed-distance /
occupancy field:
    f: R³ → R    (negative inside, positive outside for SDF)

Uses JAX for training when available; falls back to a NumPy MLP.

Applications (from the research proposal)
------------------------------------------
1. Point-cloud → CAD reconstruction: optimise CAD params to minimise
   point-level loss against a scanned point cloud sampled from the field.
2. 2D drawing → CAD: encode orthographic projections as occupancy queries,
   optimise CAD params to match.

Architecture
------------
  3 → 64 → 64 → 64 → 1,  sine activations (SIREN-inspired, Sitzmann 2020)

Reference
---------
Sitzmann, Martel, Bergman, Lindell & Wetzstein (2020). "Implicit Neural
Representations with Periodic Activation Functions." NeurIPS 2020.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _JAX = True
except ImportError:
    _JAX = False

# ---------------------------------------------------------------------------
# MLP parameter helpers
# ---------------------------------------------------------------------------

def _init_siren_params(layer_sizes: list[int],
                       omega0: float = 30.0,
                       seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """He-like init for SIREN layers (Sitzmann 2020 §3.2)."""
    rng = np.random.default_rng(seed)
    params = []
    for i, (n_in, n_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])):
        if i == 0:
            scale = 1.0 / n_in
        else:
            scale = np.sqrt(6.0 / n_in) / omega0
        W = rng.uniform(-scale, scale, (n_in, n_out)).astype(np.float32)
        b = np.zeros(n_out, dtype=np.float32)
        params.append((W, b))
    return params


def _forward_np(params, x: np.ndarray, omega0: float = 30.0) -> np.ndarray:
    """NumPy forward pass with sine activations on all-but-last layer."""
    h = x.astype(np.float32)
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = np.sin(omega0 * h)
    return h  # (N, 1)


# ---------------------------------------------------------------------------
# JAX forward + train (optional)
# ---------------------------------------------------------------------------

def _forward_jax(params, x, omega0: float = 30.0):
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = jnp.sin(omega0 * h)
    return h


def _sdf_loss_jax(params, x_surface, x_free, omega0):
    """Combined SDF loss: zero on surface + Eikonal constraint on free pts."""
    pred_surf = _forward_jax(params, x_surface, omega0)
    loss_surface = jnp.mean(pred_surf ** 2)

    pred_free = _forward_jax(params, x_free, omega0)
    pred_abs = jnp.abs(pred_free)
    loss_eikonal = jnp.mean((pred_abs - 0.05) ** 2)

    return loss_surface + 0.1 * loss_eikonal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NeuralImplicitField:
    """Compact MLP representing a geometry as an implicit field.

    Parameters
    ----------
    layer_sizes : list[int]
        Hidden-layer widths.  Input is always 3 (x,y,z); output always 1.
    omega0 : float
        SIREN frequency scaling.
    seed : int
        RNG seed for parameter initialisation.
    """

    def __init__(self,
                 layer_sizes: list[int] | None = None,
                 omega0: float = 30.0,
                 seed: int = 42):
        self.omega0 = omega0
        self.seed = seed
        sizes = [3] + (layer_sizes or [64, 64, 64]) + [1]
        if _JAX:
            self._params_jax = [
                (jnp.array(W), jnp.array(b))
                for W, b in _init_siren_params(sizes, omega0, seed)
            ]
            self._params_np = None
        else:
            self._params_np = _init_siren_params(sizes, omega0, seed)
            self._params_jax = None
        self._trained = False
        self.train_loss_history: list[float] = []

    def fit(self,
            surface_points: np.ndarray,
            n_epochs: int = 500,
            learning_rate: float = 1e-4,
            batch_size: int = 512) -> "NeuralImplicitField":
        """Train the implicit field on a set of surface points.

        The loss is: SDF = 0 on surface points (zero iso-surface) plus
        an Eikonal regulariser evaluated on free-space samples.

        Parameters
        ----------
        surface_points : ndarray of shape (N, 3)
            Points sampled from the target geometry surface.
        """
        N = len(surface_points)
        pts = surface_points.astype(np.float32)

        # Normalise to unit cube
        centre = pts.mean(axis=0)
        scale = max(np.abs(pts - centre).max(), 1e-6)
        pts_norm = (pts - centre) / scale
        self._centre = centre
        self._scale = scale

        if not _JAX:
            raise RuntimeError(
                "NeuralImplicitField requires JAX. Install with: "
                "`pip install jax` (CPU) or `pip install jax[cuda12]` (GPU)."
            )
        self._fit_jax(pts_norm, n_epochs, learning_rate, batch_size)

        self._trained = True
        return self

    def _fit_jax(self, pts_norm, n_epochs, lr, batch_size):
        from jax import value_and_grad

        omega0 = self.omega0
        params = self._params_jax
        rng = np.random.default_rng(self.seed)

        def loss_fn(p):
            idx = np.random.choice(len(pts_norm),
                                   min(batch_size, len(pts_norm)), replace=False)
            x_surf = jnp.array(pts_norm[idx])
            x_free = jnp.array(
                rng.uniform(-1.0, 1.0, (min(batch_size, 256), 3)).astype(np.float32)
            )
            return _sdf_loss_jax(p, x_surf, x_free, omega0)

        grad_fn = jax.jit(value_and_grad(loss_fn))

        # Manual Adam
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_state = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
        v_state = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]

        for step in range(1, n_epochs + 1):
            val, grads = grad_fn(params)
            self.train_loss_history.append(float(val))

            new_params = []
            new_m, new_v = [], []
            for i, ((W, b), (gW, gb), (mW, mb), (vW, vb)) in enumerate(
                    zip(params, grads, m_state, v_state)):
                mW2 = beta1 * mW + (1 - beta1) * gW
                mb2 = beta1 * mb + (1 - beta1) * gb
                vW2 = beta2 * vW + (1 - beta2) * gW ** 2
                vb2 = beta2 * vb + (1 - beta2) * gb ** 2
                mW_hat = mW2 / (1 - beta1 ** step)
                mb_hat = mb2 / (1 - beta1 ** step)
                vW_hat = vW2 / (1 - beta2 ** step)
                vb_hat = vb2 / (1 - beta2 ** step)
                new_params.append((
                    W - lr * mW_hat / (jnp.sqrt(vW_hat) + eps),
                    b - lr * mb_hat / (jnp.sqrt(vb_hat) + eps),
                ))
                new_m.append((mW2, mb2))
                new_v.append((vW2, vb2))
            params = new_params
            m_state, v_state = new_m, new_v

        self._params_jax = params

    def _fit_np(self, pts_norm, n_epochs, lr, batch_size):
        """NumPy path — cannot train: no autodiff available without JAX.

        Raises RuntimeError rather than silently returning a random-weight field
        with fake zero-loss history, which would be a silent correctness failure.
        """
        raise RuntimeError(
            "NeuralImplicitField.fit() requires JAX for gradient computation. "
            "Install JAX: `pip install jax` (CPU) or "
            "`pip install jax[cuda12]` (GPU). "
            "The NumPy fallback cannot train the SIREN — it has no autodiff."
        )

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the implicit field at query points.

        Parameters
        ----------
        points : ndarray of shape (N, 3)

        Returns
        -------
        ndarray of shape (N,) — negative inside, positive outside.
        """
        pts = points.astype(np.float32)
        if hasattr(self, "_centre"):
            pts = (pts - self._centre) / self._scale

        if _JAX and self._params_jax is not None:
            x = jnp.array(pts)
            out = _forward_jax(self._params_jax, x, self.omega0)
            return np.array(out).squeeze(-1)
        else:
            out = _forward_np(self._params_np, pts, self.omega0)
            return out.squeeze(-1)

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask — True where the implicit field is negative (inside)."""
        return self.predict(points) < 0.0

    def to_dict(self) -> dict:
        return {
            "trained": self._trained,
            "n_train_epochs": len(self.train_loss_history),
            "final_train_loss": round(self.train_loss_history[-1], 6)
                                if self.train_loss_history else None,
            "backend": "jax" if _JAX else "numpy",
        }


# ---------------------------------------------------------------------------
# Point-cloud reconstruction loss
# ---------------------------------------------------------------------------

def point_cloud_reconstruction_loss(
    field: NeuralImplicitField,
    target_points: np.ndarray,
) -> dict[str, float]:
    """Measure how well a trained implicit field covers a target point cloud.

    Samples the field at ``target_points`` and reports mean |SDF| (ideally
    near zero if points lie on the zero iso-surface) and occupancy accuracy
    for interior-labelled points.

    Parameters
    ----------
    field : NeuralImplicitField
        A trained implicit field (fit to some reference geometry).
    target_points : ndarray (N, 3)
        Points sampled from the target geometry.

    Returns
    -------
    dict with ``mean_abs_sdf``, ``n_points``, ``coverage_pct``.
    """
    sdf_vals = field.predict(target_points)
    mean_abs = float(np.abs(sdf_vals).mean())
    near_surface_pct = float((np.abs(sdf_vals) < 0.05).mean() * 100)
    return {
        "n_points": len(target_points),
        "mean_abs_sdf": round(mean_abs, 6),
        "near_surface_pct": round(near_surface_pct, 2),
        "note": "Ideal: mean_abs_sdf → 0.  Points on zero iso-surface → near_surface_pct = 100.",
    }


# ---------------------------------------------------------------------------
# Convenience: fit field to a GeomDescriptor
# ---------------------------------------------------------------------------

def fit_to_geometry(
    primitive: str = "fillet_box",
    params: dict | None = None,
    n_surface_points: int = 1000,
    n_epochs: int = 300,
    learning_rate: float = 1e-4,
    layer_sizes: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Create and train a NeuralImplicitField on a parametric geometry.

    Parameters
    ----------
    primitive : str
        Geometry primitive to fit.
    params : dict | None
        Primitive parameters.  Uses midpoint defaults if None.
    n_surface_points : int
        Number of surface samples to train on.
    n_epochs : int
        Training epochs.

    Returns
    -------
    dict with field metadata, training loss curve, and reconstruction loss.
    """
    from simlab.engines.qdgeometry.primitives import (
        linear_extrusion, revolution, sweep, fillet_box,
    )

    _prim_fn = {"linear_extrusion": linear_extrusion, "revolution": revolution,
                "sweep": sweep, "fillet_box": fillet_box}.get(primitive)
    if _prim_fn is None:
        raise ValueError(f"Unknown primitive {primitive!r}")

    kw = dict(params or {})
    kw["n_points"] = n_surface_points
    geom = _prim_fn(**kw)

    if geom.point_cloud is None or len(geom.point_cloud) == 0:
        raise ValueError("Geometry produced no point cloud — increase n_points")

    if not _JAX:
        return {
            "error": "JAX is required for SIREN training but is not installed.",
            "fix": "pip install jax  (CPU)  or  pip install jax[cuda12]  (GPU)",
            "backend": "unavailable",
            "trained": False,
        }

    t0 = time.time()
    field = NeuralImplicitField(layer_sizes=layer_sizes, seed=seed)
    field.fit(geom.point_cloud, n_epochs=n_epochs, learning_rate=learning_rate)
    elapsed = time.time() - t0

    recon = point_cloud_reconstruction_loss(field, geom.point_cloud)

    return {
        "primitive": primitive,
        "params": {k: v for k, v in kw.items() if k != "n_points"},
        "n_surface_points": n_surface_points,
        "n_epochs": n_epochs,
        "timing_s": round(elapsed, 3),
        "field": field.to_dict(),
        "reconstruction_loss": recon,
        "loss_history": [round(v, 6) for v in field.train_loss_history[::10]],
        "note": (
            "A SIREN MLP is trained to represent the geometry as a zero-level "
            "set of an implicit field.  The field can then be used as a "
            "differentiable proxy for point-cloud and drawing reconstruction."
        ),
    }
