"""JAX-differentiable parametric geometry primitives.

Expresses all scalar geometric quantities (volume, aspect ratio, wall thickness,
constraint losses) as pure JAX functions so that ``jax.grad`` can propagate
gradients back to the continuous CAD parameters.

Gradient-based optimiser entry-point
-------------------------------------
``gradient_descent_geometry(primitive, init_params, constraint_targets, ...)``

Falls back gracefully to ``scipy.optimize.minimize`` when JAX is unavailable.

Reference
---------
Gaier, Asteroth & Mouret (2017) Surrogate-Assisted Illumination (SAIL).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional JAX import
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    _JAX = True
except ImportError:  # pragma: no cover
    _JAX = False
    jnp = np  # type: ignore[assignment]

try:
    import optax as _optax  # type: ignore
    _OPTAX = True
except ImportError:
    _OPTAX = False


# ---------------------------------------------------------------------------
# Pure scalar geometry functions (JAX-traceable)
# ---------------------------------------------------------------------------

def _jax_linear_extrusion_volume(w, d, h, taper_deg):
    """Volume of a tapered rectangular extrusion (frustum)."""
    t = taper_deg * (math.pi / 180.0)
    shrink = jnp.tan(t) * h
    w_top = jnp.maximum(w - 2 * shrink, 1e-6)
    d_top = jnp.maximum(d - 2 * shrink, 1e-6)
    return h * (w * d + w_top * d_top + jnp.sqrt(w * d * w_top * d_top)) / 3.0


def _jax_linear_extrusion_aspect(w, d, h, taper_deg):
    return h / jnp.sqrt(jnp.maximum(w * d, 1e-12))


def _jax_linear_extrusion_wall(w, d, h, taper_deg):
    return jnp.minimum(w, d) / 2.0


def _jax_revolution_volume(r_out, r_in, h):
    return math.pi * h * jnp.maximum(r_out**2 - r_in**2, 0.0)


def _jax_revolution_aspect(r_out, r_in, h):
    return h / jnp.maximum(2.0 * r_out, 1e-6)


def _jax_revolution_wall(r_out, r_in, h):
    return jnp.maximum(r_out - r_in, 1e-6)


def _jax_sweep_volume(radius, spine_length, spine_curve):
    return math.pi * radius**2 * spine_length


def _jax_sweep_aspect(radius, spine_length, spine_curve):
    return spine_length / jnp.maximum(2.0 * radius, 1e-6)


def _jax_sweep_wall(radius, spine_length, spine_curve):
    return radius


def _jax_fillet_box_volume(w, d, h, fillet_r):
    fr = jnp.minimum(fillet_r, jnp.minimum(w, d) / 2.0)
    return w * d * h - (4.0 - math.pi) * fr**2 * h


def _jax_fillet_box_aspect(w, d, h, fillet_r):
    return h / jnp.sqrt(jnp.maximum(w * d, 1e-12))


def _jax_fillet_box_wall(w, d, h, fillet_r):
    return jnp.minimum(w, d) / 2.0


# Per-primitive dispatch tables
_VOLUME_FNS = {
    "linear_extrusion": _jax_linear_extrusion_volume,
    "revolution":       _jax_revolution_volume,
    "sweep":            _jax_sweep_volume,
    "fillet_box":       _jax_fillet_box_volume,
}
_ASPECT_FNS = {
    "linear_extrusion": _jax_linear_extrusion_aspect,
    "revolution":       _jax_revolution_aspect,
    "sweep":            _jax_sweep_aspect,
    "fillet_box":       _jax_fillet_box_aspect,
}
_WALL_FNS = {
    "linear_extrusion": _jax_linear_extrusion_wall,
    "revolution":       _jax_revolution_wall,
    "sweep":            _jax_sweep_wall,
    "fillet_box":       _jax_fillet_box_wall,
}

_PARAM_ORDERS = {
    "linear_extrusion": ["width", "depth", "height", "taper_angle_deg"],
    "revolution":       ["outer_radius", "inner_radius", "height"],
    "sweep":            ["radius", "spine_length", "spine_curve"],
    "fillet_box":       ["width", "depth", "height", "fillet_radius"],
}


# ---------------------------------------------------------------------------
# Composable constraint loss (JAX-traceable)
# ---------------------------------------------------------------------------

def _jax_total_loss(params_vec, primitive: str, ct: dict):
    """Scalar constraint loss as a JAX-traceable function of params_vec."""
    vol_fn   = _VOLUME_FNS[primitive]
    ar_fn    = _ASPECT_FNS[primitive]
    wall_fn  = _WALL_FNS[primitive]

    volume      = vol_fn(*params_vec)
    aspect      = ar_fn(*params_vec)
    wall        = wall_fn(*params_vec)

    loss = 0.0

    if "target_volume_m3" in ct and ct["target_volume_m3"] is not None:
        tv = float(ct["target_volume_m3"])
        loss = loss + ((volume - tv) / max(tv, 1e-9)) ** 2

    if "target_aspect_ratio" in ct and ct["target_aspect_ratio"] is not None:
        ta = float(ct["target_aspect_ratio"])
        loss = loss + ((aspect - ta) / max(ta, 1e-9)) ** 2

    if "min_wall_m" in ct and ct["min_wall_m"] is not None:
        mw = float(ct["min_wall_m"])
        viol = jnp.maximum(mw - wall, 0.0)
        loss = loss + (viol / max(mw, 1e-9)) ** 2

    if primitive == "linear_extrusion" and "min_draft_deg" in ct:
        min_d = float(ct.get("min_draft_deg", 1.0))
        taper = params_vec[3]
        viol = jnp.maximum(min_d - taper, 0.0)
        loss = loss + (viol / max(min_d, 1e-9)) ** 2

    return loss


# ---------------------------------------------------------------------------
# Gradient-descent optimiser
# ---------------------------------------------------------------------------

def gradient_descent_geometry(
    primitive: str,
    init_params: dict[str, float],
    constraint_targets: dict[str, Any] | None = None,
    n_steps: int = 200,
    learning_rate: float = 1e-3,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Tune continuous CAD parameters via gradient descent against constraint losses.

    Uses JAX autodiff when available; falls back to ``scipy.optimize.minimize``.

    Parameters
    ----------
    primitive : str
        One of ``"linear_extrusion"``, ``"revolution"``, ``"sweep"``, ``"fillet_box"``.
    init_params : dict
        Initial parameter values, keyed by parameter name.
    constraint_targets : dict | None
        Constraint targets passed to the loss function.
    n_steps : int
        Number of gradient-descent steps (JAX path) or evaluations (scipy path).
    learning_rate : float
        Step size for Adam optimiser (JAX path).
    param_bounds : dict | None
        Box bounds per parameter.  Applied as clip after each step.

    Returns
    -------
    dict with ``optimized_params``, ``final_loss``, ``loss_history``, ``backend``.
    """
    if primitive not in _PARAM_ORDERS:
        raise ValueError(f"Unknown primitive {primitive!r}")

    ct = constraint_targets or {}
    order = _PARAM_ORDERS[primitive]

    defaults_bounds: dict[str, dict[str, tuple[float, float]]] = {
        "linear_extrusion": {"width": (0.001, 1.0), "depth": (0.001, 1.0),
                             "height": (0.001, 2.0), "taper_angle_deg": (0.0, 15.0)},
        "revolution":       {"outer_radius": (0.001, 0.5), "inner_radius": (0.0, 0.4),
                             "height": (0.001, 1.0)},
        "sweep":            {"radius": (0.001, 0.2), "spine_length": (0.01, 2.0),
                             "spine_curve": (-3.0, 3.0)},
        "fillet_box":       {"width": (0.001, 1.0), "depth": (0.001, 1.0),
                             "height": (0.001, 2.0), "fillet_radius": (0.0001, 0.1)},
    }
    pb = dict(defaults_bounds[primitive])
    if param_bounds:
        pb.update(param_bounds)

    l_bounds = np.array([pb[p][0] for p in order], dtype=np.float64)
    u_bounds = np.array([pb[p][1] for p in order], dtype=np.float64)

    init_vec = np.array(
        [float(init_params.get(p, (pb[p][0] + pb[p][1]) / 2.0)) for p in order],
        dtype=np.float64,
    )
    init_vec = np.clip(init_vec, l_bounds, u_bounds)

    loss_history: list[float] = []

    # ── JAX + optax path ──────────────────────────────────────────────────
    if _JAX and _OPTAX:
        import optax

        params = jnp.array(init_vec)
        lb = jnp.array(l_bounds)
        ub = jnp.array(u_bounds)

        def loss_fn(p):
            return _jax_total_loss(p, primitive, ct)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn))
        opt = optax.adam(learning_rate)
        opt_state = opt.init(params)

        for _ in range(n_steps):
            val, grads = grad_fn(params)
            loss_history.append(float(val))
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            params = jnp.clip(params, lb, ub)

        final_params = np.array(params)
        final_loss = float(loss_fn(params))
        backend = "jax+optax"

    # ── JAX only — manual Adam ────────────────────────────────────────────
    elif _JAX:
        params = jnp.array(init_vec)
        lb = jnp.array(l_bounds)
        ub = jnp.array(u_bounds)

        def loss_fn(p):
            return _jax_total_loss(p, primitive, ct)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        # Adam state
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = jnp.zeros_like(params)
        v = jnp.zeros_like(params)

        for step in range(1, n_steps + 1):
            val, grads = grad_fn(params)
            loss_history.append(float(val))
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads ** 2
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            params = params - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
            params = jnp.clip(params, lb, ub)

        final_params = np.array(params)
        final_loss = float(loss_fn(params))
        backend = "jax"

    # ── scipy fallback ────────────────────────────────────────────────────
    else:
        from scipy.optimize import minimize

        def loss_np(vec):
            v = [float(x) for x in vec]
            try:
                return float(_jax_total_loss(v, primitive, ct))
            except Exception:
                return 1e9

        bounds_list = [(l_bounds[i], u_bounds[i]) for i in range(len(order))]
        res = minimize(loss_np, init_vec, method="L-BFGS-B", bounds=bounds_list,
                       options={"maxiter": n_steps})
        loss_history = [float(res.fun)]
        final_params = np.clip(res.x, l_bounds, u_bounds)
        final_loss = float(res.fun)
        backend = "scipy"

    optimized_params = {p: float(final_params[i]) for i, p in enumerate(order)}
    return {
        "primitive": primitive,
        "backend": backend,
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "init_params": {p: float(init_params.get(p, (pb[p][0] + pb[p][1]) / 2.0)) for p in order},
        "optimized_params": optimized_params,
        "final_loss": round(final_loss, 8),
        "loss_history": [round(v, 8) for v in loss_history],
        "constraint_targets": ct,
        "note": (
            f"Gradient descent via {backend}. "
            "Parameters tuned to minimise the weighted constraint loss. "
            "Use mapelites_geometry_search for diverse population search."
        ),
    }
