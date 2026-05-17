"""NVIDIA GPU-accelerated QD geometry pipeline using PyTorch + CUDA.

All heavy computation runs on the GPU:
  - SIREN implicit field: PyTorch nn.Module on CUDA
  - Gradient descent: torch.autograd through differentiable geometry functions
  - Batch MAP-Elites: entire population evaluated in a single CUDA kernel call
  - SAIL surrogate: GPU-batched UCB acquisition

Requires: torch with CUDA (torch>=2.0, CUDA 12.x).
Falls back to CPU if CUDA is unavailable.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Differentiable geometry — pure torch ops (GPU-traceable)
# ---------------------------------------------------------------------------

def _vol_linear_extrusion(w, d, h, taper):
    shrink = torch.tan(taper * (math.pi / 180.0)) * h
    wt = torch.clamp(w - 2 * shrink, min=1e-6)
    dt = torch.clamp(d - 2 * shrink, min=1e-6)
    return h * (w * d + wt * dt + torch.sqrt(w * d * wt * dt + 1e-12)) / 3.0

def _ar_linear_extrusion(w, d, h, taper):
    return h / torch.sqrt(torch.clamp(w * d, min=1e-12))

def _wall_linear_extrusion(w, d, h, taper):
    return torch.min(w, d) / 2.0

def _vol_revolution(r_out, r_in, h):
    return math.pi * h * torch.clamp(r_out**2 - r_in**2, min=0.0)

def _ar_revolution(r_out, r_in, h):
    return h / torch.clamp(2.0 * r_out, min=1e-6)

def _wall_revolution(r_out, r_in, h):
    return torch.clamp(r_out - r_in, min=1e-6)

def _vol_fillet_box(w, d, h, fr):
    fr_c = torch.min(fr, torch.min(w, d) / 2.0)
    return w * d * h - (4.0 - math.pi) * fr_c**2 * h

def _ar_fillet_box(w, d, h, fr):
    return h / torch.sqrt(torch.clamp(w * d, min=1e-12))

def _wall_fillet_box(w, d, h, fr):
    return torch.min(w, d) / 2.0

def _vol_sweep(r, L, c):
    return math.pi * r**2 * L

def _ar_sweep(r, L, c):
    return L / torch.clamp(2.0 * r, min=1e-6)

def _wall_sweep(r, L, c):
    return r


_GEOM = {
    "linear_extrusion": (_vol_linear_extrusion, _ar_linear_extrusion, _wall_linear_extrusion),
    "revolution":        (_vol_revolution,       _ar_revolution,       _wall_revolution),
    "fillet_box":        (_vol_fillet_box,       _ar_fillet_box,       _wall_fillet_box),
    "sweep":             (_vol_sweep,            _ar_sweep,            _wall_sweep),
}

_PARAM_ORDERS = {
    "linear_extrusion": ["width", "depth", "height", "taper_angle_deg"],
    "revolution":       ["outer_radius", "inner_radius", "height"],
    "fillet_box":       ["width", "depth", "height", "fillet_radius"],
    "sweep":            ["radius", "spine_length", "spine_curve"],
}

_DEFAULT_BOUNDS = {
    "linear_extrusion": {"width":(0.01,0.5),"depth":(0.01,0.5),
                         "height":(0.01,0.8),"taper_angle_deg":(0.0,10.0)},
    "revolution":       {"outer_radius":(0.01,0.3),"inner_radius":(0.0,0.25),"height":(0.01,0.6)},
    "fillet_box":       {"width":(0.01,0.5),"depth":(0.01,0.5),
                         "height":(0.01,0.8),"fillet_radius":(0.001,0.05)},
    "sweep":            {"radius":(0.005,0.1),"spine_length":(0.05,1.0),"spine_curve":(-2.0,2.0)},
}


def _constraint_loss_batch(params_batch: torch.Tensor, primitive: str,
                            ct: dict) -> torch.Tensor:
    """Vectorised constraint loss for a batch of parameter vectors (N, D) → (N,)."""
    vol_fn, ar_fn, wall_fn = _GEOM[primitive]
    cols = params_batch.unbind(dim=1)

    vol  = vol_fn(*cols)
    ar   = ar_fn(*cols)
    wall = wall_fn(*cols)

    loss = torch.zeros(len(params_batch), device=params_batch.device)

    if ct.get("target_volume_m3"):
        tv = float(ct["target_volume_m3"])
        loss = loss + ((vol - tv) / max(tv, 1e-9)) ** 2

    if ct.get("target_aspect_ratio"):
        ta = float(ct["target_aspect_ratio"])
        loss = loss + ((ar - ta) / max(ta, 1e-9)) ** 2

    if ct.get("min_wall_m"):
        mw = float(ct["min_wall_m"])
        loss = loss + (torch.clamp(mw - wall, min=0.0) / max(mw, 1e-9)) ** 2

    return loss   # (N,)


# ---------------------------------------------------------------------------
# 1. GPU Gradient Descent
# ---------------------------------------------------------------------------

def gpu_gradient_descent(
    primitive: str,
    init_params: dict[str, float],
    constraint_targets: dict | None = None,
    n_steps: int = 500,
    lr: float = 5e-3,
    param_bounds: dict | None = None,
) -> dict[str, Any]:
    """Adam gradient descent through differentiable geometry on GPU."""
    ct  = constraint_targets or {}
    pb  = dict(_DEFAULT_BOUNDS[primitive])
    if param_bounds: pb.update(param_bounds)
    order = _PARAM_ORDERS[primitive]

    l_bounds = torch.tensor([pb[p][0] for p in order], device=DEVICE)
    u_bounds = torch.tensor([pb[p][1] for p in order], device=DEVICE)
    init_vec = torch.tensor(
        [float(init_params.get(p, (pb[p][0]+pb[p][1])/2)) for p in order],
        device=DEVICE, dtype=torch.float32
    ).clamp(l_bounds, u_bounds)

    params = nn.Parameter(init_vec.clone())
    opt = torch.optim.Adam([params], lr=lr)

    t0 = time.time()
    loss_hist = []
    for _ in range(n_steps):
        opt.zero_grad()
        p_clamp = params.clamp(l_bounds, u_bounds)
        loss = _constraint_loss_batch(p_clamp.unsqueeze(0), primitive, ct).squeeze()
        loss.backward()
        opt.step()
        with torch.no_grad():
            params.data.clamp_(l_bounds, u_bounds)
        loss_hist.append(float(loss.detach()))

    elapsed = time.time() - t0
    final_params = {p: float(params[i].detach()) for i, p in enumerate(order)}

    return {
        "method": "GPU gradient descent (torch.autograd)",
        "device": str(DEVICE),
        "primitive": primitive,
        "init_params": init_params,
        "optimized_params": final_params,
        "final_loss": round(float(loss_hist[-1]), 8),
        "loss_history": loss_hist,
        "n_steps": n_steps,
        "timing_s": round(elapsed, 4),
    }


# ---------------------------------------------------------------------------
# 2. GPU Batch MAP-Elites
# ---------------------------------------------------------------------------

def gpu_mapelites(
    primitive: str,
    constraint_targets: dict | None = None,
    n_iterations: int = 1000,
    population_size: int = 512,
    dim1_bins: int = 10,
    dim2_bins: int = 10,
    mutation_sigma: float = 0.04,
    seed: int = 42,
) -> dict[str, Any]:
    """MAP-Elites with fully batched GPU evaluation.

    Each iteration generates ``population_size`` mutants simultaneously and
    evaluates all of them in a single GPU call — no Python loop over candidates.
    """
    ct = constraint_targets or {}
    pb = dict(_DEFAULT_BOUNDS[primitive])
    order = _PARAM_ORDERS[primitive]
    n_params = len(order)

    torch.manual_seed(seed)
    l_b = torch.tensor([pb[p][0] for p in order], device=DEVICE)
    u_b = torch.tensor([pb[p][1] for p in order], device=DEVICE)

    vol_fn, ar_fn, _ = _GEOM[primitive]

    def _features_batch(batch):
        cols = batch.unbind(dim=1)
        ar   = ar_fn(*cols)
        vol  = vol_fn(*cols)
        return ar, vol

    ar_range  = (0.5, 25.0)
    vol_max   = float((u_b[0] * u_b[1] * u_b[2]).item()) if n_params >= 3 else 0.05
    vol_range = (1e-6, vol_max)

    def _cell_idx(ar_t, vol_t):
        ar_n  = (ar_t  - ar_range[0])  / max(ar_range[1]  - ar_range[0],  1e-9)
        vol_n = (vol_t - vol_range[0]) / max(vol_range[1] - vol_range[0], 1e-9)
        ci = (ar_n  * dim1_bins).long().clamp(0, dim1_bins  - 1)
        cj = (vol_n * dim2_bins).long().clamp(0, dim2_bins  - 1)
        return ci * dim2_bins + cj   # flat cell index

    n_cells = dim1_bins * dim2_bins
    # Archive stored as tensors on GPU
    archive_params  = torch.zeros(n_cells, n_params, device=DEVICE)
    archive_quality = torch.full((n_cells,), -1e9, device=DEVICE)
    archive_filled  = torch.zeros(n_cells, dtype=torch.bool, device=DEVICE)

    t0 = time.time()
    n_evals = 0
    n_improvements = 0

    # Phase 0: random init (batched)
    init_batch = torch.rand(population_size * 4, n_params, device=DEVICE)
    init_batch = init_batch * (u_b - l_b) + l_b
    init_loss  = _constraint_loss_batch(init_batch, primitive, ct)
    init_q     = -init_loss
    ar_b, vol_b = _features_batch(init_batch)
    cells_b    = _cell_idx(ar_b, vol_b)

    for k in range(len(init_batch)):
        c = int(cells_b[k])
        if float(init_q[k]) > float(archive_quality[c]):
            archive_params[c]  = init_batch[k]
            archive_quality[c] = init_q[k]
            archive_filled[c]  = True
            n_improvements += 1
    n_evals += len(init_batch)

    # Phase 1: MAP-Elites iterations (all batched)
    sigma_t = (u_b - l_b) * mutation_sigma

    for _ in range(n_iterations):
        # Sample parents from filled cells
        filled_idx = archive_filled.nonzero(as_tuple=True)[0]
        if len(filled_idx) == 0:
            parents = torch.rand(population_size, n_params, device=DEVICE) * (u_b - l_b) + l_b
        else:
            ridx = filled_idx[torch.randint(len(filled_idx), (population_size,), device=DEVICE)]
            parents = archive_params[ridx]

        # Gaussian mutation — entire batch at once
        noise    = torch.randn_like(parents) * sigma_t
        mutants  = (parents + noise).clamp(l_b, u_b)

        # Evaluate entire batch on GPU in one call
        loss_b   = _constraint_loss_batch(mutants, primitive, ct)
        quality_b = -loss_b
        ar_b, vol_b = _features_batch(mutants)
        cells_b  = _cell_idx(ar_b, vol_b)

        # Fully batched archive update using scatter — zero Python loops
        # For each cell find the best candidate in this batch
        # scatter_reduce: for each cell index, keep the max quality
        best_q_per_cell = torch.full((n_cells,), -1e9, device=DEVICE)
        best_q_per_cell.scatter_reduce_(0, cells_b, quality_b.detach(),
                                        reduce="amax", include_self=True)
        # Which candidates are the best for their cell AND beat the archive?
        is_cell_best = quality_b.detach() >= best_q_per_cell[cells_b] - 1e-9
        beats_archive = quality_b.detach() > archive_quality[cells_b]
        update_mask = is_cell_best & beats_archive   # (pop,)

        update_cells = cells_b[update_mask]
        if len(update_cells) > 0:
            archive_quality.scatter_(0, update_cells, quality_b.detach()[update_mask])
            archive_params[update_cells] = mutants[update_mask].detach()
            archive_filled[update_cells] = True
            n_improvements += int(update_mask.sum())

        n_evals += population_size

    elapsed = time.time() - t0

    filled_mask = archive_filled.cpu().numpy()
    qualities   = archive_quality.cpu().numpy()
    params_np   = archive_params.cpu().numpy()
    n_filled    = int(filled_mask.sum())

    best_cell = int(qualities[filled_mask].argmax()) if n_filled else 0
    best_flat = np.where(filled_mask)[0][best_cell] if n_filled else 0
    best_q    = float(qualities[best_flat]) if n_filled else None
    best_p    = {p: float(params_np[best_flat, i]) for i, p in enumerate(order)} if n_filled else None

    elites = []
    for flat in np.where(filled_mask)[0]:
        ci = int(flat) // dim2_bins
        cj = int(flat) % dim2_bins
        elites.append({
            "cell": [ci, cj],
            "quality": float(qualities[flat]),
            "params": {p: float(params_np[flat, i]) for i, p in enumerate(order)},
        })

    return {
        "method": "GPU Batch MAP-Elites",
        "device": str(DEVICE),
        "primitive": primitive,
        "population_size": population_size,
        "n_iterations": n_iterations,
        "n_evaluations": n_evals,
        "n_improvements": n_improvements,
        "timing_s": round(elapsed, 4),
        "evals_per_second": round(n_evals / max(elapsed, 1e-9)),
        "archive": {
            "dim1_bins": dim1_bins, "dim2_bins": dim2_bins,
            "n_elites": n_filled,
            "coverage": round(n_filled / n_cells, 4),
            "best_quality": round(best_q, 6) if best_q else None,
            "elites": elites,
        },
        "best_params": best_p,
        "best_quality": round(best_q, 6) if best_q else None,
    }


# ---------------------------------------------------------------------------
# 3. GPU SIREN Implicit Field
# ---------------------------------------------------------------------------

class SIREN(nn.Module):
    """Sinusoidal Representation Network (Sitzmann et al. 2020) on CUDA."""

    def __init__(self, in_dim=3, hidden=64, n_layers=4, out_dim=1, omega0=30.0):
        super().__init__()
        self.omega0 = omega0
        layers = []
        dims = [in_dim] + [hidden] * n_layers + [out_dim]
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(d_in, d_out)
            if i == 0:
                nn.init.uniform_(linear.weight, -1/d_in, 1/d_in)
            else:
                nn.init.uniform_(linear.weight,
                                  -math.sqrt(6/d_in)/omega0,
                                   math.sqrt(6/d_in)/omega0)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.sin(self.omega0 * h)
        return h


def gpu_siren_fit(
    surface_points: np.ndarray,
    n_epochs: int = 500,
    lr: float = 1e-4,
    hidden: int = 128,
    n_layers: int = 4,
    batch_size: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    """Train SIREN on GPU to represent a geometry as an implicit SDF."""
    torch.manual_seed(seed)

    pts = torch.tensor(surface_points, dtype=torch.float32)
    centre = pts.mean(dim=0)
    scale  = pts.std().clamp(min=1e-6)
    pts_norm = (pts - centre) / scale
    pts_norm = pts_norm.to(DEVICE)

    model = SIREN(in_dim=3, hidden=hidden, n_layers=n_layers, omega0=30.0).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    n_params_total = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    loss_hist = []

    for epoch in range(n_epochs):
        idx = torch.randperm(len(pts_norm))[:batch_size]
        x_surf = pts_norm[idx]
        x_free = torch.rand(batch_size // 2, 3, device=DEVICE) * 2 - 1

        pred_surf = model(x_surf)
        loss_surf = (pred_surf ** 2).mean()

        pred_free = model(x_free)
        loss_eik  = ((pred_free.abs() - 0.05) ** 2).mean()

        loss = loss_surf + 0.1 * loss_eik
        opt.zero_grad(); loss.backward(); opt.step()
        loss_hist.append(float(loss.detach()))

    elapsed = time.time() - t0

    # Measure inference throughput
    test_pts = torch.rand(100_000, 3, device=DEVICE)
    t_inf = time.time()
    with torch.no_grad():
        _ = model(test_pts)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inf_elapsed = time.time() - t_inf

    # Near-surface eval
    with torch.no_grad():
        pred_check = model(pts_norm).squeeze()
    near_surface_pct = float((pred_check.abs() < 0.05).float().mean() * 100)

    gpu_mem_mb = 0
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        gpu_mem_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    return {
        "model": model,
        "centre": centre.numpy(),
        "scale": float(scale),
        "device": str(DEVICE),
        "n_params": n_params_total,
        "architecture": f"3→{hidden}×{n_layers}→1  (SIREN)",
        "n_epochs": n_epochs,
        "loss_history": loss_hist,
        "final_loss": round(loss_hist[-1], 8),
        "near_surface_pct": round(near_surface_pct, 1),
        "timing_s": round(elapsed, 3),
        "inf_100k_pts_ms": round(inf_elapsed * 1000, 2),
        "gpu_peak_mem_mb": gpu_mem_mb,
    }


def gpu_siren_slice(model_info: dict,
                    res: int = 120,
                    axis: str = "xz") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample SDF cross-section on GPU, return (XX, YY, SDF) arrays."""
    model  = model_info["model"]
    centre = model_info["centre"]
    scale  = model_info["scale"]

    coords = np.linspace(-1.2, 1.2, res)
    A, B   = np.meshgrid(coords, coords)
    zeros  = np.zeros(res * res)

    if axis == "xz":
        pts = np.stack([A.ravel(), zeros, B.ravel()], axis=1)
    else:
        pts = np.stack([A.ravel(), B.ravel(), zeros], axis=1)

    t_pts = torch.tensor(pts, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        sdf = model(t_pts).squeeze().cpu().numpy().reshape(res, res)

    # Convert back to world coords for plotting
    A_world = A * scale + (centre[0] if axis == "xz" else centre[0])
    B_world = B * scale + (centre[2] if axis == "xz" else centre[1])
    return A_world, B_world, sdf


# ---------------------------------------------------------------------------
# 4. Multi-primitive parallel GPU search
# ---------------------------------------------------------------------------

def gpu_multi_primitive_search(
    constraint_targets: dict | None = None,
    n_iterations: int = 1000,
    population_size: int = 4096,
    dim1_bins: int = 12,
    dim2_bins: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Search all 4 primitives simultaneously using separate CUDA streams.

    Each primitive gets its own archive and runs in parallel on the GPU.
    Returns the best elite across all primitive types.
    """
    primitives = ["linear_extrusion", "revolution", "fillet_box", "sweep"]
    ct = constraint_targets or {}

    results = {}
    streams = [torch.cuda.Stream() for _ in primitives] if DEVICE.type == "cuda" else [None]*4

    t0 = time.time()
    jobs = []
    for prim, stream in zip(primitives, streams):
        if stream:
            with torch.cuda.stream(stream):
                r = gpu_mapelites(prim, ct, n_iterations=n_iterations,
                                  population_size=population_size,
                                  dim1_bins=dim1_bins, dim2_bins=dim2_bins, seed=seed)
        else:
            r = gpu_mapelites(prim, ct, n_iterations=n_iterations,
                              population_size=population_size,
                              dim1_bins=dim1_bins, dim2_bins=dim2_bins, seed=seed)
        results[prim] = r

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Find global best
    best_prim = max(results, key=lambda p: results[p]["best_quality"] or -1e9)
    total_evals = sum(r["n_evaluations"] for r in results.values())

    return {
        "method": "Multi-primitive parallel GPU search",
        "device": str(DEVICE),
        "primitives_searched": primitives,
        "n_iterations_each": n_iterations,
        "population_size": population_size,
        "total_evaluations": total_evals,
        "timing_s": round(elapsed, 3),
        "total_evals_per_second": round(total_evals / max(elapsed, 1e-9)),
        "results_per_primitive": {
            p: {
                "best_quality": r["best_quality"],
                "coverage": r["archive"]["coverage"],
                "n_elites": r["archive"]["n_elites"],
                "timing_s": r["timing_s"],
            } for p, r in results.items()
        },
        "global_best_primitive": best_prim,
        "global_best_quality": results[best_prim]["best_quality"],
        "global_best_params": results[best_prim]["best_params"],
        "all_archives": {p: r["archive"] for p, r in results.items()},
    }


# ---------------------------------------------------------------------------
# 5. Neural surrogate SAIL on GPU
# ---------------------------------------------------------------------------

class _NeuralSurrogate(nn.Module):
    """Small MLP surrogate for quality prediction: feature_dim → 1."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 2),      # [mean, log_var] for uncertainty
        )
    def forward(self, x):
        out = self.net(x)
        mean    = out[:, 0]
        log_var = out[:, 1].clamp(-6, 2)
        return mean, log_var


def gpu_neural_sail(
    primitive: str = "linear_extrusion",
    constraint_targets: dict | None = None,
    n_sail_iterations: int = 6,
    n_initial: int = 512,
    population_size: int = 4096,
    n_acquisition_steps: int = 200,
    n_candidates_per_iter: int = 64,
    surrogate_epochs: int = 100,
    dim1_bins: int = 12,
    dim2_bins: int = 12,
    ucb_beta: float = 2.0,
    seed: int = 42,
) -> dict[str, Any]:
    """SAIL with a GPU neural network surrogate (mean + uncertainty head).

    The surrogate is a small MLP trained on the evaluation history.
    Acquisition uses UCB: mean + beta * std.
    All candidate generation and surrogate inference run on GPU.
    """
    ct = constraint_targets or {}
    pb = dict(_DEFAULT_BOUNDS[primitive])
    order = _PARAM_ORDERS[primitive]
    n_params = len(order)

    torch.manual_seed(seed)
    l_b = torch.tensor([pb[p][0] for p in order], device=DEVICE)
    u_b = torch.tensor([pb[p][1] for p in order], device=DEVICE)

    vol_fn, ar_fn, _ = _GEOM[primitive]
    ar_range  = (0.5, 25.0)
    vol_max   = float((u_b[0] * u_b[1] * u_b[2]).item()) if n_params >= 3 else 0.05
    vol_range = (1e-6, vol_max)
    n_cells   = dim1_bins * dim2_bins

    def _features(batch):
        cols = batch.unbind(dim=1)
        ar   = ar_fn(*cols)
        vol  = vol_fn(*cols)
        return ar, vol

    def _cell_idx(ar_t, vol_t):
        ar_n  = (ar_t  - ar_range[0])  / max(ar_range[1] - ar_range[0], 1e-9)
        vol_n = (vol_t - vol_range[0]) / max(vol_range[1] - vol_range[0], 1e-9)
        ci = (ar_n * dim1_bins).long().clamp(0, dim1_bins - 1)
        cj = (vol_n * dim2_bins).long().clamp(0, dim2_bins - 1)
        return ci * dim2_bins + cj

    def _true_quality_batch(batch):
        return -_constraint_loss_batch(batch, primitive, ct)

    # Feature vector = [params..., ar, vol_norm]
    feat_dim = n_params + 2

    def _make_features(batch):
        ar_b, vol_b = _features(batch)
        vol_n = (vol_b - vol_range[0]) / max(vol_range[1] - vol_range[0], 1e-9)
        return torch.cat([batch, ar_b.unsqueeze(1), vol_n.unsqueeze(1)], dim=1)

    # Archives
    archive_params   = torch.zeros(n_cells, n_params, device=DEVICE)
    archive_quality  = torch.full((n_cells,), -1e9, device=DEVICE)
    archive_filled   = torch.zeros(n_cells, dtype=torch.bool, device=DEVICE)

    # History buffers
    hist_feat = torch.zeros(0, feat_dim, device=DEVICE)
    hist_q    = torch.zeros(0, device=DEVICE)

    surrogate = _NeuralSurrogate(feat_dim).to(DEVICE)
    surr_opt  = torch.optim.Adam(surrogate.parameters(), lr=1e-3)

    sigma_mut = (u_b - l_b) * 0.05
    t0 = time.time()
    n_evals = 0
    sail_log = []

    def _update_archive(batch, quality):
        nonlocal n_evals
        ar_b, vol_b = _features(batch)
        cells_b = _cell_idx(ar_b, vol_b)
        best_q_cell = torch.full((n_cells,), -1e9, device=DEVICE)
        best_q_cell.scatter_reduce_(0, cells_b, quality, reduce="amax", include_self=True)
        mask = (quality >= best_q_cell[cells_b] - 1e-9) & (quality > archive_quality[cells_b])
        uc = cells_b[mask]
        if len(uc):
            archive_quality.scatter_(0, uc, quality[mask])
            archive_params[uc] = batch[mask].detach()
            archive_filled[uc] = True
        n_evals += len(batch)

    # Phase 0: random seed on GPU
    seed_batch = torch.rand(n_initial, n_params, device=DEVICE) * (u_b - l_b) + l_b
    seed_q     = _true_quality_batch(seed_batch)
    _update_archive(seed_batch, seed_q)
    hist_feat = torch.cat([hist_feat, _make_features(seed_batch)], 0)
    hist_q    = torch.cat([hist_q,    seed_q.detach()], 0)

    for sail_iter in range(n_sail_iterations):
        # Train neural surrogate
        y_mu  = hist_q.mean(); y_std = hist_q.std().clamp(min=1e-6)
        y_norm = (hist_q - y_mu) / y_std
        for _ in range(surrogate_epochs):
            idx = torch.randperm(len(hist_feat))[:min(512, len(hist_feat))]
            pred_mean, pred_lv = surrogate(hist_feat[idx])
            nll = 0.5 * (torch.exp(-pred_lv) * (pred_mean - y_norm[idx])**2 + pred_lv)
            surr_opt.zero_grad(); nll.mean().backward(); surr_opt.step()

        # Acquisition MAP-Elites with surrogate UCB
        acq_params   = archive_params.clone()
        acq_quality  = archive_quality.clone()
        acq_filled   = archive_filled.clone()

        for _ in range(n_acquisition_steps):
            filled_idx = acq_filled.nonzero(as_tuple=True)[0]
            if len(filled_idx) == 0:
                cands = torch.rand(population_size, n_params, device=DEVICE) * (u_b - l_b) + l_b
            else:
                ridx = filled_idx[torch.randint(len(filled_idx), (population_size,), device=DEVICE)]
                cands = (acq_params[ridx] + torch.randn(population_size, n_params, device=DEVICE) * sigma_mut).clamp(l_b, u_b)

            feats = _make_features(cands)
            with torch.no_grad():
                pm, plv = surrogate(feats)
                ucb = (pm * y_std + y_mu) + ucb_beta * torch.exp(0.5 * plv) * y_std
            ar_c, vol_c = _features(cands)
            cells_c = _cell_idx(ar_c, vol_c)
            best_ucb = torch.full((n_cells,), -1e9, device=DEVICE)
            best_ucb.scatter_reduce_(0, cells_c, ucb, reduce="amax", include_self=True)
            mask = (ucb >= best_ucb[cells_c] - 1e-9) & (ucb > acq_quality[cells_c])
            uc = cells_c[mask]
            if len(uc):
                acq_quality.scatter_(0, uc, ucb[mask])
                acq_params[uc] = cands[mask]; acq_filled[uc] = True

        # Select top-k by UCB from acquisition archive
        cand_cells = acq_filled.nonzero(as_tuple=True)[0]
        cand_params_t = acq_params[cand_cells]
        feats = _make_features(cand_params_t)
        with torch.no_grad():
            pm, plv = surrogate(feats)
            ucb_scores = (pm * y_std + y_mu) + ucb_beta * torch.exp(0.5 * plv) * y_std
        topk = min(n_candidates_per_iter, len(ucb_scores))
        top_idx = ucb_scores.topk(topk).indices
        selected = cand_params_t[top_idx]

        # True evaluation
        sel_q = _true_quality_batch(selected)
        _update_archive(selected, sel_q)
        hist_feat = torch.cat([hist_feat, _make_features(selected)], 0)
        hist_q    = torch.cat([hist_q,    sel_q.detach()], 0)

        best_q = float(archive_quality[archive_filled].max()) if archive_filled.any() else None
        sail_log.append({
            "sail_iter": sail_iter + 1,
            "n_evaluated": topk,
            "real_archive_n_elites": int(archive_filled.sum()),
            "real_archive_coverage": round(int(archive_filled.sum()) / n_cells, 4),
            "best_quality": round(best_q, 6) if best_q else None,
        })

    elapsed = time.time() - t0
    n_elites = int(archive_filled.sum())
    best_q   = float(archive_quality[archive_filled].max()) if archive_filled.any() else None
    best_cell_idx = int(archive_quality.argmax())
    best_p = {p: float(archive_params[best_cell_idx, i]) for i, p in enumerate(order)}

    elites = []
    for flat in archive_filled.nonzero(as_tuple=True)[0].tolist():
        ci, cj = flat // dim2_bins, flat % dim2_bins
        elites.append({"cell": [ci, cj], "quality": float(archive_quality[flat]),
                        "params": {p: float(archive_params[flat, i]) for i, p in enumerate(order)}})

    return {
        "method": "GPU Neural SAIL",
        "device": str(DEVICE),
        "surrogate": f"MLP {feat_dim}→64→64→32→2  (mean+var head)",
        "primitive": primitive,
        "n_sail_iterations": n_sail_iterations,
        "total_evaluations": n_evals,
        "timing_s": round(elapsed, 3),
        "evals_per_second": round(n_evals / max(elapsed, 1e-9)),
        "sail_log": sail_log,
        "archive": {"dim1_bins": dim1_bins, "dim2_bins": dim2_bins,
                    "n_elites": n_elites, "coverage": round(n_elites/n_cells, 4),
                    "best_quality": round(best_q, 6) if best_q else None,
                    "elites": elites},
        "best_params": best_p,
        "best_quality": round(best_q, 6) if best_q else None,
    }
