"""FNO2d surrogate for blade thermal field prediction.

Implements a Fourier Neural Operator (Li et al., 2021, arXiv:2010.08895)
trained on (geometry encoding → temperature field) pairs from the Warp FD
solver. Uses a physics-informed PDE residual loss term (∇²T = 0 at interior
nodes) and MC-dropout for epistemic uncertainty estimation.

Architecture matches the PhysicsNeMo FNO2d design.  PhysicsNeMo 2.0.0 has
a Warp 1.13 / PyTorch 2.2 compatibility issue (wp.context.Device removed;
torch.distributed.tensor._ops.registration not in PyTorch <2.4), so this
module implements the same architecture directly in PyTorch.

Usage
-----
    from simlab.engines.qdgeometry.fno_surrogate import (
        BladeFNO, generate_training_data, train_fno, predict_with_uncertainty
    )

    dataset = generate_training_data(n_samples=200, solver="warp")
    model, history = train_fno(dataset, epochs=100)
    T_pred, uncertainty = predict_with_uncertainty(model, geom_encoding, n_mc=5)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH = False
    DEVICE = None


# ---------------------------------------------------------------------------
# Spectral convolution layer  (core FNO building block)
# ---------------------------------------------------------------------------

class _SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → complex weight multiply → IFFT.

    Li et al. (2021) eq. 4: (R · (Fφ))[x] = F⁻¹(R · Fφ)[x]
    where R is a learnable complex weight tensor.
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.modes1  = modes1
        self.modes2  = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def _complex_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2]  = self._complex_mul2d(x_ft[:, :, :m1, :m2],  self.weights1)
        out_ft[:, :, -m1:, :m2] = self._complex_mul2d(x_ft[:, :, -m1:, :m2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(H, W))


class _FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, dropout: float = 0.1):
        super().__init__()
        self.spectral = _SpectralConv2d(width, width, modes1, modes2)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(width)
        self.dropout  = nn.Dropout2d(p=dropout)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class BladeFNO(nn.Module):
    """FNO2d surrogate: geometry encoding (4-ch) → temperature field (1-ch).

    Parameters
    ----------
    in_channels  : geometry encoding channels (default 4)
    out_channels : output field channels (default 1 = T-field)
    modes        : Fourier modes in each spatial dimension
    width        : hidden channel width
    n_layers     : number of FNO blocks
    dropout      : MC-dropout rate (active at inference for uncertainty)
    """

    def __init__(
        self,
        in_channels:  int = 4,
        out_channels: int = 1,
        modes:        int = 12,
        width:        int = 32,
        n_layers:     int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes
        self.width        = width
        self.n_layers     = n_layers
        self.dropout      = dropout
        self.lift   = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_FNOBlock(width, modes, modes, dropout) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()


# ---------------------------------------------------------------------------
# Geometry encoding
# ---------------------------------------------------------------------------

def encode_geometry(
    hole_cx: list[float],
    hole_cy: list[float],
    hole_r:  list[float],
    ny: int = 128,
    nx: int = 256,
) -> np.ndarray:
    """Build 4-channel geometry encoding for FNO input.

    Uses the same mask logic as warp_blade._build_masks_warp so the FNO
    input channels are exactly consistent with the training T_fields.

    Channels:
      0  solid mask        (1 = blade solid including outer wall, 0 = exterior/hole)
      1  hole mask         (1 = cooling hole interior/wall)
      2  signed distance   to nearest hole centre (normalised)
      3  outer-wall mask   (1 = Dirichlet hot-gas boundary, top node per column)
    """
    from simlab.engines.qdgeometry.warp_blade import _build_masks_warp
    blade_np, outer_np, cool_np, _ = _build_masks_warp(ny, nx, hole_cx, hole_cy, hole_r)

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    # Signed distance to nearest hole centre (normalised by first hole radius or 1)
    dist_field = np.full((ny, nx), 1.0, dtype=np.float32)
    for cx, cy, r in zip(hole_cx, hole_cy, hole_r):
        d = np.sqrt((X - cx)**2 + (Y - cy)**2)
        dist_field = np.minimum(dist_field, (d - r) / max(r, 1e-6))
    dist_field = np.clip(dist_field, -2.0, 2.0) / 2.0  # normalise to [-1, 1]

    return np.stack([
        blade_np.astype(np.float32),
        cool_np.astype(np.float32),
        dist_field,
        outer_np.astype(np.float32),
    ], axis=0)  # (4, ny, nx)


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def _random_hole_params(n_holes: int, rng: np.random.Generator) -> tuple:
    """Generate valid random hole parameters for n_holes cooling channels.

    Positions are in the blade wall cross-section domain [0,1]×[0,1]:
      cx ∈ [0.08, 0.92]  chord-wise
      cy ∈ [0.12, 0.55]  wall-normal (below the hot-gas surface, inside blade)
      r  ∈ [0.03, 0.10]  hole radius
    """
    cx, cy, cr = [], [], []
    for _ in range(n_holes):
        x = rng.uniform(0.08, 0.92)
        y = rng.uniform(0.12, 0.55)
        r = rng.uniform(0.03, 0.10)
        cx.append(x); cy.append(y); cr.append(r)
    return cx, cy, cr


def generate_training_data(
    n_samples: int = 200,
    n_holes_range: tuple = (4, 10),
    solver: str = "warp",
    nx: int = 256,
    ny: int = 128,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate (geometry_encoding, T_field) training pairs.

    Parameters
    ----------
    n_samples      : number of geometry configurations to evaluate
    n_holes_range  : (min, max) cooling holes per configuration
    solver         : "warp" uses warp_blade.warp_blade_thermal;
                     "torch" uses torch_gpu_physics.gpu_blade_simulation
    """
    if not _TORCH:
        return {"error": "PyTorch required for data generation"}

    rng = np.random.default_rng(seed)
    X_geom, Y_tfield, T_means, meta = [], [], [], []

    if solver == "warp":
        try:
            from simlab.engines.qdgeometry.warp_blade import warp_blade_thermal
            _solver_fn = warp_blade_thermal
        except ImportError:
            solver = "torch"

    if solver == "torch":
        # torch_gpu_physics uses a fixed hole layout; use warp_blade as torch fallback
        # since it accepts parametric holes. If warp unavailable, fall through to error.
        try:
            from simlab.engines.qdgeometry.warp_blade import warp_blade_thermal
            _solver_fn = warp_blade_thermal
        except ImportError:
            return {"error": "Neither warp nor parametric torch solver available."}

    t0 = time.perf_counter()
    for i in range(n_samples):
        n = int(rng.integers(*n_holes_range))
        cx, cy, cr = _random_hole_params(n, rng)

        try:
            res = _solver_fn(cx, cy, cr, nx=nx, ny=ny, max_iters=1500)
            def _get_field(r):
                for k in ("T_field", "field"):
                    v = r.get(k)
                    if v is not None:
                        return v
                fields = r.get("_fields")
                if fields:
                    return fields.get("T_field")
                return None

            def _get_tmean(r):
                for k in ("T_mean", "T_mean_blade", "T_mean_blade_C"):
                    v = r.get(k)
                    if v is not None:
                        return v
                return None

            T_field = _get_field(res)
            T_mean  = _get_tmean(res)
            if T_field is None:
                continue
            T_field_np = np.array(T_field, dtype=np.float32)
            if T_field_np.shape != (ny, nx):
                T_field_np = T_field_np[:ny, :nx]
            geom = encode_geometry(cx, cy, cr, ny, nx)
            X_geom.append(geom)
            Y_tfield.append(T_field_np[np.newaxis])  # (1, ny, nx)
            T_means.append(float(T_mean) if T_mean is not None else 0.0)
            meta.append({"n_holes": n, "cx": cx, "cy": cy, "cr": cr})
        except Exception as exc:
            if verbose:
                print(f"  sample {i} failed: {exc}")
            continue

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  generated {i+1}/{n_samples} samples  ({elapsed:.1f}s)")

    dataset = {
        "X_geom":   np.stack(X_geom, axis=0),   # (N, 4, ny, nx)
        "Y_tfield": np.stack(Y_tfield, axis=0),  # (N, 1, ny, nx)
        "T_means":  np.array(T_means),
        "meta":     meta,
        "n_samples": len(X_geom),
        "nx": nx, "ny": ny,
        "solver": solver,
    }
    if verbose:
        print(f"Dataset: {len(X_geom)} samples, "
              f"T_mean range [{np.min(T_means):.1f}, {np.max(T_means):.1f}]°C")
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _pde_residual_loss(T_pred: torch.Tensor, solid_mask: torch.Tensor) -> torch.Tensor:
    """∇²T = 0 residual on interior solid nodes (Laplacian via finite differences).

    T_pred   : (B, 1, ny, nx)
    solid_mask: (B, 1, ny, nx)  float, 1=interior solid node
    """
    lap = (
        F.pad(T_pred, (0, 0, 1, 0))[:, :, :-1, :] +  # up
        F.pad(T_pred, (0, 0, 0, 1))[:, :, 1:, :]  +  # down
        F.pad(T_pred, (1, 0, 0, 0))[:, :, :, :-1] +  # left
        F.pad(T_pred, (0, 1, 0, 0))[:, :, :, 1:]  -  # right
        4.0 * T_pred
    )
    # Only penalise interior nodes (not BCs)
    residual = lap * solid_mask
    return (residual ** 2).mean()


def train_fno(
    dataset:      dict[str, Any],
    epochs:       int   = 200,
    batch_size:   int   = 8,
    lr:           float = 1e-3,
    pde_weight:   float = 0.1,
    modes:        int   = 12,
    width:        int   = 32,
    n_layers:     int   = 4,
    dropout:      float = 0.1,
    val_split:    float = 0.1,
    verbose:      bool  = True,
) -> tuple["BladeFNO", dict[str, Any]]:
    """Train BladeFNO on a dataset from generate_training_data().

    Returns (model, history) where history contains loss curves and metrics.
    """
    if not _TORCH:
        raise RuntimeError("PyTorch required for FNO training")

    X = torch.from_numpy(dataset["X_geom"]).float()    # (N, 4, ny, nx)
    Y = torch.from_numpy(dataset["Y_tfield"]).float()  # (N, 1, ny, nx)

    N = len(X)
    n_val  = max(1, int(N * val_split))
    n_train = N - n_val
    idx = torch.randperm(N)
    X_tr, Y_tr = X[idx[:n_train]], Y[idx[:n_train]]
    X_val, Y_val = X[idx[n_train:]], Y[idx[n_train:]]

    # Normalise T-field to [0, 1] range for stable training
    T_min = float(Y_tr.min())
    T_max = float(Y_tr.max())
    T_range = max(T_max - T_min, 1.0)
    Y_tr_n  = (Y_tr  - T_min) / T_range
    Y_val_n = (Y_val - T_min) / T_range

    model = BladeFNO(in_channels=4, out_channels=1,
                     modes=modes, width=width,
                     n_layers=n_layers, dropout=dropout).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_data_loss": [], "train_pde_loss": [], "val_l2_rel": [],
               "T_mean_mae": [], "T_norm": {"min": T_min, "max": T_max}}
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_data_loss = 0.0
        epoch_pde_loss  = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx_b = perm[i:i + batch_size]
            xb = X_tr[idx_b].to(DEVICE)
            yb = Y_tr_n[idx_b].to(DEVICE)

            opt.zero_grad()
            pred = model(xb)
            data_loss = F.mse_loss(pred, yb)

            # PDE residual: solid mask from channel 0, exclude BCs (ch3=outer)
            solid_mask = (xb[:, 0:1] - xb[:, 1:2] - xb[:, 3:4]).clamp(0, 1)
            pde_loss   = _pde_residual_loss(pred, solid_mask)

            loss = data_loss + pde_weight * pde_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_data_loss += float(data_loss.detach())
            epoch_pde_loss  += float(pde_loss.detach())
            n_batches += 1

        sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(DEVICE))
            # Relative L² error on unnormalised T-field
            val_pred_T = val_pred.cpu() * T_range + T_min
            l2_rel = float(
                (torch.norm(val_pred_T - Y_val) / torch.norm(Y_val)).item()
            )
            # T_mean MAE: interior solid nodes only (matching warp_blade_thermal metric)
            # interior = solid (ch0) minus holes (ch1) minus outer wall (ch3)
            interior_mask = (X_val[:, 0:1] - X_val[:, 1:2] - X_val[:, 3:4]).clamp(0, 1)
            # Per-sample masked mean
            mask_sum = interior_mask.sum(dim=(1, 2, 3)).clamp(min=1)
            T_mean_pred = (val_pred_T * interior_mask).sum(dim=(1, 2, 3)) / mask_sum
            T_mean_true = (Y_val       * interior_mask).sum(dim=(1, 2, 3)) / mask_sum
            T_mean_mae  = float((T_mean_pred - T_mean_true).abs().mean().item())

        history["train_data_loss"].append(epoch_data_loss / max(n_batches, 1))
        history["train_pde_loss"].append(epoch_pde_loss  / max(n_batches, 1))
        history["val_l2_rel"].append(l2_rel)
        history["T_mean_mae"].append(T_mean_mae)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            elapsed = time.perf_counter() - t0
            print(f"  epoch {epoch+1:4d}/{epochs}  "
                  f"data={epoch_data_loss/n_batches:.4f}  "
                  f"pde={epoch_pde_loss/n_batches:.4f}  "
                  f"val_l2={l2_rel:.4f}  T_mean_mae={T_mean_mae:.2f}°C  "
                  f"({elapsed:.0f}s)")

    history["final_val_l2_rel"] = history["val_l2_rel"][-1]
    history["final_T_mean_mae"] = history["T_mean_mae"][-1]
    history["epochs"] = epochs
    history["train_samples"] = n_train
    history["val_samples"]   = n_val
    history["backend"] = "pytorch_fno2d"
    history["pde_weight"] = pde_weight
    return model, history


# ---------------------------------------------------------------------------
# Inference with MC-dropout uncertainty
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model:   "BladeFNO",
    X:       np.ndarray,
    n_mc:    int   = 10,
    T_min:   float = 400.0,
    T_max:   float = 1300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MC-dropout inference to get T-field prediction + epistemic uncertainty.

    Parameters
    ----------
    X     : geometry encoding (4, ny, nx) or (B, 4, ny, nx)
    n_mc  : number of stochastic forward passes
    T_min, T_max : unnormalise range (must match training)

    Returns
    -------
    T_mean_pred : (ny, nx) or (B, ny, nx)  mean predicted T-field
    T_std       : (ny, nx) or (B, ny, nx)  std across MC passes (uncertainty)
    """
    if not _TORCH:
        raise RuntimeError("PyTorch required for FNO inference")

    single = X.ndim == 3
    if single:
        X = X[np.newaxis]

    X_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    T_range = max(T_max - T_min, 1.0)

    model.eval()
    model.enable_dropout()  # keep dropout active for MC sampling

    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            p = model(X_t).cpu().numpy()  # (B, 1, ny, nx)
            preds.append(p[:, 0] * T_range + T_min)  # (B, ny, nx)

    preds = np.stack(preds, axis=0)      # (n_mc, B, ny, nx)
    T_mean_pred = preds.mean(axis=0)
    T_std       = preds.std(axis=0)

    if single:
        return T_mean_pred[0], T_std[0]
    return T_mean_pred, T_std


def save_model(model: "BladeFNO", path: str, history: dict | None = None):
    if not _TORCH:
        raise RuntimeError("PyTorch required")
    import torch
    state = {
        "model_state": model.state_dict(),
        "model_kwargs": {
            "in_channels":  model.in_channels,
            "out_channels": model.out_channels,
            "modes":        model.modes,
            "width":        model.width,
            "n_layers":     model.n_layers,
            "dropout":      model.dropout,
        },
        "history": history or {},
    }
    torch.save(state, path)


def load_model(path: str, **model_kwargs) -> "BladeFNO":
    if not _TORCH:
        raise RuntimeError("PyTorch required")
    import torch
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    # Use saved kwargs; caller overrides take precedence
    saved_kwargs = state.get("model_kwargs", {})
    saved_kwargs.update(model_kwargs)
    model = BladeFNO(**saved_kwargs).to(DEVICE)
    model.load_state_dict(state["model_state"])
    model.eval()
    if "history" in state:
        model._history = state["history"]
    return model
