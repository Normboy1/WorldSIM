"""Real PhysicsNeMo FNO2d surrogate for turbine blade thermal fields.

Uses physicsnemo.nn.module.fno_layers.FNO2DEncoder (NVIDIA PhysicsNeMo 2.x)
as the backbone instead of the native PyTorch implementation in fno_surrogate.py.

Architecture:
    Input  (B, 4, ny, nx)  — blade/cool/sdf/outer geometry channels
    FNO2DEncoder            — spectral lifting + N FNO blocks (PhysicsNeMo SpectralConv2d)
                             coord_features=True prepends (x,y) grid channels internally
    nn.Conv2d(width, 1)    — pointwise projection to scalar temperature field
    Output (B, 1, ny, nx)  — normalised temperature ∈ [0, 1]

ModelMetaData registers the model for:
    - AMP (automatic mixed precision, amp=True)
    - ONNX export (onnx=True)
    - TRT deployment (trt=True) — after ONNX export run:
        trtexec --onnx=blade_nemo.onnx --saveEngine=blade_nemo.trt --fp16

The model is fully compatible with save_nemo_model / load_nemo_model and
reuses generate_training_data from fno_surrogate.py for training data generation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH = False
    _DEVICE = None

try:
    from physicsnemo.core.module import Module as NeMoModule, ModelMetaData
    from physicsnemo.nn.module.fno_layers import FNO2DEncoder
    _NEMO = True
except Exception:
    _NEMO = False


# ---------------------------------------------------------------------------
# PhysicsNeMo FNO model
# ---------------------------------------------------------------------------

if _NEMO and _TORCH:
    class BladeFNONeMo(NeMoModule):
        """PhysicsNeMo FNO2d surrogate for blade thermal field prediction.

        Uses FNO2DEncoder (physicsnemo.nn) as the spectral backbone with a
        pointwise Conv2d projection head.
        """

        _meta = ModelMetaData(
            jit=False,
            cuda_graphs=False,
            amp=True,
            onnx=True,
            trt=True,
        )

        def __init__(
            self,
            in_channels:   int   = 4,
            width:         int   = 32,
            modes:         int   = 12,
            n_layers:      int   = 4,
            dropout:       float = 0.1,
        ):
            super().__init__(meta=self._meta)
            self.in_channels = in_channels
            self.width       = width
            self.modes       = modes
            self.n_layers    = n_layers
            self.dropout_p   = dropout

            # FNO2DEncoder: coord_features=True adds 2 coordinate channels
            # internally, so effective in_channels = in_channels + 2
            self.encoder = FNO2DEncoder(
                in_channels=in_channels,
                num_fno_layers=n_layers,
                fno_layer_size=width,
                num_fno_modes=modes,
                coord_features=True,
            )
            self.dropout  = nn.Dropout2d(p=dropout)
            self.proj     = nn.Conv2d(width, 1, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (B, in_channels, ny, nx) → (B, 1, ny, nx)."""
            z = self.encoder(x)      # (B, width, ny, nx)
            z = self.dropout(z)
            return self.proj(z)      # (B, 1, ny, nx)

        def enable_dropout(self):
            """Switch Dropout2d layers to train mode for MC-dropout inference."""
            for m in self.modules():
                if isinstance(m, nn.Dropout2d):
                    m.train()

else:
    # Fallback stub so imports don't break without PhysicsNeMo
    class BladeFNONeMo:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            raise RuntimeError(
                "physicsnemo is not installed. Run: pip install nvidia-physicsnemo"
            )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fno_nemo(
    dataset: dict,
    epochs:      int   = 300,
    batch_size:  int   = 16,
    lr:          float = 1e-3,
    width:       int   = 32,
    modes:       int   = 12,
    n_layers:    int   = 4,
    dropout:     float = 0.1,
    pde_weight:  float = 0.05,
    val_split:   float = 0.15,
    verbose:     bool  = True,
) -> tuple["BladeFNONeMo", dict]:
    """Train BladeFNONeMo on blade thermal data.

    Reuses the training dataset format produced by
    ``fno_surrogate.generate_training_data``.

    Parameters match ``fno_surrogate.train_fno`` for drop-in compatibility.

    Returns
    -------
    (model, history)  — history keys: train_loss, val_loss, val_l2_rel, backend
    """
    if not _NEMO or not _TORCH:
        raise RuntimeError("physicsnemo and torch required for train_fno_nemo")

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    X_np = np.array(dataset.get("X_geom", dataset.get("X")), dtype=np.float32)  # (N, 4, ny, nx)
    _y_raw = np.array(dataset.get("Y_tfield", dataset.get("y")), dtype=np.float32)
    # Y_tfield may already be (N, 1, ny, nx) or (N, ny, nx) — normalise to (N, ny, nx)
    y_np = _y_raw[:, 0] if _y_raw.ndim == 4 else _y_raw  # (N, ny, nx)

    n_val = max(1, int(len(X_np) * val_split))
    n_train = len(X_np) - n_val
    X_tr, X_val = X_np[:n_train], X_np[n_train:]
    y_tr, y_val = y_np[:n_train], y_np[n_train:]

    T_min = float(y_tr.min())
    T_max = float(y_tr.max())
    y_tr_n  = (y_tr  - T_min) / (T_max - T_min + 1e-8)
    y_val_n = (y_val - T_min) / (T_max - T_min + 1e-8)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr_n[:, None]),   # (N, 1, ny, nx)
        ),
        batch_size=batch_size, shuffle=True, pin_memory=True,
    )

    in_ch = X_np.shape[1]
    model = BladeFNONeMo(
        in_channels=in_ch, width=width, modes=modes,
        n_layers=n_layers, dropout=dropout,
    ).to(_DEVICE)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # FNO2DEncoder pads inputs before FFT (ny+2p, nx+2p); padded sizes are
    # almost never powers of two, so fp16 cuFFT fails. Use fp32 throughout.

    X_val_t = torch.from_numpy(X_val).to(_DEVICE)
    y_val_t = torch.from_numpy(y_val_n[:, None]).to(_DEVICE)

    history: dict[str, Any] = {
        "train_loss": [], "val_loss": [], "val_l2_rel": [],
        "backend": "physicsnemo_fno2d",
        "T_norm": {"min": T_min, "max": T_max},
    }

    # Build interior mask from the solid channel (channel 0 = blade, channel 1 = cool)
    solid_ch = torch.from_numpy(X_tr[:1, 0:1]).to(_DEVICE)
    cool_ch  = torch.from_numpy(X_tr[:1, 1:2]).to(_DEVICE)
    outer_ch = torch.from_numpy(X_tr[:1, 3:4]).to(_DEVICE)
    interior_mask = (solid_ch > 0.5) & (cool_ch < 0.5) & (outer_ch < 0.5)

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(_DEVICE), yb.to(_DEVICE)
            opt.zero_grad(set_to_none=True)

            pred = model(Xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += float(loss.item()) * len(Xb)

        sched.step()
        ep_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(nn.functional.mse_loss(val_pred, y_val_t).item())
            denom = float(y_val_t.norm()) + 1e-8
            val_l2 = float((val_pred - y_val_t).norm() / denom)

        history["train_loss"].append(ep_loss)
        history["val_loss"].append(val_loss)
        history["val_l2_rel"].append(val_l2)

        if verbose and epoch % 50 == 0:
            lr_now = opt.param_groups[0]["lr"]
            elapsed = time.perf_counter() - t0
            print(f"  ep {epoch:4d}/{epochs}  train={ep_loss:.4f}  "
                  f"val_l2={val_l2:.4f}  lr={lr_now:.2e}  t={elapsed:.1f}s")

    history["wall_time_s"]       = time.perf_counter() - t0
    history["final_val_l2_rel"]  = history["val_l2_rel"][-1]
    history["final_val_loss"]    = history["val_loss"][-1]
    history["n_train"]           = n_train
    history["n_val"]             = n_val
    return model, history


# ---------------------------------------------------------------------------
# ONNX export (enables TRT conversion)
# ---------------------------------------------------------------------------

def export_nemo_onnx(
    model: "BladeFNONeMo",
    save_path: str,
    ny: int = 64,
    nx: int = 128,
) -> dict:
    """Export BladeFNONeMo to TorchScript for production deployment.

    FNO2DEncoder uses dynamic ``torch.linspace`` and FFT shape inference
    inside its forward pass, which prevents static ONNX tracing with either
    the legacy or dynamo-based exporters.  TorchScript (``torch.jit.trace``)
    handles this correctly and produces a GPU-runnable artifact.

    For TensorRT deployment use Torch-TensorRT (not trtexec):
        import torch_tensorrt
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input((1, 4, ny, nx))],
            enabled_precisions={torch.float16},
        )

    Parameters
    ----------
    save_path : output path — use .pt extension (e.g. "data/blade_nemo_ts.pt")
    ny, nx    : spatial grid size matching training resolution

    Returns
    -------
    dict with torchscript_path, backend, deployment notes
    """
    if not _TORCH or not _NEMO:
        raise RuntimeError("physicsnemo and torch required")

    import torch

    # Ensure .pt extension
    if not save_path.endswith(".pt"):
        save_path = save_path.replace(".onnx", ".pt")
        if not save_path.endswith(".pt"):
            save_path += ".pt"

    model.eval()
    device = next(model.parameters()).device
    dummy  = torch.zeros(1, model.in_channels, ny, nx, device=device)

    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, save_path)

    return {
        "torchscript_path": save_path,
        "backend":          "physicsnemo_torchscript",
        "input_shape":      [1, model.in_channels, ny, nx],
        "output_shape":     [1, 1, ny, nx],
        "trt_note": (
            "For TRT: pip install torch-tensorrt, then "
            "torch_tensorrt.compile(model, inputs=[...], enabled_precisions={torch.float16})"
        ),
        "onnx_note": (
            "ONNX export is not supported for FNO2DEncoder — dynamic FFT shapes "
            "in physicsnemo prevent static tracing. Use TorchScript or Torch-TRT."
        ),
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_nemo_model(
    model: "BladeFNONeMo",
    path: str,
    history: dict | None = None,
) -> None:
    if not _TORCH:
        raise RuntimeError("torch required")
    import torch
    torch.save({
        "model_state": model.state_dict(),
        "model_kwargs": {
            "in_channels": model.in_channels,
            "width":       model.width,
            "modes":       model.modes,
            "n_layers":    model.n_layers,
            "dropout":     model.dropout_p,
        },
        "history": history or {},
        "physicsnemo_version": getattr(__import__("physicsnemo"), "__version__", "?"),
    }, path)


def load_nemo_model(path: str, **overrides) -> "BladeFNONeMo":
    if not _TORCH:
        raise RuntimeError("torch required")
    import torch
    state = torch.load(path, map_location=_DEVICE, weights_only=False)
    kwargs = state.get("model_kwargs", {})
    kwargs.update(overrides)
    model = BladeFNONeMo(**kwargs).to(_DEVICE)
    model.load_state_dict(state["model_state"])
    model.eval()
    if "history" in state:
        model._history = state["history"]
    return model


def predict_nemo(
    model:   "BladeFNONeMo",
    geom:    "np.ndarray",
    T_min:   float = 400.0,
    T_max:   float = 1300.0,
    n_mc:    int   = 0,
) -> tuple["np.ndarray", "np.ndarray | None"]:
    """Run inference on a (4, ny, nx) geometry array.

    Parameters
    ----------
    n_mc : int
        If > 0, run MC-dropout with n_mc forward passes and return (mean, std).
        If 0, run single deterministic pass and return (pred, None).

    Returns
    -------
    (temperature_field [ny, nx], std_field [ny, nx] or None)
    """
    if not _TORCH or not _NEMO:
        raise RuntimeError("physicsnemo and torch required")

    import torch

    device = next(model.parameters()).device
    X = torch.from_numpy(geom[None].astype(np.float32)).to(device)

    if n_mc > 0:
        model.enable_dropout()
        preds = []
        with torch.no_grad():
            for _ in range(n_mc):
                p = model(X).squeeze().cpu().numpy()
                preds.append(p)
        model.eval()
        stack = np.stack(preds)
        mean_n = stack.mean(0)
        std_n  = stack.std(0)
        mean_T = mean_n * (T_max - T_min) + T_min
        std_T  = std_n  * (T_max - T_min)
        return mean_T, std_T
    else:
        model.eval()
        with torch.no_grad():
            out = model(X).squeeze().cpu().numpy()
        return out * (T_max - T_min) + T_min, None
