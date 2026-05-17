"""TensorRT and ONNX Runtime inference engine for WorldSIM surrogates.

Inference priority:
    1. TensorRT engine (.trt) — fastest, GPU only
    2. ONNX Runtime (GPU EP if CUDA available, else CPU EP) — cross-platform
    3. PyTorch fallback — always available

Usage
-----
# Export model to ONNX first (from physicsnemo_fno or fno_surrogate):
from simlab.engines.qdgeometry.physicsnemo_fno import export_nemo_onnx, load_nemo_model
model = load_nemo_model("data/blade_nemo.pt")
export_nemo_onnx(model, "data/blade_nemo.onnx")

# Convert to TRT (run in shell):
# trtexec --onnx=data/blade_nemo.onnx --saveEngine=data/blade_nemo.trt --fp16

# Load inference engine:
eng = TRTInferenceEngine.from_onnx("data/blade_nemo.onnx")
# or: eng = TRTInferenceEngine.from_trt("data/blade_nemo.trt")
result = eng.infer(geom_array)  # (4, ny, nx) numpy → (ny, nx) temperature field
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import tensorrt as trt
    _TRT = True
    _TRT_VERSION = trt.__version__
except ImportError:
    _TRT = False
    _TRT_VERSION = None

try:
    import onnxruntime as ort
    _ORT = True
    _ORT_VERSION = ort.__version__
except ImportError:
    _ORT = False
    _ORT_VERSION = None

try:
    import torch
    _TORCH = True
    _TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH = False
    _TORCH_DEVICE = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ort_session(onnx_path: str, prefer_trt: bool = True):
    """Create ONNX Runtime session.

    Provider priority: TensorrtExecutionProvider → CUDAExecutionProvider → CPU.
    TRT EP compiles the ONNX to a TRT engine on first run (cached to disk).
    """
    available = ort.get_available_providers() if _ORT else []
    providers: list = []
    if prefer_trt and "TensorrtExecutionProvider" in available:
        providers.append((
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "data/trt_cache",
            },
        ))
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(onnx_path, providers=providers)


def _active_backend() -> str:
    if _ORT:
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            return "onnxruntime_trt"
        if "CUDAExecutionProvider" in available:
            return "onnxruntime_gpu"
        return "onnxruntime_cpu"
    if _TORCH:
        return "pytorch_fallback"
    return "unavailable"


# ---------------------------------------------------------------------------
# TRT engine runner (pycuda-based)
# ---------------------------------------------------------------------------

class _TRTRunner:
    """Thin wrapper around a loaded TensorRT engine for numpy I/O.

    Uses torch CUDA tensors for device memory — no pycuda required.
    Requires TensorRT 8+ Python API and a CUDA-enabled torch install.
    """

    def __init__(self, engine_path: str):
        if not _TRT:
            raise RuntimeError("tensorrt is not installed")
        if not _TORCH:
            raise RuntimeError("torch is required for _TRTRunner device memory")

        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

    def infer(self, x: np.ndarray) -> np.ndarray:
        import torch

        x_t   = torch.from_numpy(x.astype(np.float32)).cuda()
        B, C, H, W = x_t.shape
        out_t = torch.empty(B, 1, H, W, dtype=torch.float32, device="cuda")

        # TRT 10 execute_v2: bindings are raw CUDA pointers
        bindings = [x_t.data_ptr(), out_t.data_ptr()]
        self._context.execute_v2(bindings)
        torch.cuda.synchronize()
        return out_t.cpu().numpy()


# ---------------------------------------------------------------------------
# Public inference engine
# ---------------------------------------------------------------------------

class TRTInferenceEngine:
    """Unified inference engine for BladeFNO/BladeFNONeMo models.

    Wraps TensorRT, ONNX Runtime, or PyTorch in a single ``infer()`` call.
    Instantiate via class methods:
        TRTInferenceEngine.from_trt(path)   — load TRT .trt engine
        TRTInferenceEngine.from_onnx(path)  — load ONNX model
        TRTInferenceEngine.from_torch(model)— wrap existing PyTorch model
    """

    def __init__(self, runner, backend: str, metadata: dict | None = None):
        self._runner  = runner
        self.backend  = backend
        self.metadata = metadata or {}

    @classmethod
    def from_trt(cls, trt_path: str) -> "TRTInferenceEngine":
        """Load a serialised TensorRT engine (.trt file)."""
        if not _TRT:
            raise RuntimeError(
                "tensorrt is not installed. Install with: "
                "pip install tensorrt pycuda  (requires CUDA toolkit)"
            )
        runner = _TRTRunner(trt_path)
        return cls(runner, backend="tensorrt", metadata={"engine_path": trt_path})

    @classmethod
    def from_onnx(cls, onnx_path: str) -> "TRTInferenceEngine":
        """Load an ONNX model (uses ONNX Runtime, CUDA EP if available)."""
        if not _ORT:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Install: pip install onnxruntime-gpu  (or onnxruntime for CPU)"
            )
        session = _ort_session(onnx_path)
        providers = session.get_providers()
        backend = "onnxruntime_gpu" if any("CUDA" in p for p in providers) else "onnxruntime_cpu"
        return cls(session, backend=backend, metadata={
            "onnx_path": onnx_path,
            "providers": providers,
        })

    @classmethod
    def from_torch(
        cls,
        model,
        T_min: float = 400.0,
        T_max: float = 1300.0,
    ) -> "TRTInferenceEngine":
        """Wrap an existing PyTorch BladeFNO/BladeFNONeMo model."""
        if not _TORCH:
            raise RuntimeError("torch is not installed")
        runner = {"model": model, "T_min": T_min, "T_max": T_max}
        return cls(runner, backend="pytorch_fallback")

    def infer(
        self,
        geom: np.ndarray,
        T_min: float = 400.0,
        T_max: float = 1300.0,
    ) -> dict[str, Any]:
        """Run inference on a geometry array.

        Parameters
        ----------
        geom   : (4, ny, nx) or (B, 4, ny, nx) float32 numpy array
        T_min  : temperature for de-normalisation (coolant temperature)
        T_max  : temperature for de-normalisation (hot gas temperature)

        Returns
        -------
        dict with:
            T_field   (ny, nx) temperature field [°C] — squeezed if single sample
            T_mean    mean interior temperature
            backend   which engine was used
            latency_ms wall time in milliseconds
        """
        if geom.ndim == 3:
            geom = geom[None]   # (1, 4, ny, nx)

        geom = geom.astype(np.float32)
        t0   = time.perf_counter()

        if self.backend == "tensorrt":
            out_n = self._runner.infer(geom)   # (B, 1, ny, nx)

        elif self.backend in ("onnxruntime_gpu", "onnxruntime_cpu"):
            session = self._runner
            input_name  = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            out_n = session.run([output_name], {input_name: geom})[0]

        elif self.backend == "pytorch_fallback":
            import torch
            model   = self._runner["model"]
            device  = next(model.parameters()).device
            X       = torch.from_numpy(geom).to(device)
            model.eval()
            with torch.no_grad():
                out_n = model(X).cpu().numpy()

        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # De-normalise: [0,1] → [T_min, T_max]
        T_field_n = out_n[0, 0]   # (ny, nx)
        T_field   = T_field_n * (T_max - T_min) + T_min

        return {
            "T_field":    T_field.tolist(),
            "T_mean":     float(T_field.mean()),
            "T_norm":     T_field_n.tolist(),
            "backend":    self.backend,
            "latency_ms": round(latency_ms, 2),
            "shape":      list(T_field.shape),
        }

    @staticmethod
    def available_backends() -> dict[str, Any]:
        """Report which inference backends are installed."""
        ort_providers = ort.get_available_providers() if _ORT else []
        return {
            "tensorrt":              _TRT,
            "tensorrt_version":      _TRT_VERSION,
            "onnxruntime":           _ORT,
            "onnxruntime_version":   _ORT_VERSION,
            "ort_trt_ep":            "TensorrtExecutionProvider" in ort_providers,
            "ort_cuda_ep":           "CUDAExecutionProvider" in ort_providers,
            "ort_providers":         ort_providers,
            "pytorch":               _TORCH,
            "recommended":           _active_backend(),
            "note": (
                "FNO2DEncoder cannot be ONNX-exported (dynamic FFT shapes). "
                "Use TorchScript (.pt) for BladeFNONeMo deployment. "
                "ORT TRT EP works for static ONNX models (e.g. BladeFNO from fno_surrogate.py)."
            ),
        }
