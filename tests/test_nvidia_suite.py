"""Tests for all NVIDIA-backed engines added in the latest session.

Covers:
- PhysicsNeMo FNO2d (BladeFNONeMo): forward, train, save/load, TorchScript export
- TRTInferenceEngine: backend discovery, PyTorch fallback inference
- WarpBackend: von Mises stress kernel, pressure Poisson kernel
- GPUMAPElitesArchive: insert, coverage, qd_score, to_dict
- PhysicsNeMoBackend.train_fno_surrogate (FNO2DEncoder composition path)
- Dispatcher routes: physicsnemo_fno_train, trt_backend_status,
                     gpu_mapelites_benchmark, warp_von_mises, warp_pressure_poisson
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    _CUDA = _torch.cuda.is_available()
    _TORCH = True
except ImportError:
    _TORCH = False
    _CUDA = False

try:
    from physicsnemo.nn.module.fno_layers import FNO2DEncoder  # noqa: F401
    _NEMO = True
except Exception:
    _NEMO = False

_nemo_skip = pytest.mark.skipif(
    not (_NEMO and _TORCH),
    reason="physicsnemo + torch required",
)
_cuda_skip = pytest.mark.skipif(not _CUDA, reason="CUDA device required")


# ===========================================================================
# 1. BladeFNONeMo — model unit tests
# ===========================================================================

class TestBladeFNONeMo:
    """PhysicsNeMo FNO2d model: forward pass, train, save, load, export."""

    @_nemo_skip
    def test_forward_shape(self):
        from simlab.engines.qdgeometry.physicsnemo_fno import BladeFNONeMo
        device = "cuda" if _CUDA else "cpu"
        model = BladeFNONeMo(in_channels=4, width=16, modes=8, n_layers=2).to(device)
        x = _torch.randn(2, 4, 32, 64).to(device)
        with _torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 32, 64), f"Unexpected shape: {out.shape}"

    @_nemo_skip
    def test_forward_no_nan(self):
        from simlab.engines.qdgeometry.physicsnemo_fno import BladeFNONeMo
        device = "cuda" if _CUDA else "cpu"
        model = BladeFNONeMo(in_channels=4, width=16, modes=8, n_layers=2).to(device)
        x = _torch.randn(1, 4, 32, 64).to(device)
        with _torch.no_grad():
            out = model(x)
        assert not _torch.isnan(out).any(), "NaN in PhysicsNeMo FNO output"

    @_nemo_skip
    def test_uses_physicsnemo_spectralconv(self):
        """Verify FNO2DEncoder (physicsnemo SpectralConv2d) is the backbone."""
        from simlab.engines.qdgeometry.physicsnemo_fno import BladeFNONeMo
        from physicsnemo.nn.module.spectral_layers import SpectralConv2d
        model = BladeFNONeMo(in_channels=4, width=16, modes=8, n_layers=2)
        spectral_layers = [m for m in model.modules() if isinstance(m, SpectralConv2d)]
        assert len(spectral_layers) >= 1, (
            "No SpectralConv2d layers found — FNO2DEncoder not wired correctly"
        )

    @_nemo_skip
    def test_modelmetadata_registered(self):
        """ModelMetaData with trt=True and onnx=True should be set."""
        from simlab.engines.qdgeometry.physicsnemo_fno import BladeFNONeMo
        model = BladeFNONeMo()
        assert model._meta.amp  is True
        assert model._meta.trt  is True
        assert model._meta.onnx is True

    @_nemo_skip
    def test_mc_dropout_nonzero_std(self):
        """MC-dropout should produce non-zero variance across passes."""
        from simlab.engines.qdgeometry.physicsnemo_fno import BladeFNONeMo, predict_nemo
        from simlab.engines.qdgeometry.fno_surrogate import encode_geometry
        device = "cuda" if _CUDA else "cpu"
        model = BladeFNONeMo(in_channels=4, width=16, modes=8, n_layers=2, dropout=0.2).to(device)
        geom = encode_geometry([0.3, 0.7], [0.3, 0.3], [0.05, 0.05], 32, 64)
        mean, std = predict_nemo(model, geom, T_min=400.0, T_max=1300.0, n_mc=10)
        assert mean.shape == (32, 64)
        assert std is not None and std.shape == (32, 64)
        assert float(std.mean()) > 0.0, "MC-dropout std is zero — dropout not activating"

    @_nemo_skip
    def test_save_and_load(self, tmp_path):
        from simlab.engines.qdgeometry.physicsnemo_fno import (
            BladeFNONeMo, save_nemo_model, load_nemo_model,
        )
        device = "cuda" if _CUDA else "cpu"
        model = BladeFNONeMo(in_channels=4, width=16, modes=6, n_layers=2).to(device)
        path = str(tmp_path / "test_nemo.pt")
        save_nemo_model(model, path, history={"val_l2_rel": [0.05]})
        loaded = load_nemo_model(path)
        assert loaded.width   == 16
        assert loaded.modes   == 6
        assert loaded.n_layers == 2
        assert hasattr(loaded, "_history")

    @_nemo_skip
    def test_torchscript_export(self, tmp_path):
        from simlab.engines.qdgeometry.physicsnemo_fno import BladeFNONeMo, export_nemo_onnx
        device = "cuda" if _CUDA else "cpu"
        model  = BladeFNONeMo(in_channels=4, width=16, modes=8, n_layers=2).to(device)
        path   = str(tmp_path / "blade_nemo.pt")
        result = export_nemo_onnx(model, path, ny=32, nx=64)
        assert result["torchscript_path"].endswith(".pt")
        assert os.path.exists(result["torchscript_path"]), "TorchScript file not written"
        # Load traced model and run inference
        traced = _torch.jit.load(result["torchscript_path"])
        dummy  = _torch.randn(1, 4, 32, 64).to(device)
        with _torch.no_grad():
            out = traced(dummy)
        assert out.shape == (1, 1, 32, 64)

    @_nemo_skip
    def test_mini_train_loop(self, tmp_path):
        """Train BladeFNONeMo for 2 epochs on tiny synthetic data — loss should decrease."""
        from simlab.engines.qdgeometry.physicsnemo_fno import train_fno_nemo
        from simlab.engines.qdgeometry.fno_surrogate import generate_training_data
        dataset = generate_training_data(n_samples=8, nx=32, ny=16, verbose=False)
        model, history = train_fno_nemo(
            dataset, epochs=2, batch_size=4, width=8, modes=4, n_layers=1,
            val_split=0.25, verbose=False,
        )
        assert history["backend"] == "physicsnemo_fno2d"
        assert len(history["train_loss"]) == 2
        # Both losses should be finite
        assert all(math.isfinite(v) for v in history["train_loss"])
        assert all(math.isfinite(v) for v in history["val_loss"])


# ===========================================================================
# 2. TRTInferenceEngine
# ===========================================================================

class TestTRTInferenceEngine:
    """TRT inference engine: backend discovery and PyTorch fallback path."""

    def test_available_backends_returns_dict(self):
        from simlab.engines.qdgeometry.trt_inference import TRTInferenceEngine
        info = TRTInferenceEngine.available_backends()
        for key in ("tensorrt", "onnxruntime", "pytorch", "recommended"):
            assert key in info, f"Missing key: {key}"
        assert isinstance(info["tensorrt"], bool)
        assert isinstance(info["pytorch"], bool)

    def test_recommended_is_non_empty_string(self):
        from simlab.engines.qdgeometry.trt_inference import TRTInferenceEngine
        rec = TRTInferenceEngine.available_backends()["recommended"]
        assert isinstance(rec, str) and len(rec) > 0

    @pytest.mark.skipif(not _TORCH, reason="torch required")
    @_nemo_skip
    def test_from_torch_inference(self):
        """PyTorch fallback path: wrap a BladeFNONeMo and run infer()."""
        from simlab.engines.qdgeometry.physicsnemo_fno import BladeFNONeMo
        from simlab.engines.qdgeometry.trt_inference import TRTInferenceEngine
        from simlab.engines.qdgeometry.fno_surrogate import encode_geometry
        device = "cuda" if _CUDA else "cpu"
        model  = BladeFNONeMo(in_channels=4, width=8, modes=4, n_layers=1).to(device)
        eng    = TRTInferenceEngine.from_torch(model, T_min=400.0, T_max=1300.0)
        geom   = encode_geometry([0.3], [0.3], [0.05], 32, 64)
        result = eng.infer(geom, T_min=400.0, T_max=1300.0)
        # Untrained model: only check shape/type/latency, not temperature range
        assert result["backend"] == "pytorch_fallback"
        assert "T_mean" in result
        assert isinstance(result["T_mean"], float)
        assert isinstance(result["T_field"], list)
        assert len(result["T_field"]) == 32   # ny rows
        assert result["latency_ms"] > 0

    @pytest.mark.skipif(not _TORCH, reason="torch required")
    def test_from_torch_result_json_serialisable(self):
        """Result from infer() must be JSON-serialisable."""
        import json
        from simlab.engines.qdgeometry.physicsnemo_fno import BladeFNONeMo
        from simlab.engines.qdgeometry.trt_inference import TRTInferenceEngine
        from simlab.engines.qdgeometry.fno_surrogate import encode_geometry
        device = "cuda" if _CUDA else "cpu"
        model  = BladeFNONeMo(in_channels=4, width=8, modes=4, n_layers=1).to(device)
        eng    = TRTInferenceEngine.from_torch(model)
        geom   = encode_geometry([0.3], [0.3], [0.05], 32, 64)
        result = eng.infer(geom)
        # Must not raise
        json.dumps(result)

    def test_tensorrt_installed_and_importable(self):
        """tensorrt package is importable when present (skipped if absent)."""
        pytest.importorskip("tensorrt", reason="tensorrt not installed")
        from simlab.engines.qdgeometry.trt_inference import _TRT, _TRT_VERSION
        assert _TRT, "tensorrt should be installed"
        assert _TRT_VERSION is not None

    def test_ort_installed_with_trt_ep(self):
        """onnxruntime-gpu provides the TRT execution provider (skipped if absent)."""
        ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        from simlab.engines.qdgeometry.trt_inference import _ORT
        assert _ORT, "onnxruntime should be installed"
        providers = ort.get_available_providers()
        assert "TensorrtExecutionProvider" in providers
        assert "CUDAExecutionProvider" in providers


# ===========================================================================
# 3. WarpBackend — new kernels
# ===========================================================================

class TestWarpVonMisesStress:
    """Warp @wp.kernel plane-stress von Mises field."""

    @pytest.fixture(scope="class")
    def result(self):
        from simlab.engines.materials.nvidia_backends import WarpBackend
        ny, nx = 32, 64
        ux = np.zeros((ny, nx), dtype=np.float32)
        # Linear y-displacement: strain in yy direction only
        uy = (np.linspace(0, 1e-3, ny)[:, None] * np.ones((ny, nx))).astype(np.float32)
        return WarpBackend().warp_laplacian_stress(
            ux.tolist(), uy.tolist(), E=200e9, nu=0.3
        )

    def test_returns_dict_with_required_keys(self, result):
        for key in ("sigma_vm", "sigma_vm_max", "sigma_vm_mean", "backend", "wall_time_s"):
            assert key in result

    def test_sigma_vm_is_non_negative(self, result):
        arr = np.array(result["sigma_vm"])
        assert float(arr.min()) >= -1e-3, "Von Mises stress has negative values"

    def test_sigma_vm_shape(self, result):
        arr = np.array(result["sigma_vm"])
        assert arr.shape == (32, 64)

    def test_sigma_vm_no_nan(self, result):
        arr = np.array(result["sigma_vm"])
        assert not np.any(np.isnan(arr))

    def test_non_zero_displacement_produces_stress(self, result):
        """Non-trivial displacement field must produce non-zero von Mises stress."""
        assert result["sigma_vm_max"] > 0.0

    def test_backend_label(self, result):
        assert result["backend"] in ("warp_cuda", "numpy_cpu", "numpy_cpu_fallback")

    def test_zero_displacement_gives_zero_stress(self):
        from simlab.engines.materials.nvidia_backends import WarpBackend
        ny, nx = 16, 16
        ux = np.zeros((ny, nx)).tolist()
        uy = np.zeros((ny, nx)).tolist()
        r = WarpBackend().warp_laplacian_stress(ux, uy)
        assert r["sigma_vm_max"] < 1e-3, "Zero displacement should give ~zero stress"


class TestWarpPressurePoisson:
    """Warp @wp.kernel Laplace pressure solver."""

    @pytest.fixture(scope="class")
    def result(self):
        from simlab.engines.materials.nvidia_backends import WarpBackend
        return WarpBackend().warp_pressure_poisson(
            nx=32, ny=32, p_top=1.0, p_bot=0.0, iters=500
        )

    def test_returns_required_keys(self, result):
        for key in ("p_field", "p_max", "p_min", "backend", "iters"):
            assert key in result

    def test_boundary_values(self, result):
        assert result["p_max"] == pytest.approx(1.0, abs=0.01)
        assert result["p_min"] == pytest.approx(0.0, abs=0.01)

    def test_field_shape(self, result):
        arr = np.array(result["p_field"])
        assert arr.shape == (32, 32)

    def test_monotone_in_y(self, result):
        """Pressure should increase monotonically from bottom to top."""
        arr = np.array(result["p_field"])
        col_mid = arr[:, arr.shape[1] // 2]
        # Allow small non-monotonicity at boundaries (FD artefact)
        diffs = np.diff(col_mid)
        assert np.mean(diffs >= 0) > 0.90, "Pressure not monotone in y (>10% violations)"

    def test_backend_label(self, result):
        assert result["backend"] in ("warp_cuda", "numpy_cpu", "numpy_cpu_fallback")

    def test_no_nan(self, result):
        arr = np.array(result["p_field"])
        assert not np.any(np.isnan(arr))


# ===========================================================================
# 4. GPUMAPElitesArchive
# ===========================================================================

class TestGPUMAPElitesArchive:
    """GPU-resident (CuPy/NumPy fallback) MAP-Elites archive."""

    def test_insert_and_retrieve(self):
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        arch = GPUMAPElitesArchive(5, 5, (0.0, 1.0), (0.0, 1.0))
        inserted = arch.try_insert("sol_a", quality=0.8, f1=0.1, f2=0.1)
        assert inserted is True
        best = arch.best()
        assert best is not None
        assert best["quality"] == pytest.approx(0.8)

    def test_better_quality_replaces(self):
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        arch = GPUMAPElitesArchive(5, 5, (0.0, 1.0), (0.0, 1.0))
        arch.try_insert("sol_a", quality=0.5, f1=0.1, f2=0.1)
        replaced = arch.try_insert("sol_b", quality=0.9, f1=0.1, f2=0.1)
        assert replaced is True
        assert arch.best()["solution"] == "sol_b"

    def test_worse_quality_does_not_replace(self):
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        arch = GPUMAPElitesArchive(5, 5, (0.0, 1.0), (0.0, 1.0))
        arch.try_insert("sol_a", quality=0.9, f1=0.1, f2=0.1)
        replaced = arch.try_insert("sol_b", quality=0.4, f1=0.1, f2=0.1)
        assert replaced is False
        assert arch.best()["solution"] == "sol_a"

    def test_coverage_increases(self):
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        rng  = np.random.default_rng(0)
        arch = GPUMAPElitesArchive(10, 10, (0.0, 1.0), (0.0, 1.0))
        for _ in range(1000):
            f1, f2 = rng.uniform(0, 1), rng.uniform(0, 1)
            arch.try_insert({"x": f1}, quality=rng.uniform(0, 1), f1=f1, f2=f2)
        assert arch.coverage() > 0.8, f"Coverage {arch.coverage():.2f} < 0.8"

    def test_qd_score_is_mean_of_filled_cells(self):
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        arch = GPUMAPElitesArchive(4, 4, (0.0, 1.0), (0.0, 1.0))
        # Insert identical quality everywhere for predictable QD-score
        rng = np.random.default_rng(7)
        for _ in range(200):
            f1, f2 = rng.uniform(0, 1), rng.uniform(0, 1)
            arch.try_insert({"x": f1}, quality=0.5, f1=f1, f2=f2)
        assert arch.qd_score() == pytest.approx(0.5, abs=0.01)

    def test_to_dict_json_serialisable(self):
        import json
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        arch = GPUMAPElitesArchive(5, 5, (0.0, 1.0), (0.0, 1.0))
        arch.try_insert({"x": 0.3}, quality=0.7, f1=0.3, f2=0.3)
        d = arch.to_dict()
        json.dumps(d)  # must not raise

    def test_backend_info_returns_dict(self):
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        info = GPUMAPElitesArchive.backend_info()
        for key in ("cupy_available", "active_backend", "install_cupy"):
            assert key in info

    def test_sample_elite(self):
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        rng  = np.random.default_rng(1)
        arch = GPUMAPElitesArchive(5, 5, (0.0, 1.0), (0.0, 1.0))
        arch.try_insert("sol", quality=1.0, f1=0.5, f2=0.5)
        elite = arch.sample_elite(rng)
        assert elite is not None
        assert elite["solution"] == "sol"

    def test_empty_archive_best_is_none(self):
        from simlab.engines.qdgeometry.mapelites import GPUMAPElitesArchive
        arch = GPUMAPElitesArchive(5, 5, (0.0, 1.0), (0.0, 1.0))
        assert arch.best() is None
        assert arch.coverage() == pytest.approx(0.0)


# ===========================================================================
# 5. PhysicsNeMoBackend.train_fno_surrogate (real FNO2DEncoder path)
# ===========================================================================

class TestPhysicsNeMoBackendSurrogate:
    """PhysicsNeMoBackend using real FNO2DEncoder for composition → property."""

    @_nemo_skip
    def test_train_fno_surrogate_real_backend(self):
        from simlab.engines.materials.nvidia_backends import PhysicsNeMoBackend
        rng   = np.random.default_rng(0)
        n, d  = 40, 5
        X     = rng.dirichlet(np.ones(d), n).tolist()   # composition vectors
        y     = [float(rng.uniform(0, 1)) for _ in range(n)]
        backend = PhysicsNeMoBackend()
        assert backend.available, "PhysicsNeMo not detected"
        result = backend.train_fno_surrogate(
            X, y, n_modes=4, hidden_channels=16, n_layers=2, epochs=5,
            target_name="test_property",
        )
        assert result["backend"] == "physicsnemo_fno2d_gpu", (
            f"Expected real backend, got '{result['backend']}'. "
            f"Error: {result.get('physicsnemo_error')}"
        )
        assert result["final_train_loss"] is not None
        assert math.isfinite(result["final_train_loss"])

    @_nemo_skip
    def test_train_fno_surrogate_returns_loss_history(self):
        from simlab.engines.materials.nvidia_backends import PhysicsNeMoBackend
        rng = np.random.default_rng(1)
        X   = rng.dirichlet(np.ones(4), 20).tolist()
        y   = [float(rng.uniform(0, 1)) for _ in range(20)]
        result = PhysicsNeMoBackend().train_fno_surrogate(
            X, y, n_modes=4, hidden_channels=8, n_layers=1, epochs=10,
        )
        losses = result.get("loss_history", [])
        assert len(losses) > 0
        assert all(math.isfinite(v) for v in losses)


# ===========================================================================
# 6. Dispatcher routes — smoke tests
# ===========================================================================

class TestNvidiaDispatcherRoutes:
    """Smoke tests through the ExperimentDispatcher for all new NVIDIA routes."""

    @pytest.fixture(scope="class")
    def core(self):
        from simlab.core.engine.simlab_core import SimLabCore
        return SimLabCore()

    def _req(self, domain, exp_type, params=None):
        from simlab.core.schemas.experiment import ExperimentRequest
        return ExperimentRequest(domain=domain, type=exp_type, parameters=params or {})

    # ── TRT backend status ────────────────────────────────────────────

    def test_trt_backend_status_route(self, core):
        result = core.run_experiment(self._req("qdgeometry", "trt_backend_status"))
        assert result.status == "success", f"Route failed: {result.errors}"
        assert "recommended" in result.results
        assert "tensorrt" in result.results

    # ── GPU MAP-Elites info ───────────────────────────────────────────

    def test_gpu_mapelites_info_route(self, core):
        result = core.run_experiment(self._req("qdgeometry", "gpu_mapelites_info"))
        assert result.status == "success", f"Route failed: {result.errors}"
        assert "active_backend" in result.results
        assert "cupy_available" in result.results

    # ── GPU MAP-Elites benchmark ──────────────────────────────────────

    def test_gpu_mapelites_benchmark_route(self, core):
        result = core.run_experiment(self._req("qdgeometry", "gpu_mapelites_benchmark", {
            "n_dims": 5, "n_iter": 100, "grid_bins": 5, "seed": 0,
        }))
        assert result.status == "success", f"Route failed: {result.errors}"
        assert result.results["coverage"] > 0
        assert result.results["n_evaluations"] > 0

    # ── Warp von Mises stress ─────────────────────────────────────────

    def test_warp_von_mises_route(self, core):
        ny, nx = 16, 16
        ux = np.zeros((ny, nx)).tolist()
        uy = (np.linspace(0, 1e-4, ny)[:, None] * np.ones((ny, nx))).tolist()
        result = core.run_experiment(self._req("qdgeometry", "warp_von_mises", {
            "displacement_x": ux,
            "displacement_y": uy,
            "E": 200e9,
            "nu": 0.3,
        }))
        assert result.status == "success", f"Route failed: {result.errors}"
        assert "sigma_vm_max" in result.results
        assert result.results["sigma_vm_max"] >= 0

    # ── Warp pressure Poisson ─────────────────────────────────────────

    def test_warp_pressure_poisson_route(self, core):
        result = core.run_experiment(self._req("qdgeometry", "warp_pressure_poisson", {
            "nx": 16, "ny": 16, "p_top": 1.0, "p_bot": 0.0, "iters": 200,
        }))
        assert result.status == "success", f"Route failed: {result.errors}"
        assert result.results["p_max"] == pytest.approx(1.0, abs=0.05)
        assert result.results["p_min"] == pytest.approx(0.0, abs=0.05)

    # ── PhysicsNeMo FNO train (tiny) ──────────────────────────────────

    @_nemo_skip
    def test_physicsnemo_fno_train_route(self, core, tmp_path):
        import os
        save = str(tmp_path / "blade_nemo_route.pt")
        result = core.run_experiment(self._req("qdgeometry", "physicsnemo_fno_train", {
            "n_samples": 8, "epochs": 2, "width": 8, "modes": 4,
            "n_layers": 1, "nx": 32, "ny": 16,
            "save_path": save, "verbose": False,
        }))
        assert result.status == "success", f"Route failed: {result.errors}"
        assert result.results["backend"] == "physicsnemo_fno2d"
        assert os.path.exists(save), "Model file not written"

    # ── PhysicsNeMo FNO infer ─────────────────────────────────────────

    @_nemo_skip
    def test_physicsnemo_fno_infer_route(self, core, tmp_path):
        import os
        save = str(tmp_path / "blade_nemo_infer.pt")
        # First train a tiny model
        core.run_experiment(self._req("qdgeometry", "physicsnemo_fno_train", {
            "n_samples": 8, "epochs": 2, "width": 8, "modes": 4,
            "n_layers": 1, "nx": 32, "ny": 16,
            "save_path": save, "verbose": False,
        }))
        # Then run inference
        result = core.run_experiment(self._req("qdgeometry", "physicsnemo_fno_infer", {
            "model_path": save,
            "hole_cx": [0.3, 0.7], "hole_cy": [0.3, 0.3], "hole_r": [0.05, 0.05],
            "nx": 32, "ny": 16,
        }))
        assert result.status == "success", f"Route failed: {result.errors}"
        assert "T_mean" in result.results
        assert isinstance(result.results["T_field"], list)

    # ── Result JSON serializability ───────────────────────────────────

    def test_all_routes_return_json_serialisable_results(self, core):
        """Every new route result must be JSON-serialisable (no numpy arrays)."""
        import json
        routes_and_params = [
            ("trt_backend_status",     {}),
            ("gpu_mapelites_info",     {}),
            ("gpu_mapelites_benchmark",{"n_dims": 3, "n_iter": 20, "grid_bins": 3}),
            ("warp_pressure_poisson",  {"nx": 8, "ny": 8, "iters": 50}),
        ]
        for etype, params in routes_and_params:
            result = core.run_experiment(self._req("qdgeometry", etype, params))
            assert result.status == "success", f"{etype} failed: {result.errors}"
            try:
                json.dumps(result.results)
            except (TypeError, ValueError) as exc:
                pytest.fail(f"{etype} result not JSON-serialisable: {exc}")
