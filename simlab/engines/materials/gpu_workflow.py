"""GPU-oriented alloy discovery and degradation workflow.

This module keeps the NVIDIA-specific stack optional. When ALCHEMI,
PhysicsNeMo, Warp, Omniverse, or cuPyNumeric are installed, callers can use the
reported capability map to select accelerated implementations. In a plain CPU
environment, the methods still return deterministic screening results that are
useful for tests, API demos, and proposal prototypes.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from typing import Any

import numpy as np


R_GAS = 8.31446261815324

# ---------------------------------------------------------------------------
# High-fidelity backend configuration loader
# ---------------------------------------------------------------------------

def _load_materials_config() -> dict[str, Any]:
    """Load materials_config.yaml from the same directory as this file."""
    import os
    import yaml  # optional — falls back to empty config if unavailable
    cfg_path = os.path.join(os.path.dirname(__file__), "materials_config.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_materials_config_safe() -> dict[str, Any]:
    try:
        return _load_materials_config()
    except Exception:
        return {}


@dataclass(frozen=True)
class ElementProperty:
    density_g_cm3: float
    elastic_modulus_GPa: float
    melting_K: float
    atomic_radius_pm: float
    oxidation_resistance: float
    hydrogen_sensitivity: float


_ELEMENTS: dict[str, ElementProperty] = {
    "Al": ElementProperty(2.70,  69.0,  933.0, 143.0, 0.82, 0.18),
    "C":  ElementProperty(2.26,  30.0, 3915.0,  70.0, 0.35, 0.10),
    "Co": ElementProperty(8.90, 209.0, 1768.0, 125.0, 0.62, 0.28),
    "Cr": ElementProperty(7.19, 279.0, 2180.0, 128.0, 0.96, 0.22),
    "Cu": ElementProperty(8.96, 117.0, 1358.0, 128.0, 0.42, 0.12),
    "Fe": ElementProperty(7.87, 211.0, 1811.0, 126.0, 0.34, 0.45),
    "Hf": ElementProperty(13.31, 78.0, 2506.0, 156.0, 0.75, 0.55),
    "Mn": ElementProperty(7.21, 198.0, 1519.0, 127.0, 0.38, 0.40),
    "Mo": ElementProperty(10.28, 329.0, 2896.0, 139.0, 0.72, 0.20),
    "Nb": ElementProperty(8.57, 105.0, 2750.0, 146.0, 0.70, 0.48),
    "Ni": ElementProperty(8.91, 200.0, 1728.0, 124.0, 0.67, 0.30),
    "Re": ElementProperty(21.02, 463.0, 3459.0, 137.0, 0.58, 0.15),
    "Ru": ElementProperty(12.37, 447.0, 2607.0, 134.0, 0.60, 0.18),
    "Si": ElementProperty(2.33, 130.0, 1687.0, 117.0, 0.70, 0.20),
    "Ta": ElementProperty(16.65, 186.0, 3290.0, 146.0, 0.78, 0.38),
    "Ti": ElementProperty(4.51,  116.0, 1941.0, 147.0, 0.88, 0.76),
    "V":  ElementProperty(6.11,  128.0, 2183.0, 134.0, 0.55, 0.52),
    "W":  ElementProperty(19.25, 411.0, 3695.0, 139.0, 0.65, 0.18),
    "Zr": ElementProperty(6.52,  68.0,  2128.0, 160.0, 0.72, 0.60),
}


_BACKEND_MODULES: dict[str, tuple[str, ...]] = {
    "alchemi": ("nvalchemiops", "alchemi"),
    "physicsnemo": ("physicsnemo", "nvidia.physicsnemo", "modulus"),
    "warp": ("warp",),
    "omniverse": ("omni",),
    "cupynumeric": ("cupynumeric", "cunumeric"),
    "cupy": ("cupy",),
    "cuda_python": ("cuda",),
}


def _find_module(candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


def _normalize_composition(composition: dict[str, float]) -> dict[str, float]:
    if not composition:
        raise ValueError("composition must contain at least one element.")

    normalized: dict[str, float] = {}
    for symbol, value in composition.items():
        if symbol not in _ELEMENTS:
            supported = ", ".join(sorted(_ELEMENTS))
            raise ValueError(f"Unsupported element {symbol!r}. Supported elements: {supported}")
        amount = float(value)
        if amount < 0:
            raise ValueError("composition values must be non-negative.")
        if amount > 0:
            normalized[symbol] = amount

    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("composition must have a positive total amount.")

    return {symbol: amount / total for symbol, amount in normalized.items()}


def _weighted_average(composition: dict[str, float], attr: str) -> float:
    return float(
        sum(frac * getattr(_ELEMENTS[symbol], attr) for symbol, frac in composition.items())
    )


class MaterialsGPUWorkflowEngine:
    """Focused alloy discovery workflow with optional NVIDIA acceleration hooks.

    High-fidelity backends (CALPHAD, DFT, MLIP) are loaded lazily on first use
    and are configured via ``materials_config.yaml`` in this directory.
    When a backend is unavailable the corresponding method returns a structurally
    identical dict with ``"backend": "proxy"`` and explicit disclaimer text.
    """

    def __init__(self):
        self._cfg = _load_materials_config_safe()
        self._calphad: Any = None
        self._dft: Any = None
        self._mlip: Any = None

    # ------------------------------------------------------------------
    # Backend accessors (lazy init)
    # ------------------------------------------------------------------

    def _get_calphad(self):
        if self._calphad is None:
            from simlab.engines.materials.calphad_backend import CALPHADBackend
            db = (self._cfg.get("calphad") or {}).get("database_path") or None
            self._calphad = CALPHADBackend(database_path=db or None)
        return self._calphad

    def _get_dft(self):
        if self._dft is None:
            from simlab.engines.materials.dft_backend import DFTBackend
            dft_cfg = self._cfg.get("dft") or {}
            calculator = dft_cfg.get("calculator", "emt")
            calc_config = dft_cfg.get(calculator, {})
            self._dft = DFTBackend(calculator=calculator, config=calc_config)
        return self._dft

    def _get_mlip(self):
        if self._mlip is None:
            from simlab.engines.materials.mlip_backend import MLIPBackend
            mlip_cfg = self._cfg.get("mlip") or {}
            self._mlip = MLIPBackend(
                mace_model_path=mlip_cfg.get("mace_model_path") or None,
                sevennet_model_path=mlip_cfg.get("sevennet_model_path") or None,
                device=mlip_cfg.get("device", "cuda"),
            )
        return self._mlip

    def integration_status(self) -> dict[str, Any]:
        """Return detected optional backends and their intended workflow roles."""
        detected = {
            backend: _find_module(modules)
            for backend, modules in _BACKEND_MODULES.items()
        }
        return {
            "available": {name: module is not None for name, module in detected.items()},
            "modules": detected,
            "roles": {
                "alchemi": "Atomistic chemistry/materials simulation, batched MD, geometry relaxation, MLIP operations.",
                "physicsnemo": "Physics-informed ML and neural operator surrogate models.",
                "warp": "Custom GPU kernels for diffusion, phase transformation, and grain growth.",
                "omniverse": "Digital twin visualization and inspection of simulated microstructures.",
                "cupynumeric": "Drop-in accelerated array workloads for CUDA/HPC-scale numerical sweeps.",
                "cupy": "CUDA array fallback for local GPU numerical kernels.",
                "cuda_python": "Low-level CUDA runtime access when custom launch control is needed.",
            },
            "fallback_policy": "Use NumPy CPU screening when optional GPU backends are unavailable.",
        }

    def alloy_property_prediction(
        self,
        composition: dict[str, float],
        temperature_K: float = 298.15,
        heat_treatment: str | None = None,
        composition_basis: str = "atomic_fraction",
    ) -> dict[str, Any]:
        """Estimate first-pass alloy properties from composition.

        This is a screening model, not a calibrated CALPHAD/DFT replacement. It
        produces features and proxy labels that can later seed ALCHEMI atomistic
        runs and PhysicsNeMo surrogate training.
        """
        comp = _normalize_composition(composition)
        temperature_K = float(temperature_K)
        if temperature_K <= 0:
            raise ValueError("temperature_K must be positive.")

        density = _weighted_average(comp, "density_g_cm3")
        modulus = _weighted_average(comp, "elastic_modulus_GPa")
        melting = _weighted_average(comp, "melting_K")
        ox_score = _weighted_average(comp, "oxidation_resistance")

        mean_radius = _weighted_average(comp, "atomic_radius_pm")
        radius_mismatch = math.sqrt(
            sum(
                frac * ((_ELEMENTS[symbol].atomic_radius_pm - mean_radius) / mean_radius) ** 2
                for symbol, frac in comp.items()
            )
        )
        configurational_entropy = -R_GAS * sum(frac * math.log(frac) for frac in comp.values())

        carbon = comp.get("C", 0.0)
        fe = comp.get("Fe", 0.0)
        ni = comp.get("Ni", 0.0)
        cr = comp.get("Cr", 0.0)
        al = comp.get("Al", 0.0)
        ti = comp.get("Ti", 0.0)

        strengthening_MPa = 120.0 + 4200.0 * radius_mismatch
        strengthening_MPa += 900.0 * min(carbon, 0.02) / 0.02 if fe > 0.4 else 0.0
        strengthening_MPa += 280.0 * min(al + ti, 0.18) / 0.18 if ni > 0.35 else 0.0

        service_margin = (melting - temperature_K) / melting

        # Scores are intentionally continuous and NOT clamped to 1.0 at the
        # formula level so that compositions can be meaningfully ranked.
        # Raw values can exceed 1.0 for extreme compositions; callers should
        # treat the absolute number as a relative heuristic, not a probability.
        phase_stability_score_raw = (
            0.45 + 0.035 * configurational_entropy + 0.8 * service_margin
        )
        oxidation_resistance_score_raw = (
            ox_score + 0.35 * min(cr, 0.18) / 0.18 + 0.20 * min(al, 0.08) / 0.08
        )

        # VEC (valence electron count) — empirical FCC/BCC indicator for HEAs
        _VEC: dict[str, float] = {
            "Fe": 8, "Ni": 10, "Cr": 6, "Co": 9, "Al": 3,
            "Ti": 4, "Mo": 6, "W": 6, "Nb": 5, "V": 5,
            "Cu": 11, "C": 4, "Ta": 5, "Re": 7, "Ru": 8,
            "Hf": 4, "Zr": 4, "Mn": 7, "Si": 4,
        }
        vec = sum(comp.get(el, 0.0) * _VEC.get(el, 8.0) for el in comp)

        # Ni/(Al+Ti) ratio — stoichiometry check for gamma-prime stability
        ni_al_ti_ratio = ni / max(al + ti, 1e-9)

        alloy_family = self._classify_alloy(comp)
        result: dict[str, Any] = {
            "workflow": "WorldSim Materials alloy screening",
            "composition_basis": composition_basis,
            "normalized_composition": comp,
            "alloy_family": alloy_family,
            "temperature_K": temperature_K,
            "heat_treatment": heat_treatment,
            "features": {
                "mean_atomic_radius_pm": float(mean_radius),
                "atomic_radius_mismatch": float(radius_mismatch),
                "configurational_entropy_J_molK": float(configurational_entropy),
                "service_temperature_margin": float(service_margin),
                "VEC": float(vec),
                "Ni_over_Al_plus_Ti": float(ni_al_ti_ratio),
            },
            "property_estimates": {
                "density_g_cm3": float(density),
                "elastic_modulus_GPa": float(modulus),
                "solidus_proxy_K": float(melting * (1.0 - 0.20 * radius_mismatch)),
                "yield_strength_proxy_MPa": float(strengthening_MPa),
                "phase_stability_score_raw": float(phase_stability_score_raw),
                "oxidation_resistance_score_raw": float(oxidation_resistance_score_raw),
            },
            "recommended_next_runs": self._recommended_backend_runs(comp, "alloy_property_prediction"),
            "model_limits": [
                "Screening estimates are composition heuristics for proposal prototypes.",
                "Use ALCHEMI/MLIP or DFT-derived data before making material-selection claims.",
            ],
        }

        # ── Layer in CALPHAD phase equilibrium ───────────────────────────────
        calphad = self._get_calphad()
        if calphad.available():
            try:
                calphad_result = calphad.calculate_phase_equilibrium(comp, temperature_K)
                result["phase_stability_calphad"] = {
                    "phase_fractions": calphad_result["phase_fractions"],
                    "tcp_phases_detected": calphad_result["tcp_phases_detected"],
                    "tcp_risk": calphad_result["tcp_risk"],
                    "backend": calphad_result["backend"],
                }
                tcp_result = calphad.detect_tcp_phases(comp, service_T_K=temperature_K)
                result["tcp_phases"] = {
                    "tcp_risk_classification": tcp_result["tcp_risk_classification"],
                    "max_tcp_mole_fraction": tcp_result["max_tcp_mole_fraction"],
                    "recommendation": tcp_result["recommendation"],
                    "backend": tcp_result["backend"],
                }
            except Exception as exc:
                result["calphad_error"] = str(exc)
        else:
            result["phase_stability_calphad"] = {
                "backend": "proxy",
                "note": (
                    "CALPHAD unavailable. Set calphad.database_path in "
                    "materials_config.yaml and install pycalphad."
                ),
            }

        return result

    def degradation_prediction(
        self,
        composition: dict[str, float],
        environment: dict[str, float | str],
        exposure_hours: float,
    ) -> dict[str, Any]:
        """Estimate oxidation/corrosion/hydrogen degradation risk."""
        comp = _normalize_composition(composition)
        exposure_hours = float(exposure_hours)
        if exposure_hours < 0:
            raise ValueError("exposure_hours must be non-negative.")

        temperature_K = float(environment.get("temperature_K", 873.15))
        if temperature_K <= 0:
            raise ValueError("environment.temperature_K must be positive.")
        oxygen_atm = max(float(environment.get("oxygen_partial_pressure_atm", 0.21)), 0.0)
        chloride = max(float(environment.get("chloride_mol_L", 0.0)), 0.0)
        ph = float(environment.get("pH", environment.get("ph", 7.0)))
        hydrogen_atm = max(float(environment.get("hydrogen_partial_pressure_atm", 0.0)), 0.0)

        cr = comp.get("Cr", 0.0)
        al = comp.get("Al", 0.0)
        ti = comp.get("Ti", 0.0)
        ni = comp.get("Ni", 0.0)

        activation_J_mol = 82_000.0
        base_kp_m2_s = 2.5e-15 * math.exp(
            -activation_J_mol / R_GAS * (1.0 / temperature_K - 1.0 / 873.15)
        )
        protective_factor = 1.0 / (1.0 + 8.0 * cr + 6.0 * al + 1.5 * ni)
        environment_factor = (1.0 + 2.0 * oxygen_atm) * (1.0 + 3.5 * chloride)
        if ph < 4.0:
            environment_factor *= 1.0 + (4.0 - ph) * 0.35
        kp_m2_s = base_kp_m2_s * protective_factor * environment_factor

        seconds = exposure_hours * 3600.0
        oxide_thickness_um = math.sqrt(max(kp_m2_s * seconds, 0.0)) * 1e6
        mass_gain_mg_cm2 = 0.55 * oxide_thickness_um
        time_hours = np.linspace(0.0, exposure_hours, 80)
        oxide_curve_um = np.sqrt(np.maximum(kp_m2_s * time_hours * 3600.0, 0.0)) * 1e6

        h_sensitivity = _weighted_average(comp, "hydrogen_sensitivity")
        hydrogen_embrittlement_risk = max(
            0.0,
            min(1.0, h_sensitivity * (1.0 + 8.0 * hydrogen_atm) + 0.25 * ti + 0.15 * chloride),
        )
        corrosion_risk = max(
            0.0,
            min(1.0, 0.18 + 0.28 * chloride + 0.18 * max(0.0, 4.0 - ph) - 0.55 * cr - 0.25 * al),
        )

        return {
            "workflow": "WorldSim Materials degradation screening",
            "normalized_composition": comp,
            "environment": {
                "temperature_K": temperature_K,
                "oxygen_partial_pressure_atm": oxygen_atm,
                "chloride_mol_L": chloride,
                "pH": ph,
                "hydrogen_partial_pressure_atm": hydrogen_atm,
            },
            "exposure_hours": exposure_hours,
            "degradation_estimates": {
                "parabolic_rate_constant_m2_s": float(kp_m2_s),
                "oxide_thickness_um": float(oxide_thickness_um),
                "mass_gain_mg_cm2_proxy": float(mass_gain_mg_cm2),
                "corrosion_risk_score": float(corrosion_risk),
                "hydrogen_embrittlement_risk_score": float(hydrogen_embrittlement_risk),
            },
            "oxidation_curve": {
                "time_hours": time_hours.tolist(),
                "oxide_thickness_um": oxide_curve_um.tolist(),
            },
            "recommended_next_runs": self._recommended_backend_runs(comp, "degradation_prediction"),
            "model_limits": [
                "Uses a simple parabolic oxidation proxy and heuristic corrosion factors.",
                "Calibrate against experiments or atomistic/phase-field data before publication claims.",
            ],
        }

    def microstructure_diffusion(
        self,
        grid_shape: list[int] | tuple[int, int] = (48, 48),
        diffusivity_m2_s: float = 1e-15,
        time_s: float = 10.0,
        steps: int = 80,
        initial_left_concentration: float = 1.0,
        initial_right_concentration: float = 0.0,
    ) -> dict[str, Any]:
        """Run a 2D finite-difference diffusion simulation via WarpBackend.

        Delegates all computation to ``WarpBackend.run_diffusion_field`` so the
        Warp GPU kernel is used when CUDA is available, with automatic NumPy
        fallback otherwise.
        """
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must contain two dimensions.")
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        if nx < 3 or ny < 3:
            raise ValueError("grid_shape dimensions must be at least 3.")
        if nx * ny > 250_000:
            raise ValueError("grid_shape is too large for the CPU fallback demo.")
        if diffusivity_m2_s <= 0 or time_s < 0 or steps < 1:
            raise ValueError("diffusivity_m2_s must be positive, time_s non-negative, and steps >= 1.")

        dx = 1.0
        dt = float(time_s) / int(steps)
        alpha = min(float(diffusivity_m2_s) * dt / (dx * dx), 0.20)

        from simlab.engines.materials.nvidia_backends import WarpBackend
        warp = WarpBackend()
        raw = warp.run_diffusion_field(
            grid_shape=(nx, ny),
            alpha=alpha,
            steps=steps,
            C_left=float(initial_left_concentration),
            C_right=float(initial_right_concentration),
        )
        field = np.array(raw["field"])

        return {
            "workflow": "WorldSim Materials microstructure diffusion",
            "backend": raw["backend"],
            "grid_shape": [nx, ny],
            "steps": int(steps),
            "time_s": float(time_s),
            "stability_alpha_used": float(alpha),
            "field_summary": {
                "min": float(np.min(field)),
                "max": float(np.max(field)),
                "mean": float(np.mean(field)),
                "std": float(np.std(field)),
                "centerline": field[:, ny // 2].tolist(),
            },
            "field": field.tolist(),
        }

    def allen_cahn_simulation(
        self,
        grid_shape: list[int] | tuple[int, int] = (64, 64),
        M: float = 1.0,
        kappa: float = 1.0,
        dt: float = 0.01,
        steps: int = 500,
        phi_seed: int = 42,
    ) -> dict[str, Any]:
        """Run Allen-Cahn phase-field grain coarsening via WarpBackend.

        Evolves a random initial phase field (phi ≈ 0.5 ± 0.05) under the
        Allen-Cahn equation with a double-well potential driving grain growth.
        Uses the Warp GPU kernel when CUDA is available.
        """
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must contain two dimensions.")
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        if nx < 3 or ny < 3:
            raise ValueError("grid_shape dimensions must be at least 3.")
        if nx * ny > 250_000:
            raise ValueError("grid_shape is too large for the demo.")
        if float(M) <= 0 or float(kappa) <= 0:
            raise ValueError("M and kappa must be positive.")
        if float(dt) <= 0 or int(steps) < 1:
            raise ValueError("dt must be positive and steps >= 1.")

        from simlab.engines.materials.nvidia_backends import WarpBackend
        warp = WarpBackend()
        raw = warp.run_allen_cahn_field(
            grid_shape=(nx, ny),
            M=float(M),
            kappa=float(kappa),
            dt=float(dt),
            steps=int(steps),
            phi_seed=int(phi_seed),
        )
        field = np.array(raw["field"])

        return {
            "workflow": "WorldSim Materials Allen-Cahn grain coarsening",
            "backend": raw["backend"],
            "grid_shape": [nx, ny],
            "steps": int(steps),
            "M": float(M),
            "kappa": float(kappa),
            "dt": float(dt),
            "field_summary": {
                "min": float(np.min(field)),
                "max": float(np.max(field)),
                "mean": float(np.mean(field)),
                "std": float(np.std(field)),
                "grain_fraction_high": float(np.mean(field > 0.7)),
                "grain_fraction_low": float(np.mean(field < 0.3)),
                "centerline": field[:, ny // 2].tolist(),
            },
            "field": field.tolist(),
            "model_limits": [
                "Single-order-parameter Allen-Cahn — does not resolve individual grain orientations.",
                "Use a multi-phase-field model for grain boundary engineering predictions.",
            ],
        }

    def diffusion_3d(
        self,
        grid_shape: list[int] | tuple[int, int, int] = (16, 16, 16),
        alpha: float = 0.10,
        steps: int = 50,
        C_left: float = 1.0,
        C_right: float = 0.0,
    ) -> dict[str, Any]:
        """Run a 3D finite-difference diffusion simulation via WarpBackend.

        Uses a 6-point stencil Warp kernel on GPU; falls back to NumPy with
        stability-clamped alpha (≤ 1/6) on CPU.
        """
        if len(grid_shape) != 3:
            raise ValueError("grid_shape must contain three dimensions.")
        nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        if nx < 3 or ny < 3 or nz < 3:
            raise ValueError("grid_shape dimensions must be at least 3.")
        if nx * ny * nz > 500_000:
            raise ValueError("grid_shape is too large for the CPU fallback demo.")
        if float(alpha) <= 0 or int(steps) < 1:
            raise ValueError("alpha must be positive and steps >= 1.")

        from simlab.engines.materials.nvidia_backends import WarpBackend
        warp = WarpBackend()
        return warp.run_diffusion_3d(
            grid_shape=(nx, ny, nz),
            alpha=float(alpha),
            steps=int(steps),
            C_left=float(C_left),
            C_right=float(C_right),
        )

    def gpu_diagnostics(self) -> dict[str, Any]:
        """Return GPU device info, VRAM usage, and NVIDIA backend availability."""
        from simlab.engines.materials.nvidia_backends import GPUDiagnosticsBackend
        return GPUDiagnosticsBackend().gpu_status()

    def surrogate_model_plan(
        self,
        dataset_size: int,
        target_properties: list[str],
        operator: str = "fourier_neural_operator",
    ) -> dict[str, Any]:
        """Create a PhysicsNeMo-oriented surrogate training plan."""
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if not target_properties:
            raise ValueError("target_properties must contain at least one target.")

        train = max(1, int(dataset_size * 0.70))
        validation = max(1, int(dataset_size * 0.15))
        test = max(1, dataset_size - train - validation)

        return {
            "workflow": "WorldSim Materials surrogate training plan",
            "backend": "physicsnemo" if self.integration_status()["available"]["physicsnemo"] else "planned_physicsnemo",
            "operator": operator,
            "dataset_split": {
                "train": train,
                "validation": validation,
                "test": test,
            },
            "inputs": [
                "composition vector",
                "temperature and chemical environment",
                "heat-treatment/process descriptors",
                "microstructure field or latent embedding",
            ],
            "targets": target_properties,
            "loss_terms": [
                "supervised property loss",
                "physics residual for diffusion or reaction kinetics",
                "monotonicity/positivity constraints for degradation rates",
                "uncertainty calibration for active learning",
            ],
            "active_learning_loop": [
                "sample candidate alloy compositions",
                "run ALCHEMI atomistic simulations for high-value candidates",
                "run Warp microstructure kernels for process evolution",
                "train/update PhysicsNeMo surrogate",
                "visualize candidates and failure modes in Omniverse",
            ],
        }

    # ------------------------------------------------------------------
    # High-fidelity experiment methods (CALPHAD / DFT / MLIP)
    # ------------------------------------------------------------------

    def calphad_phase_equilibrium(
        self,
        composition: dict[str, float],
        temperature_K: float = 1173.15,
        pressure_Pa: float = 101_325.0,
        output_phases: list[str] | None = None,
    ) -> dict[str, Any]:
        """Calculate equilibrium phase fractions using pycalphad.

        Returns phase names, mole fractions, and TCP risk assessment.
        Falls back to a composition heuristic proxy when pycalphad or a
        TDB database is unavailable.
        """
        comp = _normalize_composition(composition)
        calphad = self._get_calphad()
        return calphad.calculate_phase_equilibrium(comp, temperature_K, pressure_Pa, output_phases)

    def calphad_phase_scan(
        self,
        composition: dict[str, float],
        T_low_K: float = 600.0,
        T_high_K: float = 1500.0,
        n_points: int = 30,
    ) -> dict[str, Any]:
        """Scan phase fractions from T_low_K to T_high_K.

        Useful for locating solvus temperatures and tracking γ/γ' balance.
        """
        comp = _normalize_composition(composition)
        calphad = self._get_calphad()
        return calphad.phase_fraction_vs_temperature(comp, T_low_K, T_high_K, n_points)

    def calphad_binary_diagram(
        self,
        element_a: str,
        element_b: str,
        T_low_K: float = 600.0,
        T_high_K: float = 1800.0,
        n_compositions: int = 40,
        n_temperatures: int = 40,
    ) -> dict[str, Any]:
        """Calculate a binary A-B phase diagram grid."""
        calphad = self._get_calphad()
        return calphad.calculate_binary_phase_diagram(
            element_a, element_b, T_low_K, T_high_K, n_compositions, n_temperatures
        )

    def tcp_phase_check(
        self,
        composition: dict[str, float],
        temperatures_K: list[float] | None = None,
        service_T_K: float = 1173.15,
    ) -> dict[str, Any]:
        """Check for TCP phase (σ, μ, Laves) stability.

        Returns risk classification and recommendations. High-Mo/W alloys
        (like Co41Cr16Ni16Mo12Al8W7) require this check before further
        development.  Falls back to a Mo+W fraction heuristic when pycalphad
        is unavailable.
        """
        comp = _normalize_composition(composition)
        calphad = self._get_calphad()
        return calphad.detect_tcp_phases(comp, temperatures_K, service_T_K)

    def oxide_phase_equilibrium(
        self,
        composition: dict[str, float],
        temperature_K: float = 1173.15,
        oxygen_partial_pressure_atm: float = 0.21,
    ) -> dict[str, Any]:
        """Check oxide phase stability and MoO₃ volatility.

        Critical for high-Mo alloys at service temperatures above 700°C.
        """
        comp = _normalize_composition(composition)
        calphad = self._get_calphad()
        return calphad.oxide_phase_equilibrium(comp, temperature_K, oxygen_partial_pressure_atm)

    def dft_formation_energy(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        relax_first: bool = True,
    ) -> dict[str, Any]:
        """Calculate DFT formation energy via ASE calculator.

        When ``calculator = "emt"`` (default), the result is not DFT quality
        but exercises the full workflow pipeline.  For real DFT set
        ``calculator`` to ``"vasp"`` or ``"espresso"`` in materials_config.yaml.
        """
        dft = self._get_dft()
        return dft.calculate_formation_energy(
            species, positions_angstrom, lattice_vectors_angstrom, relax_first
        )

    def dft_relaxation(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        fmax_eV_A: float = 0.05,
        max_steps: int = 300,
        optimizer: str = "FIRE",
    ) -> dict[str, Any]:
        """Geometry-relax a structure using an ASE calculator."""
        dft = self._get_dft()
        return dft.relax_structure(
            species, positions_angstrom, lattice_vectors_angstrom,
            fmax_eV_A, max_steps, optimizer,
        )

    def mlip_energy_forces(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        potential: str | None = None,
    ) -> dict[str, Any]:
        """Single-point MLIP energy + forces (MACE → SevenNet → EMT → LJ)."""
        mlip = self._get_mlip()
        return mlip.energy_forces(
            species, positions_angstrom, lattice_vectors_angstrom, potential
        )

    def mlip_batch_screen(
        self,
        compositions: list[dict[str, float]],
        lattice_type: str = "fcc",
        a_angstrom: float = 3.6,
        n_atoms: int = 32,
        seed: int = 42,
        potential: str | None = None,
    ) -> dict[str, Any]:
        """Screen a list of alloy compositions using MLIP single-point energies.

        Returns candidates ranked by energy per atom (most negative first).
        Replaces the LJ batch screen with MACE/SevenNet-quality energies when
        models are available.
        """
        mlip = self._get_mlip()
        return mlip.batch_screen(
            compositions, lattice_type, a_angstrom, n_atoms, seed, potential
        )

    def mlip_relax(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        fmax_eV_A: float = 0.05,
        max_steps: int = 300,
        optimizer: str = "FIRE",
        potential: str | None = None,
    ) -> dict[str, Any]:
        """Geometry-relax a structure using the best available MLIP."""
        mlip = self._get_mlip()
        return mlip.relax_structure(
            species, positions_angstrom, lattice_vectors_angstrom,
            fmax_eV_A, max_steps, optimizer, potential,
        )

    def generate_sqs(
        self,
        composition: dict[str, float],
        n_atoms: int = 32,
        lattice_type: str = "fcc",
        a_angstrom: float = 3.6,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Generate a Special Quasirandom Structure for DFT or MLIP input."""
        dft = self._get_dft()
        return dft.generate_sqs(
            composition=_normalize_composition(composition),
            n_atoms=n_atoms,
            lattice_type=lattice_type,
            a_angstrom=a_angstrom,
            seed=seed,
        )

    def _classify_alloy(self, composition: dict[str, float]) -> str:
        base = max(composition, key=composition.get)
        if base == "Fe":
            return "steel_or_iron_alloy"
        if base == "Ni":
            return "nickel_superalloy_candidate"
        if base == "Al":
            return "aluminum_alloy"
        if base == "Ti":
            return "titanium_alloy"
        return f"{base.lower()}_rich_alloy"

    def _recommended_backend_runs(
        self, composition: dict[str, float], workflow_type: str
    ) -> list[dict[str, str]]:
        base_runs = [
            {
                "backend": "ALCHEMI",
                "purpose": "Relax representative atomic cells and run batched molecular dynamics.",
            },
            {
                "backend": "Warp",
                "purpose": "Run diffusion, precipitation, phase-transformation, or grain-growth kernels.",
            },
            {
                "backend": "PhysicsNeMo",
                "purpose": "Train neural operator/PINN surrogate from simulated composition-process-property data.",
            },
            {
                "backend": "Omniverse",
                "purpose": "Inspect microstructure/degradation fields as a digital-twin scene.",
            },
            {
                "backend": "CUDA/HPC SDK/cuPyNumeric",
                "purpose": "Accelerate parameter sweeps and array workloads.",
            },
        ]
        if workflow_type == "degradation_prediction" and composition.get("Cr", 0.0) < 0.10:
            base_runs.insert(
                1,
                {
                    "backend": "ALCHEMI",
                    "purpose": "Prioritize surface oxidation and adsorbate/reaction-path sampling.",
                },
            )
        return base_runs
