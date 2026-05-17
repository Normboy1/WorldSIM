# WorldSIM SimLab — Full Stack Reference

> **Version:** 0.2.0 · **Platform:** Linux · **GPU:** NVIDIA GeForce RTX 3060 (12 GiB, sm_86)

---

## What This Is

WorldSIM SimLab is a physics simulation platform that routes experiment requests through a unified dispatcher to domain-specific engines. Every experiment is a `(domain, type)` pair dispatched via `ExperimentDispatcher`. Results are JSON-serializable by contract. The system exposes three surfaces: a Python API, a REST API (FastAPI), and two MCP servers (tool + knowledge).

---

## GPU / NVIDIA Stack

All packages installed and verified on CUDA 13.0 / RTX 3060.

| Package | Version | What it does here |
|---|---|---|
| **PyTorch** | 2.12.0+cu130 | All tensor ops, training loops, model storage |
| **Warp** | 1.13.0 | `@wp.kernel` JIT-compiled CUDA — blade Jacobi solver, diffusion, Allen-Cahn, von Mises stress, pressure Poisson |
| **PhysicsNeMo** | 2.0.0 | `FNO2DEncoder` (SpectralConv2d backbone) — blade thermal field surrogate |
| **CuPy** | 14.0.1 | GPU arrays — MAP-Elites quality grid lives on-device |
| **ONNX Runtime GPU** | 1.26.0 | `TensorrtExecutionProvider` + `CUDAExecutionProvider` for ONNX model inference |
| **TensorRT** | 10.12.0.36 | Python API for `.trt` engine loading (torch-based device memory, no pycuda) |

> **Note:** `torch-tensorrt` is ABI-incompatible with torch 2.12 (requires 2.8.x). Use ORT's `TensorrtExecutionProvider` for TRT-accelerated ONNX inference instead. `FNO2DEncoder` cannot be ONNX-exported (dynamic FFT shapes) — use TorchScript (`.pt`) for BladeFNONeMo deployment.

> **Note:** `nvalchemiops` (NVIDIA ALCHEMI) is not publicly available. `ALCHEMIBackend` falls back to Miedema/equipartition CPU proxies for MD, relaxation, and LJ screening.

> **Known patch:** `torch._inductor.select_algorithm` line 2487 and 3011 — torch 2.12 has a bug where `mm_grouped.py` and `mm_scaled_grouped.py` define `TritonTemplate`/extern kernel names that collide on import. The assert was softened to a skip in the installed wheel.

---

## Architecture

```
ExperimentRequest(domain, type, params)
        │
        ▼
ExperimentValidator      ← checks required params per (domain, type)
        │
        ▼
ExperimentDispatcher     ← _ROUTING_TABLE: 222 routes
        │
   ┌────┴─────────────────────────────────────────────────────┐
   │  domain routers (one per domain)                         │
   └──┬──────────────────────────────────────────────────────┘
      │
      ▼
   Engine modules         ← pure Python / NumPy / SciPy / GPU
      │
      ▼
_sanitize_result()        ← strips numpy arrays → list, NaN/Inf, _-prefixed keys
      │
      ▼
ExperimentResult(status, result, domain, type, experiment_id)
```

**Core files:**

| File | Role |
|---|---|
| `simlab/core/router/dispatcher.py` | `_ROUTING_TABLE`, `ExperimentDispatcher.dispatch()`, `_sanitize_result()` |
| `simlab/core/validation/validator.py` | Per-route required param lists |
| `simlab/core/schemas/experiment.py` | Pydantic v2 `ExperimentRequest` / `ExperimentResult` |
| `simlab/core/engine/simlab_core.py` | `SimLabCore` facade (thin wrapper around dispatcher) |
| `simlab/core/constants/physical.py` | SI constants, unit converters |

---

## Domain Catalog — 222 Routes

### `math` (26 routes)
Symbolic + numerical mathematics via SymPy, NumPy, SciPy.

`solve_equation` · `simplify` · `differentiate` · `integrate` · `expand` · `factor` · `taylor_series` · `critical_points` · `solve_linear` · `matrix_multiply` · `determinant` · `inverse` · `eigenvalues` · `rank` · `svd` · `solve_ode` · `solve_system_ode` · `solve_2nd_order` · `fit_curve` · `optimize` · `maximize` · `minimize_interval` · `monte_carlo` · `descriptive_stats` · `fit_distribution` · `hypothesis_test`

### `physics` (22 routes)
Classical mechanics, E&M, thermodynamics, fluid dynamics.

`projectile_motion` · `projectile_drag` · `spring_mass` · `pendulum` · `circular_motion` · `collision` · `gravity` · `relativistic` · `coulomb_force` · `electric_field` · `magnetic_field` · `capacitance` · `ideal_gas` · `carnot_efficiency` · `entropy` · `heat_transfer` · `van_der_waals` · `bernoulli` · `pipe_flow` · `reynolds_number` · `drag_force` · `terminal_velocity`

### `chemistry` (11 routes)
Reaction kinetics, molecular analysis, RDKit integration.

`reaction_kinetics` · `first_order` · `second_order` · `equilibrium` · `consecutive` · `michaelis_menten` · `molecule_analysis` · `parse_smiles` · `molecular_descriptors` · `molecule_image` · `3d_coordinates`

### `hybrid` (2 routes)
Cross-domain chemistry + physics pipelines.

`reaction_kinetics` · `consecutive_kinetics`

### `atomic` (23 routes)
Hydrogen orbitals, electron configuration, crystal structures, ASE integration.

`create_element` · `electron_config` · `compare_elements` · `compare_isotopes` · `hydrogen_energy_levels` · `hydrogen_energy_diagram` · `orbital_2d` · `orbital_diagram` · `radial_probability` · `radial_comparison` · `radial_normalization` · `shell_diagram` · `effective_nuclear_charge` · `slater_ionization` · `ionization_trend` · `molecule` · `molecule_plot` · `crystal_plot` · `compare_crystals` · `ase_element_data` · `bulk_crystal` · `surface_slab` · `fusion_to_element`

### `nuclear` (14 routes)
Radioactive decay, fission/fusion energetics, nuclear chart.

`decay` · `decay_chain` · `decay_plot` · `decay_chain_plot` · `alpha_decay` · `beta_minus` · `beta_plus` · `alpha_halflife` · `binding_energy_curve` · `analyze_nucleus` · `fission_energy` · `fusion_energy` · `separation_energy` · `nuclear_chart`

### `materials` (70 routes)
The largest domain. Covers the full alloy design pipeline from atomistics to component-level performance.

**Lattice & Structure**
`bcc_lattice` · `fcc_lattice` · `simple_cubic` · `create_lattice` · `common_structure` · `build_structure` · `lattice` · `compare_elements` · `element_properties` · `property_trends`

**Mechanical**
`stress_strain_curve` · `stress_test` · `elastic_deformation` · `elastic_plastic` · `ramberg_osgood` · `youngs_modulus` · `hall_petch` · `flow_stress` · `strain_rate_sensitivity` · `processing_map` · `dynamic_recrystallization` · `crystal_plasticity`

**Diffusion & Phase**
`diffusion_profile` · `arrhenius_diffusivity` · `steady_state_flux` · `grain_boundary_diffusion` · `microstructure_diffusion` · `darken_interdiffusion` · `diffusion_3d` · `phase_diagram` · `jmak_kinetics` · `jmak_temperature_series` · `precipitation_kinetics` · `grain_growth` · `ttt_diagram`

**CALPHAD & Thermodynamics**
`calphad_binary_diagram` · `calphad_phase_equilibrium` · `calphad_phase_scan` · `ellingham_diagram` · `oxide_phase_equilibrium` · `oxidation_states` · `tcp_phase_check`

**Casting & Solidification**
`solidification_time` · `cooling_curve` · `scheil_microsegregation` · `dendrite_arm_spacing` · `niyama_criterion` · `hot_tearing` · `mould_filling` · `die_casting` · `investment_casting` · `closed_die_analysis` · `open_die_pass_schedule` · `process_window_map`

**Fatigue, Creep & Corrosion**
`fatigue` · `degradation_prediction` · `scc_risk_index` · `corrosion_risk` · `oxidation_kinetics` · `pilling_bedworth_ratio`

**DFT / MLIP / Atomistics**
`dft_formation_energy` · `dft_relaxation` · `generate_sqs` · `mlip_energy_forces` · `mlip_relax` · `mlip_batch_screen`

**GPU / Surrogate**
`gpu_diagnostics` · `gpu_backend_status` · `allen_cahn_simulation` · `alloy_property_prediction` · `surrogate_model_plan` · `application_case` · `forging_force` · `phase_field_calphad`

### `nvidia` (13 routes)
Direct NVIDIA backend calls (Warp, PhysicsNeMo, ALCHEMI, CuPy).

| Route | Backend | Notes |
|---|---|---|
| `warp_diffusion` | Warp CUDA | 2D FD diffusion kernel |
| `warp_allen_cahn` | Warp CUDA | Allen-Cahn phase-field grain growth |
| `warp_diffusion_3d` | Warp CUDA | 3D FD diffusion kernel |
| `warp_allen_cahn_field` | Warp CUDA | Full field run + summary |
| `nemo_train_fno` | PhysicsNeMo | FNO2DEncoder composition surrogate |
| `nemo_blade_fno_train` | PhysicsNeMo | Full blade thermal FNO2d pipeline |
| `nemo_sampling_plan` | PhysicsNeMo | Latin-hypercube composition sampling |
| `alchemi_relax` | ALCHEMIBackend | FIRE2+LJ relaxation (proxy if nvalchemiops absent) |
| `alchemi_md` | ALCHEMIBackend | Langevin BAOAB MD (proxy if absent) |
| `alchemi_mlip` | ALCHEMIBackend | LJ / MACE / SevenNet energy+forces |
| `alchemi_batch_screen` | ALCHEMIBackend | LJ cohesive energy batch screening |
| `cupy_diffusion_sweep` | CuPy GPU | (T, t) diffusion parameter sweep on GPU |
| `gpu_diagnostics` | GPUDiagnosticsBackend | VRAM, compute capability, backend flags |

### `qdgeometry` (24 routes)
Quality-Diversity geometry optimization for turbine blade cooling holes.

| Route | Backend | Notes |
|---|---|---|
| `blade_simulation` | Warp CUDA | Jacobi 2D thermal solver, `T_interior_peak` |
| `warp_blade_thermal` | Warp CUDA | Same, via dispatcher |
| `warp_von_mises` | Warp CUDA | Plane-stress von Mises stress field |
| `warp_pressure_poisson` | Warp CUDA | 2D Laplace pressure via Jacobi |
| `fno_generate_data` | Warp CUDA | Generates `X_geom (N,4,ny,nx)` + `Y_tfield (N,1,ny,nx)` dataset |
| `fno_train` | PyTorch | Native FNO surrogate training |
| `physicsnemo_fno_train` | PhysicsNeMo | `BladeFNONeMo` (FNO2DEncoder) training, fp32 |
| `physicsnemo_fno_infer` | PhysicsNeMo | Single-sample inference, optional MC-dropout |
| `physicsnemo_fno_export` | TorchScript | Exports `.pt` for deployment (ONNX unsupported for FNO2DEncoder) |
| `trt_infer` | ORT TRT EP | ONNX inference with TensorrtExecutionProvider |
| `trt_backend_status` | — | Reports TRT/ORT/CuPy availability |
| `calculix_thermal` | CalculiX FEA | High-fidelity FE thermal solve |
| `calculix_validate` | CalculiX FEA | Cross-validates Warp solver vs. FEA |
| `mapelites_geometry_search` | MAP-Elites | Python-direct QD search over blade geometries |
| `mapelites_alloy_search` | MAP-Elites | Python-direct QD search over alloy spaces |
| `gpu_mapelites_info` | CuPy GPU | Archive info (`cupy_gpu` backend) |
| `gpu_mapelites_benchmark` | CuPy GPU | Sphere benchmark on GPU archive |
| `sail_geometry_search` | SAIL | Surrogate-Assisted Illumination |
| `sail_alloy_search` | SAIL | SAIL over alloy composition space |
| `grammar_guided_search` | CFG | Context-free grammar mutation search |
| `fit_neural_implicit` | PyTorch | SDF neural implicit geometry fitting |
| `gradient_descent` | PyTorch | Gradient-based geometry optimization |
| `constraint_loss` | PyTorch | Constraint penalty minimization |
| `geometry_primitive` | NumPy | Parametric shape CSG operations |

### `control` (8 routes)
Control systems via `python-control`.

`transfer_function` · `step_response` · `bode` · `nyquist` · `root_locus` · `pid_controller` · `design_pid` · `closed_loop`

### `data` (9 routes)
Live external data: PubChem molecular DB, arXiv papers.

`pubchem_search` · `pubchem_by_cid` · `pubchem_synonyms` · `pubchem_similar` · `pubchem_substructure` · `pubchem_safety` · `arxiv_search` · `arxiv_paper` · `arxiv_references`

---

## Key Engine Files

### GPU / NVIDIA

| File | Class / Function | GPU backend |
|---|---|---|
| `engines/qdgeometry/warp_blade.py` | `solve_blade_thermal()` | Warp CUDA Jacobi |
| `engines/qdgeometry/physicsnemo_fno.py` | `BladeFNONeMo`, `train_fno_nemo()` | PhysicsNeMo FNO2DEncoder |
| `engines/qdgeometry/trt_inference.py` | `TRTInferenceEngine` | ORT TRT EP / TRT / PyTorch |
| `engines/qdgeometry/mapelites.py` | `MAPElitesArchive`, `GPUMAPElitesArchive` | CuPy GPU quality grid |
| `engines/materials/nvidia_backends.py` | `WarpBackend`, `PhysicsNeMoBackend`, `ALCHEMIBackend`, `CuPyBackend`, `GPUDiagnosticsBackend` | Warp / PhysicsNeMo / CuPy |
| `engines/qdgeometry/fno_surrogate.py` | `generate_training_data()`, `train_fno()`, `export_to_onnx()` | PyTorch (CUDA) |

### Physics & Materials

| File | What it covers |
|---|---|
| `engines/materials/diffusion.py` | Fick's law, Arrhenius, Darken interdiffusion, grain boundary diffusion |
| `engines/materials/stress_strain.py` | Elastic-plastic, Ramberg-Osgood, Hall-Petch, toughness |
| `engines/materials/phase_transformation.py` | JMAK kinetics, TTT diagrams, precipitation |
| `engines/materials/casting.py` | Scheil, Niyama, dendrite arm spacing, die/investment casting |
| `engines/materials/calphad_backend.py` | pycalphad binary diagrams, phase equilibria |
| `engines/materials/mlip_backend.py` | MACE / SevenNet / LJ energy+forces |
| `engines/materials/forging.py` | Forging force, processing maps, dynamic recrystallization |
| `engines/materials/fatigue.py` | S-N curves, Goodman, Paris law |
| `engines/materials/creep_cavity.py` | Norton creep, cavity nucleation |
| `engines/materials/oxidation_stability.py` | Ellingham diagram, Pilling-Bedworth, SCC risk |
| `engines/atomic/hydrogen_orbitals.py` | Analytic hydrogen wavefunctions, normalization |
| `engines/atomic/ase_backend.py` | ASE crystal builder, surface slabs |
| `engines/nuclear/decay.py` | Radioactive decay chains, Bateman equations |

### Other Engines

| File | Domain |
|---|---|
| `engines/chemistry/kinetics.py` | Reaction kinetics ODE integration |
| `engines/chemistry/rdkit_backend.py` | RDKit molecular descriptors, SMILES |
| `engines/math/symbolic.py` | SymPy solve/differentiate/integrate |
| `engines/math/ode_solver.py` | SciPy ODE / BVP solvers |
| `engines/math/optimization.py` | scipy.optimize wrappers |
| `engines/physics/classical.py` | Projectile, pendulum, collision, spring |
| `engines/physics/electromagnetism.py` | Coulomb, Biot-Savart, capacitance |
| `engines/data/pubchem_engine.py` | PubChem REST (cached with httpx) |
| `engines/data/arxiv_engine.py` | arXiv API search + reference parsing |

---

## Blade Thermal Optimization Pipeline

The primary research target — finding optimal turbine blade cooling hole geometries via QD search:

```
generate_training_data()          # Warp Jacobi solver, NACA 4412 geometry
        │  X_geom (N,4,ny,nx)
        │  Y_tfield (N,1,ny,nx)
        ▼
train_fno_nemo()                  # PhysicsNeMo FNO2DEncoder, fp32, CosineAnnealingLR
        │  BladeFNONeMo.pt
        ▼
TRTInferenceEngine.from_torch()   # PyTorch fallback
    or .from_onnx()               # ORT TensorrtExecutionProvider (static ONNX models)
        │  infer(geom) → T_field (ny, nx), T_mean, latency_ms
        ▼
GPUMAPElitesArchive               # CuPy GPU quality grid
    try_insert(solution, T_interior_peak, f1, f2)
        │
        ▼
calculix_validate()               # CalculiX FEA cross-check on elites
```

**Objectives:** `T_mean` (mean interior temperature) and `T_interior_peak` (peak over unconstrained interior nodes). `T.max()` is boundary-pinned at T_hot — not a valid objective.

**Surrogate accuracy note:** `BladeFNONeMo` uses `FNO2DEncoder` with `coord_features=True`. AMP is disabled because FNO2DEncoder's internal `padding=8` creates padded dims `(ny+16, nx+16)` that are rarely powers of two — cuFFT fp16 rejects non-power-of-2 sizes.

---

## API

### FastAPI REST
```
POST /experiment
{
  "domain": "qdgeometry",
  "type":   "physicsnemo_fno_train",
  "params": {}
}
→ ExperimentResult { status, result, domain, type, experiment_id }
```
Start: `worldsim-api` (uvicorn, port 8000)

### MCP Servers
- `worldsim-mcp` — experiment tool server (Claude tool use)
- `worldsim-knowledge-mcp` — knowledge/documentation server

### Python Direct
```python
from simlab.core.engine.simlab_core import SimLabCore
core = SimLabCore()
result = core.run("qdgeometry", "warp_blade_thermal", {
    "hole_centers": [[0.3, 0.4]],
    "hole_radii": [0.05],
    "nx": 64, "ny": 32,
})
```

---

## Tests

```
tests/
  test_simlab_fixes.py      # 46 tests — core correctness, validation, JSON safety
  test_qdgeometry_suite.py  # 54 tests — blade solver, MAP-Elites, FNO surrogate, routes
  test_nvidia_suite.py      # 46 tests — all NVIDIA components
```

**125 tests, 0 failures** as of current build.

Run: `python3 -m pytest -q`

---

## What's Not Implemented

| Item | Status |
|---|---|
| `nvalchemiops` (ALCHEMI) | NVIDIA early access only — all ALCHEMI routes use CPU proxies |
| `torch-tensorrt` | ABI mismatch with torch 2.12 — use ORT TRT EP instead |
| Omniverse / USD | Not installed, no `pxr` |
| ONNX export for BladeFNONeMo | Blocked by dynamic FFT shapes in FNO2DEncoder |
| CalculiX physical equivalence | FD domain (NACA) vs. FE mesh not geometrically matched |
| Biology / Geology / Astronomy / Environmental / Plasma / CFD / Quantum | Engine files exist, dispatcher routes not yet wired |
