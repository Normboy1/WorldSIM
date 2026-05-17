# WorldSIM

**A unified, GPU-accelerated scientific simulation platform — and a research testbed for quality-diversity optimization of turbine-blade cooling geometries.**

> **Version** 0.2.0 · **Python** ≥ 3.11 · **License** MIT · **Author** MaxOSL AI Research
> **Reference platform** Linux · NVIDIA GeForce RTX 3060 (12 GiB, sm_86) · CUDA 13.0
> **Status** 223 simulation routes · 125/125 tests passing · GPU stack verified

---

## 1. What WorldSIM Is

WorldSIM is two things in one repository:

1. **A breadth-first simulation platform.** Every experiment is a `(domain, type)`
   pair — e.g. `("physics", "projectile_motion")` or `("materials", "calphad_phase_diagram")`.
   A single dispatcher routes the request to a domain engine, validates parameters,
   runs the science, sanitizes the output to JSON, and returns a typed result.
   It currently exposes **223 routes across 11 domains**, from symbolic algebra
   to radioactive decay chains to CALPHAD phase equilibria.

2. **A depth-first research vehicle.** The flagship workload is **differentiable
   parametric geometry generation via quality-diversity (QD) search** — discovering
   turbine-blade cooling-hole layouts that minimize peak metal temperature. This
   pipeline chains a Warp CUDA thermal solver, a PhysicsNeMo Fourier Neural
   Operator (FNO) surrogate, a CuPy-resident MAP-Elites archive, and CalculiX FEA
   cross-validation. It is the basis of the NVIDIA research proposal in
   [`docs/PROPOSAL_NVIDIA_4PAGE.pdf`](docs/PROPOSAL_NVIDIA_4PAGE.pdf).

The same code is reachable three ways: a **Python API**, a **FastAPI REST server**,
and two **MCP servers** for direct use by AI agents (Claude tool-use).

### Why it is built this way

Scientific tooling is usually fragmented — one library per field, each with its
own data model. WorldSIM imposes a single contract: *request in, typed result out,
JSON-serializable by construction*. That uniformity is what lets an AI agent, a
REST client, and a QD optimizer all drive the same engines without glue code, and
it is what makes the surrogate-training loop (Section 6) possible to automate.

---

## 2. Architecture

```
                ExperimentRequest(domain, type, parameters, outputs)
                                  │
                                  ▼
                        ┌──────────────────┐
                        │   SimLabCore     │   orchestrator / facade
                        └────────┬─────────┘
                                 │
                  ┌──────────────┴───────────────┐
                  ▼                              ▼
        ┌───────────────────┐         ┌─────────────────────┐
        │  SimLabValidator  │         │ ExperimentDispatcher│
        │  • domain check   │         │ • _ROUTING_TABLE    │
        │  • required params│  ───▶   │   (223 entries)     │
        │  • physical bounds│         │ • lazy engine import│
        │  • safety blocks  │         │ • param remapping   │
        └───────────────────┘         └──────────┬──────────┘
                                                 │
                              ┌──────────────────┼───────────────────┐
                              ▼                  ▼                   ▼
                       Domain engines      PlotEngine         _sanitize_result()
                   (math/physics/...)   (optional plots)   numpy→list, NaN→None,
                              │                                strip _-keys
                              ▼
                ExperimentResult(status, results, plots, errors, metadata)
```

**Design invariants**

- **Every result is JSON-serializable.** `_sanitize_result()` recursively converts
  numpy scalars/arrays to Python types, maps `NaN`/`Inf` to `None`, and strips
  in-process handles (keys prefixed with `_`, e.g. a live model object).
- **Engines are lazily imported.** Importing the dispatcher costs nothing; a heavy
  backend (PyTorch, Warp, pycalphad) loads only when its first route is called.
- **Failure is data, not an exception.** Any engine error becomes
  `ExperimentResult(status="error", errors=[...])` — callers never see a traceback.
- **Validation is separate from execution.** `SimLabValidator` runs pre-flight
  checks (required parameters, speed-of-light limits, absolute-zero temperatures,
  chemical-safety blocklist) before anything dispatches.

**Core files**

| File | Role |
|---|---|
| `simlab/core/engine/simlab_core.py` | `SimLabCore` — public facade over validate + dispatch |
| `simlab/core/router/dispatcher.py` | `_ROUTING_TABLE`, `ExperimentDispatcher.dispatch()`, `_sanitize_result()` |
| `simlab/core/validation/validator.py` | Per-route required-parameter lists, safety checks |
| `simlab/core/schemas/experiment.py` | Pydantic v2 `ExperimentRequest` / `ExperimentResult` |
| `simlab/core/constants/physical.py` | SI physical constants and unit converters |

---

## 3. The NVIDIA GPU Backbone

GPU acceleration is provided entirely by the NVIDIA software stack. All components
below are **installed and verified on the reference RTX 3060**.

| Component | Version | Role in WorldSIM |
|---|---|---|
| **PyTorch** | 2.12.0+cu130 | Tensor ops, surrogate training loops, model storage |
| **Warp** | 1.13.0 | `@wp.kernel` JIT-compiled CUDA — thermal Jacobi solver, 2D/3D diffusion, Allen-Cahn phase field, von Mises stress, pressure Poisson |
| **PhysicsNeMo** | 2.0.0 | `FNO2DEncoder` spectral backbone — blade thermal-field surrogate (`BladeFNONeMo`) |
| **CuPy** | 14.0.1 | GPU-resident arrays — the MAP-Elites quality grid lives on-device |
| **ONNX Runtime GPU** | 1.26.0 | `TensorrtExecutionProvider` + `CUDAExecutionProvider` for ONNX inference |
| **TensorRT** | 10.12.0 | `.trt` engine loading via the Python API (torch-managed device memory) |

**Verified live (latest run):**

```
PyTorch 2.12.0+cu130   cuda_available=True   device=NVIDIA GeForce RTX 3060
CuPy 14.0.1            GPU reduction OK
Warp 1.13.0            cuda:0  (12 GiB, sm_86, mempool enabled)
PhysicsNeMo 2.0.0      BladeFNONeMo forward (2,4,64,128)→(2,1,64,128), 2.36M params
Dispatcher routes      warp_diffusion / warp_allen_cahn / warp_blade_thermal /
                       warp_pressure_poisson / warp_von_mises / gpu_mapelites — all OK
```

**Honest limitations of the GPU stack**

- `nvalchemiops` (NVIDIA ALCHEMI) is early-access only and not publicly installable.
  `ALCHEMIBackend` therefore falls back to Miedema / equipartition / Lennard-Jones
  **CPU proxies** for MD, relaxation, and screening — clearly labelled in every result.
- `torch-tensorrt` is ABI-incompatible with PyTorch 2.12 (it targets 2.8.x). TRT
  acceleration is reached through ORT's `TensorrtExecutionProvider` instead.
- `FNO2DEncoder` cannot be exported to ONNX (dynamic FFT shapes); `BladeFNONeMo`
  is deployed via TorchScript (`.pt`).

---

## 4. Domain Catalog — 223 Routes

| Domain | Routes | Coverage |
|---|--:|---|
| **materials** | 71 | Lattices, stress-strain, diffusion, phase transformations, CALPHAD, casting/forging, fatigue/creep/corrosion, DFT/MLIP, GPU microstructure |
| **math** | 26 | Symbolic algebra, calculus, linear algebra, ODEs, optimization, statistics (SymPy / NumPy / SciPy) |
| **qdgeometry** | 24 | Quality-diversity geometry search, blade thermal solvers, FNO surrogates, MAP-Elites, SAIL, TRT inference |
| **atomic** | 23 | Electron configurations, hydrogen-like orbitals, crystal structures, ASE backend |
| **physics** | 22 | Classical mechanics, electromagnetism, thermodynamics, fluid dynamics |
| **nuclear** | 14 | Binding energy, fission/fusion energetics, radioactive decay chains (Bateman) |
| **nvidia** | 13 | Direct Warp / PhysicsNeMo / ALCHEMI / CuPy backend calls |
| **chemistry** | 11 | Reaction kinetics, equilibrium, RDKit molecular analysis |
| **data** | 9 | Live arXiv and PubChem lookups with TTL HTTP cache |
| **control** | 8 | Transfer functions, PID design, Bode / Nyquist / root-locus |
| **hybrid** | 2 | Cross-domain chemistry + physics pipelines |

The full enumerated route list lives in [`docs/STACK.md`](docs/STACK.md) and the
annotated catalog in [`docs/03-experiment-catalog.md`](docs/03-experiment-catalog.md).

**Materials is the deep domain.** It spans the entire alloy-design pipeline:

- *Structure* — FCC/BCC/SC lattices, pymatgen structure building, common-structure analysis
- *Mechanical* — elastic-plastic & Ramberg-Osgood curves, Hall-Petch, flow stress, processing maps
- *Diffusion & phase* — Fick / Arrhenius / Darken, JMAK kinetics, TTT diagrams, grain growth
- *Thermodynamics* — CALPHAD phase equilibria & binary diagrams (pycalphad), Ellingham diagrams
- *Casting & forging* — Scheil microsegregation, Niyama criterion, dendrite arm spacing, forging force
- *Degradation* — fatigue S-N / Paris law, Norton creep, oxidation kinetics, SCC and corrosion risk
- *Atomistics* — DFT formation energy, SQS generation, MLIP energy/forces
- *GPU microstructure* — Warp Allen-Cahn grain coarsening, 3D diffusion, surrogate planning

---

## 5. Quick Start

### Install

```bash
# Core platform (math, physics, the dispatcher, REST + MCP servers)
pip install -e .

# Optional domain extras
pip install -e ".[chemistry,atomic,materials,control]"

# Full NVIDIA GPU stack (Warp, PhysicsNeMo, CuPy; ALCHEMI where available)
pip install -e ".[nvidia]"

# High-fidelity materials backends (CALPHAD / DFT / MLIP)
pip install -e ".[high_fidelity]"
```

### Run a simulation — Python API

```python
from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest

core = SimLabCore()

req = ExperimentRequest(
    domain="physics",
    type="projectile_motion",
    parameters={"v0": 25.0, "angle_deg": 45.0},
    outputs=["plot"],
)
result = core.run_experiment(req)
print(result.status)          # "success"
print(result.results["range"])# computed range in metres
```

### Run the servers

```bash
worldsim-api              # FastAPI REST server (uvicorn, port 8000)
worldsim-mcp              # MCP experiment-tool server (Claude tool use)
worldsim-knowledge-mcp    # MCP knowledge / documentation server
```

### Run the test suite

```bash
python3 -m pytest -q      # 125 passed in ~7s
```

---

## 6. Flagship Pipeline — Blade Thermal QD Optimization

The research core of WorldSIM. The goal: **discover cooling-hole geometries for a
turbine blade cross-section that minimize peak metal temperature**, while mapping
the diversity of viable designs rather than collapsing to a single optimum.

```
 ┌─────────────────────────┐
 │ generate_training_data  │  Warp CUDA Jacobi thermal solver on a NACA-4412
 │                         │  cross-section → X_geom (N,4,ny,nx), Y_tfield (N,1,ny,nx)
 └────────────┬────────────┘
              ▼
 ┌─────────────────────────┐
 │ train_fno_nemo          │  PhysicsNeMo FNO2DEncoder (BladeFNONeMo), fp32,
 │                         │  CosineAnnealingLR — learns geometry → temperature field
 └────────────┬────────────┘
              ▼
 ┌─────────────────────────┐
 │ TRTInferenceEngine      │  Fast surrogate inference (PyTorch / ORT-TRT EP)
 │                         │  infer(geom) → T_field, T_mean, latency_ms
 └────────────┬────────────┘
              ▼
 ┌─────────────────────────┐
 │ GPUMAPElitesArchive     │  CuPy-resident quality-diversity archive — illuminates
 │ (MAP-Elites)            │  the design space by (behavior descriptor) cells
 └────────────┬────────────┘
              ▼
 ┌─────────────────────────┐
 │ calculix_validate       │  CalculiX FEA cross-check of elite designs vs. the
 │                         │  Warp finite-difference solver
 └─────────────────────────┘
```

**Objectives.** `T_mean` (mean interior temperature) and `T_interior_peak` (peak
over *unconstrained* interior nodes). Note that `T.max()` is invalid as an
objective — it is pinned at the hot-gas boundary condition.

**Quality-diversity, not just optimization.** MAP-Elites keeps the best design per
cell of a behavior-descriptor grid, so the output is an *atlas* of good geometries
across the trade-off space — directly useful for design exploration and for
generating diverse training data for the surrogate.

**Validation against published baselines.** The CuPy MAP-Elites archive was run on
the standard 20-D sphere benchmark (Mouret & Clune 2015, arXiv:1504.04909):

| Metric | Published baseline | WorldSIM measured |
|---|---|---|
| Coverage | ≥ 0.85 | **0.98** |
| Best quality | ≥ 0.90 | **0.943** |
| QD-score | ≥ 0.70 | **0.884** |

All three metrics exceed the baseline.

The pipeline also offers SAIL (Surrogate-Assisted Illumination), grammar-guided
search, JAX-autodiff gradient descent over geometry parameters, and neural-implicit
(SDF) geometry fitting — see the `qdgeometry` engines.

---

## 7. Interfaces

### Python API

```python
from simlab.core.engine.simlab_core import SimLabCore
core = SimLabCore()
result = core.run_experiment(req)        # validate → dispatch → typed result
```

### REST API (FastAPI)

```
POST /experiment        generic (domain, type, parameters)
POST /math /physics /chemistry /materials /atomic /nuclear /data /control
POST /visualize /report /safety
GET  /health /constants /experiments
```

CORS defaults to local-development origins; set `SIMLAB_CORS_ORIGINS` to a
comma-separated allowlist for deployment. External data lookups use a TTL cache
under `proof/cache/http` (`SIMLAB_DATA_CACHE_DIR` / `SIMLAB_DATA_CACHE_TTL_S`);
stale cache is served if a live request fails.

### MCP Servers (AI-agent tool use)

`worldsim-mcp` exposes the platform as Claude-callable tools:
`run_experiment`, `solve_math`, `simulate_physics`, `simulate_chemistry`,
`simulate_materials`, `simulate_atomic`, `simulate_nuclear`, `query_data`,
`analyze_control`, `generate_visualization`, `export_report`, `check_safety`.
`worldsim-knowledge-mcp` serves the documentation and proof corpus.

---

## 8. Testing & Validation

```
tests/test_simlab_fixes.py      core correctness, validation, JSON-safety
tests/test_qdgeometry_suite.py  blade solver, MAP-Elites, FNO surrogate, routes
tests/test_nvidia_suite.py      Warp / PhysicsNeMo / CuPy / TRT components
```

**125 tests, 0 failures** (`python3 -m pytest -q`, ~7 s on the reference machine).
The remaining 20 warnings are non-fatal PyTorch `torch.jit` deprecations.

Validation is layered: schema validation (Pydantic v2), per-route required-parameter
checks, physical-bounds checks (sub-light speeds, above absolute zero, positive mass),
a chemical-safety blocklist, and the MAP-Elites benchmark above.

---

## 9. Repository Layout

```
simlab/
├── core/
│   ├── engine/          SimLabCore orchestrator
│   ├── router/          ExperimentDispatcher + 223-route table
│   ├── validation/      SimLabValidator (safety + params)
│   ├── schemas/         Pydantic request/result models
│   └── constants/       SI constants, unit conversion
├── engines/             109 engine modules across all domains
│   ├── math/ physics/ chemistry/ materials/ atomic/ nuclear/ data/
│   ├── qdgeometry/      blade solvers, FNO, MAP-Elites, SAIL  ◀ flagship
│   ├── materials/       nvidia_backends.py, calphad/dft/mlip backends
│   └── visualization/   plots, vector fields, reports
├── api/fastapi_server/  REST surface
├── mcp/                 MCP tool + knowledge servers
└── db/                  experiment persistence

docs/      architecture, experiment catalog, STACK.md, proposals, alloy findings
examples/  runnable usage examples per domain
proof/     knowledge corpus + HTTP cache
external/  external scientific software (Quantum ESPRESSO, ABINIT, ...)
tests/     pytest suites (125 tests)
```

### Key documents for review

| Document | Purpose |
|---|---|
| [`docs/STACK.md`](docs/STACK.md) | Full stack reference — every route enumerated |
| [`docs/02-architecture.md`](docs/02-architecture.md) | Architecture deep-dive |
| [`docs/03-experiment-catalog.md`](docs/03-experiment-catalog.md) | Annotated route catalog |
| [`docs/PROPOSAL_NVIDIA_4PAGE.pdf`](docs/PROPOSAL_NVIDIA_4PAGE.pdf) | The 4-page research proposal |
| [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md) | End-user manual |
| [`docs/ISSUES_AND_TODO.md`](docs/ISSUES_AND_TODO.md) | Live issue tracker and roadmap |

---

## 10. Status & Roadmap

**Working today**
- 223 dispatcher routes across 11 domains; 125/125 tests passing
- Full NVIDIA GPU stack (Warp, PhysicsNeMo, CuPy, PyTorch CUDA) verified on RTX 3060
- Blade thermal QD pipeline end-to-end; MAP-Elites validated against published baselines
- CALPHAD phase equilibria (Al-Cr-Ni demo database)
- Quantum ESPRESSO 7.5 installed (`external/dft/`) — DFT-capable pending pseudopotentials

**Roadmap (see `docs/ISSUES_AND_TODO.md`)**
- *Easy / high value* — download MLIP model files (MACE-MP-0, SevenNet-0)
- *Medium* — full TDB database for multi-component CALPHAD; QE pseudopotential setup
- *Medium* — migrate `torch.jit` export to `torch.export` (deprecation cleanup)
- *Breadth* — wire dispatcher routes for the staged engine modules: astronomy,
  biology, CFD, geology, environmental, plasma, quantum, optimization, surrogates
  (engine code exists; routing not yet registered)

---

## 11. Dependencies

**Core** — SymPy · NumPy · SciPy · Matplotlib · Plotly · Pandas · Pydantic v2 · FastAPI · uvicorn · MCP · httpx
**Optional domains** — RDKit (chemistry) · ASE & pymatgen (atomic/materials) · python-control (control)
**GPU** — PyTorch CUDA · Warp · PhysicsNeMo · CuPy · ONNX Runtime GPU · TensorRT
**High-fidelity materials** — pycalphad (CALPHAD) · MACE / SevenNet (MLIP) · Quantum ESPRESSO (DFT)
**QD** — JAX · Optax (autodiff geometry optimization)

---

*WorldSIM v0.2.0 — MaxOSL AI Research. Built as a unified simulation substrate and
a research platform for quality-diversity optimization of engineered geometries.*
