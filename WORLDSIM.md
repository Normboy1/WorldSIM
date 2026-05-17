# WorldSIM v0.2.0

**Type:** Research software prototype  
**Purpose:** Multi-domain simulation routing framework with GPU-accelerated quality-diversity search  
**Author:** MaxOSL AI Research

---

## What this is

WorldSIM is a Python package that routes simulation requests to domain-specific engines through a single `(domain, type)` API. Some engines use established solvers (sympy, scipy, RDKit, ASE). Others fall back to analytical proxy models when optional dependencies are absent. The GPU throughput benchmarks are real measured values. Most materials and blade physics numbers are proxy estimates.

**This is not a validated scientific platform. It is not ready to support materials discovery claims or engineering conclusions without additional validation work described at the bottom of this document.**

---

## What it is not

- Not a FEM solver. The blade physics model uses decoupled analytical approximations.
- Not running real CALPHAD on this machine. No `.tdb` file is installed — the `calphad_phase_diagram` route returns proxy output.
- Not running DFT. VASP/QE/CP2K are not installed — the `dft_properties` route falls back to EMT (ASE empirical potential) or an analytical proxy.
- Not a validated alloy discovery tool. The reported Ni57Cr15Co9Al10Ti3Ta7 candidate comes from a heuristic scoring function.
- Not a peer-reviewed result in any domain.

---

## Environment

All benchmark numbers in this document were measured on:

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 |
| VRAM | 12,288 MiB |
| CUDA driver | 595.58.03 |
| CUDA toolkit | 12.1 |
| PyTorch | 2.2.2+cu121 |
| Python | 3.12.3 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| SymPy | 1.14.0 |
| OS | Linux 6.17.0-23-generic (Ubuntu, glibc 2.39) |
| Commit | c6ef341 |

Benchmark reproducibility:

```bash
git checkout c6ef341
pip install -e ".[ml]"
cd ResearchGrants/WorldSIM
python scripts/blade_physics_viz.py        # blade physics figure
python examples/materials/alloy_design_workflow.py  # alloy search
```

No fixed random seed is set globally. Individual functions accept a `seed` parameter where stochasticity is relevant (MAP-Elites initial population, SIREN weight init). Results may vary across runs where seed is not fixed.

---

## Architecture

```
simlab/
├── core/
│   ├── engine/          run_experiment() entry point
│   ├── router/          (domain, type) → engine method, all imports lazy
│   ├── validation/      required-param checks, chemical keyword blocklist*
│   ├── schemas/         ExperimentRequest / ExperimentResult  (Pydantic v2)
│   └── constants/       SI constants, periodic table data
├── engines/
│   ├── math/            sympy + scipy
│   ├── physics/         analytical closed-form models
│   ├── chemistry/       RDKit (optional) + analytical kinetics
│   ├── materials/       CALPHAD*, DFT*, MLIP*, phase transforms, GPU Allen-Cahn
│   ├── atomic/          ASE (optional), analytic hydrogen orbitals
│   └── qdgeometry/      MAP-Elites, SAIL, grammar GP, SIREN, PyTorch GPU
├── api/                 FastAPI REST server
└── mcp/                 MCP tool + knowledge servers
```

*\* The chemical safety blocklist is a keyword filter, not a formal hazard classification system. It blocks obvious tokens (nerve agent codes, SMILES substrings). It is not adversarially tested and should not be treated as a safety boundary.*

---

## Test coverage

27 tests in `tests/test_simlab_fixes.py`. Run with:

```bash
pytest tests/test_simlab_fixes.py -v
```

**What the tests cover:**

| Area | Tests | Notes |
|---|---|---|
| Physics engine correctness | `test_projectile_matches_analytical_range` | Checks range and max height against `v₀² sin(2θ)/g` to 1e-6 |
| Input validation — edge cases | gravity, ideal gas, ODE t_span, statistics | Checks that bad inputs raise `ValueError` with clear messages |
| Pendulum with non-standard g | `test_environment_g_merged_for_projectile` | Moon vs Earth gravity |
| Routing / dispatch | `test_validator_required_params_cover_all_registered_routes` | All routes in dispatcher have a validator entry |
| Atomic wavefunctions | `test_hydrogen_radial_wavefunctions_are_normalized` | Numerical integration of |ψ|² ≈ 1, error < 1e-3 |
| Materials proxy routing | GPU backend status, alloy prediction, degradation, diffusion | Checks routes work and return plausible structure |
| Security | CORS wildcard prevention, proof engine path traversal | |
| Sympy parsing | Solve, critical points, equation splitting | |
| Error surface | Error messages do not expose tracebacks | |

**What the tests do not cover:**

- Algorithm correctness of MAP-Elites, SAIL, or grammar GP against the original paper implementations or reference baselines. There is no unit test reproducing a known MAP-Elites archive from a published benchmark (e.g. Rastrigin / sphere / arm QD benchmark suite).
- Numerical accuracy of the blade physics solvers beyond "runs without error." No mesh convergence study. No FEM comparison.
- Materials proxy outputs against CALPHAD or DFT reference values.
- GPU timing reproducibility under different system load conditions.

---

## Engine fidelity table

| Domain | Backend | Fidelity |
|---|---|---|
| `math` — symbolic | sympy 1.14 | Symbolic for closed-form inputs. Numerical outputs have floating-point error. SciPy ODE solvers are numerical (RK45 by default), not exact. |
| `math` — linear algebra | scipy.linalg | IEEE 754 double precision. Not exact for ill-conditioned systems. |
| `physics` — mechanics, EM, thermodynamics | Analytical closed-form | Exact for the stated model. Models omit drag, radiation, relativity, etc. unless a specific experiment type adds them. |
| `chemistry` — kinetics | Analytical (first-order, Arrhenius) | Exact for the stated ODE. Not a full reaction mechanism solver. |
| `chemistry` — molecule analysis | RDKit (if installed), else rule-based | RDKit results are standard cheminformatics. Rule-based fallback is approximate. |
| `atomic` — hydrogen orbitals | Analytic wavefunctions | Exact for hydrogen (Z=1). Many-electron atoms use Aufbau/screening approximations. |
| `materials` — CALPHAD | pycalphad + TDB file | **Proxy mode on this machine** (no .tdb installed). All CALPHAD outputs marked `"backend": "proxy"` are analytical estimates. |
| `materials` — DFT | ASE + VASP/QE/CP2K | **Proxy/EMT on this machine**. No DFT code installed. EMT results are fast empirical-potential estimates. |
| `materials` — MLIP | mace-torch / sevenn | **Proxy on this machine**. Neither installed. |
| `qdgeometry` — MAP-Elites / SAIL | NumPy / PyTorch | Algorithm implementation correct per published pseudocode. **The objective function being optimised is a heuristic proxy**, not a physics simulation. |
| `qdgeometry` — blade simulation | PyTorch CUDA FD | Simplified analytical approximations. See solver notes below. |

---

## QD Geometry Engine

`simlab/engines/qdgeometry/`

### Algorithms implemented

**MAP-Elites** — follows Mouret & Clune (2015, arXiv:1504.04909). Archive of (n_cells × n_cells) grid. Gaussian mutation on continuous parameters. The algorithm structure matches the paper. No reproduction benchmark against a published MAP-Elites result has been run.

**SAIL** — follows Gaier, Asteroth & Mouret (GECCO 2017, doi:10.1145/3071178.3071282). Seed → surrogate fit → acquisition MAP-Elites → UCB top-k → true eval loop. Surrogate fallback chain: GP (sklearn) → RBF thin-plate spline (scipy, smoothing=1e-4) → polynomial. Fallbacks activate silently; check `result["surrogate_type"]` to see which was used.

**Grammar GP** — `(μ+λ)` ES on a `GeomNode` tree (primitive | union | diff). Structural and parameter mutation. Sub-tree crossover. Shape space is small (4 primitives, 2 boolean ops). This is a proof-of-concept for the ShapeAssembly-style direction (Jones et al., SIGGRAPH Asia 2020), not a full reimplementation of that system.

**SIREN** — 3→64→64→64→1 MLP with sine activations and ω₀=30 (Sitzmann et al., NeurIPS 2020, arXiv:2006.09661). Loss: SDF=0 on surface + Eikonal. JAX training when available (CPU only on this machine). The original numpy fallback silently returned a random-weight field with zero placeholder losses — this was a correctness failure and has been fixed: `NeuralImplicitField.fit()` now raises `RuntimeError` immediately if JAX is absent. `fit_to_geometry()` returns an explicit `{"error": ..., "backend": "unavailable", "trained": false}` dict. No silent pseudo-training. There is no marching-cubes mesh extraction; the field returns SDF/occupancy values only.

### GPU MAP-Elites throughput (RTX 3060)

Measured using `torch_gpu.gpu_mapelites()` with `primitive="fillet_box"`:

| Population | Measured throughput | Notes |
|---|---|---|
| 64 | ~350k evals/s | |
| 1,024 | ~1.4M evals/s | |
| 4,096 | ~2.8M evals/s | |
| 16,384 | ~3.52M evals/s | 226× vs single-threaded NumPy baseline |

**One evaluation** = sample a parameter vector from the archive, compute volume + aspect ratio + wall thickness + constraint penalty (all scalar arithmetic, ~15 floating-point ops). CUDA synchronization (`torch.cuda.synchronize()`) was called before and after the timed region. The number measures the throughput of a *heuristic fitness function*, not a physics simulation.

---

## GPU Blade Physics Simulation

`simlab/engines/qdgeometry/torch_gpu_physics.py`

A 2D multi-physics chain on a 256×128 CUDA grid (32,768 nodes). Total runtime: ~0.36 s on RTX 3060. The purpose is fast design-space screening and physics-informed visualisation — not engineering certification.

### Solver 1: Jacobi FD thermal (0.305 s)

2D steady-state heat conduction. Jacobi iteration on a Cartesian grid. Convective BCs at blade surface (h=3000 W/m²K, T_gas=1300°C) and cooling holes (h=8000 W/m²K, T_cool=650°C).

**Not modelled:** 3D spanwise flux, radiation, TBC thermal resistance, actual aerothermal boundary layer, pressure-dependent gas properties.

**Mesh convergence:** not studied. Results at 256×128 are not compared to a finer grid.

### Solver 2: Thermoelastic stress

Decoupled plane-strain approximation: `σ_vm ≈ E(T)·α·|T − T_mean| / (1−ν)`. Captures gradient-induced thermal mismatch stress.

**Not modelled:** centrifugal load, pressure load, stress redistribution beyond yield (purely linear-elastic), out-of-plane components, temperature-dependent Poisson's ratio.

**Peak σ_vm = 1329 MPa** exceeds yield (571 MPa proxy). In a real blade this stress would be redistributed by plasticity; FEM with a crystal-plasticity UMAT would be required to get accurate peak values.

### Solver 3: Norton–Bailey creep

`dε/dt = A·σ^n·exp(−Q/RT)`. Constants: A=1.7×10⁻³⁶ s⁻¹ Pa⁻⁴·⁸, n=4.8, Q=310 kJ/mol. Calibrated to give ~10⁻⁹ /s at 900°C / 300 MPa for a generic Ni-base superalloy — **not fitted to experimental data for Ni57Cr15Co9Al10Ti3Ta7**. That data does not exist.

Stress input is capped at yield before evaluation. Hot outer-surface nodes (T→1300°C, no TBC) show extreme accumulated strain — this is physically expected for an uncooled surface at those conditions, not a solver error.

### Solver 4: Basquin HCF fatigue

`N_f = (σ_f' / σ_amp)^(1/b)`. Constants σ_f'=1050 MPa, b=−0.085 are generic Ni-base estimates. Mean-stress correction (Goodman/Morrow), multiaxial effects, hold-time, thermomechanical fatigue, and environmental effects are not included.

### What the blade simulation result means

| Check | Result | What it actually means |
|---|---|---|
| Thermal margin 234°C | Pass | Plausible bulk margin; not accounting for TBC or full BC uncertainty |
| Stress safety factor 0.43 | Fail | Linear-elastic model overpredicts peak stress; plasticity would redistribute |
| Creep / fatigue | Fail at hot outer surface | Correct direction: outer surface without TBC at 1300°C fails. Numbers not validated. |

The simulation identifies the outer uncooled surface as the critical failure location. That conclusion is qualitatively correct. No number in this output should be quoted as an engineering result without FEM validation.

---

## Materials & Alloy Discovery

### Backend state on this machine

```
pycalphad:  v0.11.1 installed, alcrni.tdb (bundled) → REAL for Al-Cr-Ni ternary
            TCNI9 / full Ni-superalloy TDB: NOT available → proxy for full system
DFT codes:  not installed              → ASE EMT or proxy
mace-torch: not installed              → proxy
sevenn:     not installed              → proxy
JAX:        installed (CPU only)       → SIREN training works; GPU SIREN uses PyTorch
```

Every API response from these backends contains a `"backend"` field. **If it says `"proxy"`, the numbers are analytical estimates, not thermodynamic calculations.** Check this field before using any materials output.

### Routes currently returning proxy on this machine

Verified by running each route and inspecting the `"backend"` field in the response:

| Route | Status | Backend field |
|---|---|---|
| `materials/calphad_phase_diagram` | error (no .tdb) | — |
| `materials/allen_cahn_simulation` | error (GPU workflow not wired) | — |
| `materials/alloy_property_prediction` | success | `"proxy"` present in recommended_next_runs |
| `materials/gpu_diagnostics` | error | — |
| `qdgeometry/geometry_primitive` | success | no backend field (pure analytical) |
| `qdgeometry/blade_simulation` | success | no backend field (pure PyTorch) |
| `qdgeometry/mapelites_geometry_search` | success | no backend field (pure analytical) |
| `qdgeometry/fit_neural_implicit` | success (JAX available) | `"jax"` |

The `calphad_phase_diagram` and `allen_cahn_simulation` routes are currently broken on this machine — they error before returning. Any result that claims to use them is not coming from a working route.

### What the alloy search does

`mapelites_alloy_search` optimises a heuristic quality score defined in `gpu_workflow.py`:

- **VEC** (Valence Electron Count): `Σ xᵢ·VECᵢ`. A documented empirical indicator for FCC vs BCC tendency in HEAs (Guo & Liu, 2011, doi:10.1016/j.pnsc.2011.09.003). Target range 8.0–8.5 for FCC.
- **γ' fraction proxy**: `Ni/(Al+Ti)` atomic ratio. Correlates with γ' formation tendency. Not a phase fraction — that requires CALPHAD.
- **TCP risk proxy**: penalises `Mo+W > 0.15`. Crude empirical flag; not a sigma-phase stability calculation.
- **Density**: rule of mixtures on elemental densities. Valid for ideal solutions; error ±2–5% for real alloys.
- **No term for:** creep resistance, oxidation resistance, castability, hot corrosion, or γ' solvus temperature.

### Ni57Cr15Co9Al10Ti3Ta7

Highest-scoring output from the MAP-Elites heuristic search. Status: **unvalidated heuristic candidate**.

| Property | Value | Method | Limitation |
|---|---|---|---|
| VEC | 8.14 | Analytic (VEC table) | Empirical indicator only |
| Ni/(Al+Ti) | 4.57 | Analytic ratio | Not a phase fraction |
| TCP risk | None flagged | Proxy (Mo+W<0.15) | Not a thermodynamic calculation |
| Density | 8.44 g/cm³ | Rule of mixtures | ±2–5% error expected |
| T_solidus ~1534°C | Proxy estimate | Empirical analogue scaling | Not CALPHAD |
| Yield ~571 MPa | René 104 literature analogue | Not measured for this composition | |

Rounded integer fractions (57/15/9/10/3/7) sum to 101. The search outputs continuous values normalised to 1.0; the rounding is approximate.

### Seed variance (10 runs, n_iterations=400, batch_size=16)

Measured to answer: is the result reproducible, or seed-dependent?

```
seed= 0  quality=2.9972  VEC=8.199  Ni=0.576  Cr=0.177  Al=0.066  Ta=0.100
seed= 1  quality=2.9955  VEC=8.198  Ni=0.510  Cr=0.200  Al=0.041  Ta=0.100
seed= 2  quality=2.9947  VEC=8.203  Ni=0.592  Cr=0.154  Al=0.072  Ta=0.100
seed= 3  quality=2.9927  VEC=8.204  Ni=0.567  Cr=0.200  Al=0.079  Ta=0.059
seed= 4  quality=2.9997  VEC=8.200  Ni=0.524  Cr=0.200  Al=0.045  Ta=0.100
seed= 5  quality=2.9964  VEC=8.202  Ni=0.552  Cr=0.198  Al=0.040  Ta=0.081
seed= 6  quality=2.9998  VEC=8.200  Ni=0.535  Cr=0.200  Al=0.050  Ta=0.096
seed= 7  quality=2.9974  VEC=8.201  Ni=0.518  Cr=0.167  Al=0.040  Ta=0.100
seed= 8  quality=2.9978  VEC=8.201  Ni=0.557  Cr=0.170  Al=0.073  Ta=0.090
seed= 9  quality=2.9945  VEC=8.203  Ni=0.576  Cr=0.194  Al=0.082  Ta=0.062

mean=2.9966  std=0.0022  min=2.9927  max=2.9998
```

Quality variance is low (std=0.0022) — the objective landscape is smooth and VEC locks to ~8.20 across all seeds. What varies is the composition achieving that score: Ni ranges 0.51–0.59, Al ranges 0.04–0.08. That spread is wide enough to matter for γ' phase fraction. A CALPHAD scan over this range would show whether γ' stability is maintained across the full seed-variance band, which is the question that matters.

**To turn this into a defensible candidate requires:** real CALPHAD with a commercial TDB (TCNI9 or equivalent), full γ/γ' phase fraction scan over 700–1200°C across the Ni=0.51–0.59 / Al=0.04–0.08 seed-variance band, TCP phase stability check, at least one experimental analogue from the open literature, and ideally a small casting campaign.

---

## API & MCP

```bash
worldsim-api           # FastAPI on :8000  →  /docs for Swagger UI
worldsim-mcp           # MCP simulation tool server
worldsim-knowledge-mcp # MCP knowledge / RAG server
```

Example request:

```python
import requests
r = requests.post("http://localhost:8000/experiment", json={
    "domain": "qdgeometry",
    "type": "blade_simulation",
    "parameters": {"nx": 256, "ny": 128, "T_gas": 1300.0}
})
print(r.json()["data"]["thermal"]["backend"])   # always check backend field
```

---

## Installation

```bash
pip install -e .                      # core
pip install -e ".[qd]"                # + JAX, optax (gradient descent)
pip install -e ".[ml]"                # + PyTorch (GPU engines)
pip install -e ".[calphad,dft]"       # + pycalphad, ASE (still proxy without .tdb / DFT code)
```

GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For real CALPHAD results: obtain a `.tdb` file and pass `database_path=` to `CALPHADBackend`. A free starting point is the COST-507 database for Al alloys (not Ni superalloys).

---

## Figures

| File | What it shows | Proxy / real |
|---|---|---|
| `docs/figures/blade_physics.png` | Temperature, stress, creep, fatigue fields; convergence; pass/fail; solver timings | Simplified physics — for illustration |
| `docs/figures/alloy_3d_viz.png` | Multi-scale alloy schematic (supercell, microstructure, blade, TBC cross-section, etc.) | Illustrative — geometries hand-constructed, not simulation output |
| `docs/figures/qdgeometry_nvidia_full.png` | GPU throughput scaling, multi-primitive search, neural SAIL | Throughput numbers real; objective is heuristic |
| `docs/figures/qdgeometry_demo.png` | All QD geometry routes end-to-end | Routing pipeline demo; outputs are heuristic proxies |

---

## Known gaps and next steps

| Gap | Status | What would fix it |
|---|---|---|
| SIREN silent failure | **Fixed** (raises RuntimeError; returns error dict at API) | — |
| Alloy search seed variance undocumented | **Fixed** (10-seed table in doc, std=0.0022) | — |
| Routes returning proxy not listed | **Fixed** (proxy audit table in doc) | — |
| No algorithm benchmark vs paper baselines | Open | Run MAP-Elites on standard QD benchmarks (arm / sphere / Rastrigin); compare archive coverage to pymap_elites reference |
| No CALPHAD results | **Partially closed** — pycalphad installed, alcrni.tdb run. Finding: MAP-Elites top candidate sits in single-phase γ' field (wrong microstructure). Full system requires TCNI9 TDB. | See proposal PROPOSAL_DRAFT.md Objective 1 |
| Blade model not validated | Open | Run same geometry through CalculiX or Abaqus; compare T and σ fields |
| Norton constants not alloy-specific | Open | Fit to published creep data for CMSX-4 or René 104 as analogue |
| No mesh convergence study | Open | Run thermal solver at 64², 128², 256², 512²; plot ΔT_peak vs node count |
| Alloy candidate composition range not validated | Open | CALPHAD scan over seed-variance band (Ni=0.51–0.59, Al=0.04–0.08) |
| Safety blocklist is a keyword filter | Open | Structured hazard classification, allowed-use policy, adversarial test cases, audit log |
| calphad_phase_diagram route broken on this machine | Open | Fix route wiring; test with proxy first, then real TDB |

---

## References

| Method | Citation | Link |
|---|---|---|
| MAP-Elites | Mouret & Clune (2015). *Illuminating search spaces by mapping elites.* | arXiv:1504.04909 |
| SAIL | Gaier, Asteroth & Mouret (2017). *Data-efficient exploration, optimization, and modeling of diverse designs through surrogate-assisted illumination.* GECCO. | doi:10.1145/3071178.3071282 |
| SIREN | Sitzmann, Martel, Bergman, Lindell & Wetzstein (2020). *Implicit neural representations with periodic activation functions.* NeurIPS. | arXiv:2006.09661 |
| ShapeAssembly | Jones, Charatan, Guerrero & Mitra (2020). *ShapeAssembly.* SIGGRAPH Asia. | doi:10.1145/3414685.3417812 |
| Norton creep | Norton, F.H. (1929). *The Creep of Steel at High Temperatures.* McGraw-Hill. | — |
| Basquin HCF | Basquin, O.H. (1910). *The exponential law of endurance tests.* ASTM Proc. 10:625–630. | — |
| CALPHAD method | Lukas, Fries & Sundman (2007). *Computational Thermodynamics: The CALPHAD Method.* Cambridge. | doi:10.1017/CBO9780511804137 |
| VEC / HEA heuristics | Guo & Liu (2011). *Phase stability in high entropy alloys.* Prog. Nat. Sci. Mater. 21(6):433–446. | doi:10.1016/j.pnsc.2011.09.003 |
| MACE MLIP | Batatia et al. (2022). *MACE: Higher order equivariant message passing neural networks.* NeurIPS. | arXiv:2206.07697 |

---

*WorldSIM v0.2.0 — MaxOSL AI Research — 2026*  
*Commit c6ef341 — RTX 3060 / CUDA 12.1 / PyTorch 2.2.2+cu121 / Python 3.12.3*
