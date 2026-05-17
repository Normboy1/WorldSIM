# SimLab User Manual

SimLab is a unified scientific simulation API. You describe what you want to compute using a structured request; SimLab routes it to the right engine, runs it, and returns a structured result. The same pattern works across physics, materials, chemistry, optimization, and more.

---

## Table of Contents

1. [Quick start](#1-quick-start)
2. [The request/result pattern](#2-the-requestresult-pattern)
3. [Domain reference](#3-domain-reference)
   - [math](#31-math)
   - [physics](#32-physics)
   - [chemistry](#33-chemistry)
   - [materials](#34-materials)
   - [atomic / nuclear](#35-atomic--nuclear)
   - [control](#36-control)
   - [qdgeometry — blade thermal & QD optimization](#37-qdgeometry--blade-thermal--qd-optimization)
   - [nvidia — GPU backends](#38-nvidia--gpu-backends)
4. [The blade thermal pipeline end-to-end](#4-the-blade-thermal-pipeline-end-to-end)
5. [Using the engines directly (without SimLabCore)](#5-using-the-engines-directly-without-simlabcore)
6. [Understanding the result object](#6-understanding-the-result-object)
7. [Common errors and fixes](#7-common-errors-and-fixes)
8. [Running the tests](#8-running-the-tests)

---

## 1. Quick start

```python
from simlab.core.engine.simlab_core import SimLabCore

core = SimLabCore()

# Solve a physics problem
result = core.simulate_physics(
    "projectile_motion",
    {"v0": 50, "angle_deg": 45},
)
print(result.status)           # "success"
print(result.results["range"]) # 254.9 m

# Solve a math problem
result = core.solve_math(
    "solve_equation",
    {"equation_str": "x**2 - 5*x + 6 = 0"},
)
print(result.results["solutions"])  # [2, 3]

# Run a materials simulation
result = core.simulate_materials(
    "calphad_phase_equilibrium",
    {"composition": {"Al": 0.1, "Cr": 0.1, "Ni": 0.8}, "temperature_K": 1300},
)
print(result.results["backend"])        # "pycalphad"
print(result.results["phase_fractions"]) # {"L12_FCC": 1.0}
```

That's the whole API for most use cases. Pick a domain shortcut, pass a type and parameters, read the result.

---

## 2. The request/result pattern

Every call goes through one data model in and one data model out.

### ExperimentRequest

```python
from simlab.core.schemas.experiment import ExperimentRequest

req = ExperimentRequest(
    domain      = "physics",           # which domain (see §3)
    type        = "projectile_motion", # which simulation within that domain
    parameters  = {"v0": 50, "angle_deg": 45},
    environment = {"g": 1.62},         # optional overrides (e.g. moon gravity)
    outputs     = ["plot"],            # optional: "plot", "report", "json", "latex"
    solver      = "auto",              # optional: prefer a specific solver
)

result = core.run_experiment(req)
```

`environment` is for physical constants you want to override — gravity, ambient temperature, etc. `parameters` is everything else.

### ExperimentResult

```python
result.status       # "success" | "error" | "partial"
result.results      # dict — the actual numbers and arrays
result.plots        # list of base64 PNG strings (when outputs=["plot"])
result.errors       # list of error strings (when status == "error")
result.warnings     # list of non-fatal warnings
result.metadata     # backend info, timings, etc.
result.experiment_id # auto-generated ID like "exp_3a7f9c12"
```

Check `result.status` before using `result.results`. When status is `"error"`, `result.results` is empty and `result.errors` explains why.

### Domain shortcuts

Instead of building an `ExperimentRequest` manually, every domain has a helper method:

| Method | Domain |
|--------|--------|
| `core.solve_math(type, params)` | math |
| `core.simulate_physics(type, params, environment=...)` | physics |
| `core.simulate_chemistry(type, params)` | chemistry |
| `core.simulate_materials(type, params)` | materials |
| `core.simulate_atomic(type, params)` | atomic |
| `core.simulate_nuclear(type, params)` | nuclear |
| `core.query_data(type, params)` | data |
| `core.analyze_control(type, params)` | control |
| `core.run_experiment(req)` | any domain (raw request) |

There is no `simulate_qdgeometry` shortcut — use `core.run_experiment(req)` with `domain="qdgeometry"`.

---

## 3. Domain reference

### 3.1 math

```python
# Symbolic algebra
core.solve_math("solve_equation",  {"equation_str": "x**3 - x = 0"})
core.solve_math("differentiate",   {"expression": "sin(x)*exp(-x)", "variable": "x"})
core.solve_math("integrate",       {"expression": "x**2", "variable": "x"})
core.solve_math("simplify",        {"expression": "(x**2 - 1)/(x - 1)"})
core.solve_math("taylor_series",   {"expression": "exp(x)", "variable": "x", "point": 0, "n": 5})
core.solve_math("critical_points", {"expression": "x**3 - 3*x"})

# Linear algebra
core.solve_math("solve_linear",    {"A": [[2,1],[1,3]], "b": [5,10]})
core.solve_math("eigenvalues",     {"matrix": [[4,1],[2,3]]})
core.solve_math("svd",             {"A": [[1,2],[3,4],[5,6]]})
core.solve_math("determinant",     {"A": [[1,2],[3,4]]})

# ODEs
core.solve_math("solve_ode",       {"equation": "y' = -y", "y0": 1.0, "t_span": [0, 5]})
core.solve_math("solve_2nd_order", {"p": 0, "q": 1, "y0": 0, "dy0": 1, "t_span": [0, 10]})

# Optimization
core.solve_math("optimize",        {"expression": "x**2 + 3*x + 2", "x0": [0.0]})
core.solve_math("fit_curve",       {"x": [1,2,3,4], "y": [2.1,3.9,6.1,8.0],
                                    "model": "linear"})

# Statistics
core.solve_math("monte_carlo",     {"expression": "x**2", "n_samples": 10000,
                                    "x_range": [0, 1]})
core.solve_math("descriptive_stats", {"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
core.solve_math("hypothesis_test", {"group1": [5.1, 5.3, 4.9], "group2": [6.2, 6.0, 6.4],
                                    "test": "ttest"})
```

### 3.2 physics

```python
# Classical mechanics
core.simulate_physics("projectile_motion", {"v0": 50, "angle_deg": 30})
core.simulate_physics("projectile_drag",   {"v0": 50, "angle_deg": 30,
                                            "mass_kg": 0.1, "drag_coeff": 0.47, "area_m2": 0.01})
core.simulate_physics("pendulum",          {"length_m": 1.0, "theta0_deg": 20, "t_end_s": 10})
core.simulate_physics("spring_mass",       {"k": 10, "m": 1.0, "x0": 0.1, "v0": 0})
core.simulate_physics("collision",         {"m1": 1.0, "v1": 3.0, "m2": 2.0, "v2": -1.0,
                                            "type": "elastic"})

# Change the environment (constants)
core.simulate_physics("projectile_motion", {"v0": 50, "angle_deg": 45},
                      environment={"g": 1.62})  # Moon gravity

# Thermodynamics
core.simulate_physics("ideal_gas",      {"P": 101325, "T": 300, "n": 1.0})
core.simulate_physics("heat_transfer",  {"T_hot": 500, "T_cold": 300, "k": 10,
                                         "area_m2": 0.1, "thickness_m": 0.01})
core.simulate_physics("carnot_efficiency", {"T_hot_K": 800, "T_cold_K": 300})

# Fluid dynamics
core.simulate_physics("bernoulli",       {"v1": 2.0, "h1": 10.0, "A1": 0.1,
                                          "A2": 0.05, "rho": 1000})
core.simulate_physics("reynolds_number", {"rho": 1.2, "v": 10, "L": 0.1, "mu": 1.8e-5})

# Electromagnetism
core.simulate_physics("coulomb_force",   {"q1": 1e-6, "q2": -1e-6, "r_m": 0.1})
core.simulate_physics("magnetic_field",  {"current_A": 10, "r_m": 0.05, "geometry": "wire"})
```

### 3.3 chemistry

```python
# Molecule analysis (requires RDKit)
core.simulate_chemistry("molecule_analysis",     {"smiles": "CCO"})      # ethanol
core.simulate_chemistry("molecular_descriptors", {"smiles": "c1ccccc1"}) # benzene
core.simulate_chemistry("3d_coordinates",        {"smiles": "CC(=O)O"})  # acetic acid

# Reaction kinetics
core.simulate_chemistry("first_order",   {"k": 0.1, "C0": 1.0, "t_end": 30})
core.simulate_chemistry("second_order",  {"k": 0.05, "C0": 2.0, "t_end": 60})
core.simulate_chemistry("consecutive",   {"k1": 0.2, "k2": 0.1, "C_A0": 1.0, "t_end": 40})
core.simulate_chemistry("equilibrium",   {"k_f": 0.3, "k_r": 0.1, "CA0": 1.0, "CB0": 0.0,
                                          "t_end": 50})
core.simulate_chemistry("michaelis_menten", {"Km": 0.5, "Vmax": 1.0, "S0": 2.0, "t_end": 20})
```

### 3.4 materials

Materials is the largest domain — 70+ routes across lattice structure, thermodynamics, phase transformations, diffusion, stress/strain, oxidation, forging, and casting.

```python
# Crystal lattice
core.simulate_materials("fcc_lattice",  {"element": "Ni", "a": 3.52})
core.simulate_materials("bcc_lattice",  {"element": "Fe", "a": 2.87})
core.simulate_materials("create_lattice", {"lattice_type": "fcc", "element": "Al", "a": 4.05})

# CALPHAD phase equilibrium (real pycalphad for Al-Cr-Ni)
core.simulate_materials("calphad_phase_equilibrium", {
    "composition": {"Al": 0.1, "Cr": 0.1, "Ni": 0.8},
    "temperature_K": 1300,
})
# Returns: {"backend": "pycalphad", "phase_fractions": {...}, "tcp_risk": False}

# Phase scan over temperature range
core.simulate_materials("calphad_phase_scan", {
    "composition": {"Al": 0.1, "Cr": 0.1, "Ni": 0.8},
    "T_low_K": 800, "T_high_K": 1600, "n_points": 20,
})

# TCP phase stability check (important for superalloy design)
core.simulate_materials("tcp_phase_check", {
    "composition": {"Ni": 0.60, "Cr": 0.20, "Co": 0.10, "Mo": 0.05, "Al": 0.05},
    "temperature_K": 1100,
})

# Alloy property prediction (GPU-accelerated)
core.simulate_materials("alloy_property_prediction", {
    "composition": {"Ni": 0.60, "Cr": 0.20, "Co": 0.10, "Mo": 0.05, "Al": 0.05},
    "target_properties": ["yield_strength", "creep_life", "oxidation_resistance"],
    "temperature_K": 1000,
})

# Microstructure diffusion (Warp GPU)
core.simulate_materials("microstructure_diffusion", {
    "grid_shape": [64, 64],
    "time_s": 10.0,
    "steps": 50,
    "D": 1e-12,
})

# Phase transformations
core.simulate_materials("jmak_kinetics", {
    "T_C": 700, "t_end_s": 3600, "n_avrami": 2.5, "k": 1e-4,
})
core.simulate_materials("ttt_diagram", {
    "alloy": "steel_4140",
    "T_range_C": [200, 700], "n_points": 30,
})
core.simulate_materials("grain_growth", {
    "T_C": 900, "t_end_s": 7200, "d0_um": 20,
})

# Diffusion
core.simulate_materials("diffusion_profile", {
    "D": 1e-12, "C_surface": 1.0, "C_initial": 0.0,
    "x_max_m": 0.001, "t_s": 3600,
})
core.simulate_materials("arrhenius_diffusivity", {
    "D0": 2.3e-4, "Q_kJ_mol": 148, "T_low_C": 800, "T_high_C": 1200,
})

# Stress / strain
core.simulate_materials("stress_strain_curve", {
    "E_GPa": 210, "yield_MPa": 250, "UTS_MPa": 400,
    "strain_max": 0.3, "material": "steel",
})
core.simulate_materials("elastic_deformation", {
    "E_GPa": 200, "sigma_MPa": 100, "geometry": "beam",
    "length_m": 1.0, "width_m": 0.05, "height_m": 0.1,
})

# Oxidation
core.simulate_materials("oxidation_kinetics", {
    "material": "Ni", "T_C": 1000, "t_end_h": 100, "pO2": 0.21,
})
core.simulate_materials("pilling_bedworth_ratio", {"material": "Al"})
core.simulate_materials("corrosion_risk", {
    "material": "stainless_316", "environment": "seawater", "T_C": 25,
})

# Forging
core.simulate_materials("flow_stress", {
    "material": "Ti-6Al-4V", "T_C": 900, "strain_rate": 1.0, "strain": 0.3,
})
core.simulate_materials("dynamic_recrystallization", {
    "material": "Ni_superalloy", "T_C": 1100, "strain_rate": 0.1,
    "initial_grain_um": 50,
})
core.simulate_materials("forging_force", {
    "material": "steel", "T_C": 1200, "billet_dia_mm": 100,
    "reduction_ratio": 0.3,
})

# Casting
core.simulate_materials("solidification_time", {
    "alloy": "Al_7075", "mould_material": "steel",
    "wall_thickness_m": 0.02, "T_pour_C": 680,
})
core.simulate_materials("cooling_curve", {
    "alloy": "Al_7075", "T_pour_C": 680, "T_mould_C": 25,
    "wall_thickness_m": 0.015, "t_end_s": 300,
})
core.simulate_materials("scheil_microsegregation", {
    "alloy": "Al_7075", "fs_max": 0.95,
})
```

**Note on CALPHAD database coverage:** The default database covers Al-Cr-Ni only. Compositions outside that system return `backend: "proxy"` with a note explaining what's missing. For other alloy systems you need to supply a full TDB path when initialising the backend directly.

### 3.5 atomic / nuclear

```python
# Electron configuration
core.simulate_atomic("electron_config",  {"Z": 26})    # Iron
core.simulate_atomic("compare_elements", {"Z_list": [6, 7, 8]})

# Hydrogen orbitals
core.simulate_atomic("hydrogen_energy_levels", {"n_max": 5})
core.simulate_atomic("radial_probability",     {"n": 2, "l": 1})  # 2p
core.simulate_atomic("orbital_2d",             {"n": 3, "l": 2, "m": 0})  # 3d

# Nuclear
core.simulate_nuclear("analyze_nucleus",      {"Z": 92, "N": 143})  # U-235
core.simulate_nuclear("binding_energy_curve", {"A_range": [1, 250]})
core.simulate_nuclear("decay",                {"isotope": "Ra-226", "t_end_s": 1e11})
core.simulate_nuclear("decay_chain",          {"parent": "U-238", "steps": 14})
core.simulate_nuclear("fusion_energy",        {"reaction": "DT"})  # deuterium-tritium
```

### 3.6 control

```python
# Transfer functions and analysis
core.analyze_control("transfer_function", {
    "numerator": [1], "denominator": [1, 2, 1],
})
core.analyze_control("pid_controller", {
    "Kp": 1.0, "Ki": 0.5, "Kd": 0.1,
    "plant_num": [1], "plant_den": [1, 2, 0],
})
core.analyze_control("bode", {
    "numerator": [10], "denominator": [1, 3, 10],
    "omega_range": [0.01, 100],
})
core.analyze_control("step_response", {
    "numerator": [1], "denominator": [1, 1.4, 1],
    "t_end": 20,
})
core.analyze_control("design_pid", {
    "plant_num": [1], "plant_den": [1, 2, 0],
    "desired_settling_time_s": 5.0, "desired_overshoot_pct": 10,
})
```

### 3.7 qdgeometry — blade thermal & QD optimization

The `qdgeometry` domain is the turbine blade internal cooling optimization pipeline. It does not have a shortcut method — use `run_experiment` with an `ExperimentRequest`.

```python
from simlab.core.schemas.experiment import ExperimentRequest

def qdreq(exp_type, params):
    return ExperimentRequest(domain="qdgeometry", type=exp_type, parameters=params)
```

#### Blade thermal solver (Warp FD)

Runs 2D steady heat conduction (∇²T = 0) on a NACA 4412 blade wall cross-section. Boundary conditions: Dirichlet T=1300°C at the outer hot-gas wall, Robin/convective at cooling hole walls, Neumann (zero flux) at lateral edges.

```python
result = core.run_experiment(qdreq("warp_blade_thermal", {
    "hole_cx": [0.2, 0.4, 0.6, 0.8],   # x positions in [0, 1]
    "hole_cy": [0.30, 0.30, 0.30, 0.30], # y positions in [0, 1]  (0=cool side, 1=hot side)
    "hole_r":  [0.05, 0.06, 0.06, 0.05], # radii in [0, 1]
    "nx": 128, "ny": 64,                  # grid resolution
    "T_hot": 1300.0,                      # °C, hot-gas wall temperature
    "T_cool": 400.0,                      # °C, coolant temperature
    "h_cool": 5000.0,                     # W/m²K, convective coefficient
    "max_iters": 3000,
}))

print(result.results["T_mean"])        # mean temperature of interior solid (°C)
print(result.results["backend"])       # "warp_kernel" or "pytorch_fd_fallback"
print(result.results["wall_time_s"])   # seconds
# result.results["T_field"] is a (ny, nx) numpy array of the full temperature field
```

**Coordinate system:** `cx`, `cy`, `r` are all normalised to [0, 1] in the blade wall cross-section. `cy=0` is the coolant plenum side, `cy=1` is the hot-gas wall. Typical hole positions: `cx ∈ [0.08, 0.92]`, `cy ∈ [0.12, 0.55]`, `r ∈ [0.03, 0.10]`. The solver will raise `ValueError` if any hole extends outside the domain boundary.

**Input validation:** The solver raises `ValueError` for:
- Mismatched lengths of `hole_cx`, `hole_cy`, `hole_r`
- Any hole centre outside `(0, 1)` or radius outside `(0, 0.5)`
- Any hole whose boundary (cx ± r or cy ± r) reaches the domain edge
- `T_cool >= T_hot`
- Grid coarser than 8×4

#### FEM validation (CalculiX)

Validates the FD solver against a CalculiX finite-element model. Requires `ccx` installed (`sudo apt-get install calculix-ccx`).

```python
result = core.run_experiment(qdreq("calculix_validate", {
    "hole_cx": [0.2, 0.5, 0.8],
    "hole_cy": [0.25, 0.25, 0.25],
    "hole_r":  [0.05, 0.05, 0.05],
}))

cmp = result.results
print(cmp["fd_T_mean"])           # FD solver result (°C)
print(cmp["fem_T_mean"])          # CalculiX FEM result (°C)
print(cmp["delta_T_mean"])        # |FD - FEM| (°C)
print(cmp["passes_50C_criterion"]) # True if |ΔT_mean| < 50°C
```

You can also call CalculiX directly without the FD comparison:

```python
result = core.run_experiment(qdreq("calculix_thermal", {
    "hole_cx": [0.3, 0.7],
    "hole_cy": [0.3, 0.3],
    "hole_r":  [0.06, 0.06],
    "T_hot": 1300.0, "T_cool": 400.0,
    "n_elem_x": 40, "n_elem_y": 20,
}))
print(result.results["T_mean"])   # °C
print(result.results["T_peak"])   # °C (should be ~1300 at hot wall)
print(result.results["n_nodes"])  # mesh size
```

#### FNO surrogate model

Generate training data, train a Fourier Neural Operator surrogate, then use it for fast T_mean prediction.

```python
# Step 1: generate training data (runs warp_blade_thermal N times)
result = core.run_experiment(qdreq("fno_generate_data", {
    "n_samples": 200,
    "n_holes_range": [3, 8],
    "nx": 128, "ny": 64,
    "save_path": "data/fno_dataset.npy",
}))

# Step 2: train the FNO
result = core.run_experiment(qdreq("fno_train", {
    "dataset_path": "data/fno_dataset.npy",
    "epochs": 200,
    "modes": 16, "width": 48, "n_layers": 4,
    "save_path": "data/blade_fno.pt",
}))
print(result.results["final_val_l2_rel"])  # relative L² error
print(result.results["final_T_mean_mae"])  # mean absolute error in °C

# Step 3: run inference directly (not through dispatcher)
from simlab.engines.qdgeometry.fno_surrogate import load_model, encode_geometry, predict_with_uncertainty
import numpy as np

model = load_model("data/blade_fno.pt")
geom = encode_geometry([0.3, 0.7], [0.3, 0.3], [0.06, 0.06], ny=64, nx=128)
T_mean_field, T_uncertainty = predict_with_uncertainty(model, geom, n_mc=10)
# T_mean_field: (64, 128) — predicted temperature field in °C
# T_uncertainty: (64, 128) — epistemic uncertainty (std across MC-dropout passes)
```

#### MAP-Elites quality-diversity optimization

MAP-Elites maintains a grid archive of diverse solutions. Each cell holds the best solution found for a particular combination of behavioural features.

```python
# Alloy composition search
result = core.run_experiment(qdreq("mapelites_alloy_search", {
    "elements": ["Ni", "Cr", "Co", "Al", "Mo"],
    "bounds": {
        "Ni": [0.40, 0.70], "Cr": [0.10, 0.25],
        "Co": [0.05, 0.15], "Al": [0.03, 0.10],
        "Mo": [0.01, 0.05],
    },
    "n_iterations": 500,
    "dim1_bins": 10,   # archive grid size (10×10 = 100 cells)
    "dim2_bins": 10,
    "dim1_range": [6.5, 9.0],  # VEC (valence electron count) range
    "dim2_range": [7.5, 11.0], # density range (g/cm³)
}))

arch = result.results["archive"]
print(arch["coverage"])       # fraction of cells filled
print(arch["n_elites"])       # number of distinct elite solutions
print(result.results["best_composition"]) # highest-quality composition found

# Geometry shape search
result = core.run_experiment(qdreq("mapelites_geometry_search", {
    "primitive": "linear_extrusion",   # or "revolution", "sweep", "fillet_box"
    "n_iterations": 300,
    "constraint_targets": {
        "target_volume_m3": 1e-4,
        "target_aspect_ratio": 3.0,
        "min_wall_m": 0.002,
    },
}))

print(result.results["best_params"])   # best geometry parameters
print(result.results["best_quality"])  # negative constraint loss (0 = perfect)
print(result.results["archive"]["coverage"])

# Sphere benchmark (algorithm validation — Mouret & Clune 2015)
from simlab.engines.qdgeometry.mapelites import mapelites_sphere_benchmark

bench = mapelites_sphere_benchmark(n_dims=20, grid_bins=10, n_iter=2000, seed=42)
print(bench["coverage"])      # should be ≥ 0.85
print(bench["best_quality"])  # should be ≥ 0.90
print(bench["qd_score"])      # should be ≥ 0.70
print(bench["all_pass"])      # True if all three criteria met
```

#### SAIL (surrogate-assisted illumination)

SAIL wraps MAP-Elites with a surrogate model to reduce calls to the expensive quality function.

```python
result = core.run_experiment(qdreq("sail_alloy_search", {
    "elements": ["Ni", "Cr", "Co", "Al"],
    "bounds": {"Ni": [0.45, 0.65], "Cr": [0.15, 0.25],
               "Co": [0.05, 0.15], "Al": [0.04, 0.10]},
    "n_iterations": 200,
    "surrogate_type": "gp",   # Gaussian process surrogate
}))

result = core.run_experiment(qdreq("sail_geometry_search", {
    "primitive": "revolution",
    "n_iterations": 200,
    "surrogate_type": "polynomial",
}))
```

### 3.8 nvidia — GPU backends

```python
# Warp GPU diffusion field
core.simulate_nvidia("warp_diffusion", {
    "grid_shape": [128, 128],
    "D": 1e-9,
    "time_s": 100,
    "steps": 500,
})

# Warp Allen-Cahn grain growth
core.simulate_nvidia("warp_allen_cahn", {
    "grid_shape": [64, 64],
    "n_grains": 20,
    "time_steps": 200,
})

# PhysicsNeMo FNO surrogate training plan
core.simulate_nvidia("nemo_train_fno", {
    "domain": "heat_transfer",
    "n_samples": 500,
    "epochs": 100,
})

# ALCHEMI ML interatomic potential
core.simulate_nvidia("alchemi_relax", {
    "structure": {"formula": "Ni3Al", "lattice": "L12"},
    "potential": "chgnet",
})
```

---

## 4. The blade thermal pipeline end-to-end

This example runs the full SAIL pipeline: Warp FD → FNO surrogate → MAP-Elites optimization → CalculiX FEM validation.

```python
import numpy as np
from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest
from simlab.engines.qdgeometry.warp_blade import warp_blade_thermal
from simlab.engines.qdgeometry.fno_surrogate import (
    load_model, encode_geometry, predict_with_uncertainty
)
from simlab.engines.qdgeometry.mapelites import MAPElitesArchive
from simlab.engines.qdgeometry.calculix_validate import validate_fd_vs_fem

core = SimLabCore()

# ── Step 1: baseline — no holes ──────────────────────────────────────────────
baseline = warp_blade_thermal([], [], [], nx=128, ny=64, max_iters=2000)
print(f"No holes: T_mean = {baseline['T_mean']:.1f}°C")

# ── Step 2: run the FNO surrogate to predict T_mean fast ─────────────────────
model = load_model("data/blade_fno.pt")
history = getattr(model, "_history", {})
T_norm = history.get("T_norm", {"min": 400.0, "max": 1300.0})

def fno_predict_T_mean(cx, cy, r):
    geom = encode_geometry(cx, cy, r, 64, 128)
    T_field, uncertainty = predict_with_uncertainty(
        model, geom, n_mc=5,
        T_min=T_norm["min"], T_max=T_norm["max"],
    )
    interior = (geom[0] - geom[1] - geom[3]).clip(0, 1)
    if interior.sum() > 0:
        return float((T_field * interior).sum() / interior.sum())
    return float(T_field.mean())

# ── Step 3: MAP-Elites on (T_mean × n_holes) feature space ───────────────────
archive = MAPElitesArchive(
    dim1_bins=10, dim2_bins=5,
    dim1_range=(600.0, 1200.0),   # T_mean range
    dim2_range=(2, 10),           # n_holes range
)

rng = np.random.default_rng(42)

# Initial population
for _ in range(100):
    n = int(rng.integers(2, 9))
    cx = rng.uniform(0.08, 0.92, n).tolist()
    cy = rng.uniform(0.12, 0.45, n).tolist()
    r  = rng.uniform(0.03, 0.09, n).tolist()
    try:
        T_pred = fno_predict_T_mean(cx, cy, r)
        quality = 1300.0 - T_pred   # lower T_mean = higher quality
        archive.try_insert({"cx": cx, "cy": cy, "r": r}, quality, T_pred, float(n))
    except Exception:
        continue

print(f"Archive coverage: {archive.coverage():.0%} ({archive.best()['quality']:.1f} best quality)")

# ── Step 4: verify best candidate with the FD solver ─────────────────────────
best = archive.best()["solution"]
fd_result = warp_blade_thermal(
    best["cx"], best["cy"], best["r"],
    nx=128, ny=64, max_iters=3000,
)
print(f"Best design: T_mean = {fd_result['T_mean']:.1f}°C "
      f"(reduction from baseline: {baseline['T_mean'] - fd_result['T_mean']:.1f}°C)")

# ── Step 5: CalculiX FEM validation ──────────────────────────────────────────
cmp = validate_fd_vs_fem(best["cx"], best["cy"], best["r"])
print(f"FD: {cmp['fd_T_mean']:.1f}°C   FEM: {cmp['fem_T_mean']:.1f}°C   "
      f"|Δ|: {cmp.get('delta_T_mean', 'N/A'):.1f}°C   "
      f"passes 50°C: {cmp.get('passes_50C_criterion', 'N/A')}")
```

---

## 5. Using the engines directly (without SimLabCore)

Every engine can be imported and called directly, bypassing routing and validation. This is faster for scripting but skips input checking.

```python
# Warp FD solver
from simlab.engines.qdgeometry.warp_blade import warp_blade_thermal
r = warp_blade_thermal([0.3, 0.5, 0.7], [0.3, 0.3, 0.3], [0.05, 0.05, 0.05],
                       nx=128, ny=64, max_iters=2000)

# CalculiX FEM
from simlab.engines.qdgeometry.calculix_validate import calculix_thermal, validate_fd_vs_fem
r = calculix_thermal([0.3, 0.7], [0.3, 0.3], [0.05, 0.05])

# MAP-Elites archive
from simlab.engines.qdgeometry.mapelites import MAPElitesArchive, mapelites_sphere_benchmark
arch = MAPElitesArchive(10, 10, (0.0, 1.0), (0.0, 1.0))

# CALPHAD
from simlab.engines.materials.calphad_backend import CALPHADBackend
b = CALPHADBackend()  # auto-uses demo Al-Cr-Ni database
r = b.calculate_phase_equilibrium({"Al": 0.1, "Cr": 0.1, "Ni": 0.8}, 1300)

# FNO surrogate
from simlab.engines.qdgeometry.fno_surrogate import (
    generate_training_data, train_fno, save_model, load_model,
    encode_geometry, predict_with_uncertainty,
)
dataset = generate_training_data(n_samples=200, solver="warp")
model, history = train_fno(dataset, epochs=100, modes=16, width=48)
save_model(model, "data/blade_fno.pt", history)
model = load_model("data/blade_fno.pt")
```

---

## 6. Understanding the result object

### Checking for success

Always check `result.status` before accessing `result.results`:

```python
result = core.simulate_physics("projectile_motion", {"v0": 50, "angle_deg": 45})

if result.status == "success":
    print(result.results["range"])
elif result.status == "error":
    print("Errors:", result.errors)
elif result.status == "partial":
    print("Warnings:", result.warnings)
    print("Partial results:", result.results)
```

### What `results` contains

Each route returns different keys. A few patterns that are consistent:

| Key | Present when |
|-----|-------------|
| `backend` | Almost always — names the engine used (`"warp_kernel"`, `"pycalphad"`, `"proxy"`, etc.) |
| `wall_time_s` | Most simulation routes |
| `T_mean`, `T_field` | Blade thermal routes |
| `phase_fractions` | CALPHAD routes |
| `archive` | MAP-Elites / SAIL routes |
| `best_quality`, `coverage` | MAP-Elites archive summary |

### Backend strings

When you see `backend: "proxy"` in a result, it means the real solver wasn't available and a heuristic approximation was used. Proxy results are structurally identical but less accurate. The `note` field in the result explains why.

| Backend string | Meaning |
|----------------|---------|
| `warp_kernel` | Real Warp @wp.kernel CUDA solver |
| `pytorch_fd_fallback` | Warp unavailable; used PyTorch CPU/GPU FD |
| `pycalphad` | Real thermodynamic calculation |
| `proxy` | Heuristic estimate; install optional dependency for real results |
| `calculix_fem` | CalculiX FEM (ccx binary) |
| `numpy`, `warp_cuda` | Microstructure diffusion on CPU / GPU |

---

## 7. Common errors and fixes

**`status == "error"`, errors contain `"Unknown experiment type"`**
The type string doesn't match any registered route. Check the domain reference above for the exact string.

```python
# Wrong
core.simulate_materials("calphad_equilibrium", {...})
# Right
core.simulate_materials("calphad_phase_equilibrium", {...})
```

**`ValueError: hole_cx[0]=1.1 out of (0, 1)`**
Hole positions must be in the open interval (0, 1). Check that `cx + r < 1` and `cy - r > 0`.

**`ValueError: Hole 0 (...) extends outside domain`**
The hole boundary reaches the domain edge. Reduce the radius or move the centre inward.

**`backend: "proxy"` in CALPHAD results for Al-Cr-Ni**
This means the element name had wrong case. Use title case: `"Al"`, `"Cr"`, `"Ni"` — not `"AL"` or `"al"`. The backend normalises internally, but if you're passing pre-normalised uppercase keys and seeing proxy, check that the composition doesn't include elements outside {Al, Cr, Ni} for the demo database.

**`status == "error"`, errors contain `"ccx not found"`**
Install CalculiX: `sudo apt-get install -y calculix-ccx`. CalculiX routes return a graceful error dict rather than raising, so `result.status` will be `"error"` with a clear message.

**FNO model size mismatch on `load_model`**
The model architecture is stored alongside the weights. If you get a size mismatch, the `.pt` file was saved with an older version of the code that didn't store hyperparams. Retrain the model:
```python
dataset = np.load("data/fno_dataset.npy", allow_pickle=True).item()
model, history = train_fno(dataset, modes=16, width=48, n_layers=4, epochs=300)
save_model(model, "data/blade_fno.pt", history)
```

**MAP-Elites `coverage` is very low (< 0.3)**
Either the feature ranges don't cover the actual feature values being generated, or too few iterations were run. Check that `dim1_range` and `dim2_range` bracket the real output range. For the blade thermal problem, T_mean typically falls in [500, 1250]°C.

---

## 8. Running the tests

```bash
# All tests
python -m pytest tests/ -q

# Just the QD / blade thermal suite
python -m pytest tests/test_qdgeometry_suite.py -v

# Just the original regression suite
python -m pytest tests/test_simlab_fixes.py -v

# Run with output (useful for seeing backend names, timings)
python -m pytest tests/ -s -q
```

**Expected output:**
```
79 passed in ~4s
```

The test suite requires:
- PyTorch (CPU is fine for most tests)
- `data/blade_fno.pt` (pre-trained FNO model) for FNO tests
- `ccx` binary for CalculiX tests (those are automatically skipped if ccx is absent)
- CUDA for Warp tests (falls back to CPU PyTorch FD if no GPU)
