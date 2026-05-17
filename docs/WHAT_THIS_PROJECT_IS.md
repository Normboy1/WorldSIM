# What This Project Is — A Plain-Language Guide

This document explains WorldSIM and the NVIDIA grant proposal from first principles, as if you built it in fragments and need to see the whole thing at once.

---

## The Big Picture in One Paragraph

You are building a **simulation lab** that can run physics experiments inside a computer. You designed it to be modular — different kinds of physics (materials, blade heat transfer, alloy discovery, geometry optimisation) plug into the same routing system. On top of that simulation lab, you are building a **grant-funded GPU research pipeline** specifically for one hard engineering problem: figuring out the best arrangement of cooling holes inside a turbine blade to keep it from melting. The grant asks NVIDIA to give you access to their most powerful GPU (H100) and their five main simulation tools, which your prototype is already wired up to use — just via slow placeholder code instead of the real thing.

---

## Part 1 — The Simulation Lab (WorldSIM / SimLabCore)

### What problem does it solve?

Running a physics simulation is slow, and most simulation code is a mess of one-off scripts. WorldSIM provides a **common interface**: you send it a request like `(domain="qdgeometry", type="blade_simulation")` and it routes that request to the right engine, validates the inputs, and returns a structured result. It's designed so that adding a new physics engine doesn't break anything else.

### How does routing work?

Every experiment is described as an `ExperimentRequest` — a Pydantic model with a `domain`, a `type`, and a `parameters` dict. The dispatcher looks up `(domain, type)` in a routing table and calls the corresponding engine. If the engine's library isn't installed, the dispatcher falls back to a proxy (usually NumPy or PyTorch) so the code still runs, just slower or with placeholder results.

### What engines currently exist?

| Domain | What it does |
|---|---|
| `qdgeometry` | Blade thermal + stress simulation; MAP-Elites search over cooling hole geometry |
| `materials` | Alloy discovery; MAP-Elites search over Ni-superalloy composition space |
| `nvidia` | Stubs for Warp, PhysicsNeMo, cuPyNumeric, TensorRT, Omniverse — all currently proxy fallbacks |

### What's real vs what's a proxy?

| Component | Current state |
|---|---|
| PyTorch Jacobi FD thermal solver | Real — runs on GPU, produces correct T-fields |
| MAP-Elites archive | Real algorithm — correct Mouret & Clune implementation |
| SAIL surrogate | Real GP/RBF chain — correct Gaier et al. implementation |
| SIREN neural implicit | Real architecture — but requires JAX (not installed); raises a clear error if absent |
| Warp `@wp.kernel` FD solver | Stub — routes to PyTorch FD fallback |
| PhysicsNeMo FNO | Stub — routes to sklearn/RBF fallback |
| TensorRT inference | Stub — routes to PyTorch forward pass |
| cuPyNumeric archive | Stub — routes to NumPy |
| Omniverse rendering | Stub — no rendering happens |
| pycalphad CALPHAD | Real — v0.11.1 installed; runs on the Al-Cr-Ni ternary system |

---

## Part 2 — The Blade Physics Problem

### What are we trying to optimise?

A turbine blade runs inside a jet engine at around 1300°C. The blade is hollow, and there are small circular holes drilled through the interior through which cool air flows. The question is: **how many holes, where, and how big?**

The answer matters because:
- Too few holes or badly placed holes → blade overheats → metal creeps → blade fails
- Too many holes → blade becomes structurally weak → fatigue failure

The design space has 38 parameters: number of holes (up to 16), their x/y positions, their radii, and the coolant temperature.

### What physics model are we using?

A 2D steady-state heat conduction model on a NACA-4412 airfoil cross-section (a standard wing profile shape used in turbine blades). The heat equation is:

**∇²T = 0** (Laplace equation for steady-state temperature)

with three types of boundary conditions:
- **Outer hot-gas wall**: Dirichlet BC — T is fixed at 1300°C. This is a simplification: in a real engine the hot gas convects heat into the wall. Here we just say "the outer wall is 1300°C." This means T_peak = 1300°C always and cannot be changed by hole placement.
- **Cooling hole walls**: Robin/convective BC — the hole wall exchanges heat with the coolant. The coolant absorbs heat, so T drops near the holes. This IS affected by hole placement.
- **Lateral boundaries**: Neumann (zero flux) — insulated sides.

### What can actually be optimised?

Because the outer wall is pinned at 1300°C (Dirichlet), the **peak temperature T_peak is always 1300°C regardless of hole placement** — it is not an optimisation target.

The thing that varies is **T_mean**: the bulk average temperature of the cross-section interior. Better hole placement draws more heat away, so T_mean drops. Lower T_mean = cooler blade overall = longer life.

The optimization target is: **minimise T_mean**.

### What about stress, creep, and fatigue?

These are not solved separately — they are post-processed from the temperature field analytically:

- **Thermal stress**: plane-strain thermoelastic model. The local stress is proportional to the local temperature deviation from the mean: σ_vm ∝ E(T) · α · |T − T_mean|. Holes create steep local gradients, which creates stress concentrations.
- **Creep**: Norton-Bailey model. At high temperature and stress, metal slowly deforms permanently over time. Rate: dε/dt = A · σⁿ · exp(−Q/RT). Recalibrated for realistic Ni-superalloy values (A = 1.7×10⁻³⁶).
- **Fatigue**: Basquin high-cycle fatigue. Repeated thermal cycling eventually causes crack nucleation. Life estimate from peak stress amplitude.

None of these are "learned" by the FNO — they are computed analytically once you have the T-field.

### What is the NACA-4412?

A standard 4-digit NACA airfoil profile (4% max camber, at 40% chord, 12% max thickness). It's a well-known shape used in aerodynamics and often used as a proxy for turbine blade cross-sections in research. The 2D cross-section is discretised to a 256×128 grid.

---

## Part 3 — The Search Algorithm (MAP-Elites and SAIL)

### What is MAP-Elites?

MAP-Elites (Mouret & Clune, 2015) is a **quality-diversity** optimisation algorithm. Instead of finding one best design, it finds many good designs that are *different from each other*. It maintains an archive — a grid of cells, where each cell corresponds to a region of a "feature space." Each cell holds the best design found so far for that region.

In this project:
- The archive is a 100×100 grid = 10,000 cells
- Feature axis 1: T_mean (0°C to 1300°C range) — how cool is the blade overall?
- Feature axis 2: max thermal gradient |∇T|_max — how steep are the temperature gradients (stress proxy)?

A design that is thermally efficient but has steep gradients (high stress) goes into a different cell than one that is moderately efficient with gentle gradients. The archive explores the *tradeoff space*.

### What is SAIL?

SAIL (Surrogate-Assisted Illumination, Gaier et al., 2017) accelerates MAP-Elites by using a fast surrogate (a learned approximation) instead of the expensive physics simulator for most evaluations. Only the best candidates from the surrogate acquisition phase are sent to the real simulator for validation.

The SAIL loop:
1. Run Warp FD solver on a batch of candidates → get real T-fields and quality scores
2. Train the FNO surrogate on all real evaluations so far
3. Use TensorRT-accelerated FNO to score millions of candidates cheaply (UCB acquisition)
4. Send top-50 candidates from UCB back to Warp for real evaluation
5. Repeat

This lets you effectively explore 10 million candidate designs while only running the expensive Warp FD solver ~200,000 times.

### What is UCB?

Upper Confidence Bound — a strategy for exploration vs. exploitation. For each archive cell, UCB score = predicted quality + β × uncertainty. A cell with high predicted quality gets explored because it's likely good. A cell with high uncertainty gets explored because we don't know much about it yet. β controls the tradeoff.

---

## Part 4 — The Five NVIDIA Technologies

### Warp (the physics engine)

NVIDIA Warp is a Python package that compiles Python functions to CUDA kernels at runtime. Instead of writing a Jacobi iteration as a Python loop (slow), you write it as a `@wp.kernel` function and Warp compiles it to optimised CUDA.

The key extra feature is `wp.Tape`: Warp can automatically differentiate through the kernel, giving you `∂T_mean/∂(hole positions, radii)` — the gradient of the temperature with respect to the hole placement. This is used for **gradient-assisted MAP-Elites**: instead of only mutating randomly, we also take gradient steps toward better designs within each archive cell.

Current state: prototype uses PyTorch Jacobi FD. Grant implements the real `@wp.kernel`.

### PhysicsNeMo (the surrogate)

NVIDIA PhysicsNeMo is a library for physics-informed neural networks. We use it to train a **Fourier Neural Operator (FNO)** — a neural network that takes a geometry encoding (where the holes are) and predicts the full temperature field (all 256×128 values) in one forward pass.

The FNO is physics-informed: in addition to matching the Warp-computed T-fields (data loss), it also minimises the PDE residual ∇²T = 0 at interior nodes (physics loss). This makes it generalise better to unseen hole configurations — it can't just memorise training examples, it has to be thermodynamically consistent.

The surrogate predicts T-fields at around 1ms (PyTorch FP16) vs 0.05s for the real Warp solver — a 50× speedup per evaluation.

### TensorRT (the acceleration)

TensorRT is NVIDIA's inference engine. It takes a trained neural network (here: the PhysicsNeMo FNO) and compiles it to a highly optimised FP16 engine that runs much faster than the training-time PyTorch model.

The FNO exported to TensorRT runs a *batch* of 1,024 geometry encodings in about 0.1ms — roughly 10 million T-field evaluations per second. This makes the UCB acquisition phase (scoring millions of candidates) fast enough to run inside the SAIL loop without becoming a bottleneck.

**Important constraint**: TensorRT handles only the FNO forward pass. All the archive logic (storing designs, mutating parameters, selecting cells, computing UCB scores) is in cuPyNumeric — TensorRT can't run a stateful optimisation loop.

**Known risk**: FNO uses FFT operations (spectral layers). FFT is supported in TensorRT via ONNX opset ≥17, but this requires a smoke test. If TRT compilation fails, the fallback is `torch.compile` FP16, which gets ~3M evaluations/sec — still fast enough.

### cuPyNumeric (the archive)

cuPyNumeric is NVIDIA's drop-in GPU replacement for NumPy. You replace `import numpy as np` with `import cupynumeric as np` and your existing array code runs on GPU without changes.

The MAP-Elites archive stores 10,000 × 38-dimensional parameter vectors — about 380,000 floats. This is small enough to fit easily in GPU memory, but at 512 candidates per batch and 200,000 iterations, the cumulative host↔device transfer overhead of regular NumPy (~0.5ms per iteration) adds up to about 100 seconds wasted. cuPyNumeric eliminates this by keeping the archive on-device the whole time.

The 100×100 archive (10,000 cells) is also the key scientific contribution relative to prior SAIL work (which used 20×20 = 400 cells). At 10,000 cells, GPU-resident archive operations are a genuine necessity.

### Omniverse (the validation interface)

NVIDIA Omniverse is a 3D simulation and visualisation platform. In this project it has a specific scientific function — not just pretty pictures:

**Uncertainty-guided FEM prioritisation**: After the SAIL campaign, the archive has 10,000 designs, each with a T_mean prediction and an FNO uncertainty estimate (from MC-dropout). Running the expensive CalculiX FEM solver on all 10,000 is impossible. Omniverse renders the archive as a 3D blade with two overlays: predicted T_mean and FNO uncertainty. The engineer looks at the uncertainty map and picks which 50 cells to send to CalculiX — the ones with high uncertainty are most likely to have wrong FNO predictions and most need real validation.

Secondary function: Omniverse Replicator generates 10,000 synthetic blade geometry variants (varying chord, leading edge, aspect ratio) as extra FNO training data.

---

## Part 5 — The CALPHAD Episode (Why It Matters to This Proposal)

There is a separate sub-project in WorldSIM for **alloy discovery**: using MAP-Elites to search over Ni-superalloy compositions (Ni-Cr-Co-Al-Ti-Ta six-component space) to find alloys with good properties.

When the MAP-Elites top candidate (Ni=0.694, Cr=0.183, Al=0.122, a simplified 3-element proxy) was run through real CALPHAD thermodynamics (pycalphad v0.11.1), it showed **100% single-phase γ' (L12_FCC)** from 800–1350°C. A real Ni-superalloy needs two phases: γ (FCC matrix) + γ' (precipitate). The algorithm found a composition that scores highly on the heuristic quality function but is the wrong microstructure.

This is not a random failure. The heuristic quality function rewarded high Ni/(Al+Ti) ratio without checking the actual phase fractions. The fast surrogate was optimising a proxy, not the physical objective.

This exact failure mode — surrogate optimises a proxy → physics says it's wrong — is the core scientific argument for **why you need a tiered evaluation system**: FNO fast surrogate → Warp FD real physics → CalculiX FEM oracle. At each tier, some fraction of candidates get rejected that the previous tier would have approved.

The CALPHAD story is included in the proposal appendix as a concrete, reproducible demonstration of this principle.

---

## Part 6 — The Six-Month Plan

| Month | What gets built |
|---|---|
| 1 | Replace PyTorch FD with Warp `@wp.kernel`. Add `wp.Tape` gradient test. Upgrade archive to 10,000 cells with cuPyNumeric. |
| 2 | Validate Warp FD against CalculiX FEM on 3 reference geometries (6, 10, 16 holes). Acceptance criteria: mean |ΔT| < 50°C, peak |ΔT| < 80°C. |
| 3 | Train PhysicsNeMo FNO2d on 15,000 Warp-generated T-fields. Export to TensorRT. Smoke test: output error vs PyTorch < 0.5%. |
| 4 | Activate TensorRT acquisition engine. Benchmark: ≥1M candidates/sec. |
| 5 | Run full SAIL campaign: 200,000 Warp evaluations, 10 FNO retraining cycles, 10,000-cell archive. Target: T_mean ≥ 50°C below 6-hole baseline. |
| 6 | Omniverse archive viewer with uncertainty overlay. Replicator augmentation. Open-source release + paper draft. |

---

## Part 7 — What the Grant Is Actually Asking For

**107 H100 80GB GPU-hours.** This is a modest amount relative to the maximum allowed (30,000 hours). The H100 is needed because:
- FNO training at batch=512 with PDE residual loss exceeds the RTX 3060's 12GB VRAM
- TensorRT 10.x requires CUDA capability ≥ 8.0 (H100 = 9.0)
- Warp kernel profiling (Nsight Compute) needs CUDA 9.0
- 200,000 FD evaluations at 0.05s/eval = 2.8 GPU-hours, which would take 17+ hours on the RTX 3060 and block fast iteration

**1–2 concurrent GPUs.** Mostly 1; the 2nd is only for a cuPyNumeric multi-GPU scaling benchmark.

**500 GB storage.** Training data, Omniverse assets, checkpoints, logs.

**No physical hardware.** Everything is cloud.

---

## What This Is Not

- **Not a real film-cooling model.** Real film cooling involves 3D fluid dynamics, jet-mainstream interaction, turbulence, and conjugate heat transfer. This model is 2D steady heat conduction with simplified cooling BCs. It is a valid optimisation testbed for hole geometry, but not a substitute for CFD.
- **Not a validated superalloy design tool.** The CALPHAD sub-project found the wrong phase with a simplified 3-element system. The real 6-element superalloy space requires a proper TDB database (TCNI9), which is not yet available.
- **Not a working NVIDIA GPU pipeline yet.** All five NVIDIA technologies are routed but currently use proxy fallbacks. The grant builds the real implementations.
- **Not CalculiX-validated.** The Warp FD solver has not yet been compared to a proper FEM oracle. That comparison is Month 2 of the grant.
