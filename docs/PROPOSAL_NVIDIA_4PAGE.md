# GPU-Accelerated Physics-Informed Optimization of Turbine Blade Internal Cooling Channel Geometry

---

## PI and Collaborator Information

**Principal Investigator:** [Full Name, Title (e.g. Associate Professor), Department, Institution — required; must be full-time faculty at a Ph.D.-granting accredited institution]  
**Co-Investigator / Lead Developer:** [Name, Role, Institution]  
**Contact email:** [PI institutional email]

---

## Abstract *(≤ 300 words)*

Turbine blade internal cooling channel geometry must minimise mean interior metal temperature T_mean (cooling effectiveness), thermal stress, creep strain, and fatigue damage across a 38-dimensional parameter space. The outer hot-gas wall carries a Dirichlet BC (T = 1300°C fixed); T_mean — not T_peak — is the design-controllable optimization target. Current finite-element (FE) evaluation is one to eight hours per design; existing workflows explore only tens of configurations per design cycle, leaving most of the design space uncharted.

This project builds a **Surrogate-Assisted Illumination (SAIL)** pipeline that maintains a 100×100 archive of high-performing, diverse cooling channel designs — indexed by two independent feature axes: **T_mean** (bulk mean interior temperature; the primary optimization target) and **max thermal gradient |∇T|_max** (peak gradient magnitude in the T-field; governs thermal stress independently of T_mean). This yields 10,000 archive cells covering designs that are thermally efficient, structurally mild, or both. The archive reduces FE oracle calls by more than 50× while covering the full design space. The pipeline is governed by 2D steady-state internal heat conduction (∇²T = 0) with parametric cooling holes on a NACA-4412 cross-section; thermal stress, creep strain, and fatigue damage are analytical post-processes of the computed temperature field.

Five NVIDIA technologies own distinct, non-overlapping roles: **Warp** provides a JIT-compiled `@wp.kernel` finite-difference thermal solver with `wp.Tape` reverse-mode autodiff for gradient-assisted archive updates; **PhysicsNeMo** trains a Fourier Neural Operator (FNO2d) surrogate on Warp-generated temperature fields using a physics-informed PDE residual loss; **cuPyNumeric** keeps the 10,000-cell MAP-Elites archive GPU-resident as a drop-in `import cupynumeric as np`, eliminating host↔device overhead; **TensorRT** compiles the trained FNO to an FP16 batch-inference engine (<0.1 ms per batch of 1,024 candidates, ~10M field evaluations/s) for rapid UCB acquisition scoring; and **Omniverse** renders the archive as a digital twin with FNO epistemic uncertainty overlay, guiding engineers to prioritise which archive cells receive expensive CalculiX FEM validation.

A working prototype (PyTorch FD solver, proxy NVIDIA backends) runs on an RTX 3060. The grant activates the live GPU backends, scales the archive to 10,000 cells, and validates the SAIL pipeline against CalculiX FEM on three reference geometries.

---

## Keywords *(≤ 5)*

physics-informed neural operator · MAP-Elites · turbine blade thermal optimisation · GPU surrogate · digital twin

---

## NVIDIA Platforms

| Technology | Version target | Role |
|---|---|---|
| NVIDIA Warp | ≥ 1.0 | Differentiable FD thermal solver (`@wp.kernel`, `wp.Tape`) |
| NVIDIA PhysicsNeMo | ≥ 0.7 | FNO2d surrogate with PDE residual loss |
| NVIDIA cuPyNumeric | ≥ 23.x | GPU-resident MAP-Elites archive (drop-in NumPy) |
| NVIDIA TensorRT | ≥ 10.x (ONNX opset ≥ 17) | Batched FNO inference engine for UCB acquisition |
| NVIDIA Omniverse | Replicator + USD | Uncertainty-guided FEM prioritisation; synthetic training data |

---

## Dataset and Model

**Training data (generated, not pre-existing):** 15,000 (geometry encoding → temperature field) pairs produced by the Warp FD solver on H100. Geometry encodings are 4-channel 256×128 arrays (solid mask, hole mask, distance field, boundary condition field). Temperature fields are 256×128 single-channel arrays. No proprietary or restricted dataset is required; all training data is generated from open physics.

**Model:** FNO2d — 4 Fourier layers, modes=(12,12), width=32, physics-informed loss (data loss + λ=0.1 × PDE residual ∇²T = 0 on interior non-hole nodes). MC-dropout (p=0.1) at inference for epistemic uncertainty. Exported to TensorRT FP16 for acquisition; fallback to `torch.compile` FP16 if spectral layers fail ONNX-TRT compilation.

**Baseline model / reference:** CalculiX open-source FEM (CPU) used as tier-2 oracle for top-10 archive candidates and for Warp FD validation on 3 reference geometries.

---

## Introduction

Turbine blade designers optimise internal cooling channel geometry to maximise component life. The governing physics is 2D steady heat conduction:

∇²T = 0 on the solid cross-section, with three distinct boundary conditions: **Dirichlet** (T = T_hot = 1300°C) on the outer hot-gas wall; **Robin/convective** (−k ∂T/∂n = h_c(T − T_cool)) on cooling hole walls; **Neumann** (zero flux, ∂T/∂n = 0) on lateral symmetry boundaries. The Dirichlet outer wall pins the absolute maximum temperature at 1300°C regardless of hole placement. The optimisation target is therefore **T_mean** — the bulk mean interior temperature, which varies with hole geometry and captures how effectively the holes draw heat away from the blade. Stress, creep strain, and fatigue damage are computed analytically from the T-field gradient (plane-strain thermoelastic, Norton-Bailey, Basquin respectively) — they do not require additional learned quantities.

**Scope.** This project optimises the *internal conduction* problem: placement and sizing of circular cooling holes in a 2D blade cross-section. Film-cooling fluid dynamics (coolant jet injection, gas-path boundary layer, conjugate heat transfer) require 3D CFD and are explicitly out of scope; they are identified as a Phase 2 extension.

**Prototype.** WorldSIM v0.2.0 implements the routing architecture, PyTorch Jacobi FD solver, and proxy NVIDIA backends. Mesh convergence (ΔT_mean = 0.3°C at 512×256) and a live CALPHAD result showing surrogate-without-physics-gate failure are documented in Appendix A.

---

## Methods

### Five-technology pipeline

```
Design parameters (38-D)
        │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ cuPyNumeric  — MAP-Elites archive (10,000 cells, 38-D vectors)    │
│ Population sampling, Gaussian mutation, UCB cell selection,        │
│ archive updates — all as cupynumeric GPU arrays, zero CPU          │
│ round-trips. 10k × 38 = 380k floats; NumPy host↔device at 512-   │
│ vector batches × 200k iters would add ~100 s overhead.            │
└────────────────────┬───────────────────────────────────────────────┘
                     │ Candidate batch (N=512)
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│ Warp  — FD thermal solver (tier-1 evaluator)                      │
│ @wp.kernel: 2D steady heat conduction, NACA-4412, 256×128 grid    │
│ wp.Tape autodiff  →  ∂T_mean/∂(hole positions, radii)            │
│ Gradients feed gradient-assisted archive update (Nilsson 2021)    │
│ Target: ~0.05 s/eval on H100 (6× vs RTX 3060 PyTorch FD)         │
└────────────────────┬───────────────────────────────────────────────┘
                     │ T-fields + quality labels
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│ PhysicsNeMo  — FNO2d surrogate                                    │
│ Input: geometry encoding (4-ch, 256×128)                          │
│ Output: T-field + MC-dropout uncertainty estimate                  │
│ Loss: data loss + λ=0.1 × PDE residual (∇²T = 0)                 │
│ Retrained incrementally each SAIL cycle (~30 min/cycle)           │
└────────────────────┬───────────────────────────────────────────────┘
                     │ Trained FNO weights
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│ TensorRT  — batched FNO inference for UCB acquisition only        │
│ ONNX (opset ≥17)  →  TRT FP16 engine                             │
│ <0.1 ms/batch (batch=1,024) → ~10M T-field evals/s               │
│ Archive logic (mutation, UCB, update) remains in cuPyNumeric      │
│ Fallback: torch.compile FP16 (~3M evals/s) if TRT export fails   │
└────────────────────┬───────────────────────────────────────────────┘
                     │ Top-50 UCB candidates per SAIL cycle
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│ Warp tier-1  →  CalculiX FEM tier-2 (CPU oracle)                 │
│ Top-10 archive cells → full thermoelastic FEM validation          │
└────────────────────┬───────────────────────────────────────────────┘
                     │ Archive + FNO uncertainty map
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│ Omniverse  — uncertainty-guided engineer-in-the-loop              │
│ USD blade with T-field prediction overlay + uncertainty heat map  │
│ Engineer selects high-uncertainty cells for CalculiX validation   │
│ (scientific function: reduces wasted FEM oracle budget)           │
│ Replicator generates geometry variants for FNO data augmentation  │
└────────────────────────────────────────────────────────────────────┘
```

| Technology | Key detail | Boundary |
|---|---|---|
| **Warp** | `@wp.kernel` Jacobi stencil; `wp.Tape` reverse-mode AD through iteration. Target 0.05 s/eval H100; minimum 0.10 s/eval | FD thermal solver only |
| **PhysicsNeMo** | FNO2d predicts T-field only; stress/creep/fatigue are analytical post-processes. PDE loss ∇²T=0. MC-dropout → uncertainty | Surrogate training |
| **cuPyNumeric** | `import cupynumeric as np` drop-in, no algorithm changes. Scientific necessity: 100×100 archive (vs prior 20×20); GPU-resident required for tractability | Archive operations |
| **TensorRT** | FNO forward passes during UCB acquisition only; archive logic stays in cuPyNumeric. <0.1 ms/batch-1024 → ~10M evals/s. Month 3 smoke test; `torch.compile` FP16 fallback if TRT export fails | Batch inference |
| **Omniverse** | Uncertainty overlay → engineer ranks 10,000 cells by FNO epistemic uncertainty to allocate 50-eval CalculiX budget. Replicator: 10k geometry variants for FNO augmentation | Validation guidance |

---

## Expected Results

| Month | Deliverable | Success criterion |
|---|---|---|
| 1 | Warp `@wp.kernel` FD solver | Max \|ΔT\| < 0.1°C vs PyTorch FD; ≥ 3× speedup on H100; `wp.Tape` gradient matches finite-difference check |
| 1 | cuPyNumeric 10k-cell archive | Correctness verified vs NumPy on 10k iterations; ≥ 2× wall-time reduction vs NumPy at 10,000 cells |
| 2 | Warp vs CalculiX validation | Mean \|ΔT\| < 50°C **and** peak \|ΔT\| < 80°C **and** hot-spot location error < 10 mm on 3 reference geometries (6, 10, 16 holes) |
| 3 | PhysicsNeMo FNO trained | Mean relative L² < 5%; T_mean error < 3°C on held-out configs; PDE residual < data-only baseline on 500 held-out configs |
| 3 | TensorRT smoke test | FNO exports to TRT FP16; output error vs PyTorch < 0.5% on 100 test cases; OR fallback timing documented |
| 4 | TensorRT acquisition engine | Batch throughput ≥ 1M candidates/s (TRT or fallback path); per-batch and per-sample timing reported |
| 5 | Full SAIL campaign | ≥ 70% of 10,000 cells filled; best T_mean ≥ 50°C below 6-hole baseline; top-10 CalculiX-validated using uncertainty guidance |
| 6 | Omniverse digital twin | Uncertainty overlay operational; Replicator augmentation improves FNO L² by ≥ 10% vs no augmentation |
| 6 | Open release + paper draft | pip-installable package; methods paper submitted |

**Risk mitigation:**

| Risk | Mitigation |
|---|---|
| TensorRT FNO export fails on spectral layers | Month 3 smoke test; `torch.compile` FP16 fallback (~3M evals/s) |
| PhysicsNeMo FNO L² > 5% at 15k training samples | Increase training corpus to 30k (add 10 H100-hours); add 6 Fourier modes |
| Warp speedup < 3× on H100 | Accept 3× minimum; 200k evaluations still feasible in ~3 H100-hours |
| cuPyNumeric slower than NumPy at 10k cells | Benchmark Month 1; fallback to CuPy (same API, different import) |

---

## Project Support Details

**Personnel:**  
- PI: [Faculty Name] — project oversight, physics validation, paper authorship  
- Co-I / Lead Developer: [Name] — full implementation, GPU kernel development, SAIL pipeline  

**Existing infrastructure:**  
- RTX 3060 (12 GB) workstation for development and proxy-backend testing  
- WorldSIM v0.2.0 codebase with routing architecture and proxy fallbacks for all five NVIDIA technologies  
- pycalphad v0.11.1 installed; CalculiX open-source FEM installed for oracle evaluation  

**Why H100 cloud (not local RTX 3060):** FNO training at batch=512 with PDE loss exceeds 12 GB VRAM; TRT 10.x requires CUDA ≥ 8.0; Warp Nsight profiling requires CUDA 9.0; 200k FD evals at 0.05 s/eval = 2.8 GPU-hours (17+ hours on RTX 3060, blocking iteration).

---

## Cloud Readiness

| Field | Value |
|---|---|
| **Cloud GPU hours requested** | 107 H100 80GB GPU-hours |
| **Number of concurrent GPUs** | 1 (maximum 2 for multi-GPU cuPyNumeric scaling benchmark) |
| **Cloud storage requested** | 500 GB (15k training T-fields × ~256×128×4 bytes ≈ 4 GB; Omniverse USD assets + Replicator variants ≈ 50 GB; checkpoints + logs ≈ 400 GB) |
| **Physical hardware requested** | None |

**Breakdown of 107 H100-hours:**

| Task | Hours | H100 requirement |
|---|---|---|
| Warp kernel development and Nsight profiling | 20 | CUDA 9.0; H100 architecture-specific kernel tuning |
| FNO training corpus: 15,000 Warp evaluations | 10 | Batch=512 exceeds RTX 3060 VRAM |
| PhysicsNeMo FNO training (10 cycles × 30 min) | 5 | PDE-loss training saturates 12 GB at batch=32 |
| TensorRT export, calibration, smoke test | 4 | TRT 10.x requires CUDA ≥ 8.0 |
| cuPyNumeric scaling benchmarks (incl. 2-GPU) | 5 | Multi-GPU test; device-bandwidth measurement |
| Full SAIL campaign (200k Warp evals + retraining) | 40 | Campaign runtime; 10 FNO retraining cycles |
| Omniverse rendering + Replicator augmentation | 15 | 10k geometry variants; USD real-time rendering |
| Ablations and paper figures | 8 | Archive size ablation; TRT vs fallback comparison |
| **Total** | **107** | |

*CalculiX FEM validation is CPU-based and runs on local workstation hardware — not included in H100 budget.*

---

## References

Mouret & Clune (2015). *Illuminating search spaces by mapping elites.* arXiv:1504.04909 · Gaier et al. (2017). *Surrogate-assisted illumination.* GECCO. doi:10.1145/3071178.3071282 · Li et al. (2021). *Fourier neural operator for parametric PDEs.* ICLR. arXiv:2010.08895 · Nilsson & Mouret (2021). *Accelerating MAP-Elites with gradient information.* arXiv:2103.00666 · MacKay et al. (2022). *NVIDIA Warp.* GTC. developer.nvidia.com/warp-framework · NVIDIA (2024). *PhysicsNeMo.* docs.nvidia.com/physicsnemo · NVIDIA (2024). *TensorRT.* developer.nvidia.com/tensorrt · NVIDIA (2024). *cuPyNumeric.* developer.nvidia.com/cupynumeric · NVIDIA (2024). *Omniverse.* developer.nvidia.com/omniverse

---

## Appendix A — Prototype Evidence *(excluded from 4-page limit)*

**Mesh convergence (Jacobi FD, RTX 3060, PyTorch):**

| Grid | Nodes | T_mean (°C) | T_peak (°C) | ΔT_mean vs prev |
|---|---|---|---|---|
| 64×32 | 2,048 | 1053.9 | 1300.0 | — |
| 128×64 | 8,192 | 1057.4 | 1300.0 | 3.5°C |
| 256×128 | 32,768 | 1059.1 | 1300.0 | 1.7°C |
| 512×256 | 131,072 | 1058.8 | 1300.0 | **0.3°C** |

T_peak = 1300°C is constant across all grids because it is fixed by the **Dirichlet BC** on the outer hot-gas wall (T = 1300°C prescribed; not computed). T_peak is therefore not a useful optimization metric — it cannot be changed by hole placement. The design-controllable quantity is **T_mean**, which decreases as holes are placed more effectively. Convergence manifests in T_mean (3.5°C → 1.7°C → 0.3°C per refinement) and in thermal gradients near cooling-hole walls. A future mesh study will report hole-wall gradient convergence as the primary design-relevant metric. Production grid: 256×128.

**CALPHAD finding — motivation for physics-gated evaluation:**  
Running the MAP-Elites top alloy candidate (Ni=0.694, Cr=0.183, Al=0.122, simplified Al-Cr-Ni ternary system) through pycalphad v0.11.1 with bundled alcrni.tdb shows 100% single-phase γ' (L12_FCC) from 800–1350°C. A functional Ni-base superalloy requires two-phase γ + γ'. The heuristic quality function rewarded high Ni/(Al+Ti) without a phase-fraction constraint — yielding a high score for a microstructurally wrong result. This example demonstrates the failure mode that motivates the tiered evaluation architecture in this proposal: a fast surrogate without a physics gate optimises a proxy objective. The alloy system here is a simplified proxy (not the target Ni-Co-Cr-Al-Ti-Ta superalloy); the lesson — not the specific composition — carries over to the blade cooling pipeline.

**Existing routing stubs (not working NVIDIA implementations):**  
`simlab/engines/materials/nvidia_backends.py` contains `WarpBackend`, `PhysicsNeMoBackend`, `CuPyBackend` classes, each with an `.available` flag and proxy fallback. `dispatcher.py` routes `("nvidia", "nemo_train_fno")` and `("nvidia", "cupy_diffusion_sweep")`. These are integration scaffolding — they confirm the routing architecture and fail gracefully when libraries are absent. The grant replaces each proxy with a live GPU backend.

---

## Appendix B — CV *(required; attach separately per template instructions)*

[Attach PI CV and Co-I CV as separate pages per NVIDIA template requirements.]
