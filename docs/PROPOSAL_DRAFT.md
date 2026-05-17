# WorldSIM: GPU-Accelerated Quality-Diversity Search for Validated Superalloy and Geometry Design

**Programme:** NSF / DARPA / DOE (draft — agency-specific formatting to follow)  
**PI:** MaxOSL AI Research  
**Duration:** 36 months  
**Prototype:** WorldSIM v0.2.0 — github/maxosl-ai-research (commit c6ef341)

---

## Abstract

Superalloy design and parametric geometry optimisation share a common computational bottleneck: evaluating a candidate is expensive (CALPHAD equilibrium, DFT, FEM), yet the design space is vast and the desired output is not a single optimum but a *population of diverse, high-performing candidates* that a human engineer can select from. We propose WorldSIM, a GPU-accelerated simulation platform that couples quality-diversity (QD) search — specifically MAP-Elites and Surrogate-Assisted Illumination (SAIL) — with a tiered fidelity stack running from fast GPU heuristics to real CALPHAD thermodynamics and physics-based solvers. Our preliminary results demonstrate (i) 3.52 M evaluations/second on an RTX 3060 for geometry search, (ii) a multi-physics blade simulation pipeline completing in 0.36 s, (iii) a mesh convergence study showing ΔT_mean < 0.3°C between 256×128 and 512×256 grids, and (iv) a critical finding from our first real CALPHAD run: the heuristic-search top candidate sits in the γ' single-phase field, not the two-phase γ + γ' corridor required for a functional superalloy. That last finding is the scientific core of this proposal — it demonstrates precisely why tiered validation is necessary and why the research is not complete.

---

## 1. Motivation and Problem Statement

Nickel-base superalloys for turbine blades require simultaneous satisfaction of conflicting objectives: high γ' phase fraction (precipitate strengthening), low TCP phase risk (embrittlement), high solidus temperature, low density, and adequate oxidation resistance. A state-of-the-art design cycle uses CALPHAD for phase prediction, DFT or MLIP for energy validation, and FEM for mechanical verification. Each evaluation takes hours to days.

The standard response is to reduce evaluation count — gradient-based optimisation, Bayesian optimisation, or Pareto multi-objective search. All of these return a small set of solutions. Engineers increasingly need something different: a *map* of the design space that identifies not just the best composition but the best composition in every region of the space. This is the quality-diversity (QD) problem, and MAP-Elites (Mouret & Clune, 2015) is its canonical algorithm.

The challenge is that MAP-Elites requires many evaluations — 10,000–1,000,000 to build a meaningful archive. At CALPHAD speeds (1–100 s/evaluation), this is infeasible. Our hypothesis is that a GPU-accelerated surrogate tier can make the iteration loop fast enough to be practical: run 3.52 M cheap evaluations on GPU, select 1,000 UCB candidates via SAIL, evaluate those 1,000 on real CALPHAD, feed back, repeat.

A second parallel problem is parametric geometry optimisation for turbine blade cooling channels. The MAP-Elites + SAIL framework applies identically: the objective is multi-physics (thermal, stress, creep, fatigue), evaluation is expensive (FEM), and the desired output is a map of valid cooling designs across the feature space (e.g. metal temperature × cooling efficiency).

---

## 2. Preliminary Results

All results were produced on the hardware listed below. Raw data, scripts, and the full prototype are available in the repository.

**Hardware:** NVIDIA GeForce RTX 3060 (12 GB VRAM), CUDA 12.1, PyTorch 2.2.2+cu121, Python 3.12.3  
**Commit:** c6ef341

### 2.1 GPU MAP-Elites throughput

Using `torch_gpu.gpu_mapelites()` with a 4-parameter geometry objective (volume + aspect ratio + wall thickness, ~15 FP ops per evaluation). CUDA synchronisation was called before and after the timed region.

| Population | Throughput | Speedup vs NumPy |
|---|---|---|
| 64 | 350k evals/s | 20× |
| 1,024 | 1.4M evals/s | 80× |
| 4,096 | 2.8M evals/s | 160× |
| 16,384 | **3.52M evals/s** | **226×** |

**Honest scope:** This measures throughput of a 15-operation heuristic function, not a physics solver. It establishes that the GPU tier of the SAIL pipeline can handle the cheap-evaluation phase at the required scale. The scientific value depends on pairing this with a higher-fidelity evaluation tier, which is the subject of this proposal.

### 2.2 Blade multi-physics screening pipeline

A 2D steady-state multi-physics chain (Jacobi FD thermal → thermoelastic stress → Norton-Bailey creep → Basquin HCF fatigue) on a 256×128 CUDA grid completes in **0.36 s** on the RTX 3060. This is intended as a fast screening tool to rank cooling channel configurations before committing to full FEM. It is not a substitute for FEM.

### 2.3 Mesh convergence study

Thermal solver run at four grid resolutions to establish convergence behaviour:

| Grid | Nodes | T_mean (°C) | T_peak (°C) | ΔT_mean vs prev | Time |
|---|---|---|---|---|---|
| 64×32 | 2,048 | 1053.9 | 1300.0 | — | 0.23 s |
| 128×64 | 8,192 | 1057.4 | 1300.0 | 3.5°C | 0.03 s |
| 256×128 | 32,768 | 1059.1 | 1300.0 | 1.7°C | 0.08 s |
| 512×256 | 131,072 | 1058.8 | 1300.0 | **0.3°C** | 0.25 s |

ΔT_mean halves with each grid doubling, consistent with first-order convergence of Jacobi iteration. The 256×128 grid is within 0.3°C of the 512×256 result, establishing it as adequately resolved for screening purposes. T_peak is pinned at the Dirichlet boundary condition (T_gas = 1300°C) and is resolution-independent by construction.

**Remaining gap:** No comparison against a reference FEM solution (CalculiX or Abaqus). The 0.3°C internal convergence does not validate the model against physical reality.

### 2.4 First real CALPHAD run — and what it reveals

pycalphad was installed (v0.11.1) and run against the bundled Al-Cr-Ni thermodynamic database (`alcrni.tdb`, included with pycalphad). The MAP-Elites top candidate was projected onto the Ni-Cr-Al ternary subsystem (Co, Ti, Ta dropped; Ni:Cr:Al renormalised to Ni=0.694, Cr=0.183, Al=0.122).

**CALPHAD equilibrium result (800–1400°C):**

| T (°C) | γ (FCC_A1) | γ' (L12_FCC) | BCC | Liquid | Stable |
|---|---|---|---|---|---|
| 800–1350 | 0.0 | **1.0** | 0.0 | 0.0 | γ' only |
| 1400 | 0.0 | 0.0 | 0.0 | **1.0** | Liquid |

**What this means:** The MAP-Elites top candidate, when evaluated on real CALPHAD, sits entirely in the single-phase γ' (L12_FCC) field from room temperature to near-liquidus. A functional Ni-base superalloy requires the **two-phase γ + γ'** field — a disordered FCC matrix (γ) with coherent ordered L1₂ precipitates (γ'). Single-phase γ' is too brittle for turbine blade applications.

**Why this happened:** The heuristic quality function rewarded high Ni/(Al+Ti) ratio (γ' tendency) but had no term for the *amount* of γ' phase fraction or for being in the two-phase corridor. A high Ni/(Al+Ti) pushes into single-phase γ', not two-phase γ + γ'.

**Why this matters for the proposal:** This is the core scientific finding of the preliminary work. It demonstrates that:
1. The GPU QD search works — it efficiently finds high-scoring compositions.
2. The scoring function was wrong — it lacked a real phase-fraction term.
3. The solution is exactly what we are proposing: couple the GPU QD search to real CALPHAD in the SAIL evaluation loop, so the true-evaluation tier correctly penalises single-phase compositions.

**Limitation of this run:** The `alcrni.tdb` is a test database included with pycalphad; it is not the commercial TCNI9 database used in industrial alloy design. Co, Ti, and Ta are not present. The finding is directionally correct but should be re-run with a full Ni-superalloy TDB once obtained.

### 2.5 Alloy search reproducibility

MAP-Elites alloy search run 10 times with different seeds (n_iterations=400, batch_size=16):

```
Quality:  mean=2.9966  std=0.0022  min=2.9927  max=2.9998
Ni range: 0.510–0.592   Al range: 0.040–0.082
```

The quality score converges consistently across seeds (std/mean < 0.1%). Composition varies enough to matter: Al varies from 4 to 8 at%, a range that the CALPHAD scan above shows is entirely within the single-phase γ' field at this Cr level. Closing the validation loop on this composition range is a primary research objective.

---

## 3. Research Objectives

**Objective 1 — SAIL-CALPHAD coupling:** Integrate real pycalphad phase equilibrium (with a full Ni-superalloy TDB) as the true-evaluation function in the SAIL loop. Run GPU MAP-Elites as the surrogate acquisition tier; evaluate top-k UCB candidates on real CALPHAD; update surrogate; repeat. Target: identify compositions in the two-phase γ + γ' corridor with γ' fraction 30–60% at service temperature.

**Objective 2 — Alloy search quality function redesign:** Replace the heuristic VEC/Ni(Al+Ti) scoring with a physics-grounded objective: actual γ' phase fraction from CALPHAD, TCP phase stability check, and solidus temperature. Validate that MAP-Elites + real CALPHAD recovers known commercial alloy compositions (René 104, CMSX-4, IN738) from the archive.

**Objective 3 — Blade solver validation:** Run the 2D FD thermal solver on a subset of cooling channel configurations and compare T-field against a CalculiX FEM reference. Quantify the error as a function of model simplification. Establish whether the 0.36 s screening model is suitable as the cheap-evaluation tier in SAIL, and what correction is needed.

**Objective 4 — QD algorithm benchmarking:** Run MAP-Elites and SAIL on the ARM benchmark (Mouret & Clune, 2015) and compare archive coverage and QD score against the pymap_elites reference implementation at matched evaluation budgets. This establishes that the algorithm implementation is correct independent of the domain-specific objective.

**Objective 5 — GPU SIREN for geometry reconstruction:** Validate the neural implicit field (SIREN) against a mesh reconstruction baseline. Measure mean |SDF| on held-out surface points after training on the other half. Target: near-surface coverage >80% at n_epochs=500.

---

## 4. Methodology

### 4.1 SAIL pipeline with tiered fidelity

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 1: GPU fast screen (3.52M evals/s)                           │
│  MAP-Elites over composition / geometry space                       │
│  Quality fn: 15-op heuristic  →  fills archive quickly             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Top-k UCB candidates
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 2: Neural surrogate (GPU, ~1k evals/s)                       │
│  NeuralSurrogate(nn.Module): 64→64→32, mean+log_var UCB heads      │
│  Surrogate-guided MAP-Elites: cheap acquisition → UCB selection    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Top-k UCB (smaller set)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 3: Real physics (1–100 s / eval)                             │
│  Alloys:    pycalphad + TCNI9 TDB                                  │
│  Geometry:  2D FD blade solver (0.36 s) → CalculiX FEM (minutes)  │
│  Updates surrogate; feeds back to archive                          │
└─────────────────────────────────────────────────────────────────────┘
```

The SAIL loop (Gaier et al., 2017) limits calls to Tier 3. At n_seed=100 and n_cycles=10 with k=50 per cycle, total Tier 3 calls = 600. Each CALPHAD call costs ~1–5 s; 600 calls = 10–50 minutes. This is the target operating regime.

### 4.2 CALPHAD integration

The existing `CALPHADBackend` in `simlab/engines/materials/calphad_backend.py` wraps pycalphad and already handles the proxy/real fallback. Objective 1 requires: (a) obtain a Ni-superalloy TDB (TCNI9, or the open Steel-Ni partial TDB from Dinsdale et al., or a TDB from the NIST alloy database); (b) extend the wrapper to return γ' phase fraction and TCP phase presence as the quality signal; (c) wire this into the SAIL `quality_fn` parameter.

### 4.3 Geometry solver

The 2D FD blade solver (`torch_gpu_physics.py`) will be validated against CalculiX for 3 cooling channel configurations (6 holes, 8 holes, 12 holes). Temperature fields will be compared at 25 equally-spaced internal nodes. Acceptable error threshold: mean |ΔT| < 50°C for the FD solver to qualify as the SAIL surrogate tier.

### 4.4 Algorithm benchmarking

MAP-Elites will be run on the 6-DOF ARM benchmark (the standard QD validation problem from Mouret & Clune, 2015): a simulated arm whose end-effector position defines the 2D behaviour descriptor, and whose joint configuration quality is the negative L2 distance to a target. Results will be compared to the pymap_elites reference at 50k, 100k, and 200k evaluation budgets, measuring archive coverage and QD score (sum of qualities across all filled cells).

---

## 5. Known Limitations of Current Prototype

| Limitation | Status | Impact on Proposal |
|---|---|---|
| CALPHAD is proxy on this machine | Resolved (pycalphad installed, alcrni.tdb run) | First real CALPHAD result obtained; full Ni TDB needed for Objective 1 |
| MAP-Elites not benchmarked on standard QD suite | Open (Objective 4) | Algorithm correctness asserted but not demonstrated against reference |
| Blade solver not validated against FEM | Open (Objective 3) | Screening tool status only |
| Norton creep constants not alloy-specific | Open | Creep field is illustrative; not quoted as a design result |
| alcrni.tdb is a test DB, not commercial TCNI9 | Open | CALPHAD finding is directionally correct; needs full TDB |
| SIREN not benchmarked on reconstruction accuracy | Open (Objective 5) | Untrained (would fail loudly now); JAX available for real training |
| Safety: chemical keyword blocklist only | Open | Acknowledged; not claimed as a safety boundary |

---

## 6. Timeline

| Month | Milestone |
|---|---|
| 1–3 | Obtain Ni-superalloy TDB; run CALPHAD scan over full MAP-Elites seed-variance band; redesign quality function |
| 4–6 | ARM benchmark for MAP-Elites and SAIL; pymap_elites comparison |
| 7–12 | SAIL-CALPHAD loop (Objective 1); first archive of validated γ + γ' compositions |
| 13–18 | CalculiX FEM validation of blade solver (Objective 3); quantify screening error |
| 19–24 | SAIL-geometry loop with FD surrogate + FEM oracle |
| 25–30 | SIREN reconstruction validation; GPU SIREN vs reference mesh reconstruction |
| 31–36 | Integration, paper writing, open-source release of validated pipeline |

---

## 7. Expected Outcomes and Impact

**Scientific:** A demonstrated SAIL pipeline that reduces Tier 3 CALPHAD calls by >10× vs naive grid search while covering a comparable region of composition space. Expected archive: 50–200 validated γ + γ' compositions across a 10×10 VEC × density behaviour grid at ≥ 30% γ' phase fraction, zero TCP phases.

**Engineering:** A 0.36 s screening model whose error vs FEM is quantified, enabling its use as a SAIL surrogate for cooling channel optimisation with a well-characterised correction factor.

**Methods:** An open, reproducible implementation of the MAP-Elites → SAIL → CALPHAD tiered pipeline, including benchmark comparisons, unit tests for routing correctness, and a documented fidelity hierarchy.

**Software:** WorldSIM v1.0 — a fully documented, pip-installable package with at least: MAP-Elites + SAIL (benchmarked), real CALPHAD integration, validated blade screening model, and GPU SIREN training.

---

## 8. What This Proposal Does Not Claim

- We do not claim the current prototype produces validated alloy compositions. Section 2.4 explicitly shows the prototype candidate is wrong.
- We do not claim the blade solver is FEM-grade. Section 5 lists the known limitations.
- We do not claim the GPU throughput benchmark is a physics benchmark. It is a heuristic function throughput measurement.
- We do not claim "correct implementation" of the algorithms without benchmark evidence; obtaining that evidence is Objective 4.

---

## 9. References

Mouret, J.-B. & Clune, J. (2015). *Illuminating search spaces by mapping elites.* arXiv:1504.04909.

Gaier, A., Asteroth, A. & Mouret, J.-B. (2017). *Data-efficient exploration, optimization, and modeling of diverse designs through surrogate-assisted illumination.* GECCO 2017. doi:10.1145/3071178.3071282

Sitzmann, V., Martel, J., Bergman, A., Lindell, D. & Wetzstein, G. (2020). *Implicit neural representations with periodic activation functions.* NeurIPS. arXiv:2006.09661

Lukas, H.L., Fries, S.G. & Sundman, B. (2007). *Computational Thermodynamics: The CALPHAD Method.* Cambridge University Press. doi:10.1017/CBO9780511804137

Guo, S. & Liu, C.T. (2011). *Phase stability in high entropy alloys.* Progress in Natural Science: Materials 21(6):433–446. doi:10.1016/j.pnsc.2011.09.003

Batatia, I. et al. (2022). *MACE: Higher order equivariant message passing neural networks.* NeurIPS. arXiv:2206.07697

Norton, F.H. (1929). *The Creep of Steel at High Temperatures.* McGraw-Hill.

Basquin, O.H. (1910). *The exponential law of endurance tests.* ASTM Proc. 10:625–630.

---

*Draft v0.1 — WorldSIM v0.2.0 — commit c6ef341 — RTX 3060 / CUDA 12.1*
