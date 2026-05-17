# Preliminary Screening Note: Ni57Cr15Co9Al10Ti3Ta7

**Platform:** WorldSIM v0.2 — MaxOSL AI Research  
**Date:** 2026-05-16  
**GPU:** NVIDIA GeForce RTX 3060 (12 GB, sm_86)  
**Status: Concept-level screening only. No claim of validated superalloy behavior.**

---

## What This Document Is

This is an automated composition screening note, not a materials discovery paper.  
The pipeline generated a Ni-rich alloy candidate worth investigating further.  
It does **not** prove phase stability, γ' volume fraction, oxidation resistance,  
heat treatment suitability, or any other materials property.

Every number is a hypothesis for experimental or high-fidelity computational  
(CALPHAD, DFT) follow-up — not validated data.

---

## 1. Discovery Workflow

```
Latin Hypercube Sampling (40 compositions, seed 113)
    → ALCHEMI GPU batch screen (LJ energy ranking, a = 3.57 Å, FCC)
        → Combined scoring (LJ energy + heuristic property estimates)
            → Atomistic relaxation + heat treatment + phase-field characterisation
```

### 1.1 Composition Space

| Element | Min (at.%) | Max (at.%) | Role |
|---------|-----------|-----------|------|
| Ni | 50 | 70 | Matrix + γ' former |
| Cr | 5 | 15 | Oxidation resistance (Cr₂O₃) |
| Co | 3 | 12 | Solid solution, lowers γ' solvus slightly |
| Al | 5 | 12 | γ' former (Ni₃Al), Al₂O₃ scale |
| Ti | 1 | 6 | γ' former (substitutes Al site), modest strengthening |
| Ta | 1 | 8 | γ' former (heavy partitioner, misfit strengthening) |

Rationale for this space: Ta is a key element in 2nd–4th generation single-crystal superalloys
(CMSX-2/4, René N5, TMS series). It partitions strongly to γ', increases lattice misfit (δ),
and strengthens the γ'/γ interface. This search excludes refractory TCP-drivers (Mo, W, Re)
to test whether a "clean" composition still competes on screening scores.

40 compositions sampled with Latin Hypercube (scipy.stats.qmc, seed 113).  
Iterative clip-and-renormalise (≤10 passes) enforces element bounds after normalisation.

### 1.2 Bounds Verification

| Element | Min sampled | Max sampled | Bounded |
|---------|-------------|-------------|---------|
| Ni | 54.9% | 70.0% | ✓ |
| Cr | 5.2% | 15.0% | ✓ |
| Co | 3.4% | 12.0% | ✓ |
| Al | 5.3% | 12.0% | ✓ |
| Ti | 1.2% | 6.0% | ✓ |
| Ta | 1.1% | 8.0% | ✓ |

### 1.3 Scoring

```
score = 0.30 × (−E_LJ / 10.0)                 (relative ranking)
      + 0.35 × oxidation_resistance_score_raw   (heuristic)
      + 0.35 × phase_stability_score_raw         (heuristic)
```

Combined score: **1.045** (top of 40 candidates).

---

## 2. Candidate Composition

**Ni57Cr15Co9Al10Ti3Ta7** (at.%)

| Element | at. % |
|---------|-------|
| Ni | 56.7 |
| Cr | 14.9 |
| Al | 9.9 |
| Co | 9.3 |
| Ta | 6.6 |
| Ti | 2.5 |

This composition is in the same broad neighbourhood as CMSX-4 (Ni-61.7Cr-6.4Co-9Al-5.6Ti-1Ta-6.3W-2Re-0.6Mo-0.1Hf at.%) and René N5 (Ni-62Cr-7Co-8Al-6.2Ta-6.5W-3Re-0.15Hf-0.2C at.%), but without the refractory heavy elements (W, Re, Mo) and carbide-forming additions (Hf, C, B).

---

## 3. Heuristic Property Estimates

These are composition-weighted mixing rules, **not** CALPHAD or DFT calculations.

| Property | Value | Basis |
|----------|-------|-------|
| Density | 8.44 g/cm³ | Rule of mixtures |
| Elastic modulus | 197 GPa | Voigt average |
| Yield strength proxy | 571 MPa | Mismatch hardening heuristic |
| Solidus proxy | 1534 °C | Mismatch-corrected liquidus |
| Phase stability score (raw) | 1.058 | Empirical formula — not CALPHAD |
| Oxidation resistance score (raw) | 1.227 | Empirical formula |
| Configurational entropy | 11.04 J/mol·K (1.33 R) | −R Σ xᵢ ln xᵢ |
| VEC | 8.14 | Composition-weighted valence electrons |
| Ni/(Al+Ti) | 4.57 | Stoichiometry indicator |
| δ mismatch | 0.0614 | Normalised atomic radius variance |

### Notable indicators

**VEC = 8.14**  
Above the empirical FCC stability threshold of 8.0. This is a favourable indicator for FCC
phase stability in multi-principal-element alloys, though it does not guarantee it for a
conventional Ni superalloy composition. The prediction should be validated with CALPHAD.

**Ni/(Al+Ti) = 4.57**  
Well above the Ni₃(Al,Ti) stoichiometric ratio of 3.0. This means there is more than enough
Ni to saturate all Al+Ti into γ'. Excess Ni forms the γ matrix. This is a key difference
from the previous Ni33Cr23Fe15Al11Co9Ti9 candidate (Ni/(Al+Ti) = 1.67), where Al+Ti excess
was a serious concern. Ta is not included in the denominator here — it also partitions to γ',
so the effective ratio is even more favourable. CALPHAD would quantify the actual γ' volume
fraction.

**No Mo, W, or Re**  
TCP phase risk (σ, μ, Laves) is assessed as NONE by the proxy heuristic. Mo+W = 0.00%.
This removes the most critical composition risk from the Co41Cr16Ni16Mo12Al8W7 candidate.

**Ta = 6.6%**  
Ta is a strong γ' partitioner (k_Ta^γ'/γ ~ 5–10 in commercial alloys). It increases the
γ'/γ lattice misfit (δ), which is the primary driver of creep resistance in single-crystal
superalloys. At 6.6%, it is within the range used in first-generation alloys (CMSX-2: 6%,
MAR-M200: 0%). It also slows oxidation kinetics by reducing interdiffusion.

**S_conf = 1.33 R**  
Below the 1.5R high-entropy threshold. This is a conventional alloy regime — entropy is not
the dominant phase-stability driver. Thermodynamics (CALPHAD) governs phase stability.

---

## 4. Atomistic Screen (CPU Proxy)

**Cell assignment (largest-remainder, seed 55, 32 atoms):**

| Species | Count |
|---------|-------|
| Ni | 18 |
| Cr | 5 |
| Co | 3 |
| Al | 3 |
| Ta | 2 |
| Ti | 1 |

| Result | Value |
|--------|-------|
| Backend | numpy_cpu_proxy |
| Relaxation | LJ volume proxy |
| Converged | Yes |
| LJ energy/atom | −3.00 eV (proxy) |
| Max force | 0.00 eV/Å (proxy) |

**Limitation:** The CPU proxy does not perform real FIRE2 geometry relaxation.
nvalchemiops+CUDA is required for the GPU FIRE2+LJ path.
Use `alchemi_mlip` with `potential_path=` pointing to a MACE-MP-0 model for
~DFT-quality relaxation without a full DFT code.

---

## 5. Degradation Heuristics (1050°C, 10,000 hours, dry air)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Parabolic kp | 4.55 × 10⁻¹⁴ m²/s | Higher than Co41 (2.1×10⁻¹⁴) — temperature effect |
| Oxide thickness @ 10 kh | 1279 µm | **Severe** proxy — parabolic model without protective scale credit |
| Mass gain proxy | 703 mg/cm² | Heavy — same caveat |
| Corrosion risk (Cl⁻) | 0.073 | Low (no chloride) |
| H-embrittlement risk | 0.297 | Moderate (Ti = 2.5%, no H₂) |

**Critical caveat on oxide thickness:** The parabolic proxy overestimates oxidation because it
does not account for a self-healing protective Al₂O₃ or Cr₂O₃ scale. With Cr (14.9%) + Al
(9.9%), this composition is in the range expected to form a mixed Al₂O₃/Cr₂O₃ scale in dry
air. CMSX-4 (similar Cr+Al) maintains 10–50 µm oxide at 1050°C in 1000 h laboratory tests.
TGA testing and post-exposure SEM/EPMA are required to characterise the actual scale.

**Ta and oxidation:** Ta₂O₅ has been reported to form at grain boundaries and can initially
degrade oxidation performance. In single-crystal alloys without grain boundaries this risk is
reduced. At 6.6 at.% Ta, its effect on the oxide scale is composition-specific and requires
experimental verification.

---

## 6. Heat Treatment Analysis

All kinetic parameters are representative/generic for Ni superalloys, **not fitted to this
specific composition**. Solvus temperatures are assumed, not calculated.

### 6.1 TTT Diagram (γ' precipitation, screening model)

Parameters:

| Parameter | Value | Basis |
|-----------|-------|-------|
| Estimated γ' solvus | 1250°C (1523 K) | Assumed (typical range 1180–1320°C for high-γ' alloys) |
| T_low | 650°C (923 K) | — |
| T_nose | 1100°C (1373 K) | Assumed |
| t_nose | 80 s | Representative, faster than Co-based |
| Q | 180 kJ/mol | Generic Ni-base diffusion activation energy |

| Condition | t_start (1%) | t_finish (99%) |
|-----------|-------------|--------------|
| At nose (1092°C) | 9.9 s | 211 s |
| At 650°C | 35.7 s | — |

**What this means:** Nose at ~10 s implies extremely fast γ' precipitation kinetics at
1100°C. The critical quench rate to suppress all precipitation during cooling from the
solution anneal is very high — consistent with the need for air or forced-air quench in
conventional Ni superalloys. The actual nose position and t_nose must be measured by DSC
or CALPHAD.

### 6.2 Grain Growth (1050°C, 4 hours)

Parameters: K₀ = 5×10⁻⁵ m²/s, Q = 270 kJ/mol (Ni alloy grain growth, generic)

| Parameter | Value |
|-----------|-------|
| Initial grain size | 100 µm |
| Final grain size | 100.08 µm |
| Growth factor | 1.0008 |

Negligible grain growth at 1050°C for 4 hours. Consistent with strong γ' pinning of grain
boundaries (the Zener-pinning effect). In a single-crystal alloy this parameter is irrelevant;
in a polycrystalline version it supports fine-grained microstructure stability during service.

### 6.3 Precipitation Kinetics (JMAK, 850°C, 2 hours)

Parameters: τ_ref = 500 s at 900°C, Q_nucleation = 200 kJ/mol, Q_growth = 175 kJ/mol, n = 3

| Parameter | Value |
|-----------|-------|
| Incubation time at 850°C | 1246 s (~21 min) |
| Precipitate fraction at 2h | 1.00 (fully transformed) |
| Avrami exponent | 3.0 |

The 1246 s incubation period at 850°C reflects slower kinetics than the TTT nose — consistent
with diffusion-limited nucleation at lower supersaturation. Full transformation within 2 hours
indicates adequate aging time at this temperature for a standard 2-step aging schedule.

**Proposed screening heat treatment schedule (starting point only):**
1. Solution anneal: 1260°C / 4 h → air quench  
2. Primary age: 1100°C / 2 h → air cool  
3. Secondary age: 850°C / 24 h → air cool

Caveat: solvus, nose, and aging temperatures are assumed. All three require CALPHAD + DSC
validation before use.

---

## 7. Phase Field (Allen-Cahn, CPU)

2D Allen-Cahn on 64×64 grid, 1000 steps.

| Parameter | Value |
|-----------|-------|
| Backend | numpy_cpu |
| Steps | 1000 |
| M, κ, dt | 1.0, 0.5, 0.05 |
| Final φ mean | 0.476 |
| Interface | Sharp, bimodal |

Generic phase-separation demonstration. No connection to actual γ/γ' microstructure —
same caveat as previous candidates. Real phase-field modelling requires composition-dependent
free energies, interfacial energy σ_γγ' ~ 10–30 mJ/m², and lattice misfit δ (estimated
0.1–0.5% for this composition, to be confirmed by XRD).

---

## 8. High-Fidelity Backend Assessment

| Backend | Status | What it would add |
|---------|--------|------------------|
| pycalphad + TCNI9 | Not configured | Phase diagram, γ' solvus, actual volume fraction |
| DFT (VASP/QE) | Not configured (EMT available) | Formation energy, convex hull position |
| MACE-MP-0 | Not configured | ~DFT-quality relaxation, batch screening |
| SevenNet | Not configured | Alternative MLIP validation |

With the new `calphad_phase_equilibrium`, `tcp_phase_check`, `dft_formation_energy`,
and `mlip_batch_screen` routes now available, these calculations can be triggered via:

```python
# TCP and phase equilibrium (requires pycalphad + TDB)
core.simulate_materials("calphad_phase_equilibrium", {"composition": comp, "temperature_K": 1323.15})
core.simulate_materials("tcp_phase_check", {"composition": comp})

# MLIP batch screen (requires MACE model path in materials_config.yaml)
core.simulate_materials("mlip_batch_screen", {"compositions": [comp, ...], "n_atoms": 32})

# DFT formation energy (EMT available now; VASP/QE for real DFT)
core.simulate_materials("dft_formation_energy", {"species": [...], ...})
```

---

## 9. Comparison with Previous Candidates

| Property | Ni33Cr23Fe15Al11Co9Ti9 | Co41Cr16Ni16Mo12Al8W7 | **Ni57Cr15Co9Al10Ti3Ta7** |
|----------|----------------------|----------------------|--------------------------|
| Alloy family | Ni-rich | Co-rich | **Ni superalloy** |
| VEC | 7.40 | 7.65 | **8.14** ← FCC-favourable |
| Ni/(Al+Ti) | 1.67 ← concern | 2.08 | **4.57** ← above γ' stoich |
| TCP risk | None | MODERATE (Mo+W) | **NONE** |
| MoO₃ risk | None | CRITICAL | **NONE** |
| Density | 7.29 g/cm³ | 9.01 g/cm³ | 8.44 g/cm³ |
| Solidus proxy | 1486°C | 1736°C | 1534°C |
| Modulus | 199 GPa | 236 GPa | 197 GPa |
| Yield proxy | 373 MPa | 336 MPa | **571 MPa** |
| Key concern | Al+Ti excess | TCP + MoO₃ | Ta₂O₅ at GBs, no W/Re/Hf |

Ni57Cr15Co9Al10Ti3Ta7 has the most favourable VEC, the best Ni/(Al+Ti) ratio, and zero TCP
risk. The main limitation is the absence of W, Re, and Hf — elements critical for creep
resistance in real single-crystal superalloys. Whether Ta alone provides sufficient
strengthening is the central open question.

---

## 10. Priority Validation Steps

1. **CALPHAD phase equilibrium** (Thermo-Calc TCNI9)  
   Calculate γ' volume fraction at 850–1100°C, actual γ' solvus, and check for unwanted
   phases (α-Cr, σ, δ-Ni₃Nb). This is the highest-priority next step.

2. **γ'/γ lattice misfit calculation**  
   Compute δ = 2(a_γ' − a_γ)/(a_γ' + a_γ) from CALPHAD or DFT. Values in the range
   −0.1% to −0.5% are optimal for creep in single crystals.

3. **MACE-MP-0 / SevenNet MLIP screening**  
   Configure `mace_model_path` in `materials_config.yaml` and run `mlip_batch_screen`
   to re-rank the 40 candidates with ~DFT-quality energies, replacing the LJ proxy.

4. **DFT SQS formation energy** (VASP / QE)  
   Generate a 32-atom SQS (`generate_sqs`), relax with VASP PBE, and check convex hull
   position against Ni₃Al, Ni₃Ta, NiCrAl intermetallics.

5. **Arc-melting + XRD + SEM**  
   Synthesise a button ingot. Phase identification by XRD; γ/γ' microstructure by SEM/EBSD
   after heat treatment.

6. **TGA oxidation testing** (900–1100°C, synthetic air)  
   Measure actual mass gain and characterise oxide scale composition by EPMA.
   Specifically check for Ta₂O₅ formation at the surface.

7. **Creep testing**  
   The absence of W, Re, and Mo means creep strength is unknown relative to 1st–2nd gen
   single-crystal alloys. Larson-Miller testing at 980°C / 150 MPa vs CMSX-4 baseline.

---

## Appendix: Pipeline Steps

| Step | Method | Key Parameters | Output |
|------|--------|---------------|--------|
| Sampling | LHS seed=113, n=40 | 6 elements | 40 compositions |
| Bounds fix | Iterative clip-renorm | 10 passes | Verified bounds |
| LJ batch | `ALCHEMIBackend.batch_alloy_screen` | a=3.57 Å, FCC | LJ energies |
| Scoring | LJ + heuristic | 3-component weighted | Rank order |
| Properties | `alloy_property_prediction` | T=1323 K | Density, VEC, etc. |
| Degradation | `degradation_prediction` | T=1323 K, 10 kh | Oxide proxy |
| TTT | `PhaseTransformationEngine.ttt_diagram` | solvus=1523 K | C-curve |
| Grain growth | `grain_growth` | 1323 K, 4h | 100.08 µm |
| Precipitation | `precipitation_kinetics` | 1123 K, 2h | X=1.0 |
| Phase field | `WarpBackend.run_allen_cahn_field` | 64×64, 1000 steps | Bimodal φ |
| New backends | CALPHAD/DFT/MLIP proxy | Automatic fallback | Proxy results |

---

*Generated by WorldSIM v0.2 — MaxOSL AI Research*  
*Preliminary automated screening note. All values require experimental or high-fidelity computational validation before any material selection decision.*
