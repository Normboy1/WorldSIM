# Preliminary Screening Note: Co41Cr16Ni16Mo12Al8W7

**Platform:** WorldSIM v0.2 — MaxOSL AI Research  
**Date:** 2026-05-15  
**GPU:** NVIDIA GeForce RTX 3060 (12 GB, sm_86)  
**Status: Concept-level screening only. No claim of validated superalloy behavior.**

---

## What This Document Is

This is an automated composition screening note, not a materials discovery paper.  
The pipeline generated a candidate Co-rich alloy composition worth investigating further.  
It does **not** prove phase stability, oxidation resistance, heat treatment suitability,  
γ/γ' microstructure, service temperature, or any other materials property.

Every number is a hypothesis for experimental or high-fidelity computational  
(CALPHAD, DFT) follow-up — not validated data.

---

## 1. Discovery Workflow

```
Latin Hypercube Sampling (40 compositions, seed 77)
    → ALCHEMI GPU batch screen (LJ energy ranking, a = 3.52 Å, 40 compositions)
        → Combined scoring (LJ energy + heuristic property estimates)
            → Top candidate flagged for further characterisation
```

### 1.1 Composition Space

| Element | Min (at.%) | Max (at.%) | Rationale |
|---------|-----------|-----------|-----------|
| Co | 30 | 55 | Matrix element (Co-based target) |
| Ni | 10 | 35 | Solid solution + potential γ' former |
| Cr | 8 | 20 | Oxidation and corrosion resistance |
| Mo | 3 | 12 | Solid solution strengthening, creep |
| Al | 2 | 10 | Oxidation protection (Al₂O₃), γ' former |
| W | 1 | 8 | Solid solution strengthening, density penalty |

40 compositions sampled with Latin Hypercube (scipy.stats.qmc, seed 77).  
After sampling, compositions are clip-and-renormalised iteratively (≤10 passes) to ensure all elements stay within stated bounds.

### 1.2 ALCHEMI Batch Screen

40 FCC supercells (a = 3.52 Å, 32 atoms each) evaluated in a single GPU batch.

**What the LJ screen measures:**

```
E_alloy/atom − Σ xᵢ · E_pure_i/atom  (Lennard-Jones cohesive energy, eV/atom)
```

**What it does not measure:**
- Thermodynamic formation energy (requires DFT or CALPHAD; typically < 0.5 eV/atom)
- Electronic, magnetic, or many-body bonding
- Phase competition or ordering
- FCC vs BCC vs dual-phase stability

Values are LJ cohesive energy units (~9 eV/atom), **not** DFT formation energies.  
Use only for relative ranking within a single batch on the same lattice.

### 1.3 Combined Scoring

```
score = 0.30 × (−E_LJ / 10.0)               (relative, more negative = favoured)
      + 0.35 × oxidation_resistance_score_raw  (continuous heuristic)
      + 0.35 × phase_stability_score_raw        (continuous heuristic)
```

Scores are continuous and unclamped; actual differentiation between candidates is possible.  
Heuristic scores are empirical composition formulas, not thermodynamic calculations.

### 1.4 Composition Bounds Verification

All 6 elements confirmed within stated bounds after clip-and-renormalise:

| Element | Min sampled | Max sampled | Bound OK |
|---------|-------------|-------------|---------|
| Co | 33.0% | 55.0% | ✓ |
| Ni | 12.2% | 33.3% | ✓ |
| Cr | 8.7% | 19.9% | ✓ |
| Mo | 3.8% | 12.0% | ✓ |
| Al | 2.2% | 10.0% | ✓ |
| W | 1.2% | 8.0% | ✓ |

---

## 2. Candidate Composition

**Co41Cr16Ni16Mo12Al8W7** (at.%)

| Element | at. % |
|---------|-------|
| Co | 41.5 |
| Cr | 16.3 |
| Ni | 15.8 |
| Mo | 12.0 ← at upper bound |
| Al | 7.6 |
| W | 6.7 |

Combined score: **1.1395** | LJ energy: **−9.166 eV/atom**  
Alloy family classification: co\_rich\_alloy

**Why this composition stands out (heuristically):**
- High Mo (12%) is unusual — it contributes strongly to solid-solution strengthening
  and oxidation heuristic but may stabilise TCP phases (σ, μ) in reality
- Cr (16%) + Al (7.6%) are in the range used for oxidation-resistant Co alloys
- W (6.7%) near upper bound adds density but also modulus contribution

---

## 3. Heuristic Property Estimates

These are composition-weighted mixing rules, **not** CALPHAD or DFT calculations.

| Property | Value | Basis |
|----------|-------|-------|
| Density | 9.01 g/cm³ | Rule of mixtures (W + Mo penalty) |
| Elastic modulus | 236 GPa | Voigt average |
| Yield strength proxy | 336 MPa | Mismatch hardening heuristic |
| Solidus proxy | 1736 °C | Mismatch-corrected liquidus |
| Phase stability score (raw) | 1.249 | Empirical formula — not CALPHAD |
| Oxidation resistance score (raw) | 1.220 | Empirical formula |
| Configurational entropy | 13.17 J/mol·K (1.58 R) | −R Σ xᵢ ln xᵢ |
| VEC | 7.65 | Composition-weighted valence electrons |
| Ni/(Al+Ti) | 2.08 | Stoichiometry indicator (Ti = 0) |

### Open questions raised by these numbers

**VEC = 7.65**  
Empirical HEA rules associate FCC with VEC ≥ 8, mixed FCC/BCC with 6.87–8. VEC = 7.65 does not predict FCC. Co-rich alloys may favour HCP over FCC at room temperature depending on stacking fault energy. Phase structure is unknown without CALPHAD or experiment.

**Mo = 12% — TCP phase risk**  
Mo at high fractions is a known driver of σ-phase and μ-phase (topologically close-packed) formation in Co and Ni superalloys. These brittle phases are strongly detrimental to ductility and toughness. CALPHAD calculation to check TCP stability at service temperatures is critical before this composition is taken further.

**W = 6.7% — density cost**  
W increases density (W: 19.25 g/cm³). At 6.7%, it contributes meaningfully to the 9.01 g/cm³ density estimate. Co-based turbine alloys must balance strengthening gain from W against weight penalty, particularly for rotating components.

**Ni/(Al+Ti) = 2.08**  
In Ni-based superalloys, γ' Ni₃(Al,Ti) requires Ni/(Al+Ti) = 3.0. In Co-based alloys, the relevant ordered precipitate is Co₃(Al,W) (the L1₂ cobalt γ'). The stability of Co₃Al in this composition is not predictable from this heuristic. Whether a useful volume fraction of coherent precipitate forms requires CALPHAD and experimental ageing studies.

**Configurational entropy = 1.58 R**  
Exceeds the 1.5R HEA threshold. As with the previous alloy (Ni33Cr23Fe15Al11Co9Ti9), entropy alone does not guarantee a single-phase solid solution. The high Mo and W content creates specific intermetallic formation tendencies that entropy may not suppress.

**Solidus proxy = 1736°C**  
This is higher than most Ni-based alloys (typically 1230–1340°C liquidus). However, it comes from the Burke-Turnbull rule-of-mixtures — Co (1495°C) + W (3422°C) and Mo (2623°C) contributions pull the average up. The actual liquidus of a multi-component alloy requires DSC or CALPHAD.

---

## 4. Atomistic Screen (ALCHEMI, CPU proxy)

**Cell assignment (largest-remainder, seed 31, 32 atoms):**

| Species | Count |
|---------|-------|
| Co | 13 |
| Ni | 5 |
| Cr | 5 |
| Mo | 4 |
| Al | 3 |
| W | 2 |

| Result | Value |
|--------|-------|
| Backend | numpy\_cpu\_proxy (nvalchemiops not installed) |
| Relaxation | LJ volume proxy |
| Converged | Yes |
| LJ energy/atom | −3.00 eV (proxy value) |
| Max force | 0.00 eV/Å (proxy) |

**Limitation:** The CPU proxy does not run actual FIRE2 geometry relaxation. It returns a placeholder energy. The GPU path (nvalchemiops + CUDA) was used only for the batch screening; single-cell relaxation fell back to the proxy. Real atomistic relaxation requires installing nvalchemiops with CUDA.

**MD (1200°C, 500 steps, NVT):**
- Backend: numpy\_cpu\_proxy (equipartition estimate)
- Total time: 0.5 ps
- Proxy kinetic energy: 6.09 eV (equipartition, not a real trajectory)

---

## 5. Heat Treatment Analysis

All parameters are representative/generic, **not fitted to this composition**.

### 5.1 TTT Diagram (γ' precipitation, screening model)

Parameters used (not derived from CALPHAD or experiment):

| Parameter | Value | Note |
|-----------|-------|------|
| Solvus estimate | 1150°C (1423 K) | Assumed, not calculated |
| Low T bound | 600°C (873 K) | Assumed |
| Nose T | 850°C (1123 K) | Assumed |
| t_nose | 120 s | Representative |
| Q (diffusion) | 160 kJ/mol | Generic Co-alloy |

| Condition | t_start | t_finish |
|-----------|---------|---------|
| At nose (861°C) | 15 s | 321 s |
| At 600°C | 35 s | — |
| At 1100°C | 42 s | — |

**What this means:** Fast nose kinetics (15 s to 1% transformation) implies a demanding quench rate is needed to avoid precipitation during cooling from the solution anneal. Whether the actual nose is at 850°C and whether γ' (or Co₃Al, or TCP) is what forms — both require CALPHAD.

### 5.2 Grain Growth (850°C, 2 hours)

Parameters: K₀ = 1.5 × 10⁻⁴ m²/s, Q = 240 kJ/mol (generic Co-alloy, not fitted)

| Parameter | Value |
|-----------|-------|
| Initial grain size | 50 µm |
| Final grain size | 50.07 µm |
| Growth factor | 1.0015 |
| K(T) at 850°C | 1.03 × 10⁻¹⁵ m²/s |

Negligible grain growth at 850°C for 2 hours — consistent with strong pinning by precipitates or slow kinetics at this temperature for Co-rich alloys. The high Q = 240 kJ/mol is a reasonable upper estimate for Co alloys with high refractory content; the actual value is unknown.

### 5.3 Precipitation Kinetics (JMAK, 850°C, 2 hours)

Parameters: τ_ref = 300 s at 900°C, Q_nucleation = 180 kJ/mol, Q_growth = 160 kJ/mol, n = 3

| Parameter | Value |
|-----------|-------|
| Incubation time at 850°C | 682 s (~11 min) |
| Precipitate fraction at 2h | 1.00 (fully transformed) |
| Avrami exponent | 3.0 |

Full transformation within 2 hours at 850°C. The 682 s incubation shifts the nose left relative to a Ni-based alloy with shorter τ_ref. Whether this is γ', Co₃Al, TCP, or something else — unknown without CALPHAD.

---

## 6. Phase Field (Allen-Cahn, CPU)

A 2D Allen-Cahn simulation was run on a 64×64 grid.

| Parameter | Value |
|-----------|-------|
| Backend | numpy\_cpu |
| Grid | 64 × 64 |
| Steps | 800 |
| M, κ, dt | 1.0, 0.5, 0.05 |
| Final φ mean | 0.476 |
| Final φ std | ~0.5 (bimodal — two phases) |
| Centerline profile | Sharp interface visible |

**What it shows:** The double-well PDE separates into two domains over 800 steps. Sharp interface at grid centre with φ ≈ 0 on one side and φ ≈ 1 on the other.

**What it does not show:** No connection to γ/Co₃Al microstructure in this alloy. Parameters (M, κ, dt) are not derived from this composition. Phase field modelling of actual precipitate morphology requires composition-dependent free energies, interfacial energy, and lattice misfit data — none of which are available from this screening.

---

## 7. Oxidation / Corrosion Heuristics (900°C, 10,000 hours, dry air)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Parabolic kp | 2.13 × 10⁻¹⁴ m²/s | Heuristic rate constant |
| Oxide thickness @ 10 kh | 875 µm | **Severe** — 0.875 mm scale |
| Mass gain proxy | 481 mg/cm² | Heavy oxidation |
| Corrosion risk score | 0.071 | Low Cl⁻ environment |
| H-embrittlement risk | 0.249 | Moderate (Ti = 0, but Mo present) |

**Warning:** 875 µm oxide at 10,000 hours is severe degradation, not acceptable for a precision high-temperature component. The parabolic proxy does not account for the protective Al₂O₃/Cr₂O₃ scale that may form at these Cr + Al levels — it is a conservative worst-case upper bound.

**What is plausible:** Cr (16%) + Al (7.6%) is at the lower edge of alumina-former territory. MCrAlY bond coats typically use 8–12% Al for reliable Al₂O₃ scale formation. Whether a continuous protective scale actually forms from this composition requires Ellingham analysis, TGA exposure testing, and SEM/EPMA characterisation.

**Mo oxidation concern:** Mo forms MoO₃, which is volatile above ~750°C ("pest oxidation"). At 12% Mo, oxidation in air above 700–800°C may involve volatile MoO₃ formation, which would break down any protective scale. This is a critical risk that the parabolic proxy model does not capture.

---

## 8. What the Simulation Stack Can and Cannot Do

| Claim | Valid | Requires |
|-------|-------|---------|
| Candidate composition identified from 6-element space | ✓ | — |
| Relative LJ energy ranking within batch | ✓ | Same lattice, same potential |
| Composition-weighted density, modulus | ✓ (±10%) | — |
| Phase structure (FCC vs HCP vs BCC) | ✗ | CALPHAD or DFT |
| Co₃Al γ' stability and volume fraction | ✗ | CALPHAD + phase field with real parameters |
| TCP phase stability (σ, μ) | ✗ | CALPHAD — critical for high-Mo alloys |
| Actual oxidation rate | ✗ | TGA + exposure testing |
| MoO₃ pest oxidation risk | ✗ | Oxidation testing above 700°C |
| Actual yield and creep strength | ✗ | Mechanical testing |
| Actual solvus temperature | ✗ | CALPHAD + DSC |
| Service temperature rating | ✗ | Creep testing + oxidation testing |

---

## 9. Priority Validation Steps

Ordered by what would most rapidly confirm or falsify this candidate:

1. **CALPHAD phase equilibrium** (Thermo-Calc TCNI or TCCOB / Pandat)  
   Calculate Co-Ni-Cr-Mo-Al-W phase diagram. Critical questions: Is γ' (Co₃Al type) stable? At what fraction? Where do σ and μ phases appear? This is the most important missing calculation.

2. **TCP phase check**  
   Mo and W are strong TCP promoters. Compute σ/μ solvus and phase fractions at 700–1000°C. If TCP phases are stable, this composition is likely not viable as a superalloy without further adjustment.

3. **MoO₃ volatility assessment**  
   Ellingham + thermodynamic calculation of MoO₃ partial pressure at 900°C in air. If p(MoO₃) is significant, the oxidation behaviour will be catastrophic above ~750°C.

4. **MACE-MP-0 / SevenNet energy**  
   Run `alchemi_mlip` with `potential_path=` to get ~DFT-quality energies in seconds, replacing the LJ proxy.

5. **DFT SQS single-point** (VASP / QE)  
   Formation energy and convex hull position. Check whether Co₃Al is a stable competing phase.

6. **Arc-melting + XRD**  
   Synthesise a button ingot and characterise phases present. Compare to CALPHAD prediction.

7. **TGA oxidation at 700–1000°C**  
   Check for MoO₃ volatilisation. If mass loss occurs above 750°C, Mo must be reduced.

8. **Ageing study (hardness vs time at 700–950°C)**  
   Map the actual TTT curve. Confirm whether hardening or softening occurs, and what phase is responsible.

---

## 10. Comparison with Previous Candidate (Ni33Cr23Fe15Al11Co9Ti9)

| Property | Ni33Cr23Fe15Al11Co9Ti9 | Co41Cr16Ni16Mo12Al8W7 |
|----------|----------------------|----------------------|
| Alloy family | Ni-rich | Co-rich |
| Density | 7.29 g/cm³ | 9.01 g/cm³ |
| Elastic modulus | 199 GPa | 236 GPa |
| Solidus proxy | 1486°C | 1736°C |
| VEC | 7.40 | 7.65 |
| S_conf | 1.66 R | 1.58 R |
| Key risk | Ni/(Al+Ti) = 1.67 → excess Al+Ti | Mo TCP risk, MoO₃ volatility |
| Oxidation thickness @ 10 kh | 759 µm | 875 µm |

Neither composition is validated. The Co-rich candidate has a higher modulus and higher solidus proxy, but a worse oxidation proxy and a critical Mo-related risk that the Ni-based candidate does not have.

---

## Appendix: Pipeline Steps Run

| Step | Method | Input | Output |
|------|--------|-------|--------|
| 1. Sampling | `scipy.stats.qmc.LatinHypercube` | seed=77, n=40, d=6 | 40 compositions |
| 2. Bounds fix | Iterative clip-renorm (10 passes) | Raw LHS | Bounded compositions |
| 3. LJ screen | `ALCHEMIBackend.batch_alloy_screen` | 40 comps, a=3.52 Å | LJ energies |
| 4. Scoring | Combined LJ + heuristic | 40 comps | Top candidate |
| 5. Properties | `MaterialsGPUWorkflowEngine.alloy_property_prediction` | Co41… | Density, modulus, VEC |
| 6. Degradation | `MaterialsGPUWorkflowEngine.degradation_prediction` | Co41…, 900°C, 10 kh | Oxide thickness |
| 7. Relaxation | `ALCHEMIBackend.relax_structure` (CPU proxy) | 32-atom cell | Proxy energy |
| 8. TTT | `PhaseTransformationEngine.ttt_diagram` | Generic Co params | C-curve |
| 9. Grain growth | `PhaseTransformationEngine.grain_growth` | 850°C, 2h, K₀=1.5e-4 | d=50.07 µm |
| 10. Precipitation | `PhaseTransformationEngine.precipitation_kinetics` | 850°C, τ_ref=300s | X=1.00 at 2h |
| 11. Phase field | `WarpBackend.run_allen_cahn_field` | 64×64, 800 steps | Bimodal φ |

---

*Generated by WorldSIM v0.2 — MaxOSL AI Research*  
*This is a preliminary automated screening note. All values require experimental or high-fidelity computational validation before any material selection decision.*
