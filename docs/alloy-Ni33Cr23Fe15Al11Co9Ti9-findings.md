# Preliminary Screening Note: Ni33Cr23Fe15Al11Co9Ti9

**Platform:** WorldSIM v0.2 — MaxOSL AI Research  
**Date:** 2026-05-15  
**GPU:** NVIDIA GeForce RTX 3060 (12 GB, sm_86)  
**Status: Concept-level screening only. No claim of validated superalloy behavior.**

---

## What This Document Is

This is an automated composition screening note, not a materials discovery paper.  
The pipeline described below generated a candidate composition worth investigating further.  
It does **not** prove phase stability, oxidation resistance, heat treatment suitability,  
γ' precipitation, service temperature, or any other materials property.

Every number should be treated as a hypothesis for experimental or high-fidelity  
computational (CALPHAD, DFT) follow-up — not as validated data.

---

## 1. Discovery Workflow

```
Latin Hypercube Sampling (40 compositions)
    → ALCHEMI GPU batch screen (LJ pairwise energy difference)
        → Property scoring (heuristic, continuous, unclamped)
            → Top candidate flagged for further review
```

### 1.1 Composition Space

| Element | Min (at.%) | Max (at.%) |
|---------|-----------|-----------|
| Fe | 5 | 40 |
| Ni | 25 | 55 |
| Cr | 10 | 25 |
| Co | 5 | 20 |
| Al | 2 | 12 |
| Ti | 1 | **8** |

40 compositions sampled with Latin Hypercube (scipy.stats.qmc, seed 42).  
After sampling, compositions are clip-and-renormalised iteratively to ensure all elements stay within their stated bounds (fix applied after peer review).

### 1.2 ALCHEMI Batch Screen

40 FCC supercells (a = 3.57 Å, 32 atoms each) evaluated in **0.51 s** on the RTX 3060.

**What the LJ screen measures:**

```
lj_diff = E_alloy/atom − Σ xᵢ · E_pure_i/atom
```

This is the difference in Lennard-Jones cohesive energy between the mixed alloy and the composition-weighted pure elements on the same lattice.

**What it does not measure:**
- Thermodynamic formation energy (that requires DFT or CALPHAD)
- Electronic, magnetic, or many-body metallic bonding
- Phase competition or ordering energetics
- Anything that would let you distinguish a stable superalloy from an unstable intermetallic

Values like −15 eV/atom are in LJ cohesive energy units. They are **not** DFT formation energies, which are typically < 0.5 eV/atom in magnitude. Use this metric only for relative ranking within a single batch screened on the same lattice.

### 1.3 Scoring

```
score = 0.30 × lj_diff_component  (relative within batch)
      + 0.35 × oxidation_resistance_score_raw  (continuous heuristic)
      + 0.35 × phase_stability_score_raw        (continuous heuristic)
```

Scores are now continuous and unclamped after peer review, allowing actual differentiation between candidates. The heuristic scores are empirical composition formulas, not thermodynamic calculations.

### 1.4 Flag: Scoring Saturation (Resolved)

In the original run, all top-5 candidates scored identically (1.000) because sub-scores were clamped to [0, 1]. After the fix, raw scores differentiate candidates (e.g. 1.250 vs 1.190 for phase stability). The selection of this specific composition over others of similar LJ diff should be treated as provisional.

---

## 2. Candidate Composition

**Ni33Cr23Fe15Al11Co9Ti9** (at.%)  
All elements confirmed within their stated search bounds after the clip-renorm fix.

| Element | at. % |
|---------|-------|
| Ni | 32.8 |
| Cr | 23.0 |
| Fe | 15.5 |
| Al | 10.9 |
| Co | 9.2 |
| Ti | 8.0 ← clipped to stated bound |

---

## 3. Heuristic Property Estimates

These are composition-weighted mixing rules, **not** CALPHAD or DFT calculations.

| Property | Value | Basis |
|----------|-------|-------|
| Density | 7.29 g/cm³ | Rule of mixtures |
| Elastic modulus | 199 GPa | Voigt average |
| Yield strength proxy | 373 MPa | Mismatch hardening heuristic |
| Solidus proxy | 1486 °C | Mismatch-corrected liquidus |
| Phase stability score (raw) | 1.25 | Empirical formula — not CALPHAD |
| Oxidation resistance score (raw) | 1.27 | Empirical formula |
| Configurational entropy | 13.84 J/mol·K | −R Σ xᵢ ln xᵢ |
| VEC | 7.40 | Composition-weighted valence electrons |
| Ni / (Al+Ti) | 1.67 | Stoichiometry indicator |

### Open questions raised by these numbers

**VEC = 7.40**  
Empirical HEA rules associate FCC stability with VEC ≥ 8, mixed FCC/BCC with 6.87–8. VEC = 7.40 does not prove FCC. The phase structure is unknown without CALPHAD or experiment.

**Ni/(Al+Ti) = 1.67**  
Ideal γ' Ni₃(Al,Ti) stoichiometry requires Ni/(Al+Ti) = 3.0. This alloy has Al+Ti ≈ 19.6 at.%, with only 32.8 at.% Ni — far below the ratio needed for all Al+Ti to enter γ'. Excess Al and Ti could drive B2 NiAl, η Ni₃Ti, or Laves phase formation. This is a serious concern requiring CALPHAD phase equilibrium calculation.

**Configurational entropy = 13.84 J/mol·K**  
Exceeds the 1.5R HEA threshold. However, modern HEA research shows that entropy alone does not guarantee simple-phase FCC microstructure. Mixing enthalpy, intermetallic formation tendency, atomic size, and specific element chemistry all matter.

**Self-contradiction in the original report**  
Claiming "entropy suppresses ordered intermetallics" while simultaneously relying on ordered γ' for strengthening is contradictory. The composition may or may not form γ'; that depends on thermodynamics, not the claim.

---

## 4. Atomistic Screen (ALCHEMI, FIRE2 + LJ)

**Cell after peer review fix — largest-remainder assignment, seed 7:**

| Species | Count |
|---------|-------|
| Ni | 2 |
| Cr | 2 |
| Al | 1 |
| Co | 1 |
| Fe | 1 |
| Ti | **1** ← present after fix |

Previous cell was Ni/Cr/Al/Co/Fe with no Ti. Fixed.

| Result | Value |
|--------|-------|
| Relaxation | FIRE2 + Lennard-Jones |
| Converged | Yes (92 steps) |
| LJ energy/atom | −0.17 eV |
| Max force | 0.049 eV/Å |

**What this means:** The LJ cell converged to a local energy minimum. This is a pairwise potential calculation on 8 atoms. It does not characterise electronic structure, magnetism, ordering tendencies, or phase stability of the bulk alloy.

---

## 5. Oxidation / Corrosion Heuristics

Two separate unconnected models were run. They should not be read together.

| Model | Output | Value | Interpretation |
|-------|--------|-------|----------------|
| Corrosion risk heuristic | Risk score | 0.026 / unbounded | Low Cl⁻ environment → low score. **Not a validated corrosion rate.** |
| Parabolic oxidation proxy | Oxide thickness @ 10,000 h | 759 µm | This is **severe** degradation. Not "excellent." |
| Parabolic oxidation proxy | Mass gain | 418 mg/cm² | Consistent with heavy oxide scale |

**Original document error:** The report called corrosion "excellent" based on the heuristic risk score while ignoring the 759 µm oxide thickness from a different model. A 0.76 mm oxide scale is not acceptable for a precision high-temperature alloy.

**Environment inputs at 900 °C:** pH and chloride mol/L are valid for aqueous or molten-salt corrosion environments, not dry gas turbine conditions. For high-temperature oxidation without condensate, the relevant input is p(O₂), which was set to 0.21 atm. The pH-dependent terms in the heuristic are not physically meaningful at 900 °C in dry air.

**What is defensible:** High Cr (23%) + Al (11%) is the same passivation strategy used in MCrAlY bond coats and alumina-forming austenitic alloys. Whether this specific composition actually forms a protective Al₂O₃/Cr₂O₃ scale at these concentrations requires Ellingham analysis, TGA, and experimental exposure testing.

---

## 6. Phase Field (Allen-Cahn)

A 2D Allen-Cahn simulation was run with generic parameters (M, κ, dt not derived from this alloy).

**What it shows:** A double-well PDE separates into two domains over 800 time steps on a 48×48 grid.

**What it does not show:** This has no connection to γ/γ' microstructure in this alloy. Phase field modelling of γ' precipitation requires composition-dependent free energies, interfacial energy data (σ ~ 10–30 mJ/m²), lattice misfit δ, and phase-specific mobility. None of those were used.

---

## 7. Heat Treatment Estimates

The TTT diagram, JMAK kinetics, grain growth, and precipitation kinetics were computed with parameters described as "representative, not fitted to this composition."

**What can be said:**
- A solution anneal above the γ' solvus followed by aging at a nose temperature is the standard approach for Ni superalloys
- Grain growth was negligible in the Burke-Turnbull model at 750 °C — this is consistent with γ'-pinned grain boundaries generally
- The prescribed schedule is physically plausible as a starting point for experimental aging studies

**What cannot be said:**
- The actual solvus temperature for this composition (unknown without CALPHAD/DSC)
- Whether the nose is actually at 750 °C for this alloy
- The correct aging time for target γ' fraction
- Whether secondary phases (B2, η, Laves) precipitate during the schedule

**Regarding IN-738 / IN-939 analogy:** Those alloys contain W, Mo, Ta, Nb, C, B, and Zr — critical for creep strength, carbide pinning, grain boundary chemistry, and castability. This composition lacks all of those. The analogy is not justified.

---

## 8. What the Simulation Stack Can and Cannot Do

| Claim | Valid | Requires |
|-------|-------|---------|
| Candidate composition identified from a 6-element space | ✓ | — |
| Relative LJ pairwise energy ranking within a batch | ✓ | Same lattice, same potential |
| Composition-weighted density, modulus | ✓ (± ~10%) | — |
| Phase structure (FCC vs BCC vs dual-phase) | ✗ | CALPHAD or DFT |
| γ' volume fraction and morphology | ✗ | CALPHAD + phase field with real parameters |
| Thermodynamic formation energy | ✗ | DFT or CALPHAD |
| Actual oxidation rate | ✗ | TGA + exposure testing |
| Actual yield strength | ✗ | Tensile testing |
| Heat treatment schedule | ✗ (starting point only) | CALPHAD + DSC + aging studies |
| Service temperature rating | ✗ | Creep testing |

---

## 9. Priority Validation Steps

Ordered by what would most rapidly falsify or confirm the candidate:

1. **CALPHAD phase equilibrium** (Thermo-Calc TCNI / Pandat)  
   Calculate phase diagram, γ' solvus, γ' fraction at temperature, and check for B2/η/Laves stability. This is the most critical missing step.

2. **VEC and Miedema analysis**  
   Compute mixing enthalpy from Miedema's model to check whether the alloy favours solid solution or intermetallic formation.

3. **DFT single-point** (VASP / Quantum ESPRESSO)  
   SQS cell, PBE functional. Get actual formation energy and check convex hull position.

4. **MACE-MP-0 neural potential** (available now, free)  
   Replace LJ with `potential_path=` in `alchemi_mlip`. Gets ~DFT-quality energies in seconds. Would immediately show whether the LJ ranking is meaningful.

5. **TGA oxidation test** at 900–1100 °C in synthetic air  
   Confirm whether protective Al₂O₃/Cr₂O₃ scale actually forms.

6. **Arc-melting + DSC**  
   Measure actual solvus, liquidus, and phase transformation temperatures. Compare to CALPHAD.

7. **Aging study** (hardness vs time at 650–850 °C)  
   Map the actual TTT curve experimentally.

---

## Appendix: Bugs Fixed After Peer Review

| Bug | Description | Fix |
|-----|-------------|-----|
| Ti search bound violation | Normalisation pushed Ti to 8.7%, above stated 8% max | Iterative clip-and-renormalise loop (≤10 passes) |
| Ti dropped from atomistic cell | Simple rounding + truncation silently removed Ti from 8-atom cell | Largest-remainder integer assignment |
| Formation energy mislabelling | LJ cohesive energy difference (∼10s of eV) presented as thermodynamic formation energy (should be < 0.5 eV) | Renamed to `lj_cohesive_energy_diff_eV_atom` with disclaimer in output |
| Scoring saturation | All top-5 candidates scored 1.000 — no differentiation | Scores now continuous and unclamped |

---

*Generated by WorldSIM v0.2 — MaxOSL AI Research*  
*This is a preliminary automated screening note. All values require experimental or high-fidelity computational validation before any material selection decision.*
