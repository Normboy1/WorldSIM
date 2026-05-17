"""WorldSim Materials — alloy design workflow demo.

Demonstrates the full ALCHEMI-oriented pipeline:
1. Alloy property screening (composition → property proxies)
2. Carburization case-depth prediction (diffusion)
3. JMAK transformation kinetics (heat treatment planning)
4. Grain growth prediction (microstructure evolution)
5. Oxidation kinetics and Ellingham stability
6. Corrosion risk and SCC susceptibility assessment
7. Surrogate model training plan (PhysicsNeMo)
"""

from simlab.core.engine.simlab_core import SimLabCore

core = SimLabCore()

print("=" * 60)
print("WorldSim Materials — Fe-Cr-Ni Superalloy Design Pipeline")
print("=" * 60)

# ------------------------------------------------------------------
# 1. Alloy property screening
# ------------------------------------------------------------------
print("\n[1] Alloy property screening")
r = core.simulate_materials(
    "alloy_property_prediction",
    {
        "composition": {"Ni": 0.62, "Cr": 0.18, "Co": 0.10, "Al": 0.06, "Ti": 0.04},
        "temperature_K": 1050.0,
        "heat_treatment": "solution_anneal + aging",
    },
)
props = r.results["property_estimates"]
print(f"  Alloy family       : {r.results['alloy_family']}")
print(f"  Yield proxy        : {props['yield_strength_proxy_MPa']:.0f} MPa")
print(f"  Oxidation resistance: {props['oxidation_resistance_score']:.2f}")
print(f"  Phase stability    : {props['phase_stability_score']:.2f}")

# ------------------------------------------------------------------
# 2. Carburization case depth (diffusion)
# ------------------------------------------------------------------
print("\n[2] Carburization case depth (C in γ-Fe, 940°C, 6 hours)")
r = core.simulate_materials(
    "diffusion_profile",
    {
        "D": 0.0,          # will be overridden by Arrhenius below
        "time_s": 6 * 3600.0,
        "C_surface": 0.85,
        "C_initial": 0.15,
        "case_threshold": 0.50,
        "preset": "C_in_austenite",
        "temperature_K": 1213.0,   # 940°C
    },
)
print(f"  D at 940°C         : {r.results['D_m2_s']:.3e} m²/s")
print(f"  Case depth (0.50%) : {r.results['case_depth_mm']:.3f} mm")
print(f"  sqrt(D*t)          : {r.results['sqrt_Dt_m']*1000:.4f} mm")

# ------------------------------------------------------------------
# 3. JMAK bainite transformation kinetics
# ------------------------------------------------------------------
print("\n[3] JMAK bainite transformation at 500°C (773 K)")
r = core.simulate_materials(
    "jmak_kinetics",
    {
        "k": 1.2e-4,          # typical bainite rate constant at 500°C
        "n": 2.0,             # 2D plate growth
        "t_max_s": 1800.0,
        "mode": "2d_growth_site_saturated",
    },
)
print(f"  10% transformed: {r.results['t_10pct_s']:.1f} s")
print(f"  50% transformed: {r.results['t_50pct_s']:.1f} s")
print(f"  90% transformed: {r.results['t_90pct_s']:.1f} s")

# ------------------------------------------------------------------
# 4. Grain growth after solution anneal
# ------------------------------------------------------------------
print("\n[4] Grain growth during solution anneal (1373 K, 1 hr)")
r = core.simulate_materials(
    "grain_growth",
    {
        "d0_um": 15.0,
        "temperature_K": 1373.0,
        "time_s": 3600.0,
        "preset": "austenitic_steel",
    },
)
print(f"  d0 = {r.results['d0_um']} µm → d_final = {r.results['final_grain_size_um']:.1f} µm")
print(f"  Growth factor: {r.results['growth_factor']:.2f}×")

# ------------------------------------------------------------------
# 5. Oxidation kinetics at service temperature
# ------------------------------------------------------------------
print("\n[5] Oxidation kinetics — 18Cr stainless in air at 900°C (1173 K)")
r = core.simulate_materials(
    "oxidation_kinetics",
    {
        "preset": "Fe_Cr18_air",
        "temperature_K": 1173.0,
        "time_s": 10_000.0,
    },
)
print(f"  Regime         : {r.results['regime']}")
print(f"  Oxide after 10 ks : {r.results['final_thickness_um']:.4f} µm")

# Ellingham: check which oxides form preferentially at 900°C
print("\n[5b] Ellingham stability at 1173 K (most stable first)")
r = core.simulate_materials("ellingham_diagram", {"T_min_K": 1173.0, "T_max_K": 1173.0, "n_points": 1})
for rxn in sorted(r.results["reactions"], key=lambda x: x["dG_kJ_molO2_at_Tmin"]):
    print(f"  {rxn['reaction'][:38]:38s}  ΔG = {rxn['dG_kJ_molO2_at_Tmin']:+.0f} kJ/mol O₂")

# ------------------------------------------------------------------
# 6. Pilling-Bedworth + corrosion risk
# ------------------------------------------------------------------
print("\n[6] Pilling-Bedworth ratio for candidate oxides")
for metal in ["Al", "Cr", "Ni", "Fe"]:
    r = core.simulate_materials("pilling_bedworth_ratio", {"metal": metal})
    print(f"  {metal:4s}: PBR = {r.results['PBR']:.3f}  {r.results['protectiveness'][:35]}")

print("\n[6b] Corrosion risk — Fe at various pH values (E = -0.30 V SHE)")
for ph in [3.0, 7.0, 12.0]:
    r = core.simulate_materials("corrosion_risk", {"metal": "Fe", "pH": ph, "electrode_potential_V": -0.30})
    print(f"  pH {ph:.0f}: {r.results['stability_region']:15s}  {r.results['description'][:40]}")

print("\n[6c] SCC susceptibility — austenitic stainless vs environments")
for env in ["chloride", "caustic", "high_temp_water"]:
    r = core.simulate_materials("scc_risk_index", {
        "material": "austenitic_stainless",
        "environment": env,
        "temperature_K": 373.0,
        "stress_MPa": 250.0,
    })
    print(f"  {env:20s}: risk = {r.results['scc_risk_index']:.2f}  ({r.results['severity'][:25]})")

# ------------------------------------------------------------------
# 7. GPU backend status + surrogate model plan
# ------------------------------------------------------------------
print("\n[7] GPU backend detection")
r = core.simulate_materials("gpu_backend_status", {})
for backend, detected in r.results["available"].items():
    status = "✓ detected" if detected else "  not installed"
    print(f"  {backend:15s}: {status}")

print("\n[7b] PhysicsNeMo surrogate training plan")
r = core.simulate_materials(
    "surrogate_model_plan",
    {
        "dataset_size": 2000,
        "target_properties": ["yield_strength_proxy_MPa", "oxide_thickness_um",
                              "phase_stability_score", "case_depth_mm"],
        "operator": "fourier_neural_operator",
    },
)
split = r.results["dataset_split"]
print(f"  Dataset split: train={split['train']}  val={split['validation']}  test={split['test']}")
print(f"  Loss terms: {len(r.results['loss_terms'])} terms")
print(f"  Active learning loop steps: {len(r.results['active_learning_loop'])}")

print("\n" + "=" * 60)
print("Pipeline complete — all WorldSim Materials engines operational.")
print("Install nvidia stack: pip install -e '.[nvidia]' for ALCHEMI/Warp/PhysicsNeMo.")
print("=" * 60)
