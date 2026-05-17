"""Forging and Casting simulation visuals for WorldSim Materials."""

from __future__ import annotations
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm

sys.path.insert(0, "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM-Materials")

from simlab.engines.materials.forging import ForgingEngine, FORGING_PRESETS
from simlab.engines.materials.casting import CastingEngine, CASTING_PRESETS

BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
FG    = "#e6edf3"
MUTED = "#8b949e"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f0883e"
RED     = "#f85149"
PURPLE  = "#bc8cff"
YELLOW  = "#e3b341"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,  "axes.labelcolor": FG,
    "xtick.color": MUTED,    "ytick.color": MUTED,
    "text.color": FG,        "grid.color": GRID, "grid.linewidth": 0.5,
    "font.family": "monospace", "font.size": 8.5,
})

fe = ForgingEngine()
ce = CastingEngine()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — FORGING
# ═══════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(16, 10), facecolor=BG)
fig1.suptitle("WorldSim Materials  ·  Forging Simulation Suite", fontsize=13,
              color=FG, fontweight="bold", y=1.01)
gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.52, wspace=0.40)

# ── A: Flow stress curves (4 materials, 3 temperatures) ────────────────────
ax_a = fig1.add_subplot(gs1[0, 0])
colors_T = [ACCENT, ORANGE, RED]
temps = [20, 600, 1100]
for mat, ls in [("AISI_4340", "-"), ("Ti6Al4V", "--"), ("Al_6061", ":")]:
    for T, col in zip(temps, colors_T):
        r = fe.flow_stress(mat, strain_max=0.8, temperature_C=T, model="johnson_cook")
        label = f"{mat.replace('_','-')} {T}°C" if T == 20 else None
        ax_a.plot(r["strain"], r["flow_stress_MPa"], color=col, ls=ls,
                  lw=1.4, alpha=0.85, label=label)
# legend for materials
leg_lines = [
    plt.Line2D([0],[0], color=FG,   ls="-",  lw=1.5, label="AISI 4340"),
    plt.Line2D([0],[0], color=FG,   ls="--", lw=1.5, label="Ti-6Al-4V"),
    plt.Line2D([0],[0], color=FG,   ls=":",  lw=1.5, label="Al 6061"),
    plt.Line2D([0],[0], color=ACCENT, lw=2,  label="20°C"),
    plt.Line2D([0],[0], color=ORANGE, lw=2,  label="600°C"),
    plt.Line2D([0],[0], color=RED,    lw=2,  label="1100°C"),
]
ax_a.legend(handles=leg_lines, fontsize=6.5, ncol=2, loc="upper right", framealpha=0.3)
ax_a.set_xlabel("True strain")
ax_a.set_ylabel("Flow stress  [MPa]")
ax_a.set_title("A · Flow Stress (Johnson-Cook)", color=FG, fontsize=9, pad=5)
ax_a.grid(True, lw=0.4)

# ── B: Hall-Petch — yield strength vs grain size ────────────────────────────
ax_b = fig1.add_subplot(gs1[0, 1])
pal = [ACCENT, ORANGE, GREEN, RED, PURPLE]
for (mat, col) in zip(["AISI_4340","AISI_1020","Al_6061","Ti6Al4V","Inconel_718"], pal):
    r = fe.hall_petch(mat)
    ax_b.semilogx(r["grain_size_sweep_um"], r["yield_strength_sweep_MPa"],
                  color=col, lw=1.6, label=mat.replace("_","-"))
ax_b.axvspan(10, 50, alpha=0.06, color=GREEN)
ax_b.text(22, 400, "fine\ngrain\nzone", color=GREEN, fontsize=7, ha="center")
ax_b.set_xlabel("Grain size  [µm]")
ax_b.set_ylabel("Yield strength  [MPa]")
ax_b.set_title("B · Hall-Petch  σ_y vs d", color=FG, fontsize=9, pad=5)
ax_b.legend(fontsize=6.5, loc="upper right", framealpha=0.3)
ax_b.grid(True, lw=0.4)

# ── C: Dynamic recrystallization ─────────────────────────────────────────────
ax_c = fig1.add_subplot(gs1[0, 2])
ax_c2 = ax_c.twinx(); ax_c2.set_facecolor("none")
r_drx = fe.dynamic_recrystallization("AISI_4340", temperature_C=1100,
                                      strain_rate=1.0, initial_grain_um=100.0)
eps  = np.array(r_drx["strain"])
X    = np.array(r_drx["X_recrystallized"])
davg = np.array(r_drx["avg_grain_size_um"])

l1, = ax_c.plot(eps, X * 100, color=ACCENT, lw=2, label="X_DRX (%)")
l2, = ax_c2.plot(eps, davg, color=ORANGE, lw=1.8, ls="--", label="d_avg (µm)")
ax_c.axvline(r_drx["eps_critical"], color=RED, lw=1, ls=":", alpha=0.8)
ax_c.text(r_drx["eps_critical"]+0.02, 10, f"ε_c={r_drx['eps_critical']:.3f}",
          color=RED, fontsize=7)
ax_c.set_xlabel("True strain")
ax_c.set_ylabel("Recrystallized fraction  [%]", color=ACCENT)
ax_c2.set_ylabel("Avg grain size  [µm]", color=ORANGE)
ax_c.tick_params(axis="y", colors=ACCENT)
ax_c2.tick_params(axis="y", colors=ORANGE)
ax_c2.spines["right"].set_color(ORANGE)
ax_c.set_title("C · Dynamic Recrystallization\nAISI 4340 @ 1100°C, ε̇=1/s", color=FG, fontsize=9, pad=5)
lines = [l1, l2]; ax_c.legend(lines, [l.get_label() for l in lines], fontsize=7.5, framealpha=0.3)
ax_c.grid(True, lw=0.4)

# ── D: Forging force vs reduction ────────────────────────────────────────────
ax_d = fig1.add_subplot(gs1[1, 0])
for mat, col in zip(["AISI_4340","Ti6Al4V","Inconel_718"], [ACCENT, ORANGE, PURPLE]):
    r_ff = fe.forging_force(mat, billet_diameter_mm=120, original_height_mm=120,
                             final_height_mm=40, temperature_C=1100)
    ax_d.plot(r_ff["reduction_pct"], r_ff["force_vs_reduction_MN"],
              color=col, lw=1.7, label=mat.replace("_","-"))
ax_d.set_xlabel("Height reduction  [%]")
ax_d.set_ylabel("Press force  [MN]")
ax_d.set_title("D · Forging Force vs Reduction\n120mm billet @ 1100°C, μ=0.15", color=FG, fontsize=9, pad=5)
ax_d.legend(fontsize=7.5, framealpha=0.3)
ax_d.grid(True, lw=0.4)

# ── E: Processing map (hot workability) ──────────────────────────────────────
ax_e = fig1.add_subplot(gs1[1, 1])
r_pm = fe.processing_map("AISI_4340", temp_range_C=(900,1200),
                          strain_rate_range=(0.01, 100), strain=0.5, n_T=40, n_edot=40)
T_g   = np.array(r_pm["temperatures_C"])
ed_g  = np.array(r_pm["strain_rates_1_s"])
eta_g = np.array(r_pm["efficiency_eta"])
xi_g  = np.array(r_pm["instability_xi"])
safe_g = np.array(r_pm["safe_window"])

TT, EE = np.meshgrid(T_g, ed_g, indexing="ij")
cf = ax_e.contourf(TT, np.log10(EE), eta_g,
                   levels=np.linspace(0, 0.7, 20), cmap="plasma")
fig1.colorbar(cf, ax=ax_e, label="η efficiency")
# Safe window overlay
ax_e.contour(TT, np.log10(EE), safe_g, levels=[0.5], colors=GREEN, linewidths=1.5)
# Instability hatching
ax_e.contourf(TT, np.log10(EE), (xi_g < 0).astype(float),
              levels=[0.5, 1.5], colors=RED, alpha=0.2)
ax_e.set_xlabel("Temperature  [°C]")
ax_e.set_ylabel("log₁₀(ε̇  [1/s])")
ax_e.set_title("E · Prasad Processing Map\nAISI 4340, ε=0.5", color=FG, fontsize=9, pad=5)
ax_e.text(0.03, 0.92, f"Optimal: {r_pm['optimal_T_C']:.0f}°C", transform=ax_e.transAxes,
          color=GREEN, fontsize=7.5)
leg = [Patch(edgecolor=GREEN, facecolor="none", lw=1.5, label="Safe window"),
       Patch(facecolor=RED, alpha=0.4, label="Flow instability")]
ax_e.legend(handles=leg, fontsize=7, framealpha=0.3)

# ── F: Strain-rate sensitivity m-value ───────────────────────────────────────
ax_f = fig1.add_subplot(gs1[1, 2])
for mat, col, T in [("AISI_4340", ACCENT, 1100), ("Ti6Al4V", ORANGE, 900),
                     ("Al_6061", GREEN, 500), ("Inconel_718", PURPLE, 1100)]:
    r_m = fe.strain_rate_sensitivity(mat, temperature_C=T, strain=0.5)
    ax_f.semilogx(r_m["strain_rates_1_s"], r_m["m_value"],
                  color=col, lw=1.6, label=f"{mat.replace('_','-')} {T}°C")
ax_f.axhline(0.5, color=YELLOW, lw=1, ls="--", alpha=0.8, label="m=0.5 (superplastic)")
ax_f.set_xlabel("Strain rate  [1/s]")
ax_f.set_ylabel("m-value  (∂ln σ / ∂ln ε̇)")
ax_f.set_title("F · Strain-Rate Sensitivity", color=FG, fontsize=9, pad=5)
ax_f.legend(fontsize=6.5, framealpha=0.3)
ax_f.set_ylim(-0.05, 0.8)
ax_f.grid(True, lw=0.4)

out1 = "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM-Materials/examples/materials/forging_simulation.png"
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out1}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — CASTING
# ═══════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 10), facecolor=BG)
fig2.suptitle("WorldSim Materials  ·  Casting Simulation Suite", fontsize=13,
              color=FG, fontweight="bold", y=1.01)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.52, wspace=0.42)

# ── A: Cooling curves (all alloys, sand mould) ───────────────────────────────
ax_A = fig2.add_subplot(gs2[0, 0])
pal2 = [ACCENT, ORANGE, GREEN, RED, PURPLE]
for (alloy, col) in zip(CASTING_PRESETS.keys(), pal2):
    r_cc = ce.cooling_curve(alloy, mould_material="sand",
                             volume_cm3=500, surface_area_cm2=300)
    t_arr = np.array(r_cc["time_s"])
    T_arr = np.array(r_cc["temperature_C"])
    T_liq = CASTING_PRESETS[alloy]["T_liq"] - 273.15
    T_sol = CASTING_PRESETS[alloy]["T_sol"] - 273.15
    ax_A.plot(t_arr / 60, T_arr, color=col, lw=1.6,
              label=CASTING_PRESETS[alloy]["name"].split()[0])
    ax_A.axhline(T_liq, color=col, lw=0.6, ls=":", alpha=0.5)
ax_A.set_xlabel("Time  [min]")
ax_A.set_ylabel("Temperature  [°C]")
ax_A.set_title("A · Cooling Curves (Sand Mould)\n500 cm³ casting", color=FG, fontsize=9, pad=5)
ax_A.legend(fontsize=7, framealpha=0.3)
ax_A.grid(True, lw=0.4)

# ── B: SDAS vs cooling rate ───────────────────────────────────────────────────
ax_B = fig2.add_subplot(gs2[0, 1])
for alloy, col in zip(["A356_Al","Gray_Iron","316L_SS","Copper_C110"], [ACCENT,ORANGE,GREEN,RED]):
    r_sdas = ce.dendrite_arm_spacing(alloy)
    ax_B.loglog(r_sdas["cooling_rates_C_s"], r_sdas["SDAS_um"],
                color=col, lw=1.8, label=CASTING_PRESETS[alloy]["name"].split()[0])
# Process zone bands
for (lo, hi, lbl, col) in [(0.1,1,"Sand",MUTED),(1,10,"Die",ORANGE),(10,100,"P.Die",GREEN),(100,1000,"RSP",PURPLE)]:
    ax_B.axvspan(lo, hi, alpha=0.06, color=col)
    ax_B.text(math.sqrt(lo*hi), 8, lbl, color=col, fontsize=6.5, ha="center", rotation=90)
ax_B.set_xlabel("Cooling rate  [°C/s]")
ax_B.set_ylabel("SDAS  [µm]")
ax_B.set_title("B · Dendrite Arm Spacing vs Cooling Rate", color=FG, fontsize=9, pad=5)
ax_B.legend(fontsize=7, framealpha=0.3)
ax_B.grid(True, which="both", lw=0.3)
import math

# ── C: Scheil microsegregation ────────────────────────────────────────────────
ax_C = fig2.add_subplot(gs2[0, 2])
for alloy, col in zip(["A356_Al","Gray_Iron","316L_SS"], [ACCENT, ORANGE, GREEN]):
    r_sch = ce.scheil_microsegregation(alloy)
    C0 = CASTING_PRESETS[alloy]["C0"]
    C_s = np.array(r_sch["C_solid_wt_pct"])
    f_s = np.array(r_sch["solid_fraction"])
    label = f"{CASTING_PRESETS[alloy]['name'].split()[0]} ({r_sch['solute']})"
    ax_C.plot(f_s, C_s / C0, color=col, lw=1.8, label=label)
ax_C.axhline(1.0, color=MUTED, lw=0.8, ls="--", label="C₀ (nominal)")
ax_C.set_xlabel("Solid fraction  f_s")
ax_C.set_ylabel("C_S / C₀  (normalised)")
ax_C.set_title("C · Scheil Microsegregation\n(solute in solid vs f_s)", color=FG, fontsize=9, pad=5)
ax_C.legend(fontsize=7, framealpha=0.3)
ax_C.set_xlim(0, 1); ax_C.set_ylim(0, None)
ax_C.grid(True, lw=0.4)

# ── D: Niyama criterion map ───────────────────────────────────────────────────
ax_D = fig2.add_subplot(gs2[1, 0])
G_arr  = np.logspace(2, 5, 80)  # K/m
cr_arr = np.logspace(-1, 3, 80)  # K/s
GG, CC = np.meshgrid(G_arr / 1000, cr_arr, indexing="ij")  # K/mm, K/s
Ny_map = GG / np.sqrt(CC)
cf2 = ax_D.contourf(np.log10(G_arr), np.log10(cr_arr), Ny_map.T,
                    levels=[0,1,2,3,5,10,20], cmap="RdYlGn",
                    extend="both")
fig2.colorbar(cf2, ax=ax_D, label="Niyama  [(K·s)^0.5/mm]")
ax_D.contour(np.log10(G_arr), np.log10(cr_arr), Ny_map.T,
             levels=[1.0, 3.0], colors=[RED, GREEN], linewidths=1.5)
ax_D.text(3.2, 1.5, "Ny=1\n(risk)", color=RED,   fontsize=7.5)
ax_D.text(4.3, 0.5, "Ny=3\n(safe)", color=GREEN, fontsize=7.5)
ax_D.set_xlabel("log₁₀(Thermal gradient  [K/m])")
ax_D.set_ylabel("log₁₀(Cooling rate  [K/s])")
ax_D.set_title("D · Niyama Criterion Map\nShrinkage porosity risk", color=FG, fontsize=9, pad=5)

# ── E: Hot tearing HTS — alloy comparison ────────────────────────────────────
ax_E = fig2.add_subplot(gs2[1, 1])
r_ht = ce.hot_tearing("316L_SS", cooling_rate_K_s=5.0, mould_constraint="medium")
alloy_names = [CASTING_PRESETS[k]["name"].split()[0] + "\n" + k.replace("_","\n")
               for k in r_ht["alloy_comparison_HTS"]]
hts_vals = list(r_ht["alloy_comparison_HTS"].values())
bar_cols = [GREEN if v < 0.1 else ORANGE if v < 0.4 else RED for v in hts_vals]
bars2 = ax_E.barh(range(len(hts_vals)), hts_vals, color=bar_cols, alpha=0.85, height=0.6)
ax_E.set_yticks(range(len(alloy_names)))
ax_E.set_yticklabels([k.replace("_","-") for k in r_ht["alloy_comparison_HTS"]], fontsize=8)
ax_E.axvline(0.10, color=ORANGE, lw=1, ls="--", alpha=0.8)
ax_E.axvline(0.40, color=RED, lw=1, ls="--", alpha=0.8)
ax_E.set_xlabel("Hot Tearing Susceptibility Index (HTS)")
ax_E.set_title("E · Hot Tearing Index\n(Clyne-Davies, 5 K/s, medium constraint)", color=FG, fontsize=9, pad=5)
leg_ht = [Patch(fc=GREEN,  label="Low (<0.10)"),
          Patch(fc=ORANGE, label="Moderate (0.10–0.40)"),
          Patch(fc=RED,    label="High (>0.40)")]
ax_E.legend(handles=leg_ht, fontsize=7, framealpha=0.3, loc="lower right")
ax_E.grid(True, axis="x", lw=0.4)

# ── F: Solidification time — modulus chart ────────────────────────────────────
ax_F = fig2.add_subplot(gs2[1, 2])
moduli = np.linspace(0.5, 8.0, 200)
for alloy, col in zip(CASTING_PRESETS.keys(), pal2):
    B = CASTING_PRESETS[alloy]["B_chvorinov"]
    t_s = B * moduli ** 2
    ax_F.plot(moduli, t_s, color=col, lw=1.8,
              label=f"{CASTING_PRESETS[alloy]['name'].split()[0]}  (B={B})")
# Typical moduli for shapes
for (label, m, col) in [("Thin plate 2cm", 0.67, MUTED),
                         ("Rod ⌀50mm",      1.25, MUTED),
                         ("Block 100mm³",   1.67, MUTED)]:
    ax_F.axvline(m, color=col, lw=0.8, ls=":", alpha=0.6)
    ax_F.text(m + 0.08, 18, label, color=col, fontsize=6.5, rotation=90)
ax_F.set_xlabel("Casting modulus  V/A  [cm]")
ax_F.set_ylabel("Solidification time  [s]")
ax_F.set_title("F · Chvorinov's Rule\nt_s = B · (V/A)²", color=FG, fontsize=9, pad=5)
ax_F.legend(fontsize=7, framealpha=0.3)
ax_F.set_ylim(0, 130)
ax_F.grid(True, lw=0.4)

out2 = "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM-Materials/examples/materials/casting_simulation.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out2}")
plt.close()
