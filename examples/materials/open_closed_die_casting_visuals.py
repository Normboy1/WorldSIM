"""Open-die / closed-die forging + casting applications visual suite."""

from __future__ import annotations
import sys, math, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simlab.engines.materials.forging import ForgingEngine, FORGING_PRESETS
from simlab.engines.materials.casting import CastingEngine, CASTING_PRESETS

FE = ForgingEngine()
CE = CastingEngine()

DARK = "#0d0d0d"
PANEL = "#141414"
ACCENT = "#e0e0e0"
COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8", "#80cbc4"]

def _style(fig, axes_flat):
    fig.patch.set_facecolor(DARK)
    for ax in axes_flat:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=ACCENT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.xaxis.label.set_color(ACCENT)
        ax.yaxis.label.set_color(ACCENT)
        ax.title.set_color(ACCENT)
        ax.grid(True, color="#222222", linewidth=0.5, linestyle="--")

# ======================================================================
# Figure 1 — Open-die vs Closed-die Forging
# ======================================================================
fig1 = plt.figure(figsize=(18, 11), facecolor=DARK)
fig1.suptitle("WorldSim Materials  ·  Open-Die & Closed-Die Forging Suite",
              color=ACCENT, fontsize=13, fontweight="bold", y=0.97)
gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.42, wspace=0.38,
                        left=0.07, right=0.97, top=0.92, bottom=0.07)

ax_A = fig1.add_subplot(gs1[0, 0])  # open-die pass schedule — force per pass
ax_B = fig1.add_subplot(gs1[0, 1])  # open-die billet geometry evolution
ax_C = fig1.add_subplot(gs1[0, 2])  # open-die penetration efficiency
ax_D = fig1.add_subplot(gs1[1, 0])  # closed-die flash pressure vs land ratio
ax_E = fig1.add_subplot(gs1[1, 1])  # closed-die force sweep (force vs reduction)
ax_F = fig1.add_subplot(gs1[1, 2])  # closed-die: part + flash force breakdown

# --- A: Open-die — force per pass for 3 materials ---
materials_od = ["AISI_4340", "Ti6Al4V", "Inconel_718"]
labels_od    = ["AISI 4340", "Ti-6Al-4V", "Inconel 718"]
for mat, lbl, col in zip(materials_od, labels_od, COLORS):
    res = FE.open_die_pass_schedule(
        mat,
        billet_length_mm=400, billet_width_mm=180, billet_height_mm=180,
        target_thickness_mm=60, temperature_C=1100,
        strain_rate=1.0, pass_reduction_pct=15.0,
    )
    passes = res["pass_schedule"]
    ax_A.plot([p["pass"] for p in passes],
              [p["force_MN"] for p in passes],
              color=col, lw=1.8, marker="o", ms=3.5, label=lbl)
    # Mark reheats
    for p in passes:
        if p["reheat_after"]:
            ax_A.axvline(p["pass"] + 0.5, color=col, lw=0.6, alpha=0.35, linestyle=":")

ax_A.set_xlabel("Pass number")
ax_A.set_ylabel("Forging force [MN]")
ax_A.set_title("A — Open-Die: Force per Pass (180→60 mm)")
ax_A.legend(fontsize=6.5, framealpha=0.2)

# --- B: Billet geometry evolution (height + width + length) ---
res_b = FE.open_die_pass_schedule(
    "AISI_4340",
    billet_length_mm=400, billet_width_mm=180, billet_height_mm=180,
    target_thickness_mm=50, temperature_C=1100,
    strain_rate=1.0, pass_reduction_pct=12.0,
)
passes_b = res_b["pass_schedule"]
pass_nums = [0] + [p["pass"] for p in passes_b]
heights   = [180.0] + [p["h_out_mm"] for p in passes_b]
widths    = [180.0] + [p["b_out_mm"] for p in passes_b]
lengths   = [400.0] + [p["L_out_mm"] for p in passes_b]

ax_B.plot(pass_nums, heights, color=COLORS[0], lw=1.8, label="Height h")
ax_B.plot(pass_nums, widths,  color=COLORS[1], lw=1.8, label="Width b")
ax_B2 = ax_B.twinx()
ax_B2.plot(pass_nums, lengths, color=COLORS[2], lw=1.4, linestyle="--", label="Length L")
ax_B2.set_ylabel("Length [mm]", color=COLORS[2], fontsize=7)
ax_B2.tick_params(colors=COLORS[2], labelsize=7)
ax_B2.yaxis.label.set_color(COLORS[2])
ax_B.set_xlabel("Pass number")
ax_B.set_ylabel("Cross-section [mm]")
ax_B.set_title("B — Open-Die: Billet Shape Evolution (AISI 4340)")
lines1, lbs1 = ax_B.get_legend_handles_labels()
lines2, lbs2 = ax_B2.get_legend_handles_labels()
ax_B.legend(lines1 + lines2, lbs1 + lbs2, fontsize=6.5, framealpha=0.2)

# --- C: Penetration efficiency vs pass for different pass reductions ---
for red, col in zip([10.0, 20.0, 30.0], COLORS[:3]):
    res_c = FE.open_die_pass_schedule(
        "AISI_4340",
        billet_length_mm=400, billet_width_mm=200, billet_height_mm=200,
        target_thickness_mm=60, temperature_C=1100,
        strain_rate=1.0, pass_reduction_pct=red,
    )
    ps = res_c["pass_schedule"]
    ax_C.plot([p["pass"] for p in ps],
              [p["penetration_pct"] for p in ps],
              color=col, lw=1.8, marker="s", ms=3, label=f"{red:.0f}% / pass")

ax_C.set_xlabel("Pass number")
ax_C.set_ylabel("Penetration efficiency [%]")
ax_C.set_title("C — Open-Die: Deformation Penetration")
ax_C.legend(fontsize=6.5, framealpha=0.2)
ax_C.set_ylim(0, 100)

# --- D: Closed-die flash pressure vs land ratio (3 materials) ---
R_vals = np.linspace(1.0, 10.0, 60)
for mat, lbl, col in zip(["AISI_1020", "AISI_4340", "Inconel_718"],
                          ["AISI 1020", "AISI 4340", "Inconel 718"], COLORS):
    # Re-use closed_die_analysis to get sigma_f then compute p_flash sweep
    base = FE.closed_die_analysis(mat, temperature_C=1100)
    sf = base["flow_stress_MPa"]
    p_flash_arr = sf * (1.0 + R_vals / 4.0)
    ax_D.plot(R_vals, p_flash_arr, color=col, lw=1.8, label=lbl)

ax_D.axvline(4.0, color="white", lw=0.8, linestyle="--", alpha=0.5)
ax_D.text(4.1, ax_D.get_ylim()[0] if ax_D.get_ylim()[0] > 0 else 100,
          "R=4 (typical)", color="white", fontsize=6, alpha=0.7)
ax_D.set_xlabel("Flash land ratio  R = w_f / t_f")
ax_D.set_ylabel("Flash pressure [MPa]")
ax_D.set_title("D — Closed-Die: Flash Pressure vs Land Ratio")
ax_D.legend(fontsize=6.5, framealpha=0.2)

# --- E: Closed-die total force vs reduction (force sweep) ---
for mat, lbl, col in zip(["AISI_1020", "AISI_4340", "Ti6Al4V", "Inconel_718"],
                          ["AISI 1020", "AISI 4340", "Ti-6Al-4V", "Inconel 718"],
                          COLORS):
    res_e = FE.closed_die_analysis(mat, temperature_C=1100, strain_rate=5.0)
    reds = res_e["reduction_pct_sweep"]
    Fs   = res_e["force_sweep_MN"]
    ax_E.plot(reds, Fs, color=col, lw=1.8, label=lbl)

ax_E.set_xlabel("Die reduction [%]")
ax_E.set_ylabel("Total forging force [MN]")
ax_E.set_title("E — Closed-Die: Force vs Reduction")
ax_E.legend(fontsize=6.5, framealpha=0.2)

# --- F: Part force vs flash force breakdown (bar chart, 4 materials) ---
mats_f = ["AISI_1020", "AISI_4340", "Ti6Al4V", "Inconel_718"]
lbls_f = ["AISI 1020", "AISI 4340", "Ti-6Al-4V", "Inconel 718"]
part_forces, flash_forces, trim_forces = [], [], []
for mat in mats_f:
    r = FE.closed_die_analysis(mat, temperature_C=1100, strain_rate=5.0)
    part_forces.append(r["forging_force_part_MN"])
    flash_forces.append(r["forging_force_flash_MN"])
    trim_forces.append(r["trimming_force_MN"])

x = np.arange(len(mats_f))
w = 0.25
ax_F.bar(x - w, part_forces, w, label="Part body", color=COLORS[0], alpha=0.85)
ax_F.bar(x,     flash_forces, w, label="Flash zone", color=COLORS[1], alpha=0.85)
ax_F.bar(x + w, trim_forces,  w, label="Trimming",   color=COLORS[3], alpha=0.85)
ax_F.set_xticks(x)
ax_F.set_xticklabels(lbls_f, rotation=12, fontsize=7)
ax_F.set_ylabel("Force [MN]")
ax_F.set_title("F — Closed-Die: Force Breakdown by Zone")
ax_F.legend(fontsize=6.5, framealpha=0.2)

_style(fig1, [ax_A, ax_B, ax_C, ax_D, ax_E, ax_F])
# Twin axis for B needs manual styling
ax_B2.set_facecolor(PANEL)
for sp in ax_B2.spines.values():
    sp.set_color("#333333")

out1 = ROOT / "examples/materials/open_closed_die_forging.png"
fig1.savefig(out1, dpi=140, bbox_inches="tight", facecolor=DARK)
print(f"Saved → {out1}")
plt.close(fig1)

# ======================================================================
# Figure 2 — Casting Applications
# ======================================================================
fig2 = plt.figure(figsize=(18, 12), facecolor=DARK)
fig2.suptitle("WorldSim Materials  ·  Casting Applications Suite",
              color=ACCENT, fontsize=13, fontweight="bold", y=0.97)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.38,
                        left=0.07, right=0.97, top=0.92, bottom=0.07)

ax_G = fig2.add_subplot(gs2[0, 0])  # investment casting: wall vs superheat
ax_H = fig2.add_subplot(gs2[0, 1])  # investment casting: shell layers + solidification time
ax_I = fig2.add_subplot(gs2[0, 2])  # die casting: gate velocity vs porosity index
ax_J = fig2.add_subplot(gs2[1, 0])  # application spider / summary bars
ax_K = fig2.add_subplot(gs2[1, 1])  # investment grain structure map
ax_L = fig2.add_subplot(gs2[1, 2])  # die casting: clamping force vs area

# --- G: Investment casting — superheat vs wall thickness (4 alloys) ---
wall_range = np.linspace(0.5, 8.0, 80)
for alloy, col in zip(["A356_Al", "316L_SS", "Ti6Al4V_cast", "Copper_C110"],
                       COLORS[:4]):
    superheats = []
    for wt in wall_range:
        r = CE.investment_casting(alloy, wall_thickness_mm=float(wt),
                                  part_volume_cm3=50.0, surface_area_cm2=120.0,
                                  vacuum=True)
        superheats.append(r["superheat_C"])
    p_name = CASTING_PRESETS[alloy]["name"].split()[0]
    ax_G.plot(wall_range, superheats, color=col, lw=1.8, label=p_name)
    # min wall thickness line
    r0 = CE.investment_casting(alloy, part_volume_cm3=50.0, surface_area_cm2=120.0, vacuum=True)
    ax_G.axvline(r0["min_wall_thickness_mm"], color=col, lw=0.6, linestyle=":", alpha=0.5)

ax_G.set_xlabel("Wall thickness [mm]")
ax_G.set_ylabel("Required superheat [°C]")
ax_G.set_title("G — Investment Casting: Superheat vs Wall")
ax_G.legend(fontsize=6.5, framealpha=0.2)

# --- H: Shell layers & solidification time vs part volume (investment) ---
vols = np.linspace(10.0, 500.0, 60)
sa_frac = 4.5   # surface_area = vol * 4.5 / vol^(1/3) rough scaling
for alloy, col in zip(["Ti6Al4V_cast", "316L_SS"], COLORS[2:4]):
    layers, sol_times = [], []
    for v in vols:
        sa = 4.5 * v ** (2.0 / 3.0)
        r = CE.investment_casting(alloy, wall_thickness_mm=2.0,
                                  part_volume_cm3=float(v), surface_area_cm2=sa, vacuum=True)
        layers.append(r["n_shell_layers"])
        sol_times.append(r["solidification_time_s"])
    p_name = CASTING_PRESETS[alloy]["name"].split()[0]
    ax_H.plot(vols, sol_times, color=col, lw=1.8, label=f"{p_name} sol. time")
ax_H_r = ax_H.twinx()
ax_H_r.plot(vols, layers, color=COLORS[4], lw=1.4, linestyle="--", label="Shell layers (Ti)")
ax_H_r.set_ylabel("Shell layers", color=COLORS[4], fontsize=7)
ax_H_r.tick_params(colors=COLORS[4], labelsize=7)
ax_H_r.yaxis.label.set_color(COLORS[4])

ax_H.set_xlabel("Part volume [cm³]")
ax_H.set_ylabel("Solidification time [s]")
ax_H.set_title("H — Investment: Shell Layers & Solidification")
lines_h, lbs_h = ax_H.get_legend_handles_labels()
lines_hr, lbs_hr = ax_H_r.get_legend_handles_labels()
ax_H.legend(lines_h + lines_hr, lbs_h + lbs_hr, fontsize=6.5, framealpha=0.2)

# --- I: Die casting — porosity risk index vs gate velocity ---
r_dc = CE.die_casting("A356_Al", part_volume_cm3=200.0,
                       projected_area_mm2=18000.0, gate_area_mm2=400.0,
                       plunger_velocity_m_s=3.0, intensification_pressure_MPa=70.0)
v_range = np.array(r_dc["gate_velocity_range_m_s"])
por_risk = np.array(r_dc["porosity_risk_index"])

# Color by risk level
cmap_risk = LinearSegmentedColormap.from_list("risk", ["#4fc3f7", "#ffb74d", "#f44336"])
for i in range(len(v_range) - 1):
    ax_I.fill_betweenx([0, por_risk[i]], v_range[i], v_range[i+1],
                        color=cmap_risk(por_risk[i]), alpha=0.7)
ax_I.plot(v_range, por_risk, color="white", lw=1.5)
ax_I.axhline(0.5, color="#ffb74d", lw=0.8, linestyle="--")
ax_I.text(12, 0.52, "Moderate risk", color="#ffb74d", fontsize=6.5)
ax_I.axhline(0.8, color="#f44336", lw=0.8, linestyle="--")
ax_I.text(12, 0.82, "High risk", color="#f44336", fontsize=6.5)
ax_I.set_xlabel("Gate velocity [m/s]")
ax_I.set_ylabel("Porosity risk index")
ax_I.set_title(f"I — HPDC: Porosity Risk vs Gate Velocity (A356 Al)")
ax_I.set_ylim(0, 1.05)

# --- J: Application summary bars (5 cases) ---
app_keys  = ["engine_block_al", "turbine_blade_ni", "wheel_hub_iron",
             "structural_bracket_ti", "heat_exchanger_cu"]
app_short = ["Engine\nBlock Al", "Turbine\nBlade Ti", "Wheel Hub\nGray Fe",
             "Bracket\nTi-6-4V", "Heat Exch\nCu C110"]

niy_vals, ht_vals, sdas_vals = [], [], []
for app in app_keys:
    r = CE.application_case(app)
    niy_vals.append(min(r["niyama_value"], 6.0))
    ht_vals.append(r["hot_tearing_index"])
    sdas_vals.append(r["sdas_um"] / 100.0)   # normalise to 0-1 scale

x = np.arange(len(app_keys))
w = 0.25
ax_J.bar(x - w, niy_vals,  w, label="Niyama (↑ better)", color=COLORS[0], alpha=0.85)
ax_J.bar(x,     ht_vals,   w, label="HTS index (↓ better)", color=COLORS[3], alpha=0.85)
ax_J.bar(x + w, sdas_vals, w, label="SDAS /100 µm (↓ finer)", color=COLORS[1], alpha=0.85)
ax_J.axhline(1.0, color=COLORS[0], lw=0.7, linestyle="--", alpha=0.5)
ax_J.axhline(0.4, color=COLORS[3], lw=0.7, linestyle="--", alpha=0.5)
ax_J.set_xticks(x)
ax_J.set_xticklabels(app_short, fontsize=6.5)
ax_J.set_ylabel("Index value")
ax_J.set_title("J — Application Summary: Quality Indices")
ax_J.legend(fontsize=6.5, framealpha=0.2)

# --- K: Investment grain structure map (T_pour vs wall thickness) ---
wall_g  = np.linspace(0.5, 8.0, 40)
T_off_g = np.linspace(20.0, 150.0, 40)
WW, TT  = np.meshgrid(wall_g, T_off_g)
struct_map = np.zeros_like(WW)
for i in range(40):
    for j in range(40):
        k_c = 6.7   # Ti k_cond
        cr = k_c * (TT[i,j] + 1660.0 - 25.0) / (4420.0 * 526.0 * (WW[i,j] / 10.0) * 0.01)
        if cr > 50.0:
            struct_map[i,j] = 3.0   # fine equiaxed
        elif cr > 5.0:
            struct_map[i,j] = 2.0   # mixed
        else:
            struct_map[i,j] = 1.0   # coarse columnar

cmap_gs = LinearSegmentedColormap.from_list("gs", ["#4a148c", "#4fc3f7", "#81c784"])
cf = ax_K.contourf(WW, TT, struct_map, levels=[0.5,1.5,2.5,3.5], cmap=cmap_gs, alpha=0.85)
ax_K.set_xlabel("Wall thickness [mm]")
ax_K.set_ylabel("Superheat [°C]")
ax_K.set_title("K — Ti-6Al-4V Investment: Grain Structure Map")
legend_els = [
    mpatches.Patch(color="#4a148c", label="Coarse columnar"),
    mpatches.Patch(color="#4fc3f7", label="Mixed col./equiaxed"),
    mpatches.Patch(color="#81c784", label="Fine equiaxed"),
]
ax_K.legend(handles=legend_els, fontsize=6.5, framealpha=0.3)

# --- L: HPDC clamping force vs projected area (A356, with pressure lines) ---
areas = np.linspace(5000.0, 80000.0, 80)   # mm²
for p_int, col in zip([50.0, 70.0, 100.0, 140.0], COLORS[:4]):
    F_arr = p_int * 1e6 * areas * 1e-6 / 1e6   # MN
    ax_L.plot(areas / 100.0, F_arr, color=col, lw=1.8, label=f"{p_int:.0f} MPa")

# Typical machine ratings
for Frating, lbl in [(20.0, "200t"), (50.0, "500t"), (100.0, "1000t")]:
    ax_L.axhline(Frating / 101.97, color="white", lw=0.6, linestyle=":", alpha=0.4)
    ax_L.text(areas[-1] / 100.0, Frating / 101.97 + 0.05, lbl,
              color="white", fontsize=6, alpha=0.5, ha="right")

ax_L.set_xlabel("Projected area [cm²]")
ax_L.set_ylabel("Clamping force [MN]")
ax_L.set_title("L — HPDC: Clamping Force vs Projected Area")
ax_L.legend(title="Intens. pressure", fontsize=6.5, framealpha=0.2, title_fontsize=6.5)

_style(fig2, [ax_G, ax_H, ax_I, ax_J, ax_K, ax_L])
ax_H_r.set_facecolor(PANEL)
for sp in ax_H_r.spines.values():
    sp.set_color("#333333")

out2 = ROOT / "examples/materials/casting_applications.png"
fig2.savefig(out2, dpi=140, bbox_inches="tight", facecolor=DARK)
print(f"Saved → {out2}")
plt.close(fig2)
