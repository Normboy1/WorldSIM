"""GPU multi-physics blade simulation visualization.

Runs the full thermal → stress → creep → fatigue chain on the RTX 3060
for Ni57Cr15Co9Al10Ti3Ta7 and produces a publication-quality figure.
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize

sys.path.insert(0, ".")

from simlab.engines.qdgeometry.torch_gpu_physics import (
    gpu_blade_simulation, gpu_thermal_solve, gpu_thermoelastic_stress,
    gpu_creep_field, gpu_fatigue_life, _blade_mask,
)

try:
    import torch
    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE_STR = "cpu"

# ── Run the simulation ────────────────────────────────────────────────────────
print("Running GPU multi-physics blade simulation…")
t0 = time.time()

result = gpu_blade_simulation(
    nx=256, ny=128,
    T_gas=1300.0, T_coolant=650.0,
    n_thermal_iter=3000,
    t_service_hours=25_000.0,
)

fields   = result["_fields"]
T        = fields["T"]           # (128, 256)
sigma    = fields["sigma_vm"]    # MPa
creep_s  = fields["creep_strain"]
log10_Nf = fields["log10_Nf"]
bmask    = fields["blade_mask"]  # bool

elapsed = time.time() - t0
print(f"Sim complete in {elapsed:.2f}s on {DEVICE_STR}")

# ── Masked arrays ─────────────────────────────────────────────────────────────
T_show     = np.where(bmask, T,        np.nan)
sig_show   = np.where(bmask, sigma,    np.nan)
creep_show = np.where(bmask & (creep_s > 0), np.log10(creep_s.clip(1e-20)), np.nan)
nf_show    = np.where(bmask, log10_Nf, np.nan)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14), facecolor="white")
fig.patch.set_facecolor("white")

gs = GridSpec(3, 4, figure=fig,
              wspace=0.40, hspace=0.48,
              left=0.06, right=0.97,
              top=0.91, bottom=0.07)

ax_T   = fig.add_subplot(gs[0, :2])   # temperature — wide
ax_sig = fig.add_subplot(gs[0, 2:])   # stress — wide
ax_cr  = fig.add_subplot(gs[1, :2])   # creep — wide
ax_nf  = fig.add_subplot(gs[1, 2:])   # fatigue — wide
ax_cnv = fig.add_subplot(gs[2, 0])    # convergence
ax_bar = fig.add_subplot(gs[2, 1])    # pass/fail bars
ax_col = fig.add_subplot(gs[2, 2])    # throughput comparison
ax_ann = fig.add_subplot(gs[2, 3])    # annotation / properties card

for ax in [ax_T, ax_sig, ax_cr, ax_nf, ax_cnv, ax_bar, ax_col, ax_ann]:
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")

def _blade_outline(ax, bmask_):
    """Draw blade boundary as a contour line."""
    from matplotlib.contour import QuadContourSet
    ny, nx = bmask_.shape
    X_g, Y_g = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    ax.contour(X_g, Y_g, bmask_.astype(float), levels=[0.5],
               colors=["#333333"], linewidths=[0.8])

ny, nx_val = T.shape
X_g = np.linspace(0, 1, nx_val)
Y_g = np.linspace(0, 1, ny)

# ── Panel 1: Temperature ──────────────────────────────────────────────────────
im1 = ax_T.pcolormesh(X_g, Y_g, T_show, cmap="plasma",
                      vmin=650, vmax=1300, shading="auto")
cb1 = plt.colorbar(im1, ax=ax_T, fraction=0.035, pad=0.02)
cb1.set_label("Temperature (°C)", fontsize=8)
_blade_outline(ax_T, bmask)
ax_T.set_aspect("auto")
ax_T.set_title("① Steady-State Temperature Field  (T_gas=1300°C,  T_cool=650°C)",
               fontsize=10, fontweight="bold", color="#1a1a2e", pad=6)
ax_T.set_xlabel("x / chord", fontsize=8)
ax_T.set_ylabel("y / chord", fontsize=8)

# Annotate cooling channels
for xi in [0.15, 0.25, 0.40, 0.55, 0.70, 0.82]:
    ax_T.annotate("", xy=(xi, 0.5), xytext=(xi, 0.75),
                  arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2))
ax_T.text(0.50, 0.85, "film cooling channels →", ha="center",
          fontsize=7, color="cyan", transform=ax_T.transAxes)

# ── Panel 2: Thermoelastic Stress ─────────────────────────────────────────────
im2 = ax_sig.pcolormesh(X_g, Y_g, sig_show, cmap="RdYlGn_r",
                        vmin=0, vmax=1330, shading="auto")
cb2 = plt.colorbar(im2, ax=ax_sig, fraction=0.035, pad=0.02)
cb2.set_label("σ_vm  (MPa)", fontsize=8)
_blade_outline(ax_sig, bmask)

# Yield line
cb2.ax.axhline(571.0, color="blue", lw=1.5, ls="--")
cb2.ax.text(1.05, 571.0 / 1330.0, "yield\n571 MPa", transform=cb2.ax.transAxes,
            fontsize=6.5, color="blue", va="center")

ax_sig.set_title("② Von Mises Stress  (gradient-induced, plane-strain)",
                 fontsize=10, fontweight="bold", color="#1a1a2e", pad=6)
ax_sig.set_xlabel("x / chord", fontsize=8)
ax_sig.set_ylabel("y / chord", fontsize=8)

# Safety factor text
sf = result["stress"]["safety_factor"]
ax_sig.text(0.02, 0.94, f"Safety factor = {sf:.2f}  (target >1.2)",
            transform=ax_sig.transAxes, fontsize=8,
            color="crimson" if sf < 1.2 else "green",
            bbox=dict(fc="white", alpha=0.8, boxstyle="round,pad=0.2"))

# ── Panel 3: Creep accumulated strain (log10) ─────────────────────────────────
cmin, cmax = -12, 4
im3 = ax_cr.pcolormesh(X_g, Y_g, creep_show.clip(cmin, cmax),
                        cmap="YlOrRd", vmin=cmin, vmax=cmax, shading="auto")
cb3 = plt.colorbar(im3, ax=ax_cr, fraction=0.035, pad=0.02)
cb3.set_label("log₁₀(ε_creep)", fontsize=8)
_blade_outline(ax_cr, bmask)
ax_cr.set_title("③ Creep Strain After 25,000 h  (Norton–Bailey, log scale)",
                fontsize=10, fontweight="bold", color="#1a1a2e", pad=6)
ax_cr.set_xlabel("x / chord", fontsize=8)
ax_cr.set_ylabel("y / chord", fontsize=8)

# Design limit line
design_log = np.log10(0.02)  # 2% limit
cb3.ax.axhline(design_log, color="blue", lw=1.5, ls="--")
cb3.ax.text(1.05, (design_log - cmin) / (cmax - cmin),
            "2% limit", transform=cb3.ax.transAxes,
            fontsize=6.5, color="blue", va="center")

# ── Panel 4: Fatigue life ─────────────────────────────────────────────────────
im4 = ax_nf.pcolormesh(X_g, Y_g, nf_show.clip(0, 12),
                        cmap="RdYlGn", vmin=0, vmax=12, shading="auto")
cb4 = plt.colorbar(im4, ax=ax_nf, fraction=0.035, pad=0.02)
cb4.set_label("log₁₀(N_f cycles)", fontsize=8)
_blade_outline(ax_nf, bmask)
ax_nf.set_title("④ HCF Fatigue Life  (Basquin, R=0.1)",
                fontsize=10, fontweight="bold", color="#1a1a2e", pad=6)
ax_nf.set_xlabel("x / chord", fontsize=8)
ax_nf.set_ylabel("y / chord", fontsize=8)

# 1e7 cycle target
cb4.ax.axhline(7.0, color="blue", lw=1.5, ls="--")
cb4.ax.text(1.05, 7.0 / 12.0, "10⁷\ntarget", transform=cb4.ax.transAxes,
            fontsize=6.5, color="blue", va="center")

# ── Panel 5: Convergence history ─────────────────────────────────────────────
conv = result["thermal"]["convergence_history"]
iters = [i * 100 for i in range(len(conv))]
ax_cnv.semilogy(iters, conv, "b-o", ms=4, lw=1.8)
ax_cnv.axhline(1e-4, color="red", ls="--", lw=1.2, label="tol 1e-4")
ax_cnv.set_xlabel("Iteration", fontsize=8)
ax_cnv.set_ylabel("Max ΔT", fontsize=8)
ax_cnv.set_title("⑤ Jacobi FD Convergence", fontsize=9, fontweight="bold")
ax_cnv.legend(fontsize=7)
ax_cnv.grid(True, alpha=0.3)
ax_cnv.tick_params(labelsize=7)

# ── Panel 6: Pass/fail bars ───────────────────────────────────────────────────
pf = result["pass_fail"]
checks = list(pf.keys())
labels = ["Thermal\nmargin\n>200°C", "Stress\nsafety\n>1.2", "Creep\n<2%",
          "HCF life\n>10⁶ cyc"]
vals = [1 if pf[c] else 0 for c in checks]
colors = ["#27ae60" if v else "#e74c3c" for v in vals]
bars = ax_bar.bar(range(len(checks)), [1]*len(checks), color=colors, width=0.6,
                  edgecolor="white", linewidth=2)
for i, (v, lbl) in enumerate(zip(vals, labels)):
    ax_bar.text(i, 0.5, "PASS" if v else "FAIL",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white")
ax_bar.set_xticks(range(len(checks)))
ax_bar.set_xticklabels(labels, fontsize=7)
ax_bar.set_ylim(0, 1.3)
ax_bar.set_yticks([])
ax_bar.set_title("⑥ Multi-Physics Pass/Fail", fontsize=9, fontweight="bold")
ax_bar.text(0.5, 1.12,
            "Overall: PASS" if result["overall_pass"] else "Overall: FAIL — cooling redesign needed",
            ha="center", transform=ax_bar.transAxes, fontsize=8,
            color="#27ae60" if result["overall_pass"] else "#c0392b",
            fontweight="bold")

# ── Panel 7: Solver timings ───────────────────────────────────────────────────
solver_names = ["Thermal\nJacobi FD", "Thermoelastic\nStress", "Norton–Bailey\nCreep",
                "Basquin\nFatigue"]
timings = [
    result["thermal"]["solve_time_s"],
    result["stress"]["solve_time_s"],
    result["creep"]["solve_time_s"],
    result["fatigue"]["solve_time_s"],
]
bar_colors = ["#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
bs = ax_col.bar(range(4), timings, color=bar_colors, width=0.6,
                edgecolor="white", linewidth=2)
for i, t in enumerate(timings):
    ax_col.text(i, t + 0.003, f"{t*1000:.1f} ms" if t < 0.1 else f"{t:.3f} s",
                ha="center", fontsize=7.5, fontweight="bold")
ax_col.set_xticks(range(4))
ax_col.set_xticklabels(solver_names, fontsize=7)
ax_col.set_ylabel("Wall time (s)", fontsize=8)
ax_col.set_title(f"⑦ GPU Solver Timings  ({DEVICE_STR.upper()})",
                 fontsize=9, fontweight="bold")
ax_col.text(0.97, 0.94, f"Total: {result['total_sim_time_s']:.3f}s",
            ha="right", transform=ax_col.transAxes, fontsize=8, fontweight="bold",
            color="#2c3e50",
            bbox=dict(fc="#ecf0f1", boxstyle="round,pad=0.3"))
ax_col.grid(axis="y", alpha=0.3)
ax_col.tick_params(labelsize=7)

# ── Panel 8: Alloy properties card ───────────────────────────────────────────
ax_ann.axis("off")
props = [
    ("Alloy",           "Ni₅₇Cr₁₅Co₉Al₁₀Ti₃Ta₇"),
    ("Crystal",         "FCC γ  +  L1₂ γ' precipitates"),
    ("Density",         "8.44 g/cm³"),
    ("T_solidus",       "1534 °C"),
    ("T_metal (mean)",  f"{result['thermal']['T_mean_blade_C']} °C"),
    ("T_metal (peak)",  f"{result['thermal']['T_peak_blade_C']} °C"),
    ("Margin to melt",  f"{result['thermal']['margin_to_solidus_C']} °C  ✓"),
    ("σ_vm peak",       f"{result['stress']['sigma_vm_peak_MPa']} MPa"),
    ("Yield strength",  "571 MPa"),
    ("Device",          f"{DEVICE_STR.upper()}  (256×128 grid)"),
    ("Sim time",        f"{result['total_sim_time_s']:.3f} s total"),
    ("Grid cells",      f"{256*128:,}  nodes"),
]
ax_ann.text(0.5, 1.01, "⑧  Ni57 Alloy — Simulation Summary",
            ha="center", va="top", transform=ax_ann.transAxes,
            fontsize=9, fontweight="bold", color="#1a1a2e")
y0 = 0.92
for i, (lbl, val) in enumerate(props):
    ax_ann.text(0.05, y0 - i*0.074, f"{lbl}:", transform=ax_ann.transAxes,
                fontsize=7.5, color="#555555")
    ax_ann.text(0.52, y0 - i*0.074, val, transform=ax_ann.transAxes,
                fontsize=7.5, color="#1a1a2e", fontweight="bold")

ax_ann.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.02", linewidth=1.2,
    edgecolor="#aaaaaa", facecolor="#f8f9fa",
    transform=ax_ann.transAxes, clip_on=False, zorder=0))

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "WorldSIM  ·  GPU Multi-Physics Blade Simulation  ·  Ni₅₇Cr₁₅Co₉Al₁₀Ti₃Ta₇ Superalloy",
    fontsize=13, fontweight="bold", color="#1a1a2e", y=0.975,
)

out = "docs/figures/blade_physics.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out}")
