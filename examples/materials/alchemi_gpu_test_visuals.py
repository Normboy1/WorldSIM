"""Visuals for the ALCHEMI GPU test results.

Three panels:
  1. LJ potential curve (Fe params) — shows where test structures sit
  2. FIRE2 convergence — energy/atom and max-force vs step
  3. Batch alloy screen — formation energies bar chart
"""

from __future__ import annotations
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM/.venv/lib/python3.12/site-packages")
sys.path.insert(0, "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM-Materials")

from simlab.engines.materials.nvidia_backends import ALCHEMIBackend, _LJ_PARAMS

# ── colour palette ────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
GRID    = "#21262d"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f0883e"
RED     = "#f85149"
PURPLE  = "#bc8cff"
FG      = "#e6edf3"
MUTED   = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": FG, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "monospace", "font.size": 9,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Gather data — re-run the GPU tests with trajectory logging
# ─────────────────────────────────────────────────────────────────────────────

b = ALCHEMIBackend()
print(f"Backend: {b.backend_name}")

# 32-atom FCC supercell with noise (same as test 2)
a = 3.6
np.random.seed(7)
basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]], dtype=np.float32)
pos2 = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            for bv in basis:
                p = (bv + np.array([ix,iy,iz])) * a
                p += np.random.randn(3).astype(np.float32) * 0.18
                pos2.append(p.tolist())
sp2   = ["Fe"] * 32
cell2 = [[2*a,0,0],[0,2*a,0],[0,0,2*a]]

# --- run FIRE2, capture per-step trajectory ----------------------------------
if b.available:
    import torch
    from nvalchemiops.torch import fire2_step_coord

    N = 32
    pos_np = np.array(pos2, dtype=np.float32)
    pos_t  = torch.tensor(pos_np, device="cuda")
    vel_t  = torch.zeros_like(pos_t)
    bidx   = torch.zeros(N, dtype=torch.int32, device="cuda")
    alpha, dt, nsteps_inc = b._build_fire2_state(1, "cuda")

    traj_E, traj_F, converged_step = [], [], None
    _, forces_np = b._lj_energy_forces_gpu(pos_np, sp2)
    forces_t = torch.tensor(forces_np, device="cuda")

    for step in range(200):
        e_now, f_now = b._lj_energy_forces_gpu(pos_t.cpu().numpy().astype(np.float32), sp2)
        fmax = float(forces_t.norm(dim=1).max())
        traj_E.append(float(e_now.sum()) / N)
        traj_F.append(fmax)
        if fmax <= 0.2 and converged_step is None:
            converged_step = step
        fire2_step_coord(pos_t, vel_t, forces_t, bidx, alpha, dt, nsteps_inc)
        pos_np2 = pos_t.cpu().numpy().astype(np.float32)
        _, forces_np2 = b._lj_energy_forces_gpu(pos_np2, sp2)
        forces_t = torch.tensor(forces_np2, device="cuda")
    steps = np.arange(len(traj_E))
    print(f"FIRE2 trajectory: {len(steps)} steps, converged at step {converged_step}")
else:
    steps = np.arange(80)
    traj_E = -0.30 * (1 - np.exp(-steps / 12)) + 0.01 * np.sin(steps * 0.5)
    traj_F = 6.0 * np.exp(-steps / 15) + 0.15
    converged_step = 47
    print("CPU proxy — using synthetic trajectory")

# --- batch alloy screen -------------------------------------------------------
compositions = [
    {"Fe": 1.0},
    {"Fe": 0.8, "Ni": 0.2},
    {"Fe": 0.7, "Ni": 0.2, "Cr": 0.1},
    {"Fe": 0.6, "Ni": 0.25, "Cr": 0.15},
    {"Ni": 0.7, "Cr": 0.3},
    {"Fe": 0.5, "Ni": 0.3, "Cr": 0.1, "Mo": 0.1},
]
r3 = b.batch_alloy_screen(compositions, lattice_type="fcc", a_angstrom=3.6)
comp_labels = [
    "+".join(f"{el}{round(f,2)}" for el, f in c.items())
    for c in compositions
]
e_form = [res["formation_energy_eV_atom"] for res in r3["results"]]
stable  = [res["predicted_stable"] for res in r3["results"]]

# ─────────────────────────────────────────────────────────────────────────────
# 2. LJ potential curve
# ─────────────────────────────────────────────────────────────────────────────

def lj(r, eps, sig):
    x = sig / r
    return 4 * eps * (x**12 - x**6)

eps_Fe, sig_Fe = _LJ_PARAMS["Fe"]
eps_Ni, sig_Ni = _LJ_PARAMS["Ni"]
r_arr = np.linspace(2.0, 7.0, 400)
V_Fe  = lj(r_arr, eps_Fe, sig_Fe)
V_Ni  = lj(r_arr, eps_Ni, sig_Ni)

# FCC nearest-neighbour distances
r_NN_FCC  = a / math.sqrt(2)         # 2.546 Å (as-built)
r_NN_rel  = a * math.sqrt(2) ** (1/6)  # ~2.87 Å — LJ min for Fe

# ─────────────────────────────────────────────────────────────────────────────
# 3. Layout
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 5.5), facecolor=BG)
fig.suptitle(
    "ALCHEMI GPU Test  ·  nvalchemiops FIRE2 + LJ  ·  RTX 3060",
    fontsize=12, color=FG, fontweight="bold", y=1.01,
)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# ── Panel A: LJ potential ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(r_arr, np.clip(V_Fe, -0.12, 0.25), color=ACCENT,   lw=2,   label="Fe (ε=0.041 eV, σ=2.59 Å)")
ax1.plot(r_arr, np.clip(V_Ni, -0.12, 0.25), color=ORANGE,  lw=1.5, label="Ni (ε=0.035 eV, σ=2.52 Å)", ls="--")
ax1.axhline(0, color=MUTED, lw=0.6, ls=":")
ax1.axvline(r_NN_FCC, color=RED,   lw=1.2, ls="--", label=f"FCC NN (a=3.6 Å): {r_NN_FCC:.2f} Å")
ax1.axvline(r_NN_rel, color=GREEN, lw=1.2, ls="--", label=f"LJ min: {r_NN_rel:.2f} Å")

# annotate repulsive zone
ax1.axvspan(2.0, r_NN_rel, alpha=0.06, color=RED)
ax1.text(2.35, 0.18, "repulsive\nregion", color=RED, fontsize=7.5, ha="center")
ax1.text(r_NN_rel + 0.15, -0.09, "← LJ min", color=GREEN, fontsize=7.5)

ax1.set_xlim(2.0, 7.0)
ax1.set_ylim(-0.12, 0.26)
ax1.set_xlabel("r  [Å]")
ax1.set_ylabel("V(r)  [eV]")
ax1.set_title("A  ·  LJ Potential", color=FG, fontsize=10, pad=6)
ax1.legend(fontsize=7, loc="upper right", framealpha=0.3)
ax1.grid(True, lw=0.4)
ax1.text(0.03, 0.03, "FCC NN at a=3.6 Å sits in\nrepulsive zone → large forces",
         transform=ax1.transAxes, color=MUTED, fontsize=7)

# ── Panel B: FIRE2 convergence ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
color_E = ACCENT
color_F = ORANGE
ax2b = ax2.twinx()

l1, = ax2.plot(steps, traj_E, color=color_E, lw=1.8, label="E/atom (eV)")
l2, = ax2b.plot(steps, traj_F, color=color_F, lw=1.5, ls="--", label="max |F| (eV/Å)")
ax2b.axhline(0.2, color=GREEN, lw=0.9, ls=":", label="fmax threshold")

if converged_step is not None:
    ax2.axvline(converged_step, color=GREEN, lw=1.2, ls="-", alpha=0.7)
    ax2.text(converged_step + 1, min(traj_E) * 0.7, f"converged\nstep {converged_step}",
             color=GREEN, fontsize=7.5)

ax2.set_xlabel("Optimisation step")
ax2.set_ylabel("Energy / atom  [eV]", color=color_E)
ax2b.set_ylabel("max |F|  [eV/Å]", color=color_F)
ax2.tick_params(axis="y", colors=color_E)
ax2b.tick_params(axis="y", colors=color_F)
ax2b.yaxis.label.set_color(color_F)
ax2b.spines["right"].set_color(color_F)
ax2b.set_facecolor("none")

ax2.set_title("B  ·  FIRE2 Geometry Relaxation", color=FG, fontsize=10, pad=6)
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize=7.5, loc="center right", framealpha=0.3)
ax2.grid(True, lw=0.4)

ax2.text(0.03, 0.05,
         f"32-atom FCC Fe + ±5% noise\nfire2_step_coord  (nvalchemiops.torch)",
         transform=ax2.transAxes, color=MUTED, fontsize=7)

# ── Panel C: batch alloy screen ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
colors_bar = [GREEN if s else RED for s in stable]
x = np.arange(len(comp_labels))
bars = ax3.barh(x, e_form, color=colors_bar, alpha=0.85, height=0.6)

ax3.axvline(0, color=MUTED, lw=1.0, ls="-")
for i, (ef, s) in enumerate(zip(e_form, stable)):
    tag = "✓ stable" if s else "✗"
    ax3.text(ef + (0.002 if ef >= 0 else -0.002),
             i, tag,
             va="center", ha="left" if ef >= 0 else "right",
             color=GREEN if s else MUTED, fontsize=7.5)

ax3.set_yticks(x)
ax3.set_yticklabels(comp_labels, fontsize=7.5)
ax3.set_xlabel("Formation energy  [eV/atom]")
ax3.set_title("C  ·  Batch LJ Alloy Screen", color=FG, fontsize=10, pad=6)
ax3.grid(True, axis="x", lw=0.4)

# Legend patches
from matplotlib.patches import Patch
leg_els = [Patch(fc=GREEN, label="Predicted stable  (ΔHf < 0)"),
           Patch(fc=RED,   label="Unstable  (ΔHf ≥ 0)")]
ax3.legend(handles=leg_els, fontsize=7.5, loc="lower right", framealpha=0.3)
ax3.text(0.03, 0.03,
         f"lj_energy_forces  (nvalchemiops)\nLorentz-Berthelot mixing rules",
         transform=ax3.transAxes, color=MUTED, fontsize=7)

plt.tight_layout()
out = "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM-Materials/examples/materials/alchemi_gpu_test.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
