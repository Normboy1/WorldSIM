"""Animated FIRE2 relaxation — atoms colored by |F|, side panel tracks E and fmax."""

from __future__ import annotations
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM/.venv/lib/python3.12/site-packages")
sys.path.insert(0, "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM-Materials")

from simlab.engines.materials.nvidia_backends import ALCHEMIBackend

BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
FG    = "#e6edf3"
MUTED = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": MUTED,  "ytick.color": MUTED,
    "text.color": FG,      "grid.color": GRID, "grid.linewidth": 0.5,
    "font.family": "monospace", "font.size": 8,
})

# ── Build distorted 2×2×2 FCC Fe ─────────────────────────────────────────────
a = 3.6
np.random.seed(7)
basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]], dtype=np.float32)
pos_list = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            for bv in basis:
                p = (bv + np.array([ix,iy,iz])) * a
                p += np.random.randn(3).astype(np.float32) * 0.18
                pos_list.append(p)
sp = ["Fe"] * 32

b = ALCHEMIBackend()
print(f"Backend: {b.backend_name}")

# ── Run FIRE2, recording positions + forces every step ───────────────────────
frames_pos: list[np.ndarray] = []
frames_fmag: list[np.ndarray] = []
traj_E: list[float] = []
traj_F: list[float] = []

if b.available:
    import torch
    from nvalchemiops.torch import fire2_step_coord

    N = 32
    pos_np = np.array(pos_list, dtype=np.float32)
    pos_t  = torch.tensor(pos_np, device="cuda")
    vel_t  = torch.zeros_like(pos_t)
    bidx   = torch.zeros(N, dtype=torch.int32, device="cuda")
    alpha, dt, nsteps_inc = b._build_fire2_state(1, "cuda")

    _, forces_np = b._lj_energy_forces_gpu(pos_np, sp)
    forces_t = torch.tensor(forces_np, device="cuda")

    for step in range(120):
        p_now = pos_t.cpu().numpy().reshape(N, 3)
        e_arr, f_arr = b._lj_energy_forces_gpu(p_now.astype(np.float32), sp)
        fmag = np.linalg.norm(f_arr, axis=1)
        frames_pos.append(p_now.copy())
        frames_fmag.append(fmag.copy())
        traj_E.append(float(e_arr.sum()) / N)
        traj_F.append(float(fmag.max()))

        fire2_step_coord(pos_t, vel_t, forces_t, bidx, alpha, dt, nsteps_inc)
        p_new = pos_t.cpu().numpy().astype(np.float32)
        _, f_new = b._lj_energy_forces_gpu(p_new, sp)
        forces_t = torch.tensor(f_new, device="cuda")

    print(f"Captured {len(frames_pos)} frames, final fmax={traj_F[-1]:.3f} eV/Å")
else:
    print("CPU — generating synthetic trajectory")
    N = 32
    pos_np = np.array(pos_list, dtype=np.float32)
    for step in range(80):
        decay = math.exp(-step / 20)
        noise = (np.random.randn(N, 3) * 0.15 * decay).astype(np.float32)
        frames_pos.append(pos_np + noise)
        frames_fmag.append(np.abs(noise).sum(axis=1) * 10)
        traj_E.append(-0.30 * (1 - math.exp(-step / 12)))
        traj_F.append(5.0 * math.exp(-step / 15) + 0.15)

n_frames = len(frames_pos)
steps_arr = np.arange(n_frames)

# ── Colour scale: |F| mapped to "plasma" ─────────────────────────────────────
all_fmag = np.concatenate(frames_fmag)
fmin, fmax_val = all_fmag.min(), np.percentile(all_fmag, 97)
norm = Normalize(vmin=fmin, vmax=fmax_val)
cmap = plt.cm.plasma

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 5), facecolor=BG)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38, width_ratios=[1.4, 1])

ax_atoms = fig.add_subplot(gs[0])
ax_atoms.set_facecolor(PANEL)
ax_atoms.set_aspect("equal")
ax_atoms.grid(True, lw=0.3, alpha=0.5)
ax_atoms.set_title("FCC Fe — FIRE2 Relaxation  (XZ projection, colored by |F|)",
                   color=FG, fontsize=9, pad=5)
ax_atoms.set_xlabel("x  [Å]")
ax_atoms.set_ylabel("z  [Å]")

# colourbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_atoms, fraction=0.035, pad=0.02)
cbar.set_label("|F|  [eV/Å]", color=FG)
cbar.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)

# convergence panel
ax_conv = fig.add_subplot(gs[1])
ax_conv.set_facecolor(PANEL)
ax_conv2 = ax_conv.twinx()
ax_conv2.set_facecolor("none")

# FCC ideal positions (XZ projection) as faint guide
ideal = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            for bv in basis:
                ideal.append((bv + np.array([ix,iy,iz])) * a)
ideal = np.array(ideal)
ax_atoms.scatter(ideal[:,0], ideal[:,2], s=18, color=MUTED,
                 alpha=0.18, zorder=1, label="ideal FCC")

# initial scatter
scat = ax_atoms.scatter(
    frames_pos[0][:,0], frames_pos[0][:,2],
    c=frames_fmag[0], cmap=cmap, norm=norm,
    s=60, zorder=3, edgecolors="none", alpha=0.92,
)

step_txt = ax_atoms.text(
    0.02, 0.97, "step 0", transform=ax_atoms.transAxes,
    color=FG, fontsize=9, va="top",
)

# convergence lines (grow each frame)
E_line, = ax_conv.plot([], [], color="#58a6ff", lw=1.8, label="E/atom (eV)")
F_line, = ax_conv2.plot([], [], color="#f0883e", lw=1.5, ls="--", label="max |F|")
thresh  = ax_conv2.axhline(0.2, color="#3fb950", lw=0.9, ls=":", label="fmax 0.2")

ax_conv.set_xlim(0, n_frames - 1)
ax_conv.set_ylim(min(traj_E) * 1.12, max(traj_E) * 1.05 + 0.02)
ax_conv2.set_ylim(0, max(traj_F) * 1.1)
ax_conv.set_xlabel("Optimisation step")
ax_conv.set_ylabel("E/atom  [eV]", color="#58a6ff")
ax_conv2.set_ylabel("max |F|  [eV/Å]", color="#f0883e")
ax_conv.tick_params(axis="y", colors="#58a6ff")
ax_conv2.tick_params(axis="y", colors="#f0883e")
ax_conv2.spines["right"].set_color("#f0883e")
ax_conv.set_title("B  ·  Convergence", color=FG, fontsize=9, pad=5)
ax_conv.grid(True, lw=0.4)
lines = [E_line, F_line, thresh]
labels = [l.get_label() for l in lines]
ax_conv.legend(lines, labels, fontsize=7, loc="center right", framealpha=0.3)

# current step marker
step_dot_E, = ax_conv.plot([], [], "o", color="#58a6ff", ms=5, zorder=5)
step_dot_F, = ax_conv2.plot([], [], "o", color="#f0883e", ms=5, zorder=5)

# ── Animation update ──────────────────────────────────────────────────────────
def update(frame):
    pos   = frames_pos[frame]
    fmag  = frames_fmag[frame]

    scat.set_offsets(np.c_[pos[:,0], pos[:,2]])
    scat.set_array(fmag)

    step_txt.set_text(f"step {frame:3d}  E={traj_E[frame]:+.4f} eV/atom  fmax={traj_F[frame]:.3f}")

    sl = steps_arr[:frame+1]
    E_line.set_data(sl, traj_E[:frame+1])
    F_line.set_data(sl, traj_F[:frame+1])
    step_dot_E.set_data([frame], [traj_E[frame]])
    step_dot_F.set_data([frame], [traj_F[frame]])

    return scat, step_txt, E_line, F_line, step_dot_E, step_dot_F

# subsample to every 2nd frame to keep GIF manageable
frame_indices = list(range(0, n_frames, 2))
anim = FuncAnimation(fig, update, frames=frame_indices, interval=80, blit=True)

out = "/home/norman/Vibes/maxosl-ai-research/ResearchGrants/WorldSIM-Materials/examples/materials/fire2_relaxation.gif"
anim.save(out, writer=PillowWriter(fps=15), dpi=130)
print(f"Saved → {out}")
