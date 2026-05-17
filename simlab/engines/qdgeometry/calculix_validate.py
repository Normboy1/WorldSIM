"""CalculiX FEM validation of Warp/PyTorch FD thermal solver.

Writes a CalculiX .inp file for a 2D blade cross-section with parametric
cooling holes, runs `ccx`, parses the .frd output, and returns T_mean and
T_peak alongside the FD solver result for direct comparison.

Requires: calculix-ccx  (sudo apt-get install -y calculix-ccx)
Run:      ! sudo apt-get install -y calculix-ccx   in Claude Code prompt

Falls back gracefully if ccx is not found, returning a dict with
"error": "ccx not found" so the rest of the pipeline continues.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_CCX = None
for candidate in ["ccx", "ccx_2.21", "ccx_2.20", "ccx_2.19"]:
    if subprocess.run(["which", candidate], capture_output=True).returncode == 0:
        _CCX = candidate
        break


def _available() -> bool:
    return _CCX is not None


# ---------------------------------------------------------------------------
# CalculiX input file generator
# ---------------------------------------------------------------------------

def _write_ccx_input(
    fpath:     Path,
    hole_cx:   list[float],
    hole_cy:   list[float],
    hole_r:    list[float],
    T_hot:     float = 1300.0,
    T_cool:    float = 400.0,
    h_cool:    float = 5000.0,
    chord:     float = 0.05,   # m  (blade chord length — physical scale)
    n_elem_x:  int   = 40,
    n_elem_y:  int   = 20,
) -> str:
    """Write a CalculiX heat transfer .inp for steady 2D conduction.

    Uses a single-layer 3D DC3D8 hex mesh (CalculiX has no DC2D4 element).
    BCs: Dirichlet T=T_hot at top face, Dirichlet T=T_cool at hole-ring nodes.
    Neumann (zero flux) everywhere else.  Returns the job name (without extension).
    """
    job = fpath.stem

    lx = chord
    ly = chord * 0.5  # blade half-thickness
    lz = chord * 0.1  # thin z-slice (single element layer)
    dx = lx / n_elem_x
    dy = ly / n_elem_y

    nx_nodes = n_elem_x + 1
    ny_nodes = n_elem_y + 1
    nz_nodes = 2  # single layer: z=0 and z=lz

    def node_id(i, j, k):
        # k=0 or 1
        return k * nx_nodes * ny_nodes + j * nx_nodes + i + 1

    def _nset_lines(nodes: list[int]) -> list[str]:
        out = []
        for i in range(0, len(nodes), 16):
            out.append(", ".join(str(n) for n in nodes[i:i + 16]))
        return out

    lines = ["*HEADING", "CalculiX blade steady heat transfer (WorldSIM)"]

    # Nodes (two z-planes: k=0 at z=0, k=1 at z=lz)
    lines.append("*NODE, NSET=NALL")
    for k in range(nz_nodes):
        z = k * lz
        for j in range(ny_nodes):
            y = j * dy
            for i in range(nx_nodes):
                x = i * dx
                nid = node_id(i, j, k)
                lines.append(f"{nid}, {x:.6f}, {y:.6f}, {z:.6f}")

    # Elements DC3D8 (8-node hex heat transfer)
    lines.append("*ELEMENT, TYPE=DC3D8, ELSET=EALL")
    eid = 1
    for j in range(n_elem_y):
        for i in range(n_elem_x):
            # Bottom face (k=0), top face (k=1)
            n1 = node_id(i,   j,   0); n2 = node_id(i+1, j,   0)
            n3 = node_id(i+1, j+1, 0); n4 = node_id(i,   j+1, 0)
            n5 = node_id(i,   j,   1); n6 = node_id(i+1, j,   1)
            n7 = node_id(i+1, j+1, 1); n8 = node_id(i,   j+1, 1)
            lines.append(f"{eid}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}")
            eid += 1

    # Outer hot-gas wall: top row of nodes (j = n_elem_y), both z-planes
    outer_nodes = []
    for k in range(nz_nodes):
        for i in range(nx_nodes):
            outer_nodes.append(node_id(i, n_elem_y, k))

    # Cooling hole nodes: ring at radius ≈ r (Dirichlet T=T_cool approximation)
    all_cool_nodes: list[int] = []
    for cx_n, cy_n, r_n in zip(hole_cx, hole_cy, hole_r):
        cx_m = cx_n * lx
        cy_m = cy_n * ly
        r_m  = r_n  * lx
        tol  = max(dx, dy) * 1.5
        for j in range(ny_nodes):
            y = j * dy
            for i in range(nx_nodes):
                x = i * dx
                if abs(np.sqrt((x - cx_m)**2 + (y - cy_m)**2) - r_m) < tol:
                    for k in range(nz_nodes):
                        all_cool_nodes.append(node_id(i, j, k))

    # Material
    lines += [
        "*MATERIAL, NAME=NICKEL",
        "*CONDUCTIVITY",
        "14.0,",
        "*SOLID SECTION, ELSET=EALL, MATERIAL=NICKEL",
        f"{lz:.6f},",
    ]

    # Step: steady heat transfer
    lines += [
        "*STEP",
        "*HEAT TRANSFER, STEADY STATE",
        "1.0, 1.0",
    ]

    # Dirichlet BCs: temperature DOF = 11
    lines.append("*BOUNDARY")
    for nid in outer_nodes:
        lines.append(f"{nid}, 11, 11, {T_hot:.1f}")
    for nid in sorted(set(all_cool_nodes)):
        lines.append(f"{nid}, 11, 11, {T_cool:.1f}")

    lines += ["*NODE PRINT, NSET=NALL", "NT", "*END STEP"]

    with open(fpath, "w") as f:
        f.write("\n".join(lines) + "\n")

    return job


def _parse_dat(dat_path: Path) -> dict[str, Any]:
    """Parse CalculiX *NODE PRINT .dat output for nodal temperatures.

    The .dat format is:
        temperatures for set NALL and time ...
             node_id   temperature
             ...
    """
    temperatures = []
    in_temp_block = False

    try:
        with open(dat_path) as f:
            for line in f:
                stripped = line.strip()
                if "temperatures for set" in stripped.lower():
                    in_temp_block = True
                    continue
                if in_temp_block:
                    if stripped == "":
                        continue  # skip blank lines within/before data
                    if not stripped[0].isdigit() and not stripped[0] == "-":
                        in_temp_block = False
                        continue
                    parts = stripped.split()
                    if len(parts) == 2:
                        try:
                            temperatures.append(float(parts[1]))
                        except ValueError:
                            continue
    except FileNotFoundError:
        return {"error": f"dat file not found: {dat_path}"}

    if not temperatures:
        return {"error": "no temperatures found in dat output"}

    T = np.array(temperatures)
    return {
        "T_mean":  float(T.mean()),
        "T_peak":  float(T.max()),
        "T_min":   float(T.min()),
        "n_nodes": len(T),
        "T_all":   T,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculix_thermal(
    hole_cx:   list[float],
    hole_cy:   list[float],
    hole_r:    list[float],
    T_hot:     float = 1300.0,
    T_cool:    float = 400.0,
    h_cool:    float = 5000.0,
    chord:     float = 0.05,
    n_elem_x:  int   = 40,
    n_elem_y:  int   = 20,
    keep_files: bool = False,
) -> dict[str, Any]:
    """Run CalculiX FEM heat transfer and return T_mean, T_peak, T_all.

    If ccx is not installed, returns {"error": "ccx not found", "backend": "none"}.
    """
    if not _available():
        return {
            "error": "ccx not found — run: sudo apt-get install -y calculix-ccx",
            "backend": "none",
            "T_mean": None,
        }

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path = Path(tmpdir) / "blade.inp"
        job = _write_ccx_input(
            inp_path, hole_cx, hole_cy, hole_r,
            T_hot, T_cool, h_cool, chord, n_elem_x, n_elem_y,
        )

        result = subprocess.run(
            [_CCX, "-i", job],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return {
                "error": f"ccx returned {result.returncode}",
                "stderr": result.stderr[-500:],
                "backend": "calculix",
            }

        dat_path = Path(tmpdir) / f"{job}.dat"
        dat_result = _parse_dat(dat_path)

        if keep_files:
            out_dir = Path("calculix_output")
            out_dir.mkdir(exist_ok=True)
            import shutil
            shutil.copy(inp_path, out_dir / "blade.inp")
            frd_path = Path(tmpdir) / f"{job}.frd"
            if frd_path.exists():
                shutil.copy(frd_path, out_dir / "blade.frd")
            if dat_path.exists():
                shutil.copy(dat_path, out_dir / "blade.dat")

    wall_time = time.perf_counter() - t0
    dat_result.update({
        "backend":     "calculix_fem",
        "wall_time_s": wall_time,
        "n_holes":     len(hole_cx),
        "n_elem":      n_elem_x * n_elem_y,
        "ccx_binary":  _CCX,
    })
    dat_result.pop("T_all", None)
    return dat_result


def validate_fd_vs_fem(
    hole_cx:  list[float],
    hole_cy:  list[float],
    hole_r:   list[float],
    solver:   str = "warp",
    **kwargs,
) -> dict[str, Any]:
    """Compare FD solver (Warp or PyTorch) against CalculiX FEM.

    Returns comparison dict with |ΔT_mean|, |ΔT_peak|, and both results.
    """
    # FD result
    if solver == "warp":
        try:
            from simlab.engines.qdgeometry.warp_blade import warp_blade_thermal
            fd_result = warp_blade_thermal(hole_cx, hole_cy, hole_r)
        except Exception as exc:
            from simlab.engines.qdgeometry.torch_gpu_physics import gpu_blade_simulation
            fd_result = gpu_blade_simulation(
                n_holes=len(hole_cx),
                hole_positions=list(zip(hole_cx, hole_cy)),
                hole_radii=hole_r,
            )
            fd_result["backend"] = f"torch_fallback ({exc})"
    else:
        from simlab.engines.qdgeometry.torch_gpu_physics import gpu_blade_simulation
        fd_result = gpu_blade_simulation(
            n_holes=len(hole_cx),
            hole_positions=list(zip(hole_cx, hole_cy)),
            hole_radii=hole_r,
        )

    # FEM result
    fem_result = calculix_thermal(hole_cx, hole_cy, hole_r, **kwargs)

    comparison: dict[str, Any] = {
        "fd_backend":   fd_result.get("backend", solver),
        "fem_backend":  fem_result.get("backend", "none"),
        "fd_T_mean":    fd_result.get("T_mean") or fd_result.get("T_mean_blade"),
        "fem_T_mean":   fem_result.get("T_mean"),
        "fd_wall_time_s":  fd_result.get("wall_time_s"),
        "fem_wall_time_s": fem_result.get("wall_time_s"),
    }

    if comparison["fem_T_mean"] is not None and comparison["fd_T_mean"] is not None:
        comparison["delta_T_mean"] = abs(
            comparison["fd_T_mean"] - comparison["fem_T_mean"]
        )
        comparison["passes_50C_criterion"] = comparison["delta_T_mean"] < 50.0

    return comparison
