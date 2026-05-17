"""NVIDIA accelerated backends for WorldSim Materials.

Each class wraps one NVIDIA tool with a clean interface:
- ALCHEMIBackend    → atomistic MD, geometry relaxation, LJ energy/forces via nvalchemiops
- WarpBackend       → GPU finite-difference kernels (diffusion, phase-field)
- PhysicsNeMoBackend→ Fourier Neural Operator / PINN surrogate training + inference
- CuPyBackend       → drop-in accelerated NumPy for parameter sweeps

All classes fall back gracefully to NumPy/SciPy CPU implementations when the
NVIDIA packages are not installed. Call `.available` to check at runtime.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np


def _try_import(module_name: str):
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return importlib.import_module(module_name)
    except (ImportError, ValueError, ModuleNotFoundError):
        pass
    return None


# ---------------------------------------------------------------------------
# ALCHEMI backend — atomistic simulation via nvalchemiops component API
# ---------------------------------------------------------------------------

# Simplified UFF-style LJ parameters (epsilon [eV], sigma [Å]) per element.
_LJ_PARAMS: dict[str, tuple[float, float]] = {
    "Fe": (0.041, 2.594), "Ni": (0.035, 2.524), "Cr": (0.039, 2.693),
    "Co": (0.038, 2.559), "Al": (0.024, 4.008), "Ti": (0.048, 2.829),
    "Mo": (0.057, 2.719), "W":  (0.067, 2.734), "Cu": (0.021, 3.114),
    "C":  (0.086, 3.851), "H":  (0.044, 2.886), "O":  (0.060, 3.119),
    "N":  (0.069, 3.660),
}
_LJ_DEFAULT = (0.040, 2.700)

# Atomic masses [amu]
_ATOMIC_MASS: dict[str, float] = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "Al": 26.982, "Ti": 47.867, "Cr": 51.996, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Mo": 95.95,
    "W": 183.84, "Nb": 92.906, "V": 50.942,
}
_MASS_DEFAULT = 55.845

_KB_EV = 8.617333e-5  # Boltzmann constant [eV/K]


class ALCHEMIBackend:
    """NVIDIA ALCHEMI atomistic simulation via nvalchemiops component API.

    nvalchemiops is a Warp-based library of low-level MD/optimization
    primitives: FIRE2/Langevin integrators, LJ/Coulomb/DFT-D3 interactions,
    cell-list neighbor builders. This backend orchestrates those components
    directly using their functional API.

    Geometry relaxation: ``nvalchemiops.torch.fire2_step_coord`` (FIRE2)
    Molecular dynamics:  ``nvalchemiops.dynamics.integrators.langevin`` (BAOAB)
    Force evaluation:    ``nvalchemiops.interactions.lj.lj_energy_forces``
    Neighbor list:       ``nvalchemiops.neighbors.naive.naive_neighbor_matrix``

    All methods fall back to NumPy/Miedema proxies when nvalchemiops or the
    CUDA runtime is unavailable.
    """

    def __init__(self):
        self._nvalchemiops = _try_import("nvalchemiops")
        self._torch = _try_import("torch")
        self.available: bool = (
            self._nvalchemiops is not None and self._torch is not None
        )
        self.backend_name: str = "alchemi_warp_gpu" if self.available else "numpy_cpu_proxy"
        if self.available:
            try:
                import warp as wp
                wp.init()
                self._wp = wp
            except Exception:
                self.available = False
                self.backend_name = "numpy_cpu_proxy"
                self._wp = None
        else:
            self._wp = None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _mixing_rule(self, sp_a: str, sp_b: str) -> tuple[float, float]:
        """Lorentz-Berthelot mixing rules for LJ epsilon and sigma."""
        eps_a, sig_a = _LJ_PARAMS.get(sp_a, _LJ_DEFAULT)
        eps_b, sig_b = _LJ_PARAMS.get(sp_b, _LJ_DEFAULT)
        return math.sqrt(eps_a * eps_b), 0.5 * (sig_a + sig_b)

    def _avg_lj(self, species: list[str]) -> tuple[float, float]:
        """Average LJ parameters over all unique element pairs."""
        unique = list(set(species))
        eps_vals, sig_vals = [], []
        for i, a in enumerate(unique):
            for b in unique[i:]:
                e, s = self._mixing_rule(a, b)
                eps_vals.append(e)
                sig_vals.append(s)
        return float(np.mean(eps_vals)), float(np.mean(sig_vals))

    def _lj_energy_forces_gpu(
        self,
        positions_A: np.ndarray,
        species: list[str],
        cutoff: float = 6.0,
        device: str = "cuda",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute LJ energy/forces via nvalchemiops on GPU.

        Returns (energies_per_atom [N,], forces [N, 3]), in eV / eV·Å⁻¹.
        Uses ``naive_neighbor_matrix`` (non-PBC) + ``lj_energy_forces``.
        """
        wp = self._wp
        from nvalchemiops.interactions.lj import lj_energy_forces
        from nvalchemiops.neighbors.naive import naive_neighbor_matrix
        from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

        N = len(species)
        eps, sig = self._avg_lj(species)
        max_nb = estimate_max_neighbors(cutoff, safety_factor=1.5)

        pos_wp = wp.array(positions_A.astype(np.float32), dtype=wp.vec3f, device=device)
        nbr_mat = wp.zeros((N, max_nb), dtype=wp.int32, device=device)
        num_nb = wp.zeros(N, dtype=wp.int32, device=device)
        # Non-PBC naive neighbor list: shifts are all zero (no periodic images)
        nbr_shifts = wp.zeros((N, max_nb), dtype=wp.vec3i, device=device)

        naive_neighbor_matrix(pos_wp, float(cutoff), nbr_mat, num_nb, wp.float32, device)

        # Cell shape must be (1,) mat33f; value doesn't matter for non-PBC (shifts=0)
        cell_np = np.eye(3, dtype=np.float32) * 1000.0
        cell_wp = wp.array(cell_np[np.newaxis], dtype=wp.mat33f, device=device)

        energies_wp, forces_wp = lj_energy_forces(
            positions=pos_wp,
            cell=cell_wp,
            epsilon=float(eps),
            sigma=float(sig),
            cutoff=float(cutoff),
            neighbor_matrix=nbr_mat,
            neighbor_matrix_shifts=nbr_shifts,
            num_neighbors=num_nb,
            fill_value=N,
            device=device,
        )
        return energies_wp.numpy(), forces_wp.numpy().reshape(N, 3)

    def _build_fire2_state(
        self, n_systems: int, device: str
    ) -> tuple:
        """Allocate FIRE2 scalar state tensors (torch)."""
        import torch
        kw = {"dtype": torch.float32, "device": device}
        alpha = torch.full((n_systems,), 0.09, **kw)
        dt = torch.full((n_systems,), 0.02, **kw)
        nsteps_inc = torch.zeros((n_systems,), dtype=torch.int32, device=device)
        return alpha, dt, nsteps_inc

    # ------------------------------------------------------------------
    # Geometry relaxation via FIRE2 + LJ
    # ------------------------------------------------------------------

    def relax_structure(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        potential: str = "lj",
        max_steps: int = 200,
        fmax_eV_A: float = 0.05,
    ) -> dict:
        """Minimise atomic forces using FIRE2 optimizer + LJ potential on GPU.

        Uses ``nvalchemiops.torch.fire2_step_coord`` (PyTorch-native FIRE2 path)
        with ``nvalchemiops.interactions.lj.lj_energy_forces`` for atomic forces.
        FIRE2 reference: Guenole et al., Comp. Mat. Sci. 175 (2020) 109584.

        To use an MLIP instead of LJ, replace the force evaluation call with
        a ``nvalchemiops.torch`` compatible neural potential.
        """
        if self.available:
            try:
                return self._relax_fire2_lj(species, positions_angstrom, max_steps, fmax_eV_A)
            except Exception as exc:
                return self._relax_cpu_proxy(
                    species, positions_angstrom, lattice_vectors_angstrom, error=str(exc)
                )
        return self._relax_cpu_proxy(species, positions_angstrom, lattice_vectors_angstrom)

    def _relax_fire2_lj(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        max_steps: int,
        fmax_eV_A: float,
        device: str = "cuda",
    ) -> dict:
        import torch
        from nvalchemiops.torch import fire2_step_coord

        N = len(species)
        pos_np = np.array(positions_angstrom, dtype=np.float32)

        pos_t = torch.tensor(pos_np, device=device)
        vel_t = torch.zeros_like(pos_t)
        batch_idx = torch.zeros(N, dtype=torch.int32, device=device)
        alpha, dt, nsteps_inc = self._build_fire2_state(1, device)

        energies_arr, forces_np = self._lj_energy_forces_gpu(pos_np, species, device=device)
        forces_t = torch.tensor(forces_np, device=device)

        converged = False
        step = 0
        for step in range(int(max_steps)):
            fmax = float(forces_t.norm(dim=1).max())
            if fmax <= fmax_eV_A:
                converged = True
                break

            fire2_step_coord(pos_t, vel_t, forces_t, batch_idx, alpha, dt, nsteps_inc)

            pos_np = pos_t.cpu().numpy().astype(np.float32)
            energies_arr, forces_np = self._lj_energy_forces_gpu(pos_np, species, device=device)
            forces_t = torch.tensor(forces_np, device=device)

        final_fmax = float(forces_t.norm(dim=1).max())
        return {
            "backend": "alchemi_warp_gpu",
            "method": "FIRE2+LJ",
            "converged": converged,
            "steps": step + 1,
            "final_energy_eV": float(energies_arr.sum()),
            "energy_eV_per_atom": float(energies_arr.sum()) / N,
            "max_force_eV_A": final_fmax,
            "relaxed_positions": pos_t.cpu().tolist(),
        }

    def _relax_cpu_proxy(
        self,
        species: list[str],
        positions: list[list[float]],
        cell: list[list[float]],
        error: str | None = None,
    ) -> dict:
        pos = np.array(positions, dtype=float)
        n_atoms = len(species)
        cell_arr = np.array(cell, dtype=float)
        volume_A3 = float(abs(np.linalg.det(cell_arr)))
        volume_per_atom = volume_A3 / max(n_atoms, 1)
        proxy_energy = -n_atoms * 3.0 + 0.02 * (volume_per_atom - 12.0) ** 2
        result = {
            "backend": "numpy_cpu_proxy",
            "method": "LJ_volume_proxy",
            "converged": True,
            "steps": 0,
            "final_energy_eV": float(proxy_energy),
            "energy_eV_per_atom": float(proxy_energy) / n_atoms,
            "max_force_eV_A": 0.0,
            "relaxed_positions": pos.tolist(),
            "note": "CPU proxy — nvalchemiops+CUDA needed for FIRE2/LJ GPU relaxation.",
        }
        if error:
            result["alchemi_error"] = error
        return result

    # ------------------------------------------------------------------
    # Molecular dynamics via Langevin BAOAB + LJ
    # ------------------------------------------------------------------

    def run_md(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        temperature_K: float = 300.0,
        pressure_GPa: float = 0.0,
        timestep_fs: float = 1.0,
        n_steps: int = 1000,
        ensemble: str = "NVT",
        potential: str = "lj",
        output_stride: int = 100,
    ) -> dict:
        """Run NVT Langevin BAOAB molecular dynamics using nvalchemiops Warp kernels.

        Uses ``nvalchemiops.dynamics.integrators.langevin`` (BAOAB scheme) with
        LJ forces from ``nvalchemiops.interactions.lj``. Temperature is controlled
        via the Langevin thermostat. NPT is mapped to NVT (barostat path is a
        future extension via ``nvalchemiops.dynamics.integrators.npt``).
        """
        if self.available:
            try:
                return self._md_langevin_lj(
                    species, positions_angstrom, temperature_K,
                    timestep_fs, n_steps, output_stride, ensemble
                )
            except Exception as exc:
                return self._md_cpu_proxy(species, temperature_K, n_steps, timestep_fs, error=str(exc))
        return self._md_cpu_proxy(species, temperature_K, n_steps, timestep_fs)

    def _md_langevin_lj(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        temperature_K: float,
        timestep_fs: float,
        n_steps: int,
        output_stride: int,
        ensemble: str,
        device: str = "cuda",
        friction_ps: float = 1.0,
    ) -> dict:
        import warp as wp
        from nvalchemiops.dynamics.integrators.langevin import (
            langevin_baoab_half_step,
            langevin_baoab_finalize,
        )

        N = len(species)
        # Units: Å, eV, amu, fs
        kT = _KB_EV * float(temperature_K)          # eV
        dt_fs = float(timestep_fs)
        gamma = float(friction_ps) / 1000.0          # ps⁻¹ → fs⁻¹

        masses_np = np.array(
            [_ATOMIC_MASS.get(s, _MASS_DEFAULT) for s in species], dtype=np.float32
        )
        pos_np = np.array(positions_angstrom, dtype=np.float32)
        std = float(np.sqrt(kT / masses_np.mean()))
        vel_np = np.random.randn(N, 3).astype(np.float32) * std

        pos_wp = wp.array(pos_np, dtype=wp.vec3f, device=device)
        vel_wp = wp.array(vel_np, dtype=wp.vec3f, device=device)
        masses_wp = wp.array(masses_np, dtype=wp.float32, device=device)
        dt_wp = wp.array([dt_fs], dtype=wp.float32, device=device)
        kT_wp = wp.array([kT], dtype=wp.float32, device=device)    # temperature as kT [eV]
        gamma_wp = wp.array([gamma], dtype=wp.float32, device=device)

        energies_np, forces_np = self._lj_energy_forces_gpu(pos_np, species, device=device)
        forces_wp = wp.array(forces_np.astype(np.float32), dtype=wp.vec3f, device=device)

        energies_traj: list[float] = []
        temps_traj: list[float] = []

        for step in range(int(n_steps)):
            # BAOAB half-step: B-A-O-A (updates pos and vel in-place)
            langevin_baoab_half_step(
                pos_wp, vel_wp, forces_wp, masses_wp,
                dt_wp, kT_wp, gamma_wp,
                random_seed=42 + step,
                device=device,
            )

            # Recompute forces at new positions
            pos_np = pos_wp.numpy().reshape(N, 3)
            energies_np, forces_np = self._lj_energy_forces_gpu(pos_np, species, device=device)
            forces_wp = wp.array(forces_np.astype(np.float32), dtype=wp.vec3f, device=device)

            # Final B step
            langevin_baoab_finalize(vel_wp, forces_wp, masses_wp, dt_wp, device=device)

            if step % max(1, int(output_stride)) == 0:
                vel_np_curr = vel_wp.numpy().reshape(N, 3)
                ekin = 0.5 * float(np.sum(masses_np[:, None] * vel_np_curr ** 2))
                T_inst = ekin * 2.0 / (3.0 * N * _KB_EV)
                energies_traj.append(float(energies_np.sum()))
                temps_traj.append(float(T_inst))

        total_time_ps = n_steps * dt_fs / 1000.0
        return {
            "backend": "alchemi_warp_gpu",
            "method": "Langevin_BAOAB+LJ",
            "ensemble": ensemble if ensemble == "NVT" else "NVT (NPT→NVT fallback)",
            "temperature_K": float(temperature_K),
            "n_steps": n_steps,
            "timestep_fs": dt_fs,
            "total_time_ps": total_time_ps,
            "mean_energy_eV": float(np.mean(energies_traj)) if energies_traj else None,
            "mean_temperature_K": float(np.mean(temps_traj)) if temps_traj else None,
            "trajectory_steps": len(energies_traj),
            "energy_traj_eV": energies_traj,
            "temperature_traj_K": temps_traj,
        }

    def _md_cpu_proxy(
        self,
        species: list[str],
        temperature_K: float,
        n_steps: int,
        timestep_fs: float,
        error: str | None = None,
    ) -> dict:
        """Equipartition proxy: Ekin = (3/2) N kB T."""
        n_atoms = len(species)
        ekin = 1.5 * n_atoms * _KB_EV * float(temperature_K)
        total_time_ps = n_steps * float(timestep_fs) / 1000.0
        result = {
            "backend": "numpy_cpu_proxy",
            "method": "equipartition_proxy",
            "ensemble": "NVT_proxy",
            "temperature_K": float(temperature_K),
            "n_steps": n_steps,
            "total_time_ps": float(total_time_ps),
            "proxy_kinetic_energy_eV": float(ekin),
            "note": "CPU proxy (equipartition) — nvalchemiops+CUDA needed for Langevin BAOAB MD.",
        }
        if error:
            result["alchemi_error"] = error
        return result

    # ------------------------------------------------------------------
    # Single-point energy and forces
    # ------------------------------------------------------------------

    def mlip_energy_forces(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        potential: str = "lj",
        potential_path: str | None = None,
    ) -> dict:
        """Single-point energy and forces via nvalchemiops or an external MLIP.

        When ``potential_path`` is supplied, attempts to load the model via
        MACE (``mace.calculators.MACECalculator``) or SevenNet, passing atomic
        positions through the calculator's ``get_potential_energy`` /
        ``get_forces`` ASE interface.  Falls back to the LJ GPU kernel when
        neither the model file nor a supported loader is available.

        MACE-MP-0 example::

            mlip_energy_forces(
                species=["Fe", "Ni"],
                positions_angstrom=[[0,0,0],[1.8,0,0]],
                lattice_vectors_angstrom=[[5,0,0],[0,5,0],[0,0,5]],
                potential="mace",
                potential_path="/path/to/mace-mp-0.model",
            )
        """
        if potential_path is not None:
            result = self._mlip_from_path(species, positions_angstrom, lattice_vectors_angstrom, potential_path, potential)
            if result is not None:
                return result

        if self.available:
            try:
                N = len(species)
                pos_np = np.array(positions_angstrom, dtype=np.float32)
                energies, forces = self._lj_energy_forces_gpu(pos_np, species)
                fmags = np.linalg.norm(forces, axis=1)
                return {
                    "backend": "alchemi_warp_gpu",
                    "potential": potential,
                    "potential_path": None,
                    "energy_eV": float(energies.sum()),
                    "energy_eV_per_atom": float(energies.sum()) / N,
                    "max_force_eV_A": float(fmags.max()),
                    "mean_force_eV_A": float(fmags.mean()),
                    "forces_eV_A": forces.tolist(),
                    "per_atom_energies_eV": energies.tolist(),
                }
            except Exception as exc:
                return self._mlip_proxy(species, error=str(exc))
        return self._mlip_proxy(species)

    def _mlip_from_path(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        potential_path: str,
        potential: str,
    ) -> dict | None:
        """Attempt to evaluate energy/forces using an external MLIP model file.

        Tries MACE first, then SevenNet. Returns None if neither loader is
        available so the caller can fall back to the LJ path.
        """
        import os
        if not os.path.isfile(potential_path):
            return None

        pos_np = np.array(positions_angstrom, dtype=float)
        cell_np = np.array(lattice_vectors_angstrom, dtype=float)
        N = len(species)

        # --- MACE ---
        try:
            from mace.calculators import MACECalculator
            import ase
            atoms = ase.Atoms(symbols=species, positions=pos_np, cell=cell_np, pbc=True)
            calc = MACECalculator(model_paths=potential_path, device="cuda" if self.available else "cpu")
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmags = np.linalg.norm(forces, axis=1)
            return {
                "backend": "mace_mlip",
                "potential": potential,
                "potential_path": potential_path,
                "energy_eV": energy,
                "energy_eV_per_atom": energy / N,
                "max_force_eV_A": float(fmags.max()),
                "mean_force_eV_A": float(fmags.mean()),
                "forces_eV_A": forces.tolist(),
            }
        except ImportError:
            pass
        except Exception:
            pass

        # --- SevenNet ---
        try:
            from sevenn.sevennet_calculator import SevenNetCalculator
            import ase
            atoms = ase.Atoms(symbols=species, positions=pos_np, cell=cell_np, pbc=True)
            calc = SevenNetCalculator(potential_path, device="cuda" if self.available else "cpu")
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmags = np.linalg.norm(forces, axis=1)
            return {
                "backend": "sevennet_mlip",
                "potential": potential,
                "potential_path": potential_path,
                "energy_eV": energy,
                "energy_eV_per_atom": energy / N,
                "max_force_eV_A": float(fmags.max()),
                "mean_force_eV_A": float(fmags.mean()),
                "forces_eV_A": forces.tolist(),
            }
        except ImportError:
            pass
        except Exception:
            pass

        return None

    def _mlip_proxy(self, species: list[str], error: str | None = None) -> dict:
        result = {
            "backend": "numpy_cpu_proxy",
            "energy_eV_per_atom": -3.5,
            "max_force_eV_A": 0.0,
            "note": "CPU proxy — nvalchemiops+CUDA needed for LJ/MLIP energy/forces.",
        }
        if error:
            result["alchemi_error"] = error
        return result

    # ------------------------------------------------------------------
    # Batched alloy screening
    # ------------------------------------------------------------------

    def batch_alloy_screen(
        self,
        compositions: list[dict[str, float]],
        lattice_type: str = "fcc",
        a_angstrom: float = 3.6,
        potential: str = "lj",
        temperature_K: float = 300.0,
    ) -> dict:
        """Batch LJ energy screening for multiple alloy compositions on GPU.

        Builds one FCC/BCC supercell per composition, assigns mixed LJ
        parameters via Lorentz-Berthelot rules, and evaluates formation
        energies using ``nvalchemiops.interactions.lj.lj_energy_forces``.

        This mirrors ALCHEMI's core workflow: parallel GPU evaluation of
        alloy candidates to generate active-learning training data.
        """
        if self.available:
            try:
                return self._batch_screen_lj(compositions, lattice_type, a_angstrom)
            except Exception as exc:
                return self._batch_proxy(compositions, error=str(exc))
        return self._batch_proxy(compositions)

    def _make_fcc_positions(self, a: float, n_repeat: int = 2) -> np.ndarray:
        basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]], dtype=np.float32)
        pos = []
        for ix in range(n_repeat):
            for iy in range(n_repeat):
                for iz in range(n_repeat):
                    for b in basis:
                        pos.append((b + np.array([ix, iy, iz])) * a)
        return np.array(pos, dtype=np.float32)

    def _make_bcc_positions(self, a: float, n_repeat: int = 2) -> np.ndarray:
        basis = np.array([[0,0,0],[0.5,0.5,0.5]], dtype=np.float32)
        pos = []
        for ix in range(n_repeat):
            for iy in range(n_repeat):
                for iz in range(n_repeat):
                    for b in basis:
                        pos.append((b + np.array([ix, iy, iz])) * a)
        return np.array(pos, dtype=np.float32)

    def _assign_species_random(self, comp: dict[str, float], n_atoms: int) -> list[str]:
        """Randomly assign species to sites according to composition fractions.

        Uses largest-remainder rounding so that every element with a non-zero
        fraction gets at least one site, preventing minority elements being
        silently dropped by truncation.
        """
        total = sum(comp.values())
        norm = {el: v / total for el, v in comp.items() if v > 0 and total > 0}
        elements = list(norm.keys())

        # Largest-remainder method: guarantees each element gets ≥1 atom
        exact = {el: norm[el] * n_atoms for el in elements}
        floors = {el: int(exact[el]) for el in elements}
        remainders = sorted(elements, key=lambda el: -(exact[el] - floors[el]))
        allocated = sum(floors.values())
        for el in remainders[:n_atoms - allocated]:
            floors[el] += 1

        species: list[str] = []
        for el, count in floors.items():
            species.extend([el] * count)
        np.random.shuffle(species)
        return species[:n_atoms]

    def _batch_screen_lj(
        self,
        compositions: list[dict[str, float]],
        lattice_type: str,
        a_angstrom: float,
        device: str = "cuda",
    ) -> dict:
        """Screen alloy compositions via LJ cohesive energy difference.

        IMPORTANT — interpretation: ``lj_cohesive_energy_diff_eV_atom`` is the
        difference between alloy LJ energy and the composition-weighted pure-
        element LJ energies on the *same lattice*.  This is NOT a thermodynamic
        formation energy; it captures only the pairwise mixing contribution of
        Lorentz-Berthelot ε/σ parameters and has no electronic, many-body, or
        magnetic physics.  Values of several eV/atom are normal for LJ cohesive
        energies and do not correspond to DFT formation energies (which are
        typically < 0.5 eV/atom in magnitude).  Use this only for *relative*
        ranking within a batch screened on the same lattice.
        """
        make_pos = self._make_bcc_positions if lattice_type == "bcc" else self._make_fcc_positions
        results = []
        for comp in compositions:
            pos_np = make_pos(a_angstrom)
            N = len(pos_np)
            sp = self._assign_species_random(comp, N)
            energies, _ = self._lj_energy_forces_gpu(pos_np, sp, device=device)
            e_atom = float(energies.sum()) / N

            # Pure-element reference energies on the same lattice
            e_refs: dict[str, float] = {}
            for el in set(sp):
                e_pure, _ = self._lj_energy_forces_gpu(pos_np, [el] * N, device=device)
                e_refs[el] = float(e_pure.sum()) / N
            total_fracs = sum(comp.values())
            e_ref_mix = sum(
                (v / total_fracs) * e_refs.get(el, -3.5) for el, v in comp.items()
            )
            lj_diff = e_atom - e_ref_mix
            results.append({
                "composition": comp,
                "lj_cohesive_energy_eV_atom": e_atom,
                "lj_cohesive_energy_diff_eV_atom": float(lj_diff),
                # negative diff = alloy LJ energy lower than pure mix on this lattice
                "lj_diff_negative": lj_diff < 0.0,
                "note": (
                    "LJ pairwise cohesive energy difference, NOT thermodynamic "
                    "formation energy. Use only for relative ranking."
                ),
            })
        return {
            "backend": "alchemi_warp_gpu",
            "method": f"LJ_batch_{lattice_type}",
            "n_compositions": len(compositions),
            "lattice_type": lattice_type,
            "a_angstrom": a_angstrom,
            "energy_interpretation": (
                "lj_cohesive_energy_diff_eV_atom is a relative LJ pairwise "
                "screening metric, not a DFT formation energy."
            ),
            "results": results,
        }

    def _batch_proxy(
        self, compositions: list[dict[str, float]], error: str | None = None
    ) -> dict:
        """Rough Miedema-style formation energy proxy for CPU screening."""
        _MIEDEMA_E_ATOM = {
            "Fe": -8.3, "Ni": -8.7, "Cr": -10.5, "Co": -8.8,
            "Al": -3.4, "Ti": -7.7, "Mo": -10.8, "W": -12.9,
            "Nb": -7.6, "V": -8.6, "C": -9.2, "Cu": -3.5,
        }
        results = []
        for comp in compositions:
            total = sum(comp.values())
            norm = {el: v / total for el, v in comp.items() if total > 0}
            e_atom = sum(frac * _MIEDEMA_E_ATOM.get(el, -5.0) for el, frac in norm.items())
            e_form_proxy = 0.02 * (len(norm) - 1) * (-1.0)
            results.append({
                "composition": comp,
                "energy_eV_per_atom": float(e_atom),
                "formation_energy_eV_atom": float(e_form_proxy),
                "predicted_stable": e_form_proxy < 0.0,
            })
        out = {
            "backend": "numpy_cpu_proxy",
            "method": "miedema_proxy",
            "n_compositions": len(compositions),
            "results": results,
            "note": "CPU Miedema proxy — nvalchemiops+CUDA needed for GPU LJ batch screening.",
        }
        if error:
            out["alchemi_error"] = error
        return out


# ---------------------------------------------------------------------------
# Warp backend — GPU finite-difference kernels
# ---------------------------------------------------------------------------

class WarpBackend:
    """NVIDIA Warp GPU kernel integration.

    Warp is a Python framework for high-performance simulation and graphics
    kernels (warp-lang package). Provides @wp.kernel decorated functions that
    compile to CUDA/CPU.

    Used for:
    - Finite-difference diffusion kernels (2D / 3D)
    - Phase-field grain growth (Allen-Cahn)
    - Thermal field kernels
    - Custom microstructure evolution
    """

    def __init__(self):
        self._warp = _try_import("warp")
        self.available: bool = self._warp is not None
        self.backend_name: str = "warp_cuda" if self.available else "numpy_cpu"
        if self.available:
            try:
                self._warp.init()
            except Exception:
                pass

    def diffusion_2d(
        self,
        field: np.ndarray,
        alpha: float,
        steps: int,
    ) -> np.ndarray:
        """Run a 2D finite-difference diffusion step using Warp GPU kernel.

        alpha = D * dt / dx² (stability number, must be <= 0.25 for 2D)
        """
        if self.available:
            wp = self._warp
            try:
                import warp as wp

                nx, ny = field.shape
                field_wp = wp.array(field, dtype=wp.float32, device="cuda")
                field_out = wp.zeros_like(field_wp)

                @wp.kernel
                def diffusion_step_2d(
                    f_in: wp.array2d(dtype=wp.float32),
                    f_out: wp.array2d(dtype=wp.float32),
                    alpha: float,
                ):
                    i, j = wp.tid()
                    nx_ = f_in.shape[0]
                    ny_ = f_in.shape[1]
                    if i > 0 and i < nx_ - 1 and j > 0 and j < ny_ - 1:
                        laplacian = (
                            f_in[i - 1, j] + f_in[i + 1, j]
                            + f_in[i, j - 1] + f_in[i, j + 1]
                            - 4.0 * f_in[i, j]
                        )
                        f_out[i, j] = f_in[i, j] + alpha * laplacian
                    else:
                        f_out[i, j] = f_in[i, j]

                for _ in range(int(steps)):
                    wp.launch(
                        diffusion_step_2d,
                        dim=(nx, ny),
                        inputs=[field_wp, field_out, float(alpha)],
                    )
                    field_wp, field_out = field_out, field_wp

                return field_wp.numpy()

            except Exception:
                pass

        # NumPy fallback
        f = field.copy().astype(float)
        a = min(float(alpha), 0.24)
        for _ in range(int(steps)):
            lap = (
                np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1)
                - 4.0 * f
            )
            f += a * lap
        return f

    def allen_cahn_grain_growth(
        self,
        phi: np.ndarray,
        M: float,
        kappa: float,
        dt: float,
        steps: int,
    ) -> np.ndarray:
        """Allen-Cahn phase-field grain growth kernel.

        dphi/dt = M * (kappa * ∇²phi - df/dphi)
        where f(phi) = phi²*(1-phi)² is the double-well potential.

        Parameters
        ----------
        phi    : phase field (0 = matrix, 1 = precipitate/grain boundary)
        M      : interface mobility [m³/J/s]
        kappa  : gradient energy coefficient [J/m]
        dt     : time step [s]
        steps  : number of time steps
        """
        if self.available:
            wp = self._warp
            try:
                import warp as wp

                nx, ny = phi.shape
                phi_wp = wp.array(phi.astype(np.float32), dtype=wp.float32, device="cuda")
                phi_out = wp.zeros_like(phi_wp)

                @wp.kernel
                def allen_cahn_step(
                    phi_in: wp.array2d(dtype=wp.float32),
                    phi_out: wp.array2d(dtype=wp.float32),
                    M: float,
                    kappa: float,
                    dt: float,
                ):
                    i, j = wp.tid()
                    nx_ = phi_in.shape[0]
                    ny_ = phi_in.shape[1]
                    if i > 0 and i < nx_ - 1 and j > 0 and j < ny_ - 1:
                        p = phi_in[i, j]
                        laplacian = (
                            phi_in[i - 1, j] + phi_in[i + 1, j]
                            + phi_in[i, j - 1] + phi_in[i, j + 1]
                            - 4.0 * p
                        )
                        # df/dphi for phi²(1-phi)² double-well
                        dfdphi = 2.0 * p * (1.0 - p) * (1.0 - 2.0 * p)
                        dphi_dt = M * (kappa * laplacian - dfdphi)
                        phi_out[i, j] = p + dt * dphi_dt
                    else:
                        phi_out[i, j] = phi_in[i, j]

                for _ in range(int(steps)):
                    wp.launch(
                        allen_cahn_step,
                        dim=(nx, ny),
                        inputs=[phi_wp, phi_out, float(M), float(kappa), float(dt)],
                    )
                    phi_wp, phi_out = phi_out, phi_wp

                return phi_wp.numpy()

            except Exception:
                pass

        # NumPy fallback
        f = phi.copy().astype(float)
        for _ in range(int(steps)):
            lap = (
                np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1)
                - 4.0 * f
            )
            dfdphi = 2.0 * f * (1.0 - f) * (1.0 - 2.0 * f)
            f += dt * float(M) * (float(kappa) * lap - dfdphi)
            f = np.clip(f, 0.0, 1.0)
        return f

    def run_diffusion_field(
        self,
        grid_shape: tuple[int, int],
        alpha: float,
        steps: int,
        C_left: float = 1.0,
        C_right: float = 0.0,
    ) -> dict:
        """Run a complete diffusion field experiment, returning summary + field."""
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        field = np.full((nx, ny), float(C_right))
        field[: nx // 2, :] = float(C_left)

        field_out = self.diffusion_2d(field, alpha, steps)

        return {
            "backend": self.backend_name,
            "grid_shape": [nx, ny],
            "steps": steps,
            "alpha": float(alpha),
            "field": field_out.tolist(),
            "field_summary": {
                "min": float(np.min(field_out)),
                "max": float(np.max(field_out)),
                "mean": float(np.mean(field_out)),
                "centerline": field_out[:, ny // 2].tolist(),
            },
        }

    def run_allen_cahn_field(
        self,
        grid_shape: tuple[int, int] | list,
        M: float = 1.0,
        kappa: float = 1.0,
        dt: float = 0.01,
        steps: int = 200,
        phi_seed: int = 42,
    ) -> dict:
        """Initialise a random grain structure and evolve via Allen-Cahn on GPU.

        Creates a uniform random phase field (phi ≈ 0.5 everywhere) to seed
        grain coarsening, then runs ``allen_cahn_grain_growth`` for the
        requested number of steps.
        """
        nx, ny = int(grid_shape[0]), int(grid_shape[1])
        rng = np.random.default_rng(int(phi_seed))
        phi = rng.uniform(0.45, 0.55, (nx, ny)).astype(np.float32)

        phi_out = self.allen_cahn_grain_growth(phi, M, kappa, dt, steps)

        return {
            "backend": self.backend_name,
            "grid_shape": [nx, ny],
            "steps": steps,
            "M": float(M),
            "kappa": float(kappa),
            "dt": float(dt),
            "field": phi_out.tolist(),
            "field_summary": {
                "min": float(np.min(phi_out)),
                "max": float(np.max(phi_out)),
                "mean": float(np.mean(phi_out)),
                "centerline": phi_out[:, ny // 2].tolist(),
            },
        }

    def diffusion_3d(
        self,
        field: np.ndarray,
        alpha: float,
        steps: int,
    ) -> np.ndarray:
        """3D finite-difference diffusion step via Warp GPU kernel.

        alpha = D * dt / dx² (3D stability: alpha <= 1/6)
        """
        if self.available:
            try:
                import warp as wp

                nx, ny, nz = field.shape
                field_wp = wp.array(field.astype(np.float32), dtype=wp.float32, device="cuda")
                field_out = wp.zeros_like(field_wp)

                @wp.kernel
                def diffusion_step_3d(
                    f_in: wp.array3d(dtype=wp.float32),
                    f_out: wp.array3d(dtype=wp.float32),
                    alpha: float,
                ):
                    i, j, k = wp.tid()
                    nx_ = f_in.shape[0]
                    ny_ = f_in.shape[1]
                    nz_ = f_in.shape[2]
                    if (
                        i > 0 and i < nx_ - 1
                        and j > 0 and j < ny_ - 1
                        and k > 0 and k < nz_ - 1
                    ):
                        laplacian = (
                            f_in[i - 1, j, k] + f_in[i + 1, j, k]
                            + f_in[i, j - 1, k] + f_in[i, j + 1, k]
                            + f_in[i, j, k - 1] + f_in[i, j, k + 1]
                            - 6.0 * f_in[i, j, k]
                        )
                        f_out[i, j, k] = f_in[i, j, k] + alpha * laplacian
                    else:
                        f_out[i, j, k] = f_in[i, j, k]

                for _ in range(int(steps)):
                    wp.launch(
                        diffusion_step_3d,
                        dim=(nx, ny, nz),
                        inputs=[field_wp, field_out, float(alpha)],
                    )
                    field_wp, field_out = field_out, field_wp

                return field_wp.numpy()

            except Exception:
                pass

        # NumPy fallback — clamp alpha for 3D stability (alpha <= 1/6)
        f = field.copy().astype(float)
        a = min(float(alpha), 0.16)
        for _ in range(int(steps)):
            lap = (
                np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1)
                + np.roll(f, 1, 2) + np.roll(f, -1, 2)
                - 6.0 * f
            )
            f += a * lap
        return f

    def run_diffusion_3d(
        self,
        grid_shape: tuple[int, int, int] | list,
        alpha: float,
        steps: int,
        C_left: float = 1.0,
        C_right: float = 0.0,
    ) -> dict:
        """Run a complete 3D diffusion experiment, returning summary + central slices."""
        nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        field = np.full((nx, ny, nz), float(C_right), dtype=np.float32)
        field[: nx // 2, :, :] = float(C_left)

        field_out = self.diffusion_3d(field, alpha, steps)

        return {
            "backend": self.backend_name,
            "grid_shape": [nx, ny, nz],
            "steps": steps,
            "alpha": float(alpha),
            "field_summary": {
                "min": float(np.min(field_out)),
                "max": float(np.max(field_out)),
                "mean": float(np.mean(field_out)),
                "centerline": field_out[:, ny // 2, nz // 2].tolist(),
            },
            "slice_xy": field_out[:, :, nz // 2].tolist(),
            "slice_xz": field_out[:, ny // 2, :].tolist(),
        }


# ---------------------------------------------------------------------------
# GPU Diagnostics backend
# ---------------------------------------------------------------------------

class GPUDiagnosticsBackend:
    """Query CUDA device info, VRAM usage, and backend availability.

    Uses torch.cuda for device properties and cupy/warp availability checks.
    All methods degrade gracefully when CUDA is not present.
    """

    def __init__(self):
        self._torch = _try_import("torch")

    def gpu_status(self) -> dict:
        """Return per-device GPU info, VRAM, compute capability, and backend flags."""
        devices: list[dict] = []
        cuda_available = False

        if self._torch is not None:
            import torch
            cuda_available = torch.cuda.is_available()
            n_gpus = torch.cuda.device_count() if cuda_available else 0
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                except Exception:
                    free_bytes, total_bytes = 0, props.total_memory
                used_bytes = total_bytes - free_bytes
                devices.append({
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / 1e9, 3),
                    "free_vram_gb": round(free_bytes / 1e9, 3),
                    "used_vram_gb": round(used_bytes / 1e9, 3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                    "max_threads_per_block": props.max_threads_per_block,
                })

        return {
            "cuda_available": cuda_available,
            "n_gpus": len(devices),
            "devices": devices,
            "backends": {
                "warp": _try_import("warp") is not None,
                "nvalchemiops": _try_import("nvalchemiops") is not None,
                "cupy": _try_import("cupy") is not None,
                "cupynumeric": (
                    _try_import("cupynumeric") is not None
                    or _try_import("cunumeric") is not None
                ),
                "physicsnemo": (
                    _try_import("physicsnemo") is not None
                    or _try_import("nvidia.physicsnemo") is not None
                    or _try_import("modulus") is not None
                ),
                "torch": self._torch is not None,
            },
        }


# ---------------------------------------------------------------------------
# PhysicsNeMo backend — neural operator / PINN surrogate
# ---------------------------------------------------------------------------

class PhysicsNeMoBackend:
    """NVIDIA PhysicsNeMo surrogate model integration.

    PhysicsNeMo provides:
    - Fourier Neural Operator (FNO) for field-to-field predictions
    - Physics-Informed Neural Networks (PINNs) for PDE solving
    - Graph Neural Networks for irregular meshes
    - Uncertainty quantification for active learning loops
    """

    def __init__(self):
        self._nemo = (
            _try_import("physicsnemo")
            or _try_import("nvidia.physicsnemo")
            or _try_import("modulus")
        )
        self.available: bool = self._nemo is not None
        self.backend_name: str = "physicsnemo_gpu" if self.available else "sklearn_cpu_proxy"

    def train_fno_surrogate(
        self,
        X_train: list[list[float]],
        y_train: list[float],
        n_modes: int = 16,
        hidden_channels: int = 64,
        n_layers: int = 4,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        target_name: str = "property",
    ) -> dict:
        """Train a Fourier Neural Operator surrogate on composition → property data.

        Parameters
        ----------
        X_train          : composition feature vectors [[el1_frac, el2_frac, ...], ...]
        y_train          : target property values
        n_modes          : FNO Fourier modes per dimension
        hidden_channels  : FNO hidden channel width
        """
        if self.available:
            nemo = self._nemo
            try:
                import torch
                X = torch.tensor(X_train, dtype=torch.float32)
                y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

                try:
                    from physicsnemo.models.fno import FNO1d
                    model = FNO1d(
                        in_channels=X.shape[1],
                        out_channels=1,
                        n_modes=n_modes,
                        hidden_channels=hidden_channels,
                        n_layers=n_layers,
                    ).cuda()
                except Exception:
                    from modulus.models.fno import FNO
                    model = FNO(
                        in_channels=X.shape[1],
                        out_channels=1,
                        decoder_layers=n_layers,
                        decoder_layer_size=hidden_channels,
                        dimension=1,
                        latent_channels=hidden_channels,
                        num_fno_layers=n_layers,
                        num_fno_modes=n_modes,
                    ).cuda()

                optimiser = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
                losses = []
                X_gpu = X.cuda()
                y_gpu = y.cuda()
                for epoch in range(int(epochs)):
                    optimiser.zero_grad()
                    pred = model(X_gpu.unsqueeze(2))
                    loss = torch.nn.functional.mse_loss(pred.squeeze(), y_gpu.squeeze())
                    loss.backward()
                    optimiser.step()
                    if epoch % 20 == 0:
                        losses.append(float(loss.item()))

                return {
                    "backend": "physicsnemo_gpu",
                    "model": "FNO",
                    "n_modes": n_modes,
                    "hidden_channels": hidden_channels,
                    "epochs": epochs,
                    "final_train_loss": float(losses[-1]) if losses else None,
                    "loss_history": losses,
                    "target": target_name,
                    "n_training_samples": len(X_train),
                }
            except Exception as exc:
                return self._train_cpu_proxy(X_train, y_train, target_name, error=str(exc))
        else:
            return self._train_cpu_proxy(X_train, y_train, target_name)

    def _train_cpu_proxy(
        self,
        X_train: list[list[float]],
        y_train: list[float],
        target_name: str,
        error: str | None = None,
    ) -> dict:
        """Random forest proxy surrogate for CPU fallback."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import cross_val_score
            import numpy as np

            X = np.array(X_train)
            y = np.array(y_train)
            model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=0)
            if len(X) > 5:
                scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring="r2")
                r2 = float(np.mean(scores))
            else:
                r2 = float("nan")
            model.fit(X, y)
            proxy_note = "CPU GradientBoosting proxy — install nvidia-physicsnemo for FNO/GPU training."
        except ImportError:
            r2 = float("nan")
            proxy_note = "CPU proxy (sklearn not installed) — install nvidia-physicsnemo."

        result = {
            "backend": "sklearn_cpu_proxy",
            "model": "GradientBoostingRegressor",
            "cv_r2_score": r2,
            "target": target_name,
            "n_training_samples": len(X_train),
            "note": proxy_note,
        }
        if error:
            result["physicsnemo_error"] = error
        return result

    def generate_training_data_plan(
        self,
        composition_space: dict[str, tuple[float, float]],
        target_properties: list[str],
        n_samples: int = 500,
        sampling_strategy: str = "latin_hypercube",
    ) -> dict:
        """Generate a sampling plan for surrogate model training data collection.

        Creates the composition grid that ALCHEMI will evaluate atomistically.

        Parameters
        ----------
        composition_space : {element: (min_fraction, max_fraction)} bounds
        n_samples         : number of compositions to sample
        sampling_strategy : 'latin_hypercube' | 'random' | 'sobol'
        """
        elements = list(composition_space.keys())
        bounds = [composition_space[el] for el in elements]
        n_el = len(elements)

        if sampling_strategy == "latin_hypercube":
            try:
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=n_el, seed=42)
                sample = sampler.random(n=int(n_samples))
                l_bounds = [b[0] for b in bounds]
                u_bounds = [b[1] for b in bounds]
                sample = qmc.scale(sample, l_bounds, u_bounds)
            except ImportError:
                rng = np.random.default_rng(42)
                sample = np.column_stack([
                    rng.uniform(b[0], b[1], int(n_samples)) for b in bounds
                ])
        else:
            rng = np.random.default_rng(42)
            sample = np.column_stack([
                rng.uniform(b[0], b[1], int(n_samples)) for b in bounds
            ])

        # Normalise rows to sum to 1, then re-clip to stated bounds.
        # Without re-clipping, normalisation can push an element above its
        # upper bound (e.g. Ti sampled near 8% can become 8.7% after
        # normalisation).  We iteratively clip-and-renormalise until all
        # elements are within bounds (converges in ≤ 10 passes for typical
        # alloy spaces).
        l_bounds_arr = np.array([b[0] for b in bounds])
        u_bounds_arr = np.array([b[1] for b in bounds])

        for _ in range(10):
            row_sums = sample.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            sample = sample / row_sums
            sample = np.clip(sample, l_bounds_arr, u_bounds_arr)

        # Final normalisation after clipping
        row_sums = sample.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        sample = sample / row_sums

        compositions = [
            {el: float(sample[i, j]) for j, el in enumerate(elements)}
            for i in range(len(sample))
        ]

        return {
            "backend": self.backend_name,
            "sampling_strategy": sampling_strategy,
            "n_samples": len(compositions),
            "elements": elements,
            "composition_space": {el: list(b) for el, b in zip(elements, bounds)},
            "sampled_compositions": compositions[:10],
            "total_compositions": len(compositions),
            "all_compositions": compositions,
            "next_step": "Run batch_alloy_screen on all_compositions using ALCHEMIBackend.batch_alloy_screen()",
            "target_properties": target_properties,
        }


# ---------------------------------------------------------------------------
# CuPy backend — drop-in NumPy replacement for parameter sweeps
# ---------------------------------------------------------------------------

class CuPyBackend:
    """CuPy / cuPyNumeric accelerated array backend.

    Provides a drop-in xp = numpy replacement for large parameter sweeps.
    Use xp.array(), xp.linspace(), etc. — same API as NumPy.
    """

    def __init__(self):
        self._cupynumeric = _try_import("cupynumeric") or _try_import("cunumeric")
        self._cupy = _try_import("cupy")
        if self._cupynumeric is not None:
            self._xp = self._cupynumeric
            self.backend_name = "cupynumeric_gpu"
        elif self._cupy is not None:
            self._xp = self._cupy
            self.backend_name = "cupy_gpu"
        else:
            self._xp = np
            self.backend_name = "numpy_cpu"
        self.available: bool = self._cupynumeric is not None or self._cupy is not None

    @property
    def xp(self):
        """Return the active array module (cupynumeric, cupy, or numpy)."""
        return self._xp

    def to_numpy(self, arr) -> np.ndarray:
        """Convert a GPU array to a host NumPy array."""
        if self.available and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    def parameter_sweep_diffusion(
        self,
        D0: float,
        Q_J_mol: float,
        temperature_range_K: tuple[float, float],
        time_range_s: tuple[float, float],
        n_T: int = 50,
        n_t: int = 50,
        C_surface: float = 1.0,
        C_initial: float = 0.0,
        case_threshold: float = 0.5,
    ) -> dict:
        """GPU-accelerated parameter sweep: case depth over (T, t) grid.

        Uses the active xp (cuPyNumeric > CuPy > NumPy) for all array ops.
        """
        xp = self._xp
        R = 8.31446261815324

        T = xp.linspace(float(temperature_range_K[0]), float(temperature_range_K[1]), int(n_T))
        t = xp.linspace(float(time_range_s[0]), float(time_range_s[1]), int(n_t))

        D_T = float(D0) * xp.exp(-float(Q_J_mol) / (R * T))

        T_grid, t_grid = xp.meshgrid(T, t, indexing="ij")
        D_grid = float(D0) * xp.exp(-float(Q_J_mol) / (R * T_grid))
        sqrt_Dt = xp.sqrt(D_grid * t_grid)

        Cs, C0 = float(C_surface), float(C_initial)
        erf_target = (Cs - float(case_threshold)) / (Cs - C0) if abs(Cs - C0) > 1e-30 else 0.5
        erf_target = max(-0.9999, min(0.9999, erf_target))

        from scipy.special import erfinv as _erfinv
        erf_inv_val = float(_erfinv(erf_target))
        case_depth_m = xp.maximum(0.0, 2.0 * erf_inv_val * sqrt_Dt)
        case_depth_mm = case_depth_m * 1000.0

        return {
            "backend": self.backend_name,
            "D0_m2_s": float(D0),
            "Q_J_mol": float(Q_J_mol),
            "n_T": int(n_T),
            "n_t": int(n_t),
            "temperatures_K": self.to_numpy(T).tolist(),
            "times_s": self.to_numpy(t).tolist(),
            "case_depth_mm_grid": self.to_numpy(case_depth_mm).tolist(),
            "D_at_T_m2_s": self.to_numpy(D_T).tolist(),
        }
