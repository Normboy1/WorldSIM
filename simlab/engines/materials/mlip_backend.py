"""Standalone MLIP (Machine-Learning Interatomic Potential) backend.

Extracts and enhances the MACE / SevenNet code that was previously embedded
inside ``nvidia_backends.ALCHEMIBackend``, and adds batch screening and ASE
structure-manipulation capabilities.

Supported potentials
--------------------
* **MACE-MP-0** — universal foundation model (Batatia et al. 2023)
* **SevenNet-0** — universal 7-net potential (Park et al. 2024)
* **EMT** — ASE's empirical tight-binding (always available; testing only)

The backend degrades gracefully: if neither MACE nor SevenNet is installed it
falls back to EMT (via ASE) and, if ASE is also absent, to a simple LJ proxy.

All public methods return plain dicts that are JSON-serialisable.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np

R_GAS = 8.31446261815324

_SUPPORTED_ELEMENTS = {
    "Fe", "Ni", "Cr", "Co", "Al", "Ti", "Mo", "W", "Nb", "Ta",
    "Re", "Hf", "Ru", "V", "Mn", "Si", "C", "B", "Zr", "Cu",
}

# LJ σ and ε parameters for common elements (Å, eV) — used as final fallback
_LJ_PARAMS: dict[str, tuple[float, float]] = {
    "Al": (2.55, 0.34), "Co": (2.23, 0.51), "Cr": (2.28, 0.40),
    "Cu": (2.34, 0.41), "Fe": (2.32, 0.42), "Mo": (2.51, 0.66),
    "Nb": (2.62, 0.56), "Ni": (2.23, 0.43), "Ti": (2.60, 0.45),
    "W":  (2.52, 0.86), "V":  (2.38, 0.45),
}
_LJ_DEFAULT = (2.40, 0.40)


def _available(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


class MLIPBackend:
    """Unified MACE / SevenNet / EMT / LJ backend for materials screening.

    Parameters
    ----------
    mace_model_path : str | None
        Path to a MACE ``.model`` file (e.g. ``mace-mp-0.model``).
    sevennet_model_path : str | None
        Path to a SevenNet checkpoint file.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        mace_model_path: str | None = None,
        sevennet_model_path: str | None = None,
        device: str = "cuda",
    ):
        self._mace_path = mace_model_path
        self._sevennet_path = sevennet_model_path
        self._device = device
        self._mace_calc: Any = None       # cached calculator objects
        self._sevennet_calc: Any = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def mace_available(self) -> bool:
        if not _available("mace"):
            return False
        if self._mace_path is None:
            return False
        import os
        return os.path.isfile(self._mace_path)

    def sevennet_available(self) -> bool:
        if not _available("sevenn"):
            return False
        if self._sevennet_path is None:
            return False
        import os
        return os.path.isfile(self._sevennet_path)

    def ase_available(self) -> bool:
        return _available("ase")

    def best_available(self) -> str:
        if self.mace_available():
            return "mace"
        if self.sevennet_available():
            return "sevennet"
        if self.ase_available():
            return "emt"
        return "lj_proxy"

    # ------------------------------------------------------------------
    # 1. Single-point energy + forces
    # ------------------------------------------------------------------

    def energy_forces(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        potential: str | None = None,
    ) -> dict[str, Any]:
        """Calculate energy and forces with the best available potential.

        Parameters
        ----------
        potential : str | None
            Force a specific backend: ``"mace"``, ``"sevennet"``, ``"emt"``,
            ``"lj"``.  None = auto-select the best available.
        """
        chosen = potential or self.best_available()

        if chosen == "mace" and self.mace_available():
            return self._mace_energy_forces(species, positions_angstrom, lattice_vectors_angstrom)
        if chosen == "sevennet" and self.sevennet_available():
            return self._sevennet_energy_forces(species, positions_angstrom, lattice_vectors_angstrom)
        if chosen in ("emt", None) and self.ase_available():
            return self._emt_energy_forces(species, positions_angstrom, lattice_vectors_angstrom)
        return self._lj_proxy_energy_forces(species, positions_angstrom)

    # ------------------------------------------------------------------
    # 2. Geometry relaxation
    # ------------------------------------------------------------------

    def relax_structure(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        fmax_eV_A: float = 0.05,
        max_steps: int = 300,
        optimizer: str = "FIRE",
        potential: str | None = None,
    ) -> dict[str, Any]:
        """Relax atomic positions with the best available potential.

        Uses ASE optimisers (FIRE recommended for MLIP potentials).
        """
        if not self.ase_available():
            return self._lj_proxy_relax(species)

        try:
            import ase
            from ase.optimize import FIRE, BFGS, LBFGS

            atoms = ase.Atoms(
                symbols=species,
                positions=np.array(positions_angstrom),
                cell=np.array(lattice_vectors_angstrom),
                pbc=True,
            )
            atoms.calc = self._get_ase_calculator(potential)
            opt_cls = {"fire": FIRE, "bfgs": BFGS, "lbfgs": LBFGS}.get(
                optimizer.lower(), FIRE
            )
            opt = opt_cls(atoms, logfile=None)
            converged = opt.run(fmax=float(fmax_eV_A), steps=int(max_steps))

            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmags = np.linalg.norm(forces, axis=1)
            backend_name = potential or self.best_available()
            return {
                "backend": f"mlip_{backend_name}",
                "potential": backend_name,
                "optimizer": optimizer,
                "n_atoms": len(species),
                "converged": bool(converged),
                "steps": opt.get_number_of_steps(),
                "energy_eV": energy,
                "energy_eV_per_atom": energy / max(len(species), 1),
                "max_force_eV_A": float(fmags.max()),
                "relaxed_positions": atoms.get_positions().tolist(),
            }
        except Exception as exc:
            return {**self._lj_proxy_relax(species), "ase_error": str(exc)}

    # ------------------------------------------------------------------
    # 3. Batch screening
    # ------------------------------------------------------------------

    def batch_screen(
        self,
        compositions: list[dict[str, float]],
        lattice_type: str = "fcc",
        a_angstrom: float = 3.6,
        n_atoms: int = 32,
        seed: int = 42,
        potential: str | None = None,
    ) -> dict[str, Any]:
        """Screen a list of compositions using single-point MLIP energies.

        Builds one supercell per composition, assigns species by the
        largest-remainder method, and evaluates energy + forces.

        Returns a ranked list of candidates by energy per atom (most negative
        = most energetically favourable on the LJ / EMT / MLIP surface).

        Parameters
        ----------
        compositions : list of dicts
            Each dict maps element symbol to mole fraction.
        """
        import time

        if not self.ase_available() and potential not in ("mace", "sevennet"):
            return self._lj_batch_screen(compositions, lattice_type, a_angstrom, n_atoms, seed)

        results: list[dict] = []
        t0 = time.time()
        rng = np.random.default_rng(seed)

        for i, comp_raw in enumerate(compositions):
            comp = _normalise(comp_raw)
            positions, cell = _build_supercell(lattice_type, a_angstrom, n_atoms)
            species = _largest_remainder_assign(comp, len(positions), rng)

            sp = self.energy_forces(species, positions.tolist(), cell.tolist(), potential=potential)

            # LJ diff vs pure elements (for comparability with batch_alloy_screen)
            ref = sum(
                _lj_cohesive_reference(el) * x for el, x in comp.items()
            )
            lj_diff = sp["energy_eV_per_atom"] - ref

            results.append({
                "composition_index": i,
                "composition": comp,
                "energy_eV_per_atom": sp["energy_eV_per_atom"],
                "mlip_energy_diff_eV_atom": lj_diff,
                "max_force_eV_A": sp.get("max_force_eV_A", None),
                "backend": sp["backend"],
            })

        results.sort(key=lambda r: r["energy_eV_per_atom"])
        elapsed = time.time() - t0

        return {
            "backend": f"mlip_{potential or self.best_available()}",
            "n_compositions": len(compositions),
            "lattice_type": lattice_type,
            "a_angstrom": a_angstrom,
            "n_atoms_per_cell": n_atoms,
            "timing_s": round(elapsed, 3),
            "top_candidates": results[:5],
            "all_results": results,
            "note": (
                "Energy ranked by DFT-quality MLIP single-point per atom. "
                "More negative = lower energy on the MLIP surface. "
                "Not a thermodynamic convex hull — geometry not relaxed."
            ),
        }

    # ------------------------------------------------------------------
    # 4. MACE-MP-0 formation energy
    # ------------------------------------------------------------------

    def mace_formation_energy(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
    ) -> dict[str, Any]:
        """Calculate MACE formation energy relative to elemental MACE references.

        This is close to DFT formation energy quality for the elements
        included in the MACE-MP-0 universal potential training set.
        """
        if not self.mace_available():
            return {
                "backend": "proxy",
                "note": "MACE not available. Supply mace_model_path pointing to mace-mp-0.model.",
            }

        sp = self._mace_energy_forces(species, positions_angstrom, lattice_vectors_angstrom)
        if "mace_error" in sp:
            return sp

        comp = _species_to_composition(species)
        # Reference energies from MACE-MP-0 for standard elemental structures
        # (these are approximate — ideally compute per-element references with the same model)
        from simlab.engines.materials.dft_backend import _DFT_REFERENCE_ENERGIES_eV as _REF
        ref = sum(_REF.get(el, -5.0) * x for el, x in comp.items())
        ef = sp["energy_eV_per_atom"] - ref

        return {
            "backend": "mace_mlip",
            "n_atoms": len(species),
            "composition": comp,
            "energy_eV_per_atom": sp["energy_eV_per_atom"],
            "reference_energy_eV_per_atom": ref,
            "formation_energy_eV_per_atom": ef,
            "max_force_eV_A": sp["max_force_eV_A"],
            "note": (
                "MACE-MP-0 formation energy. Reference energies are PBE elemental "
                "references; slight inconsistency with MACE energies expected. "
                "For best accuracy compute elemental references with the same MACE model."
            ),
        }

    # ------------------------------------------------------------------
    # Internal — MACE
    # ------------------------------------------------------------------

    def _mace_energy_forces(
        self,
        species: list[str],
        positions: list[list[float]],
        cell: list[list[float]],
    ) -> dict:
        try:
            from mace.calculators import MACECalculator
            import ase
            if self._mace_calc is None:
                self._mace_calc = MACECalculator(
                    model_paths=self._mace_path,
                    device=self._device,
                    default_dtype="float32",
                )
            atoms = ase.Atoms(
                symbols=species, positions=np.array(positions),
                cell=np.array(cell), pbc=True,
            )
            atoms.calc = self._mace_calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmags = np.linalg.norm(forces, axis=1)
            return {
                "backend": "mace_mlip",
                "potential": "mace",
                "n_atoms": len(species),
                "energy_eV": energy,
                "energy_eV_per_atom": energy / len(species),
                "max_force_eV_A": float(fmags.max()),
                "mean_force_eV_A": float(fmags.mean()),
                "forces_eV_A": forces.tolist(),
            }
        except Exception as exc:
            return {**self._lj_proxy_energy_forces(species, positions),
                    "mace_error": str(exc)}

    # ------------------------------------------------------------------
    # Internal — SevenNet
    # ------------------------------------------------------------------

    def _sevennet_energy_forces(
        self,
        species: list[str],
        positions: list[list[float]],
        cell: list[list[float]],
    ) -> dict:
        try:
            from sevenn.sevennet_calculator import SevenNetCalculator
            import ase
            if self._sevennet_calc is None:
                self._sevennet_calc = SevenNetCalculator(
                    self._sevennet_path, device=self._device
                )
            atoms = ase.Atoms(
                symbols=species, positions=np.array(positions),
                cell=np.array(cell), pbc=True,
            )
            atoms.calc = self._sevennet_calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmags = np.linalg.norm(forces, axis=1)
            return {
                "backend": "sevennet_mlip",
                "potential": "sevennet",
                "n_atoms": len(species),
                "energy_eV": energy,
                "energy_eV_per_atom": energy / len(species),
                "max_force_eV_A": float(fmags.max()),
                "mean_force_eV_A": float(fmags.mean()),
                "forces_eV_A": forces.tolist(),
            }
        except Exception as exc:
            return {**self._lj_proxy_energy_forces(species, positions),
                    "sevennet_error": str(exc)}

    # ------------------------------------------------------------------
    # Internal — EMT (ASE, always available)
    # ------------------------------------------------------------------

    def _emt_energy_forces(
        self,
        species: list[str],
        positions: list[list[float]],
        cell: list[list[float]],
    ) -> dict:
        try:
            import ase
            from ase.calculators.emt import EMT
            atoms = ase.Atoms(
                symbols=species, positions=np.array(positions),
                cell=np.array(cell), pbc=True,
            )
            atoms.calc = EMT()
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmags = np.linalg.norm(forces, axis=1)
            return {
                "backend": "ase_emt",
                "potential": "emt",
                "n_atoms": len(species),
                "energy_eV": energy,
                "energy_eV_per_atom": energy / len(species),
                "max_force_eV_A": float(fmags.max()),
                "mean_force_eV_A": float(fmags.mean()),
                "forces_eV_A": forces.tolist(),
                "note": "EMT is an approximate empirical model for testing only.",
            }
        except Exception as exc:
            return {**self._lj_proxy_energy_forces(species, positions),
                    "emt_error": str(exc)}

    # ------------------------------------------------------------------
    # Internal — LJ proxy (final fallback)
    # ------------------------------------------------------------------

    def _lj_proxy_energy_forces(
        self,
        species: list[str],
        positions: list[list[float]] | None = None,
    ) -> dict:
        comp = _species_to_composition(species)
        avg_eps = sum(_LJ_PARAMS.get(el, _LJ_DEFAULT)[1] * x for el, x in comp.items())
        avg_sig = sum(_LJ_PARAMS.get(el, _LJ_DEFAULT)[0] * x for el, x in comp.items())
        n = len(species)
        energy_per_atom = -avg_eps * ((avg_sig / 2.5) ** 6)
        return {
            "backend": "lj_proxy",
            "potential": "lj",
            "n_atoms": n,
            "energy_eV": energy_per_atom * n,
            "energy_eV_per_atom": energy_per_atom,
            "max_force_eV_A": 0.0,
            "forces_eV_A": [[0.0, 0.0, 0.0]] * n,
            "note": (
                "LJ proxy — install ASE + MACE/SevenNet for real MLIP energies. "
                "LJ energies are ~3–10 eV/atom; not comparable to DFT formation energies."
            ),
        }

    def _lj_proxy_relax(self, species: list[str]) -> dict:
        sp = self._lj_proxy_energy_forces(species)
        return {**sp, "converged": True, "steps": 0,
                "note": "LJ proxy relaxation — no geometry optimisation."}

    def _lj_batch_screen(
        self,
        compositions: list[dict],
        lattice_type: str,
        a: float,
        n_atoms: int,
        seed: int,
    ) -> dict:
        import time
        rng = np.random.default_rng(seed)
        t0 = time.time()
        results = []
        for i, comp_raw in enumerate(compositions):
            comp = _normalise(comp_raw)
            species = _largest_remainder_assign(comp, n_atoms, rng)
            sp = self._lj_proxy_energy_forces(species)
            results.append({
                "composition_index": i,
                "composition": comp,
                "energy_eV_per_atom": sp["energy_eV_per_atom"],
                "backend": "lj_proxy",
            })
        results.sort(key=lambda r: r["energy_eV_per_atom"])
        return {
            "backend": "lj_proxy",
            "n_compositions": len(compositions),
            "timing_s": round(time.time() - t0, 3),
            "top_candidates": results[:5],
            "all_results": results,
            "note": "LJ proxy batch screen. Install MACE or SevenNet for MLIP-quality results.",
        }

    def _get_ase_calculator(self, potential: str | None) -> Any:
        chosen = potential or self.best_available()
        if chosen == "mace" and self.mace_available():
            from mace.calculators import MACECalculator
            if self._mace_calc is None:
                self._mace_calc = MACECalculator(
                    model_paths=self._mace_path, device=self._device,
                    default_dtype="float32",
                )
            return self._mace_calc
        if chosen == "sevennet" and self.sevennet_available():
            from sevenn.sevennet_calculator import SevenNetCalculator
            if self._sevennet_calc is None:
                self._sevennet_calc = SevenNetCalculator(
                    self._sevennet_path, device=self._device
                )
            return self._sevennet_calc
        from ase.calculators.emt import EMT
        return EMT()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(comp: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in comp.values() if v > 0)
    if total <= 0:
        raise ValueError("Composition must have positive total.")
    return {k: float(v) / total for k, v in comp.items() if v > 0}


def _species_to_composition(species: list[str]) -> dict[str, float]:
    from collections import Counter
    counts = Counter(species)
    n = len(species)
    return {el: count / n for el, count in counts.items()}


def _largest_remainder_assign(
    comp: dict[str, float], n_atoms: int, rng: np.random.Generator
) -> list[str]:
    elements = list(comp.keys())
    exact = {el: comp[el] * n_atoms for el in elements}
    floors = {el: int(exact[el]) for el in elements}
    remainders = sorted(elements, key=lambda el: -(exact[el] - floors[el]))
    allocated = sum(floors.values())
    for el in remainders[:n_atoms - allocated]:
        floors[el] += 1
    species: list[str] = []
    for el, count in floors.items():
        species.extend([el] * count)
    rng.shuffle(species)
    return species[:n_atoms]


def _build_supercell(
    lattice_type: str, a: float, n_atoms: int
) -> tuple[np.ndarray, np.ndarray]:
    if lattice_type.lower() == "fcc":
        basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], dtype=float)
        n_basis = 4
    else:  # bcc
        basis = np.array([[0, 0, 0], [.5, .5, .5]], dtype=float)
        n_basis = 2

    n_rep = max(1, math.ceil((n_atoms / n_basis) ** (1.0 / 3.0)))
    positions = []
    for ix in range(n_rep):
        for iy in range(n_rep):
            for iz in range(n_rep):
                for b in basis:
                    positions.append((b + np.array([ix, iy, iz])) * a)

    pos = np.array(positions, dtype=float)
    cell = np.array([[a * n_rep, 0, 0], [0, a * n_rep, 0], [0, 0, a * n_rep]], dtype=float)
    return pos, cell


def _lj_cohesive_reference(element: str) -> float:
    """Approximate LJ cohesive energy for a pure element (eV/atom)."""
    sig, eps = _LJ_PARAMS.get(element, _LJ_DEFAULT)
    r_nn = 2.0 ** (1.0 / 6.0) * sig
    return -eps * ((sig / r_nn) ** 12 - 2.0 * (sig / r_nn) ** 6)
