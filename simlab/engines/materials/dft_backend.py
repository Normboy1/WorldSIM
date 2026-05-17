"""ASE-based DFT backend for formation energy and geometry relaxation.

Wraps ASE calculator interfaces (VASP, Quantum ESPRESSO, CP2K, EMT) with a
graceful fallback proxy so the system remains functional without DFT codes.

All public methods return plain dicts that are JSON-serialisable.

Terminology note
----------------
``formation_energy_eV_atom`` here refers to the DFT formation energy per atom
relative to the elemental reference energies, typically in the range ±0.5 eV/atom
for metallic alloys.  This is *not* the Lennard-Jones cohesive energy (which is
~3–10 eV/atom in magnitude).
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np


def _ase_available() -> bool:
    return importlib.util.find_spec("ase") is not None


# DFT reference energies from standard VASP PBE calculations (eV/atom).
# These are approximate elemental reference energies for formation energy
# calculation relative to standard states. Sourced from AFLOW/Materials Project.
_DFT_REFERENCE_ENERGIES_eV: dict[str, float] = {
    "Al": -3.747, "C":  -9.210, "Co": -7.108, "Cr": -9.508, "Cu": -3.727,
    "Fe": -8.311, "Mo": -10.849, "Nb": -10.101, "Ni": -5.468, "Ta": -11.858,
    "Ti": -7.837, "V":  -8.986, "W":  -12.958, "Re": -12.428, "Ru": -9.219,
    "Hf": -9.956, "Mn": -9.031, "Si": -5.425, "B":  -6.679, "Zr": -8.547,
}

R_GAS = 8.31446261815324


class DFTBackend:
    """Wraps ASE calculators for DFT single-point and relaxation calculations.

    Parameters
    ----------
    calculator : str
        One of ``"vasp"``, ``"espresso"``, ``"cp2k"``, or ``"emt"`` (always
        available via ASE for quick demos).
    config : dict | None
        Calculator-specific keyword arguments forwarded to the ASE calculator
        constructor.  See each calculator's ASE documentation.
    """

    def __init__(
        self,
        calculator: str = "emt",
        config: dict[str, Any] | None = None,
    ):
        self._calculator_name = calculator.lower()
        self._config: dict[str, Any] = config or {}
        self._ase_available = _ase_available()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def available(self) -> bool:
        """True when ASE is installed and the chosen calculator is accessible."""
        if not self._ase_available:
            return False
        if self._calculator_name == "emt":
            return True   # ASE always ships EMT
        return self._calculator_available(self._calculator_name)

    def _calculator_available(self, name: str) -> bool:
        mapping = {
            "vasp":     "ase.calculators.vasp",
            "espresso": "ase.calculators.espresso",
            "cp2k":     "ase.calculators.cp2k",
        }
        mod = mapping.get(name)
        if mod is None:
            return False
        try:
            importlib.import_module(mod)
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # 1. Single-point energy and forces
    # ------------------------------------------------------------------

    def calculate_single_point(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
    ) -> dict[str, Any]:
        """Run a single-point DFT (or EMT) energy + forces calculation.

        Returns total energy, per-atom energy, max force, and forces array.
        """
        if not self._ase_available:
            return self._proxy_single_point(species)

        try:
            import ase
            atoms = ase.Atoms(
                symbols=species,
                positions=np.array(positions_angstrom),
                cell=np.array(lattice_vectors_angstrom),
                pbc=True,
            )
            calc = self._build_calculator()
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmags = np.linalg.norm(forces, axis=1)
            return {
                "backend": f"ase_{self._calculator_name}",
                "calculator": self._calculator_name,
                "n_atoms": len(species),
                "energy_eV": energy,
                "energy_eV_per_atom": energy / len(species),
                "max_force_eV_A": float(fmags.max()),
                "mean_force_eV_A": float(fmags.mean()),
                "forces_eV_A": forces.tolist(),
            }
        except Exception as exc:
            return {**self._proxy_single_point(species), "ase_error": str(exc)}

    # ------------------------------------------------------------------
    # 2. Geometry relaxation
    # ------------------------------------------------------------------

    def relax_structure(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        fmax_eV_A: float = 0.05,
        max_steps: int = 200,
        optimizer: str = "BFGS",
    ) -> dict[str, Any]:
        """Relax atomic positions until max(|F|) < ``fmax_eV_A``.

        Uses the ASE optimizer specified (BFGS, FIRE, LBFGS).

        Returns converged energy, final max force, and step count.
        """
        if not self._ase_available:
            return self._proxy_relax(species)

        try:
            import ase
            from ase.optimize import BFGS, FIRE, LBFGS

            atoms = ase.Atoms(
                symbols=species,
                positions=np.array(positions_angstrom),
                cell=np.array(lattice_vectors_angstrom),
                pbc=True,
            )
            atoms.calc = self._build_calculator()

            opt_cls = {"bfgs": BFGS, "fire": FIRE, "lbfgs": LBFGS}.get(
                optimizer.lower(), BFGS
            )
            opt = opt_cls(atoms, logfile=None)
            converged = opt.run(fmax=float(fmax_eV_A), steps=int(max_steps))

            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmags = np.linalg.norm(forces, axis=1)
            return {
                "backend": f"ase_{self._calculator_name}",
                "calculator": self._calculator_name,
                "optimizer": optimizer,
                "n_atoms": len(species),
                "converged": bool(converged),
                "steps": opt.get_number_of_steps(),
                "energy_eV": energy,
                "energy_eV_per_atom": energy / len(species),
                "max_force_eV_A": float(fmags.max()),
                "relaxed_positions": atoms.get_positions().tolist(),
            }
        except Exception as exc:
            return {**self._proxy_relax(species), "ase_error": str(exc)}

    # ------------------------------------------------------------------
    # 3. Formation energy
    # ------------------------------------------------------------------

    def calculate_formation_energy(
        self,
        species: list[str],
        positions_angstrom: list[list[float]],
        lattice_vectors_angstrom: list[list[float]],
        relax_first: bool = True,
    ) -> dict[str, Any]:
        """Compute DFT formation energy relative to elemental references.

        ΔHf/atom = E_alloy/atom − Σ xᵢ · E_ref_i

        where E_ref_i is the DFT reference energy of pure element i.

        With the EMT calculator these values are not DFT quality but give
        structurally correct formation energies for testing the workflow.
        """
        if not self._ase_available:
            comp = _species_to_composition(species)
            return self._proxy_formation_energy(comp)

        try:
            if relax_first:
                relax_result = self.relax_structure(
                    species, positions_angstrom, lattice_vectors_angstrom
                )
                energy_per_atom = relax_result["energy_eV_per_atom"]
                n_atoms = relax_result["n_atoms"]
                converged = relax_result.get("converged", False)
            else:
                sp_result = self.calculate_single_point(
                    species, positions_angstrom, lattice_vectors_angstrom
                )
                energy_per_atom = sp_result["energy_eV_per_atom"]
                n_atoms = sp_result["n_atoms"]
                converged = True

            comp = _species_to_composition(species)
            ref_energy = sum(
                x * _DFT_REFERENCE_ENERGIES_eV.get(el, -5.0)
                for el, x in comp.items()
            )
            formation_energy = energy_per_atom - ref_energy

            on_hull = formation_energy < 0.0

            return {
                "backend": f"ase_{self._calculator_name}",
                "calculator": self._calculator_name,
                "n_atoms": n_atoms,
                "converged": converged,
                "composition": comp,
                "energy_eV_per_atom": energy_per_atom,
                "reference_energy_eV_per_atom": ref_energy,
                "formation_energy_eV_per_atom": formation_energy,
                "likely_below_convex_hull": on_hull,
                "note": (
                    "Formation energy relative to elemental references from "
                    "_DFT_REFERENCE_ENERGIES_eV (PBE approximation). "
                    "Convex hull position requires comparison against all "
                    "competing phases, not just elemental references."
                ),
            }
        except Exception as exc:
            comp = _species_to_composition(species)
            return {**self._proxy_formation_energy(comp), "ase_error": str(exc)}

    # ------------------------------------------------------------------
    # 4. SQS generation
    # ------------------------------------------------------------------

    def generate_sqs(
        self,
        composition: dict[str, float],
        n_atoms: int = 32,
        lattice_type: str = "fcc",
        a_angstrom: float = 3.6,
        method: str = "random",
        seed: int = 42,
    ) -> dict[str, Any]:
        """Generate a Special Quasirandom Structure (SQS).

        When ``atat`` or ``sqsgenerator`` is unavailable, falls back to a
        random assignment on the specified lattice (a random SQS approximation).
        The returned structure can be used as input for single-point or relax
        calculations.

        Parameters
        ----------
        method : str
            ``"random"`` (always available) or ``"sqsgenerator"`` (requires
            the ``sqsgenerator`` package).
        """
        comp = _normalise(composition)

        if not self._ase_available:
            return self._proxy_sqs(comp, n_atoms, lattice_type, a_angstrom)

        try:
            import ase
            import ase.build

            rng = np.random.default_rng(seed)
            a = float(a_angstrom)

            if lattice_type.lower() == "fcc":
                bulk = ase.build.bulk("X", crystalstructure="fcc", a=a)
            else:
                bulk = ase.build.bulk("X", crystalstructure="bcc", a=a)

            # Build supercell large enough to hold n_atoms
            n_rep = max(1, math.ceil((n_atoms / len(bulk)) ** (1.0 / 3.0)))
            supercell = bulk.repeat(n_rep)
            N = len(supercell)

            # Assign species using largest-remainder
            assigned = _largest_remainder_assign(comp, N, rng)
            supercell.set_chemical_symbols(assigned)

            return {
                "backend": f"ase_{self._calculator_name}",
                "method": method,
                "lattice_type": lattice_type,
                "a_angstrom": a,
                "n_atoms": N,
                "composition": comp,
                "species": assigned,
                "positions_angstrom": supercell.get_positions().tolist(),
                "lattice_vectors_angstrom": supercell.get_cell().tolist(),
                "note": (
                    "Random SQS approximation. For true SQS generation use "
                    "the sqsgenerator package or ATAT mcsqs."
                ),
            }
        except Exception as exc:
            return {**self._proxy_sqs(comp, n_atoms, lattice_type, a_angstrom),
                    "ase_error": str(exc)}

    # ------------------------------------------------------------------
    # Proxy implementations
    # ------------------------------------------------------------------

    def _proxy_single_point(self, species: list[str]) -> dict:
        comp = _species_to_composition(species)
        ref = sum(_DFT_REFERENCE_ENERGIES_eV.get(el, -5.0) * x for el, x in comp.items())
        return {
            "backend": "proxy",
            "calculator": self._calculator_name,
            "n_atoms": len(species),
            "energy_eV_per_atom": ref,
            "max_force_eV_A": 0.0,
            "forces_eV_A": [[0.0, 0.0, 0.0]] * len(species),
            "note": "Proxy result — ASE not installed or calculator not available.",
        }

    def _proxy_relax(self, species: list[str]) -> dict:
        sp = self._proxy_single_point(species)
        return {**sp, "converged": True, "steps": 0,
                "note": "Proxy relaxation — no geometry optimisation performed."}

    def _proxy_formation_energy(self, comp: dict) -> dict:
        ref = sum(_DFT_REFERENCE_ENERGIES_eV.get(el, -5.0) * x for el, x in comp.items())
        return {
            "backend": "proxy",
            "calculator": self._calculator_name,
            "composition": comp,
            "energy_eV_per_atom": ref,
            "reference_energy_eV_per_atom": ref,
            "formation_energy_eV_per_atom": 0.0,
            "likely_below_convex_hull": None,
            "note": (
                "Proxy formation energy — ASE not installed. "
                "Value is composition-weighted elemental reference (ΔHf = 0 by construction). "
                "Install ASE and a DFT calculator for real formation energies."
            ),
        }

    def _proxy_sqs(self, comp: dict, n_atoms: int,
                   lattice_type: str, a: float) -> dict:
        return {
            "backend": "proxy",
            "method": "proxy",
            "lattice_type": lattice_type,
            "a_angstrom": a,
            "n_atoms": n_atoms,
            "composition": comp,
            "species": [],
            "positions_angstrom": [],
            "lattice_vectors_angstrom": [],
            "note": "ASE not installed — SQS generation unavailable.",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_calculator(self) -> Any:
        if self._calculator_name == "vasp":
            from ase.calculators.vasp import Vasp
            return Vasp(**self._config)
        if self._calculator_name in ("espresso", "quantum_espresso"):
            from ase.calculators.espresso import Espresso
            return Espresso(**self._config)
        if self._calculator_name == "cp2k":
            from ase.calculators.cp2k import CP2K
            return CP2K(**self._config)
        # Default: EMT (always available, suitable for testing and demos)
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
