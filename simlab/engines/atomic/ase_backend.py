"""ASE (Atomic Simulation Environment) backend.

Wraps ASE for crystal structure generation, geometry optimization,
molecular building, and structure visualization.
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np

try:
    import ase
    import ase.build
    import ase.lattice
    import ase.io
    from ase import Atoms
    from ase.build import bulk, molecule, surface
    from ase.visualize.plot import plot_atoms
    from ase.data import atomic_masses, chemical_symbols, atomic_numbers, covalent_radii
    from ase.geometry import get_distances
    _ASE_OK = True
except ImportError:
    _ASE_OK = False


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _require_ase() -> None:
    if not _ASE_OK:
        raise ImportError("ASE not installed. Run: pip install ase")


class ASECrystalEngine:
    """Crystal structure generation and analysis using ASE."""

    def build_bulk_crystal(self, symbol: str, crystalstructure: str = "fcc",
                           a: float | None = None) -> dict:
        """Build a bulk crystal structure.

        Parameters
        ----------
        symbol          : element symbol, e.g. 'Cu', 'Fe', 'Si'
        crystalstructure: 'fcc', 'bcc', 'sc', 'hcp', 'diamond', 'zincblende', 'rocksalt'
        a               : lattice constant in Angstroms (None = use ASE defaults)
        """
        _require_ase()
        kwargs = dict(name=symbol, crystalstructure=crystalstructure)
        if a is not None:
            kwargs["a"] = a

        atoms = bulk(**kwargs)
        pos = atoms.get_positions()
        cell = atoms.get_cell().tolist()

        return {
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            "crystal_structure": crystalstructure,
            "lattice_constant_A": float(atoms.cell.lengths()[0]),
            "cell_angles_deg": atoms.cell.angles().tolist(),
            "cell_volume_A3": round(float(atoms.get_volume()), 4),
            "positions_A": pos.tolist(),
            "symbols": list(atoms.get_chemical_symbols()),
            "cell_matrix": cell,
            "density_g_cm3": round(
                sum(atomic_masses[atomic_numbers[s]] for s in atoms.get_chemical_symbols())
                / (atoms.get_volume() * 1e-24 * 6.022e23), 4
            ),
        }

    def build_molecule(self, name: str) -> dict:
        """Build a molecule from ASE's built-in library.

        Common names: H2, H2O, CO2, CH4, NH3, C2H6, benzene, etc.
        """
        _require_ase()
        mol = molecule(name)
        pos = mol.get_positions()
        syms = mol.get_chemical_symbols()

        # Interatomic distances
        dist_matrix = get_distances(pos, pos)[1]
        bond_pairs = []
        cov_r = {s: float(covalent_radii[atomic_numbers[s]]) for s in set(syms)}
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                threshold = (cov_r[syms[i]] + cov_r[syms[j]]) * 1.2
                if dist_matrix[i, j] < threshold:
                    bond_pairs.append({
                        "atoms": [i, j],
                        "symbols": [syms[i], syms[j]],
                        "distance_A": round(float(dist_matrix[i, j]), 4),
                    })

        return {
            "name": name,
            "formula": mol.get_chemical_formula(),
            "n_atoms": len(mol),
            "symbols": syms,
            "positions_A": pos.tolist(),
            "bonds": bond_pairs,
            "center_of_mass": mol.get_center_of_mass().tolist(),
        }

    def plot_crystal_structure(self, symbol: str, crystalstructure: str = "fcc",
                               a: float | None = None, repeat: int = 2) -> str:
        """Plot a 2×2×2 supercell of a bulk crystal. Returns base64 PNG."""
        _require_ase()
        kwargs = dict(name=symbol, crystalstructure=crystalstructure)
        if a is not None:
            kwargs["a"] = a
        atoms = bulk(**kwargs) * (repeat, repeat, 1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, rot in zip(axes, ["0x,0y,0z", "90x,0y,0z"]):
            plot_atoms(atoms, ax, rotation=rot, show_unit_cell=2)
            ax.set_title(f"{symbol} {crystalstructure.upper()} — {'Top' if '0x' in rot else 'Side'} view")
            ax.set_xlabel("x (Å)")
            ax.set_ylabel("y (Å)")

        fig.suptitle(
            f"{symbol} {crystalstructure.upper()} crystal (a={atoms.cell.lengths()[0]:.3f} Å)",
            fontsize=13,
        )
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_molecule(self, name: str) -> str:
        """Plot a molecule from ASE's library. Returns base64 PNG."""
        _require_ase()
        mol = molecule(name)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, rot in zip(axes, ["0x,0y,0z", "90x,90y,0z"]):
            plot_atoms(mol, ax, rotation=rot)
            ax.set_title(f"{name} — {'Front' if '0x,0y' in rot else 'Side'} view")

        fig.suptitle(f"Molecule: {name}  ({mol.get_chemical_formula()})", fontsize=13)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def compare_crystal_structures(self, symbol: str) -> dict:
        """Compare FCC, BCC, and HCP structures for the same element."""
        _require_ase()
        results = {}
        for struct in ["fcc", "bcc", "hcp", "sc"]:
            try:
                data = self.build_bulk_crystal(symbol, struct)
                results[struct] = {
                    "lattice_constant_A": data["lattice_constant_A"],
                    "cell_volume_A3": data["cell_volume_A3"],
                    "n_atoms_unit_cell": data["n_atoms"],
                    "density_g_cm3": data["density_g_cm3"],
                }
            except Exception as exc:
                results[struct] = {"error": str(exc)}
        return {"symbol": symbol, "structures": results}

    def surface_slab(self, symbol: str, miller: tuple[int, int, int] = (1, 0, 0),
                     layers: int = 4, vacuum: float = 10.0) -> dict:
        """Create a surface slab for a given Miller index."""
        _require_ase()
        slab = surface(symbol, miller, layers, vacuum=vacuum)
        return {
            "formula": slab.get_chemical_formula(),
            "miller_index": list(miller),
            "n_atoms": len(slab),
            "layers": layers,
            "vacuum_A": vacuum,
            "cell": slab.get_cell().tolist(),
            "positions_A": slab.get_positions().tolist(),
        }

    def get_element_ase_data(self, symbol: str) -> dict:
        """Return ASE database values for an element."""
        _require_ase()
        Z = atomic_numbers.get(symbol)
        if Z is None:
            raise ValueError(f"Unknown element symbol: {symbol}")
        return {
            "symbol": symbol,
            "Z": Z,
            "atomic_mass_u": round(float(atomic_masses[Z]), 4),
            "covalent_radius_A": round(float(covalent_radii[Z]), 4),
        }
