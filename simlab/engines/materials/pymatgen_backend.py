"""Pymatgen materials science backend.

Wraps pymatgen for crystal structure analysis, phase diagrams,
electronic structure properties, and materials informatics.
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np

try:
    from pymatgen.core import Structure, Lattice, Element, Composition
    from pymatgen.core.periodic_table import Element as PmgElement
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter
    from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
    _PMG_OK = True
except ImportError:
    _PMG_OK = False


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _require_pmg() -> None:
    if not _PMG_OK:
        raise ImportError("pymatgen not installed. Run: pip install pymatgen")


class PymatgenEngine:
    """Materials analysis using pymatgen."""

    def element_properties(self, symbol: str) -> dict:
        """Return comprehensive pymatgen element properties."""
        _require_pmg()
        el = PmgElement(symbol)
        props = {
            "symbol": el.symbol,
            "name": el.long_name,
            "Z": el.Z,
            "atomic_mass_u": float(el.atomic_mass),
            "electronic_structure": el.electronic_structure,
            "row": el.row,
            "group": el.group,
            "block": el.block,
            "is_metal": el.is_metal,
            "is_metalloid": el.is_metalloid,
            "is_transition_metal": el.is_transition_metal,
            "is_rare_earth_metal": el.is_rare_earth_metal,
            "is_noble_gas": el.is_noble_gas,
            "is_halogen": el.is_halogen,
            "is_actinoid": el.is_actinoid,
            "is_lanthanoid": el.is_lanthanoid,
            "oxidation_states": list(el.common_oxidation_states),
        }
        # Optional props that may not exist for all elements
        for attr, key in [
            ("atomic_radius", "atomic_radius_A"),
            ("electronegativity", "electronegativity_pauling"),
            ("ionization_energy", "ionization_energy_eV"),
            ("electron_affinity", "electron_affinity_eV"),
            ("melting_point", "melting_point_K"),
            ("boiling_point", "boiling_point_K"),
        ]:
            try:
                val = getattr(el, attr)
                props[key] = float(val) if val is not None else None
            except Exception:
                props[key] = None
        return props

    def compare_elements(self, symbols: list[str]) -> dict:
        """Compare properties of multiple elements side by side."""
        _require_pmg()
        return {sym: self.element_properties(sym) for sym in symbols}

    def build_structure(self, lattice_params: dict, species: list[str],
                        coords: list[list[float]], coords_are_cartesian: bool = False) -> dict:
        """Build a crystal structure from lattice parameters and atomic positions.

        Parameters
        ----------
        lattice_params : dict with keys a, b, c (Å) and alpha, beta, gamma (deg)
        species        : list of element symbols, one per atom
        coords         : list of [x, y, z] coordinates
        coords_are_cartesian : True for Cartesian, False for fractional (default)
        """
        _require_pmg()
        lat = Lattice.from_parameters(
            a=lattice_params.get("a", 4.0),
            b=lattice_params.get("b", 4.0),
            c=lattice_params.get("c", 4.0),
            alpha=lattice_params.get("alpha", 90),
            beta=lattice_params.get("beta", 90),
            gamma=lattice_params.get("gamma", 90),
        )
        struct = Structure(lat, species, coords,
                           coords_are_cartesian=coords_are_cartesian)
        return self._structure_to_dict(struct)

    def analyze_common_structure(self, formula: str) -> dict:
        """Build and analyse a common binary compound structure.

        Supports formulas: NaCl, CsCl, ZnS (zincblende), TiO2 (rutile), CaF2 (fluorite).
        """
        _require_pmg()
        structs = {
            "NaCl":  (Lattice.cubic(5.64), ["Na", "Cl"],
                      [[0, 0, 0], [0.5, 0.5, 0.5]]),
            "CsCl":  (Lattice.cubic(4.12), ["Cs", "Cl"],
                      [[0, 0, 0], [0.5, 0.5, 0.5]]),
            "ZnS":   (Lattice.cubic(5.42), ["Zn", "S"],
                      [[0, 0, 0], [0.25, 0.25, 0.25]]),
            "CaF2":  (Lattice.cubic(5.46), ["Ca", "F", "F"],
                      [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]),
        }
        if formula not in structs:
            return {"error": f"Unknown formula '{formula}'. Supported: {list(structs)}"}

        lat, sps, coords = structs[formula]
        struct = Structure(lat, sps, coords)
        return self._structure_to_dict(struct)

    def _structure_to_dict(self, struct: "Structure") -> dict:
        """Convert a pymatgen Structure to a serializable dict."""
        sga = SpacegroupAnalyzer(struct)
        return {
            "formula": struct.formula,
            "reduced_formula": struct.composition.reduced_formula,
            "n_atoms": len(struct),
            "volume_A3": round(struct.volume, 4),
            "density_g_cm3": round(struct.density, 4),
            "lattice": {
                "a": round(struct.lattice.a, 4),
                "b": round(struct.lattice.b, 4),
                "c": round(struct.lattice.c, 4),
                "alpha": round(struct.lattice.alpha, 3),
                "beta": round(struct.lattice.beta, 3),
                "gamma": round(struct.lattice.gamma, 3),
            },
            "space_group": sga.get_space_group_symbol(),
            "space_group_number": sga.get_space_group_number(),
            "crystal_system": sga.get_crystal_system(),
            "point_group": sga.get_point_group_symbol(),
            "species": list(struct.species_string),
            "fractional_coords": [s.frac_coords.tolist() for s in struct],
        }

    def plot_phase_diagram_binary(self, el1: str, el2: str,
                                  entries: list[dict] | None = None) -> str:
        """Plot a simple binary phase diagram using provided formation energies.

        entries = list of dicts: [{"formula": "FeO", "energy_per_atom": -0.5}, ...]
        If entries is None, uses example dummy data to show the diagram structure.
        """
        _require_pmg()

        if entries is None:
            # Example Fe-O system dummy data
            el1, el2 = "Fe", "O"
            entries = [
                {"formula": el1,        "energy_per_atom": 0.0},
                {"formula": el2,        "energy_per_atom": 0.0},
                {"formula": "FeO",      "energy_per_atom": -1.40},
                {"formula": "Fe2O3",    "energy_per_atom": -1.60},
                {"formula": "Fe3O4",    "energy_per_atom": -1.55},
            ]

        pd_entries = []
        for e in entries:
            comp = Composition(e["formula"])
            pd_entries.append(PDEntry(comp, e["energy_per_atom"] * comp.num_atoms))

        try:
            pd = PhaseDiagram(pd_entries)
        except Exception as exc:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot convex hull manually for binary system
        stable = list(pd.stable_entries)
        unstable = [e for e in pd_entries if e not in stable]

        def x_frac(entry: "PDEntry") -> float:
            comp = entry.composition
            total = sum(comp.values())
            return comp.get(el2, 0) / total if total else 0

        def e_hull(entry: "PDEntry") -> float:
            return pd.get_e_above_hull(entry)

        def e_form(entry: "PDEntry") -> float:
            return pd.get_form_energy_per_atom(entry)

        xs = [x_frac(e) for e in stable]
        ys = [e_form(e) for e in stable]
        order = sorted(zip(xs, ys, [e.composition.reduced_formula for e in stable]))
        xs_s, ys_s, labels = zip(*order) if order else ([], [], [])

        ax.plot(xs_s, ys_s, "bo-", linewidth=2, markersize=8, label="Stable phases")
        for x, y, lab in zip(xs_s, ys_s, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color="#1565C0")

        for e in unstable:
            ax.plot(x_frac(e), e_form(e), "r^", markersize=7, alpha=0.7)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(f"Fraction of {el2}", fontsize=12)
        ax.set_ylabel("Formation energy per atom (eV)", fontsize=12)
        ax.set_title(f"Binary Phase Diagram: {el1}–{el2}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_element_property_trends(
        self, property_name: str = "electronegativity", z_range: tuple[int, int] = (1, 54)
    ) -> str:
        """Plot a periodic trend across elements. Returns base64 PNG.

        property_name options: 'electronegativity', 'atomic_radius', 'ionization_energy',
                               'electron_affinity', 'melting_point', 'boiling_point'
        """
        _require_pmg()
        attr_map = {
            "electronegativity": ("electronegativity", "Pauling electronegativity"),
            "atomic_radius":     ("atomic_radius", "Atomic radius (Å)"),
            "ionization_energy": ("ionization_energy", "First ionization energy (eV)"),
            "electron_affinity": ("electron_affinity", "Electron affinity (eV)"),
            "melting_point":     ("melting_point", "Melting point (K)"),
            "boiling_point":     ("boiling_point", "Boiling point (K)"),
        }
        if property_name not in attr_map:
            property_name = "electronegativity"

        attr, ylabel = attr_map[property_name]
        zs, vals, syms = [], [], []
        for Z in range(z_range[0], z_range[1] + 1):
            try:
                el = PmgElement.from_Z(Z)
                v = getattr(el, attr)
                if v is not None:
                    zs.append(Z)
                    vals.append(float(v))
                    syms.append(el.symbol)
            except Exception:
                continue

        # Highlight noble gases and alkali metals
        noble = {2, 10, 18, 36, 54}
        alkali = {1, 3, 11, 19, 37, 55}

        fig, ax = plt.subplots(figsize=(13, 5))
        colors = ["#9C27B0" if z in noble else "#F44336" if z in alkali else "#2196F3"
                  for z in zs]
        ax.scatter(zs, vals, c=colors, s=50, zorder=3)
        ax.plot(zs, vals, color="#90CAF9", linewidth=1.2, alpha=0.6, zorder=2)

        # Label notable elements
        for z, v, sym in zip(zs, vals, syms):
            if z in noble or z in alkali or v == max(vals) or v == min(vals):
                ax.annotate(sym, (z, v), textcoords="offset points",
                            xytext=(0, 6), fontsize=7, ha="center")

        from matplotlib.patches import Patch
        legend = [Patch(color="#9C27B0", label="Noble gas"),
                  Patch(color="#F44336", label="Alkali metal"),
                  Patch(color="#2196F3", label="Other")]
        ax.legend(handles=legend, fontsize=9, loc="upper right")
        ax.set_xlabel("Atomic Number (Z)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Periodic Trend: {ylabel.split('(')[0].strip()}", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def oxidation_state_analysis(self, formula: str) -> dict:
        """Predict oxidation states and check charge balance."""
        _require_pmg()
        comp = Composition(formula)
        try:
            oxi = comp.oxi_state_guesses()
            best = oxi[0] if oxi else {}
        except Exception:
            best = {}

        return {
            "formula": formula,
            "reduced_formula": comp.reduced_formula,
            "elements": list(comp.as_dict().keys()),
            "predicted_oxidation_states": best,
            "is_charge_balanced": comp.charge_balanced if hasattr(comp, "charge_balanced") else None,
        }
