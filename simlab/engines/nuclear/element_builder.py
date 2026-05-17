"""Element Builder — the core "create an element" simulation.

Given proton count Z (and optionally neutron count N), constructs a full
profile of the resulting element including nuclear properties, atomic
structure, electron configuration, stability, and visualizations.
"""

from __future__ import annotations

from simlab.engines.atomic.element_data import get_element
from simlab.engines.atomic.electron_config import ElectronConfigEngine
from simlab.engines.nuclear.nuclear_engine import NuclearEngine, semf_binding_energy, semf_binding_energy_per_nucleon

import base64
import io

import matplotlib.pyplot as plt
import numpy as np


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _most_stable_N(Z: int) -> int:
    """Find N that maximises B/A for a given Z (valley-of-stability)."""
    best_N, best_ba = 1, 0.0
    A_est = int(Z * 2.0 + Z * 0.015 * (2 * Z) ** (2 / 3))
    for N in range(max(1, Z - 20), A_est + 20):
        ba = semf_binding_energy_per_nucleon(Z, N)
        if ba > best_ba:
            best_ba = ba
            best_N = N
    return best_N


class ElementBuilder:
    """Build a complete element profile from protons (and optional neutrons)."""

    def __init__(self) -> None:
        self._ec = ElectronConfigEngine()
        self._ne = NuclearEngine()

    def create_element(
        self,
        Z: int,
        N: int | None = None,
        include_plots: bool = True,
    ) -> dict:
        """Simulate and return a complete element profile.

        Parameters
        ----------
        Z : atomic number (number of protons)
        N : number of neutrons.  If None, the most stable isotope is chosen.
        include_plots : generate base64 PNG plots for all sub-systems.

        Returns a dict with sections:
          identity, nuclear, atomic, electron_config, visualizations, summary
        """
        if Z < 1 or Z > 118:
            raise ValueError(f"Z={Z} is out of range 1–118.")

        if N is None:
            N = _most_stable_N(Z)

        A = Z + N
        el = get_element(Z)

        # ── Identity ──────────────────────────────────────────────────────────
        identity = {
            "Z": Z,
            "N": N,
            "A": A,
            "symbol": el["symbol"] if el else f"Z{Z}",
            "name": el["name"] if el else f"Synthetic element Z={Z}",
            "isotope_notation": f"¹{A}{el['symbol'] if el else '?'}" if A < 10 else f"{A}{el['symbol'] if el else '?'}",
            "category": el.get("category") if el else "unknown",
            "period": el.get("period") if el else None,
            "group": el.get("group") if el else None,
            "block": el.get("block") if el else None,
            "standard_atomic_weight_u": el.get("mass") if el else None,
            "electronegativity": el.get("electronegativity") if el else None,
        }

        # ── Nuclear ───────────────────────────────────────────────────────────
        nuclear = self._ne.analyze_nucleus(Z, N)

        # Find the most stable isotope for comparison
        N_best = _most_stable_N(Z)
        A_best = Z + N_best
        B_best = semf_binding_energy(Z, N_best)

        nuclear["most_stable_isotope"] = {
            "N": N_best,
            "A": A_best,
            "binding_energy_MeV": round(B_best, 3),
            "binding_per_nucleon_MeV": round(B_best / A_best if A_best else 0, 4),
        }

        # ── Atomic properties ─────────────────────────────────────────────────
        atomic = {
            "atomic_radius_pm": el.get("atomic_radius_pm") if el else None,
            "first_ionization_eV": el.get("ionization_energy_eV") if el else None,
            "melting_point_K": el.get("melting_point_K") if el else None,
            "boiling_point_K": el.get("boiling_point_K") if el else None,
            "density_g_cm3": el.get("density_g_cm3") if el else None,
        }

        # ── Electron configuration ────────────────────────────────────────────
        ec = self._ec.compute_configuration(Z)
        electron_config = {
            "config_str": ec["config_str"],
            "config_noble": ec["config_noble"],
            "shells": ec["shells"],
            "valence_electrons": ec["valence_electrons"],
            "is_exception": ec["is_exception"],
        }

        # ── Spectral lines (Balmer series) ────────────────────────────────────
        def balmer_wavelength_nm(n: int) -> float:
            R_H = 1.097e7  # m⁻¹
            lam = 1 / (R_H * (1 / 4 - 1 / n ** 2))
            return round(lam * 1e9, 2)

        spectral = {
            "balmer_series_nm": {
                f"n={n}→2": balmer_wavelength_nm(n) for n in range(3, 8)
            },
            "note": "Balmer series shown (H-like approximation); actual spectrum depends on Z.",
        }

        # ── Visualizations ────────────────────────────────────────────────────
        plots: dict[str, str] = {}
        if include_plots:
            try:
                plots["shell_diagram"] = self._ec.plot_shell_diagram(Z)
            except Exception as exc:
                plots["shell_diagram_error"] = str(exc)

            try:
                plots["orbital_diagram"] = self._ec.plot_orbital_diagram(Z)
            except Exception as exc:
                plots["orbital_diagram_error"] = str(exc)

            try:
                plots["binding_energy_curve"] = self._ne.binding_energy_curve(z_fixed=Z)
            except Exception as exc:
                plots["binding_energy_curve_error"] = str(exc)

            try:
                plots["isotope_stability"] = self._plot_isotope_stability(Z)
            except Exception as exc:
                plots["isotope_stability_error"] = str(exc)

        # ── Summary ───────────────────────────────────────────────────────────
        summary_lines = [
            f"Element: {identity['name']} ({identity['symbol']}), Z={Z}",
            f"Isotope: {identity['isotope_notation']} (Z={Z}, N={N}, A={A})",
            f"Category: {identity['category']}  |  Period {identity['period']}, Group {identity['group'] or 'f-block'}",
            f"Electron config: {ec['config_noble']}",
            f"Valence electrons: {ec['valence_electrons']}",
            f"Nuclear binding energy: {nuclear['binding_energy_MeV']:.2f} MeV ({nuclear['binding_per_nucleon_MeV']:.3f} MeV/nucleon)",
            f"Stability prediction: {nuclear['stability_prediction']}",
            f"Most stable isotope: A={A_best} (N={N_best})",
        ]
        if el:
            if el.get("melting_point_K"):
                summary_lines.append(f"Melting point: {el['melting_point_K']} K")
            if el.get("ionization_energy_eV"):
                summary_lines.append(f"1st ionization energy: {el['ionization_energy_eV']} eV")

        return {
            "identity": identity,
            "nuclear": nuclear,
            "atomic": atomic,
            "electron_config": electron_config,
            "spectral": spectral,
            "plots": plots,
            "summary": "\n".join(summary_lines),
        }

    def _plot_isotope_stability(self, Z: int) -> str:
        """Plot B/A vs N for all isotopes of element Z. Returns base64 PNG."""
        el = get_element(Z)
        sym = el["symbol"] if el else f"Z={Z}"

        N_arr = list(range(1, min(3 * Z + 20, 200)))
        BA_arr = [semf_binding_energy_per_nucleon(Z, N) for N in N_arr]

        N_arr_filt = [n for n, ba in zip(N_arr, BA_arr) if ba > 0]
        BA_filt = [ba for ba in BA_arr if ba > 0]

        best_N = N_arr_filt[int(np.argmax(BA_filt))] if BA_filt else 1
        best_ba = max(BA_filt) if BA_filt else 0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(N_arr_filt, BA_filt, color="#2196F3", linewidth=2.2)
        ax.fill_between(N_arr_filt, BA_filt, alpha=0.15, color="#2196F3")
        ax.axvline(best_N, color="orange", linestyle="--", linewidth=1.5,
                   label=f"Most stable: N={best_N} (A={Z+best_N})")
        ax.scatter([best_N], [best_ba], color="orange", s=80, zorder=5)
        ax.set_xlabel("Neutron Number N", fontsize=12)
        ax.set_ylabel("Binding Energy per Nucleon B/A (MeV)", fontsize=12)
        ax.set_title(f"Isotope Stability — {el['name'] if el else sym} (Z={Z})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def fusion_to_element(self, Z1: int, Z2: int, N1: int | None = None, N2: int | None = None) -> dict:
        """Simulate fusion of two elements to form a product element."""
        if N1 is None:
            N1 = _most_stable_N(Z1)
        if N2 is None:
            N2 = _most_stable_N(Z2)

        Z_out = Z1 + Z2
        N_out = N1 + N2
        nuclear_result = self._ne.fusion_energy(Z1, N1, Z2, N2)

        if 1 <= Z_out <= 118:
            product = self.create_element(Z_out, N_out, include_plots=False)
        else:
            product = {"identity": {"Z": Z_out, "N": N_out, "A": Z_out + N_out,
                                    "name": "Beyond element 118 — theoretical",
                                    "symbol": "?"}}

        return {
            "reaction": nuclear_result,
            "product_element": product,
        }

    def compare_isotopes(self, Z: int, N_list: list[int]) -> dict:
        """Compare multiple isotopes of the same element."""
        results = {}
        for N in N_list:
            nuc = self._ne.analyze_nucleus(Z, N)
            results[f"A={Z+N}"] = nuc
        return {"Z": Z, "isotopes": results}
