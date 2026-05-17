"""Electron configuration engine.

Computes ground-state electron configurations via the Aufbau principle,
orbital energy diagrams, and ionization energy trends.
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .element_data import get_element, ELEMENTS

# Aufbau filling order: (n, l, label, max_electrons)
_AUFBAU: list[tuple[int, int, str, int]] = [
    (1, 0, "1s",  2),
    (2, 0, "2s",  2),
    (2, 1, "2p",  6),
    (3, 0, "3s",  2),
    (3, 1, "3p",  6),
    (4, 0, "4s",  2),
    (3, 2, "3d", 10),
    (4, 1, "4p",  6),
    (5, 0, "5s",  2),
    (4, 2, "4d", 10),
    (5, 1, "5p",  6),
    (6, 0, "6s",  2),
    (4, 3, "4f", 14),
    (5, 2, "5d", 10),
    (6, 1, "6p",  6),
    (7, 0, "7s",  2),
    (5, 3, "5f", 14),
    (6, 2, "6d", 10),
    (7, 1, "7p",  6),
]

_NOBLE_CORES: list[tuple[int, str]] = [
    (86, "[Rn]"), (54, "[Xe]"), (36, "[Kr]"),
    (18, "[Ar]"), (10, "[Ne]"), (2,  "[He]"),
]


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class ElectronConfigEngine:
    """Compute and visualize electron configurations."""

    def compute_configuration(self, Z: int) -> dict:
        """Return the electron configuration for atomic number Z.

        Returns a dict with:
          - Z, symbol, name
          - config_str      : e.g. "1s2 2s2 2p6 3s1"
          - config_noble    : abbreviated form e.g. "[Ne] 3s1"
          - orbitals        : list of (label, electrons) tuples
          - shells          : electrons per shell n=1,2,3,...
          - valence_electrons
          - is_exception    : True if known anomalous filling
        """
        if Z < 1 or Z > 118:
            raise ValueError(f"Atomic number {Z} out of range 1–118.")

        el = get_element(Z)
        orbitals: list[tuple[str, int]] = []
        remaining = Z

        for n, l, label, cap in _AUFBAU:
            if remaining <= 0:
                break
            filled = min(remaining, cap)
            orbitals.append((label, filled))
            remaining -= filled

        config_str = " ".join(f"{lbl}{e}" for lbl, e in orbitals)

        # Noble-gas abbreviation
        noble_prefix = ""
        for z_core, sym in _NOBLE_CORES:
            if Z > z_core:
                core_orbs = self._orbitals_for_Z(z_core)
                core_str = " ".join(f"{l}{e}" for l, e in core_orbs)
                rest = config_str.replace(core_str, "").strip()
                noble_prefix = f"{sym} {rest}".strip()
                break
        if not noble_prefix:
            noble_prefix = config_str

        # Electrons per shell
        shells: dict[int, int] = {}
        for label, elec in orbitals:
            n = int(label[0])
            shells[n] = shells.get(n, 0) + elec

        # Valence = outermost shell, plus any incomplete d for transition metals
        max_n = max(shells)
        valence = shells[max_n]

        # Known exception check (simplified)
        is_exception = False
        if el:
            stored = el.get("electron_config", "")
            computed_noble = noble_prefix
            if stored and stored.replace(" ", "") != computed_noble.replace(" ", ""):
                is_exception = True

        return {
            "Z": Z,
            "symbol": el["symbol"] if el else f"Z={Z}",
            "name": el["name"] if el else f"Element {Z}",
            "config_str": config_str,
            "config_noble": noble_prefix,
            "orbitals": orbitals,
            "shells": dict(sorted(shells.items())),
            "valence_electrons": valence,
            "is_exception": is_exception,
        }

    def _orbitals_for_Z(self, Z: int) -> list[tuple[str, int]]:
        orbs = []
        rem = Z
        for n, l, label, cap in _AUFBAU:
            if rem <= 0:
                break
            filled = min(rem, cap)
            orbs.append((label, filled))
            rem -= filled
        return orbs

    def plot_orbital_diagram(self, Z: int) -> str:
        """Return base64 PNG of an orbital energy-level diagram for element Z."""
        cfg = self.compute_configuration(Z)
        orbs = cfg["orbitals"]

        # Approximate orbital energies (eV) — Slater shielding approximation
        def orbital_energy(n: int, l: int, Z_eff: float) -> float:
            return -13.6 * (Z_eff / n) ** 2

        # Build Zeff using simple Slater rules approximation
        energies: list[tuple[str, float, int]] = []
        cumulative = 0
        for label, elec in orbs:
            n = int(label[0])
            l = {"s": 0, "p": 1, "d": 2, "f": 3}[label[1]]
            # Rough Zeff
            z_eff = max(1.0, Z - cumulative * 0.35)
            cumulative += elec
            energy = orbital_energy(n, l, z_eff)
            energies.append((label, energy, elec))

        # Normalize to a reasonable range for display
        raw_e = np.array([e for _, e, _ in energies])
        e_min, e_max = raw_e.min(), raw_e.max()
        span = e_max - e_min if e_max > e_min else 1.0
        norm_e = (raw_e - e_min) / span

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {"s": "#2196F3", "p": "#4CAF50", "d": "#FF9800", "f": "#9C27B0"}
        x_positions = {"s": 1.0, "p": 2.5, "d": 4.0, "f": 5.5}

        for i, ((label, energy, elec), norm) in enumerate(zip(energies, norm_e)):
            orb_type = label[1]
            x = x_positions.get(orb_type, i)
            color = colors.get(orb_type, "gray")
            y = norm * 8  # scale

            ax.hlines(y, x - 0.3, x + 0.3, colors=color, linewidth=2.5)
            ax.text(x + 0.35, y, f"{label} ({elec}e)", va="center", fontsize=9, color=color)

            # Draw arrows for electrons (up/down spin)
            cap = {"s": 2, "p": 6, "d": 10, "f": 14}[orb_type]
            n_boxes = cap // 2
            for j in range(min(elec, cap)):
                box_x = x - 0.28 + (j % n_boxes) * 0.15
                direction = "↑" if j % 2 == 0 else "↓"
                ax.text(box_x, y + 0.05, direction, fontsize=7, color=color, ha="center")

        ax.set_xlim(0, 7)
        ax.set_ylim(-0.5, 9)
        ax.set_yticks([])
        ax.set_xticks([1.0, 2.5, 4.0, 5.5])
        ax.set_xticklabels(["s", "p", "d", "f"], fontsize=12)
        ax.set_title(
            f"Orbital Diagram — {cfg['name']} (Z={Z})\n{cfg['config_noble']}",
            fontsize=13,
        )
        ax.set_ylabel("Relative Energy (higher = less stable)", fontsize=10)

        # Legend
        patches = [mpatches.Patch(color=c, label=t) for t, c in colors.items()]
        ax.legend(handles=patches, loc="upper right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_shell_diagram(self, Z: int) -> str:
        """Bohr-model shell diagram as concentric rings. Returns base64 PNG."""
        cfg = self.compute_configuration(Z)
        shells = cfg["shells"]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal")

        # Nucleus
        nucleus_radius = 0.18
        nucleus = plt.Circle((0, 0), nucleus_radius, color="#E53935", zorder=5)
        ax.add_patch(nucleus)
        ax.text(0, 0, f"Z={Z}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", zorder=6)

        shell_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
                        "#00BCD4", "#F44336", "#8BC34A"]
        radii = [0.5 + 0.4 * i for i in range(len(shells))]

        for i, (n, elec) in enumerate(sorted(shells.items())):
            r = radii[i]
            color = shell_colors[i % len(shell_colors)]
            ring = plt.Circle((0, 0), r, fill=False, color=color,
                              linewidth=1.8, linestyle="--", alpha=0.6)
            ax.add_patch(ring)
            ax.text(r * 0.707 + 0.05, r * 0.707 + 0.05,
                    f"n={n}: {elec}e", fontsize=9, color=color)

            # Place electron dots evenly around the ring
            angles = np.linspace(0, 2 * np.pi, elec, endpoint=False)
            for angle in angles:
                ex = r * np.cos(angle)
                ey = r * np.sin(angle)
                ax.plot(ex, ey, "o", color=color, markersize=6, zorder=4)

        max_r = radii[-1] + 0.3
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.axis("off")
        ax.set_title(
            f"Bohr Shell Model — {cfg['name']} ({cfg['symbol']}, Z={Z})\n"
            f"Config: {cfg['config_noble']}",
            fontsize=12,
        )
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_ionization_energy_trend(self, z_start: int = 1, z_end: int = 36) -> str:
        """Plot first ionization energy across a range of elements. Returns base64 PNG."""
        zs, ies, syms = [], [], []
        for Z in range(z_start, z_end + 1):
            el = get_element(Z)
            if el and el.get("ionization_energy_eV") is not None:
                zs.append(Z)
                ies.append(el["ionization_energy_eV"])
                syms.append(el["symbol"])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(zs, ies, "o-", color="#2196F3", linewidth=1.8, markersize=5)
        for z, ie, sym in zip(zs, ies, syms):
            if ie > 15 or ie < 5:
                ax.annotate(sym, (z, ie), textcoords="offset points",
                            xytext=(0, 5), fontsize=7, ha="center", color="#E53935")
        ax.set_xlabel("Atomic Number (Z)", fontsize=12)
        ax.set_ylabel("First Ionization Energy (eV)", fontsize=12)
        ax.set_title(
            f"First Ionization Energy — Z={z_start} to Z={z_end}", fontsize=13
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def compare_elements(self, z_list: list[int]) -> dict:
        """Return a comparison dict for multiple elements."""
        return {Z: self.compute_configuration(Z) for Z in z_list}

    def effective_nuclear_charge(self, Z: int) -> dict:
        """Compute effective nuclear charge Z* for each occupied orbital using Slater's rules.

        Slater's rules partition electrons into groups and assign shielding constants:
        - [1s] | [2s,2p] | [3s,3p] | [3d] | [4s,4p] | [4d] | [4f] | [5s,5p] | ...
        - For ns/np: same-group e⁻ shield 0.35; (n-1) shell shields 0.85; lower shells 1.00.
        - For nd/nf: same-group e⁻ shield 0.35; all inner e⁻ shield 1.00.

        Returns Z* for each occupied orbital and the resulting approximate ionization
        energy from the hydrogen-like formula: IE ≈ 13.6 eV × (Z*)² / n².
        """
        if Z < 1 or Z > 54:
            raise ValueError("Slater's rules are calibrated for Z=1–54.")

        config = self.compute_configuration(Z)
        orbitals = config["orbitals"]

        # Build occupancy map: {(n, type): electrons}
        # type: 's' or 'p' → group together; 'd'; 'f'
        occ: dict[tuple[int, str], int] = {}
        for label, elec in orbitals:
            n = int(label[0])
            t = label[1]  # s, p, d, f
            sp_type = "sp" if t in ("s", "p") else t
            occ[(n, sp_type)] = occ.get((n, sp_type), 0) + elec

        results = []
        for label, elec in orbitals:
            if elec == 0:
                continue
            n = int(label[0])
            t = label[1]
            sp_type = "sp" if t in ("s", "p") else t

            sigma = 0.0
            if sp_type == "sp":
                for (n2, t2), e2 in occ.items():
                    if (n2, t2) == (n, "sp"):
                        sigma += (e2 - 1) * 0.35  # same group minus self
                    elif n2 == n - 1:
                        sigma += e2 * 0.85
                    elif n2 < n - 1:
                        sigma += e2 * 1.00
            else:  # d or f
                for (n2, t2), e2 in occ.items():
                    if (n2, t2) == (n, sp_type):
                        sigma += (e2 - 1) * 0.35
                    else:
                        sigma += e2 * 1.00

            z_star = Z - sigma
            ie_ev = 13.6 * z_star ** 2 / n ** 2
            results.append({
                "orbital": label,
                "electrons": elec,
                "n": n,
                "shielding_sigma": round(sigma, 3),
                "Z_star": round(z_star, 3),
                "IE_approx_eV": round(ie_ev, 2),
            })

        el = get_element(Z)
        return {
            "Z": Z,
            "symbol": el["symbol"] if el else "?",
            "name": el["name"] if el else "?",
            "orbitals": results,
            "outermost_Z_star": results[-1]["Z_star"] if results else None,
            "outermost_IE_eV": results[-1]["IE_approx_eV"] if results else None,
            "model": "Slater's rules — Z* values are reliable for trends; IE values from 13.6*(Z*/n)^2 are orbital energies (overestimate first IE by 2–4x for valence electrons; accurate for inner shells)",
            "note": "For accurate first ionization energies use experimental data (e.g. NIST). Slater IE is best interpreted as relative ordering across the periodic table.",
        }

    def slater_ionization_energy(self, Z: int) -> dict:
        """Alias for effective_nuclear_charge — returns Z* and orbital IE estimates for all occupied orbitals."""
        return self.effective_nuclear_charge(Z)
