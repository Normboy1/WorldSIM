"""Hydrogen-like orbital wave functions and probability density plots.

Uses exact quantum-mechanical solutions: ψ_{nlm}(r,θ,φ).
All plots return base64 PNG strings.
"""

from __future__ import annotations

import base64
import io
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y, genlaguerre


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _validate_quantum_numbers(n: int, l: int | None = None, m: int | None = None, Z: int = 1) -> None:
    if int(n) < 1:
        raise ValueError("n must be a positive integer.")
    if int(Z) < 1:
        raise ValueError("Z must be a positive integer.")
    if l is not None:
        if int(l) < 0 or int(l) >= int(n):
            raise ValueError(f"l must satisfy 0 <= l < n (got n={n}, l={l}).")
    if m is not None and l is not None and abs(int(m)) > int(l):
        raise ValueError(f"|m| must be <= l (got m={m}, l={l}).")


def _radial_wavefunction(n: int, l: int, r: np.ndarray, Z: int = 1) -> np.ndarray:
    """Compute the radial part R_{nl}(r) in units of a_0."""
    _validate_quantum_numbers(n, l, Z=Z)
    a0 = 1.0  # Bohr radius units
    rho = 2 * Z * r / (n * a0)
    norm_sq = (2 * Z / (n * a0)) ** 3 * factorial(n - l - 1) / (
        2 * n * factorial(n + l)
    )
    # Guard for numerical issues
    norm = np.sqrt(max(norm_sq, 0.0))
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return norm * np.exp(-rho / 2) * rho ** l * L


class HydrogenOrbitalEngine:
    """Hydrogen-like atomic orbital visualizer."""

    _ORBITAL_NAMES = {0: "s", 1: "p", 2: "d", 3: "f"}

    def plot_radial_probability(
        self, n: int, l: int, Z: int = 1, r_max: float | None = None
    ) -> str:
        """Plot the radial probability density |R(r)|² × r² for ψ_{n,l}."""
        _validate_quantum_numbers(n, l, Z=Z)
        if r_max is None:
            r_max = float(4 * n ** 2 + 10)
        r = np.linspace(0.01, r_max, 800)
        R = _radial_wavefunction(n, l, r, Z)
        prob = r ** 2 * R ** 2

        label = f"n={n}, l={l} ({self._ORBITAL_NAMES.get(l,'?')})"
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(r, prob, linewidth=2.2, color="#2196F3")
        ax.fill_between(r, prob, alpha=0.18, color="#2196F3")
        ax.set_xlabel("r (Bohr radii a₀)", fontsize=12)
        ax.set_ylabel("|R(r)|² · r²", fontsize=12)
        ax.set_title(f"Radial Probability Density — Hydrogen {label} (Z={Z})", fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_orbital_2d(
        self, n: int, l: int, m: int = 0, Z: int = 1, grid_pts: int = 300
    ) -> str:
        """Plot the 2D cross-section (xz-plane) probability density |ψ|² in colour."""
        _validate_quantum_numbers(n, l, m, Z=Z)

        r_max = float(5 * n ** 2 + 5)
        lin = np.linspace(-r_max, r_max, grid_pts)
        X, Z_grid = np.meshgrid(lin, lin)
        R_grid = np.sqrt(X ** 2 + Z_grid ** 2) + 1e-12
        theta = np.arccos(np.clip(Z_grid / R_grid, -1, 1))
        phi = np.zeros_like(X)  # xz-plane: phi=0

        R_vals = _radial_wavefunction(n, l, R_grid.ravel(), Z).reshape(grid_pts, grid_pts)
        Y_vals = sph_harm_y(l, m, theta, phi).real
        psi = R_vals * Y_vals
        density = psi ** 2

        fig, ax = plt.subplots(figsize=(7, 7))
        img = ax.imshow(
            density,
            extent=[-r_max, r_max, -r_max, r_max],
            origin="lower",
            cmap="inferno",
            aspect="equal",
        )
        plt.colorbar(img, ax=ax, label="|ψ|² (arb. units)")
        name = f"{n}{self._ORBITAL_NAMES.get(l,'?')}" + (f"_m{m}" if m != 0 else "")
        ax.set_title(f"Orbital Probability Density — {name} (Z={Z})", fontsize=13)
        ax.set_xlabel("x (a₀)", fontsize=11)
        ax.set_ylabel("z (a₀)", fontsize=11)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_radial_comparison(
        self, n: int, Z: int = 1
    ) -> str:
        """Plot all l sub-shells for a given n on one axes."""
        _validate_quantum_numbers(n, Z=Z)
        r_max = float(4 * n ** 2 + 10)
        r = np.linspace(0.01, r_max, 800)

        colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for l in range(n):
            R = _radial_wavefunction(n, l, r, Z)
            prob = r ** 2 * R ** 2
            label = f"l={l} ({self._ORBITAL_NAMES.get(l,'?')})"
            ax.plot(r, prob, linewidth=2, color=colors[l % len(colors)], label=label)

        ax.set_xlabel("r (a₀)", fontsize=12)
        ax.set_ylabel("|R(r)|² · r²", fontsize=12)
        ax.set_title(f"Radial Probability — n={n} sub-shells (Z={Z})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def energy_levels(self, Z: int = 1, n_max: int = 6) -> dict:
        """Return energy levels E_n = -13.6 * Z² / n² eV."""
        _validate_quantum_numbers(1, Z=Z)
        if int(n_max) < 1:
            raise ValueError("n_max must be a positive integer.")
        levels = {}
        for n in range(1, n_max + 1):
            E = -13.6 * Z ** 2 / n ** 2
            levels[n] = {"energy_eV": round(E, 4), "degeneracy": n ** 2}
        return {"Z": Z, "levels": levels}

    def radial_probability_integral(
        self,
        n: int,
        l: int,
        Z: int = 1,
        r_max: float | None = None,
        n_points: int = 5000,
    ) -> dict:
        """Numerically integrate |R_nl(r)|^2 r^2 dr as a normalization check."""
        _validate_quantum_numbers(n, l, Z=Z)
        if int(n_points) < 100:
            raise ValueError("n_points must be at least 100.")
        if r_max is None:
            r_max = float(20 * n ** 2 / Z + 20)
        r = np.linspace(0.0, r_max, int(n_points))
        R = _radial_wavefunction(n, l, r, Z)
        probability_density = r ** 2 * R ** 2
        integral = float(np.trapezoid(probability_density, r))
        return {
            "n": n,
            "l": l,
            "Z": Z,
            "r_max": r_max,
            "integral": integral,
            "normalization_error": abs(1.0 - integral),
        }

    def plot_energy_levels(self, Z: int = 1, n_max: int = 6) -> str:
        """Plot hydrogen-like energy level diagram. Returns base64 PNG."""
        data = self.energy_levels(Z, n_max)
        levels = data["levels"]

        fig, ax = plt.subplots(figsize=(8, 7))
        colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

        for n, info in levels.items():
            E = info["energy_eV"]
            color = colors[(n - 1) % len(colors)]
            ax.hlines(E, 0.2, 0.8, colors=color, linewidth=2.5)
            ax.text(0.82, E, f"n={n}  E={E:.2f} eV", va="center", fontsize=10, color=color)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(0.82, 0.3, "Ionization", fontsize=9, color="gray")
        ax.set_xlim(0, 1.5)
        ax.set_ylim(min(info["energy_eV"] for info in levels.values()) - 2, 3)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel("Energy (eV)", fontsize=12)
        ax.set_title(f"Hydrogen-like Energy Levels (Z={Z})", fontsize=13)
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        return _fig_to_b64(fig)
