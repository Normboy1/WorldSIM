"""Radioactive decay simulation engine.

Supports alpha, beta-minus, beta-plus decay chains and plotting.
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from simlab.engines.atomic.element_data import get_element
from simlab.engines.nuclear.nuclear_engine import semf_binding_energy


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _decay_constant(half_life_s: float) -> float:
    return np.log(2) / half_life_s


class DecayEngine:
    """Radioactive decay simulations."""

    def simulate_decay(
        self,
        N0: float,
        half_life_s: float,
        t_span: float | None = None,
        t_points: int = 300,
    ) -> dict:
        """Simulate simple exponential radioactive decay N(t) = N0 · e^{-λt}.

        Parameters
        ----------
        N0          : initial number of atoms (or activity in Bq)
        half_life_s : half-life in seconds
        t_span      : total simulation time in seconds (defaults to 5 half-lives)
        t_points    : number of time points

        Returns dict with t, N, activity arrays plus key values.
        """
        lam = _decay_constant(half_life_s)
        if t_span is None:
            t_span = 5 * half_life_s

        t = np.linspace(0, t_span, t_points)
        N = N0 * np.exp(-lam * t)
        activity = lam * N  # decays per second (Bq)

        return {
            "t_s": t.tolist(),
            "N": N.tolist(),
            "activity_Bq": activity.tolist(),
            "half_life_s": half_life_s,
            "decay_constant_per_s": round(lam, 8),
            "N0": N0,
            "N_at_t_end": round(N[-1], 4),
            "fraction_remaining": round(N[-1] / N0, 6),
            "mean_lifetime_s": round(1 / lam, 4),
        }

    def simulate_decay_chain(
        self,
        chain: list[dict],
        N0_first: float = 1e6,
        t_span: float | None = None,
        t_points: int = 500,
    ) -> dict:
        """Simulate a Bateman decay chain A → B → C → … (stable).

        Parameters
        ----------
        chain : list of dicts, each with 'symbol' and 'half_life_s'.
                Last entry can have half_life_s=None to mark stable.
        N0_first : initial count of first isotope
        t_span   : total time (default: 10× first half-life)

        Example chain for U-238 simplified:
          [{"symbol": "U-238", "half_life_s": 1.41e17},
           {"symbol": "Th-234", "half_life_s": 2.08e6},
           {"symbol": "stable",  "half_life_s": None}]
        """
        n_species = len(chain)
        lambdas = []
        for entry in chain:
            hl = entry.get("half_life_s")
            if hl is None or hl == 0:
                lambdas.append(0.0)
            else:
                lambdas.append(_decay_constant(hl))

        if t_span is None:
            first_lam = next((l for l in lambdas if l > 0), 1e-10)
            t_span = 10 / first_lam

        t_eval = np.linspace(0, t_span, t_points)
        y0 = [0.0] * n_species
        y0[0] = N0_first

        def bateman(t, y):
            dydt = []
            for i in range(n_species):
                lam_i = lambdas[i]
                production = lambdas[i - 1] * y[i - 1] if i > 0 else 0.0
                loss = lam_i * y[i]
                dydt.append(production - loss)
            return dydt

        sol = solve_ivp(bateman, [0, t_span], y0, t_eval=t_eval, method="LSODA")
        t_out = sol.t.tolist()
        populations = {
            chain[i]["symbol"]: sol.y[i].tolist() for i in range(n_species)
        }

        return {
            "t_s": t_out,
            "populations": populations,
            "chain": [c["symbol"] for c in chain],
            "lambdas": lambdas,
        }

    def plot_decay(self, N0: float, half_life_s: float, t_span: float | None = None, label: str = "Isotope") -> str:
        """Plot decay curve N(t). Returns base64 PNG."""
        result = self.simulate_decay(N0, half_life_s, t_span)
        t = np.array(result["t_s"])
        N = np.array(result["N"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(t, N, color="#E91E63", linewidth=2.2)
        axes[0].fill_between(t, N, alpha=0.15, color="#E91E63")
        axes[0].axhline(N0 / 2, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        axes[0].text(t[-1] * 0.05, N0 / 2 * 1.03, "N₀/2", fontsize=9, color="gray")
        axes[0].set_xlabel("Time (s)", fontsize=11)
        axes[0].set_ylabel("Number of atoms N(t)", fontsize=11)
        axes[0].set_title(f"{label} — Decay Curve", fontsize=12)
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(t, N, color="#2196F3", linewidth=2.2)
        axes[1].set_xlabel("Time (s)", fontsize=11)
        axes[1].set_ylabel("log N(t)", fontsize=11)
        axes[1].set_title(f"{label} — Log Scale", fontsize=12)
        axes[1].grid(True, alpha=0.3, which="both")

        fig.suptitle(f"Radioactive Decay — {label}  (t½ = {half_life_s:.3g} s)", fontsize=13)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_decay_chain(
        self, chain: list[dict], N0_first: float = 1e6, t_span: float | None = None
    ) -> str:
        """Plot population curves for a Bateman decay chain. Returns base64 PNG."""
        result = self.simulate_decay_chain(chain, N0_first, t_span)
        t = np.array(result["t_s"])

        colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (sym, pop) in enumerate(result["populations"].items()):
            ax.plot(t, pop, color=colors[i % len(colors)], linewidth=2.2, label=sym)

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Number of atoms", fontsize=12)
        ax.set_title("Decay Chain Population Dynamics", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def alpha_decay_product(self, Z: int, N: int) -> dict:
        """Return alpha-decay daughter nucleus: (Z,N) → (Z-2, N-2) + ⁴He."""
        Z_d, N_d = Z - 2, N - 2
        el = get_element(Z)
        el_d = get_element(Z_d)
        Q = (
            semf_binding_energy(Z_d, N_d)
            + semf_binding_energy(2, 2)
            - semf_binding_energy(Z, N)
        )
        return {
            "parent": f"{Z+N}{el['symbol'] if el else '?'}",
            "daughter": f"{Z_d+N_d}{el_d['symbol'] if el_d else '?'}",
            "alpha_particle": "⁴He (α)",
            "Q_value_MeV": round(Q, 4),
            "decay_possible": Q > 0,
        }

    def beta_minus_product(self, Z: int, N: int) -> dict:
        """Return β⁻ decay daughter: (Z,N) → (Z+1, N-1) + e⁻ + ν̄_e."""
        Z_d, N_d = Z + 1, N - 1
        el = get_element(Z)
        el_d = get_element(Z_d)
        Q = semf_binding_energy(Z_d, N_d) - semf_binding_energy(Z, N)
        return {
            "parent": f"{Z+N}{el['symbol'] if el else '?'}",
            "daughter": f"{Z_d+N_d}{el_d['symbol'] if el_d else '?'}",
            "particles_emitted": "e⁻ + ν̄_e",
            "Q_value_MeV": round(Q, 4),
            "decay_possible": Q > 0,
        }

    def beta_plus_product(self, Z: int, N: int) -> dict:
        """Return β⁺ decay daughter: (Z,N) → (Z-1, N+1) + e⁺ + ν_e."""
        Z_d, N_d = Z - 1, N + 1
        el = get_element(Z)
        el_d = get_element(Z_d)
        Q = semf_binding_energy(Z_d, N_d) - semf_binding_energy(Z, N) - 2 * 0.511  # 2 × m_e c²
        return {
            "parent": f"{Z+N}{el['symbol'] if el else '?'}",
            "daughter": f"{Z_d+N_d}{el_d['symbol'] if el_d else '?'}",
            "particles_emitted": "e⁺ + ν_e",
            "Q_value_MeV": round(Q, 4),
            "decay_possible": Q > 0,
        }
