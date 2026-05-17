"""Nuclear physics engine.

Implements the semi-empirical mass formula (Bethe-Weizsäcker) for binding
energy, nuclear stability analysis, and fusion/fission energy calculations.
All plot methods return base64 PNG strings.
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np

from simlab.engines.atomic.element_data import get_element, ELEMENTS

# ── Bethe-Weizsäcker (SEMF) constants (MeV) ─────────────────────────────────
_aV  = 15.835   # volume
_aS  = 18.330   # surface
_aC  = 0.714    # Coulomb
_aA  = 23.2     # asymmetry
_aP  = 11.2     # pairing

_EMPIRICAL_BINDING_ENERGY_MEV: dict[tuple[int, int], dict[str, float | str]] = {
    # Rounded reference values for common benchmark isotopes.
    # Key is (Z, N); binding energies are total nuclear binding energy in MeV.
    (1, 1): {"isotope": "2H", "binding_energy_MeV": 2.224566, "source": "AME/NIST rounded"},
    (2, 1): {"isotope": "3He", "binding_energy_MeV": 7.718043, "source": "AME/NIST rounded"},
    (1, 2): {"isotope": "3H", "binding_energy_MeV": 8.481798, "source": "AME/NIST rounded"},
    (2, 2): {"isotope": "4He", "binding_energy_MeV": 28.295674, "source": "AME/NIST rounded"},
    (6, 6): {"isotope": "12C", "binding_energy_MeV": 92.161734, "source": "AME/NIST rounded"},
    (7, 7): {"isotope": "14N", "binding_energy_MeV": 104.659, "source": "AME/NIST rounded"},
    (8, 8): {"isotope": "16O", "binding_energy_MeV": 127.619315, "source": "AME/NIST rounded"},
    (20, 20): {"isotope": "40Ca", "binding_energy_MeV": 342.052, "source": "AME/NIST rounded"},
    (26, 30): {"isotope": "56Fe", "binding_energy_MeV": 492.260, "source": "AME/NIST rounded"},
    (82, 126): {"isotope": "208Pb", "binding_energy_MeV": 1636.43, "source": "AME/NIST rounded"},
    (92, 143): {"isotope": "235U", "binding_energy_MeV": 1783.87, "source": "AME/NIST rounded"},
    (92, 146): {"isotope": "238U", "binding_energy_MeV": 1801.69, "source": "AME/NIST rounded"},
}


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def semf_binding_energy(Z: int, N: int) -> float:
    """Bethe-Weizsäcker binding energy B(Z,N) in MeV."""
    A = Z + N
    if A <= 0 or Z <= 0 or N <= 0:
        return 0.0

    # Pairing term
    if Z % 2 == 0 and N % 2 == 0:
        delta = _aP / A ** 0.5
    elif Z % 2 != 0 and N % 2 != 0:
        delta = -_aP / A ** 0.5
    else:
        delta = 0.0

    B = (
        _aV * A
        - _aS * A ** (2 / 3)
        - _aC * Z * (Z - 1) / A ** (1 / 3)
        - _aA * (A - 2 * Z) ** 2 / A
        + delta
    )
    return max(B, 0.0)


def semf_binding_energy_per_nucleon(Z: int, N: int) -> float:
    A = Z + N
    if A <= 0:
        return 0.0
    return semf_binding_energy(Z, N) / A


def empirical_binding_energy(Z: int, N: int) -> dict | None:
    """Return empirical binding-energy reference data for common isotopes."""
    ref = _EMPIRICAL_BINDING_ENERGY_MEV.get((Z, N))
    return dict(ref) if ref else None


def compare_binding_energy_to_empirical(
    Z: int, N: int, model_binding_energy_MeV: float | None = None
) -> dict | None:
    """Compare SEMF binding energy against a rounded empirical benchmark."""
    ref = empirical_binding_energy(Z, N)
    if not ref:
        return None
    model = (
        semf_binding_energy(Z, N)
        if model_binding_energy_MeV is None
        else float(model_binding_energy_MeV)
    )
    empirical = float(ref["binding_energy_MeV"])
    error = model - empirical
    return {
        "isotope": ref["isotope"],
        "empirical_binding_energy_MeV": empirical,
        "model_binding_energy_MeV": model,
        "signed_error_MeV": error,
        "absolute_error_MeV": abs(error),
        "absolute_error_per_nucleon_MeV": abs(error) / (Z + N),
        "relative_error_percent": 100.0 * error / empirical if empirical else None,
        "source": ref["source"],
    }


def nuclear_mass(Z: int, N: int) -> float:
    """Nuclear mass M(Z,N) in atomic mass units (u)."""
    A = Z + N
    mp = 1.007276  # proton mass u
    mn = 1.008665  # neutron mass u
    B = semf_binding_energy(Z, N)
    return Z * mp + N * mn - B / 931.494  # 1 u = 931.494 MeV/c²


def stability_valley(Z: int, A: int) -> dict:
    """Predict the most stable isotope for a given Z and A range."""
    best_N = round((A - Z) * 1.0)
    best_B = 0.0
    for N in range(max(1, A - Z - 10), A - Z + 11):
        B = semf_binding_energy(Z, N)
        if B > best_B:
            best_B = B
            best_N = N
    return {
        "Z": Z,
        "N_most_stable": best_N,
        "A_most_stable": Z + best_N,
        "binding_energy_MeV": round(best_B, 3),
        "binding_per_nucleon_MeV": round(best_B / (Z + best_N), 4),
    }


class NuclearEngine:
    """Semi-empirical nuclear physics computations and visualizations."""

    def analyze_nucleus(self, Z: int, N: int) -> dict:
        """Full nuclear analysis for a given (Z, N) combination."""
        A = Z + N
        el = get_element(Z)
        B = semf_binding_energy(Z, N)
        B_per_A = B / A if A > 0 else 0
        mass = nuclear_mass(Z, N)
        empirical_comparison = compare_binding_energy_to_empirical(Z, N, B)

        # Magic numbers for extra stability
        magic = {2, 8, 20, 28, 50, 82, 126}
        is_magic_Z = Z in magic
        is_magic_N = N in magic
        is_doubly_magic = is_magic_Z and is_magic_N

        # Stability heuristic: valley of stability
        N_stable_approx = round(Z * (1 + 0.015 * A ** (2 / 3)))
        delta_N = abs(N - N_stable_approx)
        if delta_N == 0:
            stability = "stable"
        elif delta_N <= 3:
            stability = "likely stable"
        elif delta_N <= 8:
            stability = "likely radioactive"
        else:
            stability = "strongly unstable / very short half-life"

        result = {
            "element": el["name"] if el else f"Z={Z}",
            "symbol": el["symbol"] if el else "?",
            "Z": Z,
            "N": N,
            "A": A,
            "isotope_notation": f"{A}{el['symbol'] if el else '?'}",
            "binding_energy_MeV": round(B, 3),
            "binding_per_nucleon_MeV": round(B_per_A, 4),
            "nuclear_mass_u": round(mass, 6),
            "is_magic_Z": is_magic_Z,
            "is_magic_N": is_magic_N,
            "is_doubly_magic": is_doubly_magic,
            "stability_prediction": stability,
            "N_valley_approx": N_stable_approx,
            "proton_to_neutron_ratio": round(Z / N, 3) if N else None,
            "model": {
                "name": "Bethe-Weizsaecker semi-empirical mass formula",
                "limits": (
                    "Bulk nuclear approximation; shell, deformation, and odd-even "
                    "effects are only approximate."
                ),
            },
        }
        if empirical_comparison:
            result["empirical_comparison"] = empirical_comparison
        return result

    def binding_energy_curve(self, z_fixed: int | None = None) -> str:
        """Plot B/A vs A (the nuclear binding energy curve). Returns base64 PNG."""
        fig, ax = plt.subplots(figsize=(11, 6))

        if z_fixed is None:
            # Most stable isotope for each A from 2 to 240
            A_arr, BA_arr = [], []
            for A in range(2, 241):
                best_ba = 0.0
                for Z in range(max(1, A // 3), min(A, 2 * A // 3 + 1)):
                    N = A - Z
                    if N < 1:
                        continue
                    ba = semf_binding_energy_per_nucleon(Z, N)
                    if ba > best_ba:
                        best_ba = ba
                A_arr.append(A)
                BA_arr.append(best_ba)

            ax.plot(A_arr, BA_arr, color="#2196F3", linewidth=2)
            ax.set_title("Nuclear Binding Energy per Nucleon — SEMF (Most Stable Isotopes)", fontsize=13)
        else:
            # Fixed Z, vary N
            N_arr, BA_arr = [], []
            for N in range(1, 160):
                ba = semf_binding_energy_per_nucleon(z_fixed, N)
                if ba > 0:
                    N_arr.append(z_fixed + N)
                    BA_arr.append(ba)
            ax.plot(N_arr, BA_arr, color="#E91E63", linewidth=2,
                    label=f"Z={z_fixed} ({get_element(z_fixed)['symbol'] if get_element(z_fixed) else '?'})")
            ax.legend(fontsize=11)
            ax.set_title(f"Binding Energy per Nucleon — Z={z_fixed} Isotopes", fontsize=13)

        # Mark Fe-56 as peak stability
        ax.axvline(56, color="orange", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(57, 1, "⁵⁶Fe (peak)", color="orange", fontsize=9)
        ax.set_xlabel("Mass Number A", fontsize=12)
        ax.set_ylabel("Binding Energy per Nucleon B/A (MeV)", fontsize=12)
        ax.set_ylim(0, 9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def nuclear_chart(
        self, z_max: int = 40, n_max: int = 50
    ) -> str:
        """Plot nuclear stability chart (Segrè chart) coloured by B/A. Returns base64 PNG."""
        Zs, Ns, BAs = [], [], []
        for Z in range(1, z_max + 1):
            for N in range(1, n_max + 1):
                ba = semf_binding_energy_per_nucleon(Z, N)
                if ba > 0:
                    Zs.append(Z)
                    Ns.append(N)
                    BAs.append(ba)

        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(Ns, Zs, c=BAs, cmap="plasma", s=12,
                        vmin=0, vmax=8.8, alpha=0.85)
        plt.colorbar(sc, ax=ax, label="B/A (MeV)")

        # Valley of stability line
        Z_line = np.arange(1, z_max + 1, dtype=float)
        N_line = Z_line * (1 + 0.015 * (2 * Z_line) ** (2 / 3))
        ax.plot(N_line, Z_line, "w--", linewidth=1.5, alpha=0.6, label="Valley of stability")

        ax.set_xlabel("Neutron Number N", fontsize=12)
        ax.set_ylabel("Proton Number Z", fontsize=12)
        ax.set_title("Segrè Nuclear Chart (SEMF binding energy)", fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def fusion_energy(self, Z1: int, N1: int, Z2: int, N2: int) -> dict:
        """Calculate energy released (Q-value) for nuclear fusion of two nuclei."""
        Z_out = Z1 + Z2
        N_out = N1 + N2
        B_in = semf_binding_energy(Z1, N1) + semf_binding_energy(Z2, N2)
        B_out = semf_binding_energy(Z_out, N_out)
        Q = B_out - B_in  # positive Q = energy released

        el1 = get_element(Z1)
        el2 = get_element(Z2)
        el_out = get_element(Z_out)

        return {
            "reactant_1": f"{Z1+N1}{el1['symbol'] if el1 else '?'}",
            "reactant_2": f"{Z2+N2}{el2['symbol'] if el2 else '?'}",
            "product": f"{Z_out+N_out}{el_out['symbol'] if el_out else '?'}",
            "Q_value_MeV": round(Q, 4),
            "energy_released": Q > 0,
            "binding_energy_in_MeV": round(B_in, 3),
            "binding_energy_out_MeV": round(B_out, 3),
            "interpretation": (
                f"Q = {Q:.3f} MeV → {'energy RELEASED (exothermic fusion)' if Q > 0 else 'energy REQUIRED (endothermic)'}"
            ),
        }

    def fission_energy(self, Z: int, N: int) -> dict:
        """Estimate symmetric fission Q-value: A → two equal fragments."""
        Z_f = Z // 2
        N_f = N // 2
        B_parent = semf_binding_energy(Z, N)
        B_frag = 2 * semf_binding_energy(Z_f, N_f)
        Q = B_frag - B_parent

        el = get_element(Z)
        el_f = get_element(Z_f)
        return {
            "parent": f"{Z+N}{el['symbol'] if el else '?'}",
            "fragment": f"{Z_f+N_f}{el_f['symbol'] if el_f else '?'} × 2",
            "Q_value_MeV": round(Q, 4),
            "energy_released": Q > 0,
            "interpretation": (
                f"Q = {Q:.3f} MeV → {'energy RELEASED (fission possible)' if Q > 0 else 'fission not energetically favoured'}"
            ),
        }

    def separation_energy(self, Z: int, N: int) -> dict:
        """Compute single-nucleon separation energies using the SEMF.

        Sn = B(Z, N) - B(Z, N-1)   [neutron separation energy]
        Sp = B(Z, N) - B(Z-1, N)   [proton separation energy]
        S2n = B(Z, N) - B(Z, N-2)  [two-neutron separation energy]
        S2p = B(Z, N) - B(Z-2, N)  [two-proton separation energy]

        Negative separation energy → that nucleon is unbound (nuclide lies
        beyond the drip line in the SEMF approximation).
        """
        if Z < 1 or N < 1:
            raise ValueError("Z and N must be at least 1.")

        B = semf_binding_energy(Z, N)
        Sn = B - semf_binding_energy(Z, N - 1) if N >= 2 else float("nan")
        Sp = B - semf_binding_energy(Z - 1, N) if Z >= 2 else float("nan")
        S2n = B - semf_binding_energy(Z, N - 2) if N >= 3 else float("nan")
        S2p = B - semf_binding_energy(Z - 2, N) if Z >= 3 else float("nan")
        Sa = B - semf_binding_energy(Z - 2, N - 2) - semf_binding_energy(2, 2) if Z >= 3 and N >= 3 else float("nan")

        el = get_element(Z)
        A = Z + N

        return {
            "isotope": f"{A}{el['symbol'] if el else '?'}",
            "Z": Z, "N": N, "A": A,
            "binding_energy_MeV": round(B, 3),
            "Sn_MeV": round(Sn, 3) if Sn == Sn else None,
            "Sp_MeV": round(Sp, 3) if Sp == Sp else None,
            "S2n_MeV": round(S2n, 3) if S2n == S2n else None,
            "S2p_MeV": round(S2p, 3) if S2p == S2p else None,
            "S_alpha_MeV": round(Sa, 3) if Sa == Sa else None,
            "neutron_bound": bool(Sn > 0) if Sn == Sn else None,
            "proton_bound": bool(Sp > 0) if Sp == Sp else None,
            "model": "SEMF (Bethe-Weizsäcker); separation energies are differences of bulk approximation",
        }

    def alpha_decay_halflife(self, Z: int, N: int) -> dict:
        """Estimate alpha-decay half-life using the Geiger-Nuttall / Viola-Seaborg law.

        log₁₀(t₁/₂ [s]) = a*Z / sqrt(Q_α) + b*Z + c

        where Q_α is the alpha decay Q-value from the SEMF in MeV.
        Uses the Viola-Seaborg parameterisation (1966), which gives results
        accurate to within ~1–2 orders of magnitude for even-even nuclei.

        Parameters
        ----------
        Z: proton number of PARENT nucleus
        N: neutron number of PARENT nucleus
        """
        if Z < 3 or N < 3:
            raise ValueError("Alpha decay requires Z≥3 and N≥3.")

        # SEMF alpha Q-value: Q = B(Z-2,N-2) + B(2,2) - B(Z,N)  [MeV]
        B_parent = semf_binding_energy(Z, N)
        B_daughter = semf_binding_energy(Z - 2, N - 2)
        B_alpha = semf_binding_energy(2, 2)
        Q_alpha = B_daughter + B_alpha - B_parent

        el_parent = get_element(Z)
        el_daughter = get_element(Z - 2)
        A = Z + N

        if Q_alpha <= 0:
            return {
                "isotope": f"{A}{el_parent['symbol'] if el_parent else '?'}",
                "Q_alpha_MeV": round(Q_alpha, 4),
                "alpha_decay_energetically_possible": False,
                "note": "Q ≤ 0: alpha decay not energetically favoured by SEMF.",
            }

        # Viola-Seaborg parameterisation
        # log10(t_1/2 [s]) = (a*Z_d + b) / sqrt(Q) + (c*Z_d + d)
        # Z_d = Z - 2 (daughter proton number)
        Z_d = Z - 2
        a_vs, b_vs = 1.66175, -8.5166
        c_vs, d_vs = -0.20228, -33.9069

        log10_t = (a_vs * Z_d + b_vs) / (Q_alpha ** 0.5) + (c_vs * Z_d + d_vs)
        t_half_s = 10 ** log10_t

        # Convert to human-readable unit
        if t_half_s < 1e-9:
            t_str = f"{t_half_s*1e12:.2e} ps"
        elif t_half_s < 1e-3:
            t_str = f"{t_half_s*1e6:.2e} μs"
        elif t_half_s < 60:
            t_str = f"{t_half_s:.2e} s"
        elif t_half_s < 3600:
            t_str = f"{t_half_s/60:.2e} min"
        elif t_half_s < 86400:
            t_str = f"{t_half_s/3600:.2e} hr"
        elif t_half_s < 3.156e7:
            t_str = f"{t_half_s/86400:.2e} days"
        elif t_half_s < 3.156e10:
            t_str = f"{t_half_s/3.156e7:.2e} yr"
        else:
            t_str = f"{t_half_s/3.156e9:.2e} Gyr"

        return {
            "isotope_parent": f"{A}{el_parent['symbol'] if el_parent else '?'}",
            "isotope_daughter": f"{A-4}{el_daughter['symbol'] if el_daughter else '?'}",
            "Z": Z, "N": N, "A": A,
            "Q_alpha_MeV": round(Q_alpha, 4),
            "alpha_decay_energetically_possible": True,
            "log10_halflife_s": round(log10_t, 3),
            "halflife_s": float(t_half_s),
            "halflife_human": t_str,
            "model": "Viola-Seaborg (Geiger-Nuttall) — accurate to ~1–2 orders of magnitude for even-even nuclei",
            "note": "Q-value from SEMF; shell effects and odd-A corrections not included.",
        }
