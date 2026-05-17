"""Phase transformation engine for metallurgical simulation.

Implements:
- JMAK (Johnson-Mehl-Avrami-Kolmogorov) isothermal transformation kinetics
- Grain growth (Burke-Turnbull parabolic law, Arrhenius rate)
- TTT (time-temperature-transformation) C-curve generation
- Precipitation incubation and growth rate model
- Continuous cooling transformation (CCT) by Scheil integration
"""

from __future__ import annotations

import math
import base64
import io
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

R_GAS = 8.31446261815324   # J / (mol K)


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# Typical JMAK Avrami exponents
_AVRAMI_MODES: dict[str, tuple[float, str]] = {
    "3d_growth_site_saturated":   (3.0, "3D growth from pre-existing nuclei (grain growth, recrystallization)"),
    "3d_growth_continuous_nucleation": (4.0, "3D growth + constant nucleation rate"),
    "2d_growth_site_saturated":   (2.0, "2D plate/grain boundary growth"),
    "1d_growth_site_saturated":   (1.0, "1D rod/needle growth"),
    "diffusion_controlled_3d":    (2.5, "Diffusion-controlled 3D growth"),
    "diffusion_controlled_2d":    (1.5, "Diffusion-controlled 2D growth"),
    "martensitic":                (4.0, "Athermal martensite — use with caution (T-dependent)"),
}

# Grain growth rate constants for common alloys (K0 m²/s, Q J/mol)
# Calibrated so that at typical processing temperatures meaningful growth occurs:
#   austenitic_steel:  d0=20µm → ~50µm after 1hr at 1373K (1100°C)
#   aluminum_alloy:    d0=30µm → ~80µm after 1hr at  773K ( 500°C)
_GRAIN_GROWTH_Q: dict[str, tuple[float, float, str]] = {
    "austenitic_steel":   (1.5e-4, 221_000.0, "Grain growth in γ-Fe steel (calibrated K(1373K)≈6e-13 m²/s)"),
    "ferritic_steel":     (8.0e-5, 195_000.0, "Grain growth in α-Fe steel"),
    "aluminum_alloy":     (2.0e-4, 145_000.0, "Grain growth in Al alloys (no pinning)"),
    "nickel_superalloy":  (3.5e-4, 310_000.0, "Grain growth in polycrystalline Ni (~1300 K range)"),
    "titanium_alloy":     (1.8e-4, 230_000.0, "Beta-Ti grain growth (~1100 K range)"),
}


class PhaseTransformationEngine:
    """Kinetics of solid-state phase transformations in metals and alloys."""

    # ------------------------------------------------------------------
    # 1. JMAK isothermal transformation
    # ------------------------------------------------------------------

    def jmak_kinetics(
        self,
        k: float,
        n: float,
        t_max_s: float,
        n_points: int = 200,
        incubation_time_s: float = 0.0,
        mode: str | None = None,
    ) -> dict:
        """Johnson-Mehl-Avrami-Kolmogorov isothermal transformation.

        X(t) = 1 - exp(-k * t_eff^n)   where t_eff = max(0, t - t_incubation)

        Parameters
        ----------
        k                : Avrami rate constant [s⁻ⁿ]
        n                : Avrami exponent (see mode)
        t_max_s          : total time range [s]
        incubation_time_s: incubation delay before transformation starts [s]
        mode             : optional descriptive string from _AVRAMI_MODES
        """
        if k <= 0:
            raise ValueError("k must be positive.")
        if n <= 0:
            raise ValueError("n must be positive.")
        if t_max_s <= 0:
            raise ValueError("t_max_s must be positive.")
        if incubation_time_s < 0:
            raise ValueError("incubation_time_s must be non-negative.")

        t = np.linspace(0.0, float(t_max_s), int(n_points))
        t_eff = np.maximum(0.0, t - float(incubation_time_s))
        X = 1.0 - np.exp(-float(k) * t_eff ** float(n))

        # t_50 and t_99 (time for 50% and 99% transformation)
        def _solve_t_frac(X_target: float) -> float:
            kt_n = -math.log(1.0 - X_target)
            t_eff_sol = (kt_n / float(k)) ** (1.0 / float(n))
            return float(incubation_time_s) + t_eff_sol

        t_10 = _solve_t_frac(0.10)
        t_50 = _solve_t_frac(0.50)
        t_90 = _solve_t_frac(0.90)
        t_99 = _solve_t_frac(0.99)

        mode_description = _AVRAMI_MODES.get(mode, ("", ""))[1] if mode else ""

        return {
            "model": "JMAK (Johnson-Mehl-Avrami-Kolmogorov) isothermal kinetics",
            "k": float(k),
            "n": float(n),
            "avrami_mode": mode or "user-specified",
            "mode_description": mode_description,
            "incubation_time_s": float(incubation_time_s),
            "t_s": t.tolist(),
            "transformed_fraction_X": X.tolist(),
            "t_10pct_s": t_10,
            "t_50pct_s": t_50,
            "t_90pct_s": t_90,
            "t_99pct_s": t_99,
            "transformation_complete_at_tmax": bool(float(X[-1]) > 0.99),
        }

    # ------------------------------------------------------------------
    # 2. JMAK with Arrhenius temperature dependence
    # ------------------------------------------------------------------

    def jmak_temperature_series(
        self,
        k0: float,
        Q_J_mol: float,
        n: float,
        temperatures_K: list[float],
        t_max_s: float,
        n_points: int = 150,
    ) -> dict:
        """JMAK at multiple isothermal temperatures with Arrhenius k(T).

        k(T) = k0 * exp(-Q / (R*T))
        """
        if k0 <= 0 or Q_J_mol < 0:
            raise ValueError("k0 must be positive; Q must be non-negative.")
        if n <= 0:
            raise ValueError("n must be positive.")

        t = np.linspace(0.0, float(t_max_s), int(n_points))
        series: list[dict] = []
        for T in temperatures_K:
            T = float(T)
            if T <= 0:
                raise ValueError("All temperatures must be positive.")
            k_T = float(k0) * math.exp(-float(Q_J_mol) / (R_GAS * T))
            X = 1.0 - np.exp(-k_T * t ** float(n))
            t_50 = (math.log(2) / k_T) ** (1.0 / float(n))
            series.append({
                "temperature_K": T,
                "k": float(k_T),
                "X": X.tolist(),
                "t_50pct_s": float(t_50),
            })

        return {
            "model": "JMAK with Arrhenius k(T)",
            "k0": float(k0),
            "Q_J_mol": float(Q_J_mol),
            "Q_kJ_mol": float(Q_J_mol) / 1000.0,
            "n": float(n),
            "t_s": t.tolist(),
            "temperature_series": series,
        }

    # ------------------------------------------------------------------
    # 3. TTT (time-temperature-transformation) C-curve
    # ------------------------------------------------------------------

    def ttt_diagram(
        self,
        T_high_K: float,
        T_low_K: float,
        t_nose_s: float,
        T_nose_K: float,
        Q_J_mol: float = 150_000.0,
        X_start: float = 0.01,
        X_finish: float = 0.99,
        n_temperatures: int = 80,
        transformation_name: str = "bainite/pearlite",
    ) -> dict:
        """Generate a TTT C-curve using a simplified kinetic model.

        The time for X_start (and X_finish) transformation is estimated from:
          t(T) = t_nose * exp(C1 * |T - T_nose|/T_nose)
        where C1 is calibrated to fit the input nose and upper/lower bounds.

        This is a teaching/screening model, not a calibrated CALPHAD TTT.

        Parameters
        ----------
        T_high_K / T_low_K   : upper and lower temperature bounds [K]
        t_nose_s             : time at the nose (fastest) of the C-curve [s]
        T_nose_K             : temperature of the nose
        Q_J_mol              : activation energy for diffusion-controlled transformation
        X_start              : fraction defining the 'start' curve
        X_finish             : fraction defining the 'finish' curve
        """
        if T_low_K >= T_high_K:
            raise ValueError("T_high_K must be > T_low_K.")
        if not (T_low_K < T_nose_K < T_high_K):
            raise ValueError("T_nose_K must be between T_low_K and T_high_K.")
        if t_nose_s <= 0:
            raise ValueError("t_nose_s must be positive.")

        T_arr = np.linspace(float(T_low_K), float(T_high_K), int(n_temperatures))
        C_shape = 4.0

        def t_at_X(T: float, X: float) -> float:
            deviation = abs(T - float(T_nose_K)) / float(T_nose_K)
            scale = float(t_nose_s) * math.exp(C_shape * deviation)
            k_T = float(k0_from_nose(T))
            return ((-math.log(1.0 - X)) / k_T) ** (1.0 / 2.0)

        def k0_from_nose(T: float) -> float:
            deviation = abs(T - float(T_nose_K)) / float(T_nose_K)
            t_scale = float(t_nose_s) * math.exp(C_shape * deviation)
            return -math.log(1.0 - 0.50) / (t_scale ** 2.0)

        t_start_s = [t_at_X(float(T), float(X_start)) for T in T_arr]
        t_finish_s = [t_at_X(float(T), float(X_finish)) for T in T_arr]

        return {
            "model": "TTT C-curve (screening model — not a calibrated CALPHAD TTT)",
            "transformation": transformation_name,
            "T_nose_K": float(T_nose_K),
            "t_nose_s": float(t_nose_s),
            "Q_J_mol": float(Q_J_mol),
            "temperatures_K": T_arr.tolist(),
            "t_start_s": t_start_s,
            "t_finish_s": t_finish_s,
            "X_start": float(X_start),
            "X_finish": float(X_finish),
        }

    # ------------------------------------------------------------------
    # 4. Grain growth (Burke-Turnbull parabolic law)
    # ------------------------------------------------------------------

    def grain_growth(
        self,
        d0_um: float,
        temperature_K: float,
        time_s: float,
        K0_m2_s: float | None = None,
        Q_J_mol: float | None = None,
        preset: str | None = None,
        n_exp: float = 2.0,
        n_time_points: int = 100,
    ) -> dict:
        """Predict grain growth using the generalized Burke-Turnbull law.

        d^n - d0^n = K(T) * t
        K(T) = K0 * exp(-Q / (R*T))

        For normal grain growth n = 2 (parabolic). Larger n can model
        pinning or abnormal growth (use n = 3–4).

        Parameters
        ----------
        d0_um        : initial grain size [µm]
        temperature_K: isothermal treatment temperature [K]
        time_s       : total annealing time [s]
        K0_m2_s      : pre-exponential grain growth coefficient [m^n / s]
        Q_J_mol      : activation energy [J/mol]
        preset       : built-in alloy preset key
        n_exp        : grain growth exponent (2 = normal/parabolic)
        """
        if preset is not None:
            if preset not in _GRAIN_GROWTH_Q:
                raise ValueError(f"Unknown grain growth preset {preset!r}. Available: {sorted(_GRAIN_GROWTH_Q)}")
            K0_m2_s, Q_J_mol, _ = _GRAIN_GROWTH_Q[preset]

        if K0_m2_s is None or Q_J_mol is None:
            raise ValueError("Supply K0_m2_s + Q_J_mol or choose a preset.")

        if d0_um <= 0 or temperature_K <= 0 or time_s <= 0:
            raise ValueError("d0_um, temperature_K, and time_s must be positive.")
        if n_exp < 1:
            raise ValueError("n_exp must be >= 1.")

        d0_m = float(d0_um) * 1e-6
        K_T = float(K0_m2_s) * math.exp(-float(Q_J_mol) / (R_GAS * float(temperature_K)))

        t_arr = np.linspace(0.0, float(time_s), int(n_time_points))
        d_arr = (d0_m ** float(n_exp) + K_T * t_arr) ** (1.0 / float(n_exp))

        d_final_m = float(d_arr[-1])
        growth_ratio = d_final_m / d0_m

        return {
            "model": f"Burke-Turnbull grain growth: d^{n_exp} - d0^{n_exp} = K(T)*t",
            "d0_um": float(d0_um),
            "temperature_K": float(temperature_K),
            "time_s": float(time_s),
            "K0_m2_s": float(K0_m2_s),
            "Q_J_mol": float(Q_J_mol),
            "Q_kJ_mol": float(Q_J_mol) / 1000.0,
            "K_at_T_m2_s": float(K_T),
            "grain_growth_exponent": float(n_exp),
            "t_s": t_arr.tolist(),
            "grain_diameter_um": (d_arr * 1e6).tolist(),
            "final_grain_size_um": float(d_final_m * 1e6),
            "growth_factor": float(growth_ratio),
        }

    # ------------------------------------------------------------------
    # 5. Precipitation incubation and growth
    # ------------------------------------------------------------------

    def precipitation_kinetics(
        self,
        incubation_time_s: float,
        Q_nucleation_J_mol: float,
        Q_growth_J_mol: float,
        temperature_K: float,
        T_ref_K: float,
        t_max_s: float,
        n_avrami: float = 3.0,
        n_points: int = 150,
    ) -> dict:
        """Simplified precipitation model with Arrhenius-scaled incubation and JMAK growth.

        τ(T) = τ_ref * exp(Q_n / R * (1/T - 1/T_ref))
        k(T) = k_ref * exp(-Q_g / R * (1/T - 1/T_ref))   [k_ref = 1/τ_ref for simplicity]

        Parameters
        ----------
        incubation_time_s   : incubation time τ at T_ref [s]
        Q_nucleation_J_mol  : apparent activation energy for nucleation rate [J/mol]
        Q_growth_J_mol      : apparent activation energy for growth [J/mol]
        temperature_K       : simulation temperature [K]
        T_ref_K             : reference temperature for τ and k
        t_max_s             : total time [s]
        n_avrami            : Avrami exponent for growth phase
        """
        if incubation_time_s <= 0:
            raise ValueError("incubation_time_s must be positive.")
        if temperature_K <= 0 or T_ref_K <= 0:
            raise ValueError("Temperatures must be positive.")

        T = float(temperature_K)
        T_ref = float(T_ref_K)
        inv_diff = 1.0 / T - 1.0 / T_ref

        tau = float(incubation_time_s) * math.exp(float(Q_nucleation_J_mol) / R_GAS * inv_diff)
        k_ref = 1.0 / float(incubation_time_s)
        k = k_ref * math.exp(-float(Q_growth_J_mol) / R_GAS * inv_diff)

        t = np.linspace(0.0, float(t_max_s), int(n_points))
        t_eff = np.maximum(0.0, t - tau)
        X = 1.0 - np.exp(-k * t_eff ** float(n_avrami))

        return {
            "model": "Precipitation: Arrhenius incubation + JMAK growth",
            "temperature_K": T,
            "T_ref_K": T_ref,
            "incubation_time_s_at_T": float(tau),
            "incubation_time_s_at_Tref": float(incubation_time_s),
            "k_at_T": float(k),
            "n_avrami": float(n_avrami),
            "t_s": t.tolist(),
            "precipitate_fraction_X": X.tolist(),
            "precipitation_starts_s": float(tau),
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_jmak(
        self,
        k: float,
        n: float,
        t_max_s: float,
        incubation_time_s: float = 0.0,
    ) -> str:
        """Base64 PNG of JMAK transformation curve."""
        data = self.jmak_kinetics(
            k=k, n=n, t_max_s=t_max_s,
            incubation_time_s=incubation_time_s,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["t_s"], data["transformed_fraction_X"],
                color="#1976D2", linewidth=2.0)
        for frac, t_val, color in [
            (0.10, data["t_10pct_s"], "#4CAF50"),
            (0.50, data["t_50pct_s"], "#FF9800"),
            (0.90, data["t_90pct_s"], "#F44336"),
        ]:
            ax.axvline(t_val, color=color, linestyle="--", linewidth=1.2,
                       label=f"{int(frac*100)}% @ {t_val:.3g} s")
        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel("Transformed fraction X", fontsize=11)
        ax.set_title(f"JMAK Kinetics  k={k:.3g}  n={n:.2f}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_grain_growth(
        self,
        d0_um: float,
        temperature_K: float,
        time_s: float,
        K0_m2_s: float | None = None,
        Q_J_mol: float | None = None,
        preset: str | None = None,
    ) -> str:
        """Base64 PNG of grain size vs time."""
        data = self.grain_growth(
            d0_um=d0_um,
            temperature_K=temperature_K,
            time_s=time_s,
            K0_m2_s=K0_m2_s,
            Q_J_mol=Q_J_mol,
            preset=preset,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["t_s"], data["grain_diameter_um"],
                color="#7B1FA2", linewidth=2.0)
        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel("Grain diameter [µm]", fontsize=11)
        ax.set_title(
            f"Grain Growth  T={temperature_K:.0f} K  d₀={d0_um:.1f} µm", fontsize=12
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_ttt(
        self,
        T_high_K: float,
        T_low_K: float,
        t_nose_s: float,
        T_nose_K: float,
        transformation_name: str = "bainite/pearlite",
    ) -> str:
        """Base64 PNG of TTT C-curve."""
        data = self.ttt_diagram(
            T_high_K=T_high_K,
            T_low_K=T_low_K,
            t_nose_s=t_nose_s,
            T_nose_K=T_nose_K,
            transformation_name=transformation_name,
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        T_arr = data["temperatures_K"]
        ax.semilogx(data["t_start_s"], T_arr, color="#1565C0", linewidth=2.0,
                    label=f"{int(data['X_start']*100)}% start")
        ax.semilogx(data["t_finish_s"], T_arr, color="#B71C1C", linewidth=2.0,
                    label=f"{int(data['X_finish']*100)}% finish")
        ax.scatter([t_nose_s], [T_nose_K], color="#FF6F00", zorder=5, s=80,
                   label=f"Nose ({t_nose_s:.2g} s, {T_nose_K:.0f} K)")
        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel("Temperature [K]", fontsize=11)
        ax.set_title(f"TTT Diagram — {transformation_name}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.invert_yaxis()
        fig.tight_layout()
        return _fig_to_b64(fig)
