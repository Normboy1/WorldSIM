"""Diffusion engine for metallurgical simulation.

Implements:
- Fick's 1st and 2nd laws (analytical solutions)
- Arrhenius temperature-dependent diffusivity
- Darken binary interdiffusion and Kirkendall shift
- Semi-infinite solid carburization / case-depth prediction
- Grain boundary diffusion (Fisher/Whipple model)
- Multi-temperature diffusivity maps for process window analysis
"""

from __future__ import annotations

import math
import base64
import io

import numpy as np
from scipy.special import erf, erfc
from scipy.integrate import quad

import matplotlib.pyplot as plt


R_GAS = 8.31446261815324   # J / (mol K)

# Preloaded Arrhenius coefficients for common diffusing species
# {label: (D0 m²/s, Q J/mol, notes)}
_DIFFUSIVITY_PRESETS: dict[str, tuple[float, float, str]] = {
    "C_in_austenite":  (2.0e-5, 142_000.0, "C in γ-Fe (austenite), Fick diffusion"),
    "C_in_ferrite":    (6.2e-7,  80_000.0, "C in α-Fe (ferrite)"),
    "N_in_austenite":  (9.1e-5, 168_000.0, "N in γ-Fe (austenite)"),
    "Cr_in_austenite": (3.6e-5, 272_000.0, "Cr in γ-Fe"),
    "Ni_in_austenite": (1.9e-5, 264_000.0, "Ni in γ-Fe"),
    "Al_in_aluminum":  (1.37e-4,122_000.0, "Al self-diffusion"),
    "Cu_in_copper":    (7.8e-5, 197_000.0, "Cu self-diffusion"),
    "Fe_self":         (1.0e-4, 239_000.0, "Fe self-diffusion in γ-Fe"),
    "H_in_steel":      (1.4e-7,  28_000.0, "H in α-Fe (NACE model, low-T)"),
}

# Grain boundary diffusion presets (D0_gb m³/s, Q_gb J/mol)
_GB_PRESETS: dict[str, tuple[float, float]] = {
    "Fe_self_gb":  (2.1e-15, 174_000.0),
    "Cr_in_Fe_gb": (3.7e-16, 218_000.0),
    "C_in_Fe_gb":  (8.8e-14,  92_000.0),
}


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


class DiffusionEngine:
    """Analytical and numerical diffusion solutions for metallurgical applications."""

    # ------------------------------------------------------------------
    # 1. Arrhenius diffusivity
    # ------------------------------------------------------------------

    def arrhenius_diffusivity(
        self,
        D0: float,
        activation_energy_J_mol: float,
        temperatures_K: list[float] | None = None,
        T_min_K: float = 500.0,
        T_max_K: float = 1500.0,
        n_temps: int = 60,
        preset: str | None = None,
    ) -> dict:
        """Compute D(T) = D0 * exp(-Q / RT) over a temperature range.

        Parameters
        ----------
        D0                   : pre-exponential factor [m²/s]
        activation_energy_J_mol : activation energy Q [J/mol]
        temperatures_K       : optional explicit temperature list [K]
        T_min_K / T_max_K    : range if temperatures_K not supplied
        preset               : optional key from built-in presets (overrides D0/Q)
        """
        if preset is not None:
            if preset not in _DIFFUSIVITY_PRESETS:
                raise ValueError(
                    f"Unknown preset {preset!r}. Available: {sorted(_DIFFUSIVITY_PRESETS)}"
                )
            D0, activation_energy_J_mol, _ = _DIFFUSIVITY_PRESETS[preset]

        if D0 <= 0:
            raise ValueError("D0 must be positive.")
        if activation_energy_J_mol < 0:
            raise ValueError("activation_energy_J_mol must be non-negative.")

        if temperatures_K is not None:
            T_arr = np.array([float(t) for t in temperatures_K])
        else:
            T_arr = np.linspace(T_min_K, T_max_K, n_temps)

        if np.any(T_arr <= 0):
            raise ValueError("All temperatures must be positive.")

        D_arr = D0 * np.exp(-activation_energy_J_mol / (R_GAS * T_arr))

        return {
            "D0_m2_s": float(D0),
            "Q_J_mol": float(activation_energy_J_mol),
            "Q_kJ_mol": float(activation_energy_J_mol / 1000.0),
            "temperatures_K": T_arr.tolist(),
            "diffusivities_m2_s": D_arr.tolist(),
            "D_at_Tmax_m2_s": float(D_arr[-1]),
            "D_at_Tmin_m2_s": float(D_arr[0]),
            "model": "Arrhenius: D = D0 * exp(-Q/RT)",
        }

    # ------------------------------------------------------------------
    # 2. Fick's 2nd law — semi-infinite solid (carburization / nitriding)
    # ------------------------------------------------------------------

    def diffusion_profile(
        self,
        D: float,
        time_s: float,
        C_surface: float = 1.0,
        C_initial: float = 0.0,
        x_max_m: float | None = None,
        n_points: int = 120,
        case_threshold: float | None = None,
        temperature_K: float | None = None,
        D0: float | None = None,
        activation_energy_J_mol: float | None = None,
        preset: str | None = None,
    ) -> dict:
        """Concentration profile in a semi-infinite solid from a constant-surface-
        concentration boundary condition (Fick's 2nd law, analytical solution).

        C(x,t) = C_s + (C_0 - C_s) * erf(x / (2*sqrt(D*t)))

        Equivalently written: C(x,t) = C_s - (C_s - C_0) * erfc_complement

        Parameters
        ----------
        D               : diffusivity [m²/s]. If temperature_K + D0 + Q supplied,
                          D is computed from Arrhenius; explicit D takes priority.
        time_s          : diffusion time [s]
        C_surface       : surface concentration [any consistent units]
        C_initial       : initial bulk concentration
        x_max_m         : depth range; defaults to 6*sqrt(D*t)
        case_threshold  : concentration at which 'case depth' is defined (default: midpoint)
        temperature_K   : temperature for Arrhenius lookup (optional)
        D0 / activation_energy_J_mol: Arrhenius parameters (optional)
        preset          : built-in Arrhenius preset key (optional)
        """
        if preset is not None and temperature_K is not None:
            if preset not in _DIFFUSIVITY_PRESETS:
                raise ValueError(f"Unknown preset {preset!r}.")
            _D0, _Q, _ = _DIFFUSIVITY_PRESETS[preset]
            D = _D0 * math.exp(-_Q / (R_GAS * float(temperature_K)))
        elif D0 is not None and activation_energy_J_mol is not None and temperature_K is not None:
            D = float(D0) * math.exp(-float(activation_energy_J_mol) / (R_GAS * float(temperature_K)))

        if D <= 0:
            raise ValueError("D must be positive.")
        if time_s <= 0:
            raise ValueError("time_s must be positive.")

        sqrt_Dt = math.sqrt(D * time_s)
        if x_max_m is None:
            x_max_m = 6.0 * sqrt_Dt

        x = np.linspace(0.0, float(x_max_m), int(n_points))
        # C(x,t) = Cs + (C0 - Cs) * erf(x / (2*sqrt(Dt)))
        erf_arg = x / (2.0 * sqrt_Dt)
        C = float(C_surface) + (float(C_initial) - float(C_surface)) * erf(erf_arg)

        # Case depth: depth where C equals threshold
        if case_threshold is None:
            case_threshold = 0.5 * (float(C_surface) + float(C_initial))

        # Solve for x_case: erf(x_case/(2*sqrt(Dt))) = (Cs - C_threshold)/(Cs - C0)
        Cs, C0 = float(C_surface), float(C_initial)
        if abs(Cs - C0) > 1e-30:
            erf_val = (Cs - float(case_threshold)) / (Cs - C0)
            erf_val = max(-0.9999, min(0.9999, erf_val))
            from scipy.special import erfinv
            x_case = 2.0 * sqrt_Dt * float(erfinv(erf_val))
            x_case = max(0.0, x_case)
        else:
            x_case = 0.0

        # Mean diffusion distance (Einstein): <x> ~ sqrt(2*D*t)
        mean_diffusion_distance_m = math.sqrt(2.0 * D * time_s)

        return {
            "model": "Fick 2nd law — semi-infinite, constant surface concentration",
            "D_m2_s": float(D),
            "time_s": float(time_s),
            "C_surface": Cs,
            "C_initial": C0,
            "sqrt_Dt_m": float(sqrt_Dt),
            "mean_diffusion_distance_m": float(mean_diffusion_distance_m),
            "case_threshold": float(case_threshold),
            "case_depth_m": float(x_case),
            "case_depth_mm": float(x_case * 1000.0),
            "x_m": x.tolist(),
            "concentration": C.tolist(),
            "surface_flux_mol_m2_s": float((Cs - C0) / (math.sqrt(math.pi) * sqrt_Dt)) if time_s > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # 3. Fick's 1st law — steady-state flux
    # ------------------------------------------------------------------

    def steady_state_flux(
        self,
        D: float,
        C_high: float,
        C_low: float,
        thickness_m: float,
    ) -> dict:
        """Steady-state diffusion flux through a membrane: J = -D * (C_low - C_high) / L.

        Parameters
        ----------
        D           : diffusivity [m²/s]
        C_high      : concentration on high side [mol/m³ or consistent units]
        C_low       : concentration on low side
        thickness_m : membrane/layer thickness [m]
        """
        if D <= 0:
            raise ValueError("D must be positive.")
        if thickness_m <= 0:
            raise ValueError("thickness_m must be positive.")

        dC_dx = (float(C_low) - float(C_high)) / float(thickness_m)
        J = -float(D) * dC_dx

        return {
            "model": "Fick 1st law — steady state",
            "D_m2_s": float(D),
            "C_high": float(C_high),
            "C_low": float(C_low),
            "concentration_gradient_per_m": float(dC_dx),
            "flux": float(J),
            "flux_units": "same as C units × m/s (e.g. mol m⁻² s⁻¹ if C in mol/m³)",
            "permeability": float(abs(J) * float(thickness_m)),
        }

    # ------------------------------------------------------------------
    # 4. Darken binary interdiffusion and Kirkendall shift
    # ------------------------------------------------------------------

    def darken_interdiffusion(
        self,
        D_A_m2_s: float,
        D_B_m2_s: float,
        x_B: float,
        dxB_dx_per_m: float = 0.0,
    ) -> dict:
        """Darken's equation for binary interdiffusion coefficient.

        D̃ = x_B * D_A + x_A * D_B
        Kirkendall velocity: v = (D_A - D_B) * dC_B/dx

        Parameters
        ----------
        D_A_m2_s     : intrinsic diffusivity of A in A-B alloy [m²/s]
        D_B_m2_s     : intrinsic diffusivity of B in A-B alloy [m²/s]
        x_B          : mole fraction of B (0 to 1)
        dxB_dx_per_m : concentration gradient dC_B/dx [1/m] for Kirkendall velocity
        """
        if not (0.0 <= float(x_B) <= 1.0):
            raise ValueError("x_B must be between 0 and 1.")
        if D_A_m2_s < 0 or D_B_m2_s < 0:
            raise ValueError("Diffusivities must be non-negative.")

        x_b = float(x_B)
        x_a = 1.0 - x_b
        D_A = float(D_A_m2_s)
        D_B = float(D_B_m2_s)

        D_tilde = x_b * D_A + x_a * D_B
        v_kirkendall = (D_A - D_B) * float(dxB_dx_per_m)

        return {
            "model": "Darken binary interdiffusion",
            "D_A_m2_s": D_A,
            "D_B_m2_s": D_B,
            "x_B": x_b,
            "x_A": x_a,
            "interdiffusion_coefficient_m2_s": float(D_tilde),
            "kirkendall_velocity_m_s": float(v_kirkendall),
            "diffusivity_ratio_DA_DB": float(D_A / D_B) if D_B > 0 else float("inf"),
            "notes": [
                "Darken equation assumes ideal solution (thermodynamic factor = 1).",
                "Kirkendall shift is nonzero when D_A ≠ D_B (porosity at slow side).",
            ],
        }

    # ------------------------------------------------------------------
    # 5. Grain boundary diffusion (Fisher / Whipple model)
    # ------------------------------------------------------------------

    def grain_boundary_diffusion(
        self,
        D_bulk_m2_s: float,
        D_gb_m2_s: float,
        delta_gb_m: float = 5e-10,
        grain_size_m: float = 20e-6,
        temperature_K: float | None = None,
        preset: str | None = None,
    ) -> dict:
        """Effective diffusivity including grain boundary short-circuit paths.

        Effective diffusivity (Hart equation):
        D_eff = f_gb * D_gb + (1 - f_gb) * D_bulk
        where f_gb = delta_gb / grain_size (volume fraction of GB)

        Fisher parameter: β = delta_gb * D_gb / (2 * D_bulk * sqrt(D_bulk * t))
        β > 1: C-type kinetics (bulk limited), β >> 1: B-type (GB dominated)

        Parameters
        ----------
        D_bulk_m2_s : bulk diffusion coefficient [m²/s]
        D_gb_m2_s   : grain boundary diffusion coefficient [m²/s]
        delta_gb_m  : grain boundary width [m] (typically 0.3–1 nm)
        grain_size_m: mean grain diameter [m]
        temperature_K: temperature for preset lookup (optional)
        preset      : built-in GB diffusion preset key
        """
        if preset is not None and temperature_K is not None:
            if preset not in _GB_PRESETS:
                raise ValueError(f"Unknown GB preset {preset!r}. Available: {sorted(_GB_PRESETS)}")
            D0_gb_vol, Q_gb = _GB_PRESETS[preset]
            D_gb_m2_s = D0_gb_vol / float(delta_gb_m) * math.exp(-Q_gb / (R_GAS * float(temperature_K)))

        if D_bulk_m2_s <= 0 or D_gb_m2_s <= 0:
            raise ValueError("Diffusivities must be positive.")
        if delta_gb_m <= 0 or grain_size_m <= 0:
            raise ValueError("delta_gb_m and grain_size_m must be positive.")

        f_gb = float(delta_gb_m) / float(grain_size_m)
        D_eff = f_gb * float(D_gb_m2_s) + (1.0 - f_gb) * float(D_bulk_m2_s)
        enhancement = D_eff / float(D_bulk_m2_s)

        return {
            "model": "Hart equation — grain boundary short-circuit diffusion",
            "D_bulk_m2_s": float(D_bulk_m2_s),
            "D_gb_m2_s": float(D_gb_m2_s),
            "delta_gb_m": float(delta_gb_m),
            "grain_size_m": float(grain_size_m),
            "grain_boundary_volume_fraction": float(f_gb),
            "effective_diffusivity_m2_s": float(D_eff),
            "enhancement_factor": float(enhancement),
            "gb_dominant": float(D_gb_m2_s) * float(delta_gb_m) > 10.0 * float(D_bulk_m2_s) * float(grain_size_m),
            "notes": [
                "Hart equation valid when grain boundaries form a connected network.",
                "B-type kinetics assumed (Fisher regime); valid when bulk diffusion is slow.",
                "Grain boundary diffusivity typically 10³–10⁶ × bulk diffusivity.",
            ],
        }

    # ------------------------------------------------------------------
    # 6. Multi-temperature process window map
    # ------------------------------------------------------------------

    def process_window_map(
        self,
        D0: float,
        activation_energy_J_mol: float,
        times_s: list[float],
        temperatures_K: list[float],
        C_surface: float = 1.0,
        C_initial: float = 0.0,
        case_threshold: float | None = None,
        preset: str | None = None,
    ) -> dict:
        """Compute case depths for a grid of (temperature, time) combinations.

        Returns a 2D map useful for process window selection:
        case_depth_mm[i][j] corresponds to (temperatures_K[i], times_s[j]).
        """
        if preset is not None:
            if preset not in _DIFFUSIVITY_PRESETS:
                raise ValueError(f"Unknown preset {preset!r}.")
            D0, activation_energy_J_mol, _ = _DIFFUSIVITY_PRESETS[preset]

        if D0 <= 0 or activation_energy_J_mol < 0:
            raise ValueError("D0 must be positive; Q must be non-negative.")

        T_arr = [float(t) for t in temperatures_K]
        t_arr = [float(t) for t in times_s]
        if any(T <= 0 for T in T_arr):
            raise ValueError("All temperatures must be positive.")
        if any(t <= 0 for t in t_arr):
            raise ValueError("All times must be positive.")

        if case_threshold is None:
            case_threshold = 0.5 * (float(C_surface) + float(C_initial))

        Cs, C0 = float(C_surface), float(C_initial)
        erf_val_target = (Cs - float(case_threshold)) / (Cs - C0) if abs(Cs - C0) > 1e-30 else 0.5

        from scipy.special import erfinv

        case_depth_grid: list[list[float]] = []
        for T in T_arr:
            row = []
            D = float(D0) * math.exp(-float(activation_energy_J_mol) / (R_GAS * T))
            for t in t_arr:
                sqrt_Dt = math.sqrt(D * t)
                erf_clamped = max(-0.9999, min(0.9999, float(erf_val_target)))
                x_case = 2.0 * sqrt_Dt * float(erfinv(erf_clamped))
                row.append(max(0.0, x_case) * 1000.0)  # mm
            case_depth_grid.append(row)

        return {
            "model": "Case-depth process window map (Fick 2nd law, Arrhenius D)",
            "D0_m2_s": float(D0),
            "Q_J_mol": float(activation_energy_J_mol),
            "case_threshold": float(case_threshold),
            "temperatures_K": T_arr,
            "times_s": t_arr,
            "case_depth_mm_grid": case_depth_grid,
            "axis_labels": {"rows": "temperature_K", "cols": "time_s"},
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_diffusion_profile(
        self,
        D: float,
        time_s: float,
        C_surface: float = 1.0,
        C_initial: float = 0.0,
        x_max_m: float | None = None,
    ) -> str:
        """Return base64 PNG of concentration vs depth profile."""
        data = self.diffusion_profile(
            D=D, time_s=time_s,
            C_surface=C_surface, C_initial=C_initial,
            x_max_m=x_max_m,
        )
        x_mm = [v * 1000.0 for v in data["x_m"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_mm, data["concentration"], color="#1565C0", linewidth=2.0)
        ax.axhline(data["case_threshold"], color="#D84315", linestyle="--",
                   linewidth=1.2, label=f"Case threshold ({data['case_threshold']:.3g})")
        ax.axvline(data["case_depth_mm"], color="#2E7D32", linestyle=":",
                   linewidth=1.5, label=f"Case depth = {data['case_depth_mm']:.3f} mm")
        ax.set_xlabel("Depth [mm]", fontsize=11)
        ax.set_ylabel("Concentration", fontsize=11)
        ax.set_title(
            f"Diffusion Profile  D={D:.2e} m²/s  t={time_s:.1f} s", fontsize=12
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_arrhenius_diffusivity(
        self,
        D0: float,
        activation_energy_J_mol: float,
        T_min_K: float = 500.0,
        T_max_K: float = 1500.0,
        preset: str | None = None,
    ) -> str:
        """Return base64 PNG of Arrhenius plot (log D vs 1/T)."""
        data = self.arrhenius_diffusivity(
            D0=D0, activation_energy_J_mol=activation_energy_J_mol,
            T_min_K=T_min_K, T_max_K=T_max_K, preset=preset,
        )
        T_arr = np.array(data["temperatures_K"])
        D_arr = np.array(data["diffusivities_m2_s"])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(1000.0 / T_arr, D_arr, color="#6A1B9A", linewidth=2.0)
        ax.set_xlabel("1000 / T  [K⁻¹]", fontsize=11)
        ax.set_ylabel("D  [m² s⁻¹]", fontsize=11)
        ax.set_title(
            f"Arrhenius Diffusivity  Q={data['Q_kJ_mol']:.1f} kJ/mol  D₀={data['D0_m2_s']:.2e}",
            fontsize=12,
        )
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)
