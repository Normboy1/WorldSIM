"""Forging simulation engine for WorldSim Materials.

Physics covered
---------------
- Flow stress (Johnson-Cook, Hollomon, Voce models)
- Hall-Petch yield strength from grain size
- Work hardening / Voce saturation
- Dynamic recrystallization (Zener-Hollomon + JMAK)
- Forging force and press capacity estimate
- Hot-forging processing map (Prasad efficiency + instability)
- Strain-rate sensitivity (m-value)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Material presets
# ---------------------------------------------------------------------------
# Each entry: (sigma_0_MPa, k_HP_MPa_mm^0.5, E_GPa, nu,
#              K_MPa, n_exp,          # Hollomon K*eps^n
#              C_JC, m_JC, T_melt_K, # Johnson-Cook thermal/rate params
#              A_JC, B_JC,            # JC A, B
#              Q_drx_J_mol,           # DRX activation energy
#              a_drx, b_drx,          # DRX grain-size constants
#              rho_kg_m3)
FORGING_PRESETS: dict[str, dict] = {
    "AISI_1020": dict(
        sigma_0=150.0, k_HP=0.60, E=200.0, nu=0.29,
        K=530.0, n_exp=0.26, A_JC=350.0, B_JC=275.0, C_JC=0.022, m_JC=1.0,
        T_melt=1793.0, T_ref=293.0,
        Q_drx=312_000, a_drx=2.8e4, b_drx=-0.15,
        rho=7870.0, name="AISI 1020 Plain Carbon Steel",
    ),
    "AISI_4340": dict(
        sigma_0=470.0, k_HP=0.72, E=205.0, nu=0.29,
        K=1470.0, n_exp=0.15, A_JC=792.0, B_JC=510.0, C_JC=0.014, m_JC=1.03,
        T_melt=1793.0, T_ref=293.0,
        Q_drx=325_000, a_drx=2.1e4, b_drx=-0.17,
        rho=7850.0, name="AISI 4340 Alloy Steel",
    ),
    "Al_6061": dict(
        sigma_0=55.0, k_HP=0.07, E=68.9, nu=0.33,
        K=397.0, n_exp=0.20, A_JC=324.0, B_JC=114.0, C_JC=0.002, m_JC=1.34,
        T_melt=925.0, T_ref=293.0,
        Q_drx=156_000, a_drx=8.0e3, b_drx=-0.20,
        rho=2700.0, name="Aluminium 6061",
    ),
    "Ti6Al4V": dict(
        sigma_0=880.0, k_HP=0.40, E=114.0, nu=0.34,
        K=1355.0, n_exp=0.10, A_JC=1098.0, B_JC=1092.0, C_JC=0.014, m_JC=1.10,
        T_melt=1933.0, T_ref=293.0,
        Q_drx=204_000, a_drx=5.5e3, b_drx=-0.18,
        rho=4430.0, name="Ti-6Al-4V",
    ),
    "Inconel_718": dict(
        sigma_0=1034.0, k_HP=0.65, E=200.0, nu=0.29,
        K=1700.0, n_exp=0.08, A_JC=1241.0, B_JC=622.0, C_JC=0.0134, m_JC=1.3,
        T_melt=1609.0, T_ref=293.0,
        Q_drx=400_000, a_drx=3.0e4, b_drx=-0.22,
        rho=8190.0, name="Inconel 718 Nickel Superalloy",
    ),
}

R_GAS = 8.31446  # J/mol/K


class ForgingEngine:
    """Physics-based forging simulation engine."""

    def _get_preset(self, material: str) -> dict:
        p = FORGING_PRESETS.get(material)
        if p is None:
            raise ValueError(
                f"Unknown material '{material}'. "
                f"Available: {list(FORGING_PRESETS)}"
            )
        return p

    # ------------------------------------------------------------------
    # Flow stress models
    # ------------------------------------------------------------------

    def flow_stress(
        self,
        material: str = "AISI_4340",
        strain_max: float = 0.8,
        strain_rate: float = 1.0,
        temperature_C: float = 20.0,
        model: str = "johnson_cook",
        n_points: int = 200,
    ) -> dict:
        """Compute flow stress vs plastic strain curve.

        Models
        ------
        - ``johnson_cook``: σ = (A+B·εⁿ)(1+C·ln(ε̇/ε̇₀))(1−T*^m)
        - ``hollomon``:     σ = K·εⁿ  (isothermal, rate-independent)
        - ``voce``:         σ = σ_sat − (σ_sat−σ_0)·exp(−ε/ε_c)
        """
        p = self._get_preset(material)
        eps = np.linspace(1e-4, float(strain_max), int(n_points))
        T_K = float(temperature_C) + 273.15
        edot = float(strain_rate)

        if model == "johnson_cook":
            A, B, n = p["A_JC"], p["B_JC"], p["n_exp"]
            C_jc, m = p["C_JC"], p["m_JC"]
            edot0 = 1.0
            T_H = max(0.0, (T_K - p["T_ref"]) / (p["T_melt"] - p["T_ref"]))
            T_H = min(T_H, 0.999)
            rate_term = 1.0 + C_jc * math.log(max(edot / edot0, 1e-6))
            thermal_term = (1.0 - T_H ** m)
            sigma = (A + B * eps ** n) * rate_term * thermal_term

        elif model == "hollomon":
            sigma = p["K"] * np.maximum(eps, 1e-6) ** p["n_exp"]

        elif model == "voce":
            sigma_0 = p["sigma_0"]
            sigma_sat = p["K"]
            eps_c = 0.2
            sigma = sigma_sat - (sigma_sat - sigma_0) * np.exp(-eps / eps_c)

        else:
            raise ValueError(f"Unknown model '{model}'. Use: johnson_cook, hollomon, voce")

        sigma = np.maximum(sigma, 0.0)
        return {
            "material": p["name"],
            "model": model,
            "temperature_C": float(temperature_C),
            "strain_rate_1_s": float(strain_rate),
            "strain": eps.tolist(),
            "flow_stress_MPa": sigma.tolist(),
            "sigma_initial_MPa": float(sigma[0]),
            "sigma_final_MPa": float(sigma[-1]),
            "UTS_proxy_MPa": float(sigma.max()),
        }

    # ------------------------------------------------------------------
    # Hall-Petch yield strength
    # ------------------------------------------------------------------

    def hall_petch(
        self,
        material: str = "AISI_4340",
        grain_size_um: float = 50.0,
    ) -> dict:
        """Compute yield strength from grain size via Hall-Petch.

        σ_y = σ_0 + k_HP / √d   (d in mm)
        """
        p = self._get_preset(material)
        d_mm = float(grain_size_um) / 1000.0
        sigma_y = p["sigma_0"] + p["k_HP"] / math.sqrt(d_mm)

        # grain size sweep 1–500 µm
        d_sweep_um = np.linspace(1.0, 500.0, 300)
        d_sweep_mm = d_sweep_um / 1000.0
        sigma_sweep = p["sigma_0"] + p["k_HP"] / np.sqrt(d_sweep_mm)

        return {
            "material": p["name"],
            "grain_size_um": float(grain_size_um),
            "sigma_0_MPa": p["sigma_0"],
            "k_HP": p["k_HP"],
            "yield_strength_MPa": float(sigma_y),
            "grain_size_sweep_um": d_sweep_um.tolist(),
            "yield_strength_sweep_MPa": sigma_sweep.tolist(),
        }

    # ------------------------------------------------------------------
    # Dynamic recrystallization
    # ------------------------------------------------------------------

    def dynamic_recrystallization(
        self,
        material: str = "AISI_4340",
        temperature_C: float = 1100.0,
        strain_rate: float = 1.0,
        initial_grain_um: float = 100.0,
        strain_max: float = 1.5,
        n_points: int = 300,
    ) -> dict:
        """Model dynamic recrystallization (DRX) during hot deformation.

        Uses Zener-Hollomon parameter Z = ε̇·exp(Q/RT) and JMAK kinetics:
          ε_c  = 0.8 · ε_peak      (critical strain for DRX onset)
          ε_p  = a·d₀^0.15·Z^0.12  (peak strain)
          X_drx = 1 − exp(−0.693·((ε−ε_c)/ε₅₀)²)
          d_rx  = a_drx · Z^b_drx   (recrystallised grain size [µm])
        """
        p = self._get_preset(material)
        T_K = float(temperature_C) + 273.15
        edot = float(strain_rate)
        d0 = float(initial_grain_um)

        Z = edot * math.exp(p["Q_drx"] / (R_GAS * T_K))

        # Peak strain (empirical power-law fit to hot-torsion data)
        eps_p = 6.97e-4 * (d0 ** 0.15) * (Z ** 0.12)
        eps_c = 0.8 * eps_p
        eps_50 = 1.5 * eps_p  # strain at 50% recrystallisation

        # Recrystallised grain size
        d_rx = max(1.0, p["a_drx"] * Z ** p["b_drx"])

        eps = np.linspace(0.0, float(strain_max), int(n_points))
        X = np.where(
            eps < eps_c,
            0.0,
            1.0 - np.exp(-0.693 * ((eps - eps_c) / max(eps_50 - eps_c, 1e-9)) ** 2),
        )
        X = np.clip(X, 0.0, 1.0)

        # Average grain size (mixture rule)
        d_avg = X * d_rx + (1.0 - X) * d0

        return {
            "material": p["name"],
            "temperature_C": float(temperature_C),
            "strain_rate_1_s": float(strain_rate),
            "initial_grain_um": float(d0),
            "Z_parameter": float(Z),
            "eps_critical": float(eps_c),
            "eps_peak": float(eps_p),
            "drx_grain_size_um": float(d_rx),
            "strain": eps.tolist(),
            "X_recrystallized": X.tolist(),
            "avg_grain_size_um": d_avg.tolist(),
        }

    # ------------------------------------------------------------------
    # Forging force
    # ------------------------------------------------------------------

    def forging_force(
        self,
        material: str = "AISI_4340",
        billet_diameter_mm: float = 100.0,
        final_height_mm: float = 40.0,
        original_height_mm: float = 100.0,
        temperature_C: float = 1100.0,
        strain_rate: float = 1.0,
        friction_coeff: float = 0.15,
    ) -> dict:
        """Estimate open-die forging press force.

        F = σ_f · A · (1 + 2μD/(3h))
        where σ_f is the flow stress at the forging condition.
        """
        p = self._get_preset(material)
        T_K = float(temperature_C) + 273.15
        edot = float(strain_rate)
        D = float(billet_diameter_mm)
        h = float(final_height_mm)
        h0 = float(original_height_mm)

        # True strain
        eps_true = math.log(h0 / h) if h > 0 else 0.0

        # Flow stress at forging condition (Johnson-Cook)
        A, B, n = p["A_JC"], p["B_JC"], p["n_exp"]
        C_jc, m = p["C_JC"], p["m_JC"]
        T_H = max(0.0, min((T_K - p["T_ref"]) / (p["T_melt"] - p["T_ref"]), 0.999))
        rate_term = 1.0 + C_jc * math.log(max(edot, 1e-6))
        sigma_f = (A + B * max(eps_true, 1e-4) ** n) * rate_term * (1.0 - T_H ** m)
        sigma_f = max(sigma_f, 1.0)

        # Contact area (volume conservation: D_f from V = π/4 * D² * h)
        V_mm3 = math.pi / 4.0 * D ** 2 * h0
        D_f = math.sqrt(4.0 * V_mm3 / (math.pi * h))
        A_mm2 = math.pi / 4.0 * D_f ** 2
        A_m2 = A_mm2 * 1e-6

        mu = float(friction_coeff)
        friction_factor = 1.0 + (2.0 * mu * D_f) / (3.0 * h)

        F_N = sigma_f * 1e6 * A_m2 * friction_factor
        F_MN = F_N / 1e6

        # Force vs reduction curve
        h_arr = np.linspace(h0 * 0.05, h0, 100)
        F_arr = []
        for h_i in h_arr:
            eps_i = math.log(h0 / h_i)
            sf_i = (A + B * max(eps_i, 1e-4) ** n) * rate_term * (1.0 - T_H ** m)
            sf_i = max(sf_i, 1.0)
            D_i = math.sqrt(4.0 * V_mm3 / (math.pi * h_i))
            A_i = math.pi / 4.0 * D_i ** 2 * 1e-6
            ff_i = 1.0 + (2.0 * mu * D_i) / (3.0 * h_i)
            F_arr.append(sf_i * 1e6 * A_i * ff_i / 1e6)

        reduction_pct = np.linspace(5.0, 100.0, 100)

        return {
            "material": p["name"],
            "temperature_C": float(temperature_C),
            "strain_rate_1_s": float(strain_rate),
            "true_strain": float(eps_true),
            "flow_stress_MPa": float(sigma_f),
            "contact_diameter_mm": float(D_f),
            "contact_area_mm2": float(A_mm2),
            "friction_factor": float(friction_factor),
            "forging_force_MN": float(F_MN),
            "forging_force_tonnes": float(F_MN * 101.97),
            "reduction_pct": reduction_pct.tolist(),
            "force_vs_reduction_MN": F_arr,
        }

    # ------------------------------------------------------------------
    # Hot forging processing map
    # ------------------------------------------------------------------

    def processing_map(
        self,
        material: str = "AISI_4340",
        temp_range_C: tuple[float, float] = (900.0, 1200.0),
        strain_rate_range: tuple[float, float] = (0.001, 100.0),
        strain: float = 0.5,
        n_T: int = 30,
        n_edot: int = 30,
    ) -> dict:
        """Generate Prasad hot-working processing map.

        Power dissipation efficiency: η = 2m/(m+1)  where m = ∂ln σ/∂ln ε̇
        Instability criterion (Ziegler): ξ = ∂ln(m/(m+1))/∂ln ε̇ + m < 0

        Safe workability regions: η > 0.3 AND ξ > 0.
        """
        p = self._get_preset(material)
        eps = max(float(strain), 1e-4)
        T_arr = np.linspace(float(temp_range_C[0]), float(temp_range_C[1]), int(n_T))
        edot_arr = np.logspace(
            math.log10(max(float(strain_rate_range[0]), 1e-4)),
            math.log10(float(strain_rate_range[1])),
            int(n_edot),
        )

        def _sigma(T_C, edot):
            T_K = T_C + 273.15
            A, B, n = p["A_JC"], p["B_JC"], p["n_exp"]
            C_jc, m = p["C_JC"], p["m_JC"]
            T_H = max(0.0, min((T_K - p["T_ref"]) / (p["T_melt"] - p["T_ref"]), 0.999))
            rate = 1.0 + C_jc * math.log(max(edot, 1e-9))
            return max((A + B * eps ** n) * rate * (1.0 - T_H ** m), 1.0)

        delta = 0.01
        eta_grid = np.zeros((n_T, n_edot))
        xi_grid = np.zeros((n_T, n_edot))

        for i, T in enumerate(T_arr):
            for j, ed in enumerate(edot_arr):
                ed_lo = ed * (1.0 - delta)
                ed_hi = ed * (1.0 + delta)
                s_lo = _sigma(T, ed_lo)
                s_hi = _sigma(T, ed_hi)
                m_val = (math.log(s_hi) - math.log(s_lo)) / (math.log(ed_hi) - math.log(ed_lo))
                m_val = max(min(m_val, 1.0), 0.001)
                eta = 2.0 * m_val / (m_val + 1.0)

                # Ziegler instability: approximate ξ
                ed_lo2 = ed_lo * (1.0 - delta)
                ed_hi2 = ed_hi * (1.0 + delta)
                s_lo2 = _sigma(T, ed_lo2)
                s_hi2 = _sigma(T, ed_hi2)
                m_lo = max((math.log(_sigma(T, ed_lo)) - math.log(s_lo2)) / (math.log(ed_lo) - math.log(ed_lo2)), 0.001)
                m_hi = max((math.log(s_hi2) - math.log(_sigma(T, ed_hi))) / (math.log(ed_hi2) - math.log(ed_hi)), 0.001)
                dm_dlog_edot = (math.log(m_hi / (m_hi + 1)) - math.log(m_lo / (m_lo + 1))) / (math.log(ed_hi) - math.log(ed_lo))
                xi = dm_dlog_edot + m_val

                eta_grid[i, j] = eta
                xi_grid[i, j] = xi

        safe = ((eta_grid > 0.30) & (xi_grid > 0.0)).astype(float)

        return {
            "material": p["name"],
            "strain": float(strain),
            "temperatures_C": T_arr.tolist(),
            "strain_rates_1_s": edot_arr.tolist(),
            "efficiency_eta": eta_grid.tolist(),
            "instability_xi": xi_grid.tolist(),
            "safe_window": safe.tolist(),
            "optimal_T_C": float(T_arr[int(np.argmax(eta_grid.max(axis=1)))]),
            "optimal_edot": float(edot_arr[int(np.argmax(eta_grid.max(axis=0)))]),
            "max_efficiency": float(eta_grid.max()),
        }

    # ------------------------------------------------------------------
    # Open-die forging — pass schedule
    # ------------------------------------------------------------------

    def open_die_pass_schedule(
        self,
        material: str = "AISI_4340",
        billet_length_mm: float = 300.0,
        billet_width_mm: float = 150.0,
        billet_height_mm: float = 150.0,
        target_thickness_mm: float = 50.0,
        temperature_C: float = 1100.0,
        strain_rate: float = 1.0,
        pass_reduction_pct: float = 15.0,
        friction_coeff: float = 0.25,
    ) -> dict:
        """Multi-pass open-die flat forging schedule.

        Each pass:
          - True strain per pass: ε = ln(h_in / h_out)
          - Spread (Townsend): b_out = b_in · (1 + 0.12·(h_in/b_in))
          - Elongation from volume conservation: L_out = V / (b_out · h_out)
          - Penetration: P = 1 − exp(−3.8·h/b)  (fraction through thickness)
          - Forging force per pass: F = σ_f·A·(1+2μD/(3h))

        Reheat assumed every 3 passes for steel/Ti (every 5 for Al).
        """
        p = self._get_preset(material)
        T_K = float(temperature_C) + 273.15
        edot = float(strain_rate)
        mu = float(friction_coeff)

        A_jc, B_jc, n = p["A_JC"], p["B_JC"], p["n_exp"]
        C_jc, m_jc = p["C_JC"], p["m_JC"]
        T_H = max(0.0, min((T_K - p["T_ref"]) / (p["T_melt"] - p["T_ref"]), 0.999))
        rate_term = 1.0 + C_jc * math.log(max(edot, 1e-6))
        thermal_term = 1.0 - T_H ** m_jc

        def _sigma_f(eps):
            return max((A_jc + B_jc * max(eps, 1e-4) ** n) * rate_term * thermal_term, 1.0)

        reheat_interval = 3 if p["T_melt"] > 1400 else 5

        h = float(billet_height_mm)
        b = float(billet_width_mm)
        L = float(billet_length_mm)
        V_mm3 = h * b * L
        h_target = float(target_thickness_mm)
        red = float(pass_reduction_pct) / 100.0

        passes = []
        cumulative_strain = 0.0
        pass_num = 0
        reheats = 0

        while h > h_target * 1.01:
            pass_num += 1
            h_out = max(h * (1.0 - red), h_target)
            eps_pass = math.log(h / h_out)
            cumulative_strain += eps_pass

            # Spread (Townsend formula)
            b_out = b * (1.0 + 0.12 * (h / b))
            L_out = V_mm3 / (b_out * h_out)

            # Contact width (flat die)
            contact_length = L_out / 4.0   # bite per stroke ≈ 25% of length
            contact_area_mm2 = contact_length * b_out
            contact_area_m2 = contact_area_mm2 * 1e-6

            sigma_f = _sigma_f(cumulative_strain)
            friction_factor = 1.0 + (2.0 * mu * b_out) / (3.0 * h_out)
            F_MN = sigma_f * 1e6 * contact_area_m2 * friction_factor / 1e6

            penetration = 1.0 - math.exp(-3.8 * h_out / max(b_out, 1e-3))

            reheat = (pass_num % reheat_interval == 0) and (h_out > h_target * 1.01)
            if reheat:
                reheats += 1

            passes.append({
                "pass": pass_num,
                "h_in_mm": round(h, 2),
                "h_out_mm": round(h_out, 2),
                "b_out_mm": round(b_out, 2),
                "L_out_mm": round(L_out, 2),
                "eps_pass": round(eps_pass, 4),
                "cumulative_strain": round(cumulative_strain, 4),
                "flow_stress_MPa": round(sigma_f, 1),
                "force_MN": round(F_MN, 3),
                "penetration_pct": round(penetration * 100.0, 1),
                "reheat_after": reheat,
            })

            h, b, L = h_out, b_out, L_out

        # Grain refinement estimate via Hall-Petch from DRX grain size
        Z = edot * math.exp(p["Q_drx"] / (R_GAS * T_K))
        d_rx = max(1.0, p["a_drx"] * Z ** p["b_drx"])

        return {
            "material": p["name"],
            "process": "open-die flat forging",
            "temperature_C": float(temperature_C),
            "pass_reduction_pct": float(pass_reduction_pct),
            "billet_initial_mm": [float(billet_length_mm), float(billet_width_mm), float(billet_height_mm)],
            "billet_final_mm": [round(L, 2), round(b, 2), round(h, 2)],
            "target_thickness_mm": float(target_thickness_mm),
            "total_passes": pass_num,
            "total_reheats": reheats,
            "total_true_strain": round(cumulative_strain, 4),
            "final_drx_grain_um": round(d_rx, 2),
            "max_force_MN": round(max(ps["force_MN"] for ps in passes), 3),
            "pass_schedule": passes,
        }

    # ------------------------------------------------------------------
    # Closed-die forging — flash, fill and trimming analysis
    # ------------------------------------------------------------------

    def closed_die_analysis(
        self,
        material: str = "AISI_4340",
        part_volume_mm3: float = 250_000.0,
        projected_area_mm2: float = 12_000.0,
        flash_thickness_mm: float = 3.0,
        flash_width_mm: float = 12.0,
        temperature_C: float = 1100.0,
        strain_rate: float = 5.0,
        die_fill_index: float = 0.95,
    ) -> dict:
        """Closed-die (impression-die) forging analysis.

        Flash design
        ------------
        - Flash land ratio:  R = w_f / t_f
        - Flash pressure:    p_f = σ_f · (1 + R/4)     [MPa]
        - Flash volume:      V_f = A_flash · t_f·1.5    (gutter + land)
        - Billet volume:     V_billet = V_part + V_f

        Total forging force
        -------------------
        F = F_part + F_flash
        F_part  = σ_f · A_part  · (1 + μ·r_eq / (3·h_avg))
        F_flash = p_f · A_flash

        Trimming force
        --------------
        F_trim = τ_shear · π · d_part · t_f
        where τ_shear ≈ 0.6·σ_f and d_part ≈ √(4·A_part/π)

        Energy
        ------
        W ≈ 0.65 · F_total · stroke   (form factor 0.65)
        """
        p = self._get_preset(material)
        T_K = float(temperature_C) + 273.15
        edot = float(strain_rate)

        A_jc, B_jc, n = p["A_JC"], p["B_JC"], p["n_exp"]
        C_jc, m_jc = p["C_JC"], p["m_JC"]
        T_H = max(0.0, min((T_K - p["T_ref"]) / (p["T_melt"] - p["T_ref"]), 0.999))
        rate_term = 1.0 + C_jc * math.log(max(edot, 1e-6))
        sigma_f = max((A_jc + B_jc * 0.3 ** n) * rate_term * (1.0 - T_H ** m_jc), 1.0)

        t_f = float(flash_thickness_mm)
        w_f = float(flash_width_mm)
        A_part = float(projected_area_mm2)
        V_part = float(part_volume_mm3)
        mu = 0.12

        R = w_f / max(t_f, 0.1)
        p_flash = sigma_f * (1.0 + R / 4.0)

        # Flash geometry (perimeter × land width)
        d_eq = math.sqrt(4.0 * A_part / math.pi)
        perimeter = math.pi * d_eq
        A_flash_mm2 = perimeter * w_f   # flash land area

        V_flash_mm3 = A_flash_mm2 * t_f * 1.5   # includes gutter
        V_billet_mm3 = V_part + V_flash_mm3

        # Part force
        h_avg = math.sqrt(A_part / math.pi) * 0.4   # rough average height
        r_eq = d_eq / 2.0
        F_part_N = sigma_f * 1e6 * A_part * 1e-6 * (1.0 + mu * r_eq / (3.0 * max(h_avg, 1.0)))
        F_flash_N = p_flash * 1e6 * A_flash_mm2 * 1e-6
        F_total_MN = (F_part_N + F_flash_N) / 1e6

        # Trimming
        tau_shear = 0.6 * sigma_f
        F_trim_N = tau_shear * 1e6 * (math.pi * d_eq * 1e-3) * (t_f * 1e-3)
        F_trim_MN = F_trim_N / 1e6
        F_trim_tonnes = F_trim_MN * 101.97

        # Energy estimate
        stroke_mm = math.sqrt(V_part / A_part) * 1.5
        W_kJ = 0.65 * F_total_MN * 1e6 * stroke_mm * 1e-3 / 1e3

        # Die fill check
        fill_pct = float(die_fill_index) * 100.0
        fill_ok = fill_pct >= 95.0

        # Draft angle recommendation (empirical)
        if sigma_f < 500.0:
            draft_deg = 3.0
        elif sigma_f < 1000.0:
            draft_deg = 5.0
        else:
            draft_deg = 7.0

        # Press capacity sweep
        reductions = np.linspace(10.0, 90.0, 80)
        forces_sweep = []
        for red in reductions:
            T_H_i = T_H
            eps_i = math.log(100.0 / (100.0 - red))
            sf_i = max((A_jc + B_jc * max(eps_i, 1e-4) ** n) * rate_term * (1.0 - T_H_i ** m_jc), 1.0)
            pf_i = sf_i * (1.0 + R / 4.0)
            F_i = (sf_i * 1e6 * A_part * 1e-6 * 1.1 + pf_i * 1e6 * A_flash_mm2 * 1e-6) / 1e6
            forces_sweep.append(round(F_i, 4))

        return {
            "material": p["name"],
            "process": "closed-die (impression-die) forging",
            "temperature_C": float(temperature_C),
            "strain_rate_1_s": float(strain_rate),
            "flow_stress_MPa": round(sigma_f, 1),
            "flash_thickness_mm": float(t_f),
            "flash_width_mm": float(w_f),
            "flash_land_ratio_R": round(R, 2),
            "flash_pressure_MPa": round(p_flash, 1),
            "part_projected_area_mm2": float(A_part),
            "flash_area_mm2": round(A_flash_mm2, 1),
            "billet_volume_mm3": round(V_billet_mm3, 0),
            "billet_mass_kg": round(V_billet_mm3 * 1e-6 * p["rho"], 3),
            "forging_force_part_MN": round(F_part_N / 1e6, 3),
            "forging_force_flash_MN": round(F_flash_N / 1e6, 3),
            "total_forging_force_MN": round(F_total_MN, 3),
            "total_forging_force_tonnes": round(F_total_MN * 101.97, 0),
            "trimming_force_MN": round(F_trim_MN, 3),
            "trimming_force_tonnes": round(F_trim_tonnes, 0),
            "energy_kJ": round(W_kJ, 1),
            "die_fill_pct": round(fill_pct, 1),
            "die_fill_ok": fill_ok,
            "recommended_draft_deg": float(draft_deg),
            "reduction_pct_sweep": reductions.tolist(),
            "force_sweep_MN": forces_sweep,
        }

    # ------------------------------------------------------------------
    # Strain-rate sensitivity
    # ------------------------------------------------------------------

    def strain_rate_sensitivity(
        self,
        material: str = "AISI_4340",
        temperature_C: float = 1100.0,
        strain: float = 0.5,
        strain_rate_range: tuple[float, float] = (0.001, 100.0),
        n_points: int = 60,
    ) -> dict:
        """Compute m-value (strain-rate sensitivity) across strain-rate range.

        m = ∂ln σ / ∂ln ε̇   (higher m = better superplastic formability)
        """
        p = self._get_preset(material)
        T_K = float(temperature_C) + 273.15
        eps = max(float(strain), 1e-4)
        edot_arr = np.logspace(
            math.log10(max(float(strain_rate_range[0]), 1e-4)),
            math.log10(float(strain_rate_range[1])),
            int(n_points),
        )

        A, B, n = p["A_JC"], p["B_JC"], p["n_exp"]
        C_jc, m_jc = p["C_JC"], p["m_JC"]
        T_H = max(0.0, min((T_K - p["T_ref"]) / (p["T_melt"] - p["T_ref"]), 0.999))
        thermal = (1.0 - T_H ** m_jc)
        base = (A + B * eps ** n) * thermal

        sigma_arr = base * (1.0 + C_jc * np.log(np.maximum(edot_arr, 1e-9)))
        sigma_arr = np.maximum(sigma_arr, 1.0)

        # local m-value: centred finite difference
        d_log_sigma = np.gradient(np.log(sigma_arr))
        d_log_edot = np.gradient(np.log(edot_arr))
        m_arr = d_log_sigma / np.where(d_log_edot != 0, d_log_edot, 1e-12)

        return {
            "material": p["name"],
            "temperature_C": float(temperature_C),
            "strain": float(strain),
            "strain_rates_1_s": edot_arr.tolist(),
            "flow_stress_MPa": sigma_arr.tolist(),
            "m_value": m_arr.tolist(),
            "mean_m_value": float(m_arr.mean()),
            "superplastic": bool(float(m_arr.mean()) > 0.5),
        }
