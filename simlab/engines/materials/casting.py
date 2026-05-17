"""Casting simulation engine for WorldSim Materials.

Physics covered
---------------
- Solidification time (Chvorinov's rule)
- Cooling curve with latent heat (enthalpy method)
- Dendrite arm spacing (SDAS) vs cooling rate
- Scheil microsegregation (solute redistribution)
- Niyama criterion for shrinkage porosity
- Hot tearing susceptibility (Clyne-Davies RDG proxy)
- Mould filling time (gate velocity + Bernoulli)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Alloy presets
# ---------------------------------------------------------------------------
# Keys: T_liq_K, T_sol_K, L_f_J_kg (latent heat), rho (kg/m³),
#       cp_J_kgK, k_W_mK (conductivity), T_pour_offset_K (superheat),
#       k_part (partition coeff for primary solute), C0 (wt%), solute_name,
#       shrinkage_pct, B_chvorinov (s/mm²), SDAS_a, SDAS_n,
#       hot_tear_susceptibility (0-1), name
CASTING_PRESETS: dict[str, dict] = {
    "A356_Al": dict(
        T_liq=888.0+273.15, T_sol=555.0+273.15,
        L_f=3.97e5, rho=2680.0, cp=963.0, k_cond=151.0,
        T_pour_offset=50.0, k_part=0.13, C0=7.0, solute="Si",
        shrinkage_pct=3.5, B_chvorinov=1.35,
        SDAS_a=50.0, SDAS_n=0.33,
        hot_tear=0.25, name="A356 Aluminium Casting Alloy",
    ),
    "Gray_Iron": dict(
        T_liq=1175.0+273.15, T_sol=1145.0+273.15,
        L_f=2.47e5, rho=7150.0, cp=544.0, k_cond=51.0,
        T_pour_offset=100.0, k_part=0.30, C0=3.2, solute="C",
        shrinkage_pct=1.0, B_chvorinov=3.20,
        SDAS_a=25.0, SDAS_n=0.40,
        hot_tear=0.10, name="Gray Cast Iron FC-300",
    ),
    "316L_SS": dict(
        T_liq=1390.0+273.15, T_sol=1370.0+273.15,
        L_f=2.72e5, rho=7990.0, cp=500.0, k_cond=16.3,
        T_pour_offset=80.0, k_part=0.77, C0=2.0, solute="Cr",
        shrinkage_pct=2.5, B_chvorinov=2.85,
        SDAS_a=14.0, SDAS_n=0.40,
        hot_tear=0.55, name="316L Austenitic Stainless Steel",
    ),
    "Copper_C110": dict(
        T_liq=1083.0+273.15, T_sol=1065.0+273.15,
        L_f=2.05e5, rho=8940.0, cp=386.0, k_cond=391.0,
        T_pour_offset=60.0, k_part=0.20, C0=0.3, solute="O",
        shrinkage_pct=4.9, B_chvorinov=0.80,
        SDAS_a=30.0, SDAS_n=0.35,
        hot_tear=0.15, name="Copper C110 ETP",
    ),
    "Ti6Al4V_cast": dict(
        T_liq=1660.0+273.15, T_sol=1604.0+273.15,
        L_f=2.86e5, rho=4420.0, cp=526.0, k_cond=6.7,
        T_pour_offset=100.0, k_part=0.87, C0=6.0, solute="Al",
        shrinkage_pct=2.8, B_chvorinov=3.50,
        SDAS_a=40.0, SDAS_n=0.38,
        hot_tear=0.40, name="Ti-6Al-4V Cast",
    ),
}


class CastingEngine:
    """Physics-based casting simulation engine."""

    def _get_preset(self, alloy: str) -> dict:
        p = CASTING_PRESETS.get(alloy)
        if p is None:
            raise ValueError(
                f"Unknown alloy '{alloy}'. Available: {list(CASTING_PRESETS)}"
            )
        return p

    # ------------------------------------------------------------------
    # Solidification time — Chvorinov's rule
    # ------------------------------------------------------------------

    def solidification_time(
        self,
        alloy: str = "A356_Al",
        volume_cm3: float = 500.0,
        surface_area_cm2: float = 300.0,
    ) -> dict:
        """Estimate solidification time via Chvorinov's rule.

        t_s = B · (V/A)²   where V/A is the modulus [cm]
        """
        p = self._get_preset(alloy)
        V = float(volume_cm3)
        A = float(surface_area_cm2)
        modulus = V / A  # cm

        t_s = p["B_chvorinov"] * modulus ** 2  # seconds

        # Riser design: riser modulus must be > casting modulus
        riser_modulus_needed = modulus * 1.2
        riser_volume = (riser_modulus_needed ** 3) * math.pi * (4.0 / 3.0)  # sphere V = (4/3)πr³

        # Shape comparison
        shapes = {
            "sphere":      (4.0 / 3.0) * math.pi * modulus ** 3 / (4.0 * math.pi * modulus ** 2),
            "cylinder_h=d": math.pi * modulus ** 2 * (2 * modulus) / (2 * math.pi * modulus ** 2 + 2 * math.pi * modulus * 2 * modulus),
            "cube":        (2 * modulus) ** 3 / (6 * (2 * modulus) ** 2),
        }

        return {
            "alloy": p["name"],
            "volume_cm3": float(V),
            "surface_area_cm2": float(A),
            "casting_modulus_cm": float(modulus),
            "B_chvorinov_s_cm2": p["B_chvorinov"],
            "solidification_time_s": float(t_s),
            "solidification_time_min": float(t_s / 60.0),
            "riser_modulus_needed_cm": float(riser_modulus_needed),
            "shape_moduli_cm": shapes,
        }

    # ------------------------------------------------------------------
    # Cooling curve (enthalpy method with latent heat release)
    # ------------------------------------------------------------------

    def cooling_curve(
        self,
        alloy: str = "A356_Al",
        mould_material: str = "sand",
        volume_cm3: float = 500.0,
        surface_area_cm2: float = 300.0,
        t_total_s: float = None,
        n_points: int = 500,
    ) -> dict:
        """Simulate temperature vs time including latent heat plateau.

        Uses a lumped-capacitance model with enthalpy:
            ρ·V·dH/dt = −h_eff·A·(T − T_mould)
        where H incorporates L_f released uniformly across the mushy zone.
        """
        p = self._get_preset(alloy)

        # Effective heat transfer coefficients [W/m²K]
        h_eff = {"sand": 400.0, "metal": 3000.0, "investment": 250.0, "die": 5000.0}.get(
            mould_material, 400.0
        )
        T_mould = 25.0 + 273.15  # K

        V_m3 = float(volume_cm3) * 1e-6
        A_m2 = float(surface_area_cm2) * 1e-4

        # Time to solidify estimate for total time
        t_s_est = p["B_chvorinov"] * (float(volume_cm3) / float(surface_area_cm2)) ** 2
        t_total = float(t_total_s) if t_total_s else t_s_est * 6.0
        dt = t_total / int(n_points)

        T_pour = p["T_liq"] + p["T_pour_offset"]
        T_liq, T_sol = p["T_liq"], p["T_sol"]
        L_f = p["L_f"]
        rho, cp = p["rho"], p["cp"]
        dT_mushy = max(T_liq - T_sol, 1.0)

        T = T_pour
        times, temps = [], []
        solidified_fraction = []
        f_s = 0.0

        for i in range(int(n_points)):
            t = i * dt
            times.append(t)
            temps.append(T - 273.15)

            # Enthalpy capacity: in mushy zone add latent heat contribution
            if T > T_liq:
                C_eff = rho * cp
            elif T > T_sol:
                C_eff = rho * (cp + L_f / dT_mushy)
                f_s = (T_liq - T) / dT_mushy
            else:
                C_eff = rho * cp
                f_s = 1.0
            solidified_fraction.append(float(f_s))

            dT = -(h_eff * A_m2 * (T - T_mould)) / (C_eff * V_m3) * dt
            T = max(T + dT, T_mould)

        # Find solidification time index
        fs_arr = np.array(solidified_fraction)
        t_arr = np.array(times)
        t_liq_idx = int(np.argmax(np.array(temps) < (T_liq - 273.15))) if any(t < T_liq - 273.15 for t in temps) else 0
        t_sol_idx = int(np.argmax(fs_arr >= 0.99)) if any(fs_arr >= 0.99) else -1

        return {
            "alloy": p["name"],
            "mould_material": mould_material,
            "h_eff_W_m2K": h_eff,
            "T_pour_C": float(T_pour - 273.15),
            "T_liquidus_C": float(T_liq - 273.15),
            "T_solidus_C": float(T_sol - 273.15),
            "time_s": times,
            "temperature_C": temps,
            "solid_fraction": solidified_fraction,
            "solidification_time_s": float(times[t_sol_idx]) if t_sol_idx > 0 else None,
            "cooling_rate_C_s": float(abs(temps[0] - temps[-1]) / t_total),
        }

    # ------------------------------------------------------------------
    # Dendrite arm spacing
    # ------------------------------------------------------------------

    def dendrite_arm_spacing(
        self,
        alloy: str = "A356_Al",
        cooling_rate_range: tuple[float, float] = (0.1, 1000.0),
        n_points: int = 200,
    ) -> dict:
        """Secondary dendrite arm spacing (SDAS) vs cooling rate.

        λ₂ = a · (dT/dt)^(−n)   [µm]
        """
        p = self._get_preset(alloy)
        cr = np.logspace(
            math.log10(float(cooling_rate_range[0])),
            math.log10(float(cooling_rate_range[1])),
            int(n_points),
        )
        sdas = p["SDAS_a"] * cr ** (-p["SDAS_n"])

        # Zone labels
        zones = []
        for c in cr:
            if c < 1.0:
                zones.append("sand cast")
            elif c < 10.0:
                zones.append("gravity die")
            elif c < 100.0:
                zones.append("pressure die cast")
            else:
                zones.append("rapid solidification")

        return {
            "alloy": p["name"],
            "a_coeff": p["SDAS_a"],
            "n_exponent": p["SDAS_n"],
            "cooling_rates_C_s": cr.tolist(),
            "SDAS_um": sdas.tolist(),
            "process_zones": zones,
            "SDAS_at_1_C_s": float(p["SDAS_a"]),
            "SDAS_at_100_C_s": float(p["SDAS_a"] * 100 ** (-p["SDAS_n"])),
        }

    # ------------------------------------------------------------------
    # Scheil microsegregation
    # ------------------------------------------------------------------

    def scheil_microsegregation(
        self,
        alloy: str = "A356_Al",
        n_points: int = 500,
    ) -> dict:
        """Scheil-Gulliver solute redistribution during solidification.

        C_L = C₀ · (1 − f_s)^(k−1)
        C_S = k · C_L
        Assumes no diffusion in solid, complete mixing in liquid.
        """
        p = self._get_preset(alloy)
        k = p["k_part"]
        C0 = p["C0"]

        f_s = np.linspace(0.001, 0.999, int(n_points))
        C_L = C0 * (1.0 - f_s) ** (k - 1.0)
        C_S = k * C_L

        # Eutectic fraction estimate (where C_L hits eutectic composition)
        C_eutectic = {"Si": 12.6, "C": 4.3, "Cr": 100.0, "O": 0.8, "Al": 30.0}.get(
            p["solute"], C0 * 8.0
        )
        if C_eutectic > C0:
            f_eutectic = 1.0 - ((C_eutectic / C0) ** (1.0 / (k - 1.0))) if k < 1.0 else None
        else:
            f_eutectic = None

        # Segregation ratio
        C_S_max = float(C_S[np.argmax(C_S)])
        seg_ratio = C_S_max / max(k * C0, 1e-9)

        return {
            "alloy": p["name"],
            "solute": p["solute"],
            "C0_wt_pct": float(C0),
            "k_partition": float(k),
            "solid_fraction": f_s.tolist(),
            "C_liquid_wt_pct": C_L.tolist(),
            "C_solid_wt_pct": C_S.tolist(),
            "max_solid_concentration_wt_pct": C_S_max,
            "segregation_ratio": float(seg_ratio),
            "eutectic_fraction": float(f_eutectic) if f_eutectic else None,
        }

    # ------------------------------------------------------------------
    # Niyama criterion — shrinkage porosity
    # ------------------------------------------------------------------

    def niyama_criterion(
        self,
        alloy: str = "A356_Al",
        thermal_gradient_K_m: float = 5000.0,
        cooling_rate_K_s: float = 2.0,
    ) -> dict:
        """Niyama criterion for shrinkage porosity prediction.

        Ny = G / √(dT/dt)   [(K·s)^0.5 / mm]
        Ny < 1 → high porosity risk
        Ny 1–3 → moderate risk
        Ny > 3 → low risk (sound casting)
        """
        p = self._get_preset(alloy)
        G = float(thermal_gradient_K_m) / 1000.0   # K/mm
        cr = float(cooling_rate_K_s)

        Ny = G / math.sqrt(max(cr, 1e-9))

        if Ny < 1.0:
            risk = "HIGH — macro/micro shrinkage likely"
        elif Ny < 3.0:
            risk = "MODERATE — micro-shrinkage possible"
        else:
            risk = "LOW — sound casting expected"

        # Sweep over G values
        G_sweep = np.linspace(500.0, 50000.0, 200) / 1000.0   # K/mm
        Ny_sweep = G_sweep / math.sqrt(max(cr, 1e-9))

        return {
            "alloy": p["name"],
            "thermal_gradient_K_mm": float(G),
            "cooling_rate_K_s": float(cr),
            "niyama_value": float(Ny),
            "porosity_risk": risk,
            "G_sweep_K_mm": (G_sweep * 1000.0).tolist(),
            "Ny_sweep": Ny_sweep.tolist(),
            "shrinkage_pct": p["shrinkage_pct"],
        }

    # ------------------------------------------------------------------
    # Hot tearing susceptibility
    # ------------------------------------------------------------------

    def hot_tearing(
        self,
        alloy: str = "316L_SS",
        cooling_rate_K_s: float = 5.0,
        mould_constraint: str = "medium",
    ) -> dict:
        """Assess hot tearing susceptibility using a Clyne-Davies-style index.

        HTS = t_v / (t_v + t_r)
          t_v = vulnerable time in mushy zone (0.9 > f_s > 0.99)
          t_r = recovery time (f_s < 0.9)
        Constraint multiplier: low=0.7, medium=1.0, high=1.4
        """
        p = self._get_preset(alloy)
        cr = max(float(cooling_rate_K_s), 0.01)
        T_liq, T_sol = p["T_liq"], p["T_sol"]
        dT_total = T_liq - T_sol

        constraint = {"low": 0.7, "medium": 1.0, "high": 1.4}.get(mould_constraint, 1.0)

        # Scheil: f_s(T) via Newton on T = T_liq - dT*(1-(1-fs)^(1-k)/...)
        # Simpler: linear Scheil approximation for temperature
        k = p["k_part"]

        def T_at_fs(fs):
            return T_liq - dT_total * (1.0 - (1.0 - fs) ** (1.0 / max(1.0 - k, 0.01)))

        T_90 = T_at_fs(0.90)
        T_99 = T_at_fs(0.99)

        t_v = abs(T_99 - T_90) / cr          # seconds
        t_r = abs(T_liq - T_90) / cr

        HTS_base = t_v / max(t_v + t_r, 1e-9) * constraint

        # HTS scale: <0.1 low, 0.1-0.4 moderate, >0.4 high
        if HTS_base < 0.1:
            susceptibility = "LOW"
        elif HTS_base < 0.4:
            susceptibility = "MODERATE"
        else:
            susceptibility = "HIGH — hot tearing likely"

        # Alloy comparison
        comparisons = {}
        for a_name, a_data in CASTING_PRESETS.items():
            k_a = a_data["k_part"]
            dT_a = a_data["T_liq"] - a_data["T_sol"]
            T90_a = a_data["T_liq"] - dT_a * (1.0 - (1.0 - 0.90) ** (1.0 / max(1.0 - k_a, 0.01)))
            T99_a = a_data["T_liq"] - dT_a * (1.0 - (1.0 - 0.99) ** (1.0 / max(1.0 - k_a, 0.01)))
            tv_a = abs(T99_a - T90_a) / cr
            tr_a = abs(a_data["T_liq"] - T90_a) / cr
            comparisons[a_name] = round(tv_a / max(tv_a + tr_a, 1e-9), 3)

        return {
            "alloy": p["name"],
            "mould_constraint": mould_constraint,
            "cooling_rate_K_s": float(cr),
            "k_partition": float(k),
            "T_vulnerable_start_C": float(T_90 - 273.15),
            "T_vulnerable_end_C": float(T_99 - 273.15),
            "t_vulnerable_s": float(t_v),
            "t_recovery_s": float(t_r),
            "HTS_index": float(HTS_base),
            "susceptibility": susceptibility,
            "alloy_comparison_HTS": comparisons,
            "preset_susceptibility": p["hot_tear"],
        }

    # ------------------------------------------------------------------
    # Investment casting — ceramic shell, grain structure, fluidity
    # ------------------------------------------------------------------

    def investment_casting(
        self,
        alloy: str = "Ti6Al4V_cast",
        wall_thickness_mm: float = 2.0,
        part_volume_cm3: float = 50.0,
        surface_area_cm2: float = 120.0,
        vacuum: bool = True,
    ) -> dict:
        """Investment (lost-wax) casting process analysis.

        Shell thickness   t_sh = max(6, 2.5 · √(V/A) · f_T)
        Minimum wall      w_min = f(fluidity, T_pour−T_liq, alloy viscosity)
        Superheat needed  ΔT_sh = base + thermal_conductivity penalty
        Grain structure   depends on cooling rate and alloy type
        Fluidity index    FL = v_gate·t_flow  [cm] — empirical alloy rating

        Burnout schedule  dewax: 150°C, burnout: 700°C, preheat: alloy-specific
        """
        p = self._get_preset(alloy)

        # Shell thickness (rule of thumb + thermal mass factor)
        modulus = float(part_volume_cm3) / float(surface_area_cm2)
        T_factor = 1.0 + (p["T_liq"] - 1300.0) / 3000.0   # thicker shell for higher T
        t_shell_mm = max(6.0, 2.5 * modulus * 10.0 * T_factor)   # cm → mm

        # Superheat requirement (higher k → more superheat needed)
        k_cond = p["k_cond"]
        superheat_C = 80.0 + k_cond * 0.3 + (1.0 / max(float(wall_thickness_mm), 0.5)) * 20.0
        T_pour_C = p["T_liq"] - 273.15 + superheat_C

        # Minimum achievable wall thickness
        # Based on fluidity: high T alloys / low k → thinner walls possible
        fl_index = (p["T_pour_offset"] / 100.0) * (100.0 / max(k_cond, 1.0)) ** 0.3
        w_min_mm = max(0.3, 1.5 / max(fl_index, 0.1))
        wall_ok = float(wall_thickness_mm) >= w_min_mm

        # Shell preheat temperature (prevents thermal shock)
        if p["T_liq"] > 1800.0:
            preheat_C = 1200.0
        elif p["T_liq"] > 1500.0:
            preheat_C = 1000.0
        elif p["T_liq"] > 1200.0:
            preheat_C = 700.0
        else:
            preheat_C = 400.0

        # Grain structure prediction
        cooling_rate_est = k_cond * (T_pour_C - 25.0) / (p["rho"] * p["cp"] * modulus * 0.01)
        if cooling_rate_est > 50.0:
            grain_structure = "fine equiaxed"
        elif cooling_rate_est > 5.0:
            grain_structure = "mixed columnar/equiaxed"
        else:
            grain_structure = "coarse columnar (directional possible)"

        # Solidification time in ceramic shell
        h_eff = 250.0 if not vacuum else 180.0   # lower h_eff in vacuum (no convection)
        V_m3 = float(part_volume_cm3) * 1e-6
        A_m2 = float(surface_area_cm2) * 1e-4
        dT_solidification = p["T_liq"] - p["T_sol"]
        t_sol = (p["rho"] * V_m3 * (p["cp"] * dT_solidification + p["L_f"])) / (h_eff * A_m2 * (T_pour_C - 25.0))

        # Wax pattern volume (same as part + 1.2% shrinkage allowance)
        V_wax_cm3 = float(part_volume_cm3) * (1.0 + p["shrinkage_pct"] / 100.0 * 1.2)

        # Ceramic shell layers (each ~0.5 mm)
        n_layers = math.ceil(t_shell_mm / 0.5)

        # Cost index (relative, 1=cheapest)
        cost_index = 1.0 + (n_layers / 10.0) + (1.0 if vacuum else 0.0) + (p["T_liq"] / 5000.0)

        return {
            "alloy": p["name"],
            "process": "investment (lost-wax) casting",
            "vacuum": vacuum,
            "wall_thickness_mm": float(wall_thickness_mm),
            "min_wall_thickness_mm": round(w_min_mm, 2),
            "wall_thickness_ok": wall_ok,
            "fluidity_index": round(fl_index, 3),
            "shell_thickness_mm": round(t_shell_mm, 1),
            "n_shell_layers": int(n_layers),
            "T_pour_C": round(T_pour_C, 1),
            "superheat_C": round(superheat_C, 1),
            "T_dewax_C": 150.0,
            "T_burnout_C": 700.0,
            "T_preheat_shell_C": preheat_C,
            "solidification_time_s": round(t_sol, 1),
            "grain_structure": grain_structure,
            "cooling_rate_estimate_C_s": round(cooling_rate_est, 2),
            "wax_pattern_volume_cm3": round(V_wax_cm3, 2),
            "shrinkage_allowance_pct": p["shrinkage_pct"],
            "cost_index_relative": round(cost_index, 2),
        }

    # ------------------------------------------------------------------
    # High-pressure die casting — injection, fill, solidification
    # ------------------------------------------------------------------

    def die_casting(
        self,
        alloy: str = "A356_Al",
        part_volume_cm3: float = 200.0,
        projected_area_mm2: float = 18_000.0,
        gate_area_mm2: float = 400.0,
        plunger_velocity_m_s: float = 3.0,
        intensification_pressure_MPa: float = 70.0,
    ) -> dict:
        """High-pressure die casting (HPDC) process analysis.

        Fill dynamics
        -------------
        Gate velocity  v_g = A_plunger/A_gate · v_plunger
        Fill time      t_f = V_part / (A_gate · v_g)

        Clamping force
        --------------
        F_clamp = p_intens · A_proj  [MN]

        Solidification in die
        ----------------------
        h_die ≈ 5000 W/m²K (water-cooled steel die)
        t_sol via lumped capacitance

        Porosity risk
        -------------
        Reynolds: Re = ρ·v_g·(A_gate)^0.5 / η_visc
        Re > 5000 → turbulent fill → high porosity risk

        Typical HPDC alloys: A356 Al, Copper C110 (only Al/Mg/Zn in practice;
        flagged as warning for high-T alloys)
        """
        p = self._get_preset(alloy)

        # Only Al/Mg/Zn are typical HPDC alloys
        hpdc_suitable = p["T_liq"] < 1200.0
        warning = "" if hpdc_suitable else (
            "WARNING: alloy liquidus >1200°C — conventional HPDC dies not suitable; "
            "consider investment casting or squeeze casting instead."
        )

        # Plunger / gate area ratio (assume 80 mm diameter plunger)
        d_plunger_mm = 80.0
        A_plunger_mm2 = math.pi / 4.0 * d_plunger_mm ** 2
        A_gate = float(gate_area_mm2)

        v_plunger = float(plunger_velocity_m_s)
        v_gate = (A_plunger_mm2 / A_gate) * v_plunger * 1000.0   # mm/s → we keep in m/s
        v_gate_m_s = (A_plunger_mm2 / A_gate) * v_plunger

        V_cm3 = float(part_volume_cm3)
        V_m3 = V_cm3 * 1e-6
        A_gate_m2 = A_gate * 1e-6

        t_fill_s = V_m3 / (A_gate_m2 * v_gate_m_s)

        # Clamping force
        p_int = float(intensification_pressure_MPa)
        A_proj_m2 = float(projected_area_mm2) * 1e-6
        F_clamp_MN = p_int * 1e6 * A_proj_m2 / 1e6
        F_clamp_tonnes = F_clamp_MN * 101.97

        # Solidification time (water-cooled die, h_die = 5000 W/m²K)
        h_die = 5000.0
        T_die = 200.0 + 273.15   # K, typical die temperature
        T_pour = p["T_liq"] + p["T_pour_offset"]
        A_m2_part = V_m3 ** (2.0 / 3.0) * 6.0   # cube approximation
        dT_solidify = p["T_liq"] - p["T_sol"]
        H_total = p["rho"] * (p["cp"] * dT_solidify + p["L_f"])
        t_sol = H_total * V_m3 / (h_die * A_m2_part * ((T_pour + p["T_liq"]) / 2.0 - T_die))

        # Thermal check: fill time must be < solidification onset
        t_solid_onset = (p["rho"] * V_m3 * p["cp"] * float(p["T_pour_offset"])) / (h_die * A_m2_part * float(p["T_pour_offset"]))
        fill_before_freeze = t_fill_s < t_solid_onset

        # Reynolds-based porosity risk
        eta_visc = 1.5e-3   # Pa·s (typical liquid Al viscosity)
        Re = p["rho"] * v_gate_m_s * math.sqrt(A_gate_m2) / eta_visc
        if Re < 2000.0:
            flow_regime = "laminar — low porosity risk"
        elif Re < 5000.0:
            flow_regime = "transitional — moderate porosity risk"
        else:
            flow_regime = "turbulent — HIGH porosity risk (air entrapment)"

        # Cycle time (fill + solidify + ejection 5s)
        cycle_time_s = t_fill_s + t_sol + 5.0

        # Shot weight
        shot_mass_kg = V_cm3 * 1e-6 * p["rho"]

        # Velocity sweep for gate velocity vs porosity correlation
        v_range = np.linspace(10.0, 100.0, 80)   # m/s gate velocity
        Re_range = p["rho"] * v_range * math.sqrt(A_gate_m2) / eta_visc
        porosity_risk_idx = np.clip((Re_range - 2000.0) / 8000.0, 0.0, 1.0)

        return {
            "alloy": p["name"],
            "process": "high-pressure die casting (HPDC)",
            "hpdc_suitable": hpdc_suitable,
            "warning": warning,
            "gate_velocity_m_s": round(v_gate_m_s, 2),
            "fill_time_ms": round(t_fill_s * 1000.0, 2),
            "solidification_time_s": round(t_sol, 3),
            "fill_before_freeze": fill_before_freeze,
            "reynolds_number": round(Re, 0),
            "flow_regime": flow_regime,
            "intensification_pressure_MPa": float(p_int),
            "clamping_force_MN": round(F_clamp_MN, 2),
            "clamping_force_tonnes": round(F_clamp_tonnes, 0),
            "cycle_time_s": round(cycle_time_s, 2),
            "shot_mass_kg": round(shot_mass_kg, 3),
            "gate_velocity_range_m_s": v_range.tolist(),
            "porosity_risk_index": porosity_risk_idx.tolist(),
        }

    # ------------------------------------------------------------------
    # Application case studies
    # ------------------------------------------------------------------

    def application_case(
        self,
        application: str = "engine_block_al",
    ) -> dict:
        """Run a pre-defined engineering casting application case.

        Applications
        ------------
        - ``engine_block_al``     : A356 Al automotive engine block, sand cast
        - ``turbine_blade_ni``    : Inconel 718 turbine blade, investment cast (vacuum)
        - ``wheel_hub_iron``      : Gray Iron wheel hub, green sand cast
        - ``structural_bracket_ti``: Ti-6Al-4V aerospace bracket, investment cast (vacuum)
        - ``heat_exchanger_cu``   : Copper C110 heat exchanger body, die cast

        Returns a combined dict of solidification, Niyama, hot tearing,
        SDAS, and process-specific analysis for that application.
        """
        _cases = {
            "engine_block_al": dict(
                alloy="A356_Al", process="sand_cast",
                volume_cm3=4500.0, surface_area_cm2=1800.0,
                mould="sand", wall_mm=6.0, vacuum=False,
                G_K_m=3000.0, cooling_rate_K_s=1.5,
                description="Automotive 4-cylinder engine block (A356 Al, sand cast)",
            ),
            "turbine_blade_ni": dict(
                alloy="Ti6Al4V_cast", process="investment_vacuum",
                volume_cm3=45.0, surface_area_cm2=95.0,
                mould="investment", wall_mm=1.8, vacuum=True,
                G_K_m=12000.0, cooling_rate_K_s=8.0,
                description="Aerospace turbine blade (Ti-6Al-4V, vacuum investment cast)",
            ),
            "wheel_hub_iron": dict(
                alloy="Gray_Iron", process="sand_cast",
                volume_cm3=3200.0, surface_area_cm2=1400.0,
                mould="sand", wall_mm=12.0, vacuum=False,
                G_K_m=2500.0, cooling_rate_K_s=0.8,
                description="Automotive wheel hub (Gray Iron FC-300, green sand)",
            ),
            "structural_bracket_ti": dict(
                alloy="Ti6Al4V_cast", process="investment_vacuum",
                volume_cm3=180.0, surface_area_cm2=320.0,
                mould="investment", wall_mm=3.5, vacuum=True,
                G_K_m=8000.0, cooling_rate_K_s=4.0,
                description="Aerospace structural bracket (Ti-6Al-4V, vacuum investment)",
            ),
            "heat_exchanger_cu": dict(
                alloy="Copper_C110", process="die_cast",
                volume_cm3=380.0, surface_area_cm2=600.0,
                mould="die", wall_mm=4.0, vacuum=False,
                G_K_m=15000.0, cooling_rate_K_s=20.0,
                description="Industrial heat exchanger body (Copper C110, die cast)",
            ),
        }
        case = _cases.get(application)
        if case is None:
            raise ValueError(
                f"Unknown application '{application}'. "
                f"Available: {list(_cases)}"
            )

        alloy = case["alloy"]
        p = self._get_preset(alloy)

        # Solidification time
        sol = self.solidification_time(alloy, case["volume_cm3"], case["surface_area_cm2"])

        # Cooling curve
        cool = self.cooling_curve(alloy, case["mould"], case["volume_cm3"], case["surface_area_cm2"])
        cooling_rate_actual = cool["cooling_rate_C_s"]

        # SDAS at actual cooling rate
        sdas_val = p["SDAS_a"] * max(cooling_rate_actual, 0.01) ** (-p["SDAS_n"])

        # Niyama
        niy = self.niyama_criterion(alloy, case["G_K_m"], case["cooling_rate_K_s"])

        # Hot tearing
        ht = self.hot_tearing(alloy, case["cooling_rate_K_s"], "medium")

        # Scheil
        sch = self.scheil_microsegregation(alloy)

        # Process-specific
        if case["process"].startswith("investment"):
            proc_data = self.investment_casting(
                alloy, case["wall_mm"], case["volume_cm3"],
                case["surface_area_cm2"], case["vacuum"]
            )
        elif case["process"] == "die_cast":
            proc_data = self.die_casting(
                alloy, case["volume_cm3"],
                projected_area_mm2=case["surface_area_cm2"] * 100.0 * 0.3,
            )
        else:
            proc_data = {"process": "sand casting", "mould": case["mould"]}

        return {
            "application": case["description"],
            "alloy": p["name"],
            "solidification_time_s": sol["solidification_time_s"],
            "solidification_time_min": sol["solidification_time_min"],
            "casting_modulus_cm": sol["casting_modulus_cm"],
            "cooling_rate_C_s": round(cooling_rate_actual, 3),
            "sdas_um": round(sdas_val, 1),
            "niyama_value": niy["niyama_value"],
            "porosity_risk": niy["porosity_risk"],
            "hot_tearing_index": ht["HTS_index"],
            "hot_tearing_susceptibility": ht["susceptibility"],
            "scheil_segregation_ratio": sch["segregation_ratio"],
            "scheil_max_solid_conc_wt_pct": sch["max_solid_concentration_wt_pct"],
            "process_details": proc_data,
        }

    # ------------------------------------------------------------------
    # Mould filling time
    # ------------------------------------------------------------------

    def mould_filling(
        self,
        alloy: str = "A356_Al",
        mould_volume_cm3: float = 1000.0,
        gate_area_cm2: float = 2.0,
        sprue_height_cm: float = 20.0,
        friction_loss: float = 0.45,
    ) -> dict:
        """Estimate mould filling time using Bernoulli gate velocity.

        v_gate = C_d · √(2gh)
        t_fill = V / (A_gate · v_gate)
        """
        g = 9810.0  # mm/s² → use cm/s² = 981
        g_cm = 981.0
        C_d = 1.0 - float(friction_loss)

        h = float(sprue_height_cm)
        v_gate = C_d * math.sqrt(2.0 * g_cm * h)   # cm/s

        V = float(mould_volume_cm3)
        A = float(gate_area_cm2)

        t_fill = V / (A * v_gate)

        # Thermal check: must fill before metal cools below T_liq
        p = self._get_preset(alloy)
        T_pour = p["T_liq"] + p["T_pour_offset"]
        # crude estimate: max fill time for 20°C superheat loss
        t_max = 20.0 / max(p["L_f"] * p["rho"] * 1e-6 / (p["cp"] * 0.1), 0.1)

        return {
            "alloy": p["name"],
            "mould_volume_cm3": float(V),
            "gate_area_cm2": float(A),
            "sprue_height_cm": float(h),
            "discharge_coeff": float(C_d),
            "gate_velocity_cm_s": float(v_gate),
            "fill_time_s": float(t_fill),
            "fill_ok": t_fill < 60.0,
            "T_pour_C": float(T_pour - 273.15),
            "note": "Use ≥2 gates and taper sprue to prevent aspiration.",
        }
