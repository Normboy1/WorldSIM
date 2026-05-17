"""Oxidation kinetics and chemical stability engine.

Implements:
- Wagner parabolic oxidation: x² = 2*kp*t (mass transport through growing scale)
- Linear oxidation: x = kl*t (interface-reaction or breakaway controlled)
- Mixed parabolic-linear (Deal-Grove model)
- Ellingham diagram: ΔG vs T for common metal oxides per mole O₂
- Pilling-Bedworth ratio (protective scale formation criterion)
- Simplified Pourbaix stability regions (E-pH diagram)
- Corrosion electrochemical assessment (pitting potential, SCC risk index)
"""

from __future__ import annotations

import math
import base64
import io
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

R_GAS = 8.31446261815324       # J / (mol K)
F_FARADAY = 96485.33212        # C / mol


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Ellingham diagram data  (ΔG = a + b*T  in kJ / mol O₂)
# Source: Richardson/Jeffes approximations (widely used teaching values)
# Negative = oxide is stable; more negative = harder to reduce
# ---------------------------------------------------------------------------
_ELLINGHAM: dict[str, tuple[float, float, str]] = {
    "2Mg + O₂ → 2MgO":         (-1139.0,  0.205, "Mg"),
    "4/3Al + O₂ → 2/3Al₂O₃":  (-1118.0,  0.210, "Al"),
    "Ti + O₂ → TiO₂":         ( -944.0,  0.180, "Ti"),
    "Si + O₂ → SiO₂":         ( -856.0,  0.173, "Si"),
    "4/3Cr + O₂ → 2/3Cr₂O₃":  ( -748.0,  0.175, "Cr"),
    "2Mn + O₂ → 2MnO":         ( -730.0,  0.145, "Mn"),
    "2Fe + O₂ → 2FeO":         ( -528.0,  0.130, "Fe"),
    "4/3Fe + O₂ → 2/3Fe₂O₃":  ( -544.0,  0.170, "Fe₂O₃"),
    "2Ni + O₂ → 2NiO":         ( -490.0,  0.195, "Ni"),
    "2Co + O₂ → 2CoO":         ( -470.0,  0.190, "Co"),
    "2Cu + O₂ → 2CuO":         ( -310.0,  0.141, "Cu"),
    "2Mo + O₂ → 2MoO":         ( -360.0,  0.135, "Mo"),
    "W + O₂ → WO₂":            ( -520.0,  0.165, "W"),
}

# Pilling-Bedworth ratio data: (Mw_oxide g/mol, Mw_metal g/mol, n_M, rho_oxide g/cm³, rho_metal g/cm³)
# PBR = (Mw_oxide * rho_metal) / (n_M * Mw_metal * rho_oxide)
_PBR_DATA: dict[str, tuple[float, float, int, float, float]] = {
    "Al":  (101.96, 26.98, 2, 3.99, 2.70),   # Al₂O₃
    "Cr":  (152.0,  52.0,  2, 5.22, 7.19),   # Cr₂O₃
    "Fe":  (71.85,  55.85, 1, 5.74, 7.87),   # FeO
    "Fe2O3": (159.7, 55.85, 2, 5.24, 7.87),  # Fe₂O₃
    "Ni":  (74.71,  58.69, 1, 6.67, 8.91),   # NiO
    "Ti":  (79.87,  47.87, 1, 4.23, 4.51),   # TiO₂
    "Cu":  (79.55,  63.55, 1, 6.31, 8.96),   # CuO
    "Mg":  (40.30,  24.31, 1, 3.58, 1.74),   # MgO
    "Si":  (60.09,  28.09, 1, 2.65, 2.33),   # SiO₂
    "Zn":  (81.41,  65.38, 1, 5.61, 7.14),   # ZnO
}

# Oxidation rate constant presets: (kp0 m²/s, Q_kp J/mol, regime, notes)
_OXIDATION_PRESETS: dict[str, tuple[float, float, str, str]] = {
    "Fe_air":            (1.2e-7, 167_000.0, "parabolic",
                          "Iron oxidation in air 700–1200 K"),
    "Fe_Cr18_air":       (4.0e-11, 195_000.0, "parabolic",
                          "18Cr stainless steel, protective Cr₂O₃ scale"),
    "Ni_air":            (5.0e-12, 172_000.0, "parabolic",
                          "Nickel oxidation in air 700–1100 K"),
    "Al_air":            (1.3e-15, 148_000.0, "parabolic",
                          "Aluminium, thin protective Al₂O₃, very slow"),
    "Ti_air":            (8.0e-9,  182_000.0, "mixed",
                          "Titanium — TiO₂ protective up to ~850 K"),
}


class OxidationStabilityEngine:
    """Oxidation kinetics and electrochemical corrosion stability models."""

    # ------------------------------------------------------------------
    # 1. Oxidation kinetics (parabolic / linear / mixed)
    # ------------------------------------------------------------------

    def oxidation_kinetics(
        self,
        kp_m2_s: float | None = None,
        kl_m_s: float | None = None,
        time_s: float = 3600.0,
        n_points: int = 120,
        temperature_K: float | None = None,
        kp0_m2_s: float | None = None,
        Q_kp_J_mol: float | None = None,
        kl0_m_s: float | None = None,
        Q_kl_J_mol: float | None = None,
        preset: str | None = None,
        regime: str = "parabolic",
    ) -> dict:
        """Compute oxide film thickness vs time.

        Regimes:
        - 'parabolic': x² = 2*kp*t  (Wagner — diffusion through growing scale)
        - 'linear'   : x = kl*t     (interface-reaction controlled or breakaway)
        - 'mixed'    : x²/(2*kp) + x/kl = t  (Deal-Grove model)

        Parameters
        ----------
        kp_m2_s    : parabolic rate constant [m²/s]
        kl_m_s     : linear rate constant [m/s]
        temperature_K  : if provided with Arrhenius params, overrides kp/kl
        preset     : built-in material preset key
        """
        if preset is not None:
            if preset not in _OXIDATION_PRESETS:
                raise ValueError(
                    f"Unknown oxidation preset {preset!r}. Available: {sorted(_OXIDATION_PRESETS)}"
                )
            kp0_val, Q_val, regime, _ = _OXIDATION_PRESETS[preset]
            if temperature_K is not None:
                kp_m2_s = kp0_val * math.exp(-Q_val / (R_GAS * float(temperature_K)))
            else:
                kp_m2_s = kp0_val  # use pre-exponential as placeholder

        if temperature_K is not None:
            if kp0_m2_s is not None and Q_kp_J_mol is not None:
                kp_m2_s = float(kp0_m2_s) * math.exp(-float(Q_kp_J_mol) / (R_GAS * float(temperature_K)))
            if kl0_m_s is not None and Q_kl_J_mol is not None:
                kl_m_s = float(kl0_m_s) * math.exp(-float(Q_kl_J_mol) / (R_GAS * float(temperature_K)))

        if kp_m2_s is None and kl_m_s is None:
            raise ValueError("Supply kp_m2_s (parabolic) and/or kl_m_s (linear), or a preset + temperature.")

        if kp_m2_s is not None and float(kp_m2_s) < 0:
            raise ValueError("kp_m2_s must be non-negative.")
        if kl_m_s is not None and float(kl_m_s) < 0:
            raise ValueError("kl_m_s must be non-negative.")
        if time_s <= 0:
            raise ValueError("time_s must be positive.")

        t = np.linspace(0.0, float(time_s), int(n_points))
        kp = float(kp_m2_s) if kp_m2_s is not None else 0.0
        kl = float(kl_m_s) if kl_m_s is not None else 0.0

        if regime == "parabolic" or (kl == 0.0 and kp > 0.0):
            x = np.sqrt(np.maximum(2.0 * kp * t, 0.0))
            regime_used = "parabolic"
        elif regime == "linear" or (kp == 0.0 and kl > 0.0):
            x = kl * t
            regime_used = "linear"
        else:
            # Deal-Grove: x²/(2*kp) + x/kl = t → solve quadratic for each t
            # x² + (2*kp/kl)*x - 2*kp*t = 0
            a_coef = 1.0
            b_coef = 2.0 * kp / kl if kl > 0 else 0.0
            x = 0.5 * (-b_coef + np.sqrt(b_coef ** 2 + 8.0 * kp * t))
            regime_used = "mixed"

        x_final = float(x[-1])
        x_um = x * 1e6  # metres → µm

        # Mass gain estimate (oxide density ~ 5 g/cm³ typical for Fe₂O₃/Cr₂O₃)
        rho_oxide = 5000.0  # kg/m³
        mass_gain_mg_cm2 = x_final * rho_oxide * 1e-1  # kg/m² → mg/cm²

        result = {
            "model": f"Oxidation kinetics — {regime_used} rate law",
            "regime": regime_used,
            "kp_m2_s": float(kp) if kp > 0 else None,
            "kl_m_s": float(kl) if kl > 0 else None,
            "time_s": float(time_s),
            "t_s": t.tolist(),
            "oxide_thickness_um": x_um.tolist(),
            "final_thickness_um": float(x_final * 1e6),
            "final_mass_gain_mg_cm2_proxy": float(mass_gain_mg_cm2),
        }
        if temperature_K is not None:
            result["temperature_K"] = float(temperature_K)
        return result

    # ------------------------------------------------------------------
    # 2. Ellingham diagram
    # ------------------------------------------------------------------

    def ellingham_diagram(
        self,
        T_min_K: float = 300.0,
        T_max_K: float = 2000.0,
        species: list[str] | None = None,
        n_points: int = 80,
    ) -> dict:
        """Compute ΔG° vs T for metal oxide formation reactions.

        ΔG = a + b*T  [kJ / mol O₂]  (Richardson-Jeffes linear fit)

        Returns tabulated data and a base64 PNG.
        More negative ΔG = more thermodynamically stable oxide.
        """
        T_arr = np.linspace(float(T_min_K), float(T_max_K), int(n_points))

        available = list(_ELLINGHAM.keys())
        if species is not None:
            # Filter by element symbols mentioned
            selected = [rxn for rxn in available
                        if any(s in rxn for s in species)]
            if not selected:
                raise ValueError(
                    f"None of {species!r} found in Ellingham data. "
                    f"Available elements: {sorted({v[2] for v in _ELLINGHAM.values()})}"
                )
        else:
            selected = available

        table: list[dict] = []
        for rxn in selected:
            a, b, element = _ELLINGHAM[rxn]
            dG = a + b * T_arr  # kJ/mol O₂
            # Equilibrium O₂ partial pressure: ln(pO2) = 2*ΔG/(RT) ... per mole O₂
            ln_pO2 = 2.0 * dG * 1000.0 / (R_GAS * T_arr)
            pO2_eq = np.exp(np.clip(ln_pO2, -300, 0))
            table.append({
                "reaction": rxn,
                "element": element,
                "a_kJ_molO2": float(a),
                "b_J_molO2_K": float(b),
                "dG_kJ_molO2_at_Tmin": float(a + b * float(T_min_K)),
                "dG_kJ_molO2_at_Tmax": float(a + b * float(T_max_K)),
                "temperatures_K": T_arr.tolist(),
                "dG_kJ_molO2": dG.tolist(),
                "eq_pO2_atm": pO2_eq.tolist(),
            })

        # Rank by stability at 1000 K
        stability_at_1000K = {r["reaction"]: r["a_kJ_molO2"] + r["b_J_molO2_K"] * 1000.0
                              for r in table}
        ranked = sorted(stability_at_1000K, key=stability_at_1000K.get)

        return {
            "model": "Ellingham-Richardson diagram (ΔG = a + bT, linear fit)",
            "T_min_K": float(T_min_K),
            "T_max_K": float(T_max_K),
            "temperatures_K": T_arr.tolist(),
            "reactions": table,
            "stability_ranking_at_1000K": ranked,
            "notes": [
                "More negative ΔG = more thermodynamically stable oxide.",
                "Linear fit from Richardson-Jeffes — valid approximate trends.",
                "Phase transitions (melting, boiling) cause kinks not captured here.",
            ],
        }

    # ------------------------------------------------------------------
    # 3. Pilling-Bedworth ratio
    # ------------------------------------------------------------------

    def pilling_bedworth_ratio(
        self,
        metal: str | None = None,
        Mw_oxide: float | None = None,
        Mw_metal: float | None = None,
        n_metal_atoms: int = 1,
        rho_oxide_g_cm3: float | None = None,
        rho_metal_g_cm3: float | None = None,
    ) -> dict:
        """Compute Pilling-Bedworth ratio to assess oxide scale protectiveness.

        PBR = V_oxide / V_metal = (Mw_oxide * ρ_M) / (n_M * Mw_M * ρ_oxide)

        PBR < 1      : non-protective (alkali metals — volume too small, scale cracks)
        1.0–2.0      : protective (most structural alloys)
        > 2.0        : tends to spall (compressive stress, non-protective)

        Parameters
        ----------
        metal          : element symbol for built-in data (e.g. 'Fe', 'Al', 'Cr')
        Mw_oxide / Mw_metal / rho_oxide / rho_metal : manual override
        n_metal_atoms  : number of metal atoms per formula unit of oxide
        """
        if metal is not None:
            if metal not in _PBR_DATA:
                raise ValueError(
                    f"Unknown metal {metal!r} for PBR. Built-in: {sorted(_PBR_DATA)}"
                )
            Mw_ox, Mw_m, n_m, rho_ox, rho_m = _PBR_DATA[metal]
        else:
            if any(v is None for v in [Mw_oxide, Mw_metal, rho_oxide_g_cm3, rho_metal_g_cm3]):
                raise ValueError("Supply 'metal' preset or all four manual parameters.")
            Mw_ox = float(Mw_oxide)
            Mw_m = float(Mw_metal)
            n_m = int(n_metal_atoms)
            rho_ox = float(rho_oxide_g_cm3)
            rho_m = float(rho_metal_g_cm3)

        PBR = (Mw_ox * rho_m) / (n_m * Mw_m * rho_ox)

        if PBR < 1.0:
            protectiveness = "non-protective (volume too small — scale cracks/voids)"
        elif PBR <= 2.0:
            protectiveness = "protective (good compressive fit)"
        else:
            protectiveness = "non-protective (volume too large — scale spalls under compression)"

        return {
            "model": "Pilling-Bedworth ratio",
            "metal": metal or "user-defined",
            "Mw_oxide_g_mol": float(Mw_ox),
            "Mw_metal_g_mol": float(Mw_m),
            "n_metal_atoms": int(n_m),
            "rho_oxide_g_cm3": float(rho_ox),
            "rho_metal_g_cm3": float(rho_m),
            "PBR": float(PBR),
            "protectiveness": protectiveness,
            "recommended_range": "1.0–2.0 for protective scales",
        }

    # ------------------------------------------------------------------
    # 4. Corrosion risk — electrochemical (simplified Pourbaix)
    # ------------------------------------------------------------------

    def corrosion_risk(
        self,
        metal: str,
        pH: float,
        electrode_potential_V: float | None = None,
        temperature_K: float = 298.15,
        ion_activity: float = 1e-6,
    ) -> dict:
        """Simplified Pourbaix stability assessment for common engineering metals.

        Uses linear approximations of the E-pH boundary lines from published
        Pourbaix atlas data. Returns stability region and corrosion type.

        Parameters
        ----------
        metal               : element symbol ('Fe', 'Al', 'Cr', 'Ni', 'Cu', 'Ti', 'Zn')
        pH                  : solution pH (0–14)
        electrode_potential_V : electrode potential vs SHE [V] (optional)
        temperature_K       : temperature [K]
        ion_activity        : metal ion activity (default 1e-6 mol/L threshold)
        """
        _METALS = {"Fe", "Al", "Cr", "Ni", "Cu", "Ti", "Zn", "Co"}
        if metal not in _METALS:
            raise ValueError(f"Metal {metal!r} not supported. Available: {sorted(_METALS)}")

        RT_nF = R_GAS * float(temperature_K) / (2.0 * F_FARADAY)  # n=2 typical
        log_a = math.log10(max(float(ion_activity), 1e-20))

        # Simplified boundary lines (linear E-pH): E_boundary = E0 - slope*pH + offset*log_a
        # Derived from Nernst equation for each stability boundary.
        # Format: {metal: (E0_immunity, E0_passivation, passivation_pH_start, passivation_pH_end, E0_transpassive)}
        _BOUNDARIES = {
            "Fe":  {"E_immunity": -0.44 + RT_nF * log_a * 2,
                    "E_passive_start": -0.44,
                    "pH_passive_start": 8.0,
                    "pH_passive_end": 14.0,
                    "E_transpassive": 0.77},
            "Al":  {"E_immunity": -1.66 + RT_nF * log_a * 3,
                    "E_passive_start": -1.30,
                    "pH_passive_start": 4.0,
                    "pH_passive_end": 9.0,
                    "E_transpassive": 2.0},
            "Cr":  {"E_immunity": -0.74 + RT_nF * log_a * 3,
                    "E_passive_start": -0.74,
                    "pH_passive_start": 2.0,
                    "pH_passive_end": 14.0,
                    "E_transpassive": 1.35},
            "Ni":  {"E_immunity": -0.26 + RT_nF * log_a * 2,
                    "E_passive_start": -0.12,
                    "pH_passive_start": 6.0,
                    "pH_passive_end": 14.0,
                    "E_transpassive": 1.70},
            "Cu":  {"E_immunity":  0.34 + RT_nF * log_a * 2,
                    "E_passive_start": 0.34,
                    "pH_passive_start": 6.5,
                    "pH_passive_end": 14.0,
                    "E_transpassive": 1.80},
            "Ti":  {"E_immunity": -0.86 + RT_nF * log_a * 4,
                    "E_passive_start": -0.90,
                    "pH_passive_start": 0.0,
                    "pH_passive_end": 14.0,
                    "E_transpassive": 2.50},
            "Zn":  {"E_immunity": -0.76 + RT_nF * log_a * 2,
                    "E_passive_start": -0.76,
                    "pH_passive_start": 8.5,
                    "pH_passive_end": 12.0,
                    "E_transpassive": 1.20},
            "Co":  {"E_immunity": -0.28 + RT_nF * log_a * 2,
                    "E_passive_start": -0.18,
                    "pH_passive_start": 7.0,
                    "pH_passive_end": 14.0,
                    "E_transpassive": 1.60},
        }

        b = _BOUNDARIES[metal]
        E_ref = float(electrode_potential_V) if electrode_potential_V is not None else 0.0
        pH_v = float(pH)

        # Determine stability region
        in_passive_pH = b["pH_passive_start"] <= pH_v <= b["pH_passive_end"]

        if E_ref < b["E_immunity"]:
            region = "immune"
            description = "Metal is thermodynamically stable (immune from corrosion)."
        elif in_passive_pH and E_ref >= b["E_passive_start"]:
            if E_ref >= b["E_transpassive"]:
                region = "transpassive"
                description = "Above passivation range — dissolution as high-valence species."
            else:
                region = "passive"
                description = "Protective oxide film forms — low corrosion rate expected."
        else:
            region = "active"
            description = "Active dissolution — corrosion expected."

        pitting_chloride_risk = "low"
        if region == "passive" and pH_v < 7.0:
            pitting_chloride_risk = "elevated (low pH destabilizes passive film)"
        elif region == "passive" and pH_v >= 7.0:
            pitting_chloride_risk = "low (stable passive film)"

        return {
            "model": "Simplified Pourbaix stability (linear boundary approximations)",
            "metal": metal,
            "pH": pH_v,
            "electrode_potential_V_SHE": E_ref,
            "temperature_K": float(temperature_K),
            "stability_region": region,
            "description": description,
            "E_immunity_boundary_V": float(b["E_immunity"]),
            "E_passive_start_V": float(b["E_passive_start"]),
            "E_transpassive_V": float(b["E_transpassive"]),
            "passive_pH_window": [float(b["pH_passive_start"]), float(b["pH_passive_end"])],
            "pitting_chloride_risk": pitting_chloride_risk,
            "notes": [
                "Boundaries use simplified linear Nernst approximations from Pourbaix Atlas.",
                "Actual behaviour depends on alloy composition, surface condition, and local chemistry.",
            ],
        }

    # ------------------------------------------------------------------
    # 5. Stress corrosion cracking (SCC) risk index
    # ------------------------------------------------------------------

    def scc_risk_index(
        self,
        material: str,
        environment: str,
        temperature_K: float = 298.15,
        stress_MPa: float = 200.0,
        KI_MPa_sqrtm: float | None = None,
        KIscc_MPa_sqrtm: float | None = None,
    ) -> dict:
        """Qualitative SCC risk index based on material-environment susceptibility.

        Uses a heuristic susceptibility matrix: known SCC pairs are scored
        1.0 (high) to 0.1 (low). Incorporates temperature and stress intensity.

        Parameters
        ----------
        material    : alloy family ('austenitic_stainless', 'duplex_stainless',
                      'low_alloy_steel', 'Al_alloy', 'Ti_alloy', 'Ni_alloy')
        environment : environment keyword ('chloride', 'H2S_sour', 'caustic',
                      'HF', 'high_temp_water', 'H2')
        stress_MPa  : applied stress [MPa]
        KI_MPa_sqrtm     : stress intensity factor (optional) [MPa√m]
        KIscc_MPa_sqrtm  : SCC threshold KI (optional) [MPa√m]
        """
        _SCC_MATRIX: dict[tuple[str, str], float] = {
            ("austenitic_stainless", "chloride"):       0.95,
            ("austenitic_stainless", "caustic"):        0.70,
            ("austenitic_stainless", "H2S_sour"):       0.60,
            ("austenitic_stainless", "high_temp_water"):0.55,
            ("duplex_stainless",     "chloride"):       0.40,
            ("duplex_stainless",     "H2S_sour"):       0.50,
            ("low_alloy_steel",      "H2S_sour"):       0.90,
            ("low_alloy_steel",      "H2"):             0.85,
            ("low_alloy_steel",      "chloride"):       0.35,
            ("Al_alloy",             "chloride"):       0.75,
            ("Al_alloy",             "H2S_sour"):       0.40,
            ("Ti_alloy",             "HF"):             0.90,
            ("Ti_alloy",             "chloride"):       0.20,
            ("Ni_alloy",             "caustic"):        0.60,
            ("Ni_alloy",             "H2S_sour"):       0.45,
            ("Ni_alloy",             "HF"):             0.35,
        }

        base_susceptibility = _SCC_MATRIX.get((material, environment), 0.10)

        # Temperature factor (SCC typically peaks 50–150°C above ambient)
        T_C = float(temperature_K) - 273.15
        if 50.0 <= T_C <= 150.0:
            T_factor = 1.0 + 0.4 * (1.0 - abs(T_C - 100.0) / 50.0)
        elif T_C < 50.0:
            T_factor = 0.5 + 0.01 * T_C
        else:
            T_factor = 1.2

        # Stress factor
        stress_factor = min(1.0 + (float(stress_MPa) - 100.0) / 400.0, 2.0)
        stress_factor = max(0.5, stress_factor)

        risk = min(1.0, base_susceptibility * T_factor * stress_factor)

        KI_above_threshold = None
        if KI_MPa_sqrtm is not None and KIscc_MPa_sqrtm is not None:
            KI_above_threshold = float(KI_MPa_sqrtm) >= float(KIscc_MPa_sqrtm)

        if risk > 0.7:
            severity = "HIGH — use corrosion-resistant alloy or inhibitor"
        elif risk > 0.35:
            severity = "MODERATE — monitor, consider alloy upgrade"
        else:
            severity = "LOW — typical engineering use acceptable"

        return {
            "model": "SCC risk index (heuristic susceptibility matrix)",
            "material": material,
            "environment": environment,
            "temperature_K": float(temperature_K),
            "stress_MPa": float(stress_MPa),
            "base_susceptibility": float(base_susceptibility),
            "temperature_factor": float(T_factor),
            "stress_factor": float(stress_factor),
            "scc_risk_index": float(risk),
            "severity": severity,
            "KI_exceeds_KIscc": KI_above_threshold,
            "notes": [
                "Heuristic matrix — use NACE/ISO standards for engineering decisions.",
                "KISCC values depend strongly on alloy condition, heat treatment, and environment.",
            ],
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_oxidation_kinetics(
        self,
        kp_m2_s: float | None = None,
        kl_m_s: float | None = None,
        time_s: float = 3600.0,
        regime: str = "parabolic",
        temperature_K: float | None = None,
        preset: str | None = None,
    ) -> str:
        """Base64 PNG of oxide thickness vs time."""
        data = self.oxidation_kinetics(
            kp_m2_s=kp_m2_s, kl_m_s=kl_m_s,
            time_s=time_s, regime=regime,
            temperature_K=temperature_K, preset=preset,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["t_s"], data["oxide_thickness_um"],
                color="#B71C1C", linewidth=2.0)
        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel("Oxide thickness [µm]", fontsize=11)
        ax.set_title(
            f"Oxidation Kinetics ({data['regime']})", fontsize=12
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return _fig_to_b64(fig)

    def plot_ellingham(
        self,
        T_min_K: float = 300.0,
        T_max_K: float = 2000.0,
        species: list[str] | None = None,
    ) -> str:
        """Base64 PNG of Ellingham diagram."""
        data = self.ellingham_diagram(
            T_min_K=T_min_K, T_max_K=T_max_K, species=species
        )
        fig, ax = plt.subplots(figsize=(11, 7))
        colors = plt.cm.tab20.colors
        T_arr = np.array(data["temperatures_K"])
        for i, rxn in enumerate(data["reactions"]):
            dG = rxn["dG_kJ_molO2"]
            ax.plot(T_arr, dG, color=colors[i % len(colors)],
                    linewidth=1.8, label=rxn["reaction"])

        ax.set_xlabel("Temperature [K]", fontsize=11)
        ax.set_ylabel("ΔG°  [kJ / mol O₂]", fontsize=11)
        ax.set_title("Ellingham-Richardson Diagram", fontsize=13)
        ax.legend(fontsize=7, loc="upper right", ncol=1)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return _fig_to_b64(fig)
