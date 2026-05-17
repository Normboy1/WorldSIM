"""Thermodynamics simulation engine using NumPy/SciPy.

Provides heat transfer, ideal gas law, entropy calculation, and Carnot efficiency.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from simlab.core.constants.physical import CONSTANTS

# Van der Waals constants: a [L^2·bar/mol^2], b [L/mol]
# Source: NIST / Atkins Physical Chemistry
_VDW_CONSTANTS: dict[str, tuple[float, float]] = {
    "He":  (0.03457, 0.02370),
    "H2":  (0.2444,  0.02661),
    "N2":  (1.370,   0.03870),
    "O2":  (1.360,   0.03183),
    "Ar":  (1.363,   0.03219),
    "CO2": (3.640,   0.04267),
    "H2O": (5.536,   0.03049),
    "CH4": (2.300,   0.04301),
    "NH3": (4.169,   0.03707),
    "Cl2": (6.579,   0.05622),
    "SO2": (6.803,   0.05636),
    "C2H6":(5.562,   0.06380),
}


class ThermodynamicsEngine:
    """Thermodynamics simulations backed by NumPy/SciPy."""

    def simulate_heat_transfer(
        self,
        T_hot: float,
        T_cold: float,
        k_conductivity: float,
        area: float,
        thickness: float,
        t_span: float = 100.0,
        mass_hot: float = 1.0,
        mass_cold: float = 1.0,
        Cp: float = 4186.0,
        n_points: int = 300,
    ) -> dict:
        """Simulate transient conductive heat transfer between two thermal reservoirs.

        Uses Newton's law of cooling: dT_hot/dt = -Q_dot/m*Cp,
        where Q_dot = k*A*(T_hot - T_cold)/thickness.

        Parameters
        ----------
        T_hot:          initial temperature of hot body [K]
        T_cold:         initial temperature of cold body [K]
        k_conductivity: thermal conductivity [W/(m·K)]
        area:           cross-sectional area [m^2]
        thickness:      thickness of conductor [m]
        t_span:         simulation duration [s]
        mass_hot:       mass of hot body [kg]
        mass_cold:      mass of cold body [kg]
        Cp:             specific heat capacity [J/(kg·K)]
        """
        if k_conductivity <= 0 or area <= 0 or thickness <= 0:
            raise ValueError(
                "heat_transfer requires positive k_conductivity, area, and thickness."
            )
        if mass_hot <= 0 or mass_cold <= 0:
            raise ValueError("mass_hot and mass_cold must be positive.")
        R_thermal = thickness / (k_conductivity * area)  # thermal resistance [K/W]

        def ode(t, state):
            Th, Tc = state
            Q_dot = (Th - Tc) / R_thermal
            dTh = -Q_dot / (mass_hot * Cp)
            dTc = Q_dot / (mass_cold * Cp)
            return [dTh, dTc]

        t_eval = np.linspace(0, t_span, n_points)
        sol = solve_ivp(ode, [0, t_span], [T_hot, T_cold], t_eval=t_eval, rtol=1e-8, atol=1e-10)

        T_equil = (mass_hot * T_hot + mass_cold * T_cold) / (mass_hot + mass_cold)

        # Fourier heat conduction: steady-state flux
        Q_dot_initial = (T_hot - T_cold) / R_thermal

        return {
            "t": sol.t.tolist(),
            "T_hot": sol.y[0].tolist(),
            "T_cold": sol.y[1].tolist(),
            "T_equilibrium": float(T_equil),
            "R_thermal": float(R_thermal),
            "Q_dot_initial": float(Q_dot_initial),
            "success": bool(sol.success),
            "k_conductivity": k_conductivity,
            "area": area,
            "thickness": thickness,
        }

    def ideal_gas_law(
        self,
        P: float | None = None,
        V: float | None = None,
        n: float | None = None,
        T: float | None = None,
    ) -> dict:
        """Solve the ideal gas law PV = nRT for the missing variable.

        Exactly one of P, V, n, T must be None.
        """
        R = CONSTANTS.R
        given = {"P": P, "V": V, "n": n, "T": T}
        missing = [k for k, v in given.items() if v is None]

        if len(missing) != 1:
            raise ValueError(
                f"Exactly one of P, V, n, T must be None. Missing: {missing}"
            )

        var = missing[0]
        for key, val in given.items():
            if val is not None and float(val) <= 0:
                raise ValueError(
                    f"Ideal gas law requires positive P, V, n, T when specified; got {key}={val}."
                )
        if var == "P":
            result = n * R * T / V
        elif var == "V":
            result = n * R * T / P
        elif var == "n":
            result = P * V / (R * T)
        elif var == "T":
            result = P * V / (n * R)
        else:
            raise ValueError(f"Unexpected variable: {var}")

        given[var] = float(result)

        return {
            "solved_for": var,
            "value": float(result),
            "P": float(given["P"]),
            "V": float(given["V"]),
            "n": float(given["n"]),
            "T": float(given["T"]),
            "R": R,
            "unit_check": f"PV = {given['P'] * given['V']:.4e}, nRT = {given['n'] * R * given['T']:.4e}",
        }

    def calculate_entropy_change(
        self,
        n_moles: float,
        T1: float,
        T2: float,
        Cp: float,
    ) -> dict:
        """Calculate entropy change for an ideal gas heated at constant pressure.

        ΔS = n * Cp * ln(T2/T1)

        Parameters
        ----------
        n_moles: number of moles
        T1:      initial temperature [K]
        T2:      final temperature [K]
        Cp:      molar heat capacity at constant pressure [J/(mol·K)]
        """
        if T1 <= 0 or T2 <= 0:
            raise ValueError("Temperatures must be positive (Kelvin).")

        delta_S = n_moles * Cp * math.log(T2 / T1)
        delta_H = n_moles * Cp * (T2 - T1)

        return {
            "delta_S": float(delta_S),
            "delta_H": float(delta_H),
            "n_moles": n_moles,
            "T1": T1,
            "T2": T2,
            "Cp": Cp,
            "process": "constant_pressure",
            "unit": "J/K",
        }

    def carnot_efficiency(self, T_hot: float, T_cold: float) -> dict:
        """Compute the Carnot efficiency of an ideal heat engine.

        Parameters
        ----------
        T_hot:  temperature of the hot reservoir [K]
        T_cold: temperature of the cold reservoir [K]
        """
        if T_hot <= T_cold:
            raise ValueError("T_hot must be greater than T_cold for a heat engine.")

        eta = 1.0 - T_cold / T_hot
        cop_heat_pump = T_hot / (T_hot - T_cold)
        cop_refrigerator = T_cold / (T_hot - T_cold)

        return {
            "efficiency": float(eta),
            "efficiency_percent": float(eta * 100),
            "cop_heat_pump": float(cop_heat_pump),
            "cop_refrigerator": float(cop_refrigerator),
            "T_hot": T_hot,
            "T_cold": T_cold,
            "Q_ratio": float(T_cold / T_hot),
        }

    def van_der_waals_gas(
        self,
        gas: str | None = None,
        a: float | None = None,
        b: float | None = None,
        P: float | None = None,
        V: float | None = None,
        n: float | None = None,
        T: float | None = None,
    ) -> dict:
        """Solve the Van der Waals equation of state for a real gas.

        (P + a*n^2/V^2)(V - n*b) = nRT

        Provide a gas name OR explicit (a, b) constants, plus exactly three of (P, V, n, T).
        Units: P [Pa], V [m^3], n [mol], T [K].
        Note: a and b must be in SI units (Pa·m^6/mol^2 and m^3/mol); if using a named gas
        the stored L/mol values are converted automatically.

        Parameters
        ----------
        gas:  named gas from built-in table (He, H2, N2, O2, Ar, CO2, H2O, CH4, NH3, Cl2...)
        a:    attraction parameter [Pa·m^6/mol^2] — overrides gas table
        b:    excluded volume [m^3/mol] — overrides gas table
        """
        R = CONSTANTS.R

        if gas is not None:
            gas_upper = gas.upper() if gas not in _VDW_CONSTANTS else gas
            key = next((k for k in _VDW_CONSTANTS if k.upper() == gas_upper), None)
            if key is None:
                raise ValueError(
                    f"Gas {gas!r} not in table. Available: {list(_VDW_CONSTANTS)}. "
                    "Pass explicit a and b in SI units instead."
                )
            a_L, b_L = _VDW_CONSTANTS[key]
            # Convert from L-bar units: a [L^2·bar/mol^2] → [Pa·m^6/mol^2], b [L/mol] → [m^3/mol]
            a_si = a_L * 0.1  # 1 L^2·bar/mol^2 = 0.1 Pa·m^6/mol^2
            b_si = b_L * 1e-3  # 1 L/mol = 1e-3 m^3/mol
            gas_label = key
        elif a is not None and b is not None:
            a_si, b_si = float(a), float(b)
            gas_label = "custom"
        else:
            raise ValueError("Provide either a gas name or explicit a and b constants.")

        given = {k: v for k, v in {"P": P, "V": V, "n": n, "T": T}.items() if v is not None}
        missing_vars = [k for k in ("P", "V", "n", "T") if k not in given]
        if len(missing_vars) != 1:
            raise ValueError(f"Provide exactly three of P, V, n, T. Missing: {missing_vars}")

        for k, v in given.items():
            if float(v) <= 0:
                raise ValueError(f"{k}={v} must be positive.")

        solve_for = missing_vars[0]
        P_g = float(given.get("P", 0))
        V_g = float(given.get("V", 0))
        n_g = float(given.get("n", 0))
        T_g = float(given.get("T", 0))

        if solve_for == "P":
            result = n_g * R * T_g / (V_g - n_g * b_si) - a_si * n_g ** 2 / V_g ** 2
        elif solve_for == "T":
            result = (P_g + a_si * n_g ** 2 / V_g ** 2) * (V_g - n_g * b_si) / (n_g * R)
        elif solve_for == "n":
            # Quadratic in n — use numerical solve
            def f_n(n_try):
                if n_try <= 0 or V_g <= n_try * b_si:
                    return 1e30
                return (P_g + a_si * n_try ** 2 / V_g ** 2) * (V_g - n_try * b_si) - n_try * R * T_g
            result = brentq(f_n, 1e-9, V_g / b_si * 0.99, xtol=1e-12)
        else:  # solve for V — cubic equation
            # P*V^3 - (nb*P + nRT)*V^2 + a*n^2*V - a*n^3*b = 0
            def f_V(V_try):
                if V_try <= n_g * b_si:
                    return 1e30
                return (P_g + a_si * n_g ** 2 / V_try ** 2) * (V_try - n_g * b_si) - n_g * R * T_g
            V_ideal = n_g * R * T_g / P_g
            lo = n_g * b_si * 1.0001
            hi = max(V_ideal * 10, lo * 100)
            result = brentq(f_V, lo, hi, xtol=1e-15)

        values = {"P": P_g, "V": V_g, "n": n_g, "T": T_g}
        values[solve_for] = float(result)
        P_f, V_f, n_f, T_f = values["P"], values["V"], values["n"], values["T"]

        V_ideal = n_f * R * T_f / P_f
        Z = P_f * V_f / (n_f * R * T_f)
        deviation_pct = 100.0 * (V_f - V_ideal) / V_ideal

        return {
            "solved_for": solve_for,
            "value": float(result),
            "P_Pa": float(P_f),
            "V_m3": float(V_f),
            "n_mol": float(n_f),
            "T_K": float(T_f),
            "compressibility_Z": float(Z),
            "ideal_V_m3": float(V_ideal),
            "volume_deviation_percent": float(deviation_pct),
            "gas": gas_label,
            "a_SI": float(a_si),
            "b_SI": float(b_si),
            "model": "Van der Waals equation of state",
            "note": "Z<1 → attractive forces dominate; Z>1 → repulsive (excluded volume) dominates",
        }

    def stefan_boltzmann_radiation(
        self, T: float, emissivity: float = 1.0, area: float = 1.0
    ) -> dict:
        """Compute blackbody radiation power via the Stefan-Boltzmann law.

        P = epsilon * sigma * A * T^4

        Parameters
        ----------
        T:           surface temperature [K]
        emissivity:  emissivity (0-1, 1 = perfect blackbody)
        area:        radiating area [m^2]
        """
        if T <= 0:
            raise ValueError("Temperature T must be positive (Kelvin).")
        if not (0.0 <= emissivity <= 1.0):
            raise ValueError("emissivity must be between 0 and 1.")
        if area < 0:
            raise ValueError("area must be non-negative.")

        sigma = CONSTANTS.sigma_SB
        power = emissivity * sigma * area * T**4
        peak_wavelength = 2.898e-3 / T  # Wien's displacement law [m]

        return {
            "power": float(power),
            "power_per_area": float(emissivity * sigma * T**4),
            "peak_wavelength_m": float(peak_wavelength),
            "peak_wavelength_nm": float(peak_wavelength * 1e9),
            "T": T,
            "emissivity": emissivity,
            "area": area,
        }
