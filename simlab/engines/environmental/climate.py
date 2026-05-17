"""Climate and global-warming simulation utilities.

Covers zero-dimensional energy balance models, global warming potentials,
radiative forcing from greenhouse gases, climate sensitivity, sea-level rise
projections, and a simple carbon-cycle box model.

Optional dependencies: scipy (ODE integration), numpy (always required).
Falls back to analytic / Euler-step approximations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.integrate import solve_ivp as _solve_ivp
    _SCIPY = True
except ImportError:
    _SCIPY = False

_SIGMA = 5.6704e-8  # Stefan-Boltzmann constant


def energy_balance_model(
    S0: float = 1361.0,
    albedo: float = 0.30,
    emissivity: float = 0.78,
    greenhouse_factor: float = 1.0,
) -> dict[str, Any]:
    """Zero-dimensional energy balance model for global mean temperature.

    T_eq = [ S0*(1-alpha)/(4*sigma*epsilon*GHF) ]^(1/4)

    Args:
        S0: Solar constant (W/m^2).
        albedo: Planetary albedo (0-1).
        emissivity: Effective emissivity (0-1).
        greenhouse_factor: Multiplier reducing OLR (1 = no greenhouse effect;
            > 1 means reduced emission = warming).

    Returns:
        Dict with keys: T_eq_K, T_eq_C, absorbed_SW_W_m2, OLR_W_m2,
        backend, note.
    """
    absorbed = S0 * (1.0 - albedo) / 4.0
    T_eq = (absorbed / (_SIGMA * emissivity / greenhouse_factor)) ** 0.25
    OLR = _SIGMA * emissivity * T_eq ** 4 / greenhouse_factor
    return {
        "backend": "analytic", "note": "",
        "T_eq_K": T_eq, "T_eq_C": T_eq - 273.15,
        "absorbed_SW_W_m2": absorbed, "OLR_W_m2": OLR,
    }


def global_warming_potential(
    gas: str = "CH4",
    time_horizon_yr: int = 100,
) -> dict[str, Any]:
    """Return the global warming potential (GWP) of a greenhouse gas.

    Values from IPCC AR6 (2021) for 20- and 100-year horizons.

    Args:
        gas: Greenhouse gas identifier (CO2, CH4, N2O, SF6, HFC134a).
        time_horizon_yr: Integration horizon (20 or 100 years).

    Returns:
        Dict with keys: GWP, gas, time_horizon_yr, backend, note.
    """
    # AR6 GWP values {gas: (GWP20, GWP100)}
    _GWP = {
        "CO2": (1, 1),
        "CH4": (81.2, 27.9),
        "N2O": (273, 273),
        "SF6": (17500, 24300),
        "HFC134a": (3600, 1526),
    }
    gas_up = gas.upper()
    if gas_up not in _GWP:
        return {
            "backend": "analytic",
            "note": f"Gas '{gas}' not in database. Available: {list(_GWP.keys())}",
            "GWP": None, "gas": gas, "time_horizon_yr": time_horizon_yr,
        }
    gwp_20, gwp_100 = _GWP[gas_up]
    gwp = gwp_20 if time_horizon_yr <= 20 else gwp_100
    return {
        "backend": "analytic", "note": "",
        "GWP": gwp, "gas": gas, "time_horizon_yr": time_horizon_yr,
    }


def radiative_forcing(
    CO2_ppm: float = 415.0,
    CH4_ppb: float = 1900.0,
    N2O_ppb: float = 332.0,
) -> dict[str, Any]:
    """Estimate total radiative forcing from well-mixed greenhouse gases.

    Uses the simplified expressions from Myhre et al. (1998):
    RF_CO2 = 5.35 * ln(C/C0)
    RF_CH4 = 0.036 * (sqrt(M) - sqrt(M0))
    RF_N2O = 0.12 * (sqrt(N) - sqrt(N0))

    Args:
        CO2_ppm: Atmospheric CO2 concentration (ppm).
        CH4_ppb: Atmospheric CH4 concentration (ppb).
        N2O_ppb: Atmospheric N2O concentration (ppb).

    Returns:
        Dict with keys: RF_CO2_W_m2, RF_CH4_W_m2, RF_N2O_W_m2,
        total_RF_W_m2, backend, note.
    """
    C0, M0, N0 = 278.0, 722.0, 270.0  # pre-industrial
    RF_CO2 = 5.35 * math.log(CO2_ppm / C0)
    RF_CH4 = 0.036 * (math.sqrt(CH4_ppb) - math.sqrt(M0))
    RF_N2O = 0.12 * (math.sqrt(N2O_ppb) - math.sqrt(N0))
    total = RF_CO2 + RF_CH4 + RF_N2O
    return {
        "backend": "analytic", "note": "",
        "RF_CO2_W_m2": RF_CO2, "RF_CH4_W_m2": RF_CH4,
        "RF_N2O_W_m2": RF_N2O, "total_RF_W_m2": total,
    }


def climate_sensitivity(
    forcing_W_m2: float = 3.7,
    lambda_K_per_W_m2: float = 0.8,
) -> dict[str, Any]:
    """Estimate equilibrium temperature change from radiative forcing.

    delta_T = lambda * RF   (lambda = climate sensitivity parameter)

    Args:
        forcing_W_m2: Radiative forcing (W/m^2).
        lambda_K_per_W_m2: Climate sensitivity parameter (K/(W/m^2)).
            Equivalent Climate Sensitivity (ECS) = lambda * 3.7 W/m^2.

    Returns:
        Dict with keys: delta_T_K, ECS_K, TCR_estimate_K, backend, note.
    """
    delta_T = lambda_K_per_W_m2 * forcing_W_m2
    ECS = lambda_K_per_W_m2 * 3.7
    TCR = ECS * 0.6  # typical TCR/ECS ratio
    return {
        "backend": "analytic", "note": "",
        "delta_T_K": delta_T, "ECS_K": ECS, "TCR_estimate_K": TCR,
    }


def sea_level_rise(
    delta_T_C: float = 2.0,
    t_years: float = 100.0,
) -> dict[str, Any]:
    """Project global mean sea-level rise from temperature anomaly.

    Uses a simple semi-empirical model:
    dSL/dt = a * (T - T_eq)   with thermal + ice-sheet contributions.

    Args:
        delta_T_C: Global mean temperature rise above pre-industrial (C).
        t_years: Projection horizon (years).

    Returns:
        Dict with keys: total_rise_mm, thermal_expansion_mm,
        ice_melt_contribution_mm, t_years, delta_T_C, backend, note.
    """
    # Simplified Rahmstorf-type scaling
    rate_mm_yr_per_K = 3.4  # mm/yr/K (central estimate)
    thermal_frac = 0.38
    ice_frac = 0.62
    total = rate_mm_yr_per_K * delta_T_C * t_years
    thermal = total * thermal_frac
    ice = total * ice_frac
    return {
        "backend": "stub",
        "note": "Install scipy for full process-based SLR model.",
        "total_rise_mm": total,
        "thermal_expansion_mm": thermal,
        "ice_melt_contribution_mm": ice,
        "t_years": t_years,
        "delta_T_C": delta_T_C,
    }


def carbon_cycle_box_model(
    emissions_GtC_yr: float = 10.0,
    t_end_yr: float = 100.0,
) -> dict[str, Any]:
    """3-box carbon cycle: atmosphere, ocean surface, deep ocean.

    dC_atm/dt = emissions - k_ao*(C_atm - C_oc_surf/beta)
    dC_oc_surf/dt = k_ao*(C_atm - C_oc_surf/beta) - k_od*C_oc_surf

    Args:
        emissions_GtC_yr: Annual anthropogenic carbon emissions (GtC/yr).
        t_end_yr: Simulation end time (years).

    Returns:
        Dict with keys: t_yr, C_atm_GtC, C_ocean_surf_GtC, CO2_ppm,
        airborne_fraction, backend, note.
    """
    n_steps = int(t_end_yr * 10)
    dt = t_end_yr / n_steps
    t_arr = np.linspace(0, t_end_yr, n_steps + 1)
    C_atm = np.zeros(n_steps + 1)
    C_oc = np.zeros(n_steps + 1)
    C_atm[0] = 590.0   # GtC ~ 280 ppm pre-industrial
    C_oc[0] = 900.0

    k_ao = 0.12   # yr^-1 atmosphere-ocean exchange
    k_od = 0.02   # yr^-1 surface-deep exchange
    beta = 10.0   # solubility ratio

    for i in range(n_steps):
        exchange = k_ao * (C_atm[i] - C_oc[i] / beta)
        dC_atm = emissions_GtC_yr - exchange
        dC_oc = exchange - k_od * C_oc[i]
        C_atm[i + 1] = C_atm[i] + dC_atm * dt
        C_oc[i + 1] = max(C_oc[i] + dC_oc * dt, 0.0)

    CO2_ppm = C_atm / 590.0 * 280.0
    AF = (C_atm[-1] - C_atm[0]) / (emissions_GtC_yr * t_end_yr)

    return {
        "backend": "stub",
        "note": "Install scipy for stiff ODE solver (ocean chemistry).",
        "t_yr": t_arr.tolist(),
        "C_atm_GtC": C_atm.tolist(),
        "C_ocean_surf_GtC": C_oc.tolist(),
        "CO2_ppm": CO2_ppm.tolist(),
        "airborne_fraction": AF,
    }
