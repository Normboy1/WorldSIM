"""GPU-accelerated physics simulation for superalloy turbine blade geometry.

Implements on-device solvers using PyTorch CUDA tensor ops:
  1. Steady-state 2-D thermal solver (FD, Jacobi iterations) with film cooling
  2. Thermoelastic von Mises stress field (plane-strain approximation)
  3. Norton-Bailey creep rate field (dε/dt = A · σ^n · exp(-Q/RT))
  4. High-cycle fatigue (Basquin) life estimate

All heavy ops stay on-device (no CPU round-trips per iteration).
Material properties tuned to Ni57Cr15Co9Al10Ti3Ta7 superalloy (from WorldSIM
materials engine alloy candidates).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH = False
    DEVICE = None  # type: ignore


# ---------------------------------------------------------------------------
# Ni57Cr15Co9Al10Ti3Ta7 material constants
# ---------------------------------------------------------------------------

class Ni57Props:
    """Temperature-dependent properties for Ni57Cr15Co9Al10Ti3Ta7."""
    rho = 8440.0          # kg/m³
    T_solidus = 1534.0    # °C
    T_service = 1050.0    # °C  (typical HPT blade metal temperature)

    # Thermal conductivity  k(T) ≈ k0 + k1·T  [W/m·K]
    k0 = 9.8;  k1 = 0.009

    # Elastic modulus E(T) ≈ E0 · (1 - β·(T - 20))  [GPa]
    E0 = 200.0;  beta_E = 2.5e-4

    # Poisson ratio (weakly T-dependent, treat as constant)
    nu = 0.30

    # Thermal expansion α [1/K]  (average 20–1050°C)
    alpha = 13.5e-6

    # Norton creep:  dε/dt = A · σ_vm^n · exp(-Q/RT)
    # Calibrated for Ni-base superalloy: ~1e-9 /s at 900°C, 300 MPa
    A_norton = 1.7e-36   # s⁻¹ Pa⁻ⁿ
    n_norton = 4.8
    Q_norton = 310_000.0  # J/mol
    R_gas = 8.314         # J/(mol·K)

    # Basquin HCF:  N_f = (σ_f' / Δσ/2)^(1/b)
    sigma_f_prime = 1050e6  # Pa  (fatigue strength coefficient)
    b_basquin = -0.085      # fatigue strength exponent


# ---------------------------------------------------------------------------
# Geometry mask — 2-D cross-section of HPT blade
# ---------------------------------------------------------------------------

def _blade_mask(nx: int, ny: int, device) -> tuple:
    """Return (blade_mask, cooling_mask, exterior_mask) boolean tensors.

    Simple analytical approximation:
    - Blade cross-section: NACA-4412 profile at chord=0.8·span
    - 6 circular film-cooling channels along camber line
    """
    x = torch.linspace(0.0, 1.0, nx, device=device)
    y = torch.linspace(0.0, 1.0, ny, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")  # (ny, nx)

    # NACA 4412 half-thickness function (simplified)
    tc = 0.12  # max thickness ratio
    xc = (X - 0.1).clamp(0.0, 1.0)
    t = 5 * tc * (0.2969 * xc.sqrt() - 0.1260 * xc
                  - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    # Camber line (m=0.04, p=0.4)
    m, p = 0.04, 0.40
    yc = torch.where(
        xc <= p,
        (m / p**2) * (2*p*xc - xc**2),
        (m / (1-p)**2) * (1 - 2*p + 2*p*xc - xc**2),
    )
    # Remap y to blade-local coords
    y_rel = Y - 0.5  # centre about 0
    blade_mask = (y_rel > yc - t) & (y_rel < yc + t) & (X > 0.05) & (X < 0.97)

    # 6 cooling holes along camber
    cooling_mask = torch.zeros_like(blade_mask, dtype=torch.bool)
    r_hole = 0.025
    for xi in [0.15, 0.25, 0.40, 0.55, 0.70, 0.82]:
        # find camber y at this x
        xci = torch.tensor(xi, device=device)
        yc_i = float((m / (1-p)**2) * (1 - 2*p + 2*p*xci - xci**2)
                     if xi > p else
                     (m / p**2) * (2*p*xci - xci**2))
        dist2 = (X - xi)**2 + (y_rel - yc_i)**2
        cooling_mask |= (dist2 < r_hole**2)

    cooling_mask = cooling_mask & blade_mask
    exterior_mask = ~blade_mask

    return blade_mask, cooling_mask, exterior_mask


# ---------------------------------------------------------------------------
# Thermal solver (Jacobi FD on GPU)
# ---------------------------------------------------------------------------

def gpu_thermal_solve(
    nx: int = 256,
    ny: int = 128,
    T_gas: float = 1300.0,    # °C  hot gas temperature
    T_coolant: float = 650.0,  # °C  film cooling temperature
    h_gas: float = 3000.0,    # W/m²K  hot side HTC
    h_cool: float = 8000.0,   # W/m²K  coolant HTC
    n_iter: int = 2000,
    convergence_tol: float = 1e-4,
    chord: float = 0.05,      # m  blade chord
) -> dict[str, Any]:
    """Steady-state 2-D conduction + convective BCs on GPU.

    Returns temperature field (numpy), solve stats, and peak temperatures.
    """
    if not _TORCH:
        return {"error": "torch not available"}

    t0 = time.time()
    dx = chord / nx
    dy = chord / ny

    blade_mask, cool_mask, ext_mask = _blade_mask(nx, ny, DEVICE)

    # Initial temperature field
    T = torch.full((ny, nx), float(T_gas), device=DEVICE)
    T[cool_mask] = T_coolant

    # Thermal conductivity field (°C → use service temp for k)
    T_ref = (T_gas + T_coolant) / 2.0
    k = Ni57Props.k0 + Ni57Props.k1 * T_ref  # scalar W/m·K

    alpha_gas = h_gas * dx / k     # Biot-like number
    alpha_cool = h_cool * dx / k

    conv_hist = []

    for it in range(n_iter):
        # Jacobi step: average of 4 neighbours (interior blade nodes)
        pad = F.pad(T.unsqueeze(0).unsqueeze(0),
                    (1, 1, 1, 1), mode="replicate").squeeze()
        T_new = 0.25 * (pad[1:-1, 2:] + pad[1:-1, :-2]
                       + pad[2:, 1:-1] + pad[:-2, 1:-1])

        # Apply boundary conditions
        # Gas-side convection (exterior boundary of blade)
        T_new = torch.where(ext_mask, torch.tensor(T_gas, device=DEVICE), T_new)

        # Cooling holes: convective sink
        T_cool_bc = (T + alpha_cool * T_coolant) / (1.0 + alpha_cool)
        T_new = torch.where(cool_mask, T_cool_bc, T_new)

        # Interior blade: update
        blade_interior = blade_mask & ~cool_mask
        T_final = torch.where(blade_interior, T_new, T)

        # Check convergence
        if it % 100 == 0:
            delta = float((T_final - T).abs().max())
            conv_hist.append(delta)
            if delta < convergence_tol:
                break

        T = T_final

    elapsed = time.time() - t0

    T_cpu = T.cpu().numpy()
    blade_cpu = blade_mask.cpu().numpy()

    T_blade_vals = T_cpu[blade_cpu]
    T_peak = float(T_blade_vals.max())
    T_mean = float(T_blade_vals.mean())
    T_min = float(T_blade_vals.min())

    margin_to_solidus = Ni57Props.T_solidus - T_peak

    return {
        "solver": "jacobi_fd_gpu",
        "device": str(DEVICE),
        "grid_size": [ny, nx],
        "n_iter_run": it + 1,
        "convergence_history": [round(v, 6) for v in conv_hist],
        "T_gas_C": T_gas,
        "T_coolant_C": T_coolant,
        "T_peak_blade_C": round(T_peak, 1),
        "T_mean_blade_C": round(T_mean, 1),
        "T_min_blade_C": round(T_min, 1),
        "margin_to_solidus_C": round(margin_to_solidus, 1),
        "solve_time_s": round(elapsed, 3),
        "field": T_cpu,
        "blade_mask": blade_cpu,
    }


# ---------------------------------------------------------------------------
# Thermoelastic stress field
# ---------------------------------------------------------------------------

def gpu_thermoelastic_stress(
    T_field: np.ndarray,
    blade_mask: np.ndarray,
    T_ref: float = 20.0,
    chord: float = 0.05,
) -> dict[str, Any]:
    """Plane-strain thermoelastic von Mises stress from temperature field.

    Uses gradient-induced thermal stress (temperature deviation from mean):
        σ_vm ≈ E·α·|T - T_mean_blade| / (1 - ν)

    This captures thermal gradient stress across the wall rather than
    the unconstrained mean expansion (which is accommodated by the root attachment).
    """
    if not _TORCH:
        return {"error": "torch not available"}

    t0 = time.time()
    T = torch.tensor(T_field, device=DEVICE, dtype=torch.float32)
    mask = torch.tensor(blade_mask, device=DEVICE)

    # Local deviation from mean blade temperature (gradient stress driver)
    T_mean = T[mask].mean()
    dT = T - T_mean

    # Temperature-dependent E (clamp to physical range)
    E = (Ni57Props.E0 * (1.0 - Ni57Props.beta_E * (T - 20.0))).clamp(80.0, 210.0)
    E_Pa = E * 1e9  # GPa → Pa

    # Plane-strain von Mises from thermal gradient mismatch
    nu = Ni57Props.nu
    alpha = Ni57Props.alpha

    sigma_vm = (E_Pa * alpha * dT.abs()) / (1.0 - nu)

    # Only inside blade
    sigma_blade = sigma_vm * mask.float()

    blade_vals = sigma_blade[mask]
    sigma_peak = float(blade_vals.max()) / 1e6  # MPa
    sigma_mean = float(blade_vals.mean()) / 1e6
    sigma_stress_field = sigma_blade.cpu().numpy() / 1e6  # MPa

    elapsed = time.time() - t0

    return {
        "solver": "thermoelastic_plane_strain_gpu",
        "sigma_vm_peak_MPa": round(sigma_peak, 1),
        "sigma_vm_mean_MPa": round(sigma_mean, 1),
        "yield_strength_proxy_MPa": 571.0,
        "safety_factor": round(571.0 / max(sigma_peak, 1.0), 3),
        "solve_time_s": round(elapsed, 4),
        "field_MPa": sigma_stress_field,
    }


# ---------------------------------------------------------------------------
# Creep rate field (Norton-Bailey)
# ---------------------------------------------------------------------------

def gpu_creep_field(
    T_field: np.ndarray,
    stress_field_MPa: np.ndarray,
    blade_mask: np.ndarray,
    t_service_hours: float = 25_000.0,
) -> dict[str, Any]:
    """Norton-Bailey creep rate and accumulated strain over service life.

    dε/dt = A · σ_vm^n · exp(-Q / R·T_K)
    """
    if not _TORCH:
        return {"error": "torch not available"}

    t0 = time.time()
    T_K = torch.tensor(T_field + 273.15, device=DEVICE, dtype=torch.float64)
    sigma = torch.tensor(stress_field_MPa * 1e6, device=DEVICE, dtype=torch.float64)
    mask = torch.tensor(blade_mask, device=DEVICE)

    A = Ni57Props.A_norton
    n = Ni57Props.n_norton
    Q = Ni57Props.Q_norton
    R = Ni57Props.R_gas

    # Cap at yield strength — beyond yield, material redistributes stress;
    # creep accumulates from the yield-limited state
    yield_Pa = Ni57Props.sigma_f_prime * 0.54  # ~571 MPa at service temp
    sigma = sigma.clamp(0.0, yield_Pa)

    # Creep rate field [1/s]
    sigma_pos = sigma.clamp(1e3)  # avoid 0^n issues
    creep_rate = A * sigma_pos.pow(n) * torch.exp(-Q / (R * T_K))

    # Accumulated strain over service life
    t_service_s = t_service_hours * 3600.0
    accumulated_strain = creep_rate * t_service_s

    # Mask to blade
    creep_blade = creep_rate * mask.float()
    strain_blade = accumulated_strain * mask.float()

    blade_vals_rate = creep_blade[mask]
    blade_vals_strain = strain_blade[mask]

    peak_creep = float(blade_vals_rate.max())
    peak_strain = float(blade_vals_strain.max())
    mean_strain = float(blade_vals_strain.mean())

    # Larson-Miller P for Ni57 (approx)
    LM_C = 20.0  # Larson-Miller constant
    T_peak = float(T_K[mask].max())
    t_rup_log = (LM_C + 20.0) / (T_peak / 1000.0) - LM_C
    t_rupture_h = 10.0 ** t_rup_log

    elapsed = time.time() - t0

    return {
        "solver": "norton_bailey_gpu",
        "service_life_h": t_service_hours,
        "peak_creep_rate_per_s": f"{peak_creep:.3e}",
        "peak_accumulated_strain_pct": round(peak_strain * 100, 4),
        "mean_accumulated_strain_pct": round(mean_strain * 100, 6),
        "larson_miller_rupture_h": round(t_rupture_h, 0),
        "pass_creep": peak_strain < 0.02,  # <2% total creep strain → pass
        "solve_time_s": round(elapsed, 4),
        "field_rate": creep_blade.float().cpu().numpy(),
        "field_strain": strain_blade.float().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# HCF life estimate (Basquin)
# ---------------------------------------------------------------------------

def gpu_fatigue_life(
    stress_field_MPa: np.ndarray,
    blade_mask: np.ndarray,
    stress_ratio: float = 0.1,  # R = σ_min / σ_max
) -> dict[str, Any]:
    """Basquin high-cycle fatigue life estimate at each node.

    N_f = (σ_f' / Δσ_a)^(1/b)    where Δσ_a = σ_amp × mean-stress correction
    """
    if not _TORCH:
        return {"error": "torch not available"}

    t0 = time.time()
    sigma = torch.tensor(stress_field_MPa, device=DEVICE, dtype=torch.float32)
    mask = torch.tensor(blade_mask, device=DEVICE)

    sigma_amp = sigma * (1.0 - stress_ratio) / 2.0

    sigma_f_prime = Ni57Props.sigma_f_prime / 1e6  # → MPa
    b = Ni57Props.b_basquin

    # Avoid log of 0
    sigma_amp_safe = sigma_amp.clamp(0.1)
    log_Nf = (1.0 / b) * torch.log10(sigma_f_prime / sigma_amp_safe)
    Nf = (10.0 ** log_Nf) * mask.float()

    blade_Nf = Nf[mask]
    min_life = float(blade_Nf[blade_Nf > 0].min()) if (blade_Nf > 0).any() else 0.0
    mean_life = float(blade_Nf.mean())

    # Typical HPT blade requires N_f > 1e7 cycles
    limit_cycles = 1e7
    fraction_pass = float((blade_Nf >= limit_cycles).float().mean())

    elapsed = time.time() - t0

    return {
        "solver": "basquin_hcf_gpu",
        "stress_ratio_R": stress_ratio,
        "min_life_cycles": f"{min_life:.2e}",
        "mean_life_cycles": f"{mean_life:.2e}",
        "fraction_above_1e7_cycles": round(fraction_pass, 4),
        "hcf_pass": min_life > 1e6,
        "solve_time_s": round(elapsed, 4),
        "field_log10_Nf": torch.log10(Nf.clamp(1.0)).cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Full integrated blade simulation
# ---------------------------------------------------------------------------

def gpu_blade_simulation(
    nx: int = 256,
    ny: int = 128,
    T_gas: float = 1300.0,
    T_coolant: float = 650.0,
    n_thermal_iter: int = 3000,
    t_service_hours: float = 25_000.0,
) -> dict[str, Any]:
    """Run the full multi-physics blade simulation pipeline on GPU.

    Chain: thermal solve → thermoelastic stress → creep → HCF fatigue
    Returns all fields and a pass/fail summary for the Ni57 alloy.
    """
    if not _TORCH:
        return {"error": "torch not available", "domain": "qdgeometry",
                "type": "blade_simulation"}

    t_total = time.time()

    # Step 1: Thermal
    thermal = gpu_thermal_solve(
        nx=nx, ny=ny,
        T_gas=T_gas, T_coolant=T_coolant,
        n_iter=n_thermal_iter,
    )

    T_field = thermal["field"]
    blade_mask = thermal["blade_mask"]

    # Step 2: Thermoelastic stress
    stress = gpu_thermoelastic_stress(T_field, blade_mask)

    # Step 3: Creep
    creep = gpu_creep_field(T_field, stress["field_MPa"], blade_mask,
                            t_service_hours=t_service_hours)

    # Step 4: HCF fatigue
    fatigue = gpu_fatigue_life(stress["field_MPa"], blade_mask)

    total_time = time.time() - t_total

    # Pass/fail summary
    passes = {
        "thermal_margin": thermal["margin_to_solidus_C"] > 200.0,
        "stress_safety": stress["safety_factor"] > 1.2,
        "creep_strain": creep["pass_creep"],
        "hcf_life": fatigue["hcf_pass"],
    }
    overall_pass = all(passes.values())

    return {
        "domain": "qdgeometry",
        "type": "blade_simulation",
        "alloy": "Ni57Cr15Co9Al10Ti3Ta7",
        "device": str(DEVICE),
        "grid_size": [ny, nx],
        "t_service_hours": t_service_hours,
        "thermal": {k: v for k, v in thermal.items()
                    if k not in ("field", "blade_mask")},
        "stress": {k: v for k, v in stress.items() if k != "field_MPa"},
        "creep": {k: v for k, v in creep.items()
                  if k not in ("field_rate", "field_strain")},
        "fatigue": {k: v for k, v in fatigue.items() if k != "field_log10_Nf"},
        "pass_fail": passes,
        "overall_pass": overall_pass,
        "total_sim_time_s": round(total_time, 3),
        # fields for visualisation
        "_fields": {
            "T": T_field,
            "sigma_vm": stress["field_MPa"],
            "creep_strain": creep["field_strain"],
            "log10_Nf": fatigue["field_log10_Nf"],
            "blade_mask": blade_mask,
        },
    }
