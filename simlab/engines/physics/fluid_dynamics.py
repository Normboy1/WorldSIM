"""Fluid dynamics engine using NumPy/SciPy.

Bernoulli flow, pipe flow (laminar + turbulent), Reynolds number,
drag on immersed bodies, and terminal velocity.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


class FluidDynamicsEngine:
    """Fluid dynamics calculations backed by NumPy/SciPy."""

    def bernoulli(
        self,
        rho: float,
        P1: float | None = None,
        v1: float | None = None,
        h1: float = 0.0,
        P2: float | None = None,
        v2: float | None = None,
        h2: float = 0.0,
        A1: float | None = None,
        A2: float | None = None,
        g: float = 9.81,
    ) -> dict:
        """Solve the Bernoulli equation for incompressible, inviscid steady flow.

        P + 0.5*rho*v^2 + rho*g*h = constant

        If cross-sectional areas A1 and A2 are provided, continuity (v1*A1 = v2*A2)
        is applied to eliminate one velocity unknown automatically.

        Provide exactly one unknown among {P1, v1, P2, v2} (h1, h2 always required).

        Parameters
        ----------
        rho:       fluid density [kg/m^3] (water≈1000, air≈1.225)
        P1, P2:    static pressures at points 1 and 2 [Pa]
        v1, v2:    flow velocities at points 1 and 2 [m/s]
        h1, h2:    elevation at points 1 and 2 [m]
        A1, A2:    cross-sectional areas [m^2] — enables continuity equation
        g:         gravitational acceleration [m/s^2]
        """
        if rho <= 0:
            raise ValueError("fluid density rho must be positive.")

        # If areas given, apply continuity to set v2 from v1 or vice versa
        if A1 is not None and A2 is not None:
            if A1 <= 0 or A2 <= 0:
                raise ValueError("A1 and A2 must be positive.")
            if v1 is not None and v2 is None:
                v2 = v1 * A1 / A2
            elif v2 is not None and v1 is None:
                v1 = v2 * A2 / A1

        unknowns = [x for x in [P1, v1, P2, v2] if x is None]
        if len(unknowns) != 1:
            raise ValueError(
                "Provide exactly one unknown among P1, v1, P2, v2 (after applying continuity)."
            )

        if P1 is None:
            P1 = P2 + 0.5 * rho * (v2 ** 2 - v1 ** 2) + rho * g * (h2 - h1)
            solved_for = "P1"
        elif v1 is None:
            v1_sq = v2 ** 2 + 2 * (P2 - P1) / rho + 2 * g * (h2 - h1)
            if v1_sq < 0:
                raise ValueError("No real solution for v1 — check input consistency.")
            v1 = math.sqrt(v1_sq)
            solved_for = "v1"
        elif P2 is None:
            P2 = P1 + 0.5 * rho * (v1 ** 2 - v2 ** 2) + rho * g * (h1 - h2)
            solved_for = "P2"
        else:
            v2_sq = v1 ** 2 + 2 * (P1 - P2) / rho + 2 * g * (h1 - h2)
            if v2_sq < 0:
                raise ValueError("No real solution for v2 — check input consistency.")
            v2 = math.sqrt(v2_sq)
            solved_for = "v2"

        dynamic_P1 = 0.5 * rho * v1 ** 2
        dynamic_P2 = 0.5 * rho * v2 ** 2
        stagnation_P1 = P1 + dynamic_P1 + rho * g * h1

        flow_rate = v1 * A1 if A1 is not None else None

        return {
            "solved_for": solved_for,
            "P1_Pa": float(P1),
            "P2_Pa": float(P2),
            "v1_m_s": float(v1),
            "v2_m_s": float(v2),
            "h1_m": float(h1),
            "h2_m": float(h2),
            "dynamic_pressure_1_Pa": float(dynamic_P1),
            "dynamic_pressure_2_Pa": float(dynamic_P2),
            "stagnation_pressure_Pa": float(stagnation_P1),
            "velocity_ratio_v2_v1": float(v2 / v1) if v1 else None,
            "pressure_drop_Pa": float(P1 - P2),
            "volumetric_flow_rate_m3_s": float(flow_rate) if flow_rate is not None else None,
            "rho": rho,
            "model": "Bernoulli (incompressible, inviscid, steady)",
        }

    def pipe_flow(
        self,
        delta_P: float,
        radius: float,
        length: float,
        viscosity: float,
        rho: float,
        roughness: float = 0.0,
    ) -> dict:
        """Compute flow through a circular pipe.

        Laminar regime (Re < 2300): Hagen-Poiseuille law (exact).
        Turbulent regime (Re ≥ 4000): Darcy-Weisbach with Colebrook friction factor.
        Transition (2300–4000): both bounds reported.

        Parameters
        ----------
        delta_P:   pressure drop [Pa]  (P_inlet - P_outlet, must be positive)
        radius:    pipe inner radius [m]
        length:    pipe length [m]
        viscosity: dynamic viscosity [Pa·s]  (water≈8.9e-4, air≈1.81e-5)
        rho:       fluid density [kg/m^3]
        roughness: pipe roughness height [m] (smooth=0, commercial steel≈4.6e-5)
        """
        if delta_P <= 0:
            raise ValueError("delta_P must be positive.")
        if radius <= 0 or length <= 0:
            raise ValueError("radius and length must be positive.")
        if viscosity <= 0 or rho <= 0:
            raise ValueError("viscosity and rho must be positive.")

        D = 2.0 * radius
        A = math.pi * radius ** 2

        # Hagen-Poiseuille (laminar exact)
        Q_laminar = math.pi * radius ** 4 * delta_P / (8.0 * viscosity * length)
        v_laminar = Q_laminar / A
        Re_laminar = rho * v_laminar * D / viscosity

        # Darcy-Weisbach with Colebrook friction factor (turbulent)
        # delta_P = f * (L/D) * 0.5 * rho * v^2  →  v = sqrt(2*delta_P*D / (f*L*rho))
        # Colebrook: 1/sqrt(f) = -2*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f)))
        rel_rough = roughness / D

        def colebrook(f):
            if f <= 0:
                return 1e30
            return 1.0 / math.sqrt(f) + 2.0 * math.log10(rel_rough / 3.7 + 2.51 / (Re_turb_guess * math.sqrt(f)))

        # Initial Re guess from smooth-pipe explicit approximation (Swamee-Jain)
        if rel_rough > 0:
            f_sw = 0.25 / (math.log10(rel_rough / 3.7 + 5.74 / (1e5 ** 0.9))) ** 2
        else:
            f_sw = 0.02  # reasonable turbulent starting point

        Re_turb_guess = rho * math.sqrt(2 * delta_P * D / (f_sw * length * rho)) * D / viscosity

        try:
            f_turb = brentq(colebrook, 1e-6, 1.0, xtol=1e-10, maxiter=100)
            v_turb = math.sqrt(2 * delta_P * D / (f_turb * length * rho))
            Q_turb = v_turb * A
            Re_turb = rho * v_turb * D / viscosity
        except Exception:
            f_turb = v_turb = Q_turb = Re_turb = float("nan")

        # Determine which regime applies
        if Re_laminar < 2300:
            regime = "laminar"
            Q_best = Q_laminar
            v_best = v_laminar
            Re_best = Re_laminar
        elif Re_laminar < 4000:
            regime = "transitional"
            Q_best = Q_laminar
            v_best = v_laminar
            Re_best = Re_laminar
        else:
            regime = "turbulent"
            Q_best = Q_turb
            v_best = v_turb
            Re_best = Re_turb

        return {
            "flow_rate_m3_s": float(Q_best),
            "mean_velocity_m_s": float(v_best),
            "reynolds_number": float(Re_best),
            "flow_regime": regime,
            "laminar": {
                "Q_m3_s": float(Q_laminar),
                "v_m_s": float(v_laminar),
                "Re": float(Re_laminar),
                "friction_factor": float(64.0 / Re_laminar) if Re_laminar > 0 else None,
            },
            "turbulent": {
                "Q_m3_s": float(Q_turb),
                "v_m_s": float(v_turb),
                "Re": float(Re_turb),
                "friction_factor": float(f_turb),
            },
            "delta_P_Pa": float(delta_P),
            "pipe_diameter_m": float(D),
            "pipe_length_m": float(length),
            "relative_roughness": float(rel_rough),
            "model": "Hagen-Poiseuille (laminar) / Darcy-Weisbach + Colebrook (turbulent)",
        }

    def reynolds_number(
        self,
        rho: float,
        velocity: float,
        length: float,
        viscosity: float,
    ) -> dict:
        """Compute Reynolds number and characterise the flow regime.

        Re = rho * v * L / mu

        Parameters
        ----------
        rho:       fluid density [kg/m^3]
        velocity:  characteristic velocity [m/s]
        length:    characteristic length (pipe diameter, plate length…) [m]
        viscosity: dynamic viscosity [Pa·s]
        """
        if viscosity <= 0:
            raise ValueError("viscosity must be positive.")
        Re = rho * velocity * length / viscosity

        if Re < 2300:
            regime = "laminar"
            description = "Smooth, layered flow. Hagen-Poiseuille applies in pipes."
        elif Re < 4000:
            regime = "transitional"
            description = "Unstable — may switch between laminar and turbulent."
        else:
            regime = "turbulent"
            description = "Chaotic mixing. Use turbulence models or Darcy-Weisbach."

        kinematic_visc = viscosity / rho

        return {
            "reynolds_number": float(Re),
            "flow_regime": regime,
            "description": description,
            "rho": rho,
            "velocity": velocity,
            "length": length,
            "dynamic_viscosity": viscosity,
            "kinematic_viscosity": float(kinematic_visc),
        }

    def terminal_velocity(
        self,
        mass: float,
        diameter: float,
        rho_fluid: float,
        Cd: float | None = None,
        g: float = 9.81,
    ) -> dict:
        """Compute terminal velocity of a sphere falling through a fluid.

        Three drag regimes are checked:
        - Stokes (Re < 1): F_d = 3*pi*mu*D*v  →  Cd = 24/Re
        - Intermediate (1 ≤ Re < 1000): Schiller-Naumann correlation
        - Newton (Re ≥ 1000): Cd ≈ 0.44 (sphere)

        If Cd is provided, uses that constant value directly.

        Parameters
        ----------
        mass:      object mass [kg]
        diameter:  sphere diameter [m]
        rho_fluid: fluid density [kg/m^3]
        Cd:        drag coefficient — if None, auto-selects regime
        g:         gravitational acceleration [m/s^2]
        """
        if mass <= 0 or diameter <= 0 or rho_fluid <= 0:
            raise ValueError("mass, diameter, and rho_fluid must be positive.")

        A = math.pi * (diameter / 2) ** 2
        W = mass * g
        rho_obj = mass / (math.pi / 6 * diameter ** 3)
        buoyancy = rho_fluid * (math.pi / 6 * diameter ** 3) * g
        F_net = W - buoyancy

        if F_net <= 0:
            return {
                "terminal_velocity_m_s": 0.0,
                "note": "Object floats — buoyancy exceeds weight.",
                "buoyancy_N": float(buoyancy),
                "weight_N": float(W),
            }

        if Cd is not None:
            vt = math.sqrt(2 * F_net / (Cd * rho_fluid * A))
            regime = "user-specified Cd"
            Re_t = rho_fluid * vt * diameter / 1e-3  # approximate Re (viscosity unknown)
            return {
                "terminal_velocity_m_s": float(vt),
                "reynolds_number_approx": float(Re_t),
                "drag_force_N": float(F_net),
                "Cd_used": float(Cd),
                "regime": regime,
                "note": "Re approximated using water viscosity. Provide viscosity for exact Re.",
            }

        # Stokes terminal velocity (Re << 1 limit)
        # vt_Stokes = W_net * 18 * mu / (rho_f * g * D^2) — needs viscosity
        # We use Newton's law regime as upper bound and iterate via Schiller-Naumann
        # Use Newton regime Cd=0.44 as first estimate
        vt_newton = math.sqrt(2 * F_net / (0.44 * rho_fluid * A))

        # Check which regime applies using Re from Newton estimate
        # Without viscosity we can only give Newton result
        return {
            "terminal_velocity_m_s": float(vt_newton),
            "drag_force_N": float(F_net),
            "weight_N": float(W),
            "buoyancy_N": float(buoyancy),
            "Cd_used": 0.44,
            "regime": "Newton (Cd=0.44 sphere approximation)",
            "object_density_kg_m3": float(rho_obj),
            "note": (
                "Uses Cd=0.44 (turbulent sphere, Re>1000). "
                "For Stokes/Schiller-Naumann regime, use pipe_flow with viscosity."
            ),
        }

    def drag_force(
        self,
        velocity: float,
        diameter: float,
        Cd: float,
        rho_fluid: float = 1.225,
    ) -> dict:
        """Compute aerodynamic/hydrodynamic drag force on an object.

        F_drag = 0.5 * Cd * rho * A * v^2

        Parameters
        ----------
        velocity:   object speed relative to fluid [m/s]
        diameter:   characteristic diameter [m] (used to compute A = pi*(d/2)^2)
        Cd:         drag coefficient
        rho_fluid:  fluid density [kg/m^3] (air at sea level ≈ 1.225)
        """
        if diameter <= 0 or Cd < 0 or rho_fluid <= 0:
            raise ValueError("diameter and rho_fluid must be positive; Cd must be non-negative.")

        A = math.pi * (diameter / 2) ** 2
        F = 0.5 * Cd * rho_fluid * A * velocity ** 2
        power = F * abs(velocity)

        return {
            "drag_force_N": float(F),
            "drag_power_W": float(power),
            "dynamic_pressure_Pa": float(0.5 * rho_fluid * velocity ** 2),
            "cross_sectional_area_m2": float(A),
            "velocity": velocity,
            "Cd": Cd,
            "rho_fluid": rho_fluid,
        }
