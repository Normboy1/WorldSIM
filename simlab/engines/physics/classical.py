"""Classical mechanics simulation engine.

Provides projectile motion, pendulum, spring-mass, collision, gravity,
and circular motion simulations using NumPy + SciPy.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from simlab.core.constants.physical import CONSTANTS


class ClassicalMechanicsEngine:
    """Classical mechanics simulations."""

    def simulate_projectile_motion(
        self,
        v0: float,
        angle_deg: float,
        g: float = 9.81,
        h0: float = 0.0,
        n_points: int = 300,
    ) -> dict:
        """Simulate 2D projectile motion (no air resistance).

        Parameters
        ----------
        v0:        initial speed [m/s]
        angle_deg: launch angle from horizontal [degrees]
        g:         gravitational acceleration [m/s^2]
        h0:        initial height [m]
        n_points:  number of time points for output arrays
        """
        if g <= 0:
            raise ValueError("gravitational acceleration g must be positive.")
        if v0 < 0:
            raise ValueError("initial speed v0 must be non-negative.")

        angle_rad = math.radians(angle_deg)
        vx = v0 * math.cos(angle_rad)
        vy = v0 * math.sin(angle_rad)

        # Time until trajectory intersects y = 0 (ground): h0 + vy*t - (g/2)*t^2 = 0
        # Positive root: t = (vy + sqrt(vy^2 + 2*g*h0)) / g  (launch from height h0 >= 0)
        discriminant = vy**2 + 2 * g * h0
        if discriminant < 0:
            raise ValueError("Initial height and velocity produce no real landing time (y=0).")
        t_flight = (vy + math.sqrt(discriminant)) / g
        if t_flight < 0:
            raise ValueError(
                "Negative time of flight: check launch angle, v0, and initial height."
            )
        t_flight = float(max(t_flight, 0.0))

        t_arr = np.linspace(0, t_flight, n_points)
        x_arr = vx * t_arr
        y_arr = h0 + vy * t_arr - 0.5 * g * t_arr**2

        # Clamp negative y to 0 (physical ground)
        y_arr = np.maximum(y_arr, 0.0)

        max_height = float(h0 + vy**2 / (2 * g)) if vy > 0 else float(h0)
        t_apex = vy / g if vy > 0 else 0.0
        horizontal_range = float(vx * t_flight)

        vx_arr = np.full(n_points, vx)
        vy_arr = vy - g * t_arr
        speed_arr = np.sqrt(vx_arr**2 + vy_arr**2)
        ke_arr = 0.5 * (vx**2 + (vy - g * t_arr)**2)  # per unit mass

        return {
            "t": t_arr.tolist(),
            "x": x_arr.tolist(),
            "y": y_arr.tolist(),
            "vx": vx_arr.tolist(),
            "vy": vy_arr.tolist(),
            "speed": speed_arr.tolist(),
            "kinetic_energy_per_kg": ke_arr.tolist(),
            "max_height": max_height,
            "range": horizontal_range,
            "time_of_flight": float(t_flight),
            "t_apex": float(t_apex),
            "v0": v0,
            "angle_deg": angle_deg,
            "g": g,
            "h0": h0,
        }

    def simulate_pendulum(
        self,
        length: float,
        theta0_deg: float,
        t_span: float = 10.0,
        damping: float = 0.0,
        g: float = 9.81,
        n_points: int = 500,
    ) -> dict:
        """Simulate a nonlinear (exact) pendulum using the full ODE.

        State: [theta, omega]  (angle in radians, angular velocity)
        ODE:  theta'' = -(g/L)*sin(theta) - damping*theta'

        Parameters
        ----------
        length:     pendulum length [m]
        theta0_deg: initial angle [degrees]
        t_span:     simulation duration [s]
        damping:    damping coefficient
        g:          gravitational acceleration [m/s^2]
        """
        if length <= 0:
            raise ValueError("pendulum length must be positive.")
        if g <= 0:
            raise ValueError("gravitational acceleration g must be positive.")

        theta0 = math.radians(theta0_deg)
        omega0 = 0.0  # released from rest

        def pendulum_ode(t, state):
            theta, omega = state
            dtheta = omega
            domega = -(g / length) * math.sin(theta) - damping * omega
            return [dtheta, domega]

        t_eval = np.linspace(0, t_span, n_points)
        sol = solve_ivp(
            pendulum_ode,
            [0, t_span],
            [theta0, omega0],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
        )

        theta_arr = sol.y[0]
        omega_arr = sol.y[1]
        x_arr = length * np.sin(theta_arr)
        y_arr = -length * np.cos(theta_arr)

        # Small-angle period
        T_linear = 2 * math.pi * math.sqrt(length / g)
        omega_natural = math.sqrt(g / length)

        return {
            "t": sol.t.tolist(),
            "theta_rad": theta_arr.tolist(),
            "theta_deg": np.degrees(theta_arr).tolist(),
            "omega": omega_arr.tolist(),
            "x": x_arr.tolist(),
            "y": y_arr.tolist(),
            "period_linear_approx": float(T_linear),
            "natural_frequency": float(omega_natural),
            "length": length,
            "theta0_deg": theta0_deg,
            "damping": damping,
            "success": bool(sol.success),
        }

    def simulate_spring_mass(
        self,
        mass: float,
        k: float,
        x0: float,
        v0: float,
        t_span: float = 10.0,
        damping: float = 0.0,
        n_points: int = 500,
    ) -> dict:
        """Simulate a damped harmonic oscillator: m*x'' + c*x' + k*x = 0.

        Parameters
        ----------
        mass:    mass [kg]
        k:       spring constant [N/m]
        x0:      initial displacement [m]
        v0:      initial velocity [m/s]
        t_span:  simulation time [s]
        damping: damping coefficient c [N·s/m]
        """
        if mass <= 0:
            raise ValueError("mass must be positive.")
        if k < 0:
            raise ValueError("spring constant k must be non-negative.")

        omega_n = math.sqrt(k / mass) if k > 0 else 0.0
        zeta = damping / (2 * math.sqrt(mass * k)) if k > 0 and mass > 0 else 0.0

        def ode(t, state):
            x, xp = state
            xpp = (-k * x - damping * xp) / mass
            return [xp, xpp]

        t_eval = np.linspace(0, t_span, n_points)
        sol = solve_ivp(ode, [0, t_span], [x0, v0], t_eval=t_eval, rtol=1e-9, atol=1e-11)

        x_arr = sol.y[0]
        v_arr = sol.y[1]
        ke_arr = 0.5 * mass * v_arr**2
        pe_arr = 0.5 * k * x_arr**2
        te_arr = ke_arr + pe_arr

        if k == 0:
            regime = "no_restoring_force"
        elif zeta < 1:
            regime = "underdamped"
        elif abs(zeta - 1) < 1e-6:
            regime = "critically_damped"
        else:
            regime = "overdamped"

        return {
            "t": sol.t.tolist(),
            "x": x_arr.tolist(),
            "v": v_arr.tolist(),
            "kinetic_energy": ke_arr.tolist(),
            "potential_energy": pe_arr.tolist(),
            "total_energy": te_arr.tolist(),
            "natural_frequency": float(omega_n),
            "damping_ratio": float(zeta),
            "regime": regime,
            "period": float(2 * math.pi / omega_n) if omega_n > 0 else None,
            "success": bool(sol.success),
        }

    def simulate_collision(
        self,
        m1: float,
        v1: float,
        m2: float,
        v2: float,
        collision_type: str = "elastic",
    ) -> dict:
        """Simulate a 1D collision between two objects.

        Parameters
        ----------
        m1, v1: mass and initial velocity of object 1
        m2, v2: mass and initial velocity of object 2
        collision_type: 'elastic' or 'perfectly_inelastic'
        """
        if m1 <= 0 or m2 <= 0:
            raise ValueError("masses m1 and m2 must be positive.")
        denom = m1 + m2
        if denom <= 0:
            raise ValueError("Total mass m1 + m2 must be positive.")

        p_initial = m1 * v1 + m2 * v2
        ke_initial = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2

        if collision_type == "elastic":
            # Elastic collision: both momentum and KE conserved
            v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / denom
            v2_final = ((m2 - m1) * v2 + 2 * m1 * v1) / denom
        elif collision_type == "perfectly_inelastic":
            # Objects stick together
            v_final = (m1 * v1 + m2 * v2) / denom
            v1_final = v_final
            v2_final = v_final
        else:
            raise ValueError(f"Unknown collision type: {collision_type!r}")

        p_final = m1 * v1_final + m2 * v2_final
        ke_final = 0.5 * m1 * v1_final**2 + 0.5 * m2 * v2_final**2
        ke_lost = ke_initial - ke_final

        return {
            "v1_initial": v1,
            "v2_initial": v2,
            "v1_final": float(v1_final),
            "v2_final": float(v2_final),
            "momentum_initial": float(p_initial),
            "momentum_final": float(p_final),
            "ke_initial": float(ke_initial),
            "ke_final": float(ke_final),
            "ke_lost": float(ke_lost),
            "momentum_conserved": bool(abs(p_initial - p_final) < 1e-9),
            "type": collision_type,
            "coefficient_of_restitution": abs(v2_final - v1_final) / abs(v1 - v2) if abs(v1 - v2) > 1e-15 else 1.0,
        }

    def simulate_gravity(self, m1: float, m2: float, r: float) -> dict:
        """Compute gravitational force and potential energy between two masses.

        Parameters
        ----------
        m1, m2: masses [kg]
        r:      separation distance [m]
        """
        if r <= 0:
            raise ValueError("separation distance r must be positive.")
        if m1 <= 0 or m2 <= 0:
            raise ValueError("masses m1 and m2 must be positive.")

        G = CONSTANTS.G
        force = G * m1 * m2 / r**2
        potential_energy = -G * m1 * m2 / r
        escape_velocity_from_m1 = math.sqrt(2 * G * m1 / r)

        return {
            "gravitational_force": float(force),
            "potential_energy": float(potential_energy),
            "escape_velocity_from_m1": float(escape_velocity_from_m1),
            "G": G,
            "m1": m1,
            "m2": m2,
            "r": r,
        }

    def simulate_projectile_with_drag(
        self,
        v0: float,
        angle_deg: float,
        mass: float,
        diameter: float,
        Cd: float = 0.47,
        h0: float = 0.0,
        use_isa_atmosphere: bool = True,
        rho_air: float = 1.225,
        g: float = 9.81,
        n_points: int = 500,
    ) -> dict:
        """Simulate projectile motion with quadratic aerodynamic drag.

        Uses the ODE system:
            dx/dt  = vx,  dy/dt  = vy
            dvx/dt = -(F_drag/m) * vx/v
            dvy/dt = -g - (F_drag/m) * vy/v
        where F_drag = 0.5 * Cd * rho(y) * A * v^2.

        When use_isa_atmosphere=True, air density decreases with altitude using
        the International Standard Atmosphere model.

        Parameters
        ----------
        v0:               initial speed [m/s]
        angle_deg:        launch angle [degrees]
        mass:             projectile mass [kg]
        diameter:         projectile diameter [m]
        Cd:               drag coefficient (sphere≈0.47, golf ball≈0.25, bullet≈0.1)
        h0:               launch height [m]
        use_isa_atmosphere: vary air density with altitude
        rho_air:          constant air density [kg/m^3] when ISA is off
        g:                gravitational acceleration [m/s^2]
        """
        if mass <= 0:
            raise ValueError("mass must be positive.")
        if diameter <= 0:
            raise ValueError("diameter must be positive.")
        if v0 < 0:
            raise ValueError("v0 must be non-negative.")
        if Cd < 0:
            raise ValueError("Cd must be non-negative.")

        A = math.pi * (diameter / 2) ** 2

        def _isa_rho(h: float) -> float:
            T0, L, R_air, rho0 = 288.15, 0.0065, 287.05, 1.225
            h = max(0.0, h)
            if h <= 11000.0:
                T = T0 - L * h
                return rho0 * (T / T0) ** (g / (R_air * L))
            T_strat, rho_11k = 216.65, rho0 * (216.65 / T0) ** (g / (R_air * L))
            return rho_11k * math.exp(-g * (h - 11000.0) / (R_air * T_strat))

        angle_rad = math.radians(angle_deg)
        vx0 = v0 * math.cos(angle_rad)
        vy0 = v0 * math.sin(angle_rad)

        def ode(t, state):
            _, y, vx, vy = state
            speed = math.sqrt(vx ** 2 + vy ** 2)
            if speed < 1e-12:
                return [vx, vy, 0.0, -g]
            rho = _isa_rho(y) if use_isa_atmosphere else rho_air
            drag_a = 0.5 * Cd * rho * A * speed ** 2 / mass
            return [vx, vy, -drag_a * vx / speed, -g - drag_a * vy / speed]

        def hit_ground(t, state):
            return state[1]
        hit_ground.terminal = True
        hit_ground.direction = -1

        sol = solve_ivp(
            ode, [0, 1e6], [0.0, h0, vx0, vy0],
            events=hit_ground, max_step=0.5, rtol=1e-8, atol=1e-10,
        )

        t_arr, x_arr = sol.t, sol.y[0]
        y_arr = np.maximum(sol.y[1], 0.0)

        # Vacuum reference
        disc = vy0 ** 2 + 2 * g * h0
        t_vac = (vy0 + math.sqrt(max(disc, 0.0))) / g if disc >= 0 else 0.0
        range_vac = vx0 * t_vac
        max_h_vac = h0 + vy0 ** 2 / (2 * g) if vy0 > 0 else h0

        drag_reduction_pct = (
            100.0 * (range_vac - float(x_arr[-1])) / range_vac
            if range_vac > 1e-9 else 0.0
        )

        t_out = np.linspace(0, t_arr[-1], n_points)
        return {
            "t": t_out.tolist(),
            "x": np.interp(t_out, t_arr, x_arr).tolist(),
            "y": np.maximum(np.interp(t_out, t_arr, y_arr), 0.0).tolist(),
            "range_with_drag": float(x_arr[-1]),
            "range_vacuum": float(range_vac),
            "drag_reduction_percent": float(drag_reduction_pct),
            "max_height_drag": float(np.max(y_arr)),
            "max_height_vacuum": float(max_h_vac),
            "time_of_flight": float(t_arr[-1]),
            "v0": v0, "angle_deg": angle_deg, "mass": mass, "diameter": diameter, "Cd": Cd,
            "model": "quadratic drag + ISA atmosphere" if use_isa_atmosphere else "quadratic drag",
        }

    def relativistic_kinematics(
        self,
        mass: float,
        velocity: float | None = None,
        momentum: float | None = None,
        kinetic_energy: float | None = None,
    ) -> dict:
        """Compute special-relativistic kinematics.

        Provide exactly one of velocity [m/s], momentum [kg·m/s], or kinetic_energy [J].
        Returns Lorentz factor, total energy, time dilation, length contraction,
        and comparison with classical (Newtonian) values.
        """
        c = CONSTANTS.c
        if mass <= 0:
            raise ValueError("mass must be positive.")
        given = sum(x is not None for x in [velocity, momentum, kinetic_energy])
        if given != 1:
            raise ValueError("Provide exactly one of velocity, momentum, or kinetic_energy.")

        if velocity is not None:
            if abs(float(velocity)) >= c:
                raise ValueError(f"velocity must be less than c ({c:.3e} m/s).")
            v = float(velocity)
        elif momentum is not None:
            p = float(momentum)
            v = p * c / math.sqrt((mass * c) ** 2 + p ** 2)
        else:
            KE = float(kinetic_energy)
            gamma_val = KE / (mass * c ** 2) + 1.0
            if gamma_val < 1.0:
                raise ValueError("kinetic_energy implies gamma < 1; check units (must be Joules).")
            v = c * math.sqrt(1.0 - 1.0 / gamma_val ** 2)

        beta = v / c
        gamma = 1.0 / math.sqrt(1.0 - beta ** 2)
        KE_rel = (gamma - 1.0) * mass * c ** 2
        KE_classical = 0.5 * mass * v ** 2

        return {
            "velocity_m_s": float(v),
            "beta": float(beta),
            "gamma": float(gamma),
            "rest_energy_J": float(mass * c ** 2),
            "total_energy_J": float(gamma * mass * c ** 2),
            "kinetic_energy_J": float(KE_rel),
            "momentum_kg_m_s": float(gamma * mass * v),
            "classical_KE_J": float(KE_classical),
            "classical_momentum_kg_m_s": float(mass * v),
            "KE_relativistic_excess_percent": float(100.0 * (KE_rel - KE_classical) / KE_classical) if KE_classical > 0 else 0.0,
            "time_dilation_factor": float(gamma),
            "length_contraction_factor": float(1.0 / gamma),
            "mass_kg": mass,
        }

    def simulate_circular_motion(
        self, mass: float, radius: float, angular_velocity: float
    ) -> dict:
        """Compute parameters of uniform circular motion.

        Parameters
        ----------
        mass:             mass of the orbiting object [kg]
        radius:           orbital radius [m]
        angular_velocity: omega [rad/s]
        """
        if mass <= 0 or radius <= 0:
            raise ValueError("mass and radius must be positive.")

        v_tangential = radius * angular_velocity
        centripetal_acceleration = radius * angular_velocity**2
        centripetal_force = mass * centripetal_acceleration
        period = 2 * math.pi / angular_velocity if angular_velocity != 0 else float("inf")
        frequency = 1.0 / period if period != 0 else 0.0
        ke = 0.5 * mass * v_tangential**2

        return {
            "tangential_velocity": float(v_tangential),
            "centripetal_acceleration": float(centripetal_acceleration),
            "centripetal_force": float(centripetal_force),
            "period": float(period),
            "frequency": float(frequency),
            "kinetic_energy": float(ke),
            "angular_momentum": float(mass * radius * v_tangential),
            "mass": mass,
            "radius": radius,
            "omega": angular_velocity,
        }
