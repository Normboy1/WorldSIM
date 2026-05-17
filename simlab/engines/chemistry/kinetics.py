"""Chemical reaction kinetics engine using SciPy.

Supports first/second order decay, consecutive reactions,
Michaelis-Menten enzyme kinetics, and reversible equilibrium.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


class KineticsEngine:
    """ODE-based chemical kinetics simulations."""

    def simulate_first_order(
        self,
        k: float,
        A0: float,
        t_span: float = 50.0,
        t_points: int = 200,
    ) -> dict:
        """Simulate first-order decay: dA/dt = -k*A.

        Analytical solution: A(t) = A0 * exp(-k*t)
        """
        if k < 0:
            raise ValueError("rate constant k must be non-negative for first-order decay.")
        if A0 < 0:
            raise ValueError("initial concentration A0 must be non-negative.")

        t_arr = np.linspace(0, t_span, t_points)
        A_arr = A0 * np.exp(-k * t_arr)
        half_life = math.log(2) / k if k > 0 else float("inf")
        t_99 = math.log(100) / k if k > 0 else float("inf")

        return {
            "t": t_arr.tolist(),
            "A": A_arr.tolist(),
            "half_life": float(half_life),
            "t_99_percent": float(t_99),
            "k": k,
            "A0": A0,
            "order": 1,
            "analytical": True,
        }

    def simulate_second_order(
        self,
        k: float,
        A0: float,
        t_span: float = 50.0,
        t_points: int = 200,
    ) -> dict:
        """Simulate second-order decay: dA/dt = -k*A^2.

        Analytical solution: 1/A(t) = 1/A0 + k*t
        """
        if k < 0:
            raise ValueError("rate constant k must be non-negative.")
        if A0 < 0:
            raise ValueError("initial concentration A0 must be non-negative.")

        t_arr = np.linspace(0, t_span, t_points)
        A_arr = A0 / (1 + k * A0 * t_arr)
        half_life = 1.0 / (k * A0) if k > 0 and A0 > 0 else float("inf")

        return {
            "t": t_arr.tolist(),
            "A": A_arr.tolist(),
            "half_life": float(half_life),
            "k": k,
            "A0": A0,
            "order": 2,
            "analytical": True,
        }

    def simulate_consecutive_reactions(
        self,
        k1: float,
        k2: float,
        A0: float,
        t_span: float = 50.0,
        t_points: int = 300,
    ) -> dict:
        """Simulate consecutive first-order reactions: A --k1--> B --k2--> C.

        Analytical solution:
            A(t) = A0 * exp(-k1*t)
            B(t) = A0 * k1/(k2-k1) * (exp(-k1*t) - exp(-k2*t))  [k1 != k2]
            C(t) = A0 - A(t) - B(t)
        """
        if k1 < 0 or k2 < 0:
            raise ValueError("rate constants k1 and k2 must be non-negative.")
        if A0 < 0:
            raise ValueError("initial concentration A0 must be non-negative.")

        t_arr = np.linspace(0, t_span, t_points)
        A_arr = A0 * np.exp(-k1 * t_arr)

        if abs(k1 - k2) < 1e-12:
            # Degenerate case: k1 == k2
            B_arr = k1 * t_arr * A0 * np.exp(-k1 * t_arr)
        else:
            B_arr = A0 * k1 / (k2 - k1) * (np.exp(-k1 * t_arr) - np.exp(-k2 * t_arr))

        C_arr = A0 - A_arr - B_arr

        # Time of maximum B concentration
        if abs(k1 - k2) > 1e-12 and k1 > 0 and k2 > 0:
            t_Bmax = math.log(k1 / k2) / (k1 - k2)
        else:
            t_Bmax = 1.0 / k1 if k1 > 0 else 0.0

        B_max = float(A0 * k1 / (k2 - k1) * (
            math.exp(-k1 * t_Bmax) - math.exp(-k2 * t_Bmax)
        )) if abs(k1 - k2) > 1e-12 else float(A0 * math.exp(-1))

        return {
            "t": t_arr.tolist(),
            "A": A_arr.tolist(),
            "B": np.maximum(B_arr, 0).tolist(),
            "C": np.maximum(C_arr, 0).tolist(),
            "t_Bmax": float(t_Bmax),
            "B_max": float(B_max),
            "k1": k1,
            "k2": k2,
            "A0": A0,
            "analytical": True,
        }

    def simulate_michaelis_menten(
        self,
        Vmax: float,
        Km: float,
        S0: float,
        t_span: float = 100.0,
        t_points: int = 300,
    ) -> dict:
        """Simulate Michaelis-Menten enzyme kinetics: dS/dt = -Vmax*S/(Km+S).

        Numerically integrated (no closed-form solution in t).

        Parameters
        ----------
        Vmax: maximum reaction velocity [concentration/time]
        Km:   Michaelis constant [concentration]
        S0:   initial substrate concentration [concentration]
        """
        if Vmax < 0 or Km <= 0 or S0 < 0:
            raise ValueError("Require Vmax >= 0, Km > 0, and S0 >= 0 for Michaelis-Menten kinetics.")

        def ode(t, y):
            S = max(y[0], 0.0)
            dS = -Vmax * S / (Km + S)
            return [dS]

        t_eval = np.linspace(0, t_span, t_points)
        sol = solve_ivp(ode, [0, t_span], [S0], t_eval=t_eval, rtol=1e-8, atol=1e-10)

        # Initial rate
        v0 = Vmax * S0 / (Km + S0)
        # Time to reach half of S0 (numerical approximation)
        S_arr = sol.y[0]
        half_s_idx = np.searchsorted(-S_arr, -S0 / 2)
        t_half = float(sol.t[min(half_s_idx, len(sol.t) - 1)])

        return {
            "t": sol.t.tolist(),
            "S": sol.y[0].tolist(),
            "initial_rate": float(v0),
            "t_half_substrate": float(t_half),
            "Vmax": Vmax,
            "Km": Km,
            "S0": S0,
            "success": bool(sol.success),
        }

    def calculate_half_life(self, k: float, order: int = 1, A0: float = 1.0) -> float:
        """Compute the half-life for first or second order reactions.

        Parameters
        ----------
        k:     rate constant
        order: 1 or 2
        A0:    initial concentration (only relevant for order=2)
        """
        if order == 1:
            return math.log(2) / k
        elif order == 2:
            return 1.0 / (k * A0)
        else:
            raise ValueError(f"Unsupported reaction order: {order}. Use 1 or 2.")

    def simulate_equilibrium(
        self,
        k_forward: float,
        k_reverse: float,
        A0: float,
        B0: float,
        t_span: float = 100.0,
        t_points: int = 300,
    ) -> dict:
        """Simulate reversible first-order equilibrium: A <--k_f/k_r--> B.

        ODE system:
            dA/dt = -k_f*A + k_r*B
            dB/dt =  k_f*A - k_r*B
        """
        if k_forward < 0 or k_reverse < 0:
            raise ValueError("rate constants k_forward and k_reverse must be non-negative.")
        if A0 < 0 or B0 < 0:
            raise ValueError("initial concentrations A0 and B0 must be non-negative.")

        def ode(t, y):
            A, B = y
            dA = -k_forward * A + k_reverse * B
            dB = k_forward * A - k_reverse * B
            return [dA, dB]

        t_eval = np.linspace(0, t_span, t_points)
        sol = solve_ivp(ode, [0, t_span], [A0, B0], t_eval=t_eval, rtol=1e-8, atol=1e-10)

        # Equilibrium concentrations (analytical)
        K_eq = k_forward / k_reverse if k_reverse > 0 else float("inf")
        total = A0 + B0
        A_eq_analytical = total / (1 + K_eq) if K_eq != float("inf") else 0.0
        B_eq_analytical = total - A_eq_analytical

        return {
            "t": sol.t.tolist(),
            "A": sol.y[0].tolist(),
            "B": sol.y[1].tolist(),
            "K_eq": float(K_eq),
            "A_equilibrium": float(A_eq_analytical),
            "B_equilibrium": float(B_eq_analytical),
            "k_forward": k_forward,
            "k_reverse": k_reverse,
            "A0": A0,
            "B0": B0,
            "success": bool(sol.success),
        }
