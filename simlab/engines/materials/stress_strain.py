"""Stress-strain analysis engine using NumPy.

Provides stress, strain, Young's modulus calculations, elastic deformation
simulation, and stress-strain curve generation.
"""

from __future__ import annotations

import math

import numpy as np


class StressStrainEngine:
    """Mechanical stress-strain analysis."""

    def calculate_stress(self, force: float, area: float) -> float:
        """Calculate engineering stress: σ = F / A.

        Parameters
        ----------
        force: applied force [N]
        area:  cross-sectional area [m^2]

        Returns
        -------
        stress in Pascals [Pa]
        """
        if area <= 0:
            raise ValueError("Area must be positive.")
        return float(force / area)

    def calculate_strain(self, delta_L: float, L0: float) -> float:
        """Calculate engineering strain: ε = ΔL / L0.

        Parameters
        ----------
        delta_L: change in length [m]
        L0:      original length [m]
        """
        if L0 <= 0:
            raise ValueError("Original length L0 must be positive.")
        return float(delta_L / L0)

    def calculate_youngs_modulus(self, stress: float, strain: float) -> float:
        """Calculate Young's modulus: E = σ / ε.

        Parameters
        ----------
        stress: engineering stress [Pa]
        strain: engineering strain (dimensionless)
        """
        if abs(strain) < 1e-15:
            raise ValueError("Strain is effectively zero; Young's modulus is undefined.")
        return float(stress / strain)

    def simulate_elastic_deformation(
        self, E: float, force: float, area: float, L0: float
    ) -> dict:
        """Simulate elastic deformation of a bar under axial load.

        Parameters
        ----------
        E:     Young's modulus [Pa]
        force: applied force [N]
        area:  cross-sectional area [m^2]
        L0:    original length [m]
        """
        stress = self.calculate_stress(force, area)
        strain = stress / E
        delta_L = strain * L0
        spring_constant = E * area / L0
        strain_energy = 0.5 * stress * strain * area * L0

        return {
            "stress": float(stress),
            "strain": float(strain),
            "delta_L": float(delta_L),
            "new_length": float(L0 + delta_L),
            "spring_constant": float(spring_constant),
            "strain_energy": float(strain_energy),
            "E": E,
            "force": force,
            "area": area,
            "L0": L0,
        }

    def stress_strain_curve(
        self,
        E: float,
        yield_stress: float,
        ultimate_stress: float,
        fracture_strain: float | None = None,
        n_points: int = 200,
    ) -> dict:
        """Generate an engineering stress-strain curve.

        Regime breakdown:
        - Elastic: 0 to yield_strain (slope = E)
        - Plastic hardening: yield_strain to ultimate_strain (power law)
        - Necking/fracture: ultimate_strain to fracture_strain

        Parameters
        ----------
        E:               Young's modulus [Pa]
        yield_stress:    yield stress [Pa]
        ultimate_stress: ultimate tensile strength [Pa]
        fracture_strain: strain at fracture (default: 2.0 * yield_strain + 0.2)
        """
        yield_strain = yield_stress / E
        # Plastic region ends at some larger strain where UTS is reached
        ultimate_strain = yield_strain + (ultimate_stress - yield_stress) / (
            0.1 * E
        )  # approximate
        if fracture_strain is None:
            fracture_strain = ultimate_strain * 1.3

        # Build piecewise curve
        n_elastic = n_points // 4
        n_plastic = n_points // 2
        n_fracture = n_points - n_elastic - n_plastic

        # Elastic region
        strain_e = np.linspace(0, yield_strain, n_elastic)
        stress_e = E * strain_e

        # Plastic hardening (Hollomon power law: σ = σ_y * (ε/ε_y)^n)
        n_exp = 0.2  # strain hardening exponent
        strain_p = np.linspace(yield_strain, ultimate_strain, n_plastic)
        stress_p = yield_stress * (strain_p / yield_strain) ** n_exp

        # Necking to fracture (linear drop from UTS)
        strain_f = np.linspace(ultimate_strain, fracture_strain, n_fracture)
        stress_f = np.linspace(ultimate_stress, 0.5 * ultimate_stress, n_fracture)

        strain_arr = np.concatenate([strain_e, strain_p, strain_f])
        stress_arr = np.concatenate([stress_e, stress_p, stress_f])

        return {
            "strain": strain_arr.tolist(),
            "stress": stress_arr.tolist(),
            "yield_stress": float(yield_stress),
            "yield_strain": float(yield_strain),
            "ultimate_stress": float(ultimate_stress),
            "ultimate_strain": float(ultimate_strain),
            "fracture_strain": float(fracture_strain),
            "E": E,
            "toughness_approx": float(np.trapz(stress_arr, strain_arr)),
            "resilience": float(0.5 * yield_stress * yield_strain),
        }

    def elastic_plastic_deformation(
        self,
        E: float,
        yield_stress: float,
        E_tangent: float,
        force: float,
        area: float,
        L0: float,
    ) -> dict:
        """Simulate elastic-perfectly-plastic and bilinear-hardening deformation.

        Uses a bilinear kinematic hardening model:
        - Elastic:  sigma = E * epsilon  (for sigma < yield_stress)
        - Plastic:  sigma = yield_stress + E_tangent * (epsilon - yield_strain)

        Parameters
        ----------
        E:            Young's modulus [Pa]
        yield_stress: yield strength [Pa]
        E_tangent:    tangent modulus in plastic region [Pa]
                      (0 = perfectly plastic, ~E/100 typical mild steel, ~E/10 aluminum)
        force:        applied axial force [N]
        area:         cross-sectional area [m^2]
        L0:           original length [m]
        """
        if E <= 0 or yield_stress <= 0 or area <= 0 or L0 <= 0:
            raise ValueError("E, yield_stress, area, and L0 must be positive.")
        if E_tangent < 0:
            raise ValueError("E_tangent must be non-negative.")

        sigma = force / area
        yield_strain = yield_stress / E

        if abs(sigma) <= yield_stress:
            regime = "elastic"
            epsilon = sigma / E
            sigma_true = sigma
        else:
            regime = "plastic"
            sign = 1.0 if sigma > 0 else -1.0
            epsilon = yield_strain + (abs(sigma) - yield_stress) / (E_tangent + 1e-15) * sign if E_tangent > 0 else yield_strain * sign
            sigma_true = sigma  # engineering stress applied

        delta_L = epsilon * L0
        strain_energy = (
            0.5 * yield_stress * yield_strain * area * L0
            + (abs(epsilon) - yield_strain) * (yield_stress + 0.5 * E_tangent * (abs(epsilon) - yield_strain)) * area * L0
            if regime == "plastic" else
            0.5 * sigma * epsilon * area * L0
        )

        # True (Cauchy) stress and logarithmic strain
        true_strain = math.log(1.0 + epsilon) if epsilon > -1 else float("nan")
        true_stress = sigma * (1.0 + epsilon)

        return {
            "regime": regime,
            "engineering_stress_Pa": float(sigma),
            "engineering_strain": float(epsilon),
            "true_stress_Pa": float(true_stress),
            "true_strain": float(true_strain),
            "delta_L_m": float(delta_L),
            "new_length_m": float(L0 + delta_L),
            "strain_energy_J": float(strain_energy),
            "yield_stress_Pa": yield_stress,
            "yield_strain": float(yield_strain),
            "is_yielded": regime == "plastic",
            "E": E,
            "E_tangent": E_tangent,
            "overstress_Pa": float(abs(sigma) - yield_stress) if regime == "plastic" else 0.0,
        }

    def ramberg_osgood(
        self,
        E: float,
        K: float,
        n_exp: float,
        stress_max: float,
        n_points: int = 300,
    ) -> dict:
        """Generate a Ramberg-Osgood nonlinear stress-strain curve.

        epsilon = sigma/E + (sigma/K)^n

        This is the standard engineering model for metals with continuous
        yielding (no distinct yield point): aluminum alloys, titanium, etc.

        Parameters
        ----------
        E:          Young's modulus [Pa]
        K:          strength coefficient [Pa]  (K ≈ yield_stress * (E/yield_stress)^(1/n))
        n_exp:      strain hardening exponent (typical range 3–20 for metals;
                    higher n → sharper knee)
        stress_max: maximum stress to evaluate [Pa]
        """
        if E <= 0 or K <= 0 or n_exp <= 0 or stress_max <= 0:
            raise ValueError("E, K, n_exp, and stress_max must be positive.")

        sigma_arr = np.linspace(0, stress_max, n_points)
        epsilon_elastic = sigma_arr / E
        epsilon_plastic = (sigma_arr / K) ** n_exp
        epsilon_total = epsilon_elastic + epsilon_plastic

        # 0.2% offset yield strength (sigma where epsilon_plastic = 0.002)
        try:
            from scipy.optimize import brentq as _brentq
            def f_02(s):
                return (s / K) ** n_exp - 0.002
            sigma_02 = _brentq(f_02, 0, stress_max * 2, xtol=1e-3) if (K * 0.002 ** (1 / n_exp)) <= stress_max * 2 else float("nan")
        except Exception:
            sigma_02 = float("nan")

        # Secant modulus at max stress
        E_secant = float(stress_max / epsilon_total[-1]) if epsilon_total[-1] > 0 else E

        return {
            "stress_Pa": sigma_arr.tolist(),
            "strain_total": epsilon_total.tolist(),
            "strain_elastic": epsilon_elastic.tolist(),
            "strain_plastic": epsilon_plastic.tolist(),
            "yield_stress_02pct_Pa": float(sigma_02),
            "E_initial_Pa": E,
            "E_secant_at_max_Pa": float(E_secant),
            "K_Pa": K,
            "n_exponent": n_exp,
            "stress_max_Pa": stress_max,
            "model": "Ramberg-Osgood: ε = σ/E + (σ/K)^n",
        }

    def poisson_ratio_deformation(
        self, E: float, nu: float, force: float, area: float, L0: float, width0: float
    ) -> dict:
        """Compute lateral contraction from Poisson's ratio.

        ε_lateral = -ν * ε_axial

        Parameters
        ----------
        E:      Young's modulus [Pa]
        nu:     Poisson's ratio (0 to 0.5)
        force:  axial force [N]
        area:   cross-sectional area [m^2]
        L0:     original length [m]
        width0: original width/diameter [m]
        """
        stress = force / area
        strain_axial = stress / E
        strain_lateral = -nu * strain_axial
        delta_L = strain_axial * L0
        delta_width = strain_lateral * width0

        return {
            "stress": float(stress),
            "strain_axial": float(strain_axial),
            "strain_lateral": float(strain_lateral),
            "delta_L": float(delta_L),
            "delta_width": float(delta_width),
            "new_length": float(L0 + delta_L),
            "new_width": float(width0 + delta_width),
            "poisson_ratio": nu,
        }
