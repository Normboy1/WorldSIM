"""Electromagnetism simulation engine using NumPy.

Provides electric field computation, Coulomb's law, magnetic field
from current-carrying wires, and capacitance calculation.
"""

from __future__ import annotations

import math

import numpy as np

from simlab.core.constants.physical import CONSTANTS


class ElectromagnetismEngine:
    """NumPy-backed electromagnetism simulations."""

    def simulate_electric_field(
        self,
        charges: list[float],
        positions: list[list[float]],
        grid_extent: float = 5.0,
        grid_points: int = 20,
    ) -> dict:
        """Compute the electric field of a set of point charges on a 2D grid.

        Parameters
        ----------
        charges:     list of charge magnitudes [C], positive or negative
        positions:   list of [x, y] positions [m]
        grid_extent: half-width of the grid [m]
        grid_points: number of grid points along each axis
        """
        if len(charges) != len(positions):
            raise ValueError("charges and positions must have the same length.")
        if grid_points < 2:
            raise ValueError("grid_points must be at least 2.")

        k_e = 1 / (4 * math.pi * CONSTANTS.epsilon_0)

        x_lin = np.linspace(-grid_extent, grid_extent, grid_points)
        y_lin = np.linspace(-grid_extent, grid_extent, grid_points)
        X, Y = np.meshgrid(x_lin, y_lin)

        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)

        for q, (px, py) in zip(charges, positions):
            dx = X - px
            dy = Y - py
            r2 = dx**2 + dy**2 + 1e-30  # avoid division by zero
            r = np.sqrt(r2)
            E_mag = k_e * q / r2
            Ex += E_mag * dx / r
            Ey += E_mag * dy / r

        # Electric potential V = sum k*q/r
        V = np.zeros_like(X)
        for q, (px, py) in zip(charges, positions):
            dx = X - px
            dy = Y - py
            r = np.sqrt(dx**2 + dy**2 + 1e-30)
            V += k_e * q / r

        return {
            "X": X.tolist(),
            "Y": Y.tolist(),
            "Ex": Ex.tolist(),
            "Ey": Ey.tolist(),
            "V": V.tolist(),
            "E_magnitude": np.sqrt(Ex**2 + Ey**2).tolist(),
            "charges": charges,
            "positions": positions,
            "grid_extent": grid_extent,
            "k_e": k_e,
        }

    def coulomb_force(self, q1: float, q2: float, r: float) -> dict:
        """Compute the Coulomb force between two point charges separated by r.

        Parameters
        ----------
        q1, q2: charges [C]
        r:      separation [m]
        """
        if r <= 0:
            raise ValueError("separation r must be positive.")
        k_e = 1 / (4 * math.pi * CONSTANTS.epsilon_0)
        force = k_e * q1 * q2 / r**2
        potential_energy = k_e * q1 * q2 / r

        return {
            "force": float(force),
            "force_direction": "attractive" if force < 0 else "repulsive",
            "potential_energy": float(potential_energy),
            "k_e": k_e,
            "q1": q1,
            "q2": q2,
            "r": r,
        }

    def simulate_magnetic_field(
        self,
        current: float,
        wire_positions: list[list[float]],
        grid_extent: float = 5.0,
        grid_points: int = 20,
    ) -> dict:
        """Compute the magnetic field (Bz component) of infinite straight current-carrying wires.

        Each wire is modelled as infinite and perpendicular to the XY plane,
        carrying *current* Amperes. The resulting B field lies in the XY plane.

        Parameters
        ----------
        current:        current magnitude [A] (positive = into page at each wire location)
        wire_positions: list of [x, y] positions [m]
        grid_extent:    half-width of grid [m]
        grid_points:    grid resolution
        """
        if not wire_positions:
            raise ValueError("wire_positions must contain at least one [x, y] pair.")
        if grid_points < 2:
            raise ValueError("grid_points must be at least 2.")

        mu0 = CONSTANTS.mu_0

        x_lin = np.linspace(-grid_extent, grid_extent, grid_points)
        y_lin = np.linspace(-grid_extent, grid_extent, grid_points)
        X, Y = np.meshgrid(x_lin, y_lin)

        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)

        for (wx, wy) in wire_positions:
            dx = X - wx
            dy = Y - wy
            r2 = dx**2 + dy**2 + 1e-30
            # B = (mu0 * I) / (2*pi*r) in phi direction
            # Bx = -(mu0*I)/(2*pi*r^2) * dy,  By = (mu0*I)/(2*pi*r^2) * dx
            factor = (mu0 * current) / (2 * math.pi * r2)
            Bx += -factor * dy
            By += factor * dx

        return {
            "X": X.tolist(),
            "Y": Y.tolist(),
            "Bx": Bx.tolist(),
            "By": By.tolist(),
            "B_magnitude": np.sqrt(Bx**2 + By**2).tolist(),
            "current": current,
            "wire_positions": wire_positions,
        }

    def calculate_capacitance(self, geometry: str, **params) -> dict:
        """Calculate capacitance for standard geometries.

        Supported geometries
        --------------------
        'parallel_plate': requires area [m^2], separation [m], epsilon_r (optional, default 1)
        'cylindrical':    requires length [m], inner_r [m], outer_r [m], epsilon_r (optional)
        'spherical':      requires inner_r [m], outer_r [m], epsilon_r (optional)
        """
        eps0 = CONSTANTS.epsilon_0

        if geometry == "parallel_plate":
            area = float(params["area"])
            d = float(params["separation"])
            if area <= 0 or d <= 0:
                raise ValueError("parallel_plate requires positive area and positive separation.")
            eps_r = float(params.get("epsilon_r", 1.0))
            C = eps_r * eps0 * area / d
            return {
                "capacitance": float(C),
                "geometry": geometry,
                "area": area,
                "separation": d,
                "epsilon_r": eps_r,
                "unit": "F",
            }

        elif geometry == "cylindrical":
            L = float(params["length"])
            r_inner = float(params["inner_r"])
            r_outer = float(params["outer_r"])
            if r_outer <= r_inner or r_inner <= 0 or L <= 0:
                raise ValueError(
                    "cylindrical geometry requires 0 < inner_r < outer_r and positive length."
                )
            eps_r = float(params.get("epsilon_r", 1.0))
            C = 2 * math.pi * eps_r * eps0 * L / math.log(r_outer / r_inner)
            return {
                "capacitance": float(C),
                "geometry": geometry,
                "length": L,
                "inner_r": r_inner,
                "outer_r": r_outer,
                "epsilon_r": eps_r,
                "unit": "F",
            }

        elif geometry == "spherical":
            r_inner = float(params["inner_r"])
            r_outer = float(params["outer_r"])
            if r_outer <= r_inner or r_inner <= 0:
                raise ValueError("spherical geometry requires 0 < inner_r < outer_r.")
            eps_r = float(params.get("epsilon_r", 1.0))
            C = 4 * math.pi * eps_r * eps0 * r_inner * r_outer / (r_outer - r_inner)
            return {
                "capacitance": float(C),
                "geometry": geometry,
                "inner_r": r_inner,
                "outer_r": r_outer,
                "epsilon_r": eps_r,
                "unit": "F",
            }

        else:
            raise ValueError(
                f"Unknown geometry: {geometry!r}. Use 'parallel_plate', 'cylindrical', or 'spherical'."
            )

    def biot_savart_loop(
        self,
        current: float,
        radius: float,
        z_range: list[float] | None = None,
        n_points: int = 100,
    ) -> dict:
        """Compute Bz on the axis of a circular current loop using Biot-Savart.

        Parameters
        ----------
        current: loop current [A]
        radius:  loop radius [m]
        z_range: [z_min, z_max] along axis [m]
        """
        mu0 = CONSTANTS.mu_0
        R = radius
        if R <= 0:
            raise ValueError("loop radius must be positive.")
        if z_range is None:
            z_range = [-3 * R, 3 * R]

        z_arr = np.linspace(z_range[0], z_range[1], n_points)
        Bz_arr = (mu0 * current * R**2) / (2 * (R**2 + z_arr**2) ** 1.5)

        Bz_center = float((mu0 * current) / (2 * R))
        return {
            "z": z_arr.tolist(),
            "Bz": Bz_arr.tolist(),
            "Bz_center": Bz_center,
            "current": current,
            "radius": radius,
        }
