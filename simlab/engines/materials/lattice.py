"""Crystal lattice generation engine using NumPy.

Supports simple cubic (SC), body-centered cubic (BCC), and
face-centered cubic (FCC) lattices.
"""

from __future__ import annotations

import math

import numpy as np


def _tile_basis(basis: list[list[float]], a: float, n_cells: int) -> np.ndarray:
    """Tile a unit cell basis over an n_cells x n_cells x n_cells supercell."""
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                origin = np.array([ix, iy, iz], dtype=float) * a
                for b in basis:
                    positions.append(origin + np.array(b) * a)
    return np.array(positions)


class LatticeEngine:
    """Crystal lattice creation and analysis."""

    def create_simple_cubic(self, a: float, n_cells: int = 2) -> dict:
        """Create a simple cubic (SC) lattice.

        Parameters
        ----------
        a:       lattice constant [Angstroms or any length unit]
        n_cells: number of unit cells along each axis
        """
        basis = [[0, 0, 0]]
        positions = _tile_basis(basis, a, n_cells)
        n_atoms = len(positions)
        volume = (n_cells * a) ** 3
        pf = self.calculate_packing_fraction("sc")

        return {
            "lattice_type": "simple_cubic",
            "a": a,
            "n_cells": n_cells,
            "n_atoms": n_atoms,
            "positions": positions.tolist(),
            "volume": float(volume),
            "packing_fraction": pf,
            "coordination_number": self.calculate_coordination_number("sc"),
            "basis": basis,
        }

    def create_bcc_lattice(self, a: float, n_cells: int = 2) -> dict:
        """Create a body-centered cubic (BCC) lattice.

        Parameters
        ----------
        a:       lattice constant
        n_cells: supercell size
        """
        basis = [[0, 0, 0], [0.5, 0.5, 0.5]]
        positions = _tile_basis(basis, a, n_cells)
        n_atoms = len(positions)
        volume = (n_cells * a) ** 3
        pf = self.calculate_packing_fraction("bcc")

        return {
            "lattice_type": "bcc",
            "a": a,
            "n_cells": n_cells,
            "n_atoms": n_atoms,
            "positions": positions.tolist(),
            "volume": float(volume),
            "packing_fraction": pf,
            "coordination_number": self.calculate_coordination_number("bcc"),
            "basis": basis,
            "nearest_neighbor_distance": float(a * math.sqrt(3) / 2),
        }

    def create_fcc_lattice(self, a: float, n_cells: int = 2) -> dict:
        """Create a face-centered cubic (FCC) lattice.

        Parameters
        ----------
        a:       lattice constant
        n_cells: supercell size
        """
        basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
        positions = _tile_basis(basis, a, n_cells)
        n_atoms = len(positions)
        volume = (n_cells * a) ** 3
        pf = self.calculate_packing_fraction("fcc")

        return {
            "lattice_type": "fcc",
            "a": a,
            "n_cells": n_cells,
            "n_atoms": n_atoms,
            "positions": positions.tolist(),
            "volume": float(volume),
            "packing_fraction": pf,
            "coordination_number": self.calculate_coordination_number("fcc"),
            "basis": basis,
            "nearest_neighbor_distance": float(a / math.sqrt(2)),
        }

    def calculate_packing_fraction(self, lattice_type: str) -> float:
        """Return the theoretical packing fraction for a given lattice type.

        Values:
          SC:  π/6         ≈ 0.5236
          BCC: π√3/8       ≈ 0.6802
          FCC: π/(3√2)     ≈ 0.7405
          HCP: π/(3√2)     ≈ 0.7405
        """
        lt = lattice_type.lower()
        if lt in ("sc", "simple_cubic"):
            return math.pi / 6
        elif lt == "bcc":
            return math.pi * math.sqrt(3) / 8
        elif lt in ("fcc", "hcp"):
            return math.pi / (3 * math.sqrt(2))
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type!r}")

    def calculate_coordination_number(self, lattice_type: str) -> int:
        """Return the coordination number (nearest neighbours) for a lattice type."""
        lt = lattice_type.lower()
        if lt in ("sc", "simple_cubic"):
            return 6
        elif lt == "bcc":
            return 8
        elif lt in ("fcc", "hcp"):
            return 12
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type!r}")

    def interplanar_spacing(
        self, a: float, h: int, k: int, l: int, lattice_type: str = "sc"
    ) -> float:
        """Compute interplanar spacing d_hkl for a cubic lattice (Bragg diffraction).

        d_hkl = a / sqrt(h^2 + k^2 + l^2)  (cubic systems)
        """
        return a / math.sqrt(h**2 + k**2 + l**2)

    def bragg_angle(
        self,
        a: float,
        h: int,
        k: int,
        l: int,
        wavelength: float,
        lattice_type: str = "sc",
    ) -> dict:
        """Compute the Bragg diffraction angle for given Miller indices.

        Uses nλ = 2d sin(θ), with n=1.

        Parameters
        ----------
        wavelength: X-ray wavelength [same units as a]
        """
        d = self.interplanar_spacing(a, h, k, l, lattice_type)
        sin_theta = wavelength / (2 * d)
        if abs(sin_theta) > 1:
            return {"error": "No diffraction: wavelength > 2d_hkl", "d_hkl": float(d)}
        theta_rad = math.asin(sin_theta)
        theta_deg = math.degrees(theta_rad)
        two_theta_deg = 2 * theta_deg

        return {
            "d_hkl": float(d),
            "theta_deg": float(theta_deg),
            "two_theta_deg": float(two_theta_deg),
            "miller_indices": [h, k, l],
            "wavelength": wavelength,
        }
