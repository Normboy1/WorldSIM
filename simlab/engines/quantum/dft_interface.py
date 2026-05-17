"""Density-Functional Theory interface and proxy calculations.

Provides lightweight proxy functions that mimic the outputs of a full DFT
code (e.g., VASP, Quantum ESPRESSO, GPAW) without performing real ab-initio
calculations.  Useful for workflow scaffolding, unit testing, and ML surrogate
training data generation.

Optional dependencies: ase (Atomic Simulation Environment), pymatgen.
Falls back to empirical / tabulated estimates.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import ase  # noqa: F401
    _ASE = True
except ImportError:
    _ASE = False

try:
    import pymatgen  # noqa: F401
    _PYMATGEN = True
except ImportError:
    _PYMATGEN = False


def lda_exchange_correlation(rho: float = 1e28) -> dict[str, Any]:
    """LDA exchange-correlation energy density (Dirac / VWN approximation).

    e_xc(rho) ≈ e_x(rho) + e_c(rho)
    e_x = -3/4 * (3/pi)^(1/3) * rho^(1/3)   [Dirac exchange, atomic units]

    Args:
        rho: Electron number density (electrons/m^3).

    Returns:
        Dict with keys: e_x_eV, e_c_eV, e_xc_eV, rho, backend, note.
    """
    # Convert to atomic units (a0^{-3})
    a0_m = 5.29177e-11
    rho_au = rho * a0_m ** 3
    e_x_au = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0) * rho_au ** (1.0 / 3.0)
    # VWN correlation (simplified)
    rs = (3.0 / (4.0 * math.pi * rho_au)) ** (1.0 / 3.0)
    e_c_au = -0.0311 * math.log(rs) + 0.0584 if rs < 1 else -0.144 / (1.0 + 7.8 * rs)
    Ha_to_eV = 27.2114
    return {
        "backend": "analytic", "note": "",
        "e_x_eV": e_x_au * Ha_to_eV,
        "e_c_eV": e_c_au * Ha_to_eV,
        "e_xc_eV": (e_x_au + e_c_au) * Ha_to_eV,
        "rho": rho,
    }


def band_gap_estimate(
    formula: str = "Si",
    structure_type: str = "diamond",
) -> dict[str, Any]:
    """Tabulated / heuristic band gap proxy for common semiconductors.

    Args:
        formula: Chemical formula string (e.g., 'Si', 'GaAs', 'TiO2').
        structure_type: Crystal structure (e.g., 'diamond', 'zincblende',
            'rutile', 'rocksalt').

    Returns:
        Dict with keys: band_gap_eV, gap_type, formula, structure_type,
        backend, note.
    """
    _GAPS = {
        "Si": (1.12, "indirect"), "Ge": (0.66, "indirect"), "GaAs": (1.42, "direct"),
        "GaN": (3.44, "direct"), "ZnO": (3.37, "direct"), "TiO2": (3.05, "indirect"),
        "AlAs": (2.16, "indirect"), "InP": (1.35, "direct"), "CdS": (2.42, "direct"),
        "SiC": (2.36, "indirect"), "Diamond": (5.47, "indirect"), "MoS2": (1.85, "direct"),
    }
    key = formula.capitalize()
    if key in _GAPS:
        gap, gap_type = _GAPS[key]
        note = ""
    else:
        # Rough heuristic
        gap = 2.0
        gap_type = "unknown"
        note = f"Formula '{formula}' not in table; using heuristic 2.0 eV. Install pymatgen for lookup."

    return {
        "backend": "stub" if note else "analytic",
        "note": note,
        "band_gap_eV": gap, "gap_type": gap_type,
        "formula": formula, "structure_type": structure_type,
    }


def density_of_states_proxy(
    formula: str = "Si",
    E_range: tuple[float, float] = (-5.0, 5.0),
    smearing_eV: float = 0.1,
) -> dict[str, Any]:
    """Return a proxy Gaussian-broadened density of states.

    Generates a plausible DOS shape using tabulated band gap and a simple
    parabolic band model broadened by Gaussian smearing.

    Args:
        formula: Chemical formula.
        E_range: Energy window (eV) relative to Fermi level.
        smearing_eV: Gaussian smearing width (eV).

    Returns:
        Dict with keys: E_eV, DOS_per_eV, band_gap_eV, backend, note.
    """
    bg = band_gap_estimate(formula)["band_gap_eV"]
    E = np.linspace(E_range[0], E_range[1], 300)
    # Valence band: negative energies, conduction band: positive
    DOS = np.zeros_like(E)
    # Parabolic band density of states ~ sqrt(|E - edge|)
    vb_mask = E < -bg / 2
    cb_mask = E > bg / 2
    DOS[vb_mask] = np.sqrt(np.abs(E[vb_mask] + bg / 2) + 1e-3)
    DOS[cb_mask] = np.sqrt(np.abs(E[cb_mask] - bg / 2) + 1e-3) * 1.2

    # Gaussian broadening
    DOS_smeared = np.zeros_like(DOS)
    for i, E_i in enumerate(E):
        g = np.exp(-((E - E_i) / smearing_eV) ** 2)
        DOS_smeared[i] = np.dot(DOS, g) * (E[1] - E[0])

    return {
        "backend": "stub",
        "note": "Install ase/pymatgen + DFT code for real DOS calculation.",
        "E_eV": E.tolist(), "DOS_per_eV": DOS_smeared.tolist(),
        "band_gap_eV": bg,
    }


def formation_energy_proxy(
    formula: str = "TiO2",
    reference_energies: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Proxy formation energy using tabulated values.

    Args:
        formula: Compound formula.
        reference_energies: Dict of elemental reference energies (eV/atom).
            Defaults to DFT-PBE standard references.

    Returns:
        Dict with keys: formation_energy_eV_atom, formula, backend, note.
    """
    _TABULATED = {
        "TiO2": -9.87, "Al2O3": -16.46, "SiO2": -9.11,
        "Fe2O3": -8.24, "MgO": -6.23, "ZnO": -3.63,
        "GaAs": -0.74, "GaN": -1.10, "InP": -0.63,
    }
    if formula in _TABULATED:
        Ef = _TABULATED[formula]
        note = ""
        backend = "stub"
    else:
        Ef = -2.5  # rough proxy
        note = f"'{formula}' not in table; using heuristic -2.5 eV/atom. Install pymatgen+DFT for real calc."
        backend = "stub"

    return {
        "backend": backend,
        "note": note,
        "formation_energy_eV_atom": Ef,
        "formula": formula,
    }


def fermi_energy_estimate(
    n_electrons: float = 1e28,
    volume_m3: float = 1e-27,
) -> dict[str, Any]:
    """Free-electron Fermi energy estimate.

    E_F = (hbar^2 / 2m) * (3 pi^2 n)^(2/3)

    Args:
        n_electrons: Total number of valence electrons.
        volume_m3: Volume of the simulation cell (m^3).

    Returns:
        Dict with keys: E_F_eV, E_F_J, electron_density_m3, backend, note.
    """
    _HBAR = 1.054571817e-34
    _ME = 9.1093837015e-31
    _EV = 1.602176634e-19
    n_density = n_electrons / volume_m3
    E_F = (_HBAR ** 2 / (2 * _ME)) * (3 * math.pi ** 2 * n_density) ** (2.0 / 3.0)
    return {
        "backend": "analytic", "note": "",
        "E_F_eV": E_F / _EV, "E_F_J": E_F,
        "electron_density_m3": n_density,
    }
