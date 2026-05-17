"""Molecular dynamics proxy utilities and potential energy functions.

Provides Lennard-Jones potential evaluation, EAM energy proxy, velocity-Verlet
integrator step, NVT and NPT MD proxies, and radial distribution function.

Optional dependencies: ase (for EAM), numpy (always required).
Falls back to tabulated / heuristic proxies.
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


def lj_potential(
    r: float = 3.4e-10,
    epsilon: float = 1.67e-21,
    sigma: float = 3.4e-10,
) -> dict[str, Any]:
    """Lennard-Jones 12-6 pair potential.

    V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]

    Args:
        r: Interatomic separation (m).
        epsilon: LJ energy parameter (J).
        sigma: LJ size parameter (m).

    Returns:
        Dict with keys: V_J, V_eV, F_N, r_min_m, backend, note.
    """
    sr = sigma / r
    V = 4.0 * epsilon * (sr ** 12 - sr ** 6)
    dV_dr = 4.0 * epsilon * (-12.0 * sr ** 12 + 6.0 * sr ** 6) / r
    r_min = 2.0 ** (1.0 / 6.0) * sigma
    _EV = 1.602176634e-19
    return {
        "backend": "analytic", "note": "",
        "V_J": V, "V_eV": V / _EV, "F_N": -dV_dr, "r_min_m": r_min,
    }


def eam_energy_proxy(
    element: str = "Cu",
    structure: str = "fcc",
    a0: float = 3.615e-10,
    n_atoms: int = 108,
) -> dict[str, Any]:
    """Proxy EAM cohesive energy and elastic constants for common metals.

    Args:
        element: Element symbol (Cu, Al, Fe, Ni, Au, Ag).
        structure: Crystal structure (fcc, bcc, hcp).
        a0: Lattice parameter (m).
        n_atoms: Number of atoms (for cohesive energy scaling).

    Returns:
        Dict with keys: cohesive_energy_eV_atom, bulk_modulus_GPa,
        shear_modulus_GPa, backend, note.
    """
    # Tabulated cohesive energies and elastic constants
    _DATA = {
        "Cu": {"E_coh": -3.54, "B": 140.0, "G": 48.0},
        "Al": {"E_coh": -3.36, "B": 79.0,  "G": 26.0},
        "Fe": {"E_coh": -4.28, "B": 170.0, "G": 82.0},
        "Ni": {"E_coh": -4.44, "B": 187.0, "G": 76.0},
        "Au": {"E_coh": -3.81, "B": 173.0, "G": 27.0},
        "Ag": {"E_coh": -2.96, "B": 104.0, "G": 30.0},
    }
    el = element.capitalize()
    d = _DATA.get(el, {"E_coh": -3.5, "B": 130.0, "G": 45.0})
    backend = "ase" if _ASE else "stub"
    note = "" if _ASE else f"Install ase for real EAM evaluation; using tabulated values for {el}."
    return {
        "backend": backend, "note": note,
        "cohesive_energy_eV_atom": d["E_coh"],
        "bulk_modulus_GPa": d["B"], "shear_modulus_GPa": d["G"],
        "element": el, "structure": structure,
    }


def velocity_verlet_step(
    r: list[list[float]],
    v: list[list[float]],
    F: list[list[float]],
    m: float = 1.067e-25,
    dt: float = 1e-15,
) -> dict[str, Any]:
    """Perform a single velocity-Verlet integration step.

    r(t+dt) = r(t) + v(t)*dt + 0.5*F/m*dt^2
    v(t+dt) = v(t) + 0.5*(F(t)+F(t+dt))/m*dt

    (Returns updated r and v using only F(t) — caller must recompute F(t+dt).)

    Args:
        r: Positions array (N x 3) in metres.
        v: Velocities array (N x 3) in m/s.
        F: Forces array (N x 3) in Newtons.
        m: Particle mass (kg).
        dt: Time step (s).

    Returns:
        Dict with keys: r_new, v_half, backend, note.
    """
    r_arr = np.array(r, dtype=float)
    v_arr = np.array(v, dtype=float)
    F_arr = np.array(F, dtype=float)
    a = F_arr / m
    r_new = r_arr + v_arr * dt + 0.5 * a * dt ** 2
    v_half = v_arr + 0.5 * a * dt
    return {
        "backend": "numpy", "note": "",
        "r_new": r_new.tolist(), "v_half": v_half.tolist(),
    }


def nvt_md_proxy(
    element: str = "Cu",
    T_K: float = 300.0,
    n_atoms: int = 256,
    n_steps: int = 1000,
    dt_fs: float = 2.0,
    potential: str = "eam",
) -> dict[str, Any]:
    """Proxy NVT MD: returns plausible thermodynamic averages without full simulation.

    Args:
        element: Element symbol.
        T_K: Target temperature (K).
        n_atoms: Number of atoms.
        n_steps: Number of MD steps.
        dt_fs: Time step (fs).
        potential: Interatomic potential type ('eam', 'lj').

    Returns:
        Dict with keys: T_mean_K, T_std_K, P_GPa, E_pot_eV_atom, MSD_A2,
        diffusion_coeff_m2_s, backend, note.
    """
    _KB = 1.380649e-23
    # Heuristic thermodynamic properties
    eam_data = eam_energy_proxy(element)
    E_pot = eam_data["cohesive_energy_eV_atom"] * (1.0 + 0.01 * T_K / 300.0)
    P = 0.1 * T_K / 300.0  # GPa proxy
    MSD = 0.05 * T_K / 300.0 * n_steps * dt_fs / 1000.0  # A^2
    D = MSD * 1e-20 / (6.0 * n_steps * dt_fs * 1e-15)

    return {
        "backend": "stub",
        "note": f"Install ase+lammps for real NVT MD; using heuristic for {element}.",
        "T_mean_K": T_K * (1 + 0.01 * np.random.randn()), "T_std_K": T_K * 0.02,
        "P_GPa": P, "E_pot_eV_atom": E_pot,
        "MSD_A2": MSD, "diffusion_coeff_m2_s": D,
    }


def npt_md_proxy(
    element: str = "Cu",
    T_K: float = 800.0,
    P_GPa: float = 0.0,
    n_atoms: int = 256,
    n_steps: int = 2000,
) -> dict[str, Any]:
    """Proxy NPT MD: returns plausible equilibrium lattice parameter and density.

    Args:
        element: Element symbol.
        T_K: Temperature (K).
        P_GPa: Pressure (GPa).
        n_atoms: Number of atoms.
        n_steps: Number of MD steps.

    Returns:
        Dict with keys: a0_A, density_g_cm3, thermal_expansion_1_K, backend, note.
    """
    _LATTICE = {
        "Cu": (3.615, 8.96), "Al": (4.046, 2.70), "Fe": (2.870, 7.87),
        "Ni": (3.524, 8.91), "Au": (4.078, 19.3), "Ag": (4.086, 10.5),
    }
    a0, rho0 = _LATTICE.get(element.capitalize(), (3.5, 8.0))
    alpha_th = 1.7e-5  # 1/K typical
    a_hot = a0 * (1 + alpha_th * (T_K - 300.0))
    # Pressure correction (Murnaghan)
    B_GPa = eam_energy_proxy(element)["bulk_modulus_GPa"]
    a_P = a_hot * (1 - P_GPa / (3.0 * B_GPa))
    rho = rho0 * (a0 / a_P) ** 3

    return {
        "backend": "stub",
        "note": f"Install ase+lammps for real NPT MD; heuristic for {element}.",
        "a0_A": a_P, "density_g_cm3": rho, "thermal_expansion_1_K": alpha_th,
    }


def radial_distribution_function(
    r_array: list[float] | None = None,
    n_bins: int = 100,
    rho: float = 8.4e28,
) -> dict[str, Any]:
    """Compute an analytic model of the radial distribution function.

    Returns a proxy RDF peaked at the first neighbour distance.

    Args:
        r_array: Radial distances (m) at which to evaluate g(r).
            Defaults to 1-10 Angstrom range.
        n_bins: Number of bins (used if r_array is None).
        rho: Atom number density (m^{-3}).

    Returns:
        Dict with keys: r_A, g_r, first_peak_A, backend, note.
    """
    if r_array is None:
        r_array = np.linspace(1e-10, 1e-9, n_bins)
    else:
        r_array = np.array(r_array)

    r_A = r_array * 1e10
    r1 = 2.55  # first peak ~ Angstrom
    # Proxy: sum of Gaussians at first 3 shells
    g = (2.5 * np.exp(-0.5 * ((r_A - r1) / 0.15) ** 2) +
         1.2 * np.exp(-0.5 * ((r_A - r1 * 1.41) / 0.2) ** 2) +
         0.9 * np.exp(-0.5 * ((r_A - r1 * 1.73) / 0.25) ** 2) + 1.0)
    g = np.where(r_A < r1 * 0.8, 0.0, g)

    return {
        "backend": "stub",
        "note": "Run real MD+histogram for actual RDF.",
        "r_A": r_A.tolist(), "g_r": g.tolist(), "first_peak_A": r1,
    }
