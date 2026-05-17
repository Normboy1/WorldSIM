"""Physical constants for SIMLAB.

Values sourced from CODATA 2018 / NIST.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """Fundamental physical constants (SI units)."""

    # Gravitational constant [m^3 kg^-1 s^-2]
    G: float = 6.67430e-11
    # Speed of light in vacuum [m/s]
    c: float = 2.99792458e8
    # Planck constant [J·s]
    h: float = 6.62607015e-34
    # Reduced Planck constant (hbar) [J·s]
    hbar: float = 1.054571817e-34
    # Boltzmann constant [J/K]
    k_B: float = 1.380649e-23
    # Avogadro constant [mol^-1]
    N_A: float = 6.02214076e23
    # Molar gas constant [J mol^-1 K^-1]
    R: float = 8.314462618
    # Elementary charge [C]
    e: float = 1.602176634e-19
    # Vacuum permittivity [F/m]
    epsilon_0: float = 8.8541878128e-12
    # Vacuum permeability [H/m]
    mu_0: float = 1.25663706212e-6
    # Electron mass [kg]
    m_e: float = 9.1093837015e-31
    # Proton mass [kg]
    m_p: float = 1.67262192369e-27
    # Stefan-Boltzmann constant [W m^-2 K^-4]
    sigma_SB: float = 5.670374419e-8
    # Fine-structure constant (dimensionless)
    alpha: float = 7.2973525693e-3
    # Standard gravity on Earth [m/s^2]
    g_earth: float = 9.80665
    # Surface gravity on Moon [m/s^2]
    g_moon: float = 1.62
    # Surface gravity on Mars [m/s^2]
    g_mars: float = 3.72076
    # Electron volt [J]
    eV: float = 1.602176634e-19
    # Atomic mass unit [kg]
    u: float = 1.66053906660e-27

    def as_dict(self) -> dict[str, float]:
        """Return all constants as a plain dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    def get(self, name: str, default: float | None = None) -> float | None:
        """Get a constant by name."""
        return getattr(self, name, default)


# Singleton instance for import convenience
CONSTANTS = PhysicalConstants()
