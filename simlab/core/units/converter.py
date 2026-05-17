"""Unit conversion utilities using a factor-based approach.

All conversions go through SI base units as the canonical intermediate.
Temperature conversions use offset formulas rather than multiplicative factors.
"""

from __future__ import annotations


# Conversion factors: unit -> SI base value
# e.g. 1 km = 1000 m  =>  "km": 1000.0
_LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
    "km": 1e3,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "ft": 0.3048,
    "in": 0.0254,
    "yd": 0.9144,
    "mi": 1609.344,
    "nmi": 1852.0,
}

_MASS_TO_KG: dict[str, float] = {
    "kg": 1.0,
    "g": 1e-3,
    "mg": 1e-6,
    "ug": 1e-9,
    "t": 1e3,
    "lb": 0.45359237,
    "oz": 0.028349523125,
    "slug": 14.593903,
}

_TIME_TO_S: dict[str, float] = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "min": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
}

_ENERGY_TO_J: dict[str, float] = {
    "J": 1.0,
    "kJ": 1e3,
    "MJ": 1e6,
    "cal": 4.184,
    "kcal": 4184.0,
    "eV": 1.602176634e-19,
    "keV": 1.602176634e-16,
    "MeV": 1.602176634e-13,
    "Wh": 3600.0,
    "kWh": 3.6e6,
    "BTU": 1055.06,
    "erg": 1e-7,
}

_FORCE_TO_N: dict[str, float] = {
    "N": 1.0,
    "kN": 1e3,
    "MN": 1e6,
    "lbf": 4.4482216152605,
    "dyn": 1e-5,
    "kgf": 9.80665,
}

_PRESSURE_TO_PA: dict[str, float] = {
    "Pa": 1.0,
    "kPa": 1e3,
    "MPa": 1e6,
    "GPa": 1e9,
    "bar": 1e5,
    "mbar": 100.0,
    "atm": 101325.0,
    "psi": 6894.757293168,
    "mmHg": 133.322387415,
    "torr": 133.322387415,
}

_ANGLE_TO_RAD: dict[str, float] = {
    "rad": 1.0,
    "deg": 3.141592653589793 / 180.0,
    "grad": 3.141592653589793 / 200.0,
    "rev": 2 * 3.141592653589793,
}

# Registry of supported dimension families
_FAMILIES: dict[str, dict[str, float]] = {
    "length": _LENGTH_TO_M,
    "mass": _MASS_TO_KG,
    "time": _TIME_TO_S,
    "energy": _ENERGY_TO_J,
    "force": _FORCE_TO_N,
    "pressure": _PRESSURE_TO_PA,
    "angle": _ANGLE_TO_RAD,
}

# Reverse lookup: unit symbol -> family name
_UNIT_TO_FAMILY: dict[str, str] = {}
for _family, _units in _FAMILIES.items():
    for _sym in _units:
        _UNIT_TO_FAMILY[_sym] = _family


def _temperature_to_K(value: float, unit: str) -> float:
    """Convert a temperature value to Kelvin."""
    if unit == "K":
        return value
    elif unit == "C":
        return value + 273.15
    elif unit == "F":
        return (value - 32) * 5 / 9 + 273.15
    elif unit == "R":  # Rankine
        return value * 5 / 9
    raise ValueError(f"Unknown temperature unit: {unit!r}")


def _temperature_from_K(value_K: float, unit: str) -> float:
    """Convert Kelvin to the target temperature unit."""
    if unit == "K":
        return value_K
    elif unit == "C":
        return value_K - 273.15
    elif unit == "F":
        return (value_K - 273.15) * 9 / 5 + 32
    elif unit == "R":  # Rankine
        return value_K * 9 / 5
    raise ValueError(f"Unknown temperature unit: {unit!r}")


_TEMP_UNITS = {"K", "C", "F", "R"}


class UnitConverter:
    """Simple unit converter supporting major physical quantities."""

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert *value* from *from_unit* to *to_unit*.

        Supports temperature (K, C, F, R) and any quantity in the families:
        length, mass, time, energy, force, pressure, angle.

        Raises
        ------
        ValueError
            If units are unknown or belong to different families.
        """
        # Temperature special case
        if from_unit in _TEMP_UNITS or to_unit in _TEMP_UNITS:
            if from_unit not in _TEMP_UNITS or to_unit not in _TEMP_UNITS:
                raise ValueError(
                    f"Cannot mix temperature unit {from_unit!r} with {to_unit!r}"
                )
            k = _temperature_to_K(value, from_unit)
            return _temperature_from_K(k, to_unit)

        # Factor-based conversion
        from_family = _UNIT_TO_FAMILY.get(from_unit)
        to_family = _UNIT_TO_FAMILY.get(to_unit)

        if from_family is None:
            raise ValueError(f"Unknown unit: {from_unit!r}")
        if to_family is None:
            raise ValueError(f"Unknown unit: {to_unit!r}")
        if from_family != to_family:
            raise ValueError(
                f"Unit mismatch: {from_unit!r} ({from_family}) vs {to_unit!r} ({to_family})"
            )

        table = _FAMILIES[from_family]
        value_si = value * table[from_unit]
        return value_si / table[to_unit]

    def supported_units(self, family: str | None = None) -> dict[str, list[str]]:
        """Return supported units, optionally filtered by family."""
        if family is not None:
            if family not in _FAMILIES:
                raise ValueError(f"Unknown family: {family!r}")
            return {family: list(_FAMILIES[family].keys())}
        result = {fam: list(units.keys()) for fam, units in _FAMILIES.items()}
        result["temperature"] = list(_TEMP_UNITS)
        return result

    def detect_family(self, unit: str) -> str:
        """Return the dimension family for a given unit symbol."""
        if unit in _TEMP_UNITS:
            return "temperature"
        family = _UNIT_TO_FAMILY.get(unit)
        if family is None:
            raise ValueError(f"Unknown unit: {unit!r}")
        return family


# Module-level convenience function
_converter = UnitConverter()


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert *value* from *from_unit* to *to_unit* (module-level shortcut)."""
    return _converter.convert(value, from_unit, to_unit)
