"""pycalphad-based thermodynamic backend for alloy phase equilibrium.

Wraps pycalphad's CALPHAD engine behind an availability check so the rest
of the codebase degrades gracefully when pycalphad or a TDB database is not
installed.  All public methods return a plain dict that is JSON-serialisable.

High-fidelity results require a licensed TDB database (e.g. TCNI9, TCCOB5
from Thermo-Calc, or the open NIST COST-507 database for Al alloys).
Without a database, the proxy methods return structurally identical dicts
flagged with ``"backend": "proxy"`` and explicit disclaimers.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np


def _pycalphad_available() -> bool:
    return importlib.util.find_spec("pycalphad") is not None


# TCP phase names as recognised in most commercial TDB databases
_TCP_PHASES = {"SIGMA", "MU_PHASE", "LAVES_C14", "LAVES_C15", "LAVES_C36",
               "CHI", "P_PHASE", "R_PHASE", "ETA", "DELTA"}

# Elements supported by this backend (used for input validation)
_SUPPORTED_ELEMENTS = {
    "Fe", "Ni", "Cr", "Co", "Al", "Ti", "Mo", "W", "Nb", "Ta",
    "Re", "Hf", "Ru", "V", "Mn", "Si", "C", "B", "Zr",
}


def _demo_tdb_path() -> str | None:
    """Return path to pycalphad's bundled Al-Cr-Ni demo database, if present."""
    try:
        import pycalphad, os
        p = os.path.join(os.path.dirname(pycalphad.__file__),
                         "tests", "databases", "alcrni.tdb")
        return p if os.path.exists(p) else None
    except Exception:
        return None

# Elements supported by the bundled alcrni.tdb demo database
_ALCRNI_ELEMENTS = {"Al", "Cr", "Ni"}
# Phases in the bundled demo database
_ALCRNI_PHASES   = ["LIQUID", "FCC_A1", "BCC_A2", "L12_FCC"]


class CALPHADBackend:
    """Wraps pycalphad for multicomponent phase equilibrium calculations.

    Instantiation is cheap — the TDB database is loaded lazily on first use.

    Parameters
    ----------
    database_path : str | None
        Path to a ``.tdb`` file.  When None the backend auto-detects pycalphad's
        bundled Al-Cr-Ni demo database; falls back to proxy mode if unavailable.
    """

    def __init__(self, database_path: str | None = None):
        if database_path is None and _pycalphad_available():
            database_path = _demo_tdb_path()
        self._database_path = database_path
        self._db: Any = None
        self._available = _pycalphad_available()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def available(self) -> bool:
        """True when pycalphad is installed *and* a database path is set."""
        return self._available and self._database_path is not None

    def _load_db(self) -> Any:
        if self._db is None:
            from pycalphad import Database
            self._db = Database(self._database_path)
        return self._db

    def _is_demo_db(self) -> bool:
        """True when using the bundled demo Al-Cr-Ni database."""
        if self._database_path is None:
            return False
        return "alcrni.tdb" in self._database_path

    def _demo_elements(self, composition: dict[str, float]) -> bool:
        """True when all composition elements are in the demo database."""
        return set(composition.keys()).issubset(_ALCRNI_ELEMENTS)

    # ------------------------------------------------------------------
    # 1. Multi-component phase equilibrium
    # ------------------------------------------------------------------

    def calculate_phase_equilibrium(
        self,
        composition: dict[str, float],
        temperature_K: float,
        pressure_Pa: float = 101_325.0,
        output_phases: list[str] | None = None,
    ) -> dict[str, Any]:
        """Calculate equilibrium phase fractions at a given T and composition.

        Returns phase names, mole fractions, and key site occupancies where
        available.  Falls back to a proxy when pycalphad is unavailable.

        Parameters
        ----------
        composition : dict[str, float]
            Element → mole fraction (will be normalised to sum = 1).
        temperature_K : float
            Equilibrium temperature.
        output_phases : list[str] | None
            Restrict calculation to these phase names.  None = all phases in DB.
        """
        comp = _normalise(composition)
        T = float(temperature_K)

        if not self.available():
            return self._proxy_phase_equilibrium(comp, T)

        # If using the demo database, fall back to proxy for non-Al/Cr/Ni systems
        if self._is_demo_db() and not self._demo_elements(comp):
            result = self._proxy_phase_equilibrium(comp, T)
            result["note"] = (
                "Demo database (Al-Cr-Ni only); proxy used for this composition. "
                "Supply a full TDB for multi-element alloys."
            )
            return result

        try:
            from pycalphad import equilibrium, variables as v
            import numpy as _np

            db = self._load_db()
            elements = list(comp.keys()) + ["VA"]
            # For demo DB, restrict to known phases
            if self._is_demo_db():
                phases = output_phases or _ALCRNI_PHASES
            else:
                phases = output_phases or list(db.phases.keys())

            # pycalphad expects uppercase element names
            comp_up = {k.upper(): val for k, val in comp.items()}
            elements_up = [e.upper() for e in comp_up] + ["VA"]

            # Specify only N-1 composition variables (Gibbs phase rule)
            dep_el = max(comp_up, key=comp_up.__getitem__)
            conditions = {v.T: T, v.P: pressure_Pa}
            for el, x in comp_up.items():
                if el != dep_el and el != "VA":
                    conditions[v.X(el)] = x

            eq = equilibrium(db, elements_up, phases, conditions)

            # Extract phase fractions: NP array is parallel to Phase string array
            phase_fracs: dict[str, float] = {}
            if hasattr(eq, "NP") and hasattr(eq, "Phase"):
                np_arr  = eq.NP.values.squeeze()
                ph_arr  = eq.Phase.values.squeeze()
                for i in range(min(len(np_arr), len(ph_arr))):
                    ph  = str(ph_arr[i]).strip()
                    val = float(np_arr[i]) if not _np.isnan(float(np_arr[i])) else 0.0
                    if ph and val > 1e-6:
                        phase_fracs[ph] = round(val, 6)

            tcp_present = [p for p in phase_fracs if p.upper() in _TCP_PHASES]
            return {
                "backend": "pycalphad",
                "database": self._database_path,
                "composition": comp,
                "temperature_K": T,
                "pressure_Pa": pressure_Pa,
                "phase_fractions": phase_fracs,
                "tcp_phases_detected": tcp_present,
                "tcp_risk": len(tcp_present) > 0,
                "note": "demo Al-Cr-Ni database" if self._is_demo_db() else None,
            }

        except Exception as exc:
            return {**self._proxy_phase_equilibrium(comp, T),
                    "pycalphad_error": str(exc)}

    # ------------------------------------------------------------------
    # 2. Phase fraction vs temperature scan
    # ------------------------------------------------------------------

    def phase_fraction_vs_temperature(
        self,
        composition: dict[str, float],
        T_low_K: float,
        T_high_K: float,
        n_points: int = 30,
        phases_of_interest: list[str] | None = None,
    ) -> dict[str, Any]:
        """Scan phase fractions over a temperature range.

        Useful for locating solvus temperatures and tracking the γ/γ'
        balance as a function of T.
        """
        comp = _normalise(composition)
        T_arr = np.linspace(float(T_low_K), float(T_high_K), int(n_points))

        if not self.available():
            return self._proxy_phase_fraction_scan(comp, T_arr)

        results: list[dict] = []
        for T in T_arr:
            eq = self.calculate_phase_equilibrium(comp, float(T))
            results.append({"T_K": float(T), "phases": eq["phase_fractions"]})

        # Collect all phase names seen
        all_phases: set[str] = set()
        for r in results:
            all_phases.update(r["phases"].keys())

        phase_curves: dict[str, list[float]] = {ph: [] for ph in sorted(all_phases)}
        for r in results:
            for ph in sorted(all_phases):
                phase_curves[ph].append(r["phases"].get(ph, 0.0))

        return {
            "backend": "pycalphad",
            "database": self._database_path,
            "composition": comp,
            "temperatures_K": T_arr.tolist(),
            "phase_curves": phase_curves,
            "phases_observed": sorted(all_phases),
        }

    # ------------------------------------------------------------------
    # 3. TCP phase detection
    # ------------------------------------------------------------------

    def detect_tcp_phases(
        self,
        composition: dict[str, float],
        temperatures_K: list[float] | None = None,
        service_T_K: float = 1173.15,
    ) -> dict[str, Any]:
        """Check for topologically close-packed (TCP) phase stability.

        Calculates equilibrium at ``service_T_K`` and at an optional list of
        additional temperatures, returning the TCP phases found, their mole
        fractions, and a risk classification.

        TCP phases checked: σ, μ, Laves (C14/C15/C36), χ, P, R, η, δ.
        """
        comp = _normalise(composition)
        temps = [float(service_T_K)]
        if temperatures_K:
            temps = sorted(set(temps + [float(t) for t in temperatures_K]))

        if not self.available():
            return self._proxy_tcp_detection(comp, temps)

        tcp_summary: dict[float, dict[str, float]] = {}
        for T in temps:
            eq = self.calculate_phase_equilibrium(comp, T)
            tcp_here = {ph: frac for ph, frac in eq["phase_fractions"].items()
                        if ph.upper() in _TCP_PHASES}
            if tcp_here:
                tcp_summary[T] = tcp_here

        max_tcp_frac = max(
            (sum(v.values()) for v in tcp_summary.values()),
            default=0.0,
        )

        if max_tcp_frac > 0.05:
            risk = "HIGH — >5% TCP phase fraction at service temperature"
        elif max_tcp_frac > 0.01:
            risk = "MODERATE — 1–5% TCP phase fraction"
        elif max_tcp_frac > 0.0:
            risk = "LOW — trace TCP phases (<1%)"
        else:
            risk = "NONE detected at evaluated temperatures"

        return {
            "backend": "pycalphad",
            "database": self._database_path,
            "composition": comp,
            "temperatures_evaluated_K": temps,
            "tcp_phases_by_temperature": {str(T): v for T, v in tcp_summary.items()},
            "max_tcp_mole_fraction": float(max_tcp_frac),
            "tcp_risk_classification": risk,
            "tcp_phases_checked": sorted(_TCP_PHASES),
            "recommendation": (
                "Reduce Mo and/or W fractions if TCP risk is HIGH or MODERATE. "
                "σ-phase is typically destabilised by reducing Cr+Mo below ~30 at.%."
                if max_tcp_frac > 0.01 else
                "No TCP adjustment indicated at evaluated temperatures."
            ),
        }

    # ------------------------------------------------------------------
    # 4. Binary phase diagram
    # ------------------------------------------------------------------

    def calculate_binary_phase_diagram(
        self,
        element_a: str,
        element_b: str,
        T_low_K: float = 600.0,
        T_high_K: float = 1800.0,
        n_compositions: int = 40,
        n_temperatures: int = 40,
    ) -> dict[str, Any]:
        """Compute a binary A-B phase diagram grid.

        Returns a 2-D grid of dominant phases for plotting.  Suitable for
        quick binary sub-system screening before multi-component calculations.
        """
        if not self.available():
            return self._proxy_binary_diagram(element_a, element_b, T_low_K, T_high_K)

        try:
            from pycalphad import equilibrium, variables as v

            db = self._load_db()
            x_arr = np.linspace(0.01, 0.99, int(n_compositions))
            T_arr = np.linspace(float(T_low_K), float(T_high_K), int(n_temperatures))
            grid: list[list[str]] = []

            for T in T_arr:
                row = []
                for x_b in x_arr:
                    conditions = {v.T: T, v.P: 101_325.0, v.X(element_b): x_b}
                    eq = equilibrium(db, [element_a, element_b, "VA"],
                                     list(db.phases.keys()), conditions)
                    dominant = _dominant_phase(eq)
                    row.append(dominant)
                grid.append(row)

            return {
                "backend": "pycalphad",
                "database": self._database_path,
                "element_a": element_a,
                "element_b": element_b,
                "x_b_values": x_arr.tolist(),
                "temperatures_K": T_arr.tolist(),
                "dominant_phase_grid": grid,
            }

        except Exception as exc:
            return {**self._proxy_binary_diagram(element_a, element_b, T_low_K, T_high_K),
                    "pycalphad_error": str(exc)}

    # ------------------------------------------------------------------
    # 5. Oxide phase equilibria / MoO₃ volatility
    # ------------------------------------------------------------------

    def oxide_phase_equilibrium(
        self,
        composition: dict[str, float],
        temperature_K: float,
        oxygen_partial_pressure_atm: float = 0.21,
    ) -> dict[str, Any]:
        """Check which oxide phases are stable and whether MoO₃ is volatile.

        Uses the oxidation-relevant elements (Al, Cr, Mo, W, Fe, Ni, Co, Ti)
        and evaluates the oxide phase equilibrium.  Flags MoO₃ volatility
        when Mo is present and T > 973 K (~700°C).

        This is an indicative calculation; quantitative p(MoO₃) requires
        a dedicated oxide TDB (e.g. SSUB or TCOX).
        """
        comp = _normalise(composition)
        T = float(temperature_K)
        mo = comp.get("Mo", 0.0)

        moo3_volatile = mo > 0.0 and T > 973.0
        moo3_risk = "CRITICAL" if (mo > 0.05 and T > 1023.0) else (
            "HIGH" if (mo > 0.02 and T > 973.0) else (
            "LOW" if mo > 0.0 else "NONE"))

        if not self.available():
            return {
                "backend": "proxy",
                "composition": comp,
                "temperature_K": T,
                "oxygen_partial_pressure_atm": oxygen_partial_pressure_atm,
                "moo3_volatility_risk": moo3_risk,
                "moo3_volatile_above_700C": moo3_volatile,
                "mo_fraction": mo,
                "note": (
                    "pycalphad not installed. MoO3 volatility assessed from "
                    "Mo fraction and temperature only (no thermodynamic calculation)."
                ),
                "disclaimer": (
                    "Full oxide phase equilibrium requires pycalphad + SSUB/TCOX database."
                ),
            }

        try:
            # Build oxide system: add O as element, fix chemical potential
            from pycalphad import equilibrium, variables as v

            db = self._load_db()
            ox_elements = list(comp.keys()) + ["O", "VA"]
            log_pO2 = math.log(oxygen_partial_pressure_atm * 101_325.0)
            conditions = {v.T: T, v.P: 101_325.0, v.MU("O"): R_GAS * T * log_pO2}
            for el, x in comp.items():
                conditions[v.X(el)] = x

            eq = equilibrium(db, ox_elements, list(db.phases.keys()), conditions)
            frac_dict = {}
            for ph in db.phases:
                try:
                    nf = float(eq.Phase.sel(phase_name=ph).values.squeeze())
                    if nf > 1e-6:
                        frac_dict[ph] = round(nf, 6)
                except Exception:
                    pass

            return {
                "backend": "pycalphad",
                "database": self._database_path,
                "composition": comp,
                "temperature_K": T,
                "oxygen_partial_pressure_atm": oxygen_partial_pressure_atm,
                "oxide_phase_fractions": frac_dict,
                "moo3_volatility_risk": moo3_risk,
                "moo3_volatile_above_700C": moo3_volatile,
                "mo_fraction": mo,
            }

        except Exception as exc:
            return {
                "backend": "proxy",
                "composition": comp,
                "temperature_K": T,
                "moo3_volatility_risk": moo3_risk,
                "moo3_volatile_above_700C": moo3_volatile,
                "mo_fraction": mo,
                "pycalphad_error": str(exc),
                "disclaimer": "Oxide equilibrium calculation failed; MoO3 risk from fraction+T only.",
            }

    # ------------------------------------------------------------------
    # Proxy implementations (no pycalphad / no database)
    # ------------------------------------------------------------------

    def _proxy_phase_equilibrium(self, comp: dict, T: float) -> dict:
        ni = comp.get("Ni", 0.0)
        co = comp.get("Co", 0.0)
        cr = comp.get("Cr", 0.0)
        mo = comp.get("Mo", 0.0)
        w  = comp.get("W",  0.0)
        al = comp.get("Al", 0.0)
        ti = comp.get("Ti", 0.0)

        # Very rough heuristics — for labelling / warning only
        gamma_prime_frac = min(0.60, max(0.0, (al + ti - 0.05) * 2.5)) if (ni > 0.25 or co > 0.25) else 0.0
        tcp_frac_proxy = max(0.0, (mo + w) - 0.15) * 0.8
        tcp_phases_proxy = []
        if tcp_frac_proxy > 0.01:
            tcp_phases_proxy.append("SIGMA_proxy")
        if mo > 0.10:
            tcp_phases_proxy.append("MU_PHASE_proxy")

        phase_fracs = {"FCC_A1": max(0.0, 1.0 - gamma_prime_frac - tcp_frac_proxy)}
        if gamma_prime_frac > 0.01:
            phase_fracs["GAMMA_PRIME_L12_proxy"] = round(gamma_prime_frac, 4)
        if tcp_frac_proxy > 0.01:
            phase_fracs["TCP_proxy"] = round(tcp_frac_proxy, 4)

        return {
            "backend": "proxy",
            "composition": comp,
            "temperature_K": T,
            "phase_fractions": phase_fracs,
            "tcp_phases_detected": tcp_phases_proxy,
            "tcp_risk": len(tcp_phases_proxy) > 0,
            "note": (
                "Proxy result — install pycalphad and supply a TDB database for "
                "real thermodynamic phase equilibrium. Phase fractions are "
                "rough composition heuristics, not CALPHAD calculations."
            ),
            "disclaimer": (
                "Do not use proxy phase fractions for material selection decisions."
            ),
        }

    def _proxy_phase_fraction_scan(self, comp: dict, T_arr: np.ndarray) -> dict:
        rows = [self._proxy_phase_equilibrium(comp, float(T)) for T in T_arr]
        all_phases: set[str] = set()
        for r in rows:
            all_phases.update(r["phase_fractions"].keys())
        curves = {ph: [r["phase_fractions"].get(ph, 0.0) for r in rows]
                  for ph in sorted(all_phases)}
        return {
            "backend": "proxy",
            "composition": comp,
            "temperatures_K": T_arr.tolist(),
            "phase_curves": curves,
            "phases_observed": sorted(all_phases),
            "note": "Proxy scan — heuristic phase fractions only.",
        }

    def _proxy_tcp_detection(self, comp: dict, temps: list[float]) -> dict:
        tcp_by_T: dict[str, dict] = {}
        max_frac = 0.0
        for T in temps:
            eq = self._proxy_phase_equilibrium(comp, T)
            tcp_here = {ph: frac for ph, frac in eq["phase_fractions"].items()
                        if "TCP" in ph or "SIGMA" in ph or "MU_" in ph}
            if tcp_here:
                tcp_by_T[str(T)] = tcp_here
                max_frac = max(max_frac, sum(tcp_here.values()))

        mo = comp.get("Mo", 0.0)
        w  = comp.get("W",  0.0)
        risk = ("HIGH" if max_frac > 0.05 else
                "MODERATE" if max_frac > 0.01 else
                "LOW" if max_frac > 0.0 else "NONE")

        return {
            "backend": "proxy",
            "composition": comp,
            "temperatures_evaluated_K": temps,
            "tcp_phases_by_temperature": tcp_by_T,
            "max_tcp_mole_fraction": float(max_frac),
            "tcp_risk_classification": risk,
            "tcp_phases_checked": sorted(_TCP_PHASES),
            "mo_w_sum": round(mo + w, 4),
            "recommendation": (
                "Mo+W = {:.1f}%. High Mo+W compositions (>15%) are TCP-prone. "
                "Proxy result — verify with CALPHAD.".format((mo + w) * 100)
            ),
            "note": "Proxy TCP detection — install pycalphad for thermodynamic calculation.",
        }

    def _proxy_binary_diagram(self, el_a: str, el_b: str,
                               T_low: float, T_high: float) -> dict:
        return {
            "backend": "proxy",
            "element_a": el_a,
            "element_b": el_b,
            "T_low_K": T_low,
            "T_high_K": T_high,
            "dominant_phase_grid": [],
            "note": "pycalphad not installed. Binary diagram not available in proxy mode.",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

R_GAS = 8.31446261815324


def _normalise(comp: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in comp.values() if v > 0)
    if total <= 0:
        raise ValueError("Composition must have positive total.")
    return {k: float(v) / total for k, v in comp.items() if v > 0}


def _dominant_phase(eq: Any) -> str:
    try:
        nf = eq.Phase.values.squeeze()
        ph_names = eq.coords["phase_name"].values
        idx = int(np.argmax(nf))
        return str(ph_names[idx])
    except Exception:
        return "UNKNOWN"
