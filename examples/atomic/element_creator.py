"""SIMLAB — Element Creator Demo.

Run this script to simulate building elements from protons and neutrons.
All matplotlib plots open in your system image viewer automatically.
Every run is recorded to the knowledge database and a PDF proof is generated.

Usage:
  python examples/atomic/element_creator.py                    # Carbon-12 (default)
  python examples/atomic/element_creator.py --Z 26             # Iron
  python examples/atomic/element_creator.py --Z 6 --N 6        # Carbon-12 explicit
  python examples/atomic/element_creator.py --Z 92             # Uranium
  python examples/atomic/element_creator.py --fusion 1 1       # H + H → He
  python examples/atomic/element_creator.py --chart            # Segrè chart + B/A curve
  python examples/atomic/element_creator.py --decay            # Ra-226 decay chain
  python examples/atomic/element_creator.py --orbitals --Z 1   # H orbital density maps
  python examples/atomic/element_creator.py --history          # Show past experiments
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simlab.engines.nuclear.element_builder import ElementBuilder
from simlab.engines.nuclear.nuclear_engine import NuclearEngine
from simlab.engines.nuclear.decay import DecayEngine
from simlab.engines.atomic.electron_config import ElectronConfigEngine
from simlab.engines.atomic.hydrogen_orbitals import HydrogenOrbitalEngine
from simlab.display import show, show_all
from simlab.db.experiment_db import ExperimentDB
from simlab.proof.proof_engine import ProofEngine


_db    = ExperimentDB()
_proof = ProofEngine()


def _sep(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * pad)
    else:
        print("─" * width)


def _record_and_prove(
    exp_id: str,
    domain: str,
    exp_type: str,
    parameters: dict,
    result: dict,
    plots_b64: dict[str, str],
    status: str = "success",
    notes: str = "",
) -> None:
    """Save to DB and generate LaTeX/PDF proof."""
    proof_info = _proof.save_proof(
        exp_id=exp_id,
        domain=domain,
        exp_type=exp_type,
        parameters=parameters,
        results=result,
        plots_b64=plots_b64,
        status=status,
        extra_notes=notes,
    )
    _db.save_experiment(
        exp_id=exp_id,
        domain=domain,
        exp_type=exp_type,
        parameters=parameters,
        results=result,
        status=status,
        proof_dir=proof_info.get("proof_dir"),
        pdf_path=proof_info.get("pdf_path"),
    )
    for name, path in proof_info.get("plot_paths", {}).items():
        _db.save_plot(exp_id, name, path)
    if notes:
        _db.add_note(notes, category="insight", experiment_id=exp_id)

    _sep("PROOF")
    print(f"  Proof dir  : {proof_info.get('proof_dir')}")
    print(f"  PDF        : {proof_info.get('pdf_path') or '(compile failed)'}")
    print(f"  Plots saved: {len(proof_info.get('plot_paths', {}))}")
    if proof_info.get("compiled_ok"):
        print("  LaTeX → PDF compiled successfully")


def demo_element(Z: int, N: int | None) -> None:
    t0 = time.perf_counter()
    builder = ElementBuilder()
    result = builder.create_element(Z, N, include_plots=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    nuc = result["nuclear"]
    ec  = result["electron_config"]
    ident = result["identity"]
    sym = ident["symbol"]
    A   = ident["A"]
    N_actual = ident["N"]
    exp_id = f"element_{sym.lower()}_Z{Z}_N{N_actual}"

    _sep("ELEMENT CREATED")
    print(result["summary"])

    _sep("NUCLEAR")
    print(f"  Isotope        : {nuc['isotope_notation']}")
    print(f"  Binding energy : {nuc['binding_energy_MeV']:.3f} MeV  "
          f"({nuc['binding_per_nucleon_MeV']:.4f} MeV/nucleon)")
    print(f"  Stability      : {nuc['stability_prediction']}")
    print(f"  Magic Z? {nuc['is_magic_Z']}   Magic N? {nuc['is_magic_N']}")
    if nuc["is_doubly_magic"]:
        print("  *** DOUBLY MAGIC NUCLEUS ***")
    print(f"  Most stable A  : {nuc['most_stable_isotope']['A']}")

    _sep("ELECTRON CONFIG")
    print(f"  Config (noble) : {ec['config_noble']}")
    print(f"  Valence e⁻     : {ec['valence_electrons']}")
    if ec["is_exception"]:
        print("  NOTE: anomalous filling")

    _sep("ATOMIC PROPERTIES")
    for key, val in result["atomic"].items():
        if val is not None:
            print(f"  {key:30s}: {val}")

    # Open plots
    plots = result["plots"]
    b64_list = [(name, b64) for name, b64 in plots.items()
                if not name.endswith("_error") and isinstance(b64, str)]
    if b64_list:
        _sep("PLOTS")
        print(f"  Opening {len(b64_list)} plot(s)…")
        paths = show_all([b for _, b in b64_list], titles=[n for n, _ in b64_list])
        for (name, _), path in zip(b64_list, paths):
            print(f"  {name:35s} → {path}")

    # Record & prove
    params = {"Z": Z, "N": N_actual, "A": A, "symbol": sym}
    result_flat = {
        "binding_energy_MeV": nuc["binding_energy_MeV"],
        "binding_per_nucleon_MeV": nuc["binding_per_nucleon_MeV"],
        "stability": nuc["stability_prediction"],
        "electron_config": ec["config_noble"],
        "valence_electrons": ec["valence_electrons"],
        "is_doubly_magic": nuc["is_doubly_magic"],
    }
    plots_b64 = {name: b64 for name, b64 in b64_list}
    notes = (f"Element {sym} (Z={Z}, N={N_actual}) created successfully. "
             f"Stability: {nuc['stability_prediction']}. "
             f"B/A = {nuc['binding_per_nucleon_MeV']:.4f} MeV/nucleon.")
    _record_and_prove(exp_id, "atomic", "create_element", params, result_flat,
                      plots_b64, notes=notes)
    _sep()


def demo_fusion(Z1: int, Z2: int) -> None:
    builder = ElementBuilder()
    result = builder.fusion_to_element(Z1, Z2)
    rx = result["reaction"]

    _sep("FUSION REACTION")
    print(f"  {rx['reactant_1']} + {rx['reactant_2']} → {rx['product']}")
    print(f"  Q-value : {rx['Q_value_MeV']:.4f} MeV")
    print(f"  {rx['interpretation']}")

    prod = result["product_element"]
    _sep("PRODUCT ELEMENT")
    print(prod.get("summary", "  (theoretical — Z > 118)"))

    exp_id = f"fusion_Z{Z1}_Z{Z2}"
    params = {"Z1": Z1, "Z2": Z2}
    result_flat = {
        "Q_value_MeV": rx["Q_value_MeV"],
        "energy_released": rx["energy_released"],
        "product": rx["product"],
    }
    _record_and_prove(exp_id, "nuclear", "nuclear_fusion", params, result_flat, {},
                      status="success",
                      notes=f"Q={rx['Q_value_MeV']:.4f} MeV. {'Exothermic.' if rx['energy_released'] else 'Endothermic.'}")
    _sep()


def demo_nuclear_chart() -> None:
    ne = NuclearEngine()
    exp_id = "nuclear_chart_Z40_N50"

    print("Generating Segrè nuclear chart (Z≤40, N≤50)…")
    b64_chart = ne.nuclear_chart(z_max=40, n_max=50)
    path_chart = show(b64_chart, title="Segrè Chart")
    print(f"  Chart: {path_chart}")

    print("Generating B/A binding energy curve…")
    b64_ba = ne.binding_energy_curve()
    path_ba = show(b64_ba, title="Binding Energy Curve")
    print(f"  B/A curve: {path_ba}")

    params = {"z_max": 40, "n_max": 50}
    result_flat = {"description": "Segre chart + B/A curve for Z≤40, N≤50"}
    plots = {"segre_chart": b64_chart, "binding_energy_curve": b64_ba}
    _record_and_prove(exp_id, "nuclear", "nuclear_chart", params, result_flat, plots,
                      notes="Full Segrè chart showing binding energy per nucleon.")
    _sep()


def demo_orbitals(Z: int) -> None:
    oe = HydrogenOrbitalEngine()
    exp_id = f"orbitals_Z{Z}"
    print(f"Plotting hydrogen-like orbitals for Z={Z}…")

    plots: dict[str, str] = {}
    plots["energy_levels"] = oe.plot_energy_levels(Z=Z, n_max=5)
    for n in range(1, min(4, Z + 1)):
        plots[f"radial_n{n}"] = oe.plot_radial_comparison(n=n, Z=Z)
    plots["orbital_2p_2D"] = oe.plot_orbital_2d(n=2, l=1, m=0, Z=Z)

    paths = show_all(list(plots.values()), titles=list(plots.keys()))
    for name, path in zip(plots.keys(), paths):
        print(f"  {name:30s} → {path}")

    params = {"Z": Z, "n_max": 5}
    result_flat = {"orbitals_plotted": list(plots.keys())}
    _record_and_prove(exp_id, "atomic", "hydrogen_orbitals", params, result_flat, plots,
                      notes=f"Hydrogen-like orbitals computed for Z={Z}.")
    _sep()


def demo_decay() -> None:
    de = DecayEngine()
    exp_id = "decay_chain_Ra226"
    chain = [
        {"symbol": "Ra-226", "half_life_s": 5.05e10},
        {"symbol": "Rn-222", "half_life_s": 3.30e5},
        {"symbol": "Po-218", "half_life_s": 186.0},
        {"symbol": "Pb-214", "half_life_s": None},
    ]
    print("Simulating Ra-226 → Rn-222 → Po-218 → Pb-214 decay chain…")
    b64 = de.plot_decay_chain(chain, N0_first=1e6, t_span=1e6)
    path = show(b64, title="Decay Chain")
    print(f"  Decay chain plot: {path}")

    alpha_ra = de.alpha_decay_product(88, 138)
    alpha_rn = de.alpha_decay_product(86, 136)
    print("\n  Ra-226 α-decay:", alpha_ra)
    print("  Rn-222 α-decay:", alpha_rn)

    params = {"chain": [c["symbol"] for c in chain], "N0": 1e6}
    result_flat = {"Ra226_alpha": str(alpha_ra), "Rn222_alpha": str(alpha_rn)}
    _record_and_prove(exp_id, "nuclear", "decay_chain", params, result_flat,
                      {"decay_chain": b64},
                      notes="Ra-226 three-step alpha decay chain via Bateman equations.")
    _sep()


def show_history() -> None:
    _sep("EXPERIMENT HISTORY")
    stats = _db.summary_stats()
    print(f"  Total experiments : {stats['total_experiments']}")
    print(f"  By status  : {stats['by_status']}")
    print(f"  By domain  : {stats['by_domain']}")
    print(f"  Notes      : {stats['total_notes']}")
    print(f"  Knowledge  : {stats['total_knowledge_entries']}")

    _sep("RECENT RUNS")
    for exp in _db.recent(10):
        status_mark = "✓" if exp["status"] == "success" else "✗"
        print(f"  [{status_mark}] {exp['id']:40s} | {exp['domain']:10s} | {exp['created_at'][:19]}")
        if exp.get("pdf_path"):
            print(f"       PDF: {exp['pdf_path']}")

    _sep("WHAT WORKED")
    worked = _db.what_worked()
    print(f"  {len(worked)} successful experiment(s) on record.")

    _sep("WHAT FAILED")
    failed = _db.what_failed()
    print(f"  {len(failed)} failed experiment(s) on record.")
    for f in failed:
        print(f"  ✗ {f['id']} — {f.get('errors', [])}")
    _sep()


def main() -> None:
    parser = argparse.ArgumentParser(description="SIMLAB Element Creator")
    parser.add_argument("--Z", type=int, default=6, help="Atomic number (protons)")
    parser.add_argument("--N", type=int, default=None, help="Neutron count")
    parser.add_argument("--fusion", type=int, nargs=2, metavar=("Z1", "Z2"))
    parser.add_argument("--chart", action="store_true")
    parser.add_argument("--orbitals", action="store_true")
    parser.add_argument("--decay", action="store_true")
    parser.add_argument("--history", action="store_true", help="Show experiment history")
    args = parser.parse_args()

    if args.history:
        show_history()
    elif args.chart:
        demo_nuclear_chart()
    elif args.fusion:
        demo_fusion(args.fusion[0], args.fusion[1])
    elif args.decay:
        demo_decay()
    elif args.orbitals:
        demo_orbitals(args.Z)
    else:
        demo_element(args.Z, args.N)


if __name__ == "__main__":
    main()
