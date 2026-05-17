"""Example: Hybrid math + chemistry reaction kinetics analysis.

Simulates first-order decay dA/dt = -kA with k=0.25, A0=1.0,
cross-validating the ODE numerical solution against:
- Analytical first-order formula (chemistry engine)
- SymPy ODE solution (math engine)

Also demonstrates consecutive A -> B -> C reactions.
"""

import sys
import os
import base64
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simlab.core.engine.simlab_core import SimLabCore
from simlab.engines.chemistry.kinetics import KineticsEngine
from simlab.engines.math.ode_solver import ODESolver
from simlab.engines.math.symbolic import SymbolicEngine
from simlab.engines.visualization.plots import PlotEngine


def main():
    core = SimLabCore()
    kinetics = KineticsEngine()
    ode_solver = ODESolver()
    sym_engine = SymbolicEngine()
    plot_engine = PlotEngine()

    k = 0.25
    A0 = 1.0
    t_end = 20.0

    print("Reaction Kinetics: First-Order Decay")
    print(f"dA/dt = -kA,  k = {k},  A₀ = {A0}")
    print("=" * 55)

    # 1. Chemistry engine (analytical)
    chem_result = kinetics.simulate_first_order(k, A0, t_span=t_end, t_points=200)
    t_arr = chem_result["t"]
    A_chem = chem_result["A"]
    t50 = chem_result["half_life"]
    print(f"\n[1] Chemistry engine (analytical: A(t) = A₀·e^(-kt))")
    print(f"    Half-life t½ = {t50:.4f} s  (expected: {math.log(2)/k:.4f} s)")
    print(f"    A(t=0)  = {A_chem[0]:.6f}")
    print(f"    A(t=t½) ≈ {A_chem[len(A_chem)//2]:.6f}  (expected: {A0*math.exp(-k*t50):.6f})")
    print(f"    A(t=20) = {A_chem[-1]:.6f}  (expected: {A0*math.exp(-k*t_end):.6f})")

    # 2. ODE solver (numerical)
    ode_result = ode_solver.solve_ode(
        f_str=f"-{k}*y",
        y0=A0,
        t_span=[0, t_end],
        t_eval_points=200,
        method="RK45",
    )
    A_ode = ode_result["y"]
    print(f"\n[2] ODE solver (numerical: RK45)")
    print(f"    A(t=0)  = {A_ode[0]:.6f}")
    print(f"    A(t=20) = {A_ode[-1]:.6f}  (expected: {A0*math.exp(-k*t_end):.6f})")
    max_error = max(abs(a - math.exp(-k * t) * A0) for a, t in zip(A_ode, ode_result["t"]))
    print(f"    Max abs error vs analytical: {max_error:.2e}")

    # 3. SymPy verification: solve the ODE symbolically
    print(f"\n[3] SymPy symbolic verification:")
    deriv = sym_engine.differentiate(f"-{k}*y", variable="y")
    print(f"    d/dy(-{k}y) = {deriv['symbolic']}")
    # Integrate: A(t) = A0 * exp(-k*t) [confirmed by integrating -k dt]
    integral = sym_engine.integrate(f"-{k}", variable="t")
    print(f"    ∫(-{k}) dt = {integral['symbolic']} + C  => A(t) = A₀·exp({integral['symbolic']})")

    # 4. Consecutive reactions A -> B -> C
    print(f"\n[4] Consecutive reactions A -> B -> C  (k1=0.25, k2=0.1)")
    k1, k2 = 0.25, 0.10
    consec = kinetics.simulate_consecutive_reactions(k1, k2, A0, t_span=40)
    print(f"    Maximum [B] = {consec['B_max']:.4f}  at t = {consec['t_Bmax']:.4f} s")
    print(f"    A(40s) = {consec['A'][-1]:.4f}")
    print(f"    B(40s) = {consec['B'][-1]:.4f}")
    print(f"    C(40s) = {consec['C'][-1]:.4f}")

    # 5. Equilibrium kinetics
    print(f"\n[5] Reversible equilibrium A <-> B  (kf=0.3, kr=0.1)")
    eq = kinetics.simulate_equilibrium(k_forward=0.3, k_reverse=0.1, A0=1.0, B0=0.0, t_span=30)
    print(f"    K_eq = {eq['K_eq']:.2f}")
    print(f"    A(eq) = {eq['A_equilibrium']:.4f}  B(eq) = {eq['B_equilibrium']:.4f}")

    # 6. Plot: compare analytical vs numerical + consecutive
    print(f"\nGenerating plots...")

    b64_compare = plot_engine.plot_multiple(
        datasets=[
            {"x": t_arr, "y": A_chem, "label": "Analytical A(t)", "color": "#2196F3"},
            {"x": ode_result["t"], "y": A_ode, "label": "RK45 A(t)", "color": "#FF5722"},
        ],
        title=f"First-Order Decay: k={k}, A₀={A0}",
        xlabel="Time [s]",
        ylabel="[A]",
    )
    out_compare = os.path.join(os.path.dirname(__file__), "decay_comparison.png")
    with open(out_compare, "wb") as f:
        f.write(base64.b64decode(b64_compare))
    print(f"  Comparison plot saved to: {out_compare}")

    b64_consec = plot_engine.plot_time_series(
        consec["t"],
        [consec["A"], consec["B"], consec["C"]],
        labels=["[A]", "[B]", "[C]"],
        title=f"Consecutive Reactions A→B→C (k₁={k1}, k₂={k2})",
        xlabel="Time [s]",
        ylabel="Concentration",
    )
    out_consec = os.path.join(os.path.dirname(__file__), "consecutive_reactions.png")
    with open(out_consec, "wb") as f:
        f.write(base64.b64decode(b64_consec))
    print(f"  Consecutive reactions plot saved to: {out_consec}")

    b64_eq = plot_engine.plot_time_series(
        eq["t"],
        [eq["A"], eq["B"]],
        labels=["[A]", "[B]"],
        title=f"Reversible Equilibrium A ⇌ B (K_eq={eq['K_eq']:.1f})",
        xlabel="Time [s]",
        ylabel="Concentration",
    )
    out_eq = os.path.join(os.path.dirname(__file__), "equilibrium.png")
    with open(out_eq, "wb") as f:
        f.write(base64.b64decode(b64_eq))
    print(f"  Equilibrium plot saved to: {out_eq}")

    # 7. Via SimLabCore hybrid dispatch
    print(f"\n--- Via SimLabCore (hybrid domain) ---")
    from simlab.core.schemas.experiment import ExperimentRequest
    req = ExperimentRequest(
        domain="hybrid",
        type="reaction_kinetics",
        parameters={"k": k, "A0": A0, "t_span": t_end},
        outputs=["plot"],
    )
    result = core.run_experiment(req)
    print(f"Status: {result.status}")
    if result.status == "success":
        print(f"Half-life: {result.results.get('half_life'):.4f} s")
    if result.plots:
        print(f"Plot generated: {len(result.plots[0])} chars (base64)")

    print("\nDone.")


if __name__ == "__main__":
    main()
