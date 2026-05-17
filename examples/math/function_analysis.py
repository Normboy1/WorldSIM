"""Example: Complete function analysis of f(x) = x^3 - 4x.

Demonstrates:
- Symbolic differentiation
- Critical point detection
- Taylor series expansion
- Function plotting via SimLabCore
"""

import sys
import os

# Allow running from repository root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest
from simlab.engines.math.symbolic import SymbolicEngine
from simlab.engines.visualization.plots import PlotEngine


def main():
    core = SimLabCore()
    sym = SymbolicEngine()
    plot_engine = PlotEngine()

    expr = "x**3 - 4*x"
    print(f"Analyzing f(x) = {expr}")
    print("=" * 50)

    # 1. Derivative
    deriv = sym.differentiate(expr, "x", order=1)
    print(f"\nf'(x)  = {deriv['symbolic']}")
    print(f"LaTeX  : {deriv['latex']}")

    deriv2 = sym.differentiate(expr, "x", order=2)
    print(f"f''(x) = {deriv2['symbolic']}")

    # 2. Critical points
    crit = sym.find_critical_points(expr, "x")
    print(f"\nCritical points:")
    for pt in crit["critical_points"]:
        print(f"  x = {pt['x']:>8}  (numeric: {pt['x_numeric']:.4f})  "
              f"y = {pt['y_numeric']:.4f}  => {pt['type']}")

    # 3. Taylor series
    taylor = sym.taylor_series(expr, "x", point=0, order=6)
    print(f"\nTaylor series (around x=0, order 6):")
    print(f"  {taylor['symbolic']}")

    # 4. Solve f(x) = 0
    roots = sym.solve_equation(expr, "x")
    print(f"\nRoots of f(x) = 0:")
    for s, n in zip(roots["solutions"], roots["numeric"]):
        n_str = f"{n:.4f}" if n is not None else "complex"
        print(f"  x = {s:>12}  ({n_str})")

    # 5. Definite integral from -2 to 2
    integral = sym.integrate(expr, "x", limits=[-2, 2])
    print(f"\n∫[-2,2] f(x) dx = {integral['symbolic']}  (numeric: {integral['numeric']})")

    # 6. Plot
    print("\nGenerating plot...")
    b64 = plot_engine.plot_function(
        expr, x_range=[-3, 3],
        title="f(x) = x³ - 4x: function and analysis"
    )

    # Save plot as PNG
    import base64
    out_path = os.path.join(os.path.dirname(__file__), "function_analysis.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"Plot saved to: {out_path}")

    # 7. Via SimLabCore dispatcher
    print("\n--- Via SimLabCore.solve_math ---")
    result = core.solve_math("differentiate", {"expression": expr, "variable": "x"})
    print(f"Status: {result.status}")
    print(f"f'(x) = {result.results.get('symbolic')}")

    result2 = core.solve_math("critical_points", {"expression": expr, "variable": "x"})
    print(f"\nCritical points (via core):")
    for pt in result2.results.get("critical_points", []):
        print(f"  x={pt['x_numeric']:.4f} => {pt['type']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
