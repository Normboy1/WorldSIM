"""Example: Projectile motion at 45 degrees, v0 = 25 m/s.

Demonstrates:
- Classical mechanics simulation
- Trajectory plotting
- SimLabCore integration
"""

import sys
import os
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simlab.core.engine.simlab_core import SimLabCore
from simlab.engines.physics.classical import ClassicalMechanicsEngine
from simlab.engines.visualization.plots import PlotEngine


def main():
    core = SimLabCore()
    engine = ClassicalMechanicsEngine()
    plot_engine = PlotEngine()

    v0 = 25.0       # m/s
    angle = 45.0    # degrees
    g = 9.81        # m/s^2

    print(f"Projectile Motion Simulation")
    print(f"v0 = {v0} m/s, angle = {angle}°, g = {g} m/s²")
    print("=" * 50)

    # Run simulation
    result = engine.simulate_projectile_motion(v0, angle, g=g)

    print(f"\nResults:")
    print(f"  Max height:      {result['max_height']:.3f} m")
    print(f"  Range:           {result['range']:.3f} m")
    print(f"  Time of flight:  {result['time_of_flight']:.3f} s")
    print(f"  Time to apex:    {result['t_apex']:.3f} s")

    # Theoretical verification (no air resistance)
    import math
    v0x = v0 * math.cos(math.radians(angle))
    v0y = v0 * math.sin(math.radians(angle))
    expected_range = v0**2 * math.sin(math.radians(2 * angle)) / g
    expected_height = v0y**2 / (2 * g)
    print(f"\nAnalytical check:")
    print(f"  Expected range:  {expected_range:.3f} m (diff: {abs(result['range'] - expected_range):.6f} m)")
    print(f"  Expected height: {expected_height:.3f} m (diff: {abs(result['max_height'] - expected_height):.6f} m)")

    # Plot trajectory
    print("\nGenerating trajectory plot...")
    b64 = plot_engine.plot_trajectory(
        result["x"], result["y"],
        title=f"Projectile Motion: v₀={v0} m/s, θ={angle}°",
        xlabel="Horizontal distance [m]",
        ylabel="Height [m]",
    )
    out_path = os.path.join(os.path.dirname(__file__), "projectile_trajectory.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"Trajectory plot saved to: {out_path}")

    # Speed vs time
    b64_speed = plot_engine.plot_data(
        result["t"], result["speed"],
        title="Speed vs Time",
        xlabel="Time [s]",
        ylabel="Speed [m/s]",
    )
    out_speed = os.path.join(os.path.dirname(__file__), "projectile_speed.png")
    with open(out_speed, "wb") as f:
        f.write(base64.b64decode(b64_speed))
    print(f"Speed plot saved to: {out_speed}")

    # Compare Earth vs Moon/Mars via SimLabCore (environment g merged into dispatch)
    print("\n--- Earth vs Moon/Mars (SimLabCore + environment) ---")
    for body, g_val in [("Earth", 9.81), ("Moon", 1.62), ("Mars", 3.72)]:
        r = core.simulate_physics(
            "projectile_motion",
            {"v0": v0, "angle_deg": angle},
            environment={"g": g_val},
        )
        res_core = r.results if r.status == "success" else {}
        range_core = res_core.get("range")
        res_body = engine.simulate_projectile_motion(v0, angle, g=g_val)
        match = (
            range_core is not None
            and abs(float(range_core) - res_body["range"]) < 1e-6
        )
        print(
            f"  {body:5s}: range = {res_body['range']:7.2f} m, "
            f"height = {res_body['max_height']:6.2f} m, "
            f"t_flight = {res_body['time_of_flight']:.2f} s "
            f"(core dispatch matches direct: {match})"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
