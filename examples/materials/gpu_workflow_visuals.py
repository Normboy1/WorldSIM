"""Generate WorldSim Materials visual examples.

Run from the repository root:

    python3 examples/materials/gpu_workflow_visuals.py
"""

from __future__ import annotations

import base64
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simlab.core.engine.simlab_core import SimLabCore

OUT_DIR = Path(__file__).resolve().parent


def _write_plots(prefix: str, plots: list[str]) -> list[Path]:
    paths: list[Path] = []
    for index, plot_b64 in enumerate(plots, start=1):
        path = OUT_DIR / f"{prefix}_{index}.png"
        path.write_bytes(base64.b64decode(plot_b64))
        paths.append(path)
    return paths


def main() -> None:
    core = SimLabCore()
    cases = [
        (
            "alloy_screen",
            "alloy_property_prediction",
            {
                "composition": {"Ni": 0.62, "Cr": 0.18, "Co": 0.10, "Al": 0.06, "Ti": 0.04},
                "temperature_K": 1050.0,
                "heat_treatment": "solution + aging",
            },
        ),
        (
            "degradation_screen",
            "degradation_prediction",
            {
                "composition": {"Fe": 0.82, "Cr": 0.12, "Ni": 0.06},
                "environment": {
                    "temperature_K": 900.0,
                    "oxygen_partial_pressure_atm": 0.21,
                    "chloride_mol_L": 0.1,
                    "pH": 6.5,
                    "hydrogen_partial_pressure_atm": 0.02,
                },
                "exposure_hours": 100.0,
            },
        ),
        (
            "microstructure_diffusion",
            "microstructure_diffusion",
            {
                "grid_shape": [64, 48],
                "diffusivity_m2_s": 0.5,
                "time_s": 8.0,
                "steps": 80,
            },
        ),
        (
            "surrogate_plan",
            "surrogate_model_plan",
            {
                "dataset_size": 1000,
                "target_properties": ["phase_stability_score", "oxide_thickness_um"],
            },
        ),
    ]

    for prefix, sim_type, params in cases:
        result = core.simulate_materials(sim_type, params, outputs=["plot"])
        if result.status != "success":
            raise RuntimeError(f"{sim_type} failed: {result.errors}")
        for path in _write_plots(prefix, result.plots):
            print(path)


if __name__ == "__main__":
    main()
