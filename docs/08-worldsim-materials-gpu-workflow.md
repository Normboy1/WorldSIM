# WorldSim Materials GPU workflow

This project copy is scoped as a focused research workflow for alloy discovery
and degradation prediction.

## Research question

Can GPU accelerated atomistic simulation and physics-informed machine learning
predict alloy stability, microstructure evolution, and chemical degradation
faster than traditional simulation workflows?

## Backend roles

| Backend | Role |
|---------|------|
| ALCHEMI | Atomistic chemistry/materials simulation, batched molecular dynamics, relaxation, and MLIP operations. |
| PhysicsNeMo | Physics-informed ML, neural operators, surrogate models, and active-learning loops. |
| Warp | Custom CUDA kernels for diffusion, phase transformation, precipitation, and grain growth. |
| Omniverse | Digital twin visualization and inspection of microstructure/degradation fields. |
| CUDA/HPC SDK/cuPyNumeric | Accelerated numerical sweeps and array workloads. |

## Current runnable experiments

All endpoints are under `domain="materials"`:

| Type | Required parameters | Notes |
|------|---------------------|-------|
| `gpu_backend_status` | none | Detects optional accelerator modules and reports intended roles. |
| `alloy_property_prediction` | `composition` | Screening model for density, modulus, solidus proxy, strength proxy, phase stability, and oxidation resistance. |
| `degradation_prediction` | `composition`, `environment`, `exposure_hours` | Parabolic oxidation and corrosion/hydrogen risk proxy. |
| `microstructure_diffusion` | none | Small NumPy fallback demo with Warp as the production kernel target. |
| `surrogate_model_plan` | `dataset_size`, `target_properties` | PhysicsNeMo-oriented training and active-learning plan. |

Example:

```python
from simlab.core.engine.simlab_core import SimLabCore

core = SimLabCore()
result = core.simulate_materials(
    "alloy_property_prediction",
    {
        "composition": {"Ni": 0.62, "Cr": 0.18, "Co": 0.10, "Al": 0.06, "Ti": 0.04},
        "temperature_K": 1050.0,
        "heat_treatment": "solution + aging",
    },
)
print(result.results["property_estimates"])
```

## Proposal deliverables

- GPU accelerated alloy simulation prototype.
- Dataset of simulated alloy compositions, environments, and microstructure outcomes.
- PhysicsNeMo surrogate model for property/degradation prediction.
- Omniverse visualization workflow for microstructure and chemical degradation.
- CPU baseline versus NVIDIA accelerated benchmark.
- Technical report or paper.
- Open source repository.

## Scope boundary

Do not present this as a full chemistry and metallurgy engine. Present it as a
focused GPU accelerated workflow for alloy microstructure and chemical stability
prediction. The built-in screening models are placeholders for reproducible API
development and should be calibrated with atomistic, phase-field, or
experimental data before scientific claims are made.
