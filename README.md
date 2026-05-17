# WorldSIM v0.2

**AI-powered scientific simulation platform**

WorldSIM provides a unified interface for running scientific simulations across mathematics, physics, chemistry, materials science (including metallurgy, alloy discovery, and GPU-accelerated microstructure prediction), atomic/nuclear science, data lookup, and control systems. It exposes both an MCP server (for Claude/AI agent integration) and a REST API (FastAPI).

## Domains

| Domain | Capabilities |
|--------|-------------|
| **Math** | Symbolic algebra, calculus, linear algebra, ODE solving, optimization, statistics |
| **Physics** | Classical mechanics, electromagnetism, thermodynamics, fluid dynamics |
| **Chemistry** | Molecular analysis (RDKit), reaction kinetics, equilibrium |
| **Materials** | Crystal lattices (FCC/BCC/SC), stress-strain, diffusion (Fick/Arrhenius/Darken), phase transformations (JMAK/TTT/grain growth), oxidation/corrosion (Ellingham, Pourbaix, SCC), forging, casting |
| **Metallurgy (GPU)** | NVIDIA ALCHEMI atomistic MD, Warp diffusion/grain kernels, PhysicsNeMo surrogate models, alloy screening, degradation prediction |
| **Atomic** | Electron configurations, element building, hydrogen-like orbitals, ASE backend |
| **Nuclear** | Binding energy, stability, fusion/fission, radioactive decay chains |
| **Data** | arXiv and PubChem lookups with HTTP cache |
| **Control** | Transfer functions, PID, Bode, root locus, Nyquist |
| **Visualization** | Function plots, vector fields, molecule renders, heatmaps, reports |

## Quick Start

```bash
# Base install
pip install -e .

# With optional domain extras
pip install -e ".[chemistry,atomic,materials]"

# With NVIDIA GPU stack (ALCHEMI, PhysicsNeMo, Warp, cuPyNumeric)
pip install -e ".[nvidia]"

# Run the FastAPI server
worldsim-api

# Run the MCP server
worldsim-mcp

# Run the knowledge/proof MCP server
worldsim-knowledge-mcp
```

## Usage Example

```python
from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest

core = SimLabCore()

req = ExperimentRequest(
    domain="physics",
    type="projectile_motion",
    parameters={"v0": 25.0, "angle_deg": 45.0},
    outputs=["plot"]
)
result = core.run_experiment(req)
print(result.results)
```

## MCP Tools

- `run_experiment` — Full experiment dispatch
- `solve_math` — Math engine shortcut
- `simulate_physics` — Physics engine shortcut
- `simulate_chemistry` — Chemistry engine shortcut
- `simulate_materials` — Materials engine shortcut
- `simulate_atomic` — Atomic/element engine shortcut
- `simulate_nuclear` — Nuclear engine shortcut
- `query_data` — arXiv/PubChem data lookup shortcut
- `analyze_control` — Control-system analysis shortcut
- `generate_visualization` — Standalone visualization
- `export_report` — Markdown/HTML/JSON export
- `check_safety` — Pre-flight safety check

## REST API

```
POST /experiment
POST /math
POST /physics
POST /chemistry
POST /materials
POST /atomic
POST /nuclear
POST /data
POST /control
POST /visualize
POST /report
POST /safety
GET  /health
GET  /constants
GET  /experiments
```

Browser CORS defaults to local development origins. Set `SIMLAB_CORS_ORIGINS`
to a comma-separated allowlist for deployment; wildcard origins never enable
credentialed CORS.

External data lookups use a local TTL cache under `proof/cache/http` by default.
Set `SIMLAB_DATA_CACHE_DIR` or `SIMLAB_DATA_CACHE_TTL_S` to control cache
location and freshness. If a live request fails, stale cached data is used when
available.

Nuclear simulations expose SEMF model limits and compare common benchmark
isotopes against rounded empirical binding-energy references. Atomic hydrogen
orbital tools include a radial-normalization check for probability densities.

## Architecture

```
ExperimentRequest
    → SimLabCore (orchestrator)
        → SimLabValidator (safety + param checks)
        → ExperimentDispatcher (routing table)
            → Engine (math/physics/chemistry/materials/atomic/nuclear/data/control)
            → PlotEngine (optional visualization)
    → ExperimentResult
```

## Dependencies

- **SymPy** — symbolic math
- **NumPy / SciPy** — numerical computation
- **Matplotlib / Plotly** — visualization
- **RDKit** — molecular chemistry (optional)
- **ASE / pymatgen** — atomic and materials backends (optional)
- **python-control** — control-systems backend (optional)
- **Pydantic v2** — schema validation
- **FastAPI** — REST server
- **MCP** — Anthropic MCP protocol server

## Documentation

Extended guides (purpose, full experiment catalog, architecture, MCP/REST, knowledge server, limitations) live under **[docs/](docs/README.md)**.

