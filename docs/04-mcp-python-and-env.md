# MCP, Python API, and environment

## Python API

Primary entry point:

```python
from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest

core = SimLabCore()
result = core.run_experiment(ExperimentRequest(
    domain="physics",
    type="projectile_motion",
    parameters={"v0": 25.0, "angle_deg": 45.0},
    outputs=["plot"],
    environment={},  # merged into parameters (e.g. {"g": 1.62})
))
```

Shortcuts mirror MCP tools: `solve_math`, `simulate_physics`, `simulate_chemistry`, `simulate_materials`, `simulate_atomic`, `simulate_nuclear`, `query_data`, and `analyze_control`.

## CLI entry points (`pyproject.toml`)

| Command | Module |
|---------|--------|
| `simlab-api` | `simlab.api.fastapi_server.main:start` |
| `simlab-mcp` | `simlab.mcp.simlab_mcp_server.server:main` |
| `simlab-knowledge-mcp` | `simlab.mcp.simlab_knowledge_server.server:main` |

### Running MCP without installing scripts

```bash
python -m simlab.mcp.simlab_mcp_server.server
python -m simlab.mcp.simlab_knowledge_server.server
```

## Main MCP server (`simlab-mcp`)

Exposes tools that delegate to `SimLabCore`, including (non-exhaustive):

- `run_experiment` — full `ExperimentRequest` dict.
- `solve_math`, `simulate_physics`, `simulate_chemistry`, `simulate_materials`, `simulate_atomic`, `simulate_nuclear`.
- `query_data` — arXiv / PubChem shortcuts.
- `analyze_control` — control-system shortcuts.
- `generate_visualization`, `export_report`, `check_safety`.

See tool docstrings in `simlab/mcp/simlab_mcp_server/server.py` for parameter examples.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `SIMLAB_CORS_ORIGINS` | Comma-separated allowed origins for the FastAPI app (deployment). |
| `SIMLAB_CORS_ALLOW_CREDENTIALS` | String `"true"` / `"false"`; wildcard origins do not enable credentialed CORS. |
| `SIMLAB_DATA_CACHE_DIR` | Override directory for HTTP cache used by data engines. |
| `SIMLAB_DATA_CACHE_TTL_S` | Cache TTL for data lookups. |

## Outputs

- **`outputs`** may include `plot`, `report`, `json`, `latex`, or `all` (see schema field descriptions).
- Plots in `ExperimentResult.plots` are typically **base64-encoded PNG** strings.

## Install extras

Examples:

```bash
pip install -e ".[chemistry]"
pip install -e ".[atomic,materials,control]"
pip install -e ".[all]"
```

See `[project.optional-dependencies]` in `pyproject.toml` for the canonical list.
