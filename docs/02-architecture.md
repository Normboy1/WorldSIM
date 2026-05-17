# Architecture

## High-level flow

```
ExperimentRequest
    → SimLabCore.run_experiment()
        → SimLabValidator     (required params, safety/stability hints)
        → ExperimentDispatcher (routing table + param renaming)
            → Domain engine   (math, physics, …)
            → PlotEngine      (optional, if outputs contain "plot" or "all")
    → ExperimentResult
```

## Major components

| Layer | Role |
|-------|------|
| **`simlab.core.schemas.experiment`** | `ExperimentRequest`, `ExperimentResult`, `ValidationResult` — shared JSON-serializable contracts. |
| **`simlab.core.engine.simlab_core`** | `SimLabCore` — public façade: `run_experiment`, domain shortcuts (`solve_math`, `simulate_physics`, …), utilities (`available_experiments`, result summaries). |
| **`simlab.core.validation.validator`** | `SimLabValidator` — required-parameter matrix, keyword/solver stability warnings, blocklist hooks for unsafe requests. |
| **`simlab.core.router.dispatcher`** | `ExperimentDispatcher` — `_ROUTING_TABLE` maps `(domain, type)` → `(engine_factory, method_name, param_map)`. Merges `environment` then `parameters` before calling the engine. |
| **`simlab.engines.*`** | Domain engines (math, physics, chemistry, materials, atomic, nuclear, data, visualization, control). |
| **`simlab.api.fastapi_server`** | REST HTTP API (`simlab-api` entry point). |
| **`simlab.mcp.simlab_mcp_server`** | Primary MCP server exposing tools that call `SimLabCore`. |
| **`simlab.mcp.simlab_knowledge_server`** | Secondary MCP server for experiment history, notes, and proof documents. |

## Special dispatcher behaviors

- **`materials` + `create_lattice`** — reads `lattice_type` (`fcc`, `bcc`, `sc` / `simple_cubic`) and dispatches to the correct lattice builder.
- **String return from engine** — some nuclear/plotting paths return a raw base64 PNG string; the dispatcher puts that in `plots` and clears numeric `results` for that path.
- **Visualization** — `_generate_plots` inspects result keys (`x`/`y` trajectories, `t` + series, stress–strain) and appends base64 PNGs when requested.

## Optional backends

Install extras from `pyproject.toml` (e.g. `pip install -e ".[chemistry]"`):

- **`chemistry`** — RDKit.
- **`atomic`** — ASE.
- **`materials`** — pymatgen.
- **`control`** — `python-control`.
- **`physics`** — PyBullet (listed; not all routes require it).
- **`ml`** — PyTorch (listed for future/optional use).

If an extra is missing, importing that engine may fail at dispatch time; the routing table still lists the experiment types.
