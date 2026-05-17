# REST API overview

The FastAPI app is started via **`simlab-api`**. Routes live under `simlab/api/fastapi_server/routes.py` and mirror the shortcut style of the MCP server (JSON in, JSON out).

## Endpoints

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Liveness / service metadata. |
| `GET` | `/constants` | Physical constants bundle (SI). |
| `POST` | `/experiment` | Full `ExperimentRequest` body. |
| `POST` | `/math` | `MathRequest`: `domain_type`, `parameters`, `outputs`. |
| `POST` | `/physics` | `PhysicsRequest`: adds `environment`. |
| `POST` | `/chemistry` | `ChemistryRequest`. |
| `POST` | `/materials` | `MaterialsRequest`. |
| `POST` | `/atomic` | `AtomicRequest`. |
| `POST` | `/nuclear` | `NuclearRequest`. |
| `POST` | `/data` | `DataRequest`: `query_type`, `parameters`. |
| `POST` | `/control` | `ControlRequest`. |
| `POST` | `/visualize` | Standalone visualization from supplied data. |
| `POST` | `/report` | Export report from a prior `ExperimentResult` dict. |
| `POST` | `/safety` | Validator-only preflight. |
| `GET` | `/experiments` | List or filter recorded experiments when wired to storage (see knowledge docs). |

## Typical client flow

1. `GET /health` — confirm service.
2. `POST /physics` (or domain-specific shortcut) with JSON matching the Pydantic request models in `routes.py`.
3. Parse `ExperimentResult`-shaped JSON: `status`, `results`, `plots`, `warnings`, `errors`.

For the exact request field names, refer to the **`BaseModel` subclasses** at the top of `routes.py` and the MCP tool docstrings (they are kept parallel by design).
