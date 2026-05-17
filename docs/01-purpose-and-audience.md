# Purpose and audience

## What SIMLAB is

**SIMLAB** is a **unified scientific simulation and lookup platform** (version 0.1) that exposes one consistent contract—an **experiment request** with a **domain**, **type**, and **parameters**—across mathematics, physics, chemistry, materials, atomic structure, nuclear models, external scientific data (arXiv, PubChem), and classical control systems.

It is built for:

- **AI agents and assistants** that need callable, structured tools (via **MCP**) instead of ad-hoc scripts.
- **Humans and notebooks** that want a **single Python API** (`SimLabCore`) or a **REST API** (FastAPI).
- **Teaching and prototyping** where correctness and clarity matter more than replacing domain-specific HPC codes.

## Purpose in one sentence

> Provide a **single orchestration layer** so “run this scientific task” is always the same shape—validate, dispatch, optionally visualize—regardless of whether the underlying engine is SymPy, SciPy, RDKit, or an HTTP-backed data source.

## Design goals (as reflected in the code)

1. **One schema** — `ExperimentRequest` / `ExperimentResult` (Pydantic v2) for all domains.
2. **Explicit routing** — a dispatcher table maps `(domain, type)` to an engine method (no hidden magic).
3. **Lazy engines** — heavy backends import only when a route needs them (faster cold start).
4. **Environment overlays** — e.g. `environment={"g": 1.62}` merges with `parameters` so agents can set defaults like gravity without duplicating every parameter key.
5. **Optional heavy dependencies** — chemistry (RDKit), atomic/materials/control backends are extras; core install stays lighter.
6. **Agent ergonomics** — MCP tools wrap the same core as the REST API so behavior stays aligned.

## Who it is not trying to replace

SIMLAB is **not** a full replacement for:

- General-purpose **computer algebra systems** (full Mathematica-style workflows).
- **Ab initio** quantum chemistry or **MD** production suites.
- **Relativistic** or **quantum field** simulations (many classical models assume non-relativistic, idealized conditions).

It **is** a practical **integration hub** and **educational** stack for coherent multi-domain demos, coursework, and agent toolchains.
