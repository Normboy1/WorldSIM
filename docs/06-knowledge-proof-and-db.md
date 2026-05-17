# Knowledge MCP, database, and proofs

## Second MCP server: `simlab-knowledge-mcp`

The **Knowledge** server is separate from the main simulation MCP. It focuses on **provenance**, **reuse**, and **documentation** of what was already run.

Typical agent workflows:

1. Run an experiment via **`simlab-mcp`** (or REST / `SimLabCore`).
2. Call **`record_experiment`** on the knowledge server with the same `exp_id`, parameters, and results so runs are **searchable** later.
3. Optionally auto-generate **LaTeX / PDF proof** artifacts through `ProofEngine`.

Entry point and package: `simlab.mcp.simlab_knowledge_server`.

## SQLite experiment database

`simlab.db.experiment_db.ExperimentDB` stores experiment history (default path under the repo’s `proof/` tree as `simlab.db`—see code for `_DEFAULT_DB`).

Use knowledge MCP tools to:

- Record runs with status, warnings, errors, and free-form notes.
- Query summaries and prior results instead of re-running expensive or rate-limited calls.

## Proof engine

`simlab.proof.proof_engine.ProofEngine` assembles structured **proof-style documents** (LaTeX-oriented) summarizing experiments for lab notebooks, coursework, or audit trails.

**Note:** PDF generation depends on a working LaTeX toolchain on the host when that path is enabled.

## Relationship to the main server

| Server | Responsibility |
|--------|----------------|
| **SIMLAB** (`simlab-mcp`) | Execute science: dispatch engines, return results and plots. |
| **SIMLAB-Knowledge** (`simlab-knowledge-mcp`) | Remember and explain: persist, search, export proofs. |
