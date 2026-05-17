"""SIMLAB Knowledge MCP Server.

Exposes experiment history, proof management, and accumulated scientific
knowledge as MCP tools.  An AI agent can use these tools to:

  - Record every experiment result with full provenance
  - Look up what has already been tried (and whether it worked)
  - Add notes and insights about particular simulations
  - Generate LaTeX/PDF proof documents automatically
  - Query the accumulated knowledge base

Run with:
  python -m simlab.mcp.simlab_knowledge_server.server
Or via the CLI entry point:
  simlab-knowledge-mcp
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from simlab.db.experiment_db import ExperimentDB
from simlab.proof.proof_engine import ProofEngine

mcp = FastMCP("SIMLAB-Knowledge")
_db = ExperimentDB()
_proof = ProofEngine()


# ── Experiment recording ────────────────────────────────────────────────────

@mcp.tool()
async def record_experiment(
    exp_id: str,
    domain: str,
    exp_type: str,
    parameters: dict,
    results: dict,
    status: str = "success",
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    notes: str = "",
    generate_proof: bool = True,
) -> dict:
    """Record a completed SIMLAB experiment to the knowledge database.

    Also generates a LaTeX/PDF proof document if generate_proof=True.

    Parameters
    ----------
    exp_id      : unique experiment identifier (e.g. 'exp_carbon12_001')
    domain      : 'math' | 'physics' | 'chemistry' | 'atomic' | 'nuclear' | 'hybrid'
    exp_type    : simulation type string (e.g. 'projectile_motion', 'create_element')
    parameters  : dict of input parameters
    results     : dict of output results (plots as base64 strings will be extracted)
    status      : 'success' | 'error' | 'partial'
    warnings    : list of warning messages
    errors      : list of error messages
    notes       : free-form notes about this experiment
    generate_proof : compile LaTeX PDF proof (default True)
    """
    # Extract plots from results
    plots_b64: dict[str, str] = {}
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, str) and len(v) > 200 and _is_b64(v):
            plots_b64[k] = v
        else:
            clean_results[k] = v

    # Also check nested "plots" dict
    if "plots" in results and isinstance(results["plots"], dict):
        for k, v in results["plots"].items():
            if isinstance(v, str) and len(v) > 200 and _is_b64(v):
                plots_b64[k] = v

    proof_info = {}
    if generate_proof:
        try:
            proof_info = _proof.save_proof(
                exp_id=exp_id,
                domain=domain,
                exp_type=exp_type,
                parameters=parameters,
                results=clean_results,
                plots_b64=plots_b64,
                status=status,
                warnings=warnings,
                errors=errors,
                extra_notes=notes,
            )
        except Exception as exc:
            proof_info = {"error": str(exc), "compiled_ok": False}

    _db.save_experiment(
        exp_id=exp_id,
        domain=domain,
        exp_type=exp_type,
        parameters=parameters,
        results=clean_results,
        status=status,
        warnings=warnings,
        errors=errors,
        proof_dir=proof_info.get("proof_dir"),
        pdf_path=proof_info.get("pdf_path"),
    )

    if notes:
        _db.add_note(notes, category="general", experiment_id=exp_id)

    return {
        "recorded": True,
        "exp_id": exp_id,
        "status": status,
        "plots_saved": len(plots_b64),
        "proof": proof_info,
    }


@mcp.tool()
async def get_experiment(exp_id: str) -> dict:
    """Retrieve a recorded experiment by its ID.

    Returns full record including parameters, results, status, proof paths.
    """
    result = _db.get_experiment(exp_id)
    if not result:
        return {"found": False, "exp_id": exp_id}
    plots = _db.get_plots(exp_id)
    notes = _db.get_notes(experiment_id=exp_id)
    result["plots_files"] = plots
    result["notes"] = notes
    return result


# ── Querying history ────────────────────────────────────────────────────────

@mcp.tool()
async def list_experiments(
    domain: str | None = None,
    status: str | None = None,
    exp_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List recorded experiments with optional filters.

    Parameters
    ----------
    domain   : filter by domain ('math', 'physics', 'chemistry', 'atomic', 'nuclear', 'hybrid')
    status   : filter by status ('success', 'error', 'partial')
    exp_type : filter by experiment type string
    limit    : max results to return (default 20)
    """
    rows = _db.list_experiments(domain=domain, status=status, exp_type=exp_type, limit=limit)
    return [_summarise(r) for r in rows]


@mcp.tool()
async def search_experiments(query: str, limit: int = 15) -> list[dict]:
    """Search experiment history by keyword.

    Searches across domain, type, parameters, and results fields.
    """
    rows = _db.search_experiments(query, limit=limit)
    return [_summarise(r) for r in rows]


@mcp.tool()
async def what_worked(domain: str | None = None) -> dict:
    """Return all successful experiments, optionally filtered by domain.

    Use this to understand what has already been validated.
    """
    rows = _db.what_worked(domain=domain)
    return {
        "domain": domain or "all",
        "count": len(rows),
        "experiments": [_summarise(r) for r in rows],
    }


@mcp.tool()
async def what_failed(domain: str | None = None) -> dict:
    """Return all failed experiments, optionally filtered by domain.

    Use this to avoid repeating known failures or to understand limitations.
    """
    rows = _db.what_failed(domain=domain)
    return {
        "domain": domain or "all",
        "count": len(rows),
        "experiments": [_summarise(r) for r in rows],
        "tip": "Check 'errors' field on individual records for root cause.",
    }


@mcp.tool()
async def recent_experiments(n: int = 10) -> list[dict]:
    """Return the N most recently run experiments."""
    rows = _db.recent(n)
    return [_summarise(r) for r in rows]


# ── Notes ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def add_note(
    body: str,
    category: str = "general",
    experiment_id: str | None = None,
) -> dict:
    """Add a note or insight to the knowledge base.

    Parameters
    ----------
    body          : the note text
    category      : 'worked' | 'failed' | 'insight' | 'warning' | 'general'
    experiment_id : link this note to a specific experiment (optional)
    """
    note_id = _db.add_note(body=body, category=category, experiment_id=experiment_id)
    return {"note_id": note_id, "category": category, "linked_exp": experiment_id}


@mcp.tool()
async def get_notes(
    experiment_id: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """Retrieve notes from the knowledge base.

    Filter by experiment_id and/or category ('worked', 'failed', 'insight', 'warning', 'general').
    """
    return _db.get_notes(experiment_id=experiment_id, category=category)


# ── Knowledge entries ─────────────────────────────────────────────────────────

@mcp.tool()
async def store_knowledge(
    domain: str,
    key: str,
    value: str,
    exp_type: str | None = None,
    confidence: float = 1.0,
    source_exp_id: str | None = None,
) -> dict:
    """Store a scientific insight or finding in the knowledge base.

    Examples:
      domain='nuclear', key='iron_56_most_stable',
        value='Fe-56 has the highest B/A (~8.79 MeV) of any nucleus'
      domain='physics', key='rk45_unstable_stiff',
        value='RK45 fails for stiff ODEs; use LSODA instead'

    Parameters
    ----------
    domain        : scientific domain
    key           : short identifier for this knowledge entry
    value         : the finding or insight (plain text)
    exp_type      : experiment type this knowledge applies to
    confidence    : 0.0–1.0 (1.0 = verified, 0.5 = likely, 0.0 = uncertain)
    source_exp_id : experiment ID that produced this knowledge
    """
    kid = _db.store_knowledge(
        domain=domain, key=key, value=value,
        exp_type=exp_type, confidence=confidence,
        source_exp_id=source_exp_id,
    )
    return {"knowledge_id": kid, "domain": domain, "key": key}


@mcp.tool()
async def get_knowledge(
    domain: str | None = None,
    exp_type: str | None = None,
    key: str | None = None,
) -> list[dict]:
    """Query the accumulated scientific knowledge base.

    Returns knowledge entries sorted by confidence (highest first).
    """
    return _db.get_knowledge(domain=domain, exp_type=exp_type, key=key)


# ── Summary & reporting ───────────────────────────────────────────────────────

@mcp.tool()
async def knowledge_summary() -> dict:
    """Return a high-level summary of the SIMLAB knowledge database.

    Shows total experiments, breakdown by domain and status, and counts
    of notes and knowledge entries.
    """
    stats = _db.summary_stats()
    recent = _db.recent(5)
    stats["recent_experiments"] = [_summarise(r) for r in recent]
    return stats


@mcp.tool()
async def generate_proof(
    exp_id: str,
    title: str | None = None,
    extra_notes: str = "",
) -> dict:
    """(Re)generate the LaTeX/PDF proof document for an existing experiment.

    Retrieves the experiment from the database and regenerates all artifacts
    in proof/{domain}/{exp_id}/.  Returns paths to the .tex and .pdf files.
    """
    exp = _db.get_experiment(exp_id)
    if not exp:
        return {"error": f"Experiment '{exp_id}' not found in database."}

    plots_b64 = {}
    plot_files = _db.get_plots(exp_id)
    import base64
    for pf in plot_files:
        path = Path(pf["file_path"])
        if path.exists():
            plots_b64[pf["name"]] = base64.b64encode(path.read_bytes()).decode()

    proof_info = _proof.save_proof(
        exp_id=exp_id,
        domain=exp["domain"],
        exp_type=exp["exp_type"],
        parameters=exp.get("parameters", {}),
        results=exp.get("results", {}),
        plots_b64=plots_b64,
        status=exp.get("status", "success"),
        warnings=exp.get("warnings", []),
        errors=exp.get("errors", []),
        title=title,
        extra_notes=extra_notes,
    )

    # Update DB with new paths
    _db.save_experiment(
        exp_id=exp_id,
        domain=exp["domain"],
        exp_type=exp["exp_type"],
        parameters=exp.get("parameters", {}),
        results=exp.get("results", {}),
        status=exp.get("status", "success"),
        proof_dir=proof_info.get("proof_dir"),
        pdf_path=proof_info.get("pdf_path"),
    )

    return proof_info


@mcp.tool()
async def open_proof(exp_id: str) -> dict:
    """Open the PDF proof for an experiment in the system PDF viewer."""
    import subprocess
    exp = _db.get_experiment(exp_id)
    if not exp:
        return {"error": f"Experiment '{exp_id}' not found."}
    pdf = exp.get("pdf_path")
    if not pdf or not Path(pdf).exists():
        return {"error": "PDF not found. Run generate_proof first."}
    subprocess.Popen(["xdg-open", pdf], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"opened": pdf}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarise(row: dict) -> dict:
    """Strip large fields for list views."""
    return {
        "id": row.get("id"),
        "domain": row.get("domain"),
        "type": row.get("exp_type"),
        "status": row.get("status"),
        "created_at": row.get("created_at"),
        "proof_dir": row.get("proof_dir"),
        "pdf_path": row.get("pdf_path"),
        "warnings": row.get("warnings", []),
        "errors": row.get("errors", []),
    }


def _is_b64(s: str) -> bool:
    import re
    return bool(re.match(r"^[A-Za-z0-9+/]{50,}={0,2}$", s[:100]))


def main():
    """Entry point for simlab-knowledge-mcp CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
